"""Actual model/OCR -> passage -> persistence -> localhost HTTP ground-truth check.

Still-image acceptance only: does not invent synchronized temporal recordings.
"""
import argparse
import base64
from datetime import datetime
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time
from pydantic import SecretStr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from app.config.settings import Settings, CameraConfig
from app.database.database import Database
from app.database.repositories.gate import GateRepository
from app.domain import Frame
from app.events.cloud_dispatcher import CloudOutboxDispatcher
from app.events.passage_session import ContainerPassageSession
from app.ocr.ocr_engine import EasyOCREngine
from app.runtime import GateRuntime, OCRJob
from app.snapshots.snapshot_manager import SnapshotManager
from app.detection.yolo_detector import YoloDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    labels = json.loads(args.labels.read_text())
    cfg = Settings(_env_file=None, deployment_mode='test', pipeline_enabled=False,
        database_url='sqlite:///'+str((args.output/'replay.db').resolve()),
        snapshot_directory=args.output/'snapshots', upload_directory=args.output/'uploads',
        model_path=args.model.resolve(), model_device='0', confidence_threshold=.28,
        ocr_gpu=True, ocr_download_enabled=False, ocr_model_directory=Path.home()/'.EasyOCR/model',
        camera_discovery_enabled=False, prosper_enabled=False)
    db = Database(cfg.database_url.get_secret_value()); db.initialize()
    repo = GateRepository(db); snaps = SnapshotManager(cfg.snapshot_directory)
    detector = YoloDetector(cfg); engine = EasyOCREngine(cfg)
    expected, receipts, results = {}, [], []
    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            truth = expected[payload['id']]
            event = repo.event(payload['id'])
            manifest = {s['camera_id']:s for s in snaps.build_evidence_manifest(event)['captures']}
            errors, images = [], []
            for container in payload['Containers']:
                if container['ContainerNumber'] != truth['container_number']:
                    errors.append('wrong_container_number')
                if container['SizeCode'] != truth['size_type']:
                    errors.append('wrong_iso_code')
                for image in container['Images']:
                    data = base64.b64decode(image['Image'].split(',',1)[1])
                    snapshot = manifest.get(image['CameraId'])
                    if image['CameraId'] != truth['camera_id'] or not snapshot:
                        errors.append('wrong_camera')
                    elif data != snaps.path(snapshot['image_path']).read_bytes():
                        errors.append('wrong_image_bytes')
                    images.append({'camera_id':image['CameraId'],'sha256':hashlib.sha256(data).hexdigest()})
            receipts.append({'event_id':payload['id'],'errors':errors,'images':images})
            self.send_response(422 if errors else 200); self.end_headers()
            self.wfile.write(b'{"eventId":"labelled-local-test"}')
        def log_message(self,*args): pass
    server = ThreadingHTTPServer(('127.0.0.1',0), Receiver)
    thread = threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    cloud_cfg = cfg.model_copy(update={'prosper_enabled':True,
        'prosper_api_base_url':f'http://127.0.0.1:{server.server_port}',
        'prosper_bearer_token':SecretStr('local-only'), 'prosper_max_retries':1})
    dispatcher = CloudOutboxDispatcher(cloud_cfg,repository=repo,snapshots=snaps); dispatcher.start()
    try:
        for index,label in enumerate(labels):
            metadata = json.loads((args.root/label['event_id']/'event.json').read_text())
            camera = CameraConfig(id=label['camera_id'],name=label['camera_id'],gate_id=metadata['gate_id'],
                source_env='LOCAL_ACCEPTANCE', direction=metadata['event_type'])
            repo.sync_cameras([camera])
            runtime = GateRuntime(cfg,[camera],repo,snaps); runtime.ocr = engine
            image = cv2.imread(str(args.root/label['image_path']))
            timestamp = datetime.fromisoformat(metadata['timestamp'].rstrip('Z')).replace(tzinfo=None)
            frame = Frame(camera.id,timestamp,index+1,image,str(args.root/label['image_path']))
            detections = detector.detect(frame)
            # Label constrains evaluation; it is never passed into OCR or consensus.
            bodies = sorted((d for d in detections if d.class_name=='container'),
                key=lambda d:(d.bbox[2]-d.bbox[0])*(d.bbox[3]-d.bbox[1]),reverse=True)
            result = {'source_image':label['image_path'],'expected_number':label['container_number'],
                'scenarios':label['scenarios'], 'detected_physical_containers':len(bodies)}
            if bodies:
                job = OCRJob(f'acceptance:{index}',camera.gate_id,index+1,bodies[0],frame,camera.direction,
                    timestamp,camera.ocr_roi,tuple(detections),camera.role)
                read,value = runtime._execute_ocr(job)
                result.update(raw_text=read.raw_text if read else '', recognized_number=value.normalized_text if value else None,
                    confidence=value.confidence if value else 0, recognized_iso=value.size_type if value else None)
                session = ContainerPassageSession(f'acceptance:{index}',camera.gate_id,camera.direction,[camera.id],timestamp)
                session.add_frame(camera.id,image,timestamp,[bodies[0]],source_id=frame.source_id,frame_sequence=frame.sequence)
                session.add_ocr_candidate(value)
                expected[session.event_id] = label
                runtime.dispatcher = dispatcher
                runtime._finalize_passage_session(session)
                deadline = time.monotonic()+30
                while dispatcher.queue.unfinished_tasks and time.monotonic()<deadline: time.sleep(.05)
                event = repo.event(session.event_id)
                result.update(event_id=session.event_id,status=event['status'] if event else None,
                    delivery_state=event['cloud_delivery_state'] if event else None)
                result['identity_matches_ground_truth'] = bool(value and value.normalized_text==label['container_number'])
            else:
                result.update(status='NO_PHYSICAL_CONTAINER',identity_matches_ground_truth=False)
            results.append(result); print(json.dumps(result),flush=True)
            (args.output/'results.json').write_text(json.dumps({'model':str(args.model),
                'results':results,'http_receipts':receipts,'scope':'labelled still images; not temporal multi-camera replay'},indent=2))
    finally:
        dispatcher.stop(); server.shutdown();server.server_close();thread.join(timeout=2);db.close()


if __name__ == '__main__': main()
