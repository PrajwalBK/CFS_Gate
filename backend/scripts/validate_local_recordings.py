"""Read-only source-media validation; results and replay DB stay in --output.

Uses local model weights, actual detector/OCR and the production runtime. Cloud
credentials are never loaded. Clips are independent sources unless separately
labelled as synchronized cameras; resulting identities are predictions, not truth.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import time
import base64
import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pydantic import SecretStr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from app.config.settings import Settings, CameraConfig
from app.database.database import Database
from app.database.repositories.gate import GateRepository
from app.detection.yolo_detector import YoloDetector
from app.domain import Frame
from app.ocr.container_ocr import crop_identification
from app.ocr.ocr_engine import PaddleOCREngine, EasyOCREngine
from app.runtime import GateRuntime
from app.events.cloud_dispatcher import CloudOutboxDispatcher
from app.snapshots.snapshot_manager import SnapshotManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('D:/videos'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fps', type=float, default=3)
    parser.add_argument('--confidence', type=float, default=.35)
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--end-frame', type=int)
    parser.add_argument('--samples', type=int, default=0, help='0 means whole clips at --fps')
    parser.add_argument('--detector-test', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--detector-only', action='store_true')
    parser.add_argument('--ocr-engine', choices=['paddleocr', 'easyocr'], default='paddleocr')
    parser.add_argument('--extra-root', type=Path, action='append', default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    settings = Settings(_env_file=None, deployment_mode='test', pipeline_enabled=False,
        database_url='sqlite:///' + str((args.output / 'replay.db').resolve()),
        snapshot_directory=args.output / 'snapshots', upload_directory=args.output / 'uploads',
        model_path=args.model or Path(__file__).resolve().parents[1] / 'models/best.pt',
        ocr_model_directory=Path.home() / '.EasyOCR/model',
        model_device='0' if torch.cuda.is_available() else 'cpu',
        ocr_fallback_engine='none', prosper_enabled=False, confidence_threshold=args.confidence, camera_discovery_enabled=False)
    detector = YoloDetector(settings)
    if args.detector_test:
        counts = {name: {'tp': 0, 'fp': 0, 'fn': 0} for name in ['container_number', 'iso_code', 'container']}
        images = sorted((args.detector_test / 'test/images').glob('*'))
        aliases = ['container_number', 'iso_code', 'container']
        for idx, path in enumerate(images):
            image = cv2.imread(str(path))
            if image is None:
                continue
            h, w = image.shape[:2]
            truth = []
            for line in (args.detector_test / 'test/labels' / (path.stem + '.txt')).read_text().splitlines():
                values = list(map(float, line.split()))
                if len(values) == 5:
                    cls, x, y, bw, bh = values
                else:
                    cls = values[0]
                    xs, ys = values[1::2], values[2::2]
                    x, y = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
                    bw, bh = max(xs)-min(xs), max(ys)-min(ys)
                truth.append((aliases[int(cls)], ((x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h)))
            predictions = detector.detect(Frame('label-test', datetime(2026,10,9), idx, image, str(path)))
            matched = set()
            def iou(a, b):
                intersection = max(0, min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
                union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
                return intersection / union if union else 0
            for detection in sorted(predictions, key=lambda d: d.confidence, reverse=True):
                eligible = [(iou(detection.bbox, box), j) for j, (name, box) in enumerate(truth)
                    if name == detection.class_name and j not in matched]
                score, match = max(eligible, default=(0, -1))
                if score >= .5:
                    matched.add(match)
                    counts[detection.class_name]['tp'] += 1
                else:
                    counts[detection.class_name]['fp'] += 1
            for j, (name, _) in enumerate(truth):
                if j not in matched:
                    counts[name]['fn'] += 1
            if idx % 100 == 0:
                print('LABEL_PROGRESS', idx, '/', len(images), flush=True)
        for value in counts.values():
            value['precision'] = value['tp'] / max(1,value['tp']+value['fp'])
            value['recall'] = value['tp'] / max(1,value['tp']+value['fn'])
        metrics = {'images': len(images), 'iou_threshold': .5, 'confidence_threshold': settings.confidence_threshold, 'classes': counts}
        (args.output / 'detector-metrics.json').write_text(json.dumps(metrics, indent=2))
        print('DETECTOR_METRICS', json.dumps(metrics), flush=True)
    if args.detector_only:
        return
    engine = PaddleOCREngine(settings) if args.ocr_engine == 'paddleocr' else EasyOCREngine(settings)
    print('OCR_ENGINE', args.ocr_engine, flush=True)
    db = Database(settings.database_url.get_secret_value())
    db.initialize()
    repository = GateRepository(db)
    snapshots = SnapshotManager(settings.snapshot_directory)
    receipts = []
    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            event = repository.event(payload['id'])
            proof = {s['camera_id']: s for s in snapshots.build_evidence_manifest(event)['captures']}
            errors = []
            received = []
            for container in payload['Containers']:
                if container['ContainerNumber'] != event['container_number']:
                    errors.append('container_identity_mismatch')
                for image in container['Images']:
                    data = base64.b64decode(image['Image'].split(',',1)[1])
                    snapshot = proof.get(image['CameraId'])
                    if not snapshot or data != snapshots.path(snapshot['image_path']).read_bytes():
                        errors.append('camera_image_mismatch')
                    received.append({'camera_id': image['CameraId'], 'captured_at': image['CapturedAt'],
                        'sha256': hashlib.sha256(data).hexdigest()})
            receipts.append({'event_id': payload['id'], 'number': event['container_number'],
                'images': received, 'errors': errors})
            self.send_response(422 if errors else 200)
            self.end_headers()
            self.wfile.write(b'{"eventId":"offline-receiver"}')
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Receiver)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    cloud_settings = settings.model_copy(update={'prosper_enabled': True,
        'prosper_api_base_url': f'http://127.0.0.1:{server.server_port}',
        'prosper_bearer_token': SecretStr('isolated-replay'), 'prosper_max_retries': 1})
    delivery = CloudOutboxDispatcher(cloud_settings, repository=repository, snapshots=snapshots)
    delivery.start()
    results = []
    try:
        files = sorted(p for root in [args.root] + args.extra_root for p in ([root] if root.is_file() else root.rglob('*'))
                       if p.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv'})
        for index, path in enumerate(files):
            camera = CameraConfig(id=f'clip-{index+1}', name=path.name, gate_id=f'replay-{index+1}',
                source_env='OFFLINE_ONLY', source_type='file', direction='ENTRY')
            repository.sync_cameras([camera])
            runtime = GateRuntime(settings, [camera], repository, snapshots)
            runtime.detector = detector
            runtime.ocr = engine
            cap = cv2.VideoCapture(str(path))
            fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            record = {'file': str(path), 'frames_total': total, 'source_fps': fps,
                'sample_mode': args.samples, 'frames_processed': 0, 'ocr_jobs': 0,
                'errors': [], 'reads': [], 'events': []}
            started = time.monotonic()
            base = datetime(2026, 10, 9) + timedelta(hours=index)
            if args.samples:
                frame_ids = sorted(set(int((total-1)*fraction/(args.samples+1)) for fraction in range(1,args.samples+1)))
                frames = iter(frame_ids)
            else:
                frames = range(args.start_frame, min(total, args.end_frame or total), max(1, round(fps/args.fps)))
            for seq in frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, seq)
                ok, image = cap.read()
                if not ok:
                    record['errors'].append({'frame': seq, 'error': 'decode_failed'})
                    continue
                frame = Frame(camera.id, base + timedelta(seconds=seq/fps), seq+1, image, str(path))
                try:
                    runtime.process_frame(camera, frame)
                    while not runtime.queue.empty():
                        job = runtime.queue.get_nowait()
                        try:
                            job_id = runtime.ocr_store.save(job, crop_identification(job.frame.image, job.detection.bbox, job.roi))
                            runtime._pending_event_saves[job.event_id] = max(0, runtime._pending_event_saves.get(job.event_id, 0)-1)
                            runtime._process_saved_ocr(job_id)
                            _, _, value = runtime.ocr_store.load(job_id)
                            record['reads'].append({'frame': seq, 'event_id': job.event_id,
                                'validation': asdict(value) if value else None})
                            record['ocr_jobs'] += 1
                        finally:
                            runtime.queue.task_done()
                    record['frames_processed'] += 1
                except Exception as exc:
                    import traceback
                    record['errors'].append({'frame': seq, 'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()})
                if record['frames_processed'] % 30 == 0:
                    print('PROGRESS',path.name,record['frames_processed'], 'ocr', record['ocr_jobs'], flush=True)
            cap.release()
            end = base + timedelta(seconds=(total/fps if fps else 0) + 30)
            for session in runtime.passage_sessions.check_and_finalize(end):
                runtime._finalize_passage_session(session)
            runtime._retry_passage_finalizations()
            for item in repository.events(limit=1000)['items']:
                if item['gate_id'] != camera.gate_id:
                    continue
                event = repository.event(item['id'])
                manifest = snapshots.build_evidence_manifest(event)
                if event['status'] == 'CONFIRMED':
                    delivery.enqueue(None, item['id'])
                    delivery.queue.join()
                    event = repository.event(item['id'])
                record['events'].append({'id': item['id'], 'status': item['status'],
                    'container_number': item['container_number'],
                    'delivery_state': event['cloud_delivery_state'],
                    'captured_cameras': [s['camera_id'] for s in event['snapshots']],
                    'usable_cameras': [s['camera_id'] for s in manifest['captures']]})
            record['seconds_elapsed'] = round(time.monotonic()-started, 2)
            results.append(record)
            (args.output / 'recording-results.json').write_text(json.dumps(results, indent=2))
            (args.output / 'http-receipts.json').write_text(json.dumps(receipts, indent=2))
            print('CLIP_COMPLETE', path.name, record['frames_processed'], 'frames',
                record['ocr_jobs'], 'ocr', len(record['events']), 'events', len(record['errors']), 'errors', flush=True)
    finally:
        delivery.stop()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)
        if hasattr(engine, "close"):
            engine.close()
        db.close()


if __name__ == '__main__':
    main()
