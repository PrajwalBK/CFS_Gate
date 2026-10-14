"""Send only isolated replay events to a localhost ground-truth receiver."""
import argparse
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
from pydantic import SecretStr
from app.config.settings import Settings
from app.database.database import Database
from app.database.repositories.gate import GateRepository
from app.events.cloud_dispatcher import CloudOutboxDispatcher
from app.snapshots.snapshot_manager import SnapshotManager


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--replay',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    args=parser.parse_args()
    groups=json.loads((args.replay/'results.json').read_text())
    frames=json.loads(args.manifest.read_text())['frames']
    expected={e['event_id']:group for group in groups for e in group['events'] if e['status']=='CONFIRMED'}
    cfg=Settings(_env_file=None,deployment_mode='test',pipeline_enabled=False,
        database_url='sqlite:///'+str((args.replay/'replay.db').resolve()),
        snapshot_directory=args.replay/'snapshots',upload_directory=args.replay/'delivery-artifacts',
        prosper_enabled=False, camera_discovery_enabled=False)
    db=Database(cfg.database_url.get_secret_value());db.check()
    repo=GateRepository(db);snaps=SnapshotManager(cfg.snapshot_directory)
    receipts=[]
    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            group=expected[payload['id']];event=repo.event(payload['id'])
            captures={s['camera_id']:s for s in snaps.build_evidence_manifest(event)['captures']}
            errors=[];images=[]
            for container in payload['Containers']:
                if container['ContainerNumber']!=group['expected_number']:errors.append('ground_truth_identity_mismatch')
                for image in container['Images']:
                    data=base64.b64decode(image['Image'].split(',',1)[1])
                    snapshot=captures.get(image['CameraId'])
                    if not snapshot or data!=snaps.path(snapshot['image_path']).read_bytes():
                        errors.append('stored_camera_proof_mismatch');continue
                    proof=snapshot.get('provenance') or {}
                    source=next((f for f in frames if f['event_id']==group['original_event_id']
                        and f['camera_id']==image['CameraId'] and f['sequence']==proof.get('frame_sequence')
                        and f['timestamp']==proof.get('timestamp','').rstrip('Z')),None)
                    source_matches=False
                    if source:
                        raw=cv2.imread(str(args.source/source['image_path']))
                        if raw is not None:
                            source_matches=any(cv2.imencode('.jpg',raw,[cv2.IMWRITE_JPEG_QUALITY,q])[1].tobytes()==data for q in (90,92))
                    if not source_matches:errors.append('original_capture_provenance_mismatch')
                    images.append({'camera_id':image['CameraId'],'captured_at':image['CapturedAt'],
                        'sha256':hashlib.sha256(data).hexdigest(),'source_sequence':proof.get('frame_sequence'),
                        'source_bytes_verified':source_matches})
            receipts.append({'event_id':payload['id'],'expected_number':group['expected_number'],
                'received_number':payload['Containers'][0]['ContainerNumber'],'images':images,'errors':errors})
            self.send_response(422 if errors else 200);self.end_headers();self.wfile.write(b'{"eventId":"local-archive-verified"}')
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Receiver)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    cloud=cfg.model_copy(update={'prosper_enabled':True,'prosper_max_retries':1,
        'prosper_api_base_url':f'http://127.0.0.1:{server.server_port}','prosper_bearer_token':SecretStr('local-only')})
    dispatcher=CloudOutboxDispatcher(cloud,repository=repo,snapshots=snaps);dispatcher.start()
    try:
        for event_id in expected:
            dispatcher.enqueue(None,event_id,force=True)
            deadline=time.monotonic()+30
            while dispatcher.queue.unfinished_tasks and time.monotonic()<deadline:time.sleep(.05)
        result={'receiver_scope':'localhost only','receipts':receipts,
            'delivery_states':{event_id:repo.event(event_id)['cloud_delivery_state'] for event_id in expected}}
        (args.replay/'verified-http-receipts.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2),flush=True)
    finally:
        dispatcher.stop();server.shutdown();server.server_close();thread.join(timeout=2);db.close()


if __name__=='__main__':main()
