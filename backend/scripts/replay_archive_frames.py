"""Replay saved camera frames in timestamp order into an isolated runtime DB.

These are sparse OCR-triggered captures, not continuous video. Ground-truth
labels apply to the foreground target; unrelated scene objects are not labelled.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from app.config.settings import Settings, CameraConfig
from app.database.database import Database
from app.database.repositories.gate import GateRepository
from app.domain import Frame
from app.detection.yolo_detector import YoloDetector
from app.ocr.ocr_engine import EasyOCREngine
from app.ocr.container_ocr import crop_identification
from app.runtime import GateRuntime
from app.snapshots.snapshot_manager import SnapshotManager


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reverse-arrival', action='store_true')
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    cfg = Settings(_env_file=None,deployment_mode='test',pipeline_enabled=False,
        database_url='sqlite:///'+str((args.output/'replay.db').resolve()),
        snapshot_directory=args.output/'snapshots',upload_directory=args.output/'uploads',
        model_path=args.model.resolve(), model_device='0',confidence_threshold=.28,
        ocr_gpu=True,ocr_download_enabled=False,ocr_model_directory=Path.home()/'.EasyOCR/model',
        camera_discovery_enabled=False,prosper_enabled=False)
    db = Database(cfg.database_url.get_secret_value());db.initialize()
    repo = GateRepository(db); snaps=SnapshotManager(cfg.snapshot_directory)
    detector=YoloDetector(cfg);engine=EasyOCREngine(cfg)
    manifest=json.loads(args.manifest.read_text())['frames']
    labels=json.loads(args.labels.read_text());results=[]
    for label in labels:
        original=json.loads((args.root/label['event_id']/'event.json').read_text())
        frames=sorted((f for f in manifest if f['event_id']==label['event_id']),
            key=lambda f:(f['timestamp'],f['job_id']),reverse=args.reverse_arrival)
        cameras={c:CameraConfig(id=c,name=c,gate_id=original['gate_id'],direction=original['event_type'],
            source_env='ARCHIVE_ONLY',role='REAR' if c.endswith('-3') else 'FRONT_TOP' if c.endswith('-1') else 'LEFT')
            for c in sorted({f['camera_id'] for f in frames})}
        repo.sync_cameras(list(cameras.values()))
        prior_ids={e['id'] for e in repo.events(limit=10000)['items']}
        runtime=GateRuntime(cfg,list(cameras.values()),repo,snaps)
        runtime.detector=detector;runtime.ocr=engine
        result={'original_event_id':label['event_id'],'expected_number':label['container_number'],
            'source_frame_count':len(frames),'processed_frames':0,'ocr_reads':[],'errors':[],'events':[],
            'arrival_order':'reversed' if args.reverse_arrival else 'timestamp',
            'scenarios':label['scenarios'],'continuous_video':False}
        for value in frames:
            image=cv2.imread(str(args.root/value['image_path']))
            if image is None:
                result['errors'].append({'job_id':value['job_id'],'error':'missing_source_frame'});continue
            stamp=datetime.fromisoformat(value['timestamp']).replace(tzinfo=None)
            frame=Frame(value['camera_id'],stamp,value['sequence'],image,
                f"archive:{label['event_id']}:{value['camera_id']}")
            try:
                runtime.process_frame(cameras[value['camera_id']],frame)
                while not runtime.queue.empty():
                    job=runtime.queue.get_nowait()
                    try:
                        job_id=runtime.ocr_store.save(job,crop_identification(job.frame.image,job.detection.bbox,job.roi))
                        runtime._pending_event_saves[job.event_id]=max(0,runtime._pending_event_saves.get(job.event_id,0)-1)
                        runtime._process_saved_ocr(job_id)
                        _,_,read=runtime.ocr_store.load(job_id)
                        result['ocr_reads'].append({'source_job_id':value['job_id'],'read':asdict(read) if read else None})
                    finally:runtime.queue.task_done()
                result['processed_frames']+=1
            except Exception as error:
                result['errors'].append({'job_id':value['job_id'],'error':f'{type(error).__name__}: {error}'})
        if frames:
            end=max(datetime.fromisoformat(f['timestamp']).replace(tzinfo=None) for f in frames)+timedelta(seconds=30)
            for session in runtime.passage_sessions.check_and_finalize(end):runtime._finalize_passage_session(session)
            runtime._retry_passage_finalizations()
        for item in repo.events(limit=10000)['items']:
            if item['gate_id']!=original['gate_id'] or item['id'] in prior_ids:continue
            event=repo.event(item['id'])
            result['events'].append({'event_id':event['id'],'number':event['container_number'],
                'status':event['status'],'camera_ids':[s['camera_id'] for s in event['snapshots']],
                'matches_expected_foreground':event['container_number']==label['container_number'],
                'manifest_cameras':[s['camera_id'] for s in snaps.build_evidence_manifest(event)['captures']]})
        results.append(result)
        (args.output/'results.json').write_text(json.dumps(results,indent=2))
        print('GROUP',label['event_id'],'frames',result['processed_frames'],'reads',len(result['ocr_reads']),
            'events',len(result['events']),'errors',len(result['errors']),flush=True)
    db.close()


if __name__=='__main__':main()
