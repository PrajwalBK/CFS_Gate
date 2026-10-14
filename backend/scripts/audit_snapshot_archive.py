"""Read-only archive audit with real local detector and optional cached OCR.

Stored identities are predictions, never independent ground truth. Source files
and production DB/cloud are not modified. JSON output is periodically checkpointed.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from app.config.settings import Settings
from app.domain import Frame, OCRRead
from app.detection.yolo_detector import YoloDetector
from app.ocr.ocr_engine import EasyOCREngine
from app.ocr.validator import ContainerValidator, is_valid_iso_6346
from app.snapshots.snapshot_manager import SnapshotManager


def audit(root, output, detect=False, model=None, ocr_samples=0):
    output.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    settings = Settings(_env_file=None, deployment_mode='test', pipeline_enabled=False,
        model_path=model or Path(__file__).resolve().parents[1]/'models/best.pt',
        model_device='0', confidence_threshold=.28, ocr_gpu=True,
        ocr_download_enabled=False, ocr_model_directory=Path.home()/'.EasyOCR/model', prosper_enabled=False)
    detector = YoloDetector(settings) if detect else None
    validator = ContainerValidator()
    rows, events, issues = [], [], Counter()
    files = list(root.glob('*/event.json'))
    for i, metadata_path in enumerate(sorted(files)):
        try:
            event = json.loads(metadata_path.read_text(encoding='utf-8'))
        except (ValueError, OSError) as error:
            issues['unreadable_event_json'] += 1
            events.append({'directory': metadata_path.parent.name, 'error': str(error)})
            continue
        event_issues = []
        if event.get('id') != metadata_path.parent.name:
            event_issues.append('event_directory_identity_mismatch')
        number = event.get('container_number')
        if event.get('status') == 'CONFIRMED' and not is_valid_iso_6346(number or ''):
            event_issues.append('confirmed_invalid_checksum')
        valid_reads = {v.get('normalized_text') for v in event.get('ocr_results', [])
            if is_valid_iso_6346(v.get('normalized_text') or '')}
        if len(valid_reads) > 1:
            event_issues.append('conflicting_checksum_valid_reads')
        if number and valid_reads and number not in valid_reads:
            event_issues.append('identity_without_matching_valid_read')
        for value in event.get('ocr_results', []):
            revised = validator.validate(OCRRead(
                value.get('raw_text') or '', float(value.get('confidence') or 0)))
            if value.get('valid_check_digit') and not revised.valid_check_digit:
                event_issues.append('historical_read_not_accepted_by_strict_validator')
        for snapshot in event.get('snapshots', []):
            row = {'event_id': event.get('id'), 'camera_id': snapshot.get('camera_id'),
                'snapshot_id': snapshot.get('id'), 'image_path': snapshot.get('image_path'),
                'timestamp': snapshot.get('timestamp'), 'stored_number': number,
                'status': event.get('status'), 'issues': []}
            path = (root / (snapshot.get('image_path') or '')).resolve()
            if not path.is_relative_to(root) or path == root:
                row['issues'].append('unsafe_path')
            elif not path.is_file():
                row['issues'].append('missing_recorded_image')
            else:
                image = cv2.imread(str(path))
                if image is None:
                    row['issues'].append('undecodable_recorded_image')
                else:
                    row['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    row.update(width=image.shape[1], height=image.shape[0],
                        brightness=round(float(gray.mean()), 2),
                        highlight_fraction=round(float((gray>245).mean()), 4),
                        quality=round(SnapshotManager.compute_quality(image), 3))
                    if min(image.shape[:2]) < 240 or image.shape[1] < 320:
                        row['issues'].append('undersized_proof')
                    if gray.std() < 8 or row['quality'] < 2.5:
                        row['issues'].append('weak_proof_quality')
                    proof = snapshot.get('provenance') or {}
                    if not proof:
                        row['issues'].append('legacy_no_capture_provenance')
                    elif proof.get('image_sha256') != row['sha256']:
                        row['issues'].append('image_digest_mismatch')
                    if detector:
                        stamp = datetime.fromisoformat((snapshot.get('timestamp') or event['timestamp']).rstrip('Z')).replace(tzinfo=None)
                        detections = detector.detect(Frame(row['camera_id'], stamp, i+1, image, str(path)))
                        row['actual_detections'] = [{'class_name': d.class_name, 'confidence': d.confidence, 'bbox': d.bbox} for d in detections]
                        if not any(d.class_name == 'container' for d in detections):
                            row['issues'].append('actual_model_no_physical_container')
            rows.append(row)
            issues.update(row['issues'])
        issues.update(set(event_issues))
        events.append({'event_id': event.get('id'), 'status':event.get('status'),
            'stored_number': number, 'timestamp': event.get('timestamp'), 'issues': sorted(set(event_issues))})
        if i % 25 == 0:
            print(f'ARCHIVE {i}/{len(files)} events; {len(rows)} proof references', flush=True)
            (output/'archive-audit.json').write_text(json.dumps({'events':events, 'snapshots':rows, 'issue_counts':dict(issues)}, indent=2))
    ocr = []
    if ocr_samples:
        candidates = [r for r in rows if 'actual_detections' in r and any(d['class_name']=='container_number' for d in r['actual_detections'])]
        # Diverse sampling by measured darkness, highlights, weak quality and chronology.
        selected, seen = [], set()
        rankings = [sorted(candidates, key=lambda r:r['brightness']),
            sorted(candidates, key=lambda r:r['highlight_fraction'], reverse=True),
            sorted(candidates, key=lambda r:r['quality']), candidates]
        for rank in range(len(candidates)):
            for ranking in rankings:
                row = ranking[rank]
                key = row['image_path']
                if key not in seen:
                    seen.add(key); selected.append(row)
                if len(selected) >= ocr_samples: break
            if len(selected) >= ocr_samples: break
        engine = EasyOCREngine(settings)
        for i, row in enumerate(selected):
            image = cv2.imread(str(root/row['image_path']))
            readings = []
            for detection in sorted((d for d in row['actual_detections'] if d['class_name']=='container_number'), key=lambda d:d['confidence'], reverse=True):
                x1,y1,x2,y2 = map(int,detection['bbox'])
                crop = image[max(0,y1-10):min(image.shape[0],y2+10),max(0,x1-15):min(image.shape[1],x2+15)]
                value = validator.validate(engine.read(crop))
                readings.append({'raw_text':value.raw_text, 'normalized_text':value.normalized_text,
                    'valid_check_digit':value.valid_check_digit, 'confidence':value.confidence})
            valid_numbers = {r['normalized_text'] for r in readings if r['valid_check_digit']}
            ocr.append({**row, 'actual_ocr_reads': readings,
                'comparison_to_stored_prediction': 'AGREES' if valid_numbers=={row['stored_number']} else
                    'DIFFERENT_VALID_IDENTITY' if valid_numbers else 'UNREADABLE_OR_INVALID',
                'independent_ground_truth': False})
            (output/'actual-ocr-sample.json').write_text(json.dumps(ocr, indent=2))
            print(f'OCR {i+1}/{len(selected)} {ocr[-1]["comparison_to_stored_prediction"]}', flush=True)
    result = {'source':str(root), 'source_modified':False, 'events':events, 'snapshots':rows,
        'event_count':len(events), 'proof_reference_count':len(rows), 'issue_counts':dict(issues),
        'status_counts':dict(Counter(e.get('status') for e in events)),
        'model':str(settings.model_path), 'detector_exercised': bool(detector),
        'ocr_sample_count':len(ocr), 'ocr_comparisons':dict(Counter(r['comparison_to_stored_prediction'] for r in ocr)),
        'semantic_accuracy_established':False}
    (output/'archive-audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('events','snapshots')}, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--detect', action='store_true')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--ocr-samples', type=int, default=0)
    args = parser.parse_args()
    audit(args.root,args.output,args.detect,args.model,args.ocr_samples)
