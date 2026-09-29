"""Derived inspection image; never change the original captured evidence."""
import re

import cv2
import numpy as np


def proof_jpeg(data: bytes, snapshot: dict, event: dict) -> bytes:
    # Mutable companion frames cannot be tied reliably to an OCR observation.
    if not snapshot.get('id') or snapshot['id'] not in snapshot.get('image_path', ''):
        return data
    rear = snapshot.get('image_path', '').endswith('_rear.jpg')
    number = event.get('container_number_2' if rear else 'container_number')
    readings = {o['detection_id']: o for o in event.get('ocr_results', [])
                if number and o.get('normalized_text') == number and o.get('valid_check_digit')}
    detection = next((d for d in event.get('detections', [])
                      if d.get('id') in readings
                      and d.get('camera_id') == snapshot.get('camera_id')
                      and d.get('timestamp') == snapshot.get('timestamp')
                      and (d.get('origin_key') or '').endswith(':rear') == rear), None)
    if detection is None:
        return data
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return data
    h, w = frame.shape[:2]
    canvas = np.full((h + 120, max(w + 260, 900), 3), (30, 25, 20), np.uint8)
    canvas[:h, :w] = frame
    crop = None
    # Passage snapshots are already cropped: their detector coordinates belong
    # to the full frame and must NOT be applied to this smaller image.
    bbox = detection.get('bbox')
    if not event.get('lane_id') and detection.get('class_name') == 'container_number' and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1, x2, y2 = max(0, x1 - 8), max(0, y1 - 8), min(w, x2 + 8), min(h, y2 + 8)
        if x2 > x1 and y2 > y1:
            crop = frame[y1:y2, x1:x2]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 210, 255), 2)
    if crop is not None and h > 90:
        scale = min(4.0, 230 / crop.shape[1], (h - 70) / crop.shape[0])
        inset = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        ih, iw = inset.shape[:2]
        canvas[50:50 + ih, w + 15:w + 15 + iw] = inset
        cv2.putText(canvas, 'Number crop', (w + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
    raw = str(readings[detection['id']].get('raw_text') or '')
    compact = re.sub('[^A-Z0-9]', '', raw.upper())
    note = 'OCR reading shown below; verify against photograph.' if number in compact else 'Recorded ID includes validation completion/correction; verify photograph.'
    lines = [f'Recorded ID: {number}', f'Raw OCR: {raw}', note]
    for i, line in enumerate(lines):
        line = re.sub(r'[^\x20-\x7E]', ' ', line)
        font_scale = min(.65, (canvas.shape[1] - 30) / max(1, cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0][0]))
        cv2.putText(canvas, line, (15, h + 30 + i * 32), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (240, 240, 240), 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return encoded.tobytes() if ok else data
