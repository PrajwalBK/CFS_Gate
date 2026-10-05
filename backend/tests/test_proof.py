import cv2
import numpy as np
from app.snapshots.proof import proof_jpeg


def evidence():
    frame = np.full((300, 500, 3), 180, np.uint8)
    cv2.putText(frame, 'BMOU', (25, 80), 0, 1, (0, 0, 0), 2)
    data = cv2.imencode('.jpg', frame)[1].tobytes()
    snap = dict(id='snap1', image_path='event/cam_snap1.jpg', camera_id='cam', timestamp='t')
    event = dict(container_number='BMOU1353220', detections=[dict(id='det', camera_id='cam', timestamp='t',
        origin_key='dwell', class_name='container_number', bbox=[20, 40, 125, 90])],
        ocr_results=[dict(detection_id='det', normalized_text='BMOU1353220', raw_text='BMOU 135322', valid_check_digit=True)])
    return data, snap, event


def test_clean_proof_preserves_raw_source():
    data, snap, event = evidence()
    result = proof_jpeg(data, snap, event)
    assert result == data
    decoded = cv2.imdecode(np.frombuffer(result, np.uint8), 1)
    assert decoded.shape == (300, 500, 3)


def test_unlinked_or_wrong_identity_returns_original():
    data, snap, event = evidence()
    assert proof_jpeg(data, {**snap, 'image_path': 'cam.jpg'}, event) == data
    assert proof_jpeg(data, {**snap, 'timestamp': 'other'}, event) == data
    assert proof_jpeg(data, snap, {**event, 'container_number': 'OTHER'}) == data
    assert proof_jpeg(b'broken', snap, event) == b'broken'
