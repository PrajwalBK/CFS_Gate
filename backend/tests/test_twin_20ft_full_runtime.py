"""
End-to-end Verification and Test for Detecting Every Physical Container Independently.

Mandatory Rule:
2 physical containers = 2 independent detections = 2 unique track IDs = 2 container records = 2 event/visit IDs.
When a trailer carries 2 x 20ft containers, neither container is merged or skipped.
"""
from datetime import datetime, timedelta
import numpy as np
import pytest
from unittest.mock import MagicMock

from app.config.settings import CameraConfig
from app.detection.tracker import iou
from app.domain import Detection, Frame, OCRRead, Validation
from app.runtime import GateRuntime


def create_mock_frame(camera_id, t, sequence=1):
    # Dummy 640x480 frame with sufficient contrast
    img = np.full((480, 640, 3), 100, dtype=np.uint8)
    # Draw two container-like rectangles
    img[100:380, 350:600] = 180  # Container 1 area
    img[100:380, 50:300] = 160   # Container 2 area
    return Frame(camera_id, t, sequence, img, 'live')


def test_runtime_e2e_trailer_with_two_20ft_containers(settings, camera, repository, snapshots):
    """
    Test complete GateRuntime pipeline with a trailer carrying two 20ft containers:
    - Container 1: Lead 20ft (ABCU1234567, 22G1)
    - Container 2: Trail 20ft (XYZU7654321, 22G1)
    Both containers pass through the gate, are tracked, OCR'd, finalized, and dispatched independently.
    """
    camera.gate_exit_axis = 'x'
    camera.gate_exit_positive = True
    camera.occupancy_roi = (0.0, 0.0, 1.0, 1.0)
    repository.sync_cameras([camera])

    runtime = GateRuntime(settings, [camera], repository, snapshots)
    runtime.dispatcher = MagicMock()
    runtime.dispatcher.enqueue = MagicMock()

    # Mock OCR engine to return correct container numbers and ISO codes based on crop position
    def mock_ocr(job, crop=None, **kwargs):
        bbox = job.detection.bbox
        cx = (bbox[0] + bbox[2]) / 2.0
        # If in right half of frame (> 320): Container 1 (Lead)
        if cx >= 320:
            if job.detection.class_name == 'iso_code':
                return OCRRead('22G1', 0.95), Validation('22G1', '22G1', 0.95, False, False, 'VALID_SIZE_CODE', size_type='22G1')
            return OCRRead('ABCU1234567', 0.98), Validation('ABCU1234567', 'ABCU1234567', 0.98, True, True, 'VALID', size_type='22G1')
        else:
            # Left half of frame (< 320): Container 2 (Trail)
            if job.detection.class_name == 'iso_code':
                return OCRRead('22G1', 0.94), Validation('22G1', '22G1', 0.94, False, False, 'VALID_SIZE_CODE', size_type='22G1')
            return OCRRead('XYZU7654321', 0.97), Validation('XYZU7654321', 'XYZU7654321', 0.97, True, True, 'VALID', size_type='22G1')

    runtime._execute_ocr = mock_ocr

    base_time = datetime(2026, 9, 15, 10, 0, 0)

    # Frame sequence: 10 frames of both containers moving through the gate
    for frame_idx in range(10):
        t = base_time + timedelta(milliseconds=frame_idx * 150)
        # Shift containers forward along X axis
        dx = frame_idx * 5
        # Lead container (Container 1)
        c1_box = (350.0 + dx, 100.0, 580.0 + dx, 380.0)
        c1_num_box = (400.0 + dx, 150.0, 550.0 + dx, 190.0)
        # Trail container (Container 2)
        c2_box = (60.0 + dx, 100.0, 290.0 + dx, 380.0)
        c2_num_box = (100.0 + dx, 150.0, 250.0 + dx, 190.0)

        frame = create_mock_frame(camera.id, t, sequence=frame_idx + 1)
        dets = [
            Detection(camera.id, t, 'container', 0.95, c1_box),
            Detection(camera.id, t, 'container_number', 0.93, c1_num_box),
            Detection(camera.id, t, 'container', 0.94, c2_box),
            Detection(camera.id, t, 'container_number', 0.92, c2_num_box),
        ]

        class FrameDetector:
            def __init__(self, d): self.d = d
            def detect(self, f): return self.d

        runtime.detector = FrameDetector(dets)
        runtime.process_frame(camera, frame)

    # Drain OCR queues so all reads are processed
    while not runtime.fast_ocr_queue.empty():
        job, profiler = runtime.fast_ocr_queue.get_nowait()
        runtime._process_fast_ocr(job, profiler)

    # Now simulate trailer clearing the gate: send frames with no container
    clear_start = base_time + timedelta(seconds=2)
    for clear_idx in range(8):
        t_clear = clear_start + timedelta(seconds=clear_idx + 1)
        clear_frame = Frame(camera.id, t_clear, 100 + clear_idx, np.zeros((480, 640, 3), dtype=np.uint8), 'live')
        runtime.detector = FrameDetector([])
        runtime.process_frame(camera, clear_frame)

    # Verify that BOTH containers were finalized independently
    events_data = repository.events(limit=10)
    items = events_data['items']

    assert len(items) >= 2, f"Expected at least 2 distinct events, got {len(items)}"

    nums = [it.get('container_number') for it in items]
    assert 'ABCU1234567' in nums, "Container 1 (ABCU1234567) must be detected and recorded!"
    assert 'XYZU7654321' in nums, "Container 2 (XYZU7654321) must be detected and recorded!"

    ev1 = next(it for it in items if it.get('container_number') == 'ABCU1234567')
    ev2 = next(it for it in items if it.get('container_number') == 'XYZU7654321')

    # MANDATORY RULES CHECK:
    # 1. Unique event/visit ID
    assert ev1['id'] != ev2['id']
    # 2. Never merged into single twin container event
    assert ev1['is_twin'] is False
    assert ev2['is_twin'] is False
    assert ev1['container_number_2'] is None
    assert ev2['container_number_2'] is None
    # 3. Both confirmed
    assert ev1['status'] == 'CONFIRMED'
    assert ev2['status'] == 'CONFIRMED'

    # 4. Unique tracking IDs
    det_ev1 = repository.event(ev1['id'])['detections']
    det_ev2 = repository.event(ev2['id'])['detections']
    t1_ids = {d['track_id'] for d in det_ev1}
    t2_ids = {d['track_id'] for d in det_ev2}
    assert t1_ids != t2_ids, "Container 1 and Container 2 must have distinct track IDs!"

    # 5. Both events dispatched to cloud dispatcher
    dispatched_calls = runtime.dispatcher.enqueue.call_args_list
    dispatched_event_ids = [c[0][1] for c in dispatched_calls]
    assert ev1['id'] in dispatched_event_ids
    assert ev2['id'] in dispatched_event_ids


def test_runtime_recovers_temporarily_missed_container_in_trailer(settings, camera, repository, snapshots):
    """
    Ensure that if one container on the trailer is temporarily missed for 1-2 frames,
    the system maintains its track identity and recovers it when visible again without
    creating duplicate tracks or dropping the container.
    """
    camera.occupancy_roi = (0.0, 0.0, 1.0, 1.0)
    repository.sync_cameras([camera])

    runtime = GateRuntime(settings, [camera], repository, snapshots)
    t0 = datetime(2026, 9, 15, 10, 0, 0)

    class DynamicDetector:
        def __init__(self): self.current_dets = []
        def detect(self, f): return self.current_dets

    dyn_det = DynamicDetector()
    runtime.detector = dyn_det

    # Frame 0: Both containers detected
    c1_f0 = Detection(camera.id, t0, 'container', 0.95, (350.0, 100.0, 580.0, 380.0))
    c2_f0 = Detection(camera.id, t0, 'container', 0.94, (60.0, 100.0, 290.0, 380.0))
    dyn_det.current_dets = [c1_f0, c2_f0]
    frame0 = create_mock_frame(camera.id, t0, sequence=1)
    runtime.process_frame(camera, frame0)

    track_c1_id = next(t.id for t in runtime.tracker[camera.id].tracks.values() if iou(t.detection.bbox, c1_f0.bbox) > 0.8)
    track_c2_id = next(t.id for t in runtime.tracker[camera.id].tracks.values() if iou(t.detection.bbox, c2_f0.bbox) > 0.8)
    assert track_c1_id != track_c2_id

    # Frame 1: Container 2 is TEMPORARILY MISSED (e.g. occlusion/glare); only Container 1 detected
    t1 = t0 + timedelta(milliseconds=100)
    c1_f1 = Detection(camera.id, t1, 'container', 0.95, (360.0, 100.0, 590.0, 380.0))
    dyn_det.current_dets = [c1_f1]
    frame1 = create_mock_frame(camera.id, t1, sequence=2)
    runtime.process_frame(camera, frame1)

    # Frame 2: Container 2 is VISIBLE AGAIN (recovered)
    t2 = t0 + timedelta(milliseconds=200)
    c1_f2 = Detection(camera.id, t2, 'container', 0.95, (370.0, 100.0, 600.0, 380.0))
    c2_f2 = Detection(camera.id, t2, 'container', 0.93, (75.0, 100.0, 305.0, 380.0))
    dyn_det.current_dets = [c1_f2, c2_f2]
    frame2 = create_mock_frame(camera.id, t2, sequence=3)
    runtime.process_frame(camera, frame2)

    # Verify both tracks retained their original track IDs
    recovered_c1 = next(t for t in runtime.tracker[camera.id].tracks.values() if iou(t.detection.bbox, c1_f2.bbox) > 0.8)
    recovered_c2 = next(t for t in runtime.tracker[camera.id].tracks.values() if iou(t.detection.bbox, c2_f2.bbox) > 0.8)

    assert recovered_c1.id == track_c1_id, "Container 1 track ID must be preserved!"
    assert recovered_c2.id == track_c2_id, "Container 2 track ID must be recovered after temporary miss!"
