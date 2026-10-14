from datetime import datetime, timedelta
import numpy as np
import pytest
from app.domain import Frame, Detection, Validation, Observation
from app.events.occupancy import ViewOccupancy


def make_frame_detections(t, boxes, camera_id='cam-1', source='source'):
    ts = datetime(2026, 9, 15, 10, 0, 0) + timedelta(seconds=t)
    frame = Frame(camera_id, ts, int(t * 100) + 1, np.zeros((100, 100, 3), dtype=np.uint8), source)
    dets = [Detection(camera_id, ts, 'container', 0.95, box) for box in boxes]
    return frame, dets


def test_single_container_lifecycle(settings, camera):
    view = ViewOccupancy(settings)
    frame, dets = make_frame_detections(0, [((20, 20, 70, 80))])
    det = view.update(camera, frame, dets)
    assert det is not None
    assert view.current is not None
    assert not view.current.is_twin
    assert len(view.current.slots) == 1
    assert 'FRONT' in view.current.slots

    # Confirm event
    view.confirm(view.current.origin, 'evt-single', 'MSCU1234566')
    assert view.current.closed is True
    assert view.current.container_number == 'MSCU1234566'


def test_containers_independent_no_twin_merging(settings, camera):
    """Multiple containers in view must never be merged into twin containers."""
    view = ViewOccupancy(settings)
    camera.gate_exit_axis = 'x'
    camera.gate_exit_positive = True

    box_front = (55, 20, 90, 80)
    box_rear = (10, 20, 45, 80)

    frame, dets = make_frame_detections(0, [box_front, box_rear])
    det = view.update(camera, frame, dets)
    assert det is not None
    assert view.current is not None
    # Must NOT be marked as twin
    assert view.current.is_twin is False
    assert view.status()['is_twin'] is False


def test_independent_containers_saved_under_distinct_event_ids(repository, snapshots, camera):
    """Each physical container = 1 unique tracker ID = 1 event = 1 visit ID. Never merged."""
    repo = repository

    frame1, (det1,) = make_frame_detections(0, [(55, 20, 90, 80)], camera_id=camera.id)
    val1 = Validation('MSCU1234566', 'MSCU1234566', 0.99, True, True, 'VALID', size_type='22G1')
    obs1 = Observation(f'container:{camera.id}:1', camera.gate_id, 1, det1, val1, 'IN',
                       frame1.image, frame1.timestamp, True, slot_position='SINGLE', is_twin=False)

    # 1. Container 1 creates its independent gate event
    event_id1, created1 = repo.save_observation(obs1, snapshots, 30)
    assert created1 is True
    assert event_id1 is not None

    # 2. Container 2 on same trailer is processed independently as its own container / tracker ID
    frame2, (det2,) = make_frame_detections(1, [(10, 20, 45, 80)], camera_id=camera.id)
    val2 = Validation('TGHU7654320', 'TGHU7654320', 0.98, True, True, 'VALID', size_type='22G1')
    obs2 = Observation(f'container:{camera.id}:2', camera.gate_id, 2, det2, val2, 'IN',
                       frame2.image, frame2.timestamp, True, slot_position='SINGLE', is_twin=False)

    event_id2, created2 = repo.save_observation(obs2, snapshots, 30)
    assert created2 is True
    assert event_id2 != event_id1  # MUST BE DISTINCT EVENTS!

    # 3. Retrieve events and verify both exist independently
    ev1 = repo.event(event_id1)
    ev2 = repo.event(event_id2)
    assert ev1 is not None and ev2 is not None
    assert ev1['is_twin'] is False
    assert ev2['is_twin'] is False
    assert ev1['container_number'] == 'MSCU1234566'
    assert ev2['container_number'] == 'TGHU7654320'
    assert ev1['container_number_2'] is None
    assert ev2['container_number_2'] is None
    assert ev1['status'] == 'CONFIRMED'
    assert ev2['status'] == 'CONFIRMED'


def test_trailer_with_two_20ft_containers_independent_passage_sessions(repository, snapshots, camera):
    """
    Mandatory Rule:
    2 physical containers = 2 independent detections = 2 unique track IDs = 2 container records = 2 event/visit IDs.
    Trailer with Container 1 (ABCU1234560, 22G1, track 101) and Container 2 (XYZU7654323, 22G1, track 102).
    Both must be tracked and recorded independently.
    """
    from app.events.passage_session import PassageSessionManager
    from app.domain import OCRRead

    mgr = PassageSessionManager()
    t0 = datetime(2026, 9, 15, 10, 0, 0)
    cameras = [camera.id]
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. Container 1 (Lead 20ft, track 101) detected and frames recorded
    det_c1 = Detection(camera.id, t0, 'container', 0.95, (350, 100, 600, 380))
    det_c1_num = Detection(camera.id, t0, 'container_number', 0.92, (400, 200, 580, 250))
    cand_c1 = mgr.record_frame(camera.id, camera.gate_id, 'IN', dummy_img, t0,
                               detections=[det_c1, det_c1_num],
                               gate_cameras=cameras, container_slot='CONTAINER_1', track_id=101)
    assert cand_c1 is not None

    # 2. Container 2 (Trail 20ft, track 102) detected on same trailer and frames recorded independently
    det_c2 = Detection(camera.id, t0, 'container', 0.94, (50, 100, 300, 380))
    det_c2_num = Detection(camera.id, t0, 'container_number', 0.90, (100, 200, 280, 250))
    cand_c2 = mgr.record_frame(camera.id, camera.gate_id, 'IN', dummy_img, t0,
                               detections=[det_c2, det_c2_num],
                               gate_cameras=cameras, container_slot='CONTAINER_2', track_id=102)
    assert cand_c2 is not None

    # 3. OCR results arrive independently
    mgr.record_ocr_result(camera.gate_id, OCRRead(raw_text='ABCU1234560', confidence=0.98), container_slot='CONTAINER_1')
    mgr.record_ocr_result(camera.gate_id, OCRRead(raw_text='22G1', confidence=0.95), container_slot='CONTAINER_1')

    mgr.record_ocr_result(camera.gate_id, OCRRead(raw_text='XYZU7654323', confidence=0.97), container_slot='CONTAINER_2')
    mgr.record_ocr_result(camera.gate_id, OCRRead(raw_text='22G1', confidence=0.94), container_slot='CONTAINER_2')

    # 4. Trailer clears the gate
    clear_t = t0 + timedelta(seconds=10)
    mgr.mark_clearing(camera.gate_id, t0 + timedelta(seconds=2))

    ready = mgr.check_and_finalize(clear_t, gate_clear_seconds=6.0)
    assert len(ready) == 2, f"Expected 2 independent sessions, got {len(ready)}"

    s1 = next(s for s in ready if s.container_slot == 'CONTAINER_1')
    s2 = next(s for s in ready if s.container_slot == 'CONTAINER_2')

    assert s1.track_id == 101
    assert s2.track_id == 102
    assert s1.confirmed_container_number == 'ABCU1234560'
    assert s2.confirmed_container_number == 'XYZU7654323'
    assert s1.confirmed_iso_code == '22G1'
    assert s2.confirmed_iso_code == '22G1'
    assert s1.event_id != s2.event_id

    # 5. Finalize both into repository database
    ev1_id = repository.finalize_passage_session(s1, snapshots)
    ev2_id = repository.finalize_passage_session(s2, snapshots)
    assert ev1_id != ev2_id

    ev1 = repository.event(ev1_id)
    ev2 = repository.event(ev2_id)
    assert ev1 is not None and ev2 is not None
    assert ev1['container_number'] == 'ABCU1234560'
    assert ev2['container_number'] == 'XYZU7654323'
    assert ev1['status'] == 'CONFIRMED'
    assert ev2['status'] == 'CONFIRMED'
    assert ev1['is_twin'] is False
    assert ev2['is_twin'] is False

    # Check that detection records retained distinct track IDs
    t1_ids = [d['track_id'] for d in ev1['detections']]
    t2_ids = [d['track_id'] for d in ev2['detections']]
    assert 101 in t1_ids
    assert 102 in t2_ids


def test_tracker_recovery_when_container_temporarily_missed():
    """
    Ensure the tracker continues tracking a container if it is temporarily
    missed in one or more frames and recovers the same track ID when visible again.
    """
    from app.detection.tracker import TemporalTracker

    tracker = TemporalTracker(ttl_seconds=5, match_iou=0.15)
    t0 = datetime(2026, 9, 15, 10, 0, 0)

    # Frame 0: Container detected
    det0 = Detection('cam-1', t0, 'container', 0.95, (100, 100, 300, 250))
    tracks0 = tracker.update([det0], t0)
    assert len(tracks0) == 1
    orig_track_id = tracks0[0].id

    # Frame 1: Container temporarily missed (empty detections)
    t1 = t0 + timedelta(milliseconds=100)
    tracks1 = tracker.update([], t1)
    assert len(tracks1) == 0

    # Frame 2: Container visible again, moved slightly (e.g. forward 20px)
    t2 = t0 + timedelta(milliseconds=200)
    det2 = Detection('cam-1', t2, 'container', 0.93, (120, 105, 320, 255))
    tracks2 = tracker.update([det2], t2)
    assert len(tracks2) == 1
    assert tracks2[0].id == orig_track_id, "Track ID must be preserved when container is recovered!"
    assert tracks2[0].hits == 2


def test_tracker_two_containers_maintain_separate_identities():
    """
    Two physical containers on the same trailer must maintain 2 separate track IDs.
    Temporary miss on one container must NOT cause swap or merge with the other.
    """
    from app.detection.tracker import TemporalTracker

    tracker = TemporalTracker(ttl_seconds=5, match_iou=0.15)
    t0 = datetime(2026, 9, 15, 10, 0, 0)

    # Frame 0: Container 1 (front) and Container 2 (rear)
    c1_f0 = Detection('cam-1', t0, 'container', 0.95, (350, 100, 550, 250))
    c2_f0 = Detection('cam-1', t0, 'container', 0.92, (100, 100, 300, 250))
    tracks0 = tracker.update([c1_f0, c2_f0], t0)
    assert len(tracks0) == 2
    t1_id = next(t.id for t in tracks0 if t.detection == c1_f0)
    t2_id = next(t.id for t in tracks0 if t.detection == c2_f0)
    assert t1_id != t2_id

    # Frame 1: Container 2 is temporarily missed; only Container 1 detected
    t1 = t0 + timedelta(milliseconds=100)
    c1_f1 = Detection('cam-1', t1, 'container', 0.95, (365, 100, 565, 250))
    tracks1 = tracker.update([c1_f1], t1)
    assert len(tracks1) == 1
    assert tracks1[0].id == t1_id

    # Frame 2: Both containers visible again
    t2 = t0 + timedelta(milliseconds=200)
    c1_f2 = Detection('cam-1', t2, 'container', 0.96, (380, 100, 580, 250))
    c2_f2 = Detection('cam-1', t2, 'container', 0.91, (130, 100, 330, 250))
    tracks2 = tracker.update([c1_f2, c2_f2], t2)
    assert len(tracks2) == 2
    recovered_t1 = next(t for t in tracks2 if t.detection == c1_f2)
    recovered_t2 = next(t for t in tracks2 if t.detection == c2_f2)
    assert recovered_t1.id == t1_id
    assert recovered_t2.id == t2_id
