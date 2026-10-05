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
    view.confirm(view.current.origin, 'evt-single', 'MSCU1234567')
    assert view.current.closed is True
    assert view.current.container_number == 'MSCU1234567'


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
    val1 = Validation('MSCU1234567', 'MSCU1234567', 0.99, True, True, 'VALID', size_type='22G1')
    obs1 = Observation(f'container:{camera.id}:1', camera.gate_id, 1, det1, val1, 'IN',
                       frame1.image, frame1.timestamp, True, slot_position='SINGLE', is_twin=False)

    # 1. Container 1 creates its independent gate event
    event_id1, created1 = repo.save_observation(obs1, snapshots, 30)
    assert created1 is True
    assert event_id1 is not None

    # 2. Container 2 on same trailer is processed independently as its own container / tracker ID
    frame2, (det2,) = make_frame_detections(1, [(10, 20, 45, 80)], camera_id=camera.id)
    val2 = Validation('TGHU7654321', 'TGHU7654321', 0.98, True, True, 'VALID', size_type='22G1')
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
    assert ev1['container_number'] == 'MSCU1234567'
    assert ev2['container_number'] == 'TGHU7654321'
    assert ev1['container_number_2'] is None
    assert ev2['container_number_2'] is None
    assert ev1['status'] == 'CONFIRMED'
    assert ev2['status'] == 'CONFIRMED'
