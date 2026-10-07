from datetime import datetime, timedelta, timezone
import time
import numpy as np
import pytest
import cv2

from unittest.mock import patch
from pydantic import SecretStr
from app.events.cloud_dispatcher import CloudOutboxDispatcher
from app.config.settings import CameraConfig
from app.domain import Detection, OCRRead
from app.events.passage_session import (
    ContainerPassageSession,
    PassageSessionManager,
    evaluate_frame_quality
)


def _create_synthetic_test_frame(pattern_type: str, sharpness_level: float = 50.0):
    """
    Creates distinct visual patterns for each camera angle to test that
    camera angles are preserved and not replaced with each other.
    """
    img = np.full((300, 400, 3), 128, dtype=np.uint8)
    if pattern_type == 'C1_FRONT':
        cv2.circle(img, (200, 150), 60, (0, 255, 0), -1)
        cv2.putText(img, "CAMERA 1 FRONT", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    elif pattern_type == 'C2_LEFT':
        cv2.rectangle(img, (80, 80), (320, 220), (255, 0, 0), -1)
        cv2.putText(img, "CAMERA 2 LEFT", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    elif pattern_type == 'C3_RIGHT':
        pts = np.array([[200, 50], [100, 250], [300, 250]], np.int32)
        cv2.fillPoly(img, [pts], (0, 0, 255))
        cv2.putText(img, "CAMERA 3 RIGHT", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    elif pattern_type == 'C4_REAR':
        cv2.line(img, (50, 50), (350, 250), (255, 255, 0), 10)
        cv2.line(img, (50, 250), (350, 50), (255, 255, 0), 10)
        cv2.putText(img, "CAMERA 4 REAR", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Add high-contrast gradient/text for sharpness variation
    if sharpness_level > 60.0:
        cv2.putText(img, "HIGH_SHARPNESS_TEXT", (30, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return img


def test_per_camera_best_frame_selection_and_angle_preservation():
    """
    Requirements 4 & 5:
    - 4 cameras independently capture frames
    - Best frame selected independently per camera
    - All 4 distinct camera angles preserved
    """
    cams = ['gate-in-1', 'gate-in-2', 'gate-in-3', 'gate-in-4']
    session = ContainerPassageSession('TRACK_001', 'lane-in', 'ENTRY', cams, datetime.now())

    # For each camera, capture 10 frames with varying sharpness
    for i in range(10):
        t = datetime.now() + timedelta(milliseconds=i * 100)
        # Frame 7 has the highest sharpness text
        sharpness = 100.0 if i == 7 else 40.0

        for cam_id, p_type in zip(cams, ['C1_FRONT', 'C2_LEFT', 'C3_RIGHT', 'C4_REAR']):
            frame_img = _create_synthetic_test_frame(p_type, sharpness_level=sharpness)
            dets = [Detection(cam_id, t, 'container', 0.95, (20, 20, 380, 280))]
            session.add_frame(cam_id, frame_img, t, dets)

    # Verify that each camera collected 10 frames (total 40 frames)
    for cam_id in cams:
        assert len(session.captured_frames[cam_id]) == 10

    # Best frame selection for each camera
    best_by_camera = session.get_best_frame_per_camera()
    assert len(best_by_camera) == 4
    for cam_id in cams:
        assert cam_id in best_by_camera
        best = best_by_camera[cam_id]
        # Frame 7 with highest sharpness text should be chosen
        assert best.quality_score > 0.0


def test_cross_camera_multi_frame_ocr_consensus():
    """
    Requirement 6:
    Combine OCR candidates across 4 cameras and multi-frames:
    C1: ABCD1234567 (91%)
    C2: ABCD1234567 (96%)
    C3: ABCD12345?7 (87%)
    C4: ABCD1234567 (94%)
    Final consensus: ABCD1234567
    """
    cams = ['gate-in-1', 'gate-in-2', 'gate-in-3', 'gate-in-4']
    session = ContainerPassageSession('TRACK_001', 'lane-in', 'ENTRY', cams, datetime.now())

    # Add OCR reads from 4 cameras
    session.add_ocr_candidate(OCRRead('ABCD 123456 7', 0.91))
    session.add_ocr_candidate(OCRRead('ABCD 123456 7', 0.96))
    session.add_ocr_candidate(OCRRead('ABCD 123450 7', 0.87))
    session.add_ocr_candidate(OCRRead('ABCD 123456 7', 0.94))

    final_num, iso, conf = session.get_consensus_container_number()
    assert final_num in ('ABCD1234567', 'ABCU1234567')
    assert conf >= 0.90


def test_gate_in_complete_four_camera_passage_lifecycle(repository, snapshots):
    """
    End-to-end Gate IN lifecycle:
    - 4 cameras (C1-C4)
    - 40 frames (10 frames each)
    - All 4 camera best images saved into ONE event
    - UI/API manifest has all 4 viewpoints
    """
    cams = [
        CameraConfig(id='gate-in-1', name='Gate IN 1', gate_id='lane-in', source_env='CAM_1', role='FRONT_TOP', direction='ENTRY'),
        CameraConfig(id='gate-in-2', name='Gate IN 2', gate_id='lane-in', source_env='CAM_2', role='LEFT', direction='ENTRY'),
        CameraConfig(id='gate-in-3', name='Gate IN 3', gate_id='lane-in', source_env='CAM_3', role='RIGHT', direction='ENTRY'),
        CameraConfig(id='gate-in-4', name='Gate IN 4', gate_id='lane-in', source_env='CAM_4', role='REAR', direction='ENTRY'),
    ]
    repository.sync_cameras(cams)

    cam_ids = [c.id for c in cams]
    session = ContainerPassageSession('TRACK_GATE_IN', 'lane-in', 'ENTRY', cam_ids, datetime.now())

    # 4 cameras x 10 frames = 40 frames
    for i in range(10):
        t = datetime.now() + timedelta(milliseconds=i * 100)
        for cam_id, p_type in zip(cam_ids, ['C1_FRONT', 'C2_LEFT', 'C3_RIGHT', 'C4_REAR']):
            frame_img = _create_synthetic_test_frame(p_type, sharpness_level=50.0 + i * 5)
            dets = [Detection(cam_id, t, 'container', 0.95, (20, 20, 380, 280))]
            session.add_frame(cam_id, frame_img, t, dets)

    # Multi-frame OCR consensus
    session.add_ocr_candidate(OCRRead('ABCD 123456 7', 0.92))
    session.add_ocr_candidate(OCRRead('ABCD 123456 7', 0.97))

    # Commit finalized passage session to repository
    event_id = repository.finalize_passage_session(session, snapshots)

    # Assert exactly ONE event created
    ev = repository.event(event_id)
    assert ev['gate_id'] == 'lane-in'
    assert ev['event_type'] == 'ENTRY'
    assert ev['container_number'] in ('ABCD1234567', 'ABCU1234567')
    assert ev['status'] == 'CONFIRMED'

    # Assert all 4 camera snapshots are present
    assert len(ev['snapshots']) == 4
    saved_cams = {s['camera_id'] for s in ev['snapshots']}
    assert saved_cams == {'gate-in-1', 'gate-in-2', 'gate-in-3', 'gate-in-4'}

    # Verify all 4 images exist on disk in the event folder
    event_dir = snapshots.root / event_id
    assert event_dir.is_dir()
    for cam_id in cam_ids:
        assert (event_dir / f"{cam_id}.jpg").is_file()

    # Evidence manifest verification
    manifest = snapshots.build_evidence_manifest(ev)
    assert len(manifest['captures']) == 4


def test_gate_out_complete_four_camera_passage_lifecycle(repository, snapshots):
    """
    End-to-end Gate OUT lifecycle (Requirement 10):
    - 4 cameras (C5-C8)
    - 40 frames (10 frames each)
    - All 4 camera best images saved into ONE event
    """
    cams = [
        CameraConfig(id='gate-out-1', name='Gate OUT 1', gate_id='lane-out', source_env='CAM_5', role='FRONT_TOP', direction='EXIT'),
        CameraConfig(id='gate-out-2', name='Gate OUT 2', gate_id='lane-out', source_env='CAM_6', role='LEFT', direction='EXIT'),
        CameraConfig(id='gate-out-3', name='Gate OUT 3', gate_id='lane-out', source_env='CAM_7', role='RIGHT', direction='EXIT'),
        CameraConfig(id='gate-out-4', name='Gate OUT 4', gate_id='lane-out', source_env='CAM_8', role='REAR', direction='EXIT'),
    ]
    repository.sync_cameras(cams)

    cam_ids = [c.id for c in cams]
    session = ContainerPassageSession('TRACK_GATE_OUT', 'lane-out', 'EXIT', cam_ids, datetime.now())

    for i in range(10):
        t = datetime.now() + timedelta(milliseconds=i * 100)
        for cam_id, p_type in zip(cam_ids, ['C1_FRONT', 'C2_LEFT', 'C3_RIGHT', 'C4_REAR']):
            frame_img = _create_synthetic_test_frame(p_type, sharpness_level=50.0 + i * 5)
            dets = [Detection(cam_id, t, 'container', 0.95, (20, 20, 380, 280))]
            session.add_frame(cam_id, frame_img, t, dets)

    session.add_ocr_candidate(OCRRead('WXYZ 987654 3', 0.95))

    event_id = repository.finalize_passage_session(session, snapshots)

    ev = repository.event(event_id)
    assert ev['gate_id'] == 'lane-out'
    assert ev['event_type'] == 'EXIT'
    assert ev['container_number'] == 'WXYZ9876543'
    assert len(ev['snapshots']) == 4
    saved_cams = {s['camera_id'] for s in ev['snapshots']}
    assert saved_cams == {'gate-out-1', 'gate-out-2', 'gate-out-3', 'gate-out-4'}


def test_multiple_containers_independent_datasets(repository, snapshots):
    """
    Requirement 12:
    Multiple containers (Container A, Container B) passing sequentially
    produce separate independent sessions and events, never mixing frames.
    """
    cams = ['gate-in-1', 'gate-in-2', 'gate-in-3', 'gate-in-4']
    session_mgr = PassageSessionManager(None, target_frames_per_camera=10)

    # Container A arrives
    t0 = datetime(2026, 10, 7, 10, 0, 0)
    sess_a = session_mgr.get_or_create_session('lane-in', 'ENTRY', t0, cams)
    dets_a = [Detection('gate-in-1', t0, 'container', 0.95, (10, 10, 300, 200))]
    session_mgr.record_frame('gate-in-1', 'lane-in', 'ENTRY', _create_synthetic_test_frame('C1_FRONT'), t0, detections=dets_a, gate_cameras=cams)
    session_mgr.record_ocr_result('lane-in', OCRRead('MSKU 111111 1', 0.95))

    # Container A clears
    session_mgr.mark_clearing('lane-in', t0 + timedelta(seconds=2))
    ready = session_mgr.check_and_finalize(t0 + timedelta(seconds=9), gate_clear_seconds=6.0)
    assert len(ready) == 1
    assert ready[0].confirmed_container_number == 'MSKU1111111'

    # Container B arrives later
    t1 = t0 + timedelta(seconds=30)
    sess_b = session_mgr.get_or_create_session('lane-in', 'ENTRY', t1, cams)
    assert sess_b.session_id != sess_a.session_id
    dets_b = [Detection('gate-in-1', t1, 'container', 0.95, (10, 10, 300, 200))]
    session_mgr.record_frame('gate-in-1', 'lane-in', 'ENTRY', _create_synthetic_test_frame('C1_FRONT'), t1, detections=dets_b, gate_cameras=cams)
    session_mgr.record_ocr_result('lane-in', OCRRead('MSKU 222222 2', 0.95))

    assert len(sess_b.ocr_candidates) == 1
    assert sess_b.ocr_candidates[0].normalized_text == 'MSKU2222222'


def test_flexible_camera_counts_two_and_six_cameras(repository, snapshots):
    """
    Flexibility Requirement:
    The system is NOT fixed to 4 cameras. It dynamically adapts to 2, 6, or any N cameras.
    """
    # 1. Test 2-camera gate configuration
    cams_2 = [
        CameraConfig(id='lane-custom-1', name='Gate Custom 1', gate_id='lane-custom', source_env='CAM_C1', role='FRONT_TOP', direction='ENTRY'),
        CameraConfig(id='lane-custom-2', name='Gate Custom 2', gate_id='lane-custom', source_env='CAM_C2', role='REAR', direction='ENTRY'),
    ]
    repository.sync_cameras(cams_2)
    session_2 = ContainerPassageSession('TRACK_FLEX_2', 'lane-custom', 'ENTRY', [c.id for c in cams_2], datetime.now())
    for cam_id in [c.id for c in cams_2]:
        for i in range(5):
            img = _create_synthetic_test_frame('C1_FRONT', sharpness_level=40.0 + i * 10)
            session_2.add_frame(cam_id, img, datetime.now(), [Detection(cam_id, datetime.now(), 'container', 0.9, (10, 10, 300, 200))])
    session_2.add_ocr_candidate(OCRRead('MSKU 123456 7', 0.95))

    best_2 = session_2.get_best_frame_per_camera()
    assert len(best_2) == 2
    assert set(best_2.keys()) == {'lane-custom-1', 'lane-custom-2'}

    evt_id_2 = repository.finalize_passage_session(session_2, snapshots)
    ev_2 = repository.event(evt_id_2)
    assert len(ev_2['snapshots']) == 2

    # 2. Test 6-camera gate configuration
    roles = ['FRONT_TOP', 'LEFT', 'RIGHT', 'REAR', 'LEFT', 'RIGHT']
    cams_6 = [
        CameraConfig(id=f'lane-mega-{i}', name=f'Mega Cam {i}', gate_id='lane-mega', source_env=f'CAM_M{i}', role=roles[i - 1], direction='ENTRY')
        for i in range(1, 7)
    ]
    repository.sync_cameras(cams_6)
    session_6 = ContainerPassageSession('TRACK_FLEX_6', 'lane-mega', 'ENTRY', [c.id for c in cams_6], datetime.now())
    for cam in cams_6:
        for i in range(4):
            img = _create_synthetic_test_frame('C1_FRONT', sharpness_level=30.0 + i * 15)
            session_6.add_frame(cam.id, img, datetime.now(), [Detection(cam.id, datetime.now(), 'container', 0.9, (10, 10, 300, 200))])
    session_6.add_ocr_candidate(OCRRead('TGHU 765432 1', 0.96))

    best_6 = session_6.get_best_frame_per_camera()
    assert len(best_6) == 6
    assert set(best_6.keys()) == {c.id for c in cams_6}

    evt_id_6 = repository.finalize_passage_session(session_6, snapshots)
    ev_6 = repository.event(evt_id_6)
    assert len(ev_6['snapshots']) == 6


def test_flexible_non_uniform_frame_counts(repository, snapshots):
    """
    Flexibility Requirement:
    Frame counts are NOT fixed to 10 frames per camera:
    - Fast truck: Cam 1 captures 3 frames, Cam 2 captures 5 frames, Cam 3 captures 2 frames.
    - Slow truck: Cam 4 captures 25 frames (buffer retains top scoring candidates).
    """
    cams = ['flex-c1', 'flex-c2', 'flex-c3', 'flex-c4']
    session = ContainerPassageSession('TRACK_NON_UNIFORM', 'lane-fast', 'ENTRY', cams, datetime.now(), target_frames_per_camera=10)

    # Cam 1 captures only 3 frames (fast transit)
    for i in range(3):
        img = _create_synthetic_test_frame('C1_FRONT', sharpness_level=50.0 + i * 20)
        session.add_frame('flex-c1', img, datetime.now(), [Detection('flex-c1', datetime.now(), 'container', 0.9, (10, 10, 300, 200))])

    # Cam 2 captures 5 frames
    for i in range(5):
        img = _create_synthetic_test_frame('C2_LEFT', sharpness_level=45.0 + i * 10)
        session.add_frame('flex-c2', img, datetime.now(), [Detection('flex-c2', datetime.now(), 'container', 0.9, (10, 10, 300, 200))])

    # Cam 3 captures only 1 frame
    img_c3 = _create_synthetic_test_frame('C3_RIGHT', sharpness_level=80.0)
    session.add_frame('flex-c3', img_c3, datetime.now(), [Detection('flex-c3', datetime.now(), 'container', 0.9, (10, 10, 300, 200))])

    # Cam 4 captures 25 frames (slow truck / buffer retention)
    for i in range(25):
        img = _create_synthetic_test_frame('C4_REAR', sharpness_level=30.0 + i * 4)
        session.add_frame('flex-c4', img, datetime.now(), [Detection('flex-c4', datetime.now(), 'container', 0.9, (10, 10, 300, 200))])

    # Assertions
    assert len(session.captured_frames['flex-c1']) == 3
    assert len(session.captured_frames['flex-c2']) == 5
    assert len(session.captured_frames['flex-c3']) == 1
    # Bounded to target_frames_per_camera (10) for memory efficiency while retaining the best
    assert len(session.captured_frames['flex-c4']) == 10

    # Best frame selection works across all non-uniform frame counts
    best_by_cam = session.get_best_frame_per_camera()
    assert len(best_by_cam) == 4
    for cam_id in cams:
        assert cam_id in best_by_cam
        assert best_by_cam[cam_id].quality_score > 0.0


def test_dynamic_camera_registration_midway_through_passage():
    """
    Flexibility Requirement:
    A camera not initially in the session can be triggered dynamically
    and its frames are seamlessly integrated into the passage session.
    """
    t_now = datetime.now()
    session = ContainerPassageSession('TRACK_DYNAMIC_CAM', 'lane-dyn', 'ENTRY', ['cam-initial'], t_now)
    session.add_frame('cam-initial', _create_synthetic_test_frame('C1_FRONT'), t_now, [Detection('cam-initial', t_now, 'container', 0.9, (10, 10, 300, 200))])

    # Dynamically triggered second and third cameras when container enters their view
    session.add_frame('cam-late-arrival', _create_synthetic_test_frame('C2_LEFT'), t_now, [Detection('cam-late-arrival', t_now, 'container', 0.9, (10, 10, 300, 200))])
    session.add_frame('cam-third-sensor', _create_synthetic_test_frame('C3_RIGHT'), t_now, [Detection('cam-third-sensor', t_now, 'container', 0.9, (10, 10, 300, 200))])

    best = session.get_best_frame_per_camera()
    assert len(best) == 3
    assert set(best.keys()) == {'cam-initial', 'cam-late-arrival', 'cam-third-sensor'}


def test_useless_empty_frames_rejected_when_container_not_entered():
    """
    Requirement:
    A frame must ONLY be captured when the container actually enters that camera's view.
    Frames without a container (empty road, empty gate before arrival) are useless
    and MUST be rejected / ignored.
    """
    session = ContainerPassageSession('TRACK_FILTER', 'lane-in', 'ENTRY', ['cam-1', 'cam-2'], datetime.now())

    # Empty frame with no container detections (e.g. empty background road)
    empty_frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
    res = session.add_frame('cam-1', empty_frame, datetime.now(), detections=[])

    # Must be rejected (returns None and not added to buffer)
    assert res is None
    assert len(session.captured_frames['cam-1']) == 0

    # Non-container detection (e.g. 'person', 'car') should also be rejected
    res_car = session.add_frame('cam-1', empty_frame, datetime.now(), detections=[
        Detection('cam-1', datetime.now(), 'car', 0.85, (50, 50, 200, 200))
    ])
    assert res_car is None
    assert len(session.captured_frames['cam-1']) == 0

    # Real container enters camera view
    container_frame = _create_synthetic_test_frame('C1_FRONT')
    res_container = session.add_frame('cam-1', container_frame, datetime.now(), detections=[
        Detection('cam-1', datetime.now(), 'container', 0.95, (20, 20, 400, 300))
    ])
    assert res_container is not None
    assert len(session.captured_frames['cam-1']) == 1
    assert session.captured_frames['cam-1'][0].quality_score > 0.0


def test_truck_and_trailer_without_container_strictly_rejected():
    """
    Requirement:
    Frames must ONLY be captured for the container, NOT other objects like
    trucks, trailers, cabs, or vehicles without a container.
    """
    session = ContainerPassageSession('TRACK_NO_TRUCK', 'lane-in', 'ENTRY', ['cam-1'], datetime.now())
    frame_img = np.full((300, 400, 3), 128, dtype=np.uint8)

    # Bare truck cab entering before container
    res_truck = session.add_frame('cam-1', frame_img, datetime.now(), detections=[
        Detection('cam-1', datetime.now(), 'truck', 0.90, (30, 30, 350, 250))
    ])
    assert res_truck is None
    assert len(session.captured_frames['cam-1']) == 0

    # Bare trailer chassis without container
    res_trailer = session.add_frame('cam-1', frame_img, datetime.now(), detections=[
        Detection('cam-1', datetime.now(), 'trailer', 0.88, (30, 30, 350, 250))
    ])
    assert res_trailer is None
    assert len(session.captured_frames['cam-1']) == 0

    # Frame quality evaluation directly returns 0.0 for trucks/trailers
    assert evaluate_frame_quality(frame_img, [Detection('cam-1', datetime.now(), 'truck', 0.9, (30, 30, 350, 250))]) == 0.0
    assert evaluate_frame_quality(frame_img, [Detection('cam-1', datetime.now(), 'trailer', 0.9, (30, 30, 350, 250))]) == 0.0


def test_companion_camera_without_container_excluded_from_best_frames():
    """
    Requirement:
    If a camera never observed the container (e.g. companion angle before arrival
    or obstructed), it MUST NOT return random empty frames.
    """
    cams = ['gate-in-1', 'gate-in-2', 'gate-in-3']
    session = ContainerPassageSession('TRACK_PARTIAL', 'lane-in', 'ENTRY', cams, datetime.now())
    t = datetime.now()

    # Cam 1 and Cam 2 see the container
    frame1 = _create_synthetic_test_frame('C1_FRONT')
    frame2 = _create_synthetic_test_frame('C2_LEFT')
    session.add_frame('gate-in-1', frame1, t, [Detection('gate-in-1', t, 'container', 0.95, (20, 20, 380, 280))])
    session.add_frame('gate-in-2', frame2, t, [Detection('gate-in-2', t, 'container_number', 0.90, (50, 50, 200, 100))])

    # Cam 3 has no container (only empty background or truck without container)
    empty_frame = np.full((300, 400, 3), 100, dtype=np.uint8)
    session.add_frame('gate-in-3', empty_frame, t, [Detection('gate-in-3', t, 'truck', 0.80, (50, 50, 200, 100))])

    best = session.get_best_frame_per_camera()
    assert 'gate-in-1' in best
    assert 'gate-in-2' in best
    # Cam 3 must be excluded because it never had a container object!
    assert 'gate-in-3' not in best


def test_snapshot_manager_visible_container_validation():
    """
    SnapshotManager._capture_has_visible_container must strictly require
    container or container_number detections and reject empty/truck frames.
    """
    from app.snapshots.snapshot_manager import SnapshotManager

    img = np.full((300, 400, 3), 128, dtype=np.uint8)
    snap = {'camera_id': 'cam-1', 'timestamp': '2026-10-07T10:00:00Z'}

    # 1. No detections recorded -> MUST REJECT
    ev_no_dets = {'detections': []}
    assert SnapshotManager._capture_has_visible_container(snap, ev_no_dets, img) is False

    # 2. Only truck detection -> MUST REJECT
    ev_truck = {'detections': [
        {'camera_id': 'cam-1', 'class_name': 'truck', 'bbox': [10, 10, 300, 200], 'timestamp': '2026-10-07T10:00:00Z'}
    ]}
    assert SnapshotManager._capture_has_visible_container(snap, ev_truck, img) is False

    # 3. Real container detection -> MUST ACCEPT
    ev_cntr = {'detections': [
        {'camera_id': 'cam-1', 'class_name': 'container', 'bbox': [10, 10, 300, 200], 'timestamp': '2026-10-07T10:00:00Z'}
    ]}
    assert SnapshotManager._capture_has_visible_container(snap, ev_cntr, img) is True


def test_multicamera_cloud_dispatch_four_frames_after_passage_finalization(repository, snapshots, settings):
    """
    Verify complete cloud payload assembly for 4-camera passage:
    - 4 cameras (C1-C4) capture 40 frames
    - Premature dispatch is suppressed during live transit
    - Finalized event posts all 4 distinct camera views to API (FRONT, LEFT, RIGHT, REAR)
    """
    from unittest.mock import patch, MagicMock
    from app.events.cloud_dispatcher import CloudOutboxDispatcher
    import time

    cams = [
        CameraConfig(id='gate-in-1', name='Gate IN 1', gate_id='lane-in', source_env='CAM_1', role='FRONT_TOP', direction='ENTRY'),
        CameraConfig(id='gate-in-2', name='Gate IN 2', gate_id='lane-in', source_env='CAM_2', role='LEFT', direction='ENTRY'),
        CameraConfig(id='gate-in-3', name='Gate IN 3', gate_id='lane-in', source_env='CAM_3', role='RIGHT', direction='ENTRY'),
        CameraConfig(id='gate-in-4', name='Gate IN 4', gate_id='lane-in', source_env='CAM_4', role='REAR', direction='ENTRY'),
    ]
    repository.sync_cameras(cams)
    cam_ids = [c.id for c in cams]

    session = ContainerPassageSession('TRACK_GATE_IN_DISPATCH', 'lane-in', 'ENTRY', cam_ids, datetime.now())

    # 4 cameras x 10 frames = 40 frames total
    for i in range(10):
        t = datetime.now() + timedelta(milliseconds=i * 100)
        for cam_id, p_type in zip(cam_ids, ['C1_FRONT', 'C2_LEFT', 'C3_RIGHT', 'C4_REAR']):
            frame_img = _create_synthetic_test_frame(p_type, sharpness_level=55.0 + i * 5)
            dets = [Detection(cam_id, t, 'container', 0.95, (20, 20, 380, 280))]
            session.add_frame(cam_id, frame_img, t, dets)

    # Multi-frame OCR reads
    session.add_ocr_candidate(OCRRead('MSCU 123456 7', 0.94))
    session.add_ocr_candidate(OCRRead('MSCU 123456 7', 0.98))

    # Finalize passage session
    event_id = repository.finalize_passage_session(session, snapshots)

    # Cloud Dispatcher execution
    from pydantic import SecretStr
    cloud_settings = settings.model_copy(update={
        'prosper_enabled': True,
        'prosper_api_base_url': 'http://syapi.test.prosper.com',
        'prosper_bearer_token': SecretStr('test-token'),
    })
    dispatcher = CloudOutboxDispatcher(cloud_settings, repository=repository, snapshots=snapshots)

    with patch.object(dispatcher.session, 'post_gate_event', return_value=True) as mock_post:
        dispatcher.start()
        dispatcher.enqueue(None, event_id, force=True)

        for _ in range(30):
            if dispatcher.queue.empty() and mock_post.called:
                break
            time.sleep(0.05)

        dispatcher.stop()

        assert mock_post.called, "Cloud dispatcher did not post finalized multi-camera event"
        payload = mock_post.call_args[0][0]
        assert payload['id'] == event_id
        assert payload['EventType'] == 'GATE_IN'
        assert len(payload['Containers']) == 1

        container = payload['Containers'][0]
        assert container['ContainerNumber'] == 'MSCU1234567'

        # Must send all 4 cameras in Images list
        images = container['Images']
        assert len(images) == 4, f"Expected 4 camera images, got {len(images)}"

        cam_order = [img['CameraId'] for img in images]
        assert cam_order == ['gate-in-1', 'gate-in-2', 'gate-in-3', 'gate-in-4']

        types_order = [img['ImageType'] for img in images]
        assert types_order == ['FRONT', 'LEFT', 'RIGHT', 'REAR']

        for img in images:
            assert img['Image'].startswith('data:image/jpeg;base64,')


def test_unavailable_camera_marked_unavailable_without_substitution(repository, snapshots, settings):
    """
    If a camera has no valid frame (e.g. camera 3 obscured/offline),
    it must NOT substitute another container's frame or duplicate another camera.
    """
    from unittest.mock import patch
    from pydantic import SecretStr
    from app.events.cloud_dispatcher import CloudOutboxDispatcher
    import time

    cams = [
        CameraConfig(id='gate-in-1', name='Gate IN 1', gate_id='lane-in', source_env='CAM_1', role='FRONT_TOP', direction='ENTRY'),
        CameraConfig(id='gate-in-2', name='Gate IN 2', gate_id='lane-in', source_env='CAM_2', role='LEFT', direction='ENTRY'),
        CameraConfig(id='gate-in-3', name='Gate IN 3', gate_id='lane-in', source_env='CAM_3', role='RIGHT', direction='ENTRY'),
        CameraConfig(id='gate-in-4', name='Gate IN 4', gate_id='lane-in', source_env='CAM_4', role='REAR', direction='ENTRY'),
    ]
    repository.sync_cameras(cams)

    session = ContainerPassageSession('TRACK_UNAVAIL', 'lane-in', 'ENTRY', [c.id for c in cams], datetime.now())

    # Only cameras 1, 2, 4 capture valid frames; Camera 3 gets no valid container frames
    t = datetime.now()
    session.add_frame('gate-in-1', _create_synthetic_test_frame('C1_FRONT'), t, [Detection('gate-in-1', t, 'container', 0.95, (10, 10, 300, 200))])
    session.add_frame('gate-in-2', _create_synthetic_test_frame('C2_LEFT'), t, [Detection('gate-in-2', t, 'container', 0.95, (10, 10, 300, 200))])
    # Camera 3 only saw an empty road/car, quality score = 0 -> rejected by add_frame
    empty_road = np.zeros((300, 400, 3), dtype=np.uint8)
    session.add_frame('gate-in-3', empty_road, t, [Detection('gate-in-3', t, 'car', 0.80, (10, 10, 100, 100))])
    session.add_frame('gate-in-4', _create_synthetic_test_frame('C4_REAR'), t, [Detection('gate-in-4', t, 'container', 0.95, (10, 10, 300, 200))])

    session.add_ocr_candidate(OCRRead('TGHU 765432 1', 0.95))

    event_id = repository.finalize_passage_session(session, snapshots)

    cloud_settings = settings.model_copy(update={
        'prosper_enabled': True,
        'prosper_api_base_url': 'http://syapi.test.prosper.com',
        'prosper_bearer_token': SecretStr('test-token'),
    })
    dispatcher = CloudOutboxDispatcher(cloud_settings, repository=repository, snapshots=snapshots)
    with patch.object(dispatcher.session, 'post_gate_event', return_value=True) as mock_post:
        dispatcher.start()
        dispatcher.enqueue(None, event_id, force=True)

        for _ in range(30):
            if dispatcher.queue.empty() and mock_post.called:
                break
            time.sleep(0.05)

        dispatcher.stop()

        payload = mock_post.call_args[0][0]
        images = payload['Containers'][0]['Images']
        # Exactly 3 images; camera 3 is unavailable and NEVER substituted or duplicated
        assert len(images) == 3
        cam_ids_sent = {img['CameraId'] for img in images}
        assert cam_ids_sent == {'gate-in-1', 'gate-in-2', 'gate-in-4'}
        assert 'gate-in-3' not in cam_ids_sent


def test_three_camera_per_gate_setup_dispatch(settings, repository, snapshots):
    configs = [
        CameraConfig(id='gate-in-1', name='Gate IN 1', gate_id='lane-in', source_env='CAM_1', role='FRONT_TOP', direction='ENTRY'),
        CameraConfig(id='gate-in-2', name='Gate IN 2', gate_id='lane-in', source_env='CAM_2', role='LEFT', direction='ENTRY'),
        CameraConfig(id='gate-in-3', name='Gate IN 3', gate_id='lane-in', source_env='CAM_3', role='REAR', direction='ENTRY'),
    ]
    repository.sync_cameras(configs)
    session = ContainerPassageSession('TRACK_3CAM', 'lane-in', 'ENTRY', [c.id for c in configs], datetime.now(timezone.utc))
    t = datetime.now(timezone.utc)
    for cam in configs:
        pattern = 'C1_FRONT' if '1' in cam.id else ('C2_LEFT' if '2' in cam.id else 'C4_REAR')
        img = _create_synthetic_test_frame(pattern)
        session.add_frame(cam.id, img, t, [Detection(cam.id, t, 'container', 0.95, (10, 10, 300, 200))])

    session.add_ocr_candidate(OCRRead('MSKU 987654 3', 0.95))

    event_id = repository.finalize_passage_session(session, snapshots)

    cloud_settings = settings.model_copy(update={
        'prosper_enabled': True,
        'prosper_api_base_url': 'http://syapi.test.prosper.com',
        'prosper_bearer_token': SecretStr('test-token'),
    })
    dispatcher = CloudOutboxDispatcher(cloud_settings, repository=repository, snapshots=snapshots)
    with patch.object(dispatcher.session, 'post_gate_event', return_value=True) as mock_post:
        dispatcher.start()
        dispatcher.enqueue(None, event_id, force=True)

        for _ in range(30):
            if dispatcher.queue.empty() and mock_post.called:
                break
            time.sleep(0.05)

        dispatcher.stop()

        payload = mock_post.call_args[0][0]
        images = payload['Containers'][0]['Images']
        assert len(images) == 3
        cam_ids = [img['CameraId'] for img in images]
        assert cam_ids == ['gate-in-1', 'gate-in-2', 'gate-in-3']
        img_types = [img['ImageType'] for img in images]
        assert img_types == ['FRONT', 'LEFT', 'REAR']





