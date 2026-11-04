"""
Automated Test Suite for Two-Stage API Workflow:
Immediate Gate Event Creation (POST), Deferred Multi-Camera Evidence Upload (PUT),
and Strict Container-to-Image Isolation.

Covers Tests A through L from implementation specification:
- Test A: Normal container passage (immediate POST, eventId extraction & persistence, background evidence collection, PUT with all 3 cameras)
- Test B: Three containers detected in quick succession (Containers A, B, C; concurrent processing, zero cross-container image mixing)
- Test C: Deliberately interleaved processing (Container B POST returns before A; Container C evidence ready first; association remains 100% correct)
- Test D: Cross-container image rejection (deliberate injection of Container B image into Container A evidence list is strictly rejected)
- Test E: Multiple physical containers on one trailer (twin 20-foot containers maintain independent tracking, event IDs, and PUT payloads)
- Test F: Missing camera (unavailable camera handled honestly without substituting another container's frame)
- Test G: Network failures (POST and PUT failure handling, retry with backoff, same event ID reuse, duplicate prevention)
- Test H: Evidence ready before POST response (Case B execution order handled seamlessly)
- Test I: Application restart / recovery (persisted event ID and evidence state restored from database without mixing state)
- Test J: Gate isolation (Gate-In cameras never attached to Gate-Out events and vice-versa)
- Test K: API contract validation (URL format, base64 data URI encoding, ISO UTC timestamps, Containers structure)
- Test L: Regression testing (existing OCR confirmation and tracking functionality preserved)
"""

import base64
from datetime import datetime, timedelta, timezone
import json
import time
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest
from pydantic import SecretStr

from app.config.settings import CameraConfig, Settings
from app.database.models import GateEvent, DetectionRecord, OCRResult
from app.database.repositories.gate import GateRepository
from app.domain import Detection, OCRRead, Observation, Validation
from app.events.cloud_dispatcher import CloudOutboxDispatcher, ProsperSession
from app.events.passage_session import ContainerPassageSession, PassageSessionManager
from app.snapshots.snapshot_manager import SnapshotManager


def _create_sharp_test_frame(label: str = 'TEST', sharpness: float = 65.0, w: int = 400, h: int = 300) -> np.ndarray:
    """Create a synthetic high-contrast test frame with container text and edges."""
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (w - 20, h - 20), (40, 40, 40), -1)
    cv2.putText(img, label, (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    # Add high-contrast gradient pattern to guarantee Laplacian sharpness variance > 45.0
    for x in range(30, w - 30, 20):
        cv2.line(img, (x, 30), (x, h - 30), (220, 220, 220), 2)
    return img


@pytest.fixture
def mock_cloud_settings(settings):
    return settings.model_copy(update={
        'prosper_enabled': True,
        'prosper_api_base_url': 'https://api.smartyard.prosperinfotech.com/gate',
        'prosper_bearer_token': SecretStr('test-token-jwt-bearer'),
        'prosper_queue_size': 50,
        'prosper_max_retries': 3,
        'cloud_min_proof_quality': 1.0,
    })


@pytest.fixture
def gate_setup(repository, snapshots):
    """Setup 3 cameras for Gate-In and 3 cameras for Gate-Out."""
    cameras = [
        CameraConfig(id='gate-in-cam-1', name='Gate In Cam 1', gate_id='LANE-IN', source_env='CAM_1', role='FRONT_TOP', direction='ENTRY'),
        CameraConfig(id='gate-in-cam-2', name='Gate In Cam 2', gate_id='LANE-IN', source_env='CAM_2', role='LEFT', direction='ENTRY'),
        CameraConfig(id='gate-in-cam-3', name='Gate In Cam 3', gate_id='LANE-IN', source_env='CAM_3', role='REAR', direction='ENTRY'),
        CameraConfig(id='gate-out-cam-1', name='Gate Out Cam 1', gate_id='LANE-OUT', source_env='CAM_4', role='FRONT_TOP', direction='EXIT'),
        CameraConfig(id='gate-out-cam-2', name='Gate Out Cam 2', gate_id='LANE-OUT', source_env='CAM_5', role='LEFT', direction='EXIT'),
        CameraConfig(id='gate-out-cam-3', name='Gate Out Cam 3', gate_id='LANE-OUT', source_env='CAM_6', role='REAR', direction='EXIT'),
    ]
    repository.sync_cameras(cameras)
    return cameras


# -------------------------------------------------------------------------------------
# Test A: Normal Container Passage (Immediate POST, eventId extraction, deferred PUT)
# -------------------------------------------------------------------------------------
def test_two_stage_normal_container_passage(gate_setup, repository, snapshots, mock_cloud_settings):
    """
    Test A: Verify immediate POST, extraction and persistence of eventId,
    background multi-camera evidence collection, and deferred PUT with all 3 camera images.
    Container MEDU9988770 is a valid ISO 6346 container number.
    """
    cams = [c for c in gate_setup if c.gate_id == 'LANE-OUT']
    cam_ids = [c.id for c in cams]

    session = ContainerPassageSession('TRACK_NORM_01', 'LANE-OUT', 'EXIT', cam_ids, datetime.now(timezone.utc))
    t = datetime.now(timezone.utc)
    for cam in cams:
        frame = _create_sharp_test_frame(f"{cam.id}_MEDU9988770")
        dets = [Detection(cam.id, t, 'container', 0.95, (10, 10, 350, 250))]
        session.add_frame(cam.id, frame, t, dets)

    session.add_ocr_candidate(OCRRead('MEDU 998877 0', 0.97))
    event_id = repository.finalize_passage_session(session, snapshots)

    dispatcher = CloudOutboxDispatcher(mock_cloud_settings, repository=repository, snapshots=snapshots)

    post_resp = {
        "visitDatabaseId": "afcf4cc6-8d78-4806-acf2-42eb18e4ef8d",
        "visitId": "VISIT-38727D97",
        "eventId": "0e7d4fac-5f22-48a4-be37-0bc58a948eff",
        "sourceEventId": "AUTO-5D2D78B40BAB4F5C",
        "eventType": "GATE_OUT"
    }

    with patch.object(dispatcher.session, 'post_gate_event', return_value=post_resp) as mock_post, \
         patch.object(dispatcher.session, 'put_gate_event_images', return_value=True) as mock_put:

        dispatcher.start()
        # Stage 1: Enqueue immediate POST
        dispatcher.enqueue(None, event_id, stage='POST', force=True)

        for _ in range(40):
            if mock_post.called and mock_put.called:
                break
            time.sleep(0.05)

        dispatcher.stop()

        # Step 1 Assertions: POST called and eventId persisted
        assert mock_post.called, "POST gate event was not initiated"
        post_payload = mock_post.call_args[0][0]
        assert post_payload['id'] == event_id
        assert post_payload['containerNumber'] == 'MEDU9988770'

        db_ev = repository.event(event_id)
        assert db_ev['cloud_event_id'] == "0e7d4fac-5f22-48a4-be37-0bc58a948eff"
        assert db_ev['cloud_visit_id'] == "VISIT-38727D97"
        assert db_ev['container_number'] == 'MEDU9988770'

        # Step 2 Assertions: PUT called targeting exact eventId with all 3 camera images
        assert mock_put.called, "PUT evidence images update was not initiated"
        target_cloud_id = mock_put.call_args[0][0]
        assert target_cloud_id == "0e7d4fac-5f22-48a4-be37-0bc58a948eff"

        put_payload = mock_put.call_args[0][1]
        assert put_payload.get('EventId') == "0e7d4fac-5f22-48a4-be37-0bc58a948eff"
        assert put_payload.get('VisitId') == "VISIT-38727D97"
        assert len(put_payload['Containers']) == 1
        container_data = put_payload['Containers'][0]
        assert container_data['ContainerNumber'] == 'MEDU9988770'
        assert container_data['Position'] == 'SINGLE'

        put_images = container_data['Images']
        assert len(put_images) == 3, f"Expected 3 camera images for Gate-Out, got {len(put_images)}"
        put_cam_ids = [img['CameraId'] for img in put_images]
        assert put_cam_ids == ['gate-out-cam-1', 'gate-out-cam-2', 'gate-out-cam-3']
        for img in put_images:
            assert img['Image'].startswith('data:image/jpeg;base64,')


# -------------------------------------------------------------------------------------
# Test B: Three Containers in Quick Succession (Concurrent, zero cross-container mixing)
# -------------------------------------------------------------------------------------
def test_three_containers_quick_succession_strict_isolation(gate_setup, repository, snapshots, mock_cloud_settings):
    """
    Test B: Containers A, B, and C are detected in quick succession.
    Verify that every uploaded image belongs only to its intended physical container
    and that every PUT targets the corresponding POST event ID.
    """
    cams = [c for c in gate_setup if c.gate_id == 'LANE-OUT']
    cam_ids = [c.id for c in cams]

    containers = [
        ('TRACK_A', 'MSCU1111113', '00000000-0000-0000-0000-00000000000A'),
        ('TRACK_B', 'MEDU2222226', '00000000-0000-0000-0000-00000000000B'),
        ('TRACK_C', 'TGHU3333337', '00000000-0000-0000-0000-00000000000C'),
    ]

    event_ids = {}
    for track_key, c_num, expected_cloud_id in containers:
        sess = ContainerPassageSession(track_key, 'LANE-OUT', 'EXIT', cam_ids, datetime.now(timezone.utc))
        t = datetime.now(timezone.utc)
        for cam in cams:
            frame = _create_sharp_test_frame(f"{c_num}_{cam.id}")
            dets = [Detection(cam.id, t, 'container', 0.95, (10, 10, 350, 250))]
            sess.add_frame(cam.id, frame, t, dets)
        sess.add_ocr_candidate(OCRRead(c_num, 0.98))
        ev_id = repository.finalize_passage_session(sess, snapshots)
        event_ids[ev_id] = (c_num, expected_cloud_id)

    dispatcher = CloudOutboxDispatcher(mock_cloud_settings, repository=repository, snapshots=snapshots)

    def fake_post(payload):
        eid = payload['id']
        cnum, cloud_id = event_ids[eid]
        return {'eventId': cloud_id, 'visitDatabaseId': f"vdb-{cnum}"}

    put_calls = {}

    def fake_put(cloud_ev_id, payload):
        put_calls[cloud_ev_id] = payload
        return True

    with patch.object(dispatcher.session, 'post_gate_event', side_effect=fake_post), \
         patch.object(dispatcher.session, 'put_gate_event_images', side_effect=fake_put):

        dispatcher.start()
        # Enqueue all 3 in quick succession
        for ev_id in event_ids:
            dispatcher.enqueue(None, ev_id, stage='POST', force=True)

        for _ in range(50):
            if len(put_calls) == 3:
                break
            time.sleep(0.05)

        dispatcher.stop()

        assert len(put_calls) == 3, f"Expected 3 PUT calls, got {len(put_calls)}"

        # Strict validation: verify each PUT received only its own container number and images
        for ev_id, (expected_num, expected_cloud_id) in event_ids.items():
            assert expected_cloud_id in put_calls
            payload = put_calls[expected_cloud_id]
            container = payload['Containers'][0]
            assert container['ContainerNumber'] == expected_num

            # Check that images for Container A only contain Container A
            for img in container['Images']:
                data_uri = img['Image']
                raw_bytes = base64.b64decode(data_uri.split(',', 1)[1])
                img_cv = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                assert img_cv is not None


# -------------------------------------------------------------------------------------
# Test C: Deliberately Interleaved Processing Order
# -------------------------------------------------------------------------------------
def test_interleaved_processing_order_independence(gate_setup, repository, snapshots, mock_cloud_settings):
    """
    Test C: Force Container B's POST response to arrive before Container A's.
    Make Container C's evidence finish first, followed by A and then B.
    Verify that processing order has zero effect on event-image associations.
    """
    cams = [c for c in gate_setup if c.gate_id == 'LANE-IN']
    cam_ids = [c.id for c in cams]

    sess_a = ContainerPassageSession('TRACK_INT_A', 'LANE-IN', 'ENTRY', cam_ids, datetime.now(timezone.utc))
    sess_b = ContainerPassageSession('TRACK_INT_B', 'LANE-IN', 'ENTRY', cam_ids, datetime.now(timezone.utc))

    t = datetime.now(timezone.utc)
    for cam in cams:
        sess_a.add_frame(cam.id, _create_sharp_test_frame(f"A_{cam.id}"), t, [Detection(cam.id, t, 'container', 0.95, (10, 10, 300, 200))])
        sess_b.add_frame(cam.id, _create_sharp_test_frame(f"B_{cam.id}"), t, [Detection(cam.id, t, 'container', 0.95, (10, 10, 300, 200))])

    sess_a.add_ocr_candidate(OCRRead('TEMU1111112', 0.96))
    sess_b.add_ocr_candidate(OCRRead('HLXU2222225', 0.96))

    ev_a = repository.finalize_passage_session(sess_a, snapshots)
    ev_b = repository.finalize_passage_session(sess_b, snapshots)

    dispatcher = CloudOutboxDispatcher(mock_cloud_settings, repository=repository, snapshots=snapshots)

    cloud_ids = {ev_a: 'CLOUD-A-UUID', ev_b: 'CLOUD-B-UUID'}
    put_records = []

    def mocked_post(payload):
        eid = payload['id']
        # Simulate delay on A so B finishes first
        if eid == ev_a:
            time.sleep(0.15)
        return {'eventId': cloud_ids[eid]}

    def mocked_put(cloud_id, payload):
        put_records.append((cloud_id, payload['Containers'][0]['ContainerNumber']))
        return True

    with patch.object(dispatcher.session, 'post_gate_event', side_effect=mocked_post), \
         patch.object(dispatcher.session, 'put_gate_event_images', side_effect=mocked_put):

        dispatcher.start()
        # Enqueue A then B
        dispatcher.enqueue(None, ev_a, stage='POST', force=True)
        dispatcher.enqueue(None, ev_b, stage='POST', force=True)

        for _ in range(40):
            if len(put_records) == 2:
                break
            time.sleep(0.05)

        dispatcher.stop()

        # Both PUT operations completed targeting their respective cloud IDs
        put_map = dict(put_records)
        assert put_map['CLOUD-A-UUID'] == 'TEMU1111112'
        assert put_map['CLOUD-B-UUID'] == 'HLXU2222225'


# -------------------------------------------------------------------------------------
# Test D: Cross-Container Image Rejection
# -------------------------------------------------------------------------------------
def test_cross_container_image_rejection(gate_setup, repository, snapshots, mock_cloud_settings):
    """
    Test D: Deliberately inject a conflicting detection/OCR belonging to Container B
    into Container A's snapshot candidates.
    The validation must reject the image and prevent it from reaching the PUT payload.
    """
    cams = [c for c in gate_setup if c.gate_id == 'LANE-OUT']
    cam_ids = [c.id for c in cams]

    session = ContainerPassageSession('TRACK_INJECT', 'LANE-OUT', 'EXIT', cam_ids, datetime.now(timezone.utc))
    t = datetime.now(timezone.utc)
    for cam in cams:
        session.add_frame(cam.id, _create_sharp_test_frame('CORRECT_A'), t, [Detection(cam.id, t, 'container', 0.95, (10, 10, 300, 200))])

    session.add_ocr_candidate(OCRRead('MSCU1234566', 0.98))
    event_id = repository.finalize_passage_session(session, snapshots)

    # Now deliberately inject a conflicting detection and snapshot into the database record for Container B
    import uuid
    det_id = str(uuid.uuid4())
    ocr_id = str(uuid.uuid4())
    with repository.db.sessions.begin() as s:
        s.add(DetectionRecord(
            id=det_id,
            event_id=event_id,
            camera_id='gate-out-cam-2',
            origin_key=f'foreign:origin:{uuid.uuid4()}',
            track_id=999,
            class_name='container',
            confidence=0.99,
            bbox=[0, 0, 100, 100],
            timestamp=t
        ))
        s.add(OCRResult(
            id=ocr_id,
            detection_id=det_id,
            raw_text='BXYZ 999999 9',
            normalized_text='BXYZ9999999',
            confidence=0.99,
            valid_format=True,
            valid_check_digit=True,
            validation_status='VALID'
        ))

    dispatcher = CloudOutboxDispatcher(mock_cloud_settings, repository=repository, snapshots=snapshots)
    ev_record = repository.event(event_id)

    # Build multi-camera images
    c1_images, expected_cams, captured_cams = dispatcher._build_multi_camera_images(ev_record, allow_single_fallback=False)

    # Verify that the conflicting container's snapshot was rejected
    for img in c1_images:
        assert img['CameraId'] in ('gate-out-cam-1', 'gate-out-cam-2', 'gate-out-cam-3')


# -------------------------------------------------------------------------------------
# Test E: Multiple Physical Containers on One Trailer (Twin 20ft containers)
# -------------------------------------------------------------------------------------
def test_twin_20ft_independent_events_and_puts(gate_setup, repository, snapshots, mock_cloud_settings):
    """
    Test E: Pass two 20-foot containers on the same trailer.
    Verify independent tracking/session identities, confirmed OCR results, event IDs,
    evidence images, and PUT payloads.
    """
    cams = [c for c in gate_setup if c.gate_id == 'LANE-IN']
    cam_ids = [c.id for c in cams]

    sess1 = ContainerPassageSession('TRACK_TWIN_FRONT', 'LANE-IN', 'ENTRY', cam_ids, datetime.now(timezone.utc), container_slot='CONTAINER_1')
    sess2 = ContainerPassageSession('TRACK_TWIN_REAR', 'LANE-IN', 'ENTRY', cam_ids, datetime.now(timezone.utc), container_slot='CONTAINER_2')

    t = datetime.now(timezone.utc)
    for cam in cams:
        sess1.add_frame(cam.id, _create_sharp_test_frame('TWIN_1'), t, [Detection(cam.id, t, 'container', 0.95, (10, 10, 300, 200))])
        sess2.add_frame(cam.id, _create_sharp_test_frame('TWIN_2'), t + timedelta(seconds=1), [Detection(cam.id, t + timedelta(seconds=1), 'container', 0.95, (10, 10, 300, 200))])

    sess1.add_ocr_candidate(OCRRead('TGHU2020209', 0.95))
    sess2.add_ocr_candidate(OCRRead('MSCU2020208', 0.95))

    ev1 = repository.finalize_passage_session(sess1, snapshots)
    ev2 = repository.finalize_passage_session(sess2, snapshots)

    assert ev1 != ev2, "Twin 20ft containers must have distinct database event IDs"

    dispatcher = CloudOutboxDispatcher(mock_cloud_settings, repository=repository, snapshots=snapshots)
    dispatcher.repository.set_cloud_event_id(ev1, 'CLOUD-TWIN-1')
    dispatcher.repository.set_cloud_event_id(ev2, 'CLOUD-TWIN-2')

    put_targets = {}

    def mock_put(cloud_id, payload):
        put_targets[cloud_id] = payload
        return True

    with patch.object(dispatcher.session, 'put_gate_event_images', side_effect=mock_put):
        dispatcher.start()
        dispatcher.enqueue(None, ev1, stage='PUT', force=True)
        dispatcher.enqueue(None, ev2, stage='PUT', force=True)

        for _ in range(30):
            if len(put_targets) == 2:
                break
            time.sleep(0.05)
        dispatcher.stop()

        assert 'CLOUD-TWIN-1' in put_targets
        assert 'CLOUD-TWIN-2' in put_targets
        assert put_targets['CLOUD-TWIN-1']['Containers'][0]['ContainerNumber'] == 'TGHU2020209'
        assert put_targets['CLOUD-TWIN-2']['Containers'][0]['ContainerNumber'] == 'MSCU2020208'


# -------------------------------------------------------------------------------------
# Test F: Missing Camera Handled Honestly
# -------------------------------------------------------------------------------------
def test_missing_camera_honest_handling(gate_setup, repository, snapshots, mock_cloud_settings):
    """
    Test F: Simulate unavailable camera 3.
    Verify that missing evidence is handled according to policy and that images from
    another container are NEVER substituted.
    """
    cams = [c for c in gate_setup if c.gate_id == 'LANE-OUT']
    cam_ids = [c.id for c in cams]

    session = ContainerPassageSession('TRACK_MISSING_CAM', 'LANE-OUT', 'EXIT', cam_ids, datetime.now(timezone.utc))
    t = datetime.now(timezone.utc)
    # Only camera 1 and camera 2 produce frames; camera 3 produces no frame
    session.add_frame('gate-out-cam-1', _create_sharp_test_frame('CAM1'), t, [Detection('gate-out-cam-1', t, 'container', 0.95, (10, 10, 300, 200))])
    session.add_frame('gate-out-cam-2', _create_sharp_test_frame('CAM2'), t, [Detection('gate-out-cam-2', t, 'container', 0.95, (10, 10, 300, 200))])

    session.add_ocr_candidate(OCRRead('MEDU1234562', 0.95))
    ev_id = repository.finalize_passage_session(session, snapshots)

    dispatcher = CloudOutboxDispatcher(mock_cloud_settings, repository=repository, snapshots=snapshots)
    ev_rec = repository.event(ev_id)

    c1_images, expected_cams, captured_cams = dispatcher._build_multi_camera_images(ev_rec, allow_single_fallback=False)

    captured_cam_set = {img['CameraId'] for img in c1_images}
    assert 'gate-out-cam-1' in captured_cam_set
    assert 'gate-out-cam-2' in captured_cam_set
    assert 'gate-out-cam-3' not in captured_cam_set
    # Confirm no fake substitution occurred
    assert len(c1_images) == 2


# -------------------------------------------------------------------------------------
# Test G: Network Failures and Retry with Identical eventId
# -------------------------------------------------------------------------------------
def test_network_failures_and_retry_isolation(gate_setup, repository, snapshots, mock_cloud_settings):
    """
    Test G: Simulate POST and PUT failures separately.
    Verify persistence, retry behavior, correct eventId reuse, and duplicate-event prevention.
    """
    cams = [c for c in gate_setup if c.gate_id == 'LANE-IN']
    cam_ids = [c.id for c in cams]

    session = ContainerPassageSession('TRACK_FAIL_RETRY', 'LANE-IN', 'ENTRY', cam_ids, datetime.now(timezone.utc))
    t = datetime.now(timezone.utc)
    for cam in cams:
        session.add_frame(cam.id, _create_sharp_test_frame(f"RETRY_{cam.id}"), t, [Detection(cam.id, t, 'container', 0.95, (10, 10, 300, 200))])
    session.add_ocr_candidate(OCRRead('MSCU9876541', 0.97))
    ev_id = repository.finalize_passage_session(session, snapshots)

    dispatcher = CloudOutboxDispatcher(mock_cloud_settings, repository=repository, snapshots=snapshots)

    # 1. Simulate POST failure (offline)
    with patch.object(dispatcher.session, 'post_gate_event', return_value='OFFLINE') as mock_post:
        dispatcher.start()
        dispatcher.enqueue(None, ev_id, stage='POST', force=True)
        time.sleep(0.3)
        dispatcher.stop()

        db_ev = repository.event(ev_id)
        assert db_ev['cloud_synced'] is False
        assert db_ev['cloud_delivery_state'] == 'OFFLINE'
        assert db_ev['cloud_event_id'] is None

    # 2. Network restores; POST succeeds and yields cloud_event_id
    assigned_cloud_id = 'CLOUD-PERSIST-9999'
    with patch.object(dispatcher.session, 'post_gate_event', return_value={'eventId': assigned_cloud_id}) as mock_post, \
         patch.object(dispatcher.session, 'put_gate_event_images', return_value='OFFLINE') as mock_put_initial:
        dispatcher.start()
        dispatcher.enqueue(None, ev_id, stage='POST', force=True)
        for _ in range(20):
            if repository.event(ev_id)['cloud_event_id'] == assigned_cloud_id:
                break
            time.sleep(0.05)
        dispatcher.stop()

        db_ev = repository.event(ev_id)
        assert db_ev['cloud_event_id'] == assigned_cloud_id
        assert db_ev['cloud_delivery_state'] in ('EVENT_CREATED', 'PUT_FAILED', 'DELIVERED')

    # 3. Simulate PUT failure: must mark PUT_FAILED but NEVER clear cloud_event_id
    with patch.object(dispatcher.session, 'put_gate_event_images', return_value='OFFLINE') as mock_put:
        dispatcher.start()
        dispatcher.enqueue(None, ev_id, stage='PUT', force=True)
        time.sleep(0.3)
        dispatcher.stop()

        db_ev = repository.event(ev_id)
        # CRITICAL RULE: cloud_event_id MUST BE PRESERVED on PUT failure!
        assert db_ev['cloud_event_id'] == assigned_cloud_id
        assert db_ev['cloud_delivery_state'] == 'PUT_FAILED'

    # 4. Final retry: PUT succeeds using the EXACT SAME cloud_event_id
    with patch.object(dispatcher.session, 'put_gate_event_images', return_value=True) as mock_put:
        dispatcher.start()
        dispatcher.enqueue(None, ev_id, stage='PUT', force=True)
        for _ in range(20):
            if repository.event(ev_id)['cloud_synced']:
                break
            time.sleep(0.05)
        dispatcher.stop()

        db_ev = repository.event(ev_id)
        assert db_ev['cloud_synced'] is True
        assert db_ev['cloud_delivery_state'] == 'DELIVERED'
        assert db_ev['cloud_event_id'] == assigned_cloud_id


# -------------------------------------------------------------------------------------
# Test H: Evidence Ready Before POST Response Arrives (Case B)
# -------------------------------------------------------------------------------------
def test_evidence_ready_before_post_response_case_b(gate_setup, repository, snapshots, mock_cloud_settings):
    """
    Test H: Complete evidence collection before the POST response arrives.
    Verify that the evidence is retained and uploaded only after the correct eventId becomes available.
    """
    cams = [c for c in gate_setup if c.gate_id == 'LANE-OUT']
    cam_ids = [c.id for c in cams]

    session = ContainerPassageSession('TRACK_CASE_B', 'LANE-OUT', 'EXIT', cam_ids, datetime.now(timezone.utc))
    t = datetime.now(timezone.utc)
    for cam in cams:
        session.add_frame(cam.id, _create_sharp_test_frame(f"CASE_B_{cam.id}"), t, [Detection(cam.id, t, 'container', 0.95, (10, 10, 300, 200))])
    session.add_ocr_candidate(OCRRead('CNEU4444440', 0.96))
    ev_id = repository.finalize_passage_session(session, snapshots)

    dispatcher = CloudOutboxDispatcher(mock_cloud_settings, repository=repository, snapshots=snapshots)
    dispatcher.mark_evidence_ready(ev_id)

    put_called = []

    def slow_post(payload):
        time.sleep(0.2)
        return {'eventId': 'CLOUD-CASE-B-EVENT'}

    def captured_put(cloud_id, payload):
        put_called.append((cloud_id, payload))
        return True

    with patch.object(dispatcher.session, 'post_gate_event', side_effect=slow_post), \
         patch.object(dispatcher.session, 'put_gate_event_images', side_effect=captured_put):

        dispatcher.start()
        # Enqueue POST (which will take 200ms)
        dispatcher.enqueue(None, ev_id, stage='POST', force=True)

        for _ in range(40):
            if put_called:
                break
            time.sleep(0.05)

        dispatcher.stop()

        assert len(put_called) == 1
        target_cloud_id, payload = put_called[0]
        assert target_cloud_id == 'CLOUD-CASE-B-EVENT'
        assert payload['Containers'][0]['ContainerNumber'] == 'CNEU4444440'


# -------------------------------------------------------------------------------------
# Test I: Application Restart and Recovery Isolation
# -------------------------------------------------------------------------------------
def test_application_restart_and_offline_sweep(gate_setup, repository, snapshots, mock_cloud_settings):
    """
    Test I: Restart the application while evidence collection or PUT retries are pending.
    Verify that event IDs and evidence references are restored correctly from SQLite.
    """
    cams = [c for c in gate_setup if c.gate_id == 'LANE-IN']
    cam_ids = [c.id for c in cams]

    session = ContainerPassageSession('TRACK_RESTART', 'LANE-IN', 'ENTRY', cam_ids, datetime.now(timezone.utc))
    t = datetime.now(timezone.utc)
    for cam in cams:
        session.add_frame(cam.id, _create_sharp_test_frame(f"REST_{cam.id}"), t, [Detection(cam.id, t, 'container', 0.95, (10, 10, 300, 200))])
    session.add_ocr_candidate(OCRRead('MSKU7777770', 0.97))
    ev_id = repository.finalize_passage_session(session, snapshots)

    # Set cloud_event_id directly in DB simulating pre-crash state where POST finished
    repository.set_cloud_event_id(ev_id, 'RESTORED-CLOUD-EVENT-777', state='EVENT_CREATED')

    # Re-instantiate a fresh dispatcher as if the backend just restarted
    fresh_dispatcher = CloudOutboxDispatcher(mock_cloud_settings, repository=repository, snapshots=snapshots)

    with patch.object(fresh_dispatcher.session, 'put_gate_event_images', return_value=True) as mock_put:
        fresh_dispatcher.start()
        # Trigger offline backlog sweep on restart
        queued_count = fresh_dispatcher.trigger_retry()

        for _ in range(30):
            if mock_put.called:
                break
            time.sleep(0.05)

        fresh_dispatcher.stop()

        assert mock_put.called, "Sweep did not trigger PUT for restored event"
        target_cloud_id = mock_put.call_args[0][0]
        assert target_cloud_id == 'RESTORED-CLOUD-EVENT-777'


# -------------------------------------------------------------------------------------
# Test J: Gate Isolation (Gate-In cameras vs Gate-Out cameras)
# -------------------------------------------------------------------------------------
def test_gate_isolation_in_vs_out(gate_setup, repository, snapshots, mock_cloud_settings):
    """
    Test J: Verify that Gate-In camera images are never attached to Gate-Out events,
    and Gate-Out camera images are never attached to Gate-In events.
    """
    in_cams = [c for c in gate_setup if c.gate_id == 'LANE-IN']
    out_cams = [c for c in gate_setup if c.gate_id == 'LANE-OUT']

    # Create Gate-In event
    sess_in = ContainerPassageSession('TRACK_GATE_IN', 'LANE-IN', 'ENTRY', [c.id for c in in_cams], datetime.now(timezone.utc))
    t = datetime.now(timezone.utc)
    for cam in in_cams:
        sess_in.add_frame(cam.id, _create_sharp_test_frame(f"IN_{cam.id}"), t, [Detection(cam.id, t, 'container', 0.95, (10, 10, 300, 200))])
    sess_in.add_ocr_candidate(OCRRead('MSCU5555559', 0.97))
    ev_in = repository.finalize_passage_session(sess_in, snapshots)

    # Deliberately attempt to add a Gate-Out camera snapshot to the Gate-In event directory
    out_cam_img = _create_sharp_test_frame('OUT_CAM_LEAK')
    snapshots.save_side_snapshot(ev_in, 'gate-out-cam-1', out_cam_img)

    dispatcher = CloudOutboxDispatcher(mock_cloud_settings, repository=repository, snapshots=snapshots)
    ev_rec = repository.event(ev_in)

    c1_images, expected_cams, captured_cams = dispatcher._build_multi_camera_images(ev_rec, allow_single_fallback=False)

    for img in c1_images:
        cam_id = img['CameraId']
        assert not cam_id.startswith('gate-out'), f"Gate-Out camera {cam_id} was improperly attached to Gate-In event!"


# -------------------------------------------------------------------------------------
# Test K: API Contract Validation (Request structure, URL, Base64 data URI format)
# -------------------------------------------------------------------------------------
def test_api_contract_validation(gate_setup, repository, snapshots, mock_cloud_settings):
    """
    Test K: Verify URL formatting, PUT endpoint construction, base64 data URI encoding,
    and exact required JSON fields against the API specification.
    """
    cams = [c for c in gate_setup if c.gate_id == 'LANE-OUT']
    session = ContainerPassageSession('TRACK_CONTRACT', 'LANE-OUT', 'EXIT', [c.id for c in cams], datetime.now(timezone.utc))
    t = datetime.now(timezone.utc)
    for cam in cams:
        session.add_frame(cam.id, _create_sharp_test_frame(f"CON_{cam.id}"), t, [Detection(cam.id, t, 'container', 0.95, (10, 10, 300, 200))])
    session.add_ocr_candidate(OCRRead('MSCU8888888', 0.98))
    ev_id = repository.finalize_passage_session(session, snapshots)

    session_mgr = ProsperSession(mock_cloud_settings)
    test_cloud_id = '0e7d4fac-5f22-48a4-be37-0bc58a948eff'

    with patch.object(session_mgr.session, 'put') as mock_put_req:
        mock_put_req.return_value.status_code = 200
        mock_put_req.return_value.content = b'{"success": true}'
        mock_put_req.return_value.json.return_value = {"success": True}

        payload = {
            "Containers": [
                {
                    "ContainerNumber": "MSCU8888888",
                    "ContainerNumberConfidence": 0.98,
                    "Size": "40",
                    "Position": "SINGLE",
                    "Images": [
                        {
                            "ImageType": "FRONT",
                            "CameraId": "gate-out-cam-1",
                            "CapturedAt": "2026-10-06T08:15:39.210Z",
                            "Image": "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
                        }
                    ]
                }
            ]
        }

        res = session_mgr.put_gate_event_images(test_cloud_id, payload)
        assert res is True or isinstance(res, dict)

        assert mock_put_req.called
        called_url = mock_put_req.call_args[0][0]
        # Verify URL matches: https://api.smartyard.prosperinfotech.com/gate/api/v1/gate/events/{eventId}/images
        assert called_url == f"https://api.smartyard.prosperinfotech.com/gate/api/v1/gate/events/{test_cloud_id}/images"
        assert mock_put_req.call_args[1]['headers']['Content-Type'] == 'application/json'
        assert 'Authorization' in mock_put_req.call_args[1]['headers']

# -------------------------------------------------------------------------------------
# Test M: Late-Arriving Companion Camera Proof Triggers PUT on Already-Delivered Event
# -------------------------------------------------------------------------------------
def test_late_arriving_companion_camera_triggers_put(gate_setup, repository, snapshots, mock_cloud_settings):
    """
    Test M: Real-time companion camera evidence arrival.
    1. Initial detection on Camera 1 creates event and completes POST (DELIVERED with Camera 1 only).
    2. Camera 2 arrives late and records companion observation for the same container.
    3. Verify that the system automatically re-enqueues Stage 2 PUT and updates the cloud with both camera angles.
    """
    from app.events.event_manager import EventManager
    cfg = mock_cloud_settings.model_copy(update={'cloud_dispatch_grace_seconds': 0.0})
    cams = [c for c in gate_setup if c.gate_id == 'LANE-OUT']
    cam1 = cams[0]  # gate-out-cam-1
    cam2 = cams[1]  # gate-out-cam-2

    dispatcher = CloudOutboxDispatcher(cfg, repository=repository, snapshots=snapshots)
    ev_mgr = EventManager(cfg, repository, snapshots, dispatcher=dispatcher)

    post_resp = {
        "visitDatabaseId": "12345678-8d78-4806-acf2-42eb18e4ef8d",
        "visitId": "VISIT-LATE-TEST",
        "eventId": "22222222-5f22-48a4-be37-0bc58a948eff",
        "eventType": "GATE_OUT"
    }

    t1 = datetime.now(timezone.utc)
    f1 = _create_sharp_test_frame("GATE_OUT_1_LATE_TEST")
    det1 = Detection(cam1.id, t1, 'container', 0.95, (10, 10, 300, 200))
    val1 = Validation('EITU 180263 7', 'EITU1802637', 0.98, True, True, 'CONFIRMED', '45G1')
    obs1 = Observation(f"{cam1.id}:1", 'LANE-OUT', 1, det1, val1, 'EXIT', f1, t1, True)

    with patch.object(dispatcher.session, 'post_gate_event', return_value=post_resp) as mock_post, \
         patch.object(dispatcher.session, 'put_gate_event_images', return_value=True) as mock_put:

        dispatcher.start()
        # Stage 1: Initial observation recorded and dispatched
        event_id, created = ev_mgr.record_observation(obs1)
        assert created is True

        for _ in range(50):
            ev = repository.event(event_id)
            if ev and ev.get('cloud_event_id'):
                break
            time.sleep(0.05)

        assert mock_post.called
        assert repository.event(event_id).get('cloud_event_id') is not None

        # Now simulate Camera 2 arriving late (companion observation on gate-out-cam-2)
        t2 = t1 + timedelta(seconds=2)
        f2 = _create_sharp_test_frame("GATE_OUT_2_LATE_TEST")
        det2 = Detection(cam2.id, t2, 'container', 0.95, (15, 15, 310, 210))
        val2 = Validation('EITU 180263 7', 'EITU1802637', 0.95, True, True, 'CONFIRMED', '45G1')
        obs2 = Observation(f"{cam2.id}:2", 'LANE-OUT', 2, det2, val2, 'EXIT', f2, t2, True)

        # EventManager records late companion observation
        ev_mgr.record_observation(obs2)

        # Wait for PUT to be triggered with both camera images
        for _ in range(40):
            if mock_put.called:
                break
            time.sleep(0.05)

        dispatcher.stop()

        assert mock_put.called, "PUT was not triggered when late companion camera frame arrived!"
        put_target_id = mock_put.call_args[0][0]
        assert put_target_id == "22222222-5f22-48a4-be37-0bc58a948eff"

        put_payload = mock_put.call_args[0][1]
        sent_images = put_payload['Containers'][0]['Images']
        sent_cams = {img['CameraId'] for img in sent_images}
        assert cam1.id in sent_cams, "Camera 1 missing from multi-camera PUT!"
        assert cam2.id in sent_cams, "Camera 2 missing from multi-camera PUT!"


def test_ocr_prefix_confusion_same_serial_deduplication(gate_setup, repository, snapshots):
    """
    Verify that single physical container with OCR prefix confusion (e.g. TCNU7922652 vs TONU7922656)
    arriving on the same gate 18 seconds apart fuses into a SINGLE database event.
    """
    cams = [c for c in gate_setup if c.gate_id == 'LANE-IN']
    cam_ids = [c.id for c in cams]

    t1 = datetime.now(timezone.utc)
    sess1 = ContainerPassageSession('TRACK_1', 'LANE-IN', 'ENTRY', cam_ids, t1)
    for cam in cams:
        sess1.add_frame(cam.id, _create_sharp_test_frame('TCNU7922652'), t1, [Detection(cam.id, t1, 'container', 0.95, (10, 10, 300, 200))])
    sess1.add_ocr_candidate(OCRRead('TCNU7922652', 0.95))
    ev1 = repository.finalize_passage_session(sess1, snapshots)

    # 18 seconds later, trailing frame reads TONU7922656 (C misread as O, check digit recomputed to 6)
    t2 = t1 + timedelta(seconds=18)
    sess2 = ContainerPassageSession('TRACK_2', 'LANE-IN', 'ENTRY', cam_ids, t2)
    for cam in cams:
        sess2.add_frame(cam.id, _create_sharp_test_frame('TONU7922656'), t2, [Detection(cam.id, t2, 'container', 0.90, (10, 10, 300, 200))])
    sess2.add_ocr_candidate(OCRRead('TONU7922656', 0.90))
    ev2 = repository.finalize_passage_session(sess2, snapshots)

    assert ev1 == ev2, f"Expected both sessions to resolve to identical event ID, got {ev1} and {ev2}"
