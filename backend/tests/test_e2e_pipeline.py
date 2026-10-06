"""
End-to-End Pipeline Verification and Test Suite.
Validates:
1. Fast blur filtering discarding unreadable/blurred crops.
2. Strict ISO 6346 check digit validation and rejection of invalid check digits.
3. Strict rejection of UUIDs, tracker IDs, and non-container strings (e.g. 3c780c27-e351-408d-b642-bdcf8e046d90).
4. Prevention of single-frame reads (count == 1) from creating confirmed events.
5. Multi-camera frame collection (4 cameras per gate) and 4 best images per container.
6. Multi-frame character consensus and early confirmation.
7. End-to-end database event persistence and asynchronous non-blocking API dispatch.
8. Latency profiling across all pipeline stages.
"""

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.config.settings import Settings, CameraConfig
from app.database.database import Database
from app.database.repositories.gate import GateRepository
from app.domain import Frame, Detection, OCRRead, Validation, Observation
from app.events.event_manager import EventManager
from app.events.pipeline_profiler import PipelineProfiler
from app.ocr.fast_selector import FastFrameSelector
from app.ocr.validator import ContainerValidator, is_valid_iso_6346, is_uuid, check_digit
from app.ocr.consensus import CharacterConsensus
from app.snapshots.snapshot_manager import SnapshotManager
from app.events.cloud_dispatcher import CloudOutboxDispatcher


@pytest.fixture
def e2e_env(tmp_path):
    db_path = tmp_path / "e2e_gate.db"
    db = Database(f"sqlite:///{db_path}")
    db.initialize()
    settings = Settings(ocr_confirmations=2, cloud_dispatch_grace_seconds=0.0)
    repo = GateRepository(db)
    
    # Configure 4 cameras for Gate-In (lane-in)
    cameras = [
        CameraConfig(id="gate-in-1", name="IN Front", gate_id="lane-in", source_env="CAM1", role="FRONT_TOP"),
        CameraConfig(id="gate-in-2", name="IN Left", gate_id="lane-in", source_env="CAM2", role="LEFT"),
        CameraConfig(id="gate-in-3", name="IN Right", gate_id="lane-in", source_env="CAM3", role="RIGHT"),
        CameraConfig(id="gate-in-4", name="IN Rear", gate_id="lane-in", source_env="CAM4", role="REAR"),
    ]
    repo.sync_cameras(cameras)
    snapshots = SnapshotManager(tmp_path / "snapshots")
    profiler = PipelineProfiler()
    return {
        "db": db,
        "repo": repo,
        "settings": settings,
        "snapshots": snapshots,
        "profiler": profiler,
        "tmp_path": tmp_path,
    }


def make_test_frame(text: str, blur: bool = False, w: int = 1920, h: int = 1080) -> np.ndarray:
    """Create a simulated camera frame with high contrast container markings."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 110
    # Add container body block
    cv2.rectangle(img, (200, 200), (1700, 900), (35, 75, 140), -1)
    # Add text marking
    cv2.putText(img, text, (400, 500), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 6)
    cv2.putText(img, "45G1", (400, 650), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 5)
    if blur:
        img = cv2.GaussianBlur(img, (51, 51), 0)
    return img


def test_e2e_blur_filter_rejection():
    """Verify that motion-blurred candidate crops are rejected in < 15ms without invoking OCR."""
    sharp_img = make_test_frame("CSQU3054383", blur=False)
    blurred_img = make_test_frame("CSQU3054383", blur=True)
    crop_box = (350, 420, 1100, 550)

    # Warm-up calls to avoid cold memory / JIT initialization latency
    FastFrameSelector.evaluate_candidate(sharp_img, crop_box)
    FastFrameSelector.evaluate_candidate(blurred_img, crop_box)

    t0 = time.perf_counter()
    sharp_usable, sharp_score, _ = FastFrameSelector.evaluate_candidate(sharp_img, crop_box)
    t_sharp = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    blurred_usable, blurred_score, _ = FastFrameSelector.evaluate_candidate(blurred_img, crop_box)
    t_blur = (time.perf_counter() - t0) * 1000.0

    assert sharp_usable is True, "Sharp container frame must pass filter"
    assert sharp_score > 30.0
    assert blurred_usable is False, "Motion-blurred frame must be rejected"
    assert t_blur < 15.0, f"Blur check must take < 15ms (was {t_blur:.2f}ms)"


def test_e2e_strict_iso_and_uuid_rejection():
    """Verify strict ISO 6346 check digit math and rejection of UUIDs/noise."""
    validator = ContainerValidator()

    # 1. UUID string reported by user
    uuid_str = "3c780c27-e351-408d-b642-bdcf8e046d90"
    assert is_uuid(uuid_str)
    assert not is_valid_iso_6346(uuid_str)
    val = validator.validate(OCRRead(uuid_str, 0.99))
    assert val.normalized_text == ""
    assert not val.valid_format
    assert not val.valid_check_digit
    assert val.validation_status == "INVALID_FORMAT"

    # 2. Corrupted check-digit (e.g. MSCU1234567 where mathematical check digit is 6)
    assert check_digit("MSCU123456") == 6
    assert not is_valid_iso_6346("MSCU1234567")
    corrupt_val = validator.validate(OCRRead("MSCU1234567", 0.98))
    assert corrupt_val.valid_format is True
    assert corrupt_val.valid_check_digit is False
    assert corrupt_val.validation_status == "INVALID_CHECK_DIGIT"

    # 3. Legitimate container numbers
    assert is_valid_iso_6346("CSQU3054383")
    assert is_valid_iso_6346("TRHU1545895")
    assert is_valid_iso_6346("MSCU6639870")
    valid_val = validator.validate(OCRRead("CSQU 305438-3", 0.96))
    assert valid_val.normalized_text == "CSQU3054383"
    assert valid_val.valid_format is True
    assert valid_val.valid_check_digit is True
    assert valid_val.validation_status == "VALID"


def test_e2e_single_frame_read_never_confirmed(e2e_env):
    """
    Verify that an isolated, single-frame OCR read (count == 1) does NOT
    produce a CONFIRMED event or dispatch to API. It must stay NEEDS_REVIEW.
    """
    repo = e2e_env["repo"]
    settings = e2e_env["settings"]
    snapshots = e2e_env["snapshots"]
    dispatcher = MagicMock()
    manager = EventManager(settings, repo, snapshots, dispatcher=dispatcher)

    # Candidate reported by user: e.g. IHEU9036531 on only 1 frame
    val_candidate = ContainerValidator().validate(OCRRead("IHEU9036531", 0.95))
    assert val_candidate.valid_check_digit is True

    frame_img = make_test_frame("IHEU9036531")
    frame = Frame("gate-in-1", datetime.now(timezone.utc), 1, frame_img)
    det = Detection("gate-in-1", frame.timestamp, "container", 0.95, (200, 200, 1600, 800))
    job = MagicMock()
    job.origin_key = "dwell:gate-in-1:single_test"
    job.gate_id = "lane-in"
    job.track_id = 101
    job.detection = det
    job.frame = frame
    job.direction = "ENTRY"
    job.started_at = frame.timestamp
    job.event_id = None
    job.all_detections = []

    # Process exactly 1 frame
    result = manager.observe(job, val_candidate)
    assert result is None, "A single frame must NOT confirm the container"

    # Now simulate dwell passage closing after max attempts without second confirmation
    for seq in range(2, settings.ocr_max_attempts + 1):
        unreadable_frame = Frame("gate-in-1", datetime.now(timezone.utc), seq, frame_img)
        job.frame = unreadable_frame
        unreadable_val = Validation("BLURRY", "", 0.1, False, False, "INVALID_FORMAT")
        event_res = manager.observe(job, unreadable_val)

    # Event must be saved as NEEDS_REVIEW, with NO container number confirmed and NO API dispatch
    assert event_res is not None
    event_id = event_res[0]
    ev = repo.event(event_id)
    assert ev["status"] == "NEEDS_REVIEW"
    assert ev["container_number"] is None
    assert not dispatcher.enqueue.called, "Unconfirmed event must NOT be dispatched to API"


def test_e2e_four_camera_consensus_and_best_frame_selection(e2e_env):
    """
    End-to-End multi-camera workflow:
    1. Container passes 4 cameras at Gate-In: C1 (Front), C2 (Left), C3 (Right), C4 (Rear).
    2. Frames captured across all 4 cameras.
    3. Multi-frame & multi-camera agreement triggers early confirmation.
    4. Best frame from each of the 4 cameras is selected (total 4 proof images).
    5. Event is saved to SQLite as CONFIRMED.
    6. Dispatched asynchronously to Cloud API with 4 images.
    """
    repo = e2e_env["repo"]
    settings = e2e_env["settings"]
    snapshots = e2e_env["snapshots"]
    profiler = e2e_env["profiler"]
    
    mock_post_session = MagicMock()
    mock_post_session.post_gate_event.return_value = {"visitDatabaseId": "vdb-e2e-123"}
    dispatcher = CloudOutboxDispatcher(settings)
    dispatcher.session = mock_post_session
    dispatcher.repository = repo
    dispatcher.snapshots = snapshots
    dispatcher.start()

    manager = EventManager(settings, repo, snapshots, dispatcher=dispatcher)

    # Container to recognize
    expected_container = "CSQU3054383"
    cams = [("gate-in-1", "FRONT_TOP"), ("gate-in-2", "LEFT"), ("gate-in-3", "RIGHT"), ("gate-in-4", "REAR")]
    origin_key = "transit:gate-in:container_999"
    now = datetime.now(timezone.utc)
    
    profiler_rec = profiler.new_record(origin_key)

    # All 4 cameras covering Gate-In capture frames
    companion_frames = {
        cam_id: make_test_frame(expected_container, blur=False)
        for cam_id, _ in cams if cam_id != "gate-in-1"
    }

    event_id = None
    for seq in (1, 2):
        img = make_test_frame(expected_container, blur=False)
        frame = Frame("gate-in-1", now, seq, img)
        det = Detection("gate-in-1", now, "container_number", 0.95, (200, 200, 1600, 800))
        val = ContainerValidator().validate(OCRRead(f"{expected_container} 45G1", 0.95))
        
        job = MagicMock()
        job.event_id = None
        job.all_detections = []
        job.origin_key = origin_key
        job.gate_id = "lane-in"
        job.track_id = 999
        job.detection = det
        job.frame = frame
        job.direction = "ENTRY"
        job.started_at = now
        
        res = manager.observe(job, val, additional_frames=companion_frames, profiler_record=profiler_rec)
        if res is not None:
            event_id = res[0]

    # Verify early confirmation was achieved
    assert event_id is not None
    event_data = repo.event(event_id)
    assert event_data["status"] == "CONFIRMED"
    assert event_data["container_number"] == expected_container

    # Verify best-frame manifest selected 1 frame per camera (all 4 cameras)
    manifest = snapshots.build_evidence_manifest(event_data)
    cam_ids_in_manifest = {c["camera_id"] for c in manifest["captures"]}
    assert len(cam_ids_in_manifest) == 4, f"Expected 4 camera captures, got {len(cam_ids_in_manifest)}"
    assert manifest["primary_snapshot_id"] is not None

    # Wait for non-blocking asynchronous cloud dispatcher
    for _ in range(30):
        if dispatcher.queue.empty() and mock_post_session.post_gate_event.called:
            break
        time.sleep(0.05)
    dispatcher.stop()

    assert mock_post_session.post_gate_event.called, "Confirmed event must be dispatched to API"
    sent_payload = mock_post_session.post_gate_event.call_args[0][0]
    
    # Verify API Payload Structure: exactly 4 camera images packaged
    assert sent_payload["Containers"][0]["ContainerNumber"] == expected_container
    assert sent_payload["EventType"] == "GATE_IN"
    assert len(sent_payload["Containers"][0]["Images"]) == 4, "Must send exactly 4 images (1 per camera)"

    # Print Latency Profile Breakdown
    profiler.record_and_log(profiler_rec, container_number=expected_container)
    print("\n--- MEASURED E2E PIPELINE LATENCY PROFILE ---")
    print(f"Tracking Latency:       {profiler_rec.tracking_ms:.3f} ms")
    print(f"Filtering Latency:      {profiler_rec.filtering_ms:.3f} ms")
    print(f"OCR Latency:            {profiler_rec.ocr_ms:.3f} ms")
    print(f"Consensus Latency:      {profiler_rec.consensus_ms:.3f} ms")
    print(f"Event Creation Latency: {profiler_rec.event_creation_ms:.3f} ms")
    print(f"API Queue Latency:      {profiler_rec.api_queue_ms:.3f} ms")
    print(f"Total Processing Time:  {profiler_rec.total_ms:.3f} ms")
