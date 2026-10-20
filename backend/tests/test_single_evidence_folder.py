from datetime import datetime, timezone, timedelta
from pathlib import Path
from uuid import uuid4
import numpy as np
import pytest

from app.domain import Detection, OCRRead
from app.events.passage_session import ContainerPassageSession, PassageSessionManager
from app.snapshots.evidence_manager import EvidenceFolderManager
from app.snapshots.snapshot_manager import SnapshotManager


def make_dummy_frame(w=640, h=480, text=""):
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    # Add texture so Laplacian variance and sharpness are high
    img[100:400, 100:500] = 200
    img[150:200, 150:450] = 50
    return img


def test_evidence_manager_folder_creation_and_update(tmp_path):
    evidence_root = tmp_path / "evidence"
    legacy_root = tmp_path / "snapshots"
    manager = EvidenceFolderManager(root=evidence_root, legacy_root=legacy_root)

    event_id = str(uuid4())

    # 1. Unconfirmed session creates folder with event_id
    folder = manager.get_or_create_folder(event_id, gate_id="gate-in-1", direction="GATE_IN")
    assert folder.is_dir()
    assert folder.name == event_id
    assert folder.parent.name == "GATE_IN"

    # 2. Save captured frame
    frame = make_dummy_frame()
    rel1 = manager.save_captured_frame(event_id, "cam_01", frame, frame_sequence=1, gate_id="gate-in-1", direction="GATE_IN")
    assert (folder / "cam_01_frame_001.jpg").is_file()

    # 3. Save frame from camera 2
    rel2 = manager.save_captured_frame(event_id, "cam_02", frame, frame_sequence=1, gate_id="gate-in-1", direction="GATE_IN")
    assert (folder / "cam_02_frame_001.jpg").is_file()

    # 4. Save frame from camera 3
    rel3 = manager.save_captured_frame(event_id, "cam_03", frame, frame_sequence=1, gate_id="gate-in-1", direction="GATE_IN")
    assert (folder / "cam_03_frame_001.jpg").is_file()

    # Verify no subdirectories were created inside the folder
    subdirs = [p for p in folder.iterdir() if p.is_dir()]
    assert len(subdirs) == 0

    # 5. Confirm container number -> safely renames folder to <event_id>_<container_number>
    updated_folder = manager.update_container_number(event_id, "MSCU1234567")
    assert updated_folder.name == f"{event_id}_MSCU1234567"
    assert (updated_folder / "cam_01_frame_001.jpg").is_file()
    assert (updated_folder / "cam_02_frame_001.jpg").is_file()
    assert (updated_folder / "cam_03_frame_001.jpg").is_file()

    # 6. Save best frames
    best1, leg1 = manager.save_best_frame(event_id, "cam_01", frame, gate_id="gate-in-1", direction="GATE_IN", container_number="MSCU1234567")
    best2, leg2 = manager.save_best_frame(event_id, "cam_02", frame, gate_id="gate-in-1", direction="GATE_IN", container_number="MSCU1234567")
    best3, leg3 = manager.save_best_frame(event_id, "cam_03", frame, gate_id="gate-in-1", direction="GATE_IN", container_number="MSCU1234567")

    assert (updated_folder / "best_cam_01.jpg").is_file()
    assert (updated_folder / "best_cam_02.jpg").is_file()
    assert (updated_folder / "best_cam_03.jpg").is_file()

    # Verify path resolution
    resolved = manager.resolve_path(f"{event_id}/cam_01.jpg")
    assert resolved.is_file()
    assert resolved == updated_folder / "cam_01.jpg"

    # Exactly one folder exists in GATE_IN
    gate_in_folders = list((evidence_root / "GATE_IN").iterdir())
    assert len(gate_in_folders) == 1
    assert gate_in_folders[0] == updated_folder


def test_gate_out_single_evidence_folder(tmp_path):
    evidence_root = tmp_path / "evidence"
    legacy_root = tmp_path / "snapshots"
    manager = EvidenceFolderManager(root=evidence_root, legacy_root=legacy_root)

    event_id = str(uuid4())

    folder = manager.get_or_create_folder(event_id, gate_id="gate-out-1", direction="GATE_OUT", container_number="TGHU9876543")
    assert folder.parent.name == "GATE_OUT"
    assert folder.name == f"{event_id}_TGHU9876543"

    frame = make_dummy_frame()
    manager.save_captured_frame(event_id, "cam_04", frame, 1, gate_id="gate-out-1", direction="GATE_OUT")
    manager.save_captured_frame(event_id, "cam_05", frame, 1, gate_id="gate-out-1", direction="GATE_OUT")
    manager.save_captured_frame(event_id, "cam_06", frame, 1, gate_id="gate-out-1", direction="GATE_OUT")

    assert (folder / "cam_04_frame_001.jpg").is_file()
    assert (folder / "cam_05_frame_001.jpg").is_file()
    assert (folder / "cam_06_frame_001.jpg").is_file()

    # Gate In remains untouched
    assert len(list((evidence_root / "GATE_IN").iterdir())) == 0
    assert len(list((evidence_root / "GATE_OUT").iterdir())) == 1


def test_multicamera_passage_session_single_folder(tmp_path):
    evidence_root = tmp_path / "evidence"
    legacy_root = tmp_path / "snapshots"
    ev_mgr = EvidenceFolderManager(root=evidence_root, legacy_root=legacy_root)
    session_mgr = PassageSessionManager(target_frames_per_camera=5, evidence_manager=ev_mgr)

    ts = datetime.now(timezone.utc)
    cameras = ["cam_in_left", "cam_in_top", "cam_in_right"]
    frame = make_dummy_frame()

    # 3 cameras assigned to Gate-In
    det1 = [Detection(camera_id="cam_in_left", class_name="container", confidence=0.92, bbox=(50.0, 50.0, 500.0, 400.0), frame_timestamp=ts)]
    det2 = [Detection(camera_id="cam_in_top", class_name="container", confidence=0.95, bbox=(50.0, 50.0, 500.0, 400.0), frame_timestamp=ts)]
    det3 = [Detection(camera_id="cam_in_right", class_name="container", confidence=0.90, bbox=(50.0, 50.0, 500.0, 400.0), frame_timestamp=ts)]

    # Collect multiple frames across all 3 cameras
    for seq in range(1, 4):
        session_mgr.record_frame("cam_in_left", "gate-1", "GATE_IN", frame, ts, detections=det1, gate_cameras=cameras, track_id=101)
        session_mgr.record_frame("cam_in_top", "gate-1", "GATE_IN", frame, ts, detections=det2, gate_cameras=cameras, track_id=101)
        session_mgr.record_frame("cam_in_right", "gate-1", "GATE_IN", frame, ts, detections=det3, gate_cameras=cameras, track_id=101)

    active = session_mgr.get_active_session("gate-1")
    assert active is not None
    event_id = active.event_id

    # Add OCR consensus
    read = OCRRead(raw_text="MSCU1234566", confidence=0.95)
    session_mgr.record_ocr_result("gate-1", read)

    # Check folder in evidence/GATE_IN
    gate_in_dir = evidence_root / "GATE_IN"
    folders = list(gate_in_dir.iterdir())
    assert len(folders) == 1
    folder = folders[0]
    assert folder.name == f"{event_id}_MSCU1234566"

    # Verify all frames from all 3 cameras are in this single folder
    for cam in cameras:
        for seq in range(1, 4):
            expected = folder / f"{cam}_frame_{seq:03d}.jpg"
            assert expected.is_file(), f"Missing {expected}"

    # Verify no subdirectories
    assert not any(p.is_dir() for p in folder.iterdir())


def test_merge_folders_on_dedup(tmp_path):
    evidence_root = tmp_path / "evidence"
    legacy_root = tmp_path / "snapshots"
    ev_mgr = EvidenceFolderManager(root=evidence_root, legacy_root=legacy_root)

    source_id = str(uuid4())
    target_id = str(uuid4())

    frame = make_dummy_frame()
    ev_mgr.save_captured_frame(source_id, "cam_01", frame, 1, gate_id="gate-1", direction="GATE_IN")
    ev_mgr.save_captured_frame(target_id, "cam_02", frame, 1, gate_id="gate-1", direction="GATE_IN")

    merged = ev_mgr.merge_event_folders(source_id, target_id, container_number="MEDU1112223")
    assert (merged / "cam_01_frame_001.jpg").is_file()
    assert (merged / "cam_02_frame_001.jpg").is_file()
    assert merged.name == f"{target_id}_MEDU1112223"


def test_collision_prevention_returning_container(tmp_path):
    evidence_root = tmp_path / "evidence"
    legacy_root = tmp_path / "snapshots"
    ev_mgr = EvidenceFolderManager(root=evidence_root, legacy_root=legacy_root)

    # First visit
    visit1_id = str(uuid4())
    folder1 = ev_mgr.get_or_create_folder(visit1_id, gate_id="gate-in-1", direction="GATE_IN", container_number="MSCU1234566")

    # Second visit of the exact same physical container number
    visit2_id = str(uuid4())
    folder2 = ev_mgr.get_or_create_folder(visit2_id, gate_id="gate-in-1", direction="GATE_IN", container_number="MSCU1234566")

    assert folder1.name == f"{visit1_id}_MSCU1234566"
    assert folder2.name == f"{visit2_id}_MSCU1234566"
    assert folder1 != folder2
    assert folder1.exists() and folder2.exists()

    gate_in_folders = list((evidence_root / "GATE_IN").iterdir())
    assert len(gate_in_folders) == 2


def test_concurrent_gate_in_and_gate_out(tmp_path):
    evidence_root = tmp_path / "evidence"
    legacy_root = tmp_path / "snapshots"
    ev_mgr = EvidenceFolderManager(root=evidence_root, legacy_root=legacy_root)

    # Gate-In Event
    in_event_id = str(uuid4())
    in_folder = ev_mgr.get_or_create_folder(in_event_id, gate_id="gate-in-1", direction="GATE_IN", container_number="MSCU1234566")

    # Gate-Out Event processed at the same time
    out_event_id = str(uuid4())
    out_folder = ev_mgr.get_or_create_folder(out_event_id, gate_id="gate-out-1", direction="GATE_OUT", container_number="TGHU9876543")

    frame = make_dummy_frame()
    ev_mgr.save_captured_frame(in_event_id, "cam_01", frame, 1, gate_id="gate-in-1", direction="GATE_IN")
    ev_mgr.save_captured_frame(in_event_id, "cam_02", frame, 1, gate_id="gate-in-1", direction="GATE_IN")
    ev_mgr.save_captured_frame(in_event_id, "cam_03", frame, 1, gate_id="gate-in-1", direction="GATE_IN")

    ev_mgr.save_captured_frame(out_event_id, "cam_04", frame, 1, gate_id="gate-out-1", direction="GATE_OUT")
    ev_mgr.save_captured_frame(out_event_id, "cam_05", frame, 1, gate_id="gate-out-1", direction="GATE_OUT")
    ev_mgr.save_captured_frame(out_event_id, "cam_06", frame, 1, gate_id="gate-out-1", direction="GATE_OUT")

    # Strictly isolated: in_folder has only cam_01, cam_02, cam_03
    in_files = [f.name for f in in_folder.iterdir()]
    assert "cam_01_frame_001.jpg" in in_files
    assert "cam_02_frame_001.jpg" in in_files
    assert "cam_03_frame_001.jpg" in in_files
    assert not any("cam_04" in f or "cam_05" in f or "cam_06" in f for f in in_files)

    # out_folder has only cam_04, cam_05, cam_06
    out_files = [f.name for f in out_folder.iterdir()]
    assert "cam_04_frame_001.jpg" in out_files
    assert "cam_05_frame_001.jpg" in out_files
    assert "cam_06_frame_001.jpg" in out_files
    assert not any("cam_01" in f or "cam_02" in f or "cam_03" in f for f in out_files)


def test_repository_finalize_passage_session_single_folder(repository, snapshots):
    evidence_root = snapshots.evidence_manager.evidence_root

    session_mgr = PassageSessionManager(target_frames_per_camera=5, evidence_manager=snapshots.evidence_manager)

    ts = datetime.now(timezone.utc)
    cameras = ["cam_1", "cam_2", "cam_3"]
    from app.config.settings import CameraConfig
    repository.sync_cameras([CameraConfig(id=c, name=c, gate_id="gate-in-1", source_env=c.upper()) for c in cameras])
    frame = make_dummy_frame()

    det1 = [Detection(camera_id="cam_1", class_name="container", confidence=0.92, bbox=(50.0, 50.0, 500.0, 400.0), frame_timestamp=ts)]
    det2 = [Detection(camera_id="cam_2", class_name="container", confidence=0.95, bbox=(50.0, 50.0, 500.0, 400.0), frame_timestamp=ts)]
    det3 = [Detection(camera_id="cam_3", class_name="container", confidence=0.90, bbox=(50.0, 50.0, 500.0, 400.0), frame_timestamp=ts)]

    session_mgr.record_frame("cam_1", "gate-in-1", "GATE_IN", frame, ts, detections=det1, gate_cameras=cameras, track_id=200)
    session_mgr.record_frame("cam_2", "gate-in-1", "GATE_IN", frame, ts, detections=det2, gate_cameras=cameras, track_id=200)
    session_mgr.record_frame("cam_3", "gate-in-1", "GATE_IN", frame, ts, detections=det3, gate_cameras=cameras, track_id=200)

    session = session_mgr.get_active_session("gate-in-1")
    session_mgr.record_ocr_result("gate-in-1", OCRRead(raw_text="MSCU1234566", confidence=0.96))

    # Finalize passage session via repository
    event_id = repository.finalize_passage_session(session, snapshots)
    assert event_id == session.event_id

    # Verify database event record
    event_rec = repository.event(event_id)
    assert event_rec is not None
    assert event_rec["container_number"] == "MSCU1234566"
    assert event_rec["status"] == "CONFIRMED"

    # Verify single evidence folder in evidence/GATE_IN
    ev_folder = evidence_root / "GATE_IN" / f"{event_id}_MSCU1234566"
    assert ev_folder.is_dir()

    # Verify best frames exist inside that folder
    assert (ev_folder / "best_cam_1.jpg").is_file()
    assert (ev_folder / "best_cam_2.jpg").is_file()
    assert (ev_folder / "best_cam_3.jpg").is_file()

    # Verify legacy compatible filenames exist inside that folder
    assert (ev_folder / "cam_1.jpg").is_file()
    assert (ev_folder / "cam_2.jpg").is_file()
    assert (ev_folder / "cam_3.jpg").is_file()

    # Verify event.json and evidence.json are saved inside that folder
    assert (ev_folder / "event.json").is_file()
    assert (ev_folder / "evidence.json").is_file()

    # Verify snapshots.path resolves correctly for each snapshot in the DB record
    for snap in event_rec["snapshots"]:
        p = snapshots.path(snap["image_path"])
        assert p.is_file(), f"Path {snap['image_path']} not found"


def test_camera_3_missing_handled_honestly_no_fake_images(repository, snapshots):
    """
    Scenario: Cam 1 and Cam 2 capture the container, but Cam 3 is disconnected or misses.
    Verify:
    1. Cam 1 and Cam 2 best frames are saved in the single evidence folder.
    2. Cam 3 is NOT fabricated or stolen from Cam 1/2.
    3. Status is CONFIRMED with correct OCR.
    4. Exactly 1 evidence folder is created.
    """
    evidence_root = snapshots.evidence_manager.evidence_root
    session_mgr = PassageSessionManager(target_frames_per_camera=4, evidence_manager=snapshots.evidence_manager)

    ts = datetime.now(timezone.utc)
    cameras = ["cam_1", "cam_2", "cam_3"]
    from app.config.settings import CameraConfig
    repository.sync_cameras([CameraConfig(id=c, name=c, gate_id="gate-in-missing", source_env=c.upper()) for c in cameras])
    frame1 = make_dummy_frame(text="CAM1")
    frame2 = make_dummy_frame(text="CAM2")

    det1 = [Detection(camera_id="cam_1", class_name="container", confidence=0.91, bbox=(50.0, 50.0, 500.0, 400.0), frame_timestamp=ts)]
    det2 = [Detection(camera_id="cam_2", class_name="container", confidence=0.94, bbox=(50.0, 50.0, 500.0, 400.0), frame_timestamp=ts)]

    # Only Cam 1 and Cam 2 record frames
    session_mgr.record_frame("cam_1", "gate-in-missing", "GATE_IN", frame1, ts, detections=det1, gate_cameras=cameras, track_id=301)
    session_mgr.record_frame("cam_2", "gate-in-missing", "GATE_IN", frame2, ts, detections=det2, gate_cameras=cameras, track_id=301)

    session_mgr.record_ocr_result("gate-in-missing", OCRRead(raw_text="MSKU1234565", confidence=0.98))
    session = session_mgr.get_active_session("gate-in-missing")

    # Finalize
    event_id = repository.finalize_passage_session(session, snapshots)
    ev_folder = evidence_root / "GATE_IN" / f"{event_id}_MSKU1234565"

    assert ev_folder.is_dir()
    # Cam 1 and Cam 2 exist
    assert (ev_folder / "best_cam_1.jpg").is_file()
    assert (ev_folder / "best_cam_2.jpg").is_file()
    # Cam 3 does NOT exist and was not fabricated
    assert not (ev_folder / "best_cam_3.jpg").exists()
    assert not (ev_folder / "cam_3.jpg").exists()

    # Database only has snapshots for cam_1 and cam_2
    event_rec = repository.event(event_id)
    assert event_rec["status"] == "CONFIRMED"
    saved_cams = {s["camera_id"] for s in event_rec["snapshots"]}
    assert saved_cams == {"cam_1", "cam_2"}
    assert "cam_3" not in saved_cams


def test_container_a_and_container_b_never_mix_images_or_ocr(repository, snapshots):
    """
    Scenario: Sequential containers A (TGHU1234567) and B (MEDU9876543) pass through the same gate.
    Verify:
    1. Container A has its own folder and only its own images.
    2. Container B has its own folder and only its own images.
    3. No images or OCR text from A leak into B or vice versa.
    """
    evidence_root = snapshots.evidence_manager.evidence_root
    cameras = ["cam_1", "cam_2", "cam_3"]
    from app.config.settings import CameraConfig
    repository.sync_cameras([CameraConfig(id=c, name=c, gate_id="gate-in-iso", source_env=c.upper()) for c in cameras])

    mgr = PassageSessionManager(target_frames_per_camera=4, evidence_manager=snapshots.evidence_manager)
    ts_a = datetime.now(timezone.utc)

    # --- Container A ---
    frame_a = np.full((100, 100, 3), 50, dtype=np.uint8)
    det_a = [Detection("cam_1", ts_a, "container", 0.95, (10, 10, 90, 90))]
    mgr.record_frame("cam_1", "gate-in-iso", "GATE_IN", frame_a, ts_a, detections=det_a, gate_cameras=cameras, track_id=10)
    mgr.record_ocr_result("gate-in-iso", OCRRead("TGHU1234567", 0.97))
    sess_a = mgr.get_active_session("gate-in-iso")
    eid_a = repository.finalize_passage_session(sess_a, snapshots)
    sess_a.finalized = True
    mgr.active_sessions.clear()

    # --- Container B ---
    ts_b = ts_a + timedelta(seconds=15)
    frame_b = np.full((100, 100, 3), 200, dtype=np.uint8)
    det_b = [Detection("cam_1", ts_b, "container", 0.93, (10, 10, 90, 90))]
    mgr.record_frame("cam_1", "gate-in-iso", "GATE_IN", frame_b, ts_b, detections=det_b, gate_cameras=cameras, track_id=20)
    mgr.record_ocr_result("gate-in-iso", OCRRead("MEDU9876548", 0.96))
    sess_b = mgr.get_active_session("gate-in-iso")
    eid_b = repository.finalize_passage_session(sess_b, snapshots)

    # Verify completely separate events
    assert eid_a != eid_b

    # Verify completely separate folders
    folder_a = evidence_root / "GATE_IN" / f"{eid_a}_TGHU1234567"
    folder_b = evidence_root / "GATE_IN" / f"{eid_b}_MEDU9876548"

    assert folder_a.is_dir()
    assert folder_b.is_dir()
    assert folder_a != folder_b

    # Check images inside A vs B
    img_a_bytes = (folder_a / "best_cam_1.jpg").read_bytes()
    img_b_bytes = (folder_b / "best_cam_1.jpg").read_bytes()
    assert img_a_bytes != img_b_bytes

    # Database integrity
    rec_a = repository.event(eid_a)
    rec_b = repository.event(eid_b)
    assert rec_a["container_number"] == "TGHU1234567"
    assert rec_b["container_number"] == "MEDU9876548"


