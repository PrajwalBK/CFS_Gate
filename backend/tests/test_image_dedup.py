import numpy as np
from datetime import datetime, timezone, timedelta
from app.snapshots.snapshot_manager import SnapshotManager
from app.database.database import Database
from app.database.repositories.gate import GateRepository, ocr_similarity_score
from app.database.models import GateEvent
from app.domain import Observation, Detection, Validation, OCRRead, Frame


def test_compute_similarity_identical_and_different(tmp_path):
    mgr = SnapshotManager(tmp_path)
    rng = np.random.default_rng(42)
    img1 = rng.integers(50, 200, (480, 640, 3), dtype=np.uint8)
    # Identical image
    assert SnapshotManager.compute_similarity(img1, img1) >= 0.999

    # Slightly perturbed image (e.g. minor noise)
    noise = rng.integers(-5, 5, (480, 640, 3), dtype=np.int16)
    img1_noisy = np.clip(img1.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    assert SnapshotManager.compute_similarity(img1, img1_noisy) >= 0.95

    # Completely different image
    img2 = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    assert SnapshotManager.compute_similarity(img1, img2) < 0.30


def test_ocr_similarity_score():
    # Exact match
    assert ocr_similarity_score("MSMU5388338", "MSMU5388338") == 1.0
    # Duplicate pair: MSMU5388338 vs MSMU5386335 (9 out of 11 matching characters)
    score = ocr_similarity_score("MSMU5388338", "MSMU5386335")
    assert score >= 0.80
    # Completely different
    assert ocr_similarity_score("MSMU5388338", "TCLU1234567") < 0.20


def test_save_observation_deduplicates_by_image_and_timestamp(tmp_path):
    from app.config.settings import CameraConfig
    db = Database("sqlite:///:memory:")
    db.initialize()
    repo = GateRepository(db)
    snapshots = SnapshotManager(tmp_path)
    repo.sync_cameras([CameraConfig(id='gate-in-1', name='Gate In 1', gate_id='lane-in', source_env='CAM1')])

    now = datetime(2026, 9, 25, 6, 41, 27, tzinfo=timezone.utc)
    rng = np.random.default_rng(100)
    truck_scene = rng.integers(40, 220, (480, 640, 3), dtype=np.uint8)

    # First event: camera gate-in-1, OCR read MSMU5386335
    det1 = Detection("gate-in-1", now, "container", 0.9, (10, 10, 100, 100))
    val1 = Validation("MSMU5386335", "MSMU5386335", 0.95, True, True, "VALID", size_type="45G1")
    obs1 = Observation(
        origin_key="dwell:gate-in-1:orig1",
        gate_id="lane-in",
        track_id=1,
        detection=det1,
        ocr=val1,
        direction="ENTRY",
        image=truck_scene,
        started_at=now,
        confirmed=True
    )
    event_id1, created1 = repo.save_observation(obs1, snapshots, dedup_seconds=30)
    assert created1 is True

    # Second event 0.5s later: camera gate-in-1, OCR misread MSMU5388338 (different string!)
    # But identical scene / same timestamp
    now2 = now + timedelta(milliseconds=480)
    det2 = Detection("gate-in-1", now2, "container", 0.9, (10, 10, 100, 100))
    val2 = Validation("MSMU5388338", "MSMU5388338", 0.90, True, True, "VALID", size_type="45G1")
    obs2 = Observation(
        origin_key="dwell:gate-in-1:orig2",
        gate_id="lane-in",
        track_id=2,
        detection=det2,
        ocr=val2,
        direction="ENTRY",
        image=truck_scene,
        started_at=now2,
        confirmed=True
    )
    event_id2, created2 = repo.save_observation(obs2, snapshots, dedup_seconds=30)

    # Must be merged into the first event despite the OCR character error!
    assert created2 is False
    assert event_id2 == event_id1


def test_canonical_event_id_does_not_merge_distinct_valid_numbers_by_image(tmp_path):
    from app.config.settings import CameraConfig
    db = Database("sqlite:///:memory:")
    db.initialize()
    repo = GateRepository(db)
    snapshots = SnapshotManager(tmp_path)
    repo.sync_cameras([CameraConfig(id='gate-in-1', name='Gate In 1', gate_id='lane-in', source_env='CAM1')])

    now = datetime(2026, 9, 25, 6, 41, 27, tzinfo=timezone.utc)
    rng = np.random.default_rng(200)
    truck_scene = rng.integers(40, 220, (480, 640, 3), dtype=np.uint8)

    # Save two events manually into the database simulating an earlier unmerged duplicate
    with repo.db.sessions.begin() as session:
        ev1 = GateEvent(
            id="ev-1-primary",
            gate_id="lane-in",
            timestamp=now,
            container_number="MSMU5386335",
            event_type="ENTRY",
            status="CONFIRMED"
        )
        ev2 = GateEvent(
            id="ev-2-duplicate",
            gate_id="lane-in",
            timestamp=now + timedelta(milliseconds=480),
            container_number="MSMU5388338",
            event_type="ENTRY",
            status="CONFIRMED"
        )
        session.add_all([ev1, ev2])

    # Save snapshots on disk for both events
    snapshots.save_side_snapshot("ev-1-primary", "gate-in-1", truck_scene)
    snapshots.save_side_snapshot("ev-2-duplicate", "gate-in-1", truck_scene)

    # Similar pixels cannot override two different checksum-valid identities.
    canonical = repo.canonical_event_id("ev-2-duplicate", max_gap_seconds=300, snapshots=snapshots)
    assert canonical == "ev-2-duplicate"


def test_passage_session_absorbs_unconfirmed_draft_and_deduplicates(tmp_path):
    from app.config.settings import CameraConfig
    from app.events.passage_session import ContainerPassageSession
    from sqlalchemy import select

    db = Database("sqlite:///:memory:")
    db.initialize()
    repo = GateRepository(db)
    snapshots = SnapshotManager(tmp_path)
    cams = ['gate-in-1', 'gate-in-2', 'gate-in-3']
    repo.sync_cameras([
        CameraConfig(id=cid, name=f'Gate In {i+1}', gate_id='lane-in', source_env=f'CAM{i+1}')
        for i, cid in enumerate(cams)
    ])

    t0 = datetime(2026, 10, 17, 7, 16, 50, tzinfo=timezone.utc)
    rng = np.random.default_rng(300)
    img = rng.integers(40, 220, (480, 640, 3), dtype=np.uint8)

    # 1. Early noisy observation creates an unconfirmed draft event (like FINU1788219)
    det1 = Detection("gate-in-2", t0, "container", 0.9, (10, 10, 200, 200))
    val1 = Validation("FINU1788219", "FINU1788219", 0.85, True, True, "VALID", size_type="45G1")
    obs1 = Observation(
        origin_key="dwell:gate-in-2:track_1",
        gate_id="lane-in",
        track_id=1,
        detection=det1,
        ocr=val1,
        direction="ENTRY",
        image=img,
        started_at=t0,
        confirmed=False
    )
    draft_id, created = repo.save_observation(obs1, snapshots, dedup_seconds=35)
    assert created is True

    # Check draft event is NEEDS_REVIEW
    ev_draft = repo.event(draft_id)
    assert ev_draft["status"] == "NEEDS_REVIEW"

    # 2. Multi-camera passage session completes 23 seconds later with confirmed true number FANU3788422
    t1 = t0 + timedelta(seconds=23)
    sess1 = ContainerPassageSession(
        session_id="TRACK_81615EA1",
        gate_id="lane-in",
        direction="ENTRY",
        cameras=cams,
        created_at=t1
    )
    for cid in cams:
        sess1.add_frame(cid, img, t1, [Detection(cid, t1, "container", 0.95, (10, 10, 300, 300))])
    sess1.add_ocr_candidate(OCRRead("FANU3788422", 0.99))
    sess1.add_ocr_candidate(OCRRead("45G1", 0.95))

    final_id_1 = repo.finalize_passage_session(sess1, snapshots)
    # Must absorb the unconfirmed draft event instead of creating a second row!
    assert final_id_1 == draft_id

    ev_confirmed = repo.event(final_id_1)
    assert ev_confirmed["status"] == "CONFIRMED"
    assert ev_confirmed["container_number"] == "FANU3788422"

    # 3. Tracker track-switch creates a secondary passage session 5 seconds later with FANU3788422
    t2 = t1 + timedelta(seconds=5)
    sess2 = ContainerPassageSession(
        session_id="TRACK_170B0C48",
        gate_id="lane-in",
        direction="ENTRY",
        cameras=cams,
        created_at=t2
    )
    for cid in cams:
        sess2.add_frame(cid, img, t2, [Detection(cid, t2, "container", 0.95, (10, 10, 300, 300))])
    sess2.add_ocr_candidate(OCRRead("FANU3788422", 0.99))
    sess2.add_ocr_candidate(OCRRead("45G1", 0.95))

    final_id_2 = repo.finalize_passage_session(sess2, snapshots)
    # Must deduplicate to the SAME canonical event ID!
    assert final_id_2 == draft_id

    # Verify database table contains EXACTLY ONE event row
    with repo.db.sessions() as db_session:
        all_events = db_session.scalars(select(GateEvent).where(GateEvent.gate_id == "lane-in")).all()
        assert len(all_events) == 1
        assert all_events[0].id == draft_id
        assert all_events[0].container_number == "FANU3788422"
        assert all_events[0].status == "CONFIRMED"


def test_sequential_distinct_containers_never_merge_within_15s(tmp_path):
    from app.config.settings import CameraConfig
    db = Database("sqlite:///:memory:")
    db.initialize()
    repo = GateRepository(db)
    snapshots = SnapshotManager(tmp_path)
    repo.sync_cameras([
        CameraConfig(id='gate-out-1', name='Gate Out 1', gate_id='lane-out', source_env='CAM1', role='FRONT_TOP', direction='EXIT'),
        CameraConfig(id='gate-out-2', name='Gate Out 2', gate_id='lane-out', source_env='CAM2', role='LEFT', direction='EXIT'),
    ])

    t0 = datetime(2026, 10, 17, 8, 38, 52, tzinfo=timezone.utc)
    rng = np.random.default_rng(42)
    img_cream = rng.integers(180, 240, (480, 640, 3), dtype=np.uint8)
    img_maroon = rng.integers(20, 80, (480, 640, 3), dtype=np.uint8)

    # Truck 1: IAAU1944191 seen on gate-out-2
    det1 = Detection("gate-out-2", t0, "container", 0.95, (10, 10, 200, 200))
    val1 = Validation("IAAU1944191", "IAAU1944191", 0.98, True, True, "VALID", size_type="45G1")
    obs1 = Observation(
        origin_key="dwell:gate-out-2:trk1",
        gate_id="lane-out",
        track_id=1,
        detection=det1,
        ocr=val1,
        direction="EXIT",
        image=img_cream,
        started_at=t0,
        confirmed=True
    )
    eid1, created1 = repo.save_observation(obs1, snapshots, dedup_seconds=30)
    assert created1 is True

    # Truck 2: CAAU7493269 arrives 15s later on gate-out-1
    t1 = t0 + timedelta(seconds=15)
    det2 = Detection("gate-out-1", t1, "container", 0.98, (15, 15, 220, 220))
    val2 = Validation("CAAU7493269", "CAAU7493269", 0.99, True, True, "VALID", size_type="45G1")
    obs2 = Observation(
        origin_key="dwell:gate-out-1:trk2",
        gate_id="lane-out",
        track_id=2,
        detection=det2,
        ocr=val2,
        direction="EXIT",
        image=img_maroon,
        started_at=t1,
        confirmed=True,
        event_id=eid1  # Even if an active lane tracker erroneously passed previous event_id
    )
    eid2, created2 = repo.save_observation(obs2, snapshots, dedup_seconds=30, ongoing_event_id=eid1)

    # MUST be created as a separate event, NEVER merged!
    assert created2 is True
    assert eid2 != eid1

    # Verify both events exist independently in DB with their correct containers and snapshots
    ev1 = repo.event(eid1)
    ev2 = repo.event(eid2)
    assert ev1["container_number"] == "IAAU1944191"
    assert ev2["container_number"] == "CAAU7493269"
    # Ensure snapshots didn't leak across events
    assert all(s["camera_id"] == "gate-out-2" for s in ev1["snapshots"])
    assert all(s["camera_id"] == "gate-out-1" for s in ev2["snapshots"])


