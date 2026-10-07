from datetime import timedelta
import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from app.config.settings import CameraConfig
from app.database.models import GateEvent, DetectionRecord, OCRResult, Snapshot
from app.domain import OCRRead, Observation
from app.events.event_manager import EventManager
from app.ocr.validator import ContainerValidator


def observation(job, text='CSQU3054383'):
    validation = ContainerValidator().validate(OCRRead(text, .94))
    return Observation(job.origin_key, job.gate_id, job.track_id, job.detection, validation, job.direction,
                       job.frame.image, job.started_at, validation.valid_check_digit)


def test_confirmation_and_hundred_frame_duplicate_prevention(settings, repository, snapshots, make_job):
    manager = EventManager(settings, repository, snapshots)
    value = ContainerValidator().validate(OCRRead('CSQU3054383', .94))
    assert manager.observe(make_job(1), value) is None
    event_id, created = manager.observe(make_job(2), value)
    assert created
    for i in range(3, 103):
        assert manager.observe(make_job(i), value) == (event_id, False)
    assert repository.events()['total'] == 1
    detail = repository.event(event_id)
    assert detail['container_number'] == 'CSQU3054383'
    assert len(detail['detections']) == len(detail['ocr_results']) == len(detail['snapshots']) == 1
    assert len(list(snapshots.root.rglob('*.jpg'))) == 1


def test_multicamera_identity_distinct_containers_direction_and_repeat_visit(repository, snapshots, make_job):
    repository.sync_cameras([CameraConfig(id='camera-2', name='Side', gate_id='lane-1', source_env='CAMERA_2_RTSP')])
    first = make_job(direction='ENTRY')
    event_id, _ = repository.save_observation(observation(first), snapshots, 30)
    second = make_job(camera_id='camera-2', origin='another-run:camera-2:1', direction='ENTRY')
    assert repository.save_observation(observation(second), snapshots, 30) == (event_id, False)
    assert len(repository.event(event_id)['detections']) == 2
    # Close in time, different identity: must not fuse.
    repository.save_observation(observation(make_job(origin='run:camera-1:2', direction='ENTRY'), 'MSCU6639870'), snapshots, 30)
    # Opposite direction: a different movement.
    repository.save_observation(observation(make_job(origin='run:camera-1:3', direction='EXIT')), snapshots, 30)
    # Same identity returns after the configured window.
    repository.save_observation(observation(make_job(origin='new-run:camera-1:4', direction='ENTRY',
                                                     timestamp=first.frame.timestamp + timedelta(seconds=60))), snapshots, 30)
    assert repository.events()['total'] == 4
    assert repository.events(container='CSQU3054383', event_type='ENTRY')['total'] == 2


def test_restart_dedup_is_database_backed(repository, snapshots, make_job):
    from app.database.repositories.gate import GateRepository
    first_id, _ = repository.save_observation(observation(make_job()), snapshots, 30)
    restarted = GateRepository(repository.db)
    assert restarted.save_observation(observation(make_job(origin='new-process:camera-1:5')), snapshots, 30) == (first_id, False)


def test_dwell_tracker_reassignment_keeps_waiting_container_in_one_event(repository, snapshots, make_job):
    repository.sync_cameras([
        CameraConfig(id='gate-out-1', name='Gate OUT 1', gate_id='lane-out', source_env='CAM_OUT_1'),
        CameraConfig(id='gate-out-2', name='Gate OUT 2', gate_id='lane-out', source_env='CAM_OUT_2'),
    ])
    first_job = make_job(camera_id='gate-out-1', gate_id='lane-out', origin='dwell:gate-out-1:first',
                         direction='EXIT')
    first_id, created = repository.save_observation(observation(first_job), snapshots, 30,
                                                    same_container_window_seconds=900)
    assert created
    second_job = make_job(camera_id='gate-out-2', gate_id='lane-out', origin='dwell:gate-out-2:reassigned',
                          direction='EXIT', timestamp=first_job.frame.timestamp + timedelta(seconds=210))
    second_id, second_created = repository.save_observation(observation(second_job), snapshots, 30,
                                                             same_container_window_seconds=900)
    assert (second_id, second_created) == (first_id, False)
    assert repository.events(container='CSQU3054383', event_type='EXIT')['total'] == 1


def test_invalid_ocr_is_review_without_container_identity(settings, repository, snapshots, make_job):
    manager = EventManager(settings, repository, snapshots)
    invalid = ContainerValidator().validate(OCRRead('CSQU3054384', .99))
    for i in range(settings.ocr_max_attempts):
        result = manager.observe(make_job(i), invalid)
    detail = repository.event(result[0])
    assert detail['container_number'] is None and detail['status'] == 'NEEDS_REVIEW'
    assert detail['ocr_results'][0]['raw_text'] == 'CSQU3054384'
    assert not detail['ocr_results'][0]['valid_check_digit']


def test_retrying_same_frame_is_not_independent_ocr_evidence(settings, repository, snapshots, make_job):
    manager = EventManager(settings, repository, snapshots)
    value = ContainerValidator().validate(OCRRead('CSQU3054383', .94))
    for _ in range(10):
        assert manager.observe(make_job(1), value) is None
    assert repository.events()['total'] == 0


def test_database_failure_does_not_mark_event_done(settings, repository, snapshots, make_job, monkeypatch):
    manager = EventManager(settings, repository, snapshots)
    value = ContainerValidator().validate(OCRRead('CSQU3054383', .94))
    manager.observe(make_job(1), value)
    original = repository.save_observation
    def fail(*args):
        raise OSError('Unavailable')
    monkeypatch.setattr(repository, 'save_observation', fail)
    with pytest.raises(OSError):
        manager.observe(make_job(2), value)
    assert manager.evidence[make_job().origin_key].event_id is None
    monkeypatch.setattr(repository, 'save_observation', original)
    assert manager.observe(make_job(2), value)[1]
    assert repository.events()['total'] == 1


def test_foreign_key_and_snapshot_failure_roll_back_transaction(repository, snapshots, make_job, monkeypatch):
    with pytest.raises(IntegrityError):
        repository.save_observation(observation(make_job(camera_id='missing')), snapshots, 30)
    assert repository.events()['total'] == 0
    def fail(*args):
        raise OSError('Disk full')
    monkeypatch.setattr(snapshots, 'save', fail)
    with pytest.raises(OSError):
        repository.save_observation(observation(make_job()), snapshots, 30)
    with repository.db.sessions() as session:
        for model in (GateEvent, DetectionRecord, OCRResult, Snapshot):
            assert session.scalar(select(func.count()).select_from(model)) == 0


def test_snapshot_references_cannot_escape_root(snapshots):
    with pytest.raises(ValueError):
        snapshots.path('../outside.jpg')


def test_multicamera_event_snapshot_bundling(repository, snapshots, make_job):
    import numpy as np
    # Configure 4 cameras for lane-in
    repository.sync_cameras([
        CameraConfig(id=f'gate-in-{i}', name=f'Gate IN {i}', gate_id='lane-in', source_env=f'CAM_IN_{i}')
        for i in range(1, 5)
    ])
    job = make_job(camera_id='gate-in-1', gate_id='lane-in')
    obs = observation(job, 'CSQU3054383')
    extra_frames = {
        'gate-in-2': np.zeros((100, 100, 3), dtype=np.uint8),
        'gate-in-3': np.zeros((100, 100, 3), dtype=np.uint8),
        'gate-in-4': np.zeros((100, 100, 3), dtype=np.uint8),
    }
    event_id, created = repository.save_observation(obs, snapshots, 30, additional_frames=extra_frames)
    assert created
    event = repository.event(event_id)
    assert event['gate_id'] == 'lane-in'
    assert len(event['snapshots']) == 4
    assert event['primary_camera_id'] == 'gate-in-1'
    assert event['primary_snapshot_id'] is not None
    snap_cameras = {s['camera_id'] for s in event['snapshots']}
    assert snap_cameras == {'gate-in-1', 'gate-in-2', 'gate-in-3', 'gate-in-4'}

    # Verify event directory exists and holds all 4 snapshot files
    event_dir = snapshots.root / event_id
    assert event_dir.is_dir()
    saved_files = list(event_dir.glob('*.jpg'))
    assert len(saved_files) == 4


def test_single_passage_multicamera_and_noisy_reads_fused_into_single_event(repository, snapshots, make_job):
    repository.sync_cameras([
        CameraConfig(id='gate-out-1', name='Gate OUT 1', gate_id='lane-out', source_env='CAM_OUT_1'),
        CameraConfig(id='gate-out-2', name='Gate OUT 2', gate_id='lane-out', source_env='CAM_OUT_2'),
    ])
    from app.domain import OCRRead
    from app.ocr.validator import ContainerValidator

    val_valid = ContainerValidator().validate(OCRRead('AXIU 196088 8', 0.99))
    val_partial = ContainerValidator().validate(OCRRead('AXU1960888', 0.95))
    val_noisy = ContainerValidator().validate(OCRRead('AXIU 981038 0', 0.85))

    job1 = make_job(camera_id='gate-out-1', gate_id='lane-out', origin='dwell:gate-out-1:track1', direction='EXIT')
    obs1 = Observation(job1.origin_key, job1.gate_id, job1.track_id, job1.detection, val_valid, job1.direction,
                       job1.frame.image, job1.started_at, True)
    eid1, created1 = repository.save_observation(obs1, snapshots, 35)
    assert created1

    # Second camera (gate-out-2) sees partial read AXU1960888 within 4 seconds
    job2 = make_job(camera_id='gate-out-2', gate_id='lane-out', origin='dwell:gate-out-2:track2', direction='EXIT',
                    timestamp=job1.frame.timestamp + timedelta(seconds=4))
    obs2 = Observation(job2.origin_key, job2.gate_id, job2.track_id, job2.detection, val_partial, job2.direction,
                       job2.frame.image, job2.started_at, False)
    eid2, created2 = repository.save_observation(obs2, snapshots, 35)
    assert not created2
    assert eid2 == eid1

    # First camera sees another track on the same moving container within 15 seconds
    job3 = make_job(camera_id='gate-out-1', gate_id='lane-out', origin='dwell:gate-out-1:track3', direction='EXIT',
                    timestamp=job1.frame.timestamp + timedelta(seconds=15))
    obs3 = Observation(job3.origin_key, job3.gate_id, job3.track_id, job3.detection, val_noisy, job3.direction,
                       job3.frame.image, job3.started_at, True)
    eid3, created3 = repository.save_observation(obs3, snapshots, 35)
    assert not created3
    assert eid3 == eid1

    # Exactly ONE event in repository for this single container passage
    events = repository.events()
    assert events['total'] == 1
    event = repository.event(eid1)
    assert event['container_number'] == 'AXIU1960888'


def test_gate_in_and_gate_out_multicamera_trigger_and_companion_snapshots(repository, snapshots, make_job):
    import numpy as np
    from app.ocr.validator import ContainerValidator
    from app.domain import OCRRead

    # 1. Sync full gate setup: 3 cameras for Gate IN, 2 cameras for Gate OUT
    repository.sync_cameras([
        CameraConfig(id='gate-in-1', name='Gate IN 1 (Front/Top)', gate_id='lane-in', source_env='CAM_IN_1', role='FRONT_TOP', direction='ENTRY'),
        CameraConfig(id='gate-in-2', name='Gate IN 2 (Left)', gate_id='lane-in', source_env='CAM_IN_2', role='LEFT', direction='ENTRY'),
        CameraConfig(id='gate-in-3', name='Gate IN 3 (Rear)', gate_id='lane-in', source_env='CAM_IN_3', role='REAR', direction='ENTRY'),
        CameraConfig(id='gate-out-1', name='Gate OUT 1 (Front/Top)', gate_id='lane-out', source_env='CAM_OUT_1', role='FRONT_TOP', direction='EXIT'),
        CameraConfig(id='gate-out-2', name='Gate OUT 2 (Left)', gate_id='lane-out', source_env='CAM_OUT_2', role='LEFT', direction='EXIT'),
    ])

    # ------------------ TEST GATE IN (ENTRY) ------------------
    # gate-in-1 triggers detection and captures companion frames from gate-in-2 and gate-in-3
    job_in = make_job(camera_id='gate-in-1', gate_id='lane-in', origin='dwell:gate-in-1:trk1', direction='ENTRY')
    val_in = ContainerValidator().validate(OCRRead('MSKU 123456 7', 0.98))
    extra_in = {
        'gate-in-2': np.full((100, 100, 3), 120, dtype=np.uint8),
        'gate-in-3': np.full((100, 100, 3), 180, dtype=np.uint8),
    }
    obs_in = Observation(job_in.origin_key, job_in.gate_id, job_in.track_id, job_in.detection, val_in, job_in.direction,
                         job_in.frame.image, job_in.started_at, True)
    eid_in, created_in = repository.save_observation(obs_in, snapshots, 35, additional_frames=extra_in)
    assert created_in
    ev_in = repository.event(eid_in)
    assert ev_in['gate_id'] == 'lane-in'
    assert ev_in['event_type'] == 'ENTRY'
    assert ev_in['container_number'] == 'MSKU1234567'
    assert len(ev_in['snapshots']) == 3
    in_cam_ids = {s['camera_id'] for s in ev_in['snapshots']}
    assert in_cam_ids == {'gate-in-1', 'gate-in-2', 'gate-in-3'}

    # ------------------ TEST GATE OUT (EXIT) ------------------
    # gate-out-1 triggers detection and captures companion frame from gate-out-2
    job_out = make_job(camera_id='gate-out-1', gate_id='lane-out', origin='dwell:gate-out-1:trk2', direction='EXIT')
    val_out = ContainerValidator().validate(OCRRead('AXIU 196088 8', 0.99))
    extra_out = {
        'gate-out-2': np.full((100, 100, 3), 200, dtype=np.uint8),
    }
    obs_out = Observation(job_out.origin_key, job_out.gate_id, job_out.track_id, job_out.detection, val_out, job_out.direction,
                          job_out.frame.image, job_out.started_at, True)
    eid_out, created_out = repository.save_observation(obs_out, snapshots, 35, additional_frames=extra_out)
    assert created_out
    ev_out = repository.event(eid_out)
    assert ev_out['gate_id'] == 'lane-out'
    assert ev_out['event_type'] == 'EXIT'
    assert ev_out['container_number'] == 'AXIU1960888'
    assert len(ev_out['snapshots']) == 2
    out_cam_ids = {s['camera_id'] for s in ev_out['snapshots']}
    assert out_cam_ids == {'gate-out-1', 'gate-out-2'}

    # Verify disk files exist for all snapshots of both events
    in_dir = snapshots.root / eid_in
    assert in_dir.is_dir()
    assert len(list(in_dir.glob('*.jpg'))) == 3

    out_dir = snapshots.root / eid_out
    assert out_dir.is_dir()
    assert len(list(out_dir.glob('*.jpg'))) == 2
