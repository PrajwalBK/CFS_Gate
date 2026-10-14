"""Regression tests for identity, cloud outcomes and durable evidence ordering."""
from dataclasses import replace
from datetime import timedelta
import threading
import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

from app.domain import Observation, OCRRead, Validation
from app.events.cloud_dispatcher import CloudOutboxDispatcher
from app.events.passage_outbox import PassageOutbox
from app.events.passage_session import ContainerPassageSession, PassageSessionManager
from app.ocr.consensus import CharacterConsensus
from app.runtime import GateRuntime
from app.database.database import Database


VALID = Validation('CSQU3054383', 'CSQU3054383', .99, True, True, 'VALID', '22G1')


def session_for(job):
    session = ContainerPassageSession('TRACK_RETRY', job.gate_id, 'ENTRY',
                                      [job.frame.camera_id], job.frame.timestamp, track_id=job.track_id)
    image = np.random.default_rng(42).integers(0, 255, job.frame.image.shape, dtype=np.uint8)
    session.add_frame(job.frame.camera_id, image, job.frame.timestamp, [job.detection])
    session.add_ocr_candidate(VALID)
    return session


def save_event(repository, snapshots, job):
    return repository.save_observation(Observation(job.origin_key, job.gate_id,
        job.track_id, job.detection, VALID, 'ENTRY', job.frame.image, job.started_at, True), snapshots, 30)[0]


def test_false_validation_flag_cannot_confirm_observation(repository, snapshots, make_job):
    job = make_job()
    bad = replace(VALID, raw_text='MSCU1234567', normalized_text='MSCU1234567')
    event_id, _ = repository.save_observation(Observation(job.origin_key, job.gate_id,
        job.track_id, job.detection, bad, 'ENTRY', job.frame.image, job.started_at, True), snapshots, 30)
    event = repository.event(event_id)
    assert event['status'] == 'NEEDS_REVIEW'
    assert not event['ocr_results'][0]['valid_check_digit']
    assert not CharacterConsensus.resolve([bad, bad]).valid_check_digit


def test_high_confidence_invalid_checksum_cannot_confirm_session(repository, snapshots, make_job):
    session = session_for(make_job())
    # Even a buggy consensus implementation cannot bypass the DB boundary.
    session.get_consensus_container_number = Mock(return_value=('MSCU1234567', '22G1', .99))
    event_id = repository.finalize_passage_session(session, snapshots)
    event = repository.event(event_id)
    assert event['status'] == 'NEEDS_REVIEW'
    assert not event['ocr_results'][0]['valid_check_digit']
    assert repository.get_unsynced_confirmed_events() == []


@pytest.mark.parametrize('outcome,state,pending', [
    (True, 'DELIVERED', 0), ({'eventId': 'remote-id'}, 'DELIVERED', 0),
    ('REJECTED', 'REJECTED', 0), ('CONFLICT', 'CONFLICT', 1), ('OFFLINE', 'OFFLINE', 1),
])
def test_cloud_outcomes_persist_without_false_acknowledgement(
        settings, repository, snapshots, make_job, outcome, state, pending):
    event_id = save_event(repository, snapshots, make_job())
    cfg = settings.model_copy(update={'prosper_enabled': True, 'prosper_max_retries': 1})
    dispatcher = CloudOutboxDispatcher(cfg, repository=repository)
    payload = {'VisitId': event_id, 'Containers': [{'ContainerNumber': VALID.normalized_text}]}
    with patch.object(dispatcher.session, 'post_gate_event', return_value=outcome):
        dispatcher.start()
        try:
            dispatcher.enqueue(None, event_id, payload=payload)
            deadline = time.monotonic() + 3
            while dispatcher.queue.unfinished_tasks and time.monotonic() < deadline:
                time.sleep(.01)
            assert dispatcher.queue.unfinished_tasks == 0
        finally:
            dispatcher.stop()
    event = repository.event(event_id)
    assert event['cloud_delivery_state'] == state
    assert event['cloud_synced'] is (state == 'DELIVERED')
    assert bool(event['cloud_synced_at']) is (state == 'DELIVERED')
    assert repository.count_unsynced_confirmed_events() == pending
    assert repository.cloud_delivery_counts()[state] == 1
    # Re-opening the repository/dispatcher must preserve terminal failures.
    if state == 'REJECTED':
        assert repository.get_unsynced_confirmed_events() == []
        repository.reset_cloud_failures()
        assert repository.event(event_id)['cloud_delivery_state'] == 'PENDING'
        assert repository.count_unsynced_confirmed_events() == 1


def test_invalid_prebuilt_cloud_payload_never_posts(settings, repository, snapshots, make_job):
    event_id = save_event(repository, snapshots, make_job())
    dispatcher = CloudOutboxDispatcher(settings.model_copy(update={'prosper_enabled': True}), repository=repository)
    with patch.object(dispatcher.session, 'post_gate_event') as post:
        dispatcher.start()
        try:
            dispatcher.enqueue(None, event_id, payload={'Containers': [{'ContainerNumber': 'MSCU1234567'}]})
            deadline = time.monotonic() + 3
            while dispatcher.queue.unfinished_tasks and time.monotonic() < deadline:
                time.sleep(.01)
            assert dispatcher.queue.unfinished_tasks == 0
            post.assert_not_called()
        finally:
            dispatcher.stop()
    assert repository.event(event_id)['cloud_delivery_state'] == 'BLOCKED'
    assert not repository.event(event_id)['cloud_synced']


def test_failed_finalization_survives_restart_and_replays_once(settings, repository, snapshots, camera, make_job):
    runtime = GateRuntime(settings, [camera], repository, snapshots)
    session = session_for(make_job())
    with patch.object(repository, 'finalize_passage_session', side_effect=OSError('DB offline')):
        runtime._finalize_passage_session(session)
    assert runtime.passage_outbox.pending() == 1
    assert repository.event(session.event_id) is None
    reopened = GateRuntime(settings, [camera], repository, snapshots)
    with reopened.passage_outbox.connection() as db:
        db.execute('UPDATE passages SET retry_at=0')
    restored = reopened.passage_outbox.ready()[0]
    assert restored.event_id == session.event_id
    assert restored.track_id == session.track_id
    np.testing.assert_array_equal(restored.get_best_frame_per_camera()[camera.id].image,
                                  session.get_best_frame_per_camera()[camera.id].image)
    reopened._retry_passage_finalizations()
    assert reopened.passage_outbox.pending() == 0
    event = repository.event(session.event_id)
    assert event['status'] == 'CONFIRMED'
    counts = [len(event[k]) for k in ('detections', 'snapshots', 'ocr_results')]
    # Crash after SQL commit but before local outbox acknowledgement.
    reopened.passage_outbox.save(session)
    reopened._retry_passage_finalizations()
    assert [len(repository.event(session.event_id)[k]) for k in ('detections', 'snapshots', 'ocr_results')] == counts
    assert repository.events()['total'] == 1


def test_staging_failure_does_not_discard_active_session(settings, make_job):
    store = Mock()
    store.save.side_effect = OSError('Disk full')
    manager = PassageSessionManager(settings, outbox=store)
    job = make_job()
    session = manager.get_or_create_session(job.gate_id, 'ENTRY', job.frame.timestamp, [job.frame.camera_id])
    with pytest.raises(OSError):
        manager.check_and_finalize(job.frame.timestamp + timedelta(seconds=30))
    assert manager.get_active_session(job.gate_id) is session
    assert not session.finalized


def test_json_export_failure_keeps_committed_images(repository, snapshots, make_job):
    session = session_for(make_job())
    with patch.object(snapshots, 'save_event_json', side_effect=OSError('Export failed')):
        repository.finalize_passage_session(session, snapshots)
    event = repository.event(session.event_id)
    assert event['status'] == 'CONFIRMED'
    assert all(snapshots.path(s['image_path']).is_file() for s in event['snapshots'])


def test_ocr_waits_for_durable_save(settings, repository, snapshots, camera, make_job):
    settings.ocr_confirmations = 1
    runtime = GateRuntime(settings, [camera], repository, snapshots)
    job = make_job()
    entered, release = threading.Event(), threading.Event()
    save = runtime.ocr_store.save
    def blocked_save(job, crop):
        entered.set()
        assert release.wait(3)
        return save(job, crop)
    runtime.ocr_store.save = blocked_save
    runtime._execute_ocr = Mock(return_value=(OCRRead(VALID.raw_text, .99), VALID))
    runtime.queue.put(job)
    writer = threading.Thread(target=runtime._save_ocr_loop)
    reader = threading.Thread(target=runtime._ocr_loop)
    writer.start()
    reader.start()
    try:
        assert entered.wait(2)
        time.sleep(.1)
        runtime._execute_ocr.assert_not_called()
        assert runtime.ocr_store.next_id() is None
        release.set()
        deadline = time.monotonic() + 3
        while repository.events()['total'] == 0 and time.monotonic() < deadline:
            time.sleep(.01)
        assert repository.events()['total'] == 1
        runtime._execute_ocr.assert_called_once()
    finally:
        release.set()
        runtime._stop.set()
        writer.join(3)
        reader.join(3)
    assert not writer.is_alive() and not reader.is_alive()


def test_cloud_columns_upgrade_preserves_legacy_records(repository, snapshots, make_job):
    event_id = save_event(repository, snapshots, make_job())
    repository.mark_cloud_synced(event_id)
    with repository.db.engine.begin() as conn:
        conn.exec_driver_sql('ALTER TABLE gate_events DROP COLUMN cloud_delivery_state')
        conn.exec_driver_sql('ALTER TABLE gate_events DROP COLUMN cloud_delivery_error')
    with pytest.raises(RuntimeError, match='initialize'):
        repository.db.check()
    repository.db.initialize()
    repository.db.check()
    event = repository.event(event_id)
    assert event['container_number'] == VALID.normalized_text
    assert event['cloud_delivery_state'] == 'LEGACY'
    assert event['cloud_synced'] is True


def test_failure_cannot_revoke_existing_cloud_acknowledgement(repository, snapshots, make_job):
    event_id = save_event(repository, snapshots, make_job())
    repository.mark_cloud_synced(event_id, 'remote-id')
    repository.mark_cloud_delivery(event_id, 'REJECTED', 'Late duplicate response')
    event = repository.event(event_id)
    assert event['cloud_synced'] is True
    assert event['cloud_delivery_state'] == 'DELIVERED'
    assert event['cloud_event_id'] == 'remote-id'


def test_restart_retry_worker_runs_with_ai_disabled(settings, repository, snapshots, camera, make_job):
    camera.enabled = False
    session = session_for(make_job())
    outbox = PassageOutbox(settings.upload_directory / 'ocr-spool')
    outbox.save(session)
    runtime = GateRuntime(settings, [camera], repository, snapshots)
    runtime.start()
    try:
        deadline = time.monotonic() + 3
        while runtime.passage_outbox.pending() and time.monotonic() < deadline:
            time.sleep(.01)
        assert runtime.passage_outbox.pending() == 0
        assert repository.event(session.event_id)['status'] == 'CONFIRMED'
        assert runtime.model_status['detector'] == 'DISABLED'
    finally:
        assert runtime.stop()
