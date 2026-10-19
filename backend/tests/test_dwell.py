from datetime import datetime, timedelta
from dataclasses import replace
from unittest.mock import Mock
import numpy as np
from app.domain import Frame, Detection, OCRRead
from app.events.occupancy import ViewOccupancy
from app.runtime import GateRuntime
from app.detection.tracker import TemporalTracker
from app.ocr.container_ocr import crop_identification


def measured(t, boxes=((20, 20, 70, 80),), source='source'):
    ts = datetime(2026, 9, 11) + timedelta(seconds=t)
    frame = Frame('camera-1', ts, int(t * 100) + 1, np.zeros((100,100,3), dtype=np.uint8), source)
    return frame, [Detection(frame.camera_id, ts, 'container', .95, box) for box in boxes]


def test_wait_and_clear_require_continuous_observed_empty_zone(settings, camera):
    view = ViewOccupancy(settings)
    view.update(camera, *measured(0))
    origin = view.current.origin
    view.confirm(origin, 'event')
    for t in range(1, 180):
        view.update(camera, *measured(t, ((21,20,71,80),)))
    assert view.status()['state'] == 'WAITING'
    assert view.current.origin == origin
    view.update(camera, *measured(180, ()))
    view.update(camera, *measured(181))
    assert not view.status()['clear_pending']
    # Offline time is not clear time.
    view.update(camera, *measured(1000, ()))
    assert view.current is not None
    for t in range(1001, 1008):
        view.update(camera, *measured(t, ()))
    assert view.status()['state'] == 'CLEAR'
    view.update(camera, *measured(1008))
    assert view.current.origin != origin


def test_different_box_cannot_steal_occupied_zone(settings, camera):
    view = ViewOccupancy(settings)
    view.update(camera, *measured(0, ((0,0,30,30),)))
    origin = view.current.origin
    view.confirm(origin, 'event')
    for t in range(1, 30):
        assert view.update(camera, *measured(t, ((60,60,90,90),))) is None
    assert view.current.origin == origin
    assert view.current.clear_since is None


def test_explicit_exit_line_unlocks_and_does_not_reclaim_departed_box(settings, camera):
    camera.gate_exit_axis = 'x'
    camera.gate_exit_position = .5
    view = ViewOccupancy(settings)
    view.update(camera, *measured(0, ((45,10,95,90),)))
    view.confirm(view.current.origin, 'event')
    view.update(camera, *measured(1, ((52,10,100,90),)))
    assert view.current is None
    view.update(camera, *measured(2, ((53,10,100,90),)))
    assert view.current is None


def test_source_change_resets_lock_and_slow_creep_is_not_stationary(settings, camera):
    view = ViewOccupancy(settings)
    for t in range(4):
        view.update(camera, *measured(t, ((20+3*t,20,70+3*t,80),)))
    assert not view.current.stationary
    origin = view.current.origin
    view.confirm(origin, 'old')
    view.update(camera, *measured(5, source='new'))
    assert view.current.origin != origin and view.current.event_id is None


def test_tracker_churn_has_one_event_then_clear_allows_next(settings, repository, snapshots, camera):
    settings.min_track_hits = 1
    settings.ocr_confirmations = 1
    runtime = GateRuntime(settings, [camera], repository, snapshots)
    runtime.detector = Mock()
    runtime.ocr = Mock()
    runtime.ocr.read.return_value = OCRRead('CSQU3054383', .99)
    def run(t, boxes=((20,20,70,80),)):
        frame, detections = measured(t, boxes)
        runtime.detector.detect.return_value = detections
        runtime.process_frame(camera, frame)
        while not runtime.queue.empty():
            job = runtime.queue.get_nowait()
            key = runtime.ocr_store.save(job, crop_identification(job.frame.image, job.detection.bbox, job.roi))
            runtime._process_saved_ocr(key)
            runtime.queue.task_done()
    run(0)
    event = runtime.occupancy[camera.id].current.event_id
    assert event
    for t in range(1, 100):
        runtime.tracker[camera.id] = TemporalTracker()  # Forced ID recycling every frame.
        run(t)
    assert repository.events()['total'] == 1
    assert runtime.ocr.read.call_count == 1
    assert runtime.occupancy[camera.id].status()['state'] == 'WAITING'
    for t in range(100,108):
        run(t, ())
    run(108)
    assert repository.events()['total'] == 2
    assert runtime.ocr.read.call_count == 2


def test_proof_prefers_sharp_sample_and_cleanup_protects_active(settings, repository, snapshots, camera, make_job):
    runtime = GateRuntime(settings, [camera], repository, snapshots)
    rng = np.random.default_rng(7)
    sharp = rng.integers(20,230,(100,150,3), dtype=np.uint8)
    first = make_job(1)
    first = replace(first, frame=replace(first.frame, image=sharp))
    second = make_job(2)
    first_id = runtime.ocr_store.save(first, sharp)
    second_id = runtime.ocr_store.save(second, np.zeros((100,150,3),dtype=np.uint8))
    assert runtime.ocr_store.best_proof(first.origin_key, second_id).frame.sequence == 1
    runtime.ocr_store.complete(first_id, 'event')
    runtime.ocr_store.complete(second_id, 'event')
    runtime.ocr_store.clean_completed(-1, (first.origin_key,))
    assert np.array_equal(runtime.ocr_store.load(first_id)[0].frame.image, sharp)


def test_new_cleared_passage_not_merged_by_short_dedup_window(repository, snapshots, make_job):
    from tests.test_events_database import observation
    first = make_job(origin='dwell:camera-1:first')
    second = make_job(origin='dwell:camera-1:second', timestamp=first.frame.timestamp + timedelta(seconds=10))
    one = repository.save_observation(observation(first, text='CSQU3054383'), snapshots, 30)[0]
    two = repository.save_observation(observation(second, text='TEMU4518441'), snapshots, 30)[0]
    assert one != two


def test_same_container_repeated_dwell_cycles_merged(repository, snapshots, make_job):
    from tests.test_events_database import observation
    first = make_job(origin='dwell:camera-1:cycle1')
    second = make_job(origin='dwell:camera-1:cycle2', timestamp=first.frame.timestamp + timedelta(seconds=25))
    one, created1 = repository.save_observation(observation(first, text='BMOU4128455'), snapshots, 300)
    two, created2 = repository.save_observation(observation(second, text='BMOU4128455'), snapshots, 300)
    assert one == two
    assert created1 is True
    assert created2 is False


def test_active_lane_association_beyond_time_window(repository, snapshots, camera, make_job):
    from tests.test_events_database import observation
    other = camera.model_copy(update={'id': 'camera-2'})
    repository.sync_cameras([camera, other])
    first = make_job(origin='dwell:camera-1:first')
    one = repository.save_observation(observation(first), snapshots, 30)[0]
    later = make_job(camera_id='camera-2', origin='dwell:camera-2:later',
                     timestamp=first.frame.timestamp + timedelta(seconds=180))
    two, created = repository.save_observation(observation(later), snapshots, 30, ongoing_event_id=one)
    assert one == two and not created


def test_confirmation_stores_sharp_first_image_not_blurred_last(settings, repository, snapshots, camera, make_job, monkeypatch):
    runtime = GateRuntime(settings, [camera], repository, snapshots)
    runtime.ocr = Mock()
    runtime.ocr.read.return_value = OCRRead('CSQU3054383', .99)
    first = make_job(1)
    image = np.random.default_rng(1).integers(20,230,first.frame.image.shape,dtype=np.uint8)
    first = replace(first, frame=replace(first.frame, image=image))
    second = make_job(2)
    first_id = runtime.ocr_store.save(first, image)
    second_id = runtime.ocr_store.save(second, second.frame.image)
    save = Mock(wraps=snapshots.save)
    monkeypatch.setattr(snapshots, 'save', save)
    runtime._process_saved_ocr(first_id)
    runtime._process_saved_ocr(second_id)
    assert np.array_equal(save.call_args.args[1].image, image)
    assert save.call_args.args[1].detection.frame_timestamp == first.detection.frame_timestamp


def test_queued_jobs_after_confirmation_skip_ocr(settings, repository, snapshots, camera, make_job):
    settings.ocr_confirmations = 1
    runtime = GateRuntime(settings, [camera], repository, snapshots)
    runtime.ocr = Mock()
    runtime.ocr.read.return_value = OCRRead('CSQU3054383', .99)
    ids = [runtime.ocr_store.save(make_job(i), np.zeros((20,20,3),dtype=np.uint8)) for i in (1,2,3)]
    for key in ids:
        runtime._process_saved_ocr(key)
    assert runtime.ocr.read.call_count == 1
    assert repository.events()['total'] == 1


def test_container_number_alone_triggers_occupancy(settings, camera):
    """Ensure that detecting container_number alone (without outer container class) successfully triggers occupancy."""
    view = ViewOccupancy(settings)
    ts = datetime(2026, 9, 11)
    frame = Frame('camera-1', ts, 1, np.zeros((720, 1280, 3), dtype=np.uint8), 'source')
    # Small vertical container number box (e.g. 10x100 pixels)
    det = Detection(frame.camera_id, ts, 'container_number', 0.85, (600, 100, 620, 250))
    selected = view.update(camera, frame, [det])
    assert selected is not None
    assert selected.class_name == 'container_number'
    assert view.current is not None
    assert view.status()['state'] in ('TRACKING', 'OCCUPIED')


def test_trailing_and_intermediate_unconfirmed_reads_do_not_duplicate_confirmed_event(settings, repository, snapshots, camera, make_job):
    """
    Ensure intermediate and trailing frames on different tracks/cameras with unconfirmed
    or missing container numbers do NOT create separate NEEDS_REVIEW rows when a confirmed
    container event exists on the lane. Exactly 1 CONFIRMED row should exist.
    """
    from tests.test_events_database import observation
    camera_2 = camera.model_copy(update={'id': 'camera-2'})
    repository.sync_cameras([camera, camera_2])

    # 1. First frame on camera-1 confirms the container
    job1 = make_job(1, origin='dwell:camera-1:passage_1:track_1')
    obs1 = observation(job1, text='PCIU2870680')
    ev_id1, created1 = repository.save_observation(obs1, snapshots, 35.0)
    assert created1 is True

    # 2. Subsequent intermediate frames on camera-2 (e.g. ISO code only or unconfirmed text)
    # on different tracks within 2 seconds of the confirmed event
    for i in range(2, 6):
        job_trail = make_job(i, camera_id='camera-2', origin=f'dwell:camera-2:passage_1:track_{i}',
                             timestamp=job1.frame.timestamp + timedelta(seconds=0.5 * (i - 1)))
        obs_trail = observation(job_trail, text='UNREADABLE')
        ev_id_trail, created_trail = repository.save_observation(obs_trail, snapshots, 35.0)
        assert created_trail is False
        assert ev_id_trail == ev_id1

    # Verify database has exactly 1 event total, which is CONFIRMED with the container number
    events = repository.events()['items']
    assert len(events) == 1
    assert events[0]['id'] == ev_id1
    assert events[0]['container_number'] == 'PCIU2870680'
    assert events[0]['status'] == 'CONFIRMED'


