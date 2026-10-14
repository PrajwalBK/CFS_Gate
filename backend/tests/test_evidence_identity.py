"""Adversarial tests for camera, passage and image identity boundaries."""
from dataclasses import replace
from datetime import timedelta
import hashlib
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pydantic import SecretStr
from app.events.cloud_dispatcher import CloudOutboxDispatcher
from app.runtime import GateRuntime
from app.database.models import GateEvent
from unittest.mock import Mock

import cv2
import numpy as np

from app.domain import Detection, OCRRead
from app.config.settings import CameraConfig
from app.events.passage_session import ContainerPassageSession, PassageSessionManager
from app.events.passage_outbox import PassageOutbox
from app.ocr.validator import ContainerValidator


def valid(number):
    return ContainerValidator().validate(OCRRead(number, .98))


def test_camera_local_tracks_do_not_split_one_passage(settings, make_job):
    job = make_job()
    manager = PassageSessionManager(settings)
    a = manager.get_or_create_session(job.gate_id, 'ENTRY', job.frame.timestamp,
        ['front', 'side'], track_id=101, camera_id='front', source_id='stream-a')
    a.add_ocr_candidate(valid('MSKU1234565'))
    b = manager.get_or_create_session(job.gate_id, 'ENTRY', job.frame.timestamp + timedelta(seconds=1.6),
        ['front', 'side'], track_id=7, camera_id='side', source_id='stream-b')
    assert a is b
    assert manager.get_active_session(job.gate_id, 'CONTAINER_2') is None


def test_restart_routes_delayed_ocr_to_original_event(settings, tmp_path, make_job):
    job = make_job()
    outbox = PassageOutbox(tmp_path)
    old = ContainerPassageSession('old', job.gate_id, 'ENTRY', [job.frame.camera_id], job.frame.timestamp)
    old.finalized = True
    outbox.save(old)
    restarted = PassageSessionManager(settings, outbox=PassageOutbox(tmp_path))
    current = restarted.get_or_create_session(job.gate_id, 'ENTRY', job.frame.timestamp,
        [job.frame.camera_id])
    restarted.record_ocr_result(job.gate_id, valid('MSKU1234565'), event_id=old.event_id)
    assert current.get_consensus_container_number()[0] is None
    assert outbox.get(old.event_id).get_consensus_container_number()[0] == 'MSKU1234565'


def test_distinct_valid_identities_require_review(make_job):
    job = make_job()
    session = ContainerPassageSession('conflict', job.gate_id, 'ENTRY', [], job.frame.timestamp)
    session.add_ocr_candidate(valid('MSKU1234565'))
    session.add_ocr_candidate(valid('MSCU1234566'))
    assert session.identity_conflict
    assert session.get_consensus_container_number()[0] is None


def test_review_record_cannot_suppress_later_confirmed_delivery(repository, make_job):
    job = make_job()
    with repository.db.sessions.begin() as db:
        for event_id, status, seconds in [('review', 'NEEDS_REVIEW', 0), ('good', 'CONFIRMED', 1)]:
            db.add(GateEvent(id=event_id, gate_id=job.gate_id, timestamp=job.frame.timestamp+timedelta(seconds=seconds),
                container_number='MSKU1234565', status=status, event_type='ENTRY'))
    assert repository.canonical_event_id('good') == 'good'


def test_similar_checksum_valid_identities_are_not_cloud_duplicates(repository, make_job):
    job = make_job()
    with repository.db.sessions.begin() as db:
        for event_id, number, seconds in [('first', 'MSKU1234565', 0), ('second', 'MSCU1234566', 1)]:
            db.add(GateEvent(id=event_id, gate_id=job.gate_id, timestamp=job.frame.timestamp+timedelta(seconds=seconds),
                container_number=number, status='CONFIRMED', event_type='ENTRY'))
    assert repository.canonical_event_id('second') == 'second'


def test_low_confidence_checksum_valid_read_remains_unconfirmed(make_job):
    job = make_job()
    session = ContainerPassageSession('uncertain', job.gate_id, 'ENTRY', [], job.frame.timestamp)
    session.add_ocr_candidate(replace(valid('MSKU1234565'), confidence=.2))
    assert session.get_consensus_container_number()[0] is None


def test_wrong_camera_stale_frame_and_text_only_are_rejected(make_job):
    job = make_job()
    session = ContainerPassageSession('frame', job.gate_id, 'ENTRY', [], job.frame.timestamp)
    for detection in (replace(job.detection, camera_id='other'),
            replace(job.detection, frame_timestamp=job.frame.timestamp - timedelta(seconds=1)),
            replace(job.detection, class_name='container_number')):
        session.add_frame(job.frame.camera_id, job.frame.image, job.frame.timestamp, [detection])
    assert session.get_best_frame_per_camera() == {}


def test_targeted_ocr_does_not_read_adjacent_container(settings, repository, snapshots, camera, make_job):
    runtime = GateRuntime(settings, [camera], repository, snapshots)
    job = make_job()
    image = np.full((480, 640, 3), 20, dtype=np.uint8)
    image[:, 301:] = 220
    frame = replace(job.frame, image=image)
    body = replace(job.detection, bbox=(30, 30, 300, 450))
    own = replace(body, class_name='container_number', bbox=(70, 100, 150, 130), confidence=.90)
    neighbor = replace(body, class_name='container_number', bbox=(310, 100, 390, 130), confidence=.99)
    runtime.ocr = Mock()
    runtime.ocr.read.side_effect = lambda crop: OCRRead('MSCU1234566' if crop.mean() > 100 else 'MSKU1234565', .98)
    _, result = runtime._execute_ocr(replace(job, frame=frame, detection=body, all_detections=(body, own, neighbor)))
    assert result.normalized_text == 'MSKU1234565'


def test_verified_ocr_frame_wins_over_sharper_later_frame_and_survives_restart(tmp_path, make_job):
    job = make_job()
    session = ContainerPassageSession('readable-proof', job.gate_id, 'ENTRY', [job.frame.camera_id],
        job.frame.timestamp, target_frames_per_camera=1)
    noise = np.random.default_rng(2).integers(30, 220, (480, 640, 3), dtype=np.uint8)
    readable = cv2.GaussianBlur(noise, (5, 5), 0)
    detection = replace(job.detection, bbox=(40, 40, 600, 440))
    first = session.add_frame(job.frame.camera_id, readable, job.frame.timestamp, [detection],
        valid('MSKU1234565'), source_id='stream', frame_sequence=10)
    later_time = job.frame.timestamp + timedelta(seconds=1)
    session.add_frame(job.frame.camera_id, noise, later_time,
        [replace(detection, frame_timestamp=later_time)], source_id='stream', frame_sequence=11)
    assert session.get_best_frame_per_camera()[job.frame.camera_id] is first
    outbox = PassageOutbox(tmp_path)
    outbox.save(session)
    restored = outbox.get(session.event_id).get_best_frame_per_camera()[job.frame.camera_id]
    assert restored.frame_sequence == 10
    assert restored.ocr_read.normalized_text == 'MSKU1234565'
    np.testing.assert_array_equal(restored.image, readable)


def test_manifest_binds_best_camera_bytes_and_rejects_tampering(repository, snapshots, make_job):
    job = make_job()
    session = ContainerPassageSession('proof', job.gate_id, 'ENTRY', ['front', 'side'], job.frame.timestamp)
    repository.sync_cameras([CameraConfig(id=c, name=c, gate_id=job.gate_id, source_env=c.upper()) for c in session.cameras])
    originals = {}
    for i, camera in enumerate(session.cameras):
        image = np.random.default_rng(i).integers(30, 220, (480, 640, 3), dtype=np.uint8)
        detection = Detection(camera, job.frame.timestamp, 'container', .96, (40, 40, 600, 440))
        session.add_frame(camera, image, job.frame.timestamp, [detection], source_id=f'source-{i}', frame_sequence=10+i)
        originals[camera] = image
    session.add_ocr_candidate(valid('MSKU1234565'))
    repository.finalize_passage_session(session, snapshots)
    event = repository.event(session.event_id)
    manifest = snapshots.build_evidence_manifest(event)
    assert {s['camera_id'] for s in manifest['captures']} == {'front', 'side'}
    for snapshot in manifest['captures']:
        data = snapshots.path(snapshot['image_path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == snapshot['provenance']['image_sha256']
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        expected = cv2.imdecode(cv2.imencode('.jpg', originals[snapshot['camera_id']],
            [cv2.IMWRITE_JPEG_QUALITY, 90])[1], cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(image, expected)
    front = next(s for s in event['snapshots'] if s['camera_id'] == 'front')
    snapshots.path(front['image_path']).write_bytes(snapshots.path(next(s for s in event['snapshots'] if s['camera_id'] == 'side')['image_path']).read_bytes())
    assert all(s['camera_id'] != 'front' for s in snapshots.build_evidence_manifest(event)['captures'])

    snapshots.path(front['image_path']).unlink()
    assert all(s['camera_id'] != 'front' for s in snapshots.build_evidence_manifest(event)['captures'])


def test_real_http_payload_preserves_event_camera_and_image_bytes(settings, repository, snapshots, make_job):
    received = []
    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"eventId":"accepted-local"}')
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Receiver)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    job = make_job()
    session = ContainerPassageSession('http-proof', job.gate_id, 'ENTRY', [job.frame.camera_id], job.frame.timestamp)
    image = np.random.default_rng(4).integers(30, 220, (480, 640, 3), dtype=np.uint8)
    detection = replace(job.detection, bbox=(40, 40, 600, 440))
    session.add_frame(job.frame.camera_id, image, job.frame.timestamp, [detection])
    session.add_ocr_candidate(replace(valid('MSKU1234565'), size_type='22G1'))
    repository.finalize_passage_session(session, snapshots)
    cfg = settings.model_copy(update={'prosper_enabled': True,
        'prosper_api_base_url': f'http://127.0.0.1:{server.server_port}',
        'prosper_bearer_token': SecretStr('local-test-token'), 'prosper_max_retries': 1})
    dispatcher = CloudOutboxDispatcher(cfg, repository=repository, snapshots=snapshots)
    try:
        dispatcher.start()
        dispatcher.enqueue(None, session.event_id)
        dispatcher.queue.join()
        assert len(received) == 1
        container = received[0]['Containers'][0]
        assert container['ContainerNumber'] == 'MSKU1234565'
        assert container['Size'] == '20'
        assert container['ContainerNumberConfidence'] == .98
        proof = container['Images'][0]
        assert proof['CameraId'] == job.frame.camera_id
        raw = base64.b64decode(proof['Image'].split(',', 1)[1])
        event = repository.event(session.event_id)
        assert raw == snapshots.path(event['snapshots'][0]['image_path']).read_bytes()
        assert event['cloud_delivery_state'] == 'DELIVERED'
    finally:
        dispatcher.stop()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
