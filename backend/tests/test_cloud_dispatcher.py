"""
Tests for Prosper Smart Yard Cloud Outbox Dispatcher
"""

import time
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from pydantic import SecretStr

from app.domain import Detection, Observation, Validation, utcnow
from app.events.cloud_dispatcher import CloudOutboxDispatcher, ProsperSession


@pytest.fixture
def mock_settings(settings):
    cfg = settings.model_copy(update={
        'prosper_enabled': True,
        'prosper_api_base_url': 'http://syapi.test.prosper.com',
        'prosper_site_id': 'site-uuid-1234',
        'prosper_bearer_token': SecretStr('test-bearer-token'),
        'prosper_site_code': 'SITE1',
        'prosper_email': 'test@prosper.com',
        'prosper_password': SecretStr('secret123'),
        'prosper_max_retries': 2,
        'prosper_queue_size': 10,
    })
    return cfg


def test_prosper_session_bearer_headers(mock_settings):
    session = ProsperSession(mock_settings)
    headers = session.get_auth_headers()
    assert headers.get('Authorization') == 'Bearer test-bearer-token'


def test_prosper_session_api_key_headers(mock_settings):
    cfg = mock_settings.model_copy(update={
        'prosper_bearer_token': None,
        'prosper_api_key': SecretStr('my-api-key-999'),
    })
    session = ProsperSession(cfg)
    headers = session.get_auth_headers()
    assert headers.get('x-api-key') == 'my-api-key-999'


def test_prosper_session_auto_login(mock_settings):
    cfg = mock_settings.model_copy(update={
        'prosper_bearer_token': None,
        'prosper_api_key': None,
    })
    session = ProsperSession(cfg)

    with patch.object(session.session, 'post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {'token': 'auto-login-jwt-token', 'expiresIn': 3600}

        headers = session.get_auth_headers()
        assert headers.get('Authorization') == 'Bearer auto-login-jwt-token'
        assert mock_post.called
        assert '/api/auth/login' in mock_post.call_args[0][0]


def test_prosper_session_upload_image(mock_settings):
    session = ProsperSession(mock_settings)
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch.object(session.session, 'post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {'url': 'https://s3.prosper.com/trailers/abc.jpg'}

        url = session.upload_image(dummy_img)
        assert url == 'https://s3.prosper.com/trailers/abc.jpg'
        assert mock_post.called
        assert '/api/sites/site-uuid-1234/images/trailers' in mock_post.call_args[0][0]


def test_prosper_session_post_gate_event(mock_settings):
    session = ProsperSession(mock_settings)
    payload = {
        'id': 'event-1',
        'gateId': 'gate-lane-1',
        'timestamp': '2026-09-09T12:00:00.000Z',
        'containerNumber': 'MSKU1234567',
        'eventType': 'ENTRY',
        'deviceId': 'device-uuid-1',
        'confidence': 0.95,
        'imageUrl': 'https://s3.prosper.com/trailers/abc.jpg',
    }

    with patch.object(session.session, 'post') as mock_post:
        mock_post.return_value.status_code = 200
        ok = session.post_gate_event(payload)
        assert ok is True
        assert '/api/sites/site-uuid-1234/gate-events' in mock_post.call_args[0][0]


def test_cloud_dispatcher_disabled_does_nothing(settings):
    cfg = settings.model_copy(update={'prosper_enabled': False})
    dispatcher = CloudOutboxDispatcher(cfg)
    dispatcher.start()
    assert dispatcher._worker_thread is None

    # Enqueue should safely no-op
    dummy_obs = MagicMock(confirmed=True)
    dispatcher.enqueue(dummy_obs, 'event-1')
    assert dispatcher.queue.empty()
    dispatcher.stop()


def test_cloud_dispatcher_enqueue_and_sync(mock_settings):
    metrics = MagicMock()
    dispatcher = CloudOutboxDispatcher(mock_settings, metrics=metrics)

    dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
    det = Detection(
        camera_id='gate-1',
        frame_timestamp=utcnow(),
        class_name='container',
        confidence=0.95,
        bbox=(10.0, 10.0, 50.0, 50.0),
    )
    val = Validation(raw_text='MSKU 123456-7', normalized_text='MSKU1234567',
                     confidence=0.95, valid_format=True, valid_check_digit=True,
                     validation_status='VALID')
    obs = Observation('origin-1', 'gate-lane-1', 1, det, val, 'ENTRY', dummy_img, utcnow(), confirmed=True)

    with patch.object(dispatcher.session, 'upload_image', return_value='https://img.test/1.jpg') as mock_up, \
         patch.object(dispatcher.session, 'post_gate_event', return_value=True) as mock_post:

        dispatcher.start()
        dispatcher.enqueue(obs, 'evt-uuid-123')

        # Wait briefly for background worker
        for _ in range(20):
            if dispatcher.queue.empty() and mock_post.called:
                break
            time.sleep(0.05)

        dispatcher.stop()

        assert mock_up.called
        assert mock_post.called
        call_payload = mock_post.call_args[0][0]
        assert call_payload['id'] == 'evt-uuid-123'
        assert call_payload['containerNumber'] == 'MSKU1234567'
        assert call_payload['eventType'] == 'ENTRY'
        assert call_payload['imageUrl'] == 'https://img.test/1.jpg'
        assert metrics.increment.called


def test_syapi_gate_event_schema(mock_settings):
    dispatcher = CloudOutboxDispatcher(mock_settings)
    dummy_img = np.zeros((32, 32, 3), dtype=np.uint8)
    det = Detection(
        camera_id='CAM-01',
        frame_timestamp=utcnow(),
        class_name='container',
        confidence=0.96,
        bbox=(5.0, 5.0, 25.0, 25.0),
    )
    val = Validation(raw_text='MSCU 111111 1', normalized_text='MSCU1111111',
                     confidence=0.96, valid_format=True, valid_check_digit=True,
                     validation_status='VALID', size_type='22G1')
    obs = Observation('origin-syapi', 'LANE-1', 10, det, val, 'ENTRY', dummy_img, utcnow(), confirmed=True)

    with patch.object(dispatcher.session, 'post_gate_event', return_value=True) as mock_post:
        dispatcher.start()
        dispatcher.enqueue(obs, 'event-uuid-888')

        for _ in range(20):
            if dispatcher.queue.empty() and mock_post.called:
                break
            time.sleep(0.05)

        dispatcher.stop()

        assert mock_post.called
        payload = mock_post.call_args[0][0]
        assert 'VisitId' in payload
        assert payload['EventType'] == 'GATE_IN'
        assert payload['DeviceId'] == 'EDGE-GATE-01'
        assert payload['Truck'] == {'TruckNumber': 'NA'}
        assert payload['Driver'] is None
        assert len(payload['Containers']) == 1
        container = payload['Containers'][0]
        assert container['ContainerNumber'] == 'MSCU1111111'
        assert container['ContainerNumberConfidence'] == 0.96
        assert container['Size'] == '20'
        assert container['Position'] == 'SINGLE'
        assert len(container['Images']) >= 1
        assert container['Images'][0]['CameraId'] == 'CAM-01'
        assert container['Images'][0]['Image'].startswith('data:image/jpeg;base64,')
