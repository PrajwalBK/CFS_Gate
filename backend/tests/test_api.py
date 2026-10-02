import pytest
from fastapi.testclient import TestClient
from app.main import create_app
from app.config.settings import Settings
from tests.test_events_database import observation


def test_camera_configuration_events_health_and_snapshot(settings, repository, snapshots, camera, make_job, monkeypatch):
    monkeypatch.delenv('CAMERA_1_RTSP', raising=False)
    event_id, _ = repository.save_observation(observation(make_job()), snapshots, 30)
    app = create_app(settings, repository.db, [camera])
    with TestClient(app) as client:
        cameras = client.get('/api/cameras').json()
        assert cameras[0]['status'] == 'OFFLINE'
        assert 'rtsp_url' not in cameras[0]
        assert client.get('/api/cameras/camera-1/status').status_code == 200
        assert client.get('/api/cameras/missing/status').status_code == 404
        assert not client.post('/api/cameras/camera-1/test').json()['success']
        assert client.get('/api/gate-events?container=CSQU3054383').json()['total'] == 1
        detail = client.get('/api/gate-events/' + event_id).json()
        assert detail['event_type'] == 'UNKNOWN'
        json_resp = client.get('/api/gate-events/' + event_id + '/json')
        assert json_resp.status_code == 200
        assert json_resp.headers['content-type'] == 'application/json'
        assert json_resp.json()['id'] == event_id
        assert (snapshots.root / event_id / 'event.json').is_file()
        snapshot_id = detail['snapshots'][0]['id']
        image = client.get('/api/snapshots/' + snapshot_id)
        assert image.status_code == 200 and image.headers['content-type'] == 'image/jpeg'
        proof = client.get('/api/snapshots/' + snapshot_id + '?proof=true')
        assert proof.status_code == 200 and proof.headers['content-type'] == 'image/jpeg'
        assert proof.headers['cache-control'] == 'no-cache'
        assert client.get('/api/snapshots/' + snapshot_id).content == image.content
        assert client.get('/api/ocr-results').json()[0]['raw_text'] == 'CSQU3054383'
        assert client.get('/api/gate-events/missing').status_code == 404
        assert client.get('/api/gate-events?limit=999').status_code == 422
        assert client.get('/api/gate-events?event_type=INVALID').status_code == 422
        assert client.get('/api/health').status_code == 200
        assert client.get('/api/models').json()['detector'] == 'DISABLED'
        assert client.get('/api/detections').json() == []
        assert 'cpu_percent' in client.get('/api/metrics').json()


def test_database_outage_returns_sanitized_503(settings, repository, camera, monkeypatch):
    from sqlalchemy.exc import OperationalError
    with TestClient(create_app(settings, repository.db, [camera])) as client:
        def fail(*args, **kwargs):
            raise OperationalError('mysql://user:secret@host', {}, Exception('password=secret'))
        monkeypatch.setattr(repository.db, 'check', fail)
        response = client.get('/api/health')
        assert response.status_code == 503
        assert 'secret' not in response.text
        monkeypatch.setattr(client.app.state.repository, 'events', fail)
        assert client.get('/api/gate-events').json() == {'detail': 'Database unavailable'}


def test_config_requires_persistent_sqlite_outside_tests_and_valid_geometry(camera):
    with pytest.raises(ValueError):
        Settings(_env_file=None, deployment_mode='development', database_url='sqlite://')
    data = camera.model_dump()
    data['ocr_roi'] = [1, 0, 0, 1]
    with pytest.raises(ValueError):
        type(camera).model_validate(data)


def test_compiled_frontend_is_served_by_backend(settings, repository, camera, tmp_path):
    frontend = tmp_path / 'browser'
    frontend.mkdir()
    (frontend / 'index.html').write_text('<!doctype html><title>Gate UI</title>', encoding='utf-8')
    (frontend / 'main.js').write_text('window.gate = true;', encoding='utf-8')
    with TestClient(create_app(settings, repository.db, [camera], frontend_directory=frontend)) as client:
        page = client.get('/')
        assert page.status_code == 200 and 'Gate UI' in page.text
        asset = client.get('/main.js')
        assert asset.status_code == 200 and 'javascript' in asset.headers['content-type']


def test_cloud_sync_status_and_retry_api(settings, repository, camera):
    with TestClient(create_app(settings, repository.db, [camera])) as client:
        status_resp = client.get('/api/cloud-sync/status')
        assert status_resp.status_code == 200
        data = status_resp.json()
        assert 'enabled' in data
        assert 'is_online' in data
        assert 'pending_count' in data

        retry_resp = client.post('/api/cloud-sync/retry')
        assert retry_resp.status_code == 200
        rdata = retry_resp.json()
        assert 'triggered' in rdata
        assert 'pending' in rdata

        health = client.get('/api/health').json()
        assert 'cloud_sync' in health
        assert health['cloud_sync']['pending_count'] == data['pending_count']
