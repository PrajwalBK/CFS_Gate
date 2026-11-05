from fastapi.testclient import TestClient
from sqlalchemy import text
from app.main import create_app
from app.database.database import Database
from tests.test_events_database import observation


def test_local_database_initializes_and_events_survive_restart(settings, camera, tmp_path, make_job):
    settings.deployment_mode = 'development'
    settings.camera_discovery_enabled = False
    settings.pipeline_enabled = False
    path = tmp_path / 'nested' / 'gate.db'
    from pydantic import SecretStr
    settings.database_url = SecretStr('sqlite+pysqlite:///' + path.as_posix())
    with TestClient(create_app(settings, camera_configs=[camera], camera_sources=[])) as client:
        assert path.is_file()
        assert client.get('/api/health').json()['database'] == 'READY'
        event_id, _ = client.app.state.repository.save_observation(
            observation(make_job()), client.app.state.snapshots, 30)
        with client.app.state.database.engine.connect() as connection:
            assert connection.scalar(text('PRAGMA foreign_keys')) == 1
            assert connection.scalar(text('PRAGMA journal_mode')) == 'wal'
    with TestClient(create_app(settings, camera_configs=[camera], camera_sources=[])) as client:
        assert client.get('/api/gate-events/' + event_id).status_code == 200
        assert len(client.get('/api/ocr-results').json()) == 1


def test_incompatible_schema_is_not_overwritten(settings, tmp_path):
    db = Database('sqlite+pysqlite:///' + (tmp_path / 'old.db').as_posix())
    db.initialize()
    with db.engine.begin() as connection:
        connection.execute(text('UPDATE gate_schema_version SET version=999'))
    import pytest
    with pytest.raises(RuntimeError):
        db.initialize()
    with db.engine.connect() as connection:
        assert connection.scalar(text('SELECT version FROM gate_schema_version')) == 999
    db.close()
