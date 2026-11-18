import pytest
from unittest.mock import MagicMock
from app.camera.lens_control import DahuaLensClient
from app.api.cameras import NightClaritySetting


def test_night_clarity_setting_schema():
    setting = NightClaritySetting()
    assert setting.day_night_mode == 'Auto'
    assert setting.hlc is True
    assert setting.max_shutter_ms == 8.0
    assert setting.gain_max == 85


def test_dahua_lens_client_get_night_status():
    client = DahuaLensClient('192.168.1.204')
    client._ensure_connected = MagicMock(return_value=True)
    client.session = 'mock-session'
    
    def mock_rpc(path, payload, timeout=3.0):
        name = payload.get('params', {}).get('name', '')
        if 'VideoInDayNight' in name:
            return {'result': True, 'params': {'table': [{'Mode': 'Auto'}]}}
        elif 'VideoInBacklight' in name:
            return {'result': True, 'params': {'table': [{'Mode': 'GlareInhibition', 'GlareInhibition': 60}]}}
        elif 'VideoInExposure' in name:
            return {'result': True, 'params': {'table': [{'Mode': 8, 'Value2': 8.0, 'GainMax': 85}]}}
        elif 'Lighting' in name:
            return {'result': True, 'params': {'table': [{'Mode': 'Auto'}]}}
        return {'result': True}

    client._rpc = MagicMock(side_effect=mock_rpc)
    status = client.get_night_status()
    assert status['supported'] is True
    assert status['day_night_mode'] == 'Auto'
    assert status['backlight_mode'] == 'GlareInhibition'
    assert status['glare_inhibition'] == 60
    assert status['exposure_mode'] == 8
    assert status['shutter_ms'] == 8.0
    assert status['gain_max'] == 85
    assert status['lighting_mode'] == 'Auto'


def test_dahua_lens_client_configure_night_clarity():
    client = DahuaLensClient('192.168.1.204')
    client._ensure_connected = MagicMock(return_value=True)
    client.session = 'mock-session'

    set_calls = []
    def mock_rpc(path, payload, timeout=3.0):
        method = payload.get('method', '')
        if method == 'configManager.setConfig':
            set_calls.append(payload.get('params', {}).get('name'))
            return {'result': True}
        return {'result': True, 'params': {'table': [{}]}}

    client._rpc = MagicMock(side_effect=mock_rpc)
    res = client.configure_night_clarity(daynight='Auto', hlc=True, max_shutter_ms=8.0, gain_max=85)
    assert res['success'] is True
    assert res['day_night'] == 'Auto'
    assert res['hlc'] is True
    assert 'VideoInDayNight[0]' in set_calls
    assert 'VideoInBacklight[0]' in set_calls
    assert 'VideoInExposure[0]' in set_calls
    assert 'Lighting[0]' in set_calls
