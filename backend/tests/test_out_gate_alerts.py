import os
import sys
import time
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.alerts.alert_config import AlertConfig
from app.alerts.usb_relay_controller import USBRelayController
from app.alerts.alert_manager import OutGateAlertManager, parse_container_pass_status

@pytest.fixture
def temp_alert_config(tmp_path):
    cfg_file = tmp_path / "alert_config.json"
    return AlertConfig(config_file=cfg_file)

@pytest.fixture
def mock_relay():
    relay = MagicMock(spec=USBRelayController)
    relay.is_connected.return_value = True
    relay.turn_on.return_value = True
    relay.turn_off.return_value = True
    relay.get_status.return_value = {
        'connected': True,
        'relay_state': 'OFF',
        'device_type': 'SERIAL',
        'device_description': 'Mock Relay',
        'port': 'COM3',
        'last_error': None
    }
    return relay

@pytest.fixture
def alert_mgr(temp_alert_config, mock_relay, tmp_path):
    mgr = OutGateAlertManager(config=temp_alert_config, relay=mock_relay)
    mgr._history_file = tmp_path / "alert_history.json"
    return mgr

# ---------------------------------------------------------------------------
# FUNCTIONAL TEST CASES (Section 11)
# ---------------------------------------------------------------------------

def test_1_alerts_on_and_pass_generated_siren_off(alert_mgr, mock_relay):
    """Case 1: Alerts ON + PASS GENERATED -> Siren OFF."""
    alert_mgr.config.set_enabled(True)
    res = alert_mgr.handle_gate_api_response(
        event_id="EVT_001",
        event_type="GATE_OUT",
        container_number="MSCU1234567",
        api_result={"passGenerated": True, "eventId": "cloud-1"}
    )
    assert res['backend_pass_status'] == "PASS GENERATED"
    assert res['final_status'] == "NO ALERT"
    mock_relay.turn_on.assert_not_called()

def test_2_alerts_on_and_pass_not_generated_siren_on_then_off(alert_mgr, mock_relay):
    """Case 2: Alerts ON + PASS NOT GENERATED -> Siren ON for 8s, then OFF."""
    alert_mgr.config.set_enabled(True)
    alert_mgr.config.siren_duration_seconds = 0.2  # Fast test duration

    res = alert_mgr.handle_gate_api_response(
        event_id="EVT_002",
        event_type="GATE_OUT",
        container_number="TGCU2031571",
        api_result={"passGenerated": False, "passStatus": "NOT_GENERATED"}
    )
    assert res['backend_pass_status'] == "PASS NOT GENERATED"
    assert res['relay_status'] == "SUCCESS"
    assert res['final_status'] == "COMPLETED"
    mock_relay.turn_on.assert_called_once()

    # Wait for deactivation timer
    time.sleep(0.3)
    mock_relay.turn_off.assert_called()

def test_3_alerts_off_and_pass_not_generated_siren_off(alert_mgr, mock_relay):
    """Case 3: Alerts OFF + PASS NOT GENERATED -> Siren OFF."""
    alert_mgr.config.set_enabled(False)

    res = alert_mgr.handle_gate_api_response(
        event_id="EVT_003",
        event_type="GATE_OUT",
        container_number="SIKU6009670",
        api_result={"passGenerated": False}
    )
    assert res['backend_pass_status'] == "PASS NOT GENERATED"
    assert res['final_status'] == "ALERT DISABLED"
    mock_relay.turn_on.assert_not_called()

def test_4_alerts_off_and_pass_generated_siren_off(alert_mgr, mock_relay):
    """Case 4: Alerts OFF + PASS GENERATED -> Siren OFF."""
    alert_mgr.config.set_enabled(False)

    res = alert_mgr.handle_gate_api_response(
        event_id="EVT_004",
        event_type="GATE_OUT",
        container_number="NPCU1260107",
        api_result={"passStatus": "GENERATED"}
    )
    assert res['backend_pass_status'] == "PASS GENERATED"
    assert res['final_status'] == "NO ALERT"
    mock_relay.turn_on.assert_not_called()

def test_5_gate_in_and_pass_not_generated_no_siren(alert_mgr, mock_relay):
    """Case 5: Gate-In + PASS NOT GENERATED -> No siren activation."""
    alert_mgr.config.set_enabled(True)

    res = alert_mgr.handle_gate_api_response(
        event_id="EVT_005",
        event_type="GATE_IN",
        container_number="MSCU9999999",
        api_result={"passGenerated": False}
    )
    assert res['final_status'] == "NO ALERT (GATE_IN)"
    mock_relay.turn_on.assert_not_called()

def test_6_api_timeout_no_siren(alert_mgr, mock_relay):
    """Case 6: API timeout -> No siren activation."""
    alert_mgr.config.set_enabled(True)

    res = alert_mgr.handle_gate_api_response(
        event_id="EVT_006",
        event_type="GATE_OUT",
        container_number="DFSU2004225",
        api_result="OFFLINE",
        raw_response={'status_code': 0, 'text': 'Connection timed out'}
    )
    assert res['backend_pass_status'] == "UNKNOWN / TIMEOUT"
    assert res['final_status'] == "NO ALERT (STATUS UNKNOWN)"
    mock_relay.turn_on.assert_not_called()

def test_7_api_network_failure_no_siren(alert_mgr, mock_relay):
    """Case 7: API network failure (500/503/offline) -> No siren activation."""
    alert_mgr.config.set_enabled(True)

    res = alert_mgr.handle_gate_api_response(
        event_id="EVT_007",
        event_type="GATE_OUT",
        container_number="DFSU2004225",
        api_result="SERVER_ERROR",
        raw_response={'status_code': 503, 'text': 'Service Unavailable'}
    )
    assert res['final_status'] == "NO ALERT (STATUS UNKNOWN)"
    mock_relay.turn_on.assert_not_called()

def test_8_usb_relay_disconnected_graceful_handling(alert_mgr, mock_relay):
    """Case 8: USB relay disconnected -> Application continues running, records disconnected status."""
    alert_mgr.config.set_enabled(True)
    mock_relay.is_connected.return_value = False
    mock_relay.turn_on.return_value = False

    res = alert_mgr.handle_gate_api_response(
        event_id="EVT_008",
        event_type="GATE_OUT",
        container_number="DFSU2004225",
        api_result={"passGenerated": False}
    )
    assert res['hardware_status'] == "Disconnected"
    assert res['final_status'] == "HARDWARE_DISCONNECTED"
    assert res['relay_status'] == "DISCONNECTED"

def test_9_usb_relay_reconnects_detection(alert_mgr, mock_relay):
    """Case 9: USB relay reconnects -> Controller detects availability."""
    mock_relay.is_connected.return_value = False
    mock_relay.get_status.return_value = {'connected': False, 'relay_state': 'DISCONNECTED'}
    assert alert_mgr.get_status()['usb_relay_connected'] is False

    mock_relay.is_connected.return_value = True
    mock_relay.get_status.return_value = {'connected': True, 'relay_state': 'OFF'}
    assert alert_mgr.get_status()['usb_relay_connected'] is True

def test_10_repeated_api_response_no_duplicate_alert(alert_mgr, mock_relay):
    """Case 10: Repeated API response for the same event -> No duplicate alert."""
    alert_mgr.config.set_enabled(True)
    alert_mgr.config.siren_duration_seconds = 5.0

    # First call triggers siren
    res1 = alert_mgr.handle_gate_api_response(
        event_id="EVT_010",
        event_type="GATE_OUT",
        container_number="TGCU2031571",
        api_result={"passGenerated": False}
    )
    assert res1['relay_status'] == "SUCCESS"
    assert mock_relay.turn_on.call_count == 1

    # Second call with same event_id: must be ignored
    res2 = alert_mgr.handle_gate_api_response(
        event_id="EVT_010",
        event_type="GATE_OUT",
        container_number="TGCU2031571",
        api_result={"passGenerated": False}
    )
    assert res2['final_status'] == "DUPLICATE"
    assert mock_relay.turn_on.call_count == 1  # Still 1, not duplicated!

def test_11_multiple_different_gate_out_events(alert_mgr, mock_relay):
    """Case 11: Multiple different Gate-Out events -> Correct alert handling."""
    alert_mgr.config.set_enabled(True)

    # Event A has pass generated -> OFF
    res_a = alert_mgr.handle_gate_api_response("EVT_A", "GATE_OUT", "CNTR_A", {"passGenerated": True})
    assert res_a['final_status'] == "NO ALERT"
    assert mock_relay.turn_on.call_count == 0

    # Event B has pass not generated -> ON
    res_b = alert_mgr.handle_gate_api_response("EVT_B", "GATE_OUT", "CNTR_B", {"passGenerated": False})
    assert res_b['final_status'] == "COMPLETED"
    assert mock_relay.turn_on.call_count == 1

def test_12_application_restart_restores_saved_setting(tmp_path):
    """Case 12: Application restart -> Previously saved Alerts setting is restored."""
    cfg_file = tmp_path / "persistent_alert_config.json"

    # Set to False and save
    cfg1 = AlertConfig(config_file=cfg_file)
    cfg1.set_enabled(False)
    assert cfg1.is_enabled() is False

    # Simulate restart by loading fresh instance from same file
    cfg2 = AlertConfig(config_file=cfg_file)
    assert cfg2.is_enabled() is False

    # Set to True and restart again
    cfg2.set_enabled(True)
    cfg3 = AlertConfig(config_file=cfg_file)
    assert cfg3.is_enabled() is True

def test_13_alerts_switched_off_while_idle(alert_mgr, mock_relay):
    """Case 13: Alerts switched OFF while idle -> No future alert activation."""
    alert_mgr.set_master_switch(False)
    assert alert_mgr.config.is_enabled() is False

    res = alert_mgr.handle_gate_api_response("EVT_013", "GATE_OUT", "CNTR_013", {"passGenerated": False})
    assert res['final_status'] == "ALERT DISABLED"
    mock_relay.turn_on.assert_not_called()

def test_14_alerts_switched_off_during_siren_activation(alert_mgr, mock_relay):
    """Case 14: Alerts switched OFF during siren activation -> Relay commanded OFF immediately."""
    alert_mgr.config.set_enabled(True)
    alert_mgr.config.siren_duration_seconds = 10.0

    # Trigger siren
    alert_mgr.handle_gate_api_response("EVT_014", "GATE_OUT", "CNTR_014", {"passGenerated": False})
    assert mock_relay.turn_on.called

    # User toggles master switch to OFF
    alert_mgr.set_master_switch(False)
    mock_relay.turn_off.assert_called()

def test_15_application_shutdown_relay_off_safety(mock_relay):
    """Case 15: Safe shutdown commands relay OFF."""
    controller = USBRelayController()
    controller._serial_conn = MagicMock()
    controller._connected = True
    controller.safe_shutdown()
    assert controller._relay_state == "OFF"

# ---------------------------------------------------------------------------
# API PAYLOAD EXTRACTION TESTS
# ---------------------------------------------------------------------------

def test_parse_container_pass_status_nested_containers():
    """Verify parsing when pass status is nested inside containers list."""
    payload_ok = {
        "eventId": "evt-123",
        "Containers": [
            {"ContainerNumber": "MSCU1234567", "passGenerated": True}
        ]
    }
    assert parse_container_pass_status(payload_ok) == "PASS_GENERATED"

    payload_bad = {
        "eventId": "evt-123",
        "Containers": [
            {"ContainerNumber": "MSCU1234567", "passGenerated": False}
        ]
    }
    assert parse_container_pass_status(payload_bad) == "PASS_NOT_GENERATED"

def test_parse_container_pass_status_explicit_error_text():
    """Verify parsing when backend returns HTTP 400/409 with error message."""
    raw_resp = {
        "status_code": 409,
        "text": "Conflict: Container pass not generated for MSCU1234567."
    }
    assert parse_container_pass_status(None, raw_resp) == "PASS_NOT_GENERATED"
