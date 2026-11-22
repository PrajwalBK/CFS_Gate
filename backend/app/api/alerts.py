import time
import threading
from uuid import uuid4
from datetime import datetime, timezone
from typing import Optional, List
from pydantic import BaseModel
from fastapi import APIRouter, Request

router = APIRouter(prefix='/api/alerts', tags=['alerts'])


def trigger_hardware_buzzer(pattern: str = 'alert'):
    """Trigger audible alert buzzer via host hardware speaker (Windows winsound)."""
    def _beep():
        try:
            import winsound
            if pattern in ('gate_out', 'exit'):
                winsound.Beep(950, 220)
                time.sleep(0.08)
                winsound.Beep(1200, 320)
            elif pattern in ('cloud_error', 'conflict'):
                winsound.Beep(600, 300)
                time.sleep(0.08)
                winsound.Beep(600, 300)
            else:
                winsound.Beep(1000, 250)
        except Exception:
            pass

    threading.Thread(target=_beep, daemon=True).start()


class AlertModel(BaseModel):
    id: str
    alert_type: str
    level: str
    title: str
    message: str
    container_number: Optional[str] = None
    event_id: Optional[str] = None
    timestamp: str


@router.get('', response_model=List[AlertModel])
def get_alerts(request: Request):
    """Return active recent alerts for dashboard notification and buzzer."""
    runtime = getattr(request.app.state, 'runtime', None)
    if not runtime or not hasattr(runtime, 'alerts'):
        return []
    with runtime._state_lock:
        return list(runtime.alerts)


@router.post('/test')
def test_buzzer(pattern: str = 'gate_out'):
    """Trigger a test hardware buzzer sound on the host machine."""
    trigger_hardware_buzzer(pattern)
    return {'status': 'triggered', 'pattern': pattern}


@router.post('/clear')
def clear_alerts(request: Request):
    """Clear all active alerts."""
    runtime = getattr(request.app.state, 'runtime', None)
    if runtime and hasattr(runtime, 'alerts'):
        with runtime._state_lock:
            runtime.alerts.clear()
    return {'status': 'cleared'}


# =========================================================================
# OUT-GATE ALERT SYSTEM & USB RELAY SIREN ENDPOINTS
# =========================================================================

class ToggleAlertPayload(BaseModel):
    enabled: bool

class TestSirenPayload(BaseModel):
    duration_seconds: float = 8.0

@router.get('/out-gate/status')
def get_out_gate_alert_status():
    """Return diagnostic status of Out-Gate Alert system, USB relay connection, and execution logs."""
    from app.alerts.alert_manager import get_alert_manager
    return get_alert_manager().get_status()

@router.post('/out-gate/toggle')
def toggle_out_gate_alert(payload: ToggleAlertPayload):
    """Enable or disable the Out-Gate Alert system persistently."""
    from app.alerts.alert_manager import get_alert_manager
    mgr = get_alert_manager()
    mgr.set_master_switch(payload.enabled)
    return mgr.get_status()

@router.post('/out-gate/test')
def test_out_gate_siren(payload: TestSirenPayload):
    """Trigger manual test of the external siren via USB relay for specified duration (default 8s)."""
    from app.alerts.alert_manager import get_alert_manager
    mgr = get_alert_manager()
    success = mgr.trigger_test_siren(payload.duration_seconds)
    return {'status': 'triggered' if success else 'failed', 'duration': payload.duration_seconds}

@router.post('/out-gate/siren-off')
def stop_out_gate_siren():
    """Immediately command USB relay OFF and silence the external siren."""
    from app.alerts.alert_manager import get_alert_manager
    mgr = get_alert_manager()
    mgr.stop_siren()
    return {'status': 'stopped'}

@router.get('/out-gate/logs')
def get_out_gate_alert_logs():
    """Return full historical log of Out-Gate alert evaluations and siren triggers."""
    from app.alerts.alert_manager import get_alert_manager
    return get_alert_manager()._logs

