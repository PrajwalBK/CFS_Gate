import json
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger('gate.alerts.config')

class AlertConfig:
    """
    Persistent configuration manager for Out-Gate Alert System.
    Stores and restores settings across application restarts.
    """

    def __init__(self, config_file: Optional[Path] = None):
        if config_file is None:
            # Default to backend/data/alert_config.json
            base_dir = Path(__file__).resolve().parents[2]
            config_file = base_dir / "data" / "alert_config.json"
        self.config_file = config_file
        self.enabled: bool = True
        self.siren_duration_seconds: float = 8.0
        self.relay_port: Optional[str] = None
        self.relay_type: str = "auto"  # auto, serial_lctech, hid
        self.load()

    def load(self) -> bool:
        """Load configuration from JSON file. Defaults to enabled=True if not exists."""
        if not self.config_file.exists():
            self.save()
            return True
        try:
            data = json.loads(self.config_file.read_text(encoding='utf-8'))
            self.enabled = bool(data.get('enabled', True))
            self.siren_duration_seconds = float(data.get('siren_duration_seconds', 8.0))
            self.relay_port = data.get('relay_port')
            self.relay_type = data.get('relay_type', 'auto')
            return True
        except Exception as e:
            log.warning('failed_to_load_alert_config: %s', e)
            return False

    def save(self) -> bool:
        """Save configuration persistently to disk."""
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'enabled': self.enabled,
                'siren_duration_seconds': self.siren_duration_seconds,
                'relay_port': self.relay_port,
                'relay_type': self.relay_type
            }
            tmp = self.config_file.with_suffix('.tmp')
            tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
            tmp.replace(self.config_file)
            return True
        except Exception as e:
            log.warning('failed_to_save_alert_config: %s', e)
            return False

    def set_enabled(self, enabled: bool) -> bool:
        self.enabled = bool(enabled)
        return self.save()

    def is_enabled(self) -> bool:
        return self.enabled

_global_config: Optional[AlertConfig] = None

def get_alert_config() -> AlertConfig:
    global _global_config
    if _global_config is None:
        _global_config = AlertConfig()
    return _global_config
