import atexit
import logging
import threading
import time
from typing import Optional, Dict, Any

log = logging.getLogger('gate.alerts.relay')

# NOYITO / LC Technology Serial Protocol
# Baud rate: 9600, Data: 8, Parity: None, Stop: 1
LC_RELAY_ON = bytes([0xA0, 0x01, 0x01, 0xA2])
LC_RELAY_OFF = bytes([0xA0, 0x01, 0x00, 0xA1])

# NOYITO / DCT-Tech HID Protocol (VID 0x16C0, PID 0x05DF)
HID_RELAY_ON = bytes([0x00, 0xFF, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
HID_RELAY_OFF = bytes([0x00, 0xFD, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])

class USBRelayController:
    """
    Hardware Controller for NOYITO USB 1-Channel Relay Module.
    Supports both Serial UART (CH340/PL2303/FTDI) and USB-HID interfaces.
    Thread-safe, non-blocking, with safe shutdown and robust error recovery.
    """

    def __init__(self, port: Optional[str] = None):
        self.port = port
        self.device_type: str = "NONE"  # "SERIAL", "HID", "NONE"
        self._serial_conn = None
        self._hid_device = None
        self._connected: bool = False
        self._relay_state: str = "OFF"  # "ON", "OFF"
        self._device_description: str = "Not Connected"
        self._lock = threading.Lock()
        self._last_error: Optional[str] = None
        self._last_scan_time: float = 0.0

        # Auto-detect device and ensure safe initialization to OFF state
        self._detect_and_init()
        atexit.register(self.safe_shutdown)

    def _detect_and_init(self):
        """Scans for available NOYITO USB relay hardware."""
        with self._lock:
            self._last_scan_time = time.monotonic()
            self._close_connections_unlocked()
            connected = self._try_connect_serial_unlocked()
            if not connected:
                connected = self._try_connect_hid_unlocked()

            if connected:
                self._connected = True
                # Ensure relay starts OFF on initialization
                self._send_off_unlocked()
            else:
                self._connected = False
                self.device_type = "NONE"
                self._device_description = "USB Relay Disconnected"

    def _try_connect_serial_unlocked(self) -> bool:
        """Attempt to find and connect via serial COM port."""
        try:
            import serial
            import serial.tools.list_ports
        except ImportError:
            return False

        ports = list(serial.tools.list_ports.comports())
        target_ports = []

        if self.port:
            target_ports = [p for p in ports if p.device.upper() == self.port.upper()]
        else:
            # Look for typical USB-to-UART chips: CH340, USB-SERIAL, FTDI, CP210x, Prolific
            for p in ports:
                desc = (p.description or '').lower()
                hwid = (p.hwid or '').lower()
                if any(k in desc or k in hwid for k in ('ch340', 'usb-serial', 'ftdi', 'pl2303', 'cp210', 'relay')):
                    target_ports.append(p)
            # If no specific match, test any available USB serial port
            if not target_ports:
                for p in ports:
                    if 'usb' in (p.hwid or '').lower():
                        target_ports.append(p)

        for p in target_ports:
            try:
                conn = serial.Serial(p.device, 9600, timeout=1.0, write_timeout=1.0)
                self._serial_conn = conn
                self.port = p.device
                self.device_type = "SERIAL"
                self._device_description = f"NOYITO Serial Relay ({p.device} - {p.description})"
                log.info('usb_relay_connected_serial: %s', self._device_description)
                return True
            except Exception as e:
                self._last_error = str(e)
                log.debug('serial_open_failed: %s (%s)', p.device, e)

        return False

    def _try_connect_hid_unlocked(self) -> bool:
        """Attempt to find and connect via USB HID."""
        try:
            import hid
        except ImportError:
            return False

        try:
            devices = hid.enumerate()
            for d in devices:
                vid = d.get('vendor_id', 0)
                pid = d.get('product_id', 0)
                prod = str(d.get('product_string') or '').lower()
                mfr = str(d.get('manufacturer_string') or '').lower()

                # NOYITO / DCT-Tech USB Relay: VID 0x16C0, PID 0x05DF or product contains 'relay'
                if (vid == 0x16C0 and pid == 0x05DF) or 'relay' in prod or 'relay' in mfr:
                    try:
                        dev = hid.device()
                        if d.get('path'):
                            dev.open_path(d['path'])
                        else:
                            dev.open(vid, pid)
                        self._hid_device = dev
                        self.device_type = "HID"
                        self._device_description = f"NOYITO HID Relay (VID: 0x{vid:04X}, PID: 0x{pid:04X})"
                        log.info('usb_relay_connected_hid: %s', self._device_description)
                        return True
                    except Exception as e:
                        self._last_error = str(e)
                        log.debug('hid_open_failed: %s', e)
        except Exception as e:
            self._last_error = str(e)

        return False

    def is_connected(self) -> bool:
        """Probe actual device connectivity."""
        with self._lock:
            # Verify existing connection or retry discovery
            if self.device_type == "SERIAL" and self._serial_conn:
                try:
                    if hasattr(self._serial_conn, 'is_open') and self._serial_conn.is_open:
                        return True
                except Exception:
                    pass
            elif self.device_type == "HID" and self._hid_device:
                return True

            # If disconnected, throttle re-scans to prevent blocking fast pipelines
            now = time.monotonic()
            if (now - self._last_scan_time) >= 15.0:
                self._last_scan_time = now
                self._close_connections_unlocked()
                connected = self._try_connect_serial_unlocked() or self._try_connect_hid_unlocked()
                if connected:
                    self._connected = True
                    self._send_off_unlocked()
                else:
                    self._connected = False
                    self.device_type = "NONE"
                    self._device_description = "USB Relay Disconnected"

            return self._connected

    def turn_on(self) -> bool:
        """
        Commands the relay ON (contacts close, siren energizes).
        Returns True if command sent successfully, False otherwise.
        """
        with self._lock:
            success = self._send_on_unlocked()
            if success:
                self._relay_state = "ON"
                self._last_error = None
                log.info('usb_relay_turned_on')
            else:
                self._relay_state = "OFF"
            return success

    def turn_off(self) -> bool:
        """
        Commands the relay OFF (contacts open, siren de-energizes).
        Returns True if command sent successfully, False otherwise.
        """
        with self._lock:
            success = self._send_off_unlocked()
            self._relay_state = "OFF"
            if success:
                self._last_error = None
                log.info('usb_relay_turned_off')
            return success

    def _send_on_unlocked(self) -> bool:
        """Low-level send ON command."""
        if not self._connected:
            self._try_connect_serial_unlocked()
            if not self._serial_conn:
                self._try_connect_hid_unlocked()

        if self.device_type == "SERIAL" and self._serial_conn:
            try:
                self._serial_conn.write(LC_RELAY_ON)
                self._serial_conn.flush()
                return True
            except Exception as e:
                self._last_error = f"Serial write error: {e}"
                log.warning('relay_serial_on_failed: %s', e)
                self._connected = False
                return False

        if self.device_type == "HID" and self._hid_device:
            try:
                # Send HID feature report
                self._hid_device.send_feature_report(list(HID_RELAY_ON))
                return True
            except Exception as e:
                self._last_error = f"HID write error: {e}"
                log.warning('relay_hid_on_failed: %s', e)
                self._connected = False
                return False

        self._last_error = "Device Disconnected"
        return False

    def _send_off_unlocked(self) -> bool:
        """Low-level send OFF command."""
        if not self._connected:
            self._try_connect_serial_unlocked()
            if not self._serial_conn:
                self._try_connect_hid_unlocked()

        if self.device_type == "SERIAL" and self._serial_conn:
            try:
                self._serial_conn.write(LC_RELAY_OFF)
                self._serial_conn.flush()
                return True
            except Exception as e:
                self._last_error = f"Serial write error: {e}"
                log.warning('relay_serial_off_failed: %s', e)
                self._connected = False
                return False

        if self.device_type == "HID" and self._hid_device:
            try:
                self._hid_device.send_feature_report(list(HID_RELAY_OFF))
                return True
            except Exception as e:
                self._last_error = f"HID write error: {e}"
                log.warning('relay_hid_off_failed: %s', e)
                self._connected = False
                return False

        return False

    def _close_connections_unlocked(self):
        """Safely closes open file handles."""
        if self._serial_conn:
            try:
                self._serial_conn.close()
            except Exception:
                pass
            self._serial_conn = None

        if self._hid_device:
            try:
                self._hid_device.close()
            except Exception:
                pass
            self._hid_device = None

    def safe_shutdown(self):
        """Emergency safe shutdown: ensures relay is OFF on exit."""
        try:
            self.turn_off()
        except Exception:
            pass
        with self._lock:
            self._close_connections_unlocked()

    def get_status(self) -> Dict[str, Any]:
        """Return diagnostic status for UI and monitoring."""
        connected = self.is_connected()
        return {
            'connected': connected,
            'relay_state': self._relay_state if connected else 'OFF',
            'device_type': self.device_type if connected else 'NONE',
            'device_description': self._device_description if connected else 'USB Relay Disconnected',
            'port': self.port,
            'last_error': self._last_error
        }

_global_relay: Optional[USBRelayController] = None

def get_relay_controller() -> USBRelayController:
    global _global_relay
    if _global_relay is None:
        _global_relay = USBRelayController()
    return _global_relay
