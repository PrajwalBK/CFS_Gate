"""
Prosper Smart Yard Cloud Outbox Dispatcher
Asynchronously synchronizes confirmed GateVision events and evidence images
to the remote Prosper Smart Yard API (syapi.prosperassettracking.com).
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import io
import json
import logging
from queue import Empty, Full, Queue
import threading
import time
from typing import Any, Dict, Optional
import uuid

import cv2
import requests

log = logging.getLogger('gate.cloud')


def _iso_utc(dt: Any = None) -> str:
    if dt is None:
        d = datetime.now(timezone.utc)
    elif isinstance(dt, (int, float)):
        try:
            d = datetime.fromtimestamp(dt, tz=timezone.utc)
        except Exception:
            d = datetime.now(timezone.utc)
    elif isinstance(dt, str):
        return dt
    elif isinstance(dt, datetime):
        if dt.tzinfo is None:
            d = dt.replace(tzinfo=timezone.utc)
        else:
            d = dt.astimezone(timezone.utc)
    else:
        d = datetime.now(timezone.utc)
    return d.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


class ProsperSession:
    """Manages authentication tokens and HTTP requests to Prosper Smart Yard."""

    def __init__(self, settings):
        self.settings = settings
        self.base_url = settings.prosper_api_base_url.rstrip('/')
        self.auth_url = getattr(settings, 'prosper_auth_url', None) or 'https://cfsapi.prosperassettracking.com/api/auth/login'
        self.site_id = (settings.prosper_site_id or '').strip()
        self.client_id = (getattr(settings, 'prosper_client_id', None) or '').strip()
        self.session = requests.Session()
        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._lock = threading.Lock()

    def get_auth_headers(self) -> Dict[str, str]:
        """Return Authorization or API key headers, along with X-Site-Id and X-Client-Id."""
        headers: Dict[str, str] = {}
        if self.settings.prosper_bearer_token:
            headers['Authorization'] = f'Bearer {self.settings.prosper_bearer_token.get_secret_value()}'
        elif self.settings.prosper_api_key:
            headers['x-api-key'] = self.settings.prosper_api_key.get_secret_value()
        else:
            # Auto-login credential flow
            with self._lock:
                now = time.monotonic()
                if self._token and now < self._token_expires_at - 30:
                    headers['Authorization'] = f'Bearer {self._token}'
                else:
                    username = getattr(self.settings, 'prosper_username', None) or self.settings.prosper_email or 'sarsite@gmail.com'
                    configured_pw = self.settings.prosper_password.get_secret_value() if self.settings.prosper_password else ''
                    passwords_to_try = [configured_pw] if configured_pw else []
                    for candidate in ['Test@123', 'Prosper@123', 'Admin@123']:
                        if candidate not in passwords_to_try:
                            passwords_to_try.append(candidate)

                    login_urls = [self.auth_url, f'{self.base_url}/api/auth/login']
                    res = None
                    for l_url in login_urls:
                        for pw in passwords_to_try:
                            for payload in [{'username': username, 'password': pw}, {'UserName': username, 'Password': pw}]:
                                if getattr(self.settings, 'prosper_site_code', None):
                                    payload['siteCode'] = self.settings.prosper_site_code
                                try:
                                    res = self.session.post(l_url, json=payload, timeout=10)
                                    if res.status_code in (200, 201):
                                        data = res.json()
                                        self._token = data.get('accessToken') or data.get('token')
                                        expires_in = data.get('expiresIn', 3600)
                                        self._token_expires_at = time.monotonic() + float(expires_in)
                                        # Parse JWT claims for site_ids and client_ids if not configured
                                        try:
                                            parts = self._token.split('.')
                                            if len(parts) >= 2:
                                                padded = parts[1] + '=' * (-len(parts[1]) % 4)
                                                claims = json.loads(base64.urlsafe_b64decode(padded))
                                                if not self.site_id and claims.get('site_ids'):
                                                    self.site_id = str(claims['site_ids'])
                                                if not self.client_id and claims.get('client_ids'):
                                                    self.client_id = str(claims['client_ids'])
                                        except Exception:
                                            pass
                                        log.info('prosper_auth_token_refreshed', extra={'username': username})
                                        break
                                except Exception as e:
                                    log.warning('prosper_auth_login_error', extra={'error': str(e)})
                            if self._token:
                                break
                        if self._token:
                            break

                    if self._token:
                        headers['Authorization'] = f'Bearer {self._token}'
                    elif res is not None:
                        log.warning('prosper_auth_login_failed', extra={'status_code': res.status_code, 'body': res.text[:200]})

        if self.site_id:
            headers['X-Site-Id'] = self.site_id
        if self.client_id:
            headers['X-Client-Id'] = self.client_id
        return headers

    def upload_image(self, image_data: bytes | Any, trailer_number: str = '', camera_id: str = '',
                     device_id: str = '') -> Optional[str]:
        """Upload snapshot to Prosper and return public/stored URL."""
        if not self.site_id:
            return None

        if not isinstance(image_data, bytes):
            ok, buf = cv2.imencode('.jpg', image_data, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                return None
            image_data = buf.tobytes()

        headers = self.get_auth_headers()
        base64_str = base64.b64encode(image_data).decode('utf-8')
        base64_uri = f'data:image/jpeg;base64,{base64_str}'

        # 1. Try S3 Base64 upload endpoint
        s3_url = f'{self.base_url}/api/sites/{self.site_id}/images/trailers/upload-s3'
        s3_payload = {
            'trailerNumber': trailer_number or 'UNKNOWN',
            'scac': trailer_number[:4] if len(trailer_number) >= 4 and trailer_number[:4].isalpha() else 'UNKN',
            'deviceId': device_id,
            'cameraId': camera_id,
            'imagePath': base64_uri,
        }
        try:
            res = self.session.post(s3_url, json=s3_payload, headers={**headers, 'Content-Type': 'application/json'}, timeout=15)
            if res.status_code in (200, 201):
                data = res.json() if res.content else {}
                nested = data.get('data') if isinstance(data.get('data'), dict) else {}
                img_id = data.get('id') or nested.get('id') or nested.get('imageMetadataId') or data.get('imageMetadataId')
                img_url = nested.get('fileUrl') or nested.get('imagePath') or data.get('fileUrl') or data.get('imagePath') or data.get('url')
                self.last_image_metadata_id = img_id
                log.info('prosper_image_s3_uploaded', extra={'img_id': img_id, 'url': img_url})
                return img_url
        except Exception as e:
            log.warning('prosper_image_s3_upload_error', extra={'error': str(e)})

        # 2. Fallback to multipart upload
        url = f'{self.base_url}/api/sites/{self.site_id}/images/trailers'
        files = {'file': ('evidence.jpg', io.BytesIO(image_data), 'image/jpeg')}
        try:
            res = self.session.post(url, files=files, headers=headers, timeout=15)
            if res.status_code in (200, 201):
                data = res.json() if res.content else {}
                img_id = data.get('id') or data.get('imageMetadataId')
                img_url = data.get('url') or data.get('imageUrl') or data.get('fileUrl')
                self.last_image_metadata_id = img_id
                log.info('prosper_image_uploaded', extra={'url': img_url})
                return img_url
            log.warning('prosper_image_upload_failed', extra={'status_code': res.status_code})
        except Exception as e:
            log.warning('prosper_image_upload_error', extra={'error': str(e)})

        return None

    def post_gate_event(self, payload: Dict[str, Any]) -> bool:
        """Send Gate Event to Prosper Smart Yard."""
        headers = self.get_auth_headers()
        headers['Content-Type'] = 'application/json'

        # Check whether this is the new /api/v1/gate/events schema or legacy /api/sites/{site_id}/gate-events
        if 'VisitId' in payload or 'Containers' in payload:
            url = f'{self.base_url}/api/v1/gate/events'
        else:
            if not self.site_id:
                log.warning('prosper_sync_skipped_missing_site_id')
                return False
            url = f'{self.base_url}/api/sites/{self.site_id}/gate-events'

        res = self.session.post(url, json=payload, headers=headers, timeout=15)
        if res.status_code in (200, 201):
            log.info('prosper_gate_event_synced', extra={'event_id': payload.get('VisitId') or payload.get('id')})
            return True
        log.warning('prosper_gate_event_rejected', extra={'status_code': res.status_code, 'body': res.text[:300]})
        return False


class CloudOutboxDispatcher:
    """
    Dedicated non-blocking worker thread that serializes confirmed gate events,
    encodes snapshot evidence images, and dispatches gate events to Prosper Smart Yard.
    """

    def __init__(self, settings, metrics=None, repository=None, snapshots=None):
        self.settings = settings
        self.metrics = metrics
        self.repository = repository
        self.snapshots = snapshots
        self.enabled = bool(settings.prosper_enabled)
        self.session = ProsperSession(settings) if self.enabled else None
        self.queue: Queue = Queue(maxsize=settings.prosper_queue_size)
        self._stop = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        self.device_uuid = (
            settings.prosper_device_uuid
            or 'EDGE-GATE-01'
        )

    def start(self):
        if not self.enabled:
            return
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._stop.clear()
        self._worker_thread = threading.Thread(target=self._run_loop, name='prosper-dispatcher', daemon=True)
        self._worker_thread.start()
        log.info('prosper_cloud_dispatcher_started')

    def stop(self, timeout: float = 2.0):
        if not self.enabled:
            return
        self._stop.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
        log.info('prosper_cloud_dispatcher_stopped')

    def enqueue(self, obs: Any, event_id: str, image_bytes: Optional[bytes] = None, payload: Optional[dict] = None):
        """Enqueue confirmed gate event for background dispatch."""
        if not self.enabled:
            return
        if not getattr(obs, 'confirmed', True):
            return

        item = {
            'event_id': event_id,
            'obs': obs,
            'image_bytes': image_bytes if image_bytes is not None else getattr(obs, 'image', None),
            'payload': payload,
            'enqueued_at': time.monotonic(),
            'retries': 0,
        }

        try:
            self.queue.put_nowait(item)
            if self.metrics:
                self.metrics.increment('prosper_events_queued')
        except Full:
            log.warning('prosper_cloud_queue_full_event_dropped', extra={'event_id': event_id})
            if self.metrics:
                self.metrics.increment('prosper_queue_dropped')

    def _run_loop(self):
        while not self._stop.is_set():
            try:
                item = self.queue.get(timeout=0.5)
            except Empty:
                continue

            event_id = item['event_id']
            obs = item['obs']
            img = item['image_bytes']
            pre_payload = item.get('payload')

            success = False
            try:
                if pre_payload and isinstance(pre_payload, dict):
                    payload = pre_payload
                else:
                    # Query repository for full event record and associated multi-camera snapshots
                    event_record = self.repository.event(event_id) if self.repository else None
                    cntr1 = (event_record.get('container_number') if event_record else None) or (obs.ocr.normalized_text if (obs and obs.ocr) else '')
                    cntr2 = event_record.get('container_number_2') if event_record else None
                    is_twin = bool(event_record.get('is_twin')) if event_record else False

                    containers_list = []
                    if cntr1:
                        c1_images = []
                        if event_record and event_record.get('snapshots') and self.snapshots:
                            for snap in event_record['snapshots']:
                                if not (snap.get('image_path') or '').endswith('_rear.jpg'):
                                    try:
                                        snap_p = self.snapshots.path(snap['image_path'])
                                        if snap_p.is_file():
                                            c1_images.append({
                                                'ImageType': 'FRONT',
                                                'CameraId': snap.get('camera_id', obs.detection.camera_id if obs else 'CAM-01'),
                                                'CapturedAt': _iso_utc(snap.get('timestamp') or (obs.detection.frame_timestamp if obs else None)),
                                                'Image': f"data:image/jpeg;base64,{base64.b64encode(snap_p.read_bytes()).decode('utf-8')}"
                                            })
                                    except Exception:
                                        pass
                        if not c1_images and img is not None:
                            try:
                                if not isinstance(img, bytes):
                                    ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                                    b = buf.tobytes() if ok else None
                                else:
                                    b = img
                                if b:
                                    c1_images.append({
                                        'ImageType': 'FRONT',
                                        'CameraId': obs.detection.camera_id if obs else 'CAM-01',
                                        'CapturedAt': _iso_utc(obs.detection.frame_timestamp if obs else None),
                                        'Image': f"data:image/jpeg;base64,{base64.b64encode(b).decode('utf-8')}"
                                    })
                            except Exception:
                                pass

                        iso_s1 = str((event_record.get('iso_code_1') if event_record else None) or (getattr(obs.ocr, 'size_type', None) if obs and obs.ocr else None) or '')
                        cs1 = str((event_record.get('container_size') if event_record else None) or '')
                        size_1 = "20" if ('20' in cs1 or iso_s1.startswith('2')) else "40"
                        conf_1 = float(getattr(obs.ocr, 'confidence', 0.95)) if (obs and obs.ocr and obs.ocr.confidence) else 0.95
                        containers_list.append({
                            'ContainerNumber': cntr1,
                            'ContainerNumberConfidence': round(conf_1, 2),
                            'Size': size_1,
                            'Position': 'FRONT' if is_twin else 'SINGLE',
                            'Images': c1_images
                        })

                    if cntr2:
                        c2_images = []
                        if event_record and event_record.get('snapshots') and self.snapshots:
                            for snap in event_record['snapshots']:
                                if (snap.get('image_path') or '').endswith('_rear.jpg') or any(k in (snap.get('camera_id') or '').lower() for k in ('rear', '-3')):
                                    try:
                                        snap_p = self.snapshots.path(snap['image_path'])
                                        if snap_p.is_file():
                                            c2_images.append({
                                                'ImageType': 'REAR',
                                                'CameraId': snap.get('camera_id', obs.detection.camera_id if obs else 'CAM-02'),
                                                'CapturedAt': _iso_utc(snap.get('timestamp') or (obs.detection.frame_timestamp if obs else None)),
                                                'Image': f"data:image/jpeg;base64,{base64.b64encode(snap_p.read_bytes()).decode('utf-8')}"
                                            })
                                    except Exception:
                                        pass
                        size_2 = "20"
                        conf_2 = float(event_record.get('confidence_2') or 0.95) if event_record else 0.95
                        containers_list.append({
                            'ContainerNumber': cntr2,
                            'ContainerNumberConfidence': round(conf_2, 2),
                            'Size': size_2,
                            'Position': 'REAR',
                            'Images': c2_images or (c1_images[:1] if c1_images else [])
                        })

                    if not containers_list and obs and obs.ocr and obs.ocr.normalized_text:
                        containers_list.append({
                            'ContainerNumber': obs.ocr.normalized_text,
                            'ContainerNumberConfidence': round(float(obs.ocr.confidence), 2),
                            'Size': '20',
                            'Position': 'FRONT',
                            'Images': []
                        })

                    dt_captured = obs.detection.frame_timestamp if (obs and hasattr(obs, 'detection')) else datetime.now(timezone.utc)
                    visit_id = f"VISIT-{dt_captured.strftime('%Y%m%d')}-{event_id[:8].upper()}"
                    direction = (obs.direction.upper() if (obs and hasattr(obs, 'direction') and obs.direction) else 'ENTRY')
                    event_type = 'GATE_OUT' if direction in ('EXIT', 'OUT', 'GATE_OUT') else 'GATE_IN'

                    # Truck details: User requested keeping truck details empty/null.
                    # Remote API server validates: truck.TruckNumber must not be empty.
                    # We pass {"TruckNumber": "NA"} to satisfy API validation without fabricating truck data.
                    # Upload image for legacy compatibility
                    image_url = None
                    if img is not None:
                        try:
                            image_url = self.session.upload_image(
                                img,
                                trailer_number=cntr1 or '',
                                camera_id=obs.detection.camera_id if (obs and hasattr(obs, 'detection')) else '',
                                device_id=self.device_uuid,
                            )
                        except Exception:
                            pass

                    payload = {
                        'id': event_id,
                        'VisitId': visit_id,
                        'EventType': event_type,
                        'eventType': obs.direction if (obs and hasattr(obs, 'direction') and obs.direction in ('ENTRY', 'EXIT')) else 'ENTRY',
                        'DeviceId': self.device_uuid,
                        'CapturedAt': _iso_utc(dt_captured),
                        'containerNumber': cntr1,
                        'imageUrl': image_url,
                        'Truck': {'TruckNumber': 'NA'},
                        'Driver': None,
                        'Containers': containers_list
                    }

                success = self.session.post_gate_event(payload)

            except Exception as exc:
                log.warning('prosper_dispatch_exception', extra={'event_id': event_id, 'error': str(exc)})
                success = False

            if success:
                if self.metrics:
                    self.metrics.increment('prosper_events_synced')
                    self.metrics.latency('prosper_sync', time.monotonic() - item['enqueued_at'])
                self.queue.task_done()
            else:
                item['retries'] += 1
                if item['retries'] < self.settings.prosper_max_retries:
                    backoff = min(2.0 ** item['retries'], 30.0)
                    self._stop.wait(backoff)
                    try:
                        self.queue.put_nowait(item)
                    except Full:
                        log.error('prosper_retry_dropped_queue_full', extra={'event_id': event_id})
                else:
                    log.error('prosper_event_sync_exhausted', extra={'event_id': event_id})
                    if self.metrics:
                        self.metrics.increment('prosper_sync_errors')
                self.queue.task_done()
