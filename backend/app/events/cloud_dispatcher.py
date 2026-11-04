"""
Prosper Smart Yard Cloud Outbox Dispatcher
Asynchronously synchronizes confirmed GateVision events and evidence images
to the remote Prosper Smart Yard API (syapi.prosperassettracking.com).
"""

from __future__ import annotations

import base64
import hashlib
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
import numpy as np
import requests
from app.snapshots.proof import proof_jpeg

log = logging.getLogger('gate.cloud')

FALLBACK_IMAGE_DATA_URI = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='


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


def _within_capture_window(left: Any, right: Any, seconds: float = 3.0) -> bool:
    """Compare camera timestamps safely when joining companion proof frames."""
    try:
        l = datetime.fromisoformat(str(left).replace('Z', '+00:00'))
        r = datetime.fromisoformat(str(right).replace('Z', '+00:00'))
        if l.tzinfo is None:
            l = l.replace(tzinfo=timezone.utc)
        if r.tzinfo is None:
            r = r.replace(tzinfo=timezone.utc)
        return abs((l - r).total_seconds()) <= seconds
    except Exception:
        return False


class ProsperSession:
    """Manages authentication tokens and HTTP requests to Prosper Smart Yard."""

    def __init__(self, settings):
        self.settings = settings
        self.base_url = (settings.prosper_api_base_url or '').strip().rstrip('/')
        self.auth_url = (getattr(settings, 'prosper_auth_url', None) or 'https://api.smartyard.prosperinfotech.com/admin/api/auth/login').strip()
        configured_events = getattr(settings, 'prosper_events_url', None)
        if configured_events:
            self.events_url = configured_events.strip()
        elif self.base_url.endswith('/gate'):
            self.events_url = f'{self.base_url}/api/v1/gate/events'
        elif 'smartyard.prosperinfotech.com' in self.base_url:
            self.events_url = f'{self.base_url}/gate/api/v1/gate/events'
        else:
            self.events_url = f'{self.base_url}/api/v1/gate/events'
        self.site_id = (settings.prosper_site_id or '').strip()
        self.client_id = (getattr(settings, 'prosper_client_id', None) or '').strip()
        self.session = requests.Session()
        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._last_auth_attempt: float = 0.0
        self._auth_cooldown_seconds: float = 30.0
        # A configured bearer token can expire.  After a 401, fall back to the
        # username/password login flow instead of retrying the same stale token
        # forever and leaving the outbox permanently offline.
        self._force_dynamic_auth: bool = False
        self._lock = threading.Lock()

    def get_auth_headers(self) -> Dict[str, str]:
        """Return Authorization or API key headers, along with X-Site-Id and X-Client-Id."""
        headers: Dict[str, str] = {}
        if self.settings.prosper_bearer_token and not self._force_dynamic_auth:
            headers['Authorization'] = f'Bearer {self.settings.prosper_bearer_token.get_secret_value()}'
        elif self.settings.prosper_api_key and not self._force_dynamic_auth:
            headers['x-api-key'] = self.settings.prosper_api_key.get_secret_value()
        else:
            # Auto-login credential flow
            with self._lock:
                now = time.monotonic()
                if self._token and now < self._token_expires_at - 30:
                    headers['Authorization'] = f'Bearer {self._token}'
                elif now - self._last_auth_attempt < self._auth_cooldown_seconds:
                    # In cooldown period after a failed login attempt; avoid blocking worker threads
                    pass
                else:
                    self._last_auth_attempt = now
                    username = getattr(self.settings, 'prosper_username', None) or self.settings.prosper_email or 'abhay@prosperinfotech.com'
                    configured_pw = self.settings.prosper_password.get_secret_value() if self.settings.prosper_password else ''
                    passwords_to_try = [configured_pw] if configured_pw else []

                    login_urls = [self.auth_url]
                    alt_url = f'{self.base_url}/api/auth/login'
                    if alt_url not in login_urls:
                        login_urls.append(alt_url)

                    res = None
                    for l_url in login_urls:
                        for pw in passwords_to_try:
                            for payload in [{'UserName': username, 'Password': pw}, {'username': username, 'password': pw}]:
                                try:
                                    res = self.session.post(l_url, json=payload, timeout=6)
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
                                    elif res.status_code == 204:
                                        # FastEndpoints cookie/header token check
                                        tok = res.cookies.get('token') or res.cookies.get('accessToken') or res.headers.get('Authorization')
                                        if tok:
                                            self._token = tok.replace('Bearer ', '')
                                            self._token_expires_at = time.monotonic() + 3600
                                            break
                                except Exception as e:
                                    log.debug('prosper_auth_login_error', extra={'error': str(e)})
                            if self._token:
                                break
                        if self._token:
                            break

                    if self._token:
                        headers['Authorization'] = f'Bearer {self._token}'
                    elif res is not None:
                        if res.status_code == 423:
                            log.warning('prosper_auth_account_locked: Cloud account %s is temporarily locked on the server (HTTP 423). Please unlock or wait for lockout cooldown.', username)
                        else:
                            log.warning('prosper_auth_login_failed', extra={'status_code': res.status_code, 'body': res.text[:200]})
                    else:
                        log.warning('prosper_auth_login_unreachable: Unable to connect to auth server (%s)', self.auth_url)

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

        headers = self.get_auth_headers()
        if 'Authorization' not in headers and 'x-api-key' not in headers:
            log.debug('prosper_image_upload_skipped_no_auth')
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

    def post_gate_event(self, payload: Dict[str, Any]) -> bool | str:
        """Send Gate Event to Prosper Smart Yard."""
        headers = self.get_auth_headers()
        if 'Authorization' not in headers and 'x-api-key' not in headers:
            log.info('prosper_gate_event_postponed_no_auth: Cloud authentication is not available (waiting for valid credentials or token). Dispatch buffered locally.')
            return 'OFFLINE'

        headers['Content-Type'] = 'application/json'

        # Check whether this is the new /api/v1/gate/events schema or legacy /api/sites/{site_id}/gate-events
        if 'VisitId' in payload or 'Containers' in payload:
            url = self.events_url
        else:
            if not self.site_id:
                log.warning('prosper_sync_skipped_missing_site_id')
                return 'REJECTED'
            url = f'{self.base_url}/api/sites/{self.site_id}/gate-events'

        try:
            res = self.session.post(url, json=payload, headers=headers, timeout=15)
        except requests.exceptions.RequestException as e:
            log.info('prosper_gate_event_network_offline: %s. Dispatch buffered locally in database.', e)
            return 'OFFLINE'
        except Exception as e:
            log.warning('prosper_gate_event_post_exception: %s', e)
            return 'OFFLINE'

        if res.status_code in (200, 201):
            resp_data = res.json() if res.content else {}
            log.info('prosper_gate_event_synced', extra={
                'event_id': payload.get('VisitId') or payload.get('id'),
                'visit_db_id': resp_data.get('visitDatabaseId', ''),
                'cloud_event_id': resp_data.get('eventId', ''),
            })
            # Return response data so caller can extract visitDatabaseId for GATE_OUT
            return resp_data if resp_data else True
        if res.status_code == 401:
            # 401 Unauthorized: token expired or rejected by cloud
            log.warning('prosper_gate_event_rejected_401: %s %s', res.status_code, res.text[:300],
                        extra={'status_code': res.status_code, 'body': res.text[:300]})
            with self._lock:
                self._token = None
                self._token_expires_at = 0.0
                self._force_dynamic_auth = True
            return 'OFFLINE'
        if res.status_code == 409:
            # 409 Conflict: E.g., "Unknown visit for GATE_OUT."
            # The cloud requires a prior GATE_IN visit before GATE_OUT can be processed.
            # Mark as non-retryable to avoid continuous failed loops on test/orphan exits.
            log.warning('prosper_gate_event_conflict_skipped: %s %s', res.status_code, res.text[:300],
                        extra={'status_code': res.status_code, 'body': res.text[:300]})
            return 'CONFLICT'
        if res.status_code in (500, 502, 503, 504):
            body_text = res.text[:400].strip()
            # Detect known Prosper API server-side bugs:
            # - DbUpdateConcurrencyException: optimistic concurrency failure on visits table (GATE_OUT bug)
            # - ck_departure: check constraint violation when GATE_IN has no corresponding visit
            if 'DbUpdateConcurrencyException' in body_text:
                event_type = payload.get('EventType', '')
                log.warning(
                    'prosper_gate_event_concurrency_retry: GATE_OUT concurrency conflict on cloud (known server bug). '
                    'Will retry. EventType=%s VisitId=%s',
                    event_type, payload.get('VisitId', ''),
                    extra={'status_code': 500, 'exception': 'DbUpdateConcurrencyException'}
                )
            elif 'ck_departure' in body_text:
                log.warning(
                    'prosper_gate_event_constraint_retry: visit constraint violation (ck_departure). '
                    'GATE_IN may not be committed yet. Will retry. VisitId=%s',
                    payload.get('VisitId', ''),
                    extra={'status_code': 500, 'exception': 'ck_departure'}
                )
            else:
                log.info('prosper_gate_event_cloud_service_unavailable: HTTP %s (%s). Dispatch buffered locally in database.',
                         res.status_code, body_text[:200])
            return 'OFFLINE'
        if res.status_code in (400, 422):
            log.warning('prosper_gate_event_rejected: %s %s', res.status_code, res.text[:300],
                        extra={'status_code': res.status_code, 'body': res.text[:300]})
            return 'REJECTED'

        log.warning('prosper_gate_event_failed: %s %s', res.status_code, res.text[:300],
                    extra={'status_code': res.status_code, 'body': res.text[:300]})
        return False

    def put_gate_event_images(self, cloud_event_id: str, payload: Dict[str, Any]) -> bool | str | dict:
        """
        Upload/update final multi-camera evidence images on the existing gate event.
        PUT https://api.smartyard.prosperinfotech.com/gate/api/v1/gate/events/{eventId}/images
        """
        if not cloud_event_id:
            log.warning('prosper_put_images_missing_cloud_event_id')
            return False

        headers = self.get_auth_headers()
        if 'Authorization' not in headers and 'x-api-key' not in headers:
            log.info('prosper_put_images_postponed_no_auth: Cloud authentication is not available.')
            return 'OFFLINE'

        headers['Content-Type'] = 'application/json'
        url = f'{self.events_url.rstrip("/")}/{cloud_event_id}/images'

        try:
            res = self.session.put(url, json=payload, headers=headers, timeout=25)
        except requests.exceptions.RequestException as e:
            log.info('prosper_put_images_network_offline: %s. Deferred upload buffered locally.', e)
            return 'OFFLINE'
        except Exception as e:
            log.warning('prosper_put_images_exception: %s', e)
            return 'OFFLINE'

        if res.status_code in (200, 201, 204):
            resp_data = res.json() if res.content and res.status_code != 204 else {}
            log.info('prosper_gate_event_images_updated', extra={
                'cloud_event_id': cloud_event_id,
                'status_code': res.status_code
            })
            return resp_data if resp_data else True

        if res.status_code == 401:
            log.warning('prosper_put_images_rejected_401: %s', res.text[:300])
            with self._lock:
                self._token = None
                self._token_expires_at = 0.0
                self._force_dynamic_auth = True
            return 'OFFLINE'

        if res.status_code == 404:
            log.warning('prosper_put_images_event_not_found: %s %s', res.status_code, res.text[:300])
            return 'NOT_FOUND'

        if res.status_code in (500, 502, 503, 504):
            log.info('prosper_put_images_cloud_service_unavailable: HTTP %s', res.status_code)
            return 'OFFLINE'

        if res.status_code in (400, 422):
            log.warning('prosper_put_images_rejected: %s %s', res.status_code, res.text[:300])
            return 'REJECTED'

        log.warning('prosper_put_images_failed: %s %s', res.status_code, res.text[:300])
        return False


class CloudOutboxDispatcher:
    """
    Dedicated non-blocking worker thread that serializes confirmed gate events,
    encodes snapshot evidence images, and dispatches gate events to Prosper Smart Yard.
    """

    def __init__(self, settings, metrics=None, repository=None, snapshots=None, runtime=None):
        self.settings = settings
        self.metrics = metrics
        self.repository = repository
        self.snapshots = snapshots
        self.runtime = runtime
        self.enabled = bool(settings.prosper_enabled)
        self.session = ProsperSession(settings) if self.enabled else None
        self.queue: Queue = Queue(maxsize=settings.prosper_queue_size)
        self._stop = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        self._is_online: bool = True
        self._last_sync_time: Optional[str] = None
        self._last_error: Optional[str] = None

        self._evidence_ready_events = set()
        self._state_lock = threading.Lock()

        self.device_uuid = (
            settings.prosper_device_uuid
            or 'EDGE-GATE-01'
        )

    def mark_evidence_ready(self, event_id: str):
        """Mark container session evidence collection as finalized."""
        with self._state_lock:
            self._evidence_ready_events.add(event_id)
        if self.repository:
            try:
                ev = self.repository.event(event_id)
                if ev and ev.get('cloud_event_id'):
                    self.enqueue(None, event_id, stage='PUT', force=True)
            except Exception:
                pass

    def is_evidence_ready(self, event_id: str, event_record: Optional[dict] = None) -> bool:
        """Check whether final multi-camera evidence images are ready for upload."""
        with self._state_lock:
            if event_id in self._evidence_ready_events:
                return True
        if self.runtime and getattr(self.runtime, 'passage_sessions', None):
            sess = self.runtime.passage_sessions.get_session_by_event_id(event_id)
            if sess and getattr(sess, 'evidence_ready', False):
                return True
        if event_record and event_record.get('snapshots') and len(event_record.get('snapshots', [])) > 0:
            return True
        if self.repository:
            try:
                ev = self.repository.event(event_id)
                if ev and ev.get('snapshots') and len(ev.get('snapshots', [])) > 0:
                    return True
            except Exception:
                pass
        return False

    def status(self) -> dict:
        pending_count = 0
        if self.repository and hasattr(self.repository, 'count_unsynced_confirmed_events'):
            try:
                pending_count = self.repository.count_unsynced_confirmed_events()
            except Exception:
                pass
        return {
            'enabled': self.enabled,
            'is_online': self._is_online if self.enabled else False,
            'pending_count': pending_count,
            'queue_depth': self.queue.qsize(),
            'last_sync_time': self._last_sync_time,
            'last_error': self._last_error,
            'site_id': self.session.site_id if (self.session and self.session.site_id) else None,
        }

    def trigger_retry(self) -> int:
        """Trigger an immediate offline backlog sweep and return the number of pending items queued."""
        if not self.enabled or not self.repository:
            return 0
        self._sweep_offline_backlog()
        return self.status()['pending_count']

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

    def enqueue(self, obs: Any, event_id: str, image_bytes: Optional[bytes] = None,
                payload: Optional[dict] = None, delay_seconds: float = 0.0,
                force: bool = False, stage: Optional[str] = None):
        """Enqueue confirmed gate event for background dispatch."""
        if not self.enabled:
            return
        if obs is not None and not getattr(obs, 'confirmed', False):
            return
        import re
        if obs is not None and obs.ocr:
            norm_obs = getattr(obs.ocr, 'normalized_text', '')
            if not norm_obs or len(norm_obs) != 11 or not re.match(r'^[A-Z]{4}[0-9]{7}$', norm_obs):
                return

        # Do not re-enqueue if already fully synced (unless forced) or invalid in database
        if self.repository:
            try:
                ev = self.repository.event(event_id)
                if not ev or ev.get('status') != 'CONFIRMED':
                    return
                if not force and ev.get('cloud_synced') and ev.get('cloud_delivery_state') == 'DELIVERED':
                    return
                cntr = ev.get('container_number')
                if not cntr or len(cntr) != 11 or not re.match(r'^[A-Z]{4}[0-9]{7}$', cntr):
                    return
                canonical = self.repository.canonical_event_id(event_id)
                if canonical and canonical != event_id:
                    # A later tracker/camera event for the same waiting
                    # container is represented by the original visit.
                    if hasattr(self.repository, 'mark_duplicate_event'):
                        self.repository.mark_duplicate_event(event_id, canonical)
                        primary_ev = self.repository.event(canonical)
                        if primary_ev and not primary_ev.get('cloud_synced') and primary_ev.get('status') == 'CONFIRMED':
                            self.enqueue(None, canonical, force=True)
                    else:
                        self.repository.mark_cloud_synced(event_id)
                    log.info('prosper_duplicate_event_suppressed', extra={
                        'event_id': event_id, 'canonical_event_id': canonical})
                    return
            except Exception:
                pass

        if stage is None:
            if obs is not None:
                stage = 'POST'
            else:
                stage = 'AUTO'

        item = {
            'event_id': event_id,
            'obs': obs,
            'image_bytes': image_bytes if image_bytes is not None else getattr(obs, 'image', None),
            'payload': payload,
            'enqueued_at': time.monotonic(),
            'not_before': time.monotonic() + max(0.0, float(delay_seconds or 0.0)),
            'retries': 0,
            'force': force,
            'stage': stage,
        }

        try:
            self.queue.put_nowait(item)
            if self.metrics:
                self.metrics.increment('prosper_events_queued')
        except Full:
            log.warning('prosper_cloud_queue_full_event_dropped', extra={'event_id': event_id})
            if self.metrics:
                self.metrics.increment('prosper_queue_dropped')

    def _proof_is_usable(self, image, quality: float) -> bool:
        """Reject blank, tiny, badly exposed or blurred frames before cloud POST."""
        if image is None or getattr(image, 'size', 0) == 0:
            return False
        try:
            h, w = image.shape[:2]
            if h < 240 or w < 320:
                return False
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if float(gray.std()) < 8.0:
                return False
            minimum = float(getattr(self.settings, 'cloud_min_proof_quality', 2.5))
            return float(quality) >= minimum
        except Exception:
            return False

    def _build_multi_camera_images(self, event_record: Optional[dict], obs: Any = None,
                                   img: Optional[bytes] = None, allow_single_fallback: bool = False):
        """
        Build isolated, validated multi-camera evidence images for this container event.
        Guarantees strict container-to-image association, gate isolation, and camera angle preservation.
        """
        cntr1 = (event_record.get('container_number') if event_record else None) or (obs.ocr.normalized_text if (obs and obs.ocr) else '')
        c1_images = []
        gate_id_for_event = (event_record.get('gate_id') if event_record else None) or (obs.gate_id if obs else None)
        expected_gate_cams = set()
        if self.runtime and getattr(self.runtime, 'cameras', None):
            expected_gate_cams = {
                c.id for c in self.runtime.cameras
                if c.gate_id == gate_id_for_event and getattr(c, 'enabled', True)
            }
        elif self.repository and hasattr(self.repository, 'cameras'):
            try:
                expected_gate_cams = {
                    c.id for c in self.repository.cameras()
                    if c.gate_id == gate_id_for_event and getattr(c, 'enabled', True)
                }
            except Exception:
                pass
        if not expected_gate_cams and self.settings and hasattr(self.settings, 'cameras'):
            try:
                expected_gate_cams = {
                    c.id for c in self.settings.cameras()
                    if c.gate_id == gate_id_for_event and getattr(c, 'enabled', True)
                }
            except Exception:
                pass

        primary_snap_id = event_record.get('primary_snapshot_id') if event_record else None
        if event_record and event_record.get('snapshots') and self.snapshots:
            cam_snapshots = {}
            for snap in event_record['snapshots']:
                cam_id = snap.get('camera_id')
                if not cam_id:
                    continue
                # Gate isolation: reject cameras not assigned to this gate
                if expected_gate_cams and cam_id not in expected_gate_cams:
                    continue
                cam_snapshots.setdefault(cam_id, []).append(snap)

            # Check filesystem event directory for any configured camera snapshots
            event_id_str = str(event_record.get('id') if event_record else '')
            event_dir = self.snapshots.root / event_id_str if (self.snapshots and event_id_str) else None
            if event_dir and event_dir.is_dir():
                for missing_cam in (expected_gate_cams - set(cam_snapshots.keys())):
                    disk_cands = [
                        event_dir / f"{missing_cam}.jpg",
                        event_dir / f"{missing_cam}.png",
                    ] + list(event_dir.glob(f"*{missing_cam}*.jpg")) + list(event_dir.glob(f"*{missing_cam}*.png"))
                    for cand in disk_cands:
                        if cand.is_file() and not cand.name.endswith('.quality'):
                            cam_snapshots.setdefault(missing_cam, []).append({
                                'id': f'disk_{missing_cam}',
                                'camera_id': missing_cam,
                                'image_path': f'{event_id_str}/{cand.name}',
                                'timestamp': event_record.get('timestamp')
                            })
                            break

            best_by_camera = {}
            for cam_id, snaps in cam_snapshots.items():
                best_snap = None
                best_data = None
                best_score = -1.0
                for snap in snaps:
                    try:
                        snap_cam = snap.get('camera_id')
                        snap_ts = snap.get('timestamp')
                        from app.database.repositories.gate import is_distinct_container, datetime_diff_seconds
                        is_mismatched = False
                        for det in event_record.get('detections', []):
                            if det.get('camera_id') == snap_cam:
                                det_ts = det.get('timestamp')
                                if det_ts and snap_ts and datetime_diff_seconds(det_ts, snap_ts) <= 2.5:
                                    for ocr in event_record.get('ocr_results', []):
                                        if ocr.get('detection_id') == det.get('id'):
                                            ocr_text = ocr.get('normalized_text')
                                            if ocr_text and ocr.get('valid_check_digit') and is_distinct_container(cntr1, ocr_text):
                                                is_mismatched = True
                                                break
                                    if is_mismatched:
                                        break
                        if is_mismatched:
                            log.warning('prosper_snapshot_skipped_conflicting_container', extra={
                                'camera_id': cam_id, 'snapshot_id': snap.get('id'), 'expected_container': cntr1
                            })
                            continue

                        snap_p = self.snapshots.path(snap['image_path'])
                        if not snap_p.is_file():
                            camera_id_str = str(snap.get('camera_id') or cam_id)
                            snap_id_str = str(snap.get('id') or '')
                            if event_dir and event_dir.is_dir():
                                cands = [
                                    event_dir / f"{camera_id_str}_{snap_id_str}.jpg",
                                    event_dir / f"{camera_id_str}_{snap_id_str}_rear.jpg",
                                    event_dir / f"{camera_id_str}.jpg",
                                    event_dir / f"{camera_id_str}.png",
                                ]
                                cands += list(event_dir.glob(f"*{camera_id_str}*.jpg"))
                                cands += list(event_dir.glob(f"*{camera_id_str}*.png"))
                                cands += list(event_dir.glob(f"*companion_{camera_id_str}.png"))
                                for cand in cands:
                                    if cand.is_file():
                                        snap_p = cand
                                        break
                        if not snap_p.is_file():
                            continue
                        data = snap_p.read_bytes()
                        decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if decoded is None or decoded.size == 0:
                            continue
                        if decoded.shape[0] < 240 or decoded.shape[1] < 320:
                            continue
                        quality = self.snapshots.compute_quality(decoded)
                        if not self._proof_is_usable(decoded, quality):
                            continue
                        score = quality
                        from app.snapshots.snapshot_manager import SnapshotManager
                        has_container = SnapshotManager._capture_has_visible_container(snap, event_record, decoded)
                        if not has_container:
                            continue
                        score += 10000.0
                        if primary_snap_id and snap.get('id') == primary_snap_id:
                            score += 1000.0
                        if score > best_score:
                            best_score = score
                            best_snap = snap
                            best_data = data
                    except Exception:
                        pass
                if best_snap and best_data:
                    best_by_camera[cam_id] = (best_snap, best_data)

            cam_roles = {}
            if self.runtime and getattr(self.runtime, 'cameras', None):
                for c in self.runtime.cameras:
                    cam_roles[c.id.lower()] = getattr(c, 'role', '')
            elif self.repository and hasattr(self.repository, 'cameras'):
                try:
                    for c in self.repository.cameras():
                        cam_roles[c.id.lower()] = getattr(c, 'role', '')
                except Exception:
                    pass
            if not cam_roles and self.settings and hasattr(self.settings, 'cameras'):
                try:
                    for c in self.settings.cameras():
                        cam_roles[c.id.lower()] = getattr(c, 'role', '')
                except Exception:
                    pass

            has_four_cams = any(c.lower().endswith(('-4', '-8')) or 'rear' in c.lower() for c in best_by_camera.keys())

            def _cam_image_type(c_id):
                c = (c_id or '').lower()
                role = (cam_roles.get(c) or '').upper()
                if role in ('FRONT_TOP', 'FRONT'):
                    return 'FRONT'
                if role in ('LEFT', 'SIDE'):
                    return 'LEFT'
                if role in ('RIGHT',):
                    return 'RIGHT'
                if role in ('REAR', 'BACK'):
                    return 'REAR'
                if any(k in c for k in ('rear', 'back')) or c.endswith('-4') or c.endswith('-8'):
                    return 'REAR'
                if any(k in c for k in ('left', 'side')) or c.endswith('-2') or c.endswith('-6'):
                    return 'LEFT'
                if any(k in c for k in ('right',)) or c.endswith('-7'):
                    return 'RIGHT'
                if c.endswith('-3'):
                    return 'RIGHT' if has_four_cams else 'REAR'
                return 'FRONT'

            def _cam_sort_key(c_id):
                itype = _cam_image_type(c_id)
                order = {'FRONT': 0, 'LEFT': 1, 'SIDE': 1, 'RIGHT': 2, 'REAR': 3}
                return (order.get(itype, 4), c_id.lower())

            for cam_id in sorted(best_by_camera.keys(), key=_cam_sort_key):
                snap, data = best_by_camera[cam_id]
                c1_images.append({
                    'ImageType': _cam_image_type(cam_id),
                    'CameraId': cam_id,
                    'CapturedAt': _iso_utc(snap.get('timestamp') or (obs.detection.frame_timestamp if obs else None)),
                    'Image': f"data:image/jpeg;base64,{base64.b64encode(proof_jpeg(data, snap, event_record)).decode('utf-8')}"
                })

        if not c1_images and allow_single_fallback:
            fallback_bytes = img if img is not None else (getattr(obs, 'image', None) if obs else None)
            if fallback_bytes is not None:
                try:
                    if not isinstance(fallback_bytes, bytes):
                        ok, buf = cv2.imencode('.jpg', fallback_bytes, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                        b = buf.tobytes() if ok else None
                    else:
                        b = fallback_bytes
                    if b:
                        cam_id = obs.detection.camera_id if obs and obs.detection else 'CAM-01'
                        c1_images.append({
                            'ImageType': _cam_image_type(cam_id) if '_cam_image_type' in locals() else 'FRONT',
                            'CameraId': cam_id,
                            'CapturedAt': _iso_utc(obs.detection.frame_timestamp if obs and obs.detection else None),
                            'Image': f"data:image/jpeg;base64,{base64.b64encode(b).decode('utf-8')}"
                        })
                except Exception:
                    pass

        # Strict deduplication: NEVER replicate images across camera angles.
        # Ensure each camera view is strictly unique per camera ID.
        seen_cameras = set()
        deduped_images = []
        for img_item in c1_images:
            cam = img_item.get('CameraId')
            b64_str = img_item.get('Image', '')
            if not b64_str or cam in seen_cameras:
                log.warning('prosper_duplicate_same_angle_image_suppressed', extra={'camera_id': cam})
                continue
            seen_cameras.add(cam)
            deduped_images.append(img_item)
        c1_images = deduped_images

        captured_cams = {img_item['CameraId'] for img_item in c1_images}
        return c1_images, expected_gate_cams, captured_cams

    def _sweep_offline_backlog(self):
        """
        Periodically checks SQLite database for confirmed events that were saved while offline.
        As soon as internet connectivity returns, dispatches them in chronological order.
        """
        if not self.enabled or not self.repository:
            return
        try:
            unsynced = self.repository.get_unsynced_confirmed_events(limit=20)
            if not unsynced:
                return

            if self.queue.qsize() > 0:
                return

            log.info('prosper_sweep_offline_backlog_found', extra={'count': len(unsynced)})
            for ev in unsynced:
                if self._stop.is_set():
                    break
                event_id = ev['id']
                cntr = ev.get('container_number')
                import re
                if not cntr or len(cntr) != 11 or not re.match(r'^[A-Z]{4}[0-9]{7}$', cntr):
                    self.repository.mark_cloud_synced(event_id)
                    continue
                cloud_ev_id = ev.get('cloud_event_id')
                # If cloud_event_id already exists, POST was completed; we only need to PUT the final images!
                stage = 'PUT' if cloud_ev_id else 'POST'
                try:
                    self.queue.put_nowait({
                        'event_id': event_id,
                        'obs': None,
                        'image_bytes': None,
                        'payload': None,
                        'enqueued_at': time.monotonic(),
                        'retries': 0,
                        'stage': stage,
                    })
                except Full:
                    break
        except Exception as e:
            log.warning('prosper_backlog_sweep_error', extra={'error': str(e)})

    def _run_loop(self):
        last_sweep = time.monotonic()
        while not self._stop.is_set():
            try:
                item = self.queue.get(timeout=1.0)
            except Empty:
                now = time.monotonic()
                if self.repository and (now - last_sweep) >= 15.0:
                    last_sweep = now
                    self._sweep_offline_backlog()
                continue

            event_id = item['event_id']
            obs = item['obs']
            img = item['image_bytes']
            pre_payload = item.get('payload')

            not_before = float(item.get('not_before') or 0.0)
            remaining = not_before - time.monotonic()
            if remaining > 0:
                if self._stop.wait(remaining):
                    self.queue.task_done()
                    break

            success = False
            try:
                # Persistent identity guard
                event_record = self.repository.event(event_id) if self.repository else None
                if self.repository and event_record:
                    if event_record.get('status') != 'CONFIRMED' or (event_record.get('cloud_synced') and not item.get('force') and event_record.get('cloud_delivery_state') == 'DELIVERED'):
                        self.queue.task_done()
                        continue
                    canonical = self.repository.canonical_event_id(event_id)
                    if canonical and canonical != event_id:
                        if hasattr(self.repository, 'mark_duplicate_event'):
                            self.repository.mark_duplicate_event(event_id, canonical)
                            primary_ev = self.repository.event(canonical)
                            if primary_ev and not primary_ev.get('cloud_synced') and primary_ev.get('status') == 'CONFIRMED':
                                self.enqueue(None, canonical, force=True)
                        else:
                            self.repository.mark_cloud_synced(event_id)
                        log.info('prosper_duplicate_backlog_suppressed', extra={
                            'event_id': event_id, 'canonical_event_id': canonical})
                        self.queue.task_done()
                        continue

                cntr1 = (event_record.get('container_number') if event_record else None) or (obs.ocr.normalized_text if (obs and obs.ocr) else '')
                import re
                if not cntr1 or len(cntr1) != 11 or not re.match(r'^[A-Z]{4}[0-9]{7}$', cntr1):
                    log.warning('prosper_dispatch_skipped_invalid_ocr', extra={'event_id': event_id, 'container_number': cntr1})
                    if self.repository:
                        self.repository.mark_cloud_synced(event_id)
                    self.queue.task_done()
                    continue

                if event_record and self.snapshots:
                    manifest = self.snapshots.build_evidence_manifest(
                        event_record,
                        min_quality=getattr(self.settings, 'cloud_min_proof_quality', 2.5),
                    )
                    if isinstance(manifest, dict):
                        event_record['evidence_manifest'] = manifest
                        manifest_cams = {c.get('camera_id') for c in (manifest.get('captures') or []) if c.get('camera_id')}
                        all_snaps = event_record.get('snapshots') or []
                        all_cams = {s.get('camera_id') for s in all_snaps if s.get('camera_id')}
                        if len(manifest_cams) >= len(all_cams) and manifest.get('captures'):
                            event_record['snapshots'] = manifest['captures']
                        event_record['primary_snapshot_id'] = manifest.get('primary_snapshot_id')
                        event_record['rear_snapshot_id'] = manifest.get('rear_snapshot_id')

                # Determine workflow stage: POST (immediate gate event creation) or PUT (final evidence update)
                stage = item.get('stage') or 'AUTO'
                if stage == 'AUTO':
                    cloud_ev_id = event_record.get('cloud_event_id') if event_record else None
                    stage = 'PUT' if cloud_ev_id else 'POST'

                # Common container metadata extraction
                iso_s1 = str((event_record.get('iso_code_1') if event_record else None) or (getattr(obs.ocr, 'size_type', None) if obs and obs.ocr else None) or '')
                cs1 = str((event_record.get('container_size') if event_record else None) or '')
                if not cs1 and iso_s1:
                    try:
                        from app.ocr.validator import extract_iso_size_code as _esc
                        _, cs1 = _esc(iso_s1)
                        cs1 = cs1 or ''
                    except Exception:
                        pass
                if '20' in cs1 or iso_s1.startswith('2'):
                    size_1 = '20'
                elif '45' in cs1 or iso_s1.startswith('L'):
                    size_1 = '45'
                else:
                    size_1 = '40'
                confidences = [float(o.get('confidence') or 0) for o in (event_record.get('ocr_results') or [])
                    if o.get('normalized_text') == cntr1 and o.get('valid_check_digit')] if event_record else []
                conf_1 = max(confidences, default=float(obs.ocr.confidence) if (obs and obs.ocr and obs.ocr.confidence) else 0.95)

                if stage == 'POST':
                    # STAGE 1: Immediate POST gate event creation
                    cloud_ev_id = event_record.get('cloud_event_id') if event_record else None
                    if cloud_ev_id and not item.get('force'):
                        # Event already exists on cloud; transition directly to PUT
                        item['stage'] = 'PUT'
                        self.queue.put(item)
                        self.queue.task_done()
                        continue

                    if pre_payload and isinstance(pre_payload, dict):
                        payload = pre_payload
                    else:
                        c1_images, expected_gate_cams, captured_cams = self._build_multi_camera_images(
                            event_record, obs, img, allow_single_fallback=True
                        )
                        if obs and hasattr(obs, 'detection'):
                            dt_captured = obs.detection.frame_timestamp
                        elif event_record and event_record.get('timestamp'):
                            try:
                                ts_str = event_record['timestamp'].rstrip('Z')
                                dt_captured = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
                            except Exception:
                                dt_captured = datetime.now(timezone.utc)
                        else:
                            dt_captured = datetime.now(timezone.utc)

                        visit_id = (event_record.get('cloud_visit_id') if event_record else None) or (item.get('visit_id') if item else None)
                        if not visit_id:
                            visit_id = f"VISIT-{dt_captured.strftime('%Y%m%d')}-{event_id[:8].upper()}"
                        direction = (
                            obs.direction.upper() if (obs and hasattr(obs, 'direction') and obs.direction)
                            else ((event_record.get('event_type') or 'ENTRY').upper() if event_record else 'ENTRY')
                        )
                        event_type = 'GATE_OUT' if direction in ('EXIT', 'OUT', 'GATE_OUT') else 'GATE_IN'

                        payload = {
                            'id': event_id,
                            'VisitId': visit_id,
                            'EventType': event_type,
                            'DeviceId': self.device_uuid,
                            'CapturedAt': _iso_utc(dt_captured),
                            'containerNumber': cntr1,
                            'Truck': {'TruckNumber': 'NA'},
                            'Driver': None,
                            'Containers': [{
                                'ContainerNumber': cntr1,
                                'ContainerNumberConfidence': round(conf_1, 2),
                                'Size': size_1,
                                'ContainerSize': cs1 or None,
                                'SizeCode': iso_s1 or None,
                                'Position': 'SINGLE',
                                'Images': c1_images
                            }]
                        }

                    from app.ocr.validator import is_valid_iso_6346
                    numbers = [c.get('ContainerNumber') for c in payload.get('Containers', [])]
                    if not numbers or not all(is_valid_iso_6346(n) for n in numbers):
                        if self.repository:
                            self.repository.mark_cloud_delivery(event_id, 'BLOCKED', 'Invalid container checksum')
                        self.queue.task_done()
                        continue

                    success = self.session.post_gate_event(payload)
                    if success == 'REJECTED' and self.runtime:
                        try:
                            ev_type = payload.get('EventType', 'GATE_OUT')
                            c_num = payload.get('containerNumber') or cntr1
                            self.runtime.emit_alert(
                                alert_type='CLOUD_SYNC_REJECTED',
                                title=f'Cloud Sync Rejected ({ev_type})',
                                message=f'Cloud rejected {ev_type} for {c_num or event_id[:8]} (Visit {payload.get("VisitId", "")})',
                                level='ERROR',
                                container_number=c_num,
                                event_id=event_id,
                                buzzer=False
                            )
                        except Exception:
                            log.exception('prosper_alert_failed: %s', event_id)
                    elif success == 'OFFLINE':
                        log.info('prosper_event_buffered_offline: Event %s buffered locally; will auto-sync when cloud service is available.', event_id)

                    if success is True or isinstance(success, dict):
                        self._is_online = True
                        self._last_sync_time = datetime.now(timezone.utc).isoformat()
                        self._last_error = None
                        returned_ev_id = None
                        returned_visit_id = None
                        if isinstance(success, dict):
                            returned_ev_id = success.get('eventId') or success.get('EventId')
                            returned_visit_id = success.get('visitId') or success.get('VisitId') or success.get('visitDatabaseId') or success.get('VisitDatabaseId')
                        if not returned_ev_id:
                            returned_ev_id = (str(returned_visit_id) if returned_visit_id else None) or event_id
                        if not returned_visit_id:
                            returned_visit_id = visit_id

                        if self.repository:
                            try:
                                self.repository.set_cloud_event_id(event_id, returned_ev_id, state='EVENT_CREATED', cloud_visit_id=str(returned_visit_id))
                            except Exception:
                                log.exception('prosper_set_cloud_event_id_failed: %s', event_id)

                        if self.runtime and getattr(self.runtime, 'passage_sessions', None):
                            try:
                                self.runtime.passage_sessions.set_cloud_event_id(event_id, returned_ev_id, cloud_visit_id=str(returned_visit_id))
                            except Exception:
                                pass

                        if self.metrics:
                            self.metrics.increment('prosper_events_queued')
                            self.metrics.increment('prosper_events_synced')
                            self.metrics.latency('prosper_post_sync', time.monotonic() - item['enqueued_at'])

                        log.info('prosper_gate_event_created_stage1_complete', extra={
                            'event_id': event_id,
                            'cloud_event_id': returned_ev_id,
                            'container_number': cntr1
                        })

                        # Trigger Stage 2 PUT if final evidence is already ready (Case B)
                        if self.is_evidence_ready(event_id, event_record):
                            self.enqueue(None, event_id, stage='PUT', force=True)

                        self.queue.task_done()
                        continue

                elif stage == 'PUT':
                    # STAGE 2: Deferred PUT evidence image upload using saved eventId
                    cloud_ev_id = event_record.get('cloud_event_id') if event_record else None
                    if not cloud_ev_id:
                        # Auto-heal: If POST was never dispatched or not yet completed, initiate POST immediately
                        if not item.get('post_initiated'):
                            item['post_initiated'] = True
                            self.enqueue(None, event_id, stage='POST', force=True)
                        # Case B: Evidence collection finished before POST response arrived
                        self.mark_evidence_ready(event_id)
                        put_wait = item.get('put_wait_count', 0)
                        if put_wait < 30:
                            item['put_wait_count'] = put_wait + 1
                            item['not_before'] = time.monotonic() + 0.5
                            self.queue.put(item)
                            self.queue.task_done()
                            continue
                        else:
                            log.warning('prosper_put_timeout_waiting_for_post_event_id', extra={'event_id': event_id})
                            self.queue.task_done()
                            continue

                    c1_images, expected_gate_cams, captured_cams = self._build_multi_camera_images(
                        event_record, obs, allow_single_fallback=False
                    )

                    missing_cams = expected_gate_cams - captured_cams
                    if missing_cams and not item.get('force'):
                        cam_waits = int(item.get('camera_wait_count', 0))
                        if cam_waits < 15:
                            item['camera_wait_count'] = cam_waits + 1
                            item['not_before'] = time.monotonic() + 1.0
                            self.queue.put(item)
                            self.queue.task_done()
                            continue
                        else:
                            log.warning('prosper_put_proceeding_with_partial_cameras', extra={
                                'event_id': event_id, 'cloud_event_id': cloud_ev_id,
                                'missing_cameras': sorted(list(missing_cams)),
                                'captured_cameras': sorted(list(captured_cams))
                            })

                    cloud_visit_id = (event_record.get('cloud_visit_id') if event_record else None) or item.get('visit_id')
                    if not cloud_visit_id:
                        if obs and hasattr(obs, 'detection'):
                            dt_captured = obs.detection.frame_timestamp
                        elif event_record and event_record.get('timestamp'):
                            try:
                                ts_str = event_record['timestamp'].rstrip('Z')
                                dt_captured = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
                            except Exception:
                                dt_captured = datetime.now(timezone.utc)
                        else:
                            dt_captured = datetime.now(timezone.utc)
                        cloud_visit_id = f"VISIT-{dt_captured.strftime('%Y%m%d')}-{event_id[:8].upper()}"

                    # Strict validation: verify images belong to this event and gate cameras
                    orig_cntr = event_record.get('container_number')
                    valid_images = [img_item for img_item in c1_images if img_item.get('Image', '').startswith('data:image/')]
                    if not valid_images:
                        log.warning('prosper_put_skipped_no_valid_images', extra={'event_id': event_id, 'cloud_event_id': cloud_ev_id})
                        if self.repository:
                            self.repository.mark_cloud_synced(event_id, cloud_event_id=cloud_ev_id, cloud_visit_id=str(cloud_visit_id))
                        self.queue.task_done()
                        continue

                    put_payload = {
                        'EventId': cloud_ev_id,
                        'VisitId': str(cloud_visit_id),
                        'Containers': [
                            {
                                'ContainerNumber': orig_cntr,
                                'ContainerNumberConfidence': round(conf_1, 2),
                                'Size': size_1,
                                'ContainerSize': cs1 or None,
                                'SizeCode': iso_s1 or None,
                                'Position': 'SINGLE',
                                'Images': valid_images
                            }
                        ]
                    }

                    success = self.session.put_gate_event_images(cloud_ev_id, put_payload)
                    if success is True or isinstance(success, dict):
                        self._is_online = True
                        self._last_sync_time = datetime.now(timezone.utc).isoformat()
                        self._last_error = None
                        if self.repository:
                            self.repository.mark_cloud_synced(event_id, cloud_event_id=cloud_ev_id, cloud_visit_id=str(cloud_visit_id))
                        if self.metrics:
                            self.metrics.increment('prosper_events_synced')
                            self.metrics.increment('prosper_images_updated')
                            self.metrics.latency('prosper_put_sync', time.monotonic() - item['enqueued_at'])
                        log.info('prosper_gate_event_images_updated_stage2_complete', extra={
                            'event_id': event_id,
                            'cloud_event_id': cloud_ev_id,
                            'container_number': orig_cntr,
                            'image_count': len(valid_images)
                        })
                        self.queue.task_done()
                        continue
                    else:
                        self._is_online = False
                        if self.repository:
                            self.repository.mark_cloud_delivery(
                                event_id,
                                'PUT_FAILED' if success == 'OFFLINE' else (success if success in ('REJECTED', 'NOT_FOUND') else 'PUT_FAILED'),
                                'Cloud image update PUT failed'
                            )
                        if success == 'REJECTED':
                            self._last_error = 'Cloud rejected PUT image update'
                            self.queue.task_done()
                            continue
                        item['retries'] += 1
                        if item['retries'] < self.settings.prosper_max_retries:
                            backoff = min(2.0 ** item['retries'], 30.0)
                            item['not_before'] = time.monotonic() + backoff
                            try:
                                self.queue.put_nowait(item)
                            except Full:
                                log.error('prosper_put_retry_dropped_queue_full', extra={'event_id': event_id})
                        self.queue.task_done()
                        continue

            except Exception as exc:
                log.exception('prosper_dispatch_exception: %s', exc)
                success = 'OFFLINE'

            # Fallback error handling for unhandled exceptions or initial failures
            self._is_online = False
            if self.repository:
                try:
                    self.repository.mark_cloud_delivery(
                        event_id,
                        success if success in ('REJECTED', 'CONFLICT', 'OFFLINE') else 'OFFLINE',
                        'Cloud dispatch error'
                    )
                except Exception:
                    pass
            item['retries'] += 1
            if item['retries'] < self.settings.prosper_max_retries:
                backoff = min(2.0 ** item['retries'], 30.0)
                item['not_before'] = time.monotonic() + backoff
                try:
                    self.queue.put_nowait(item)
                except Full:
                    log.error('prosper_retry_dropped_queue_full', extra={'event_id': event_id})
            self.queue.task_done()
