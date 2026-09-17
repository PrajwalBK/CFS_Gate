from collections import deque
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime
import logging
from queue import Queue, Empty, Full
import threading
import time
from uuid import uuid4

from app.camera.camera_manager import CameraManager
from app.camera.camera_worker import CameraWorker
from app.camera.video_library import VideoLibrary
from app.camera.source_assignments import SourceAssignments
from app.camera.discovery import probe, stream_uri
from app.config.settings import CameraSourceConfig
from app.config.settings import resolve_source
from app.detection.tracker import TemporalTracker
from app.detection.yolo_detector import YoloDetector
from app.domain import Frame, Detection, Validation, utcnow
from app.events.cloud_dispatcher import CloudOutboxDispatcher
from app.events.event_manager import EventManager
from app.events.gate_logic import update_direction
from app.events.occupancy import ViewOccupancy
from app.metrics import Metrics
from app.ocr.container_ocr import crop_identification
from app.ocr.ocr_engine import EasyOCREngine, create_ocr_engine
from app.ocr.validator import ContainerValidator, extract_iso_size_code
from app.ocr.job_store import OCRJobStore

log = logging.getLogger('gate')


@dataclass(frozen=True)
class OCRJob:
    origin_key: str
    gate_id: str
    track_id: int
    detection: Detection
    frame: Frame
    direction: str
    started_at: datetime
    roi: tuple
    all_detections: tuple[Detection, ...] = ()
    role: str = 'UNASSIGNED'
    companion_frames: dict = field(default_factory=dict)
    slot_position: str = 'FRONT'
    event_id: str | None = None



class GateRuntime:
    def __init__(self, settings, cameras, repository, snapshots, manager=None, detector_factory=YoloDetector,
                 ocr_factory=None, camera_sources=None):
        self.settings, self.cameras = settings, cameras
        self.manager = manager or CameraManager(cameras, settings)
        self.videos = VideoLibrary(settings.upload_directory)
        self.camera_sources = camera_sources or []
        self._sources_by_id = {source.id: source for source in self.camera_sources}
        self._default_source_by_slot = {camera.id: source.id for camera, source in zip(cameras, self.camera_sources)}
        self.source_assignments = SourceAssignments(settings.upload_directory)
        self.role_assignments = SourceAssignments(settings.upload_directory, 'camera-roles.json')
        self._registry_lock = threading.RLock()
        self._discovered = {}
        self._discovery_wake = threading.Event()
        self.discovery_status = {'state': 'WAITING', 'found': 0, 'message': 'Waiting for camera discovery'}
        for index, camera in enumerate(self.cameras):
            saved_role = self.role_assignments.get(camera.id)
            if saved_role:
                updated = type(camera).model_validate({**camera.model_dump(), **saved_role})
                self.cameras[index] = updated
                self.manager.workers[camera.id].config = updated
        self._camera_locks = {c.id: threading.RLock() for c in cameras}
        self._model_lock = threading.Lock()
        self.repository, self.snapshots = repository, snapshots
        self.detector_factory = detector_factory
        self.ocr_factory = ocr_factory or create_ocr_engine
        self.metrics = Metrics()
        self.dispatcher = CloudOutboxDispatcher(settings, self.metrics)
        self.events = EventManager(settings, repository, snapshots, dispatcher=self.dispatcher)
        self.validator = ContainerValidator()
        self.tracker = {c.id: TemporalTracker(settings.track_ttl_seconds) for c in cameras}
        self.queue = Queue(maxsize=settings.ocr_queue_size)
        self.ocr_store = OCRJobStore(settings.upload_directory / 'ocr-spool', settings.ocr_spool_max_mb)
        self.ocr_spool_error = None
        self._stop = threading.Event()
        self._accelerator = threading.RLock()
        self._state_lock = threading.Lock()
        self.occupancy = {c.id: ViewOccupancy(settings) for c in cameras}
        self._threads = []
        self._recent = deque(maxlen=300)
        self._preview_detections = {}
        self._tracks = {}
        self._run_id = str(uuid4())
        self.detector = None
        self.ocr = None
        self.model_status = {'detector': 'DISABLED', 'ocr': 'DISABLED', 'error_type': None}
        self.last_processing_error = None

    def start(self):
        writer = threading.Thread(target=self._save_ocr_loop, name='gate-ocr-save', daemon=True)
        self._threads.append(writer)
        writer.start()
        for camera in self.cameras:
            saved = self.videos.info(camera.id)
            if saved and saved.get('active') and self.videos.path(saved['stored_name']).is_file():
                try:
                    self.activate_video(camera.id, saved)
                except Exception as exc:
                    log.warning('saved_video_unavailable', extra={'camera_id': camera.id, 'error_type': type(exc).__name__})
            else:
                selected = self.source_assignments.get(camera.id)
                if selected in self._sources_by_id:
                    self._install_camera_source(camera.id, selected, persist=False)
                else:
                    self.manager.workers[camera.id].start()
        if self.settings.pipeline_enabled:
            self.start_processing()
        self.dispatcher.start()
        if self.settings.camera_discovery_enabled and self.settings.deployment_mode != 'test':
            thread = threading.Thread(target=self._discovery_loop, name='gate-camera-discovery', daemon=True)
            self._threads.append(thread)
            thread.start()
        else:
            self.discovery_status = {'state': 'DISABLED', 'found': 0, 'message': 'Automatic discovery disabled'}

    def start_processing(self):
        with self._model_lock:
            if self.model_status['detector'] in ('READY', 'LOADING') and self.model_status['ocr'] in ('READY', 'LOADING'):
                return
            self.settings.pipeline_enabled = True
            self.model_status = {'detector': 'LOADING', 'ocr': 'LOADING', 'error_type': None}
            thread = threading.Thread(target=self._initialize_pipeline, name='gate-model-loader', daemon=True)
            self._threads.append(thread)
            thread.start()

    def camera_status(self):
        result = self.manager.status()
        for camera in result:
            video = self.videos.info(camera['id'])
            camera['video'] = {k: v for k, v in video.items() if k != 'stored_name'} if video else None
            selected = self.source_assignments.get(camera['id']) or self._default_source_by_slot.get(camera['id'])
            source = self._sources_by_id.get(selected)
            camera['active_source_id'] = source.id if source else None
            active_name = source.name if source else None
            if source and active_name:
                url = self._source_url(source)
                if url:
                    try:
                        from urllib.parse import urlsplit
                        h = urlsplit(url).hostname
                        if h and h not in active_name:
                            active_name = f"{active_name} — {h}"
                    except Exception:
                        pass
            camera['active_source_name'] = active_name
            with self._state_lock:
                camera['occupancy'] = self.occupancy[camera['id']].status()
        return result

    def available_camera_sources(self):
        from urllib.parse import urlsplit
        with self._registry_lock:
            claimed_hosts = set()
            for source in self.camera_sources:
                if source.id not in self._discovered:
                    url = resolve_source(source)
                    if url:
                        try:
                            host = urlsplit(url).hostname
                            if host:
                                claimed_hosts.add(host)
                        except Exception:
                            pass

            result = []
            for source in self.camera_sources:
                if not source.enabled:
                    continue
                # Skip duplicate discovered camera if its IP is already mapped to a channel
                if source.id in self._discovered:
                    dev_host = self._discovered.get(source.id, {}).get('host')
                    if dev_host in claimed_hosts:
                        continue

                url = self._source_url(source)
                host = ''
                if url:
                    try:
                        host = urlsplit(url).hostname or ''
                    except Exception:
                        pass

                result.append({
                    'id': source.id,
                    'name': source.name,
                    'host': host,
                    'configured': bool(url),
                    'discovered': source.id in self._discovered,
                    'connection_hint': self._discovered.get(source.id, {}).get('status', '')
                })
            return result

    def _source_url(self, source):
        return self._discovered.get(source.id, {}).get('uri', '') or resolve_source(source)

    def _discovery_loop(self):
        while not self._stop.is_set():
            self._discovery_wake.clear()
            self.discovery_status = {'state': 'SEARCHING', 'found': len(self._discovered), 'message': 'Searching connected networks for cameras…'}
            try:
                devices = probe(self._stop)
                for device in devices:
                    if self._stop.is_set():
                        return
                    passwords_to_try = []
                    configured_pw = self.settings.camera_onvif_password.get_secret_value()
                    if configured_pw:
                        passwords_to_try.append(configured_pw)
                    for candidate in ['Admin@123', 'Prosper@123']:
                        if candidate not in passwords_to_try:
                            passwords_to_try.append(candidate)

                    user = self.settings.camera_onvif_username or 'admin'
                    for pw in passwords_to_try:
                        try:
                            device['uri'] = stream_uri(device, user, pw)
                            device['status'] = 'Stream available'
                            break
                        except Exception:
                            continue
                    else:
                        device['uri'] = ''
                        device['status'] = 'Found — check ONVIF credentials/settings'
                    self._register_discovered(device)
                self.discovery_status = {'state': 'COMPLETE', 'found': len(devices),
                    'message': f'{len(devices)} cameras found on this scan' if devices else 'No ONVIF cameras found. Check PoE network and camera discovery settings.'}
            except Exception as exc:
                log.warning('camera_discovery_failed', extra={'error_type': type(exc).__name__})
                self.discovery_status = {'state': 'ERROR', 'found': 0, 'message': 'Camera discovery failed; retrying automatically'}
            self._discovery_wake.wait(self.settings.camera_discovery_interval_seconds)

    def _register_discovered(self, device):
        with self._registry_lock:
            if device['id'] not in self._discovered and len(self._discovered) >= 32:
                return
            old_uri = self._discovered.get(device['id'], {}).get('uri')
            self._discovered[device['id']] = device
            from urllib.parse import urlsplit
            already_claimed = any(
                urlsplit(resolve_source(s)).hostname == device.get('host')
                for s in self.camera_sources if s.id not in self._discovered and resolve_source(s)
            )
            if not already_claimed and device['id'] not in self._sources_by_id:
                source = CameraSourceConfig(id=device['id'], name=device['name'], source_env='DISCOVERED_CAMERA_UNUSED')
                self.camera_sources.append(source)
                self._sources_by_id[source.id] = source
        # Restore stable device IDs after restart; never overwrite operator choices.
        assigned = [c for c in self.cameras if self.source_assignments.get(c.id) == device['id']]
        if not device.get('uri'):
            return
        for camera in assigned:
            with self._camera_locks[camera.id]:
                worker = self.manager.workers[camera.id]
                if worker.config.source_type != 'file' and old_uri != device['uri']:
                    self._install_camera_source(camera.id, device['id'], persist=False)
        if assigned:
            return
        from urllib.parse import urlsplit
        if any(urlsplit(resolve_source(source)).hostname == device['host']
               for source in self.camera_sources if source.id not in self._discovered):
            return  # Existing configured camera already owns its view.
        for camera in list(self.cameras):
            with self._camera_locks[camera.id]:
                worker = self.manager.workers[camera.id]
                if (self.source_assignments.get(camera.id) or resolve_source(camera)
                        or worker.config.source_type == 'file' or not camera.enabled):
                    continue
                self.configure_role(camera.id, 'UNASSIGNED', 'UNKNOWN')
                self._install_camera_source(camera.id, device['id'], persist=True)
                break

    def configure_role(self, camera_id, role, direction):
        with self._camera_locks[camera_id]:
            index = next(i for i, c in enumerate(self.cameras) if c.id == camera_id)
            original = self.cameras[index]
            lane = {'ENTRY': 'lane-in', 'EXIT': 'lane-out', 'UNKNOWN': 'unassigned-' + camera_id}[direction]
            values = {'role': role, 'direction': direction, 'gate_id': lane,
                      'name': f"View {index + 1} · {direction} · {role.replace('_', ' ').title()}",
                      'line_axis': None, 'ocr_roi': (0, 0, 1, 1)}
            self.role_assignments.set(camera_id, values)
            updated = original.model_copy(update=values)
            self.cameras[index] = updated
            worker = self.manager.workers[camera_id]
            worker.config = worker.config.model_copy(update=values)
            worker.source_id = str(uuid4())  # Reject queued observations using previous calibration.
            worker.take()
            self.tracker[camera_id] = TemporalTracker(self.settings.track_ttl_seconds)

    def select_camera_source(self, camera_id, source_id):
        if source_id not in self._sources_by_id or not self._sources_by_id[source_id].enabled:
            raise ValueError('Camera source is not available')
        self._install_camera_source(camera_id, source_id, persist=True)

    def _install_camera_source(self, camera_id, source_id, persist):
        with self._camera_locks[camera_id]:
            original = next(c for c in self.cameras if c.id == camera_id)
            source = self._sources_by_id[source_id]
            previous = self.manager.workers[camera_id]
            previous.stop()
            if not previous.join():
                raise RuntimeError('Previous source is still shutting down; retry shortly')
            saved = self.videos.info(camera_id)
            if saved:
                self.videos.assign(camera_id, {**saved, 'active': False})
            replacement = CameraWorker(original.model_copy(update={'source_type': 'rtsp'}), self._source_url(source), self.settings, previous.stream_factory)
            self.manager.workers[camera_id] = replacement
            self.tracker[camera_id] = TemporalTracker(self.settings.track_ttl_seconds)
            if persist:
                self.source_assignments.set(camera_id, source_id)
            replacement.start()

    def activate_video(self, camera_id, info):
        with self._camera_locks[camera_id]:
            original = next(c for c in self.cameras if c.id == camera_id)
            previous = self.manager.workers[camera_id]
            config = original.model_copy(update={'source_type': 'file', 'enabled': True, 'loop_file': False})
            replacement = CameraWorker(config, str(self.videos.path(info['stored_name'])), self.settings, previous.stream_factory)
            replacement.prepare_video()
            previous.stop()
            if not previous.join():
                raise RuntimeError('Previous source is still shutting down; retry shortly')
            self.videos.assign(camera_id, {**info, 'active': True})
            self.manager.workers[camera_id] = replacement
            self.tracker[camera_id] = TemporalTracker(self.settings.track_ttl_seconds)
            self.occupancy[camera_id] = ViewOccupancy(self.settings)
            with self._state_lock:
                self._tracks = {k: v for k, v in self._tracks.items() if not k.startswith(f'{camera_id}:')}

    def control_video(self, camera_id, action):
        with self._camera_locks[camera_id]:
            worker = self.manager.workers[camera_id]
            if action == 'restore-camera':
                selected = self.source_assignments.get(camera_id) or self._default_source_by_slot.get(camera_id)
                if selected not in self._sources_by_id:
                    raise ValueError('Select a configured camera source')
                self._install_camera_source(camera_id, selected, persist=False)
            elif worker.config.source_type != 'file':
                raise ValueError('Add a video to this camera slot first')
            elif action == 'pause':
                worker.pause()
            elif action in ('play', 'restart'):
                if action == 'restart' or worker.status()['reason'] == 'file_completed':
                    saved = self.videos.info(camera_id)
                    if not saved:
                        raise ValueError('Restart is available for uploaded videos')
                    self.activate_video(camera_id, saved)
                self.manager.workers[camera_id].play()

    def activate_lane_video(self, gate_id, info):
        target_cameras = [c for c in self.cameras if c.gate_id == gate_id]
        if not target_cameras:
            raise ValueError(f'No cameras found for gate {gate_id}')
        for cam in target_cameras:
            self.activate_video(cam.id, info)

    def control_lane_video(self, gate_id, action):
        target_cameras = [c for c in self.cameras if c.gate_id == gate_id]
        if not target_cameras:
            raise ValueError(f'No cameras found for gate {gate_id}')
        for cam in target_cameras:
            try:
                self.control_video(cam.id, action)
            except Exception:
                pass

    def _process_current(self, camera_id, next_at):
        if not self._accelerator.acquire(timeout=0.1):
            return
        try:
            self._process_latest(camera_id, next_at)
        finally:
            self._accelerator.release()

    def _process_latest(self, camera_id, next_at):
        with self._camera_locks[camera_id]:
            camera = self.manager.workers[camera_id].config
            now = time.monotonic()
            if now < next_at[camera_id]:
                return
            worker = self.manager.workers[camera_id]
            if worker.status()['playback_status'] == 'PAUSED':
                return
            frame = worker.take()
            if frame is None:
                return
            if frame.sequence % (camera.frame_skip + 1):
                self.metrics.increment('frames_skipped')
                return
            next_at[camera_id] = now + 1 / (camera.inference_fps or self.settings.inference_fps)
            self.process_frame(camera, frame)

    def _initialize_pipeline(self):
        try:
            self.detector = self.detector_factory(self.settings)
            self.model_status['detector'] = 'READY'
            self.ocr = self.ocr_factory(self.settings)
            self.model_status['ocr'] = 'READY'
            if self._stop.is_set():
                return
            log.info('models_loaded')
            for name, target in [('gate-inference', self._inference_loop), ('gate-ocr', self._ocr_loop)]:
                thread = threading.Thread(target=target, name=name, daemon=True)
                self._threads.append(thread)
                thread.start()
        except Exception as exc:
            if self.detector is None:
                self.model_status['detector'] = 'ERROR'
            self.model_status['ocr'] = 'ERROR'
            self.model_status['error_type'] = type(exc).__name__
            log.error('model_load_failed', extra={'error_type': type(exc).__name__})

    def stop(self):
        self._stop.set()
        self._discovery_wake.set()
        self.dispatcher.stop()
        cameras_stopped = self.manager.stop()
        for thread in list(self._threads):
            thread.join(timeout=10)
        clean = cameras_stopped and not any(t.is_alive() for t in self._threads)
        if not clean:
            log.error('worker_shutdown_timeout')
        return clean

    def recent_detections(self, limit=50, camera_id=None):
        with self._state_lock:
            values = list(reversed(self._recent))
        return [d for d in values if camera_id is None or d['camera_id'] == camera_id][:limit]

    def _inference_loop(self):
        next_at = {c.id: 0. for c in self.cameras}
        while not self._stop.is_set():
            for camera in self.cameras:
                if self._stop.is_set():
                    break
                try:
                    self._process_current(camera.id, next_at)
                except Exception as exc:
                    self.metrics.increment('inference_errors')
                    self.last_processing_error = type(exc).__name__
                    log.error('inference_failed', extra={'camera_id': camera.id, 'error_type': type(exc).__name__})
            self._stop.wait(.005)

    def process_frame(self, camera, frame):
        with self._accelerator:
            started = time.monotonic()
            detections = self.detector.detect(frame)
            self.metrics.latency('inference', time.monotonic() - started)
        self.metrics.inference(camera.id)
        self.metrics.increment('inferences')
        tracks = self.tracker[camera.id].update(detections, frame.timestamp)
        with self._state_lock:
            occupant = self.occupancy[camera.id]
            selected = occupant.update(camera, frame, detections)
            self._preview_detections[camera.id] = (frame.source_id, frame.timestamp, tuple(detections))
            for detection in detections:
                self._recent.append({**asdict(detection), 'frame_timestamp': detection.frame_timestamp.isoformat() + 'Z'})
            cutoff = self.settings.track_ttl_seconds * 4
            self._tracks = {key: value for key, value in self._tracks.items()
                            if value['pending'] or (frame.timestamp - value['seen']).total_seconds() <= cutoff}
        for track in tracks:
            if track.detection is not selected:
                continue
            direction = update_direction(camera, track, frame.image.shape)
            with self._state_lock:
                occupancy = occupant.current
                if occupancy is None:
                    continue
                slot_pos = getattr(occupancy, 'active_slot', 'FRONT')
                active_slot_obj = getattr(occupancy, 'slots', {}).get(slot_pos)
                if active_slot_obj and active_slot_obj.closed:
                    continue
                if occupancy.closed and (not occupancy.is_twin or all(s.closed for s in occupancy.slots.values())):
                    continue
                origin = occupancy.origin
                if getattr(occupancy, 'is_twin', False) and slot_pos in getattr(occupancy, 'slots', {}):
                    origin = occupancy.slots[slot_pos].origin
                gate_event_id = getattr(occupancy, 'event_id', None) or getattr(occupancy, 'provisional_event_id', None)
                if not gate_event_id:
                    for other_cam in self.cameras:
                        if other_cam.gate_id == camera.gate_id and other_cam.id != camera.id:
                            other_occ = self.occupancy[other_cam.id].current
                            if other_occ:
                                cand = getattr(other_occ, 'event_id', None) or getattr(other_occ, 'provisional_event_id', None)
                                if cand:
                                    gate_event_id = cand
                                    break
                event_id = gate_event_id or getattr(occupancy, 'event_id', None) or getattr(occupancy, 'provisional_event_id', None)
                if event_id and not getattr(occupancy, 'provisional_event_id', None):
                    occupancy.provisional_event_id = event_id

                state = self._tracks.setdefault(origin, {'pending': False, 'done': False, 'last_ocr': 0., 'seen': frame.timestamp, 'samples': 0, 'last_ocr_ts': None})
                state['seen'] = frame.timestamp
                ts_since_last = (frame.timestamp - state['last_ocr_ts']).total_seconds() if state.get('last_ocr_ts') else 999.0
                wall_since_last = time.monotonic() - state['last_ocr']
                if state['pending'] or state['done']:
                    continue
                if state.get('last_ocr_ts') and ts_since_last < self.settings.ocr_interval_seconds:
                    continue
                if getattr(camera, 'source_type', 'live') != 'file' and getattr(frame, 'source_id', '') not in ('test', 'file') and wall_since_last < self.settings.ocr_interval_seconds:
                    continue
                if state.get('samples', 0) >= self.settings.ocr_max_attempts:
                    if state.get('done') or state.get('samples', 0) >= self.settings.ocr_max_attempts * 2:
                        continue
                if track.hits < self.settings.min_track_hits:
                    continue
                if track.detection.class_name not in self.settings.ocr_classes:
                    continue
                if camera.line_axis and direction == 'UNKNOWN':
                    continue

                companion_frames = {}
                for other_cam in self.cameras:
                    if other_cam.gate_id == camera.gate_id and other_cam.id != camera.id:
                        other_worker = self.manager.workers.get(other_cam.id)
                        if other_worker:
                            other_latest = other_worker.latest()
                            if other_latest and other_latest.image is not None:
                                companion_frames[other_cam.id] = other_latest.image.copy()

                job = OCRJob(origin, camera.gate_id, track.id, track.detection, frame, direction,
                             occupancy.first_seen, camera.ocr_roi, tuple(detections), camera.role,
                             companion_frames=companion_frames, slot_position=slot_pos,
                             event_id=event_id)
                try:
                    self.queue.put_nowait(job)
                    state['pending'] = True
                    state['samples'] = state.get('samples', 0) + 1
                    if active_slot_obj and active_slot_obj.stationary:
                        state['stationary_samples'] = state.get('stationary_samples', 0) + 1
                    state['last_ocr'] = time.monotonic()
                    state['last_ocr_ts'] = frame.timestamp
                except Full:
                    self.metrics.increment('ocr_queue_dropped')

    def _save_ocr_loop(self):
        """Disk I/O is isolated from capture and YOLO. Retry the current write."""
        while not self._stop.is_set():
            try:
                job = self.queue.get(timeout=.2)
            except Empty:
                continue
            try:
                crop = crop_identification(job.frame.image, job.detection.bbox, job.roi)
                while not self._stop.is_set():
                    try:
                        self.ocr_store.save(job, crop)
                        self.ocr_spool_error = None
                        self.metrics.increment('ocr_jobs_saved')
                        with self._state_lock:
                            if job.origin_key in self._tracks:
                                self._tracks[job.origin_key]['pending'] = False
                        break
                    except Exception as exc:
                        self.ocr_spool_error = type(exc).__name__
                        self.metrics.increment('ocr_spool_write_errors')
                        self._stop.wait(1)
            except Exception as exc:
                self.ocr_spool_error = type(exc).__name__
                self.metrics.increment('ocr_spool_write_errors')
                with self._state_lock:
                    if job.origin_key in self._tracks:
                        self._tracks[job.origin_key]['pending'] = False
            finally:
                self.queue.task_done()

    def _process_saved_ocr(self, job_id):
        job, crop, validation = self.ocr_store.load(job_id)
        prior = self.ocr_store.evidence(job.origin_key, self.settings.ocr_max_attempts)
        is_rear_slot = getattr(job, 'slot_position', 'FRONT') == 'REAR' or job.origin_key.endswith(':rear')
        if prior.event_id:
            prior_ev = self.repository.event(prior.event_id)
            if prior_ev and prior_ev.get('status') == 'CONFIRMED':
                if is_rear_slot and not prior_ev.get('container_number_2'):
                    pass  # Keep processing to extract container_number_2
                elif not is_rear_slot and prior_ev.get('container_number'):
                    self.ocr_store.complete(job_id, prior.event_id)
                    self._confirm_occupancy(job, prior.event_id)
                    return  # Already committed occupancy: no repeat OCR, event write or log.
        if validation is None:
            started = time.monotonic()
            read = None
            validation = None

            # 1. DIRECT TARGETED TEXT OCR:
            # If YOLO already detected the text box (cointainer / feet), run OCR
            # directly on that small cropped region first. Do not waste ~2.5s running
            # on the entire massive container background.
            if getattr(job, 'all_detections', None) and getattr(job, 'frame', None) and getattr(job.frame, 'image', None) is not None:
                c_box = job.detection.bbox  # pixel coords: (x1, y1, x2, y2)
                c_w = max(1.0, c_box[2] - c_box[0])
                c_h = max(1.0, c_box[3] - c_box[1])
                c_area = c_w * c_h
                margin = 80  # margin around container bbox

                cointainer_candidates = []
                feet_candidates = []

                for d in job.all_detections:
                    cls_name = d.class_name.lower()
                    fc_x = (d.bbox[0] + d.bbox[2]) / 2
                    fc_y = (d.bbox[1] + d.bbox[3]) / 2
                    within = (
                        c_box[0] - margin <= fc_x <= c_box[2] + margin and
                        c_box[1] - margin <= fc_y <= c_box[3] + margin
                    )
                    if not within:
                        continue

                    d_w = max(1.0, d.bbox[2] - d.bbox[0])
                    d_h = max(1.0, d.bbox[3] - d.bbox[1])
                    d_area = d_w * d_h

                    if cls_name == 'cointainer':
                        cointainer_candidates.append(d)
                    elif cls_name == 'feet':
                        feet_candidates.append(d)
                    elif cls_name in ('container_object', 'container') and d_area < 0.6 * c_area:
                        cointainer_candidates.append(d)

                cointainer_candidates.sort(key=lambda d: d.confidence, reverse=True)
                feet_candidates.sort(key=lambda d: d.confidence, reverse=True)

                img_h, img_w = job.frame.image.shape[:2]

                # Direct OCR on 'cointainer' text crops first
                for d in cointainer_candidates:
                    pad = 10
                    y1 = max(0, int(d.bbox[1]) - pad)
                    y2 = min(img_h, int(d.bbox[3]) + pad)
                    x1 = max(0, int(d.bbox[0]) - pad)
                    x2 = min(img_w, int(d.bbox[2]) + pad)
                    sub_crop = job.frame.image[y1:y2, x1:x2]
                    if sub_crop.size > 0:
                        sub_r = self.ocr.read(sub_crop)
                        sub_v = self.validator.validate(sub_r)
                        if sub_v.valid_check_digit:
                            read = sub_r
                            validation = sub_v
                            break

                # If container number was found, check feet boxes to extract ISO size code if not present
                if validation and validation.valid_check_digit and not validation.size_type and feet_candidates:
                    for fd in feet_candidates:
                        pad = 10
                        fy1 = max(0, int(fd.bbox[1]) - pad)
                        fy2 = min(img_h, int(fd.bbox[3]) + pad)
                        fx1 = max(0, int(fd.bbox[0]) - pad)
                        fx2 = min(img_w, int(fd.bbox[2]) + pad)
                        f_crop = job.frame.image[fy1:fy2, fx1:fx2]
                        if f_crop.size > 0:
                            f_r = self.ocr.read(f_crop)
                            iso_c, human_s = extract_iso_size_code(f_r.raw_text)
                            if iso_c or human_s:
                                validation = Validation(
                                    raw_text=f"{validation.raw_text} {f_r.raw_text}",
                                    normalized_text=validation.normalized_text,
                                    confidence=validation.confidence,
                                    valid_format=validation.valid_format,
                                    valid_check_digit=validation.valid_check_digit,
                                    validation_status=validation.validation_status,
                                    size_type=iso_c or human_s
                                )
                                break

                # If still not valid, try 'feet' candidates (might contain full container number / valid code)
                if validation is None or not validation.valid_check_digit:
                    for d in feet_candidates:
                        pad = 10
                        y1 = max(0, int(d.bbox[1]) - pad)
                        y2 = min(img_h, int(d.bbox[3]) + pad)
                        x1 = max(0, int(d.bbox[0]) - pad)
                        x2 = min(img_w, int(d.bbox[2]) + pad)
                        sub_crop = job.frame.image[y1:y2, x1:x2]
                        if sub_crop.size > 0:
                            sub_r = self.ocr.read(sub_crop)
                            sub_v = self.validator.validate(sub_r)
                            if sub_v.valid_check_digit:
                                read = sub_r
                                validation = sub_v
                                break

            # 2. FALLBACK TO MAIN CONTAINER CROP:
            # Only if targeted sub-boxes were missing or failed to produce a valid check digit.
            if validation is None or not validation.valid_check_digit:
                main_read = self.ocr.read(crop)
                main_val = self.validator.validate(main_read)
                if validation is None or main_val.valid_check_digit or main_val.confidence > validation.confidence:
                    read = main_read
                    validation = main_val

            self.metrics.latency('ocr', time.monotonic() - started)
            self.ocr_store.save_validation(job_id, validation)
        # Recover confirmation votes from disk, including after a restart.
        # Saved jobs retain historical source/lane/role; never relabel them to
        # the currently selected camera or attach current-time companion images.
        self.events.evidence[job.origin_key] = self.ocr_store.evidence(job.origin_key, self.settings.ocr_max_attempts)
        try:
            additional = dict(getattr(job, 'companion_frames', {}) or {})
            for other_cam in self.cameras:
                if other_cam.gate_id == job.gate_id and other_cam.id != job.detection.camera_id:
                    if other_cam.id not in additional:
                        other_worker = self.manager.workers.get(other_cam.id)
                        if other_worker:
                            other_latest = other_worker.latest()
                            if other_latest and other_latest.image is not None:
                                additional[other_cam.id] = other_latest.image.copy()

            proof = self.ocr_store.best_proof(job.origin_key, job_id) or job
            ongoing = getattr(job, 'event_id', None)
            if validation.valid_check_digit:
                with self._state_lock:
                    for view in self.cameras:
                        state = self.occupancy[view.id].current
                        if (view.gate_id == job.gate_id and state and state.event_id):
                            if (state.container_number == validation.normalized_text or
                                    getattr(state, 'is_twin', False) or
                                    job.origin_key.endswith(':rear') or
                                    getattr(job, 'slot_position', 'FRONT') == 'REAR'):
                                ongoing = state.event_id
                                break
            result = self.events.observe(job, validation, additional_frames=additional,
                                         proof_job=proof, ongoing_event_id=ongoing)
            self.ocr_store.complete(job_id, result[0] if result else None)
        finally:
            self.events.evidence.pop(job.origin_key, None)
        with self._state_lock:
            state = self._tracks.get(job.origin_key)
            if state:
                state['pending'] = False
                ev = self.repository.event(result[0]) if result else None
                if ev:
                    if getattr(job, 'slot_position', 'FRONT') == 'REAR':
                        state['done'] = bool(ev.get('container_number_2'))
                    else:
                        state['done'] = bool(ev.get('container_number'))
                else:
                    state['done'] = False
        if result:
            self._confirm_occupancy(job, result[0])
            self.metrics.increment('events_created' if result[1] else 'evidence_associated')
            self.metrics.latency('event', (utcnow() - job.started_at).total_seconds())

    def _confirm_occupancy(self, job, event_id):
        event = self.repository.event(event_id)
        if event and event['status'] == 'CONFIRMED':
            with self._state_lock:
                num = event.get('container_number_2') if getattr(job, 'slot_position', 'FRONT') == 'REAR' else event.get('container_number')
                self.occupancy[job.detection.camera_id].confirm(job.origin_key, event_id, num or event['container_number'])

    def _ocr_loop(self):
        while not self._stop.is_set():
            job_id = None
            try:
                job_id = self.ocr_store.next_id()
                if job_id is None:
                    try:
                        with self._state_lock:
                            protected = tuple(v.current.origin for v in self.occupancy.values() if v.current)
                        self.ocr_store.clean_completed(max_age_seconds=10, protected_origins=protected)
                    except Exception:
                        pass
                    self._stop.wait(.2)
                    continue
                self._process_saved_ocr(job_id)
                self.last_processing_error = None
            except Exception as exc:
                self.last_processing_error = type(exc).__name__
                self.metrics.increment('ocr_job_errors')
                log.error('saved_ocr_failed', extra={'error_type': type(exc).__name__})
                if job_id is not None:
                    try:
                        self.ocr_store.retry(job_id, type(exc).__name__)
                    except Exception:
                        self.ocr_spool_error = 'RetryWriteFailed'
                self._stop.wait(.2)
