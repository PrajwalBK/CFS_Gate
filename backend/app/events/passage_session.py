"""
Multi-Camera & Multi-Frame Container Passage Session Manager.

Fulfills Requirements:
1. Every container passing through a gate is independently captured by ALL 4 cameras.
2. Captures multiple frames (e.g. up to 10 frames per camera = 40 frames total) per container.
3. Associates all 40 frames with ONE physical container tracking session (TRACK_xxx).
4. Selects the single BEST frame independently for each camera based on comprehensive quality scoring.
5. Preserves all distinct camera angles (C1_BEST, C2_BEST, C3_BEST, C4_BEST).
6. Multi-frame OCR consensus across all frames and cameras for ONE final ISO 6346 container number.
7. First camera recognizing the container NEVER skips or blocks other cameras.
8. Asynchronous, high-speed, non-blocking operation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from math import log1p
import threading
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import cv2
import numpy as np

from app.domain import Detection, OCRRead, Validation
from app.ocr.consensus import CharacterConsensus
from app.ocr.validator import is_valid_iso_6346
from app.ocr.fast_selector import FastFrameSelector
from app.ocr.validator import ContainerValidator

log = logging.getLogger('gate.passage_session')


def evaluate_frame_quality(image: np.ndarray, detections: list[Any] = (), camera_role: str = '') -> float:
    """
    Comprehensive multi-factor frame quality evaluation.
    Evaluates:
    1. Container visibility & bounding box size/area
    2. Bounding-box quality (ratio, minimum dimensions, edge-clipping check)
    3. Image sharpness via Laplacian variance
    4. Motion blur penalties
    5. OCR text readability (container_number / iso_code presence & confidence)
    6. Resolution & frame sizing
    7. Lighting, exposure, and contrast
    8. Perspective & camera angle relevance
    9. Occlusion penalties
    10. Overall image clarity composite score (0.0 to 1000.0+).
    """
    if image is None or getattr(image, 'size', 0) == 0:
        return 0.0

    try:
        h, w = image.shape[:2]
        if h < 100 or w < 100:
            return 0.0

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 1. Sharpness / Clarity (Laplacian variance)
        # Downsample large frames for ultra-fast calculation (< 1ms)
        calc_gray = gray
        if h > 400 or w > 600:
            calc_gray = cv2.resize(gray, (min(w, 400), min(h, 250)), interpolation=cv2.INTER_NEAREST)
        variance = float(cv2.Laplacian(calc_gray, cv2.CV_64F).var())
        sharpness_score = log1p(variance) * 20.0  # Scales from ~20 to ~160

        # Motion blur penalty: low variance indicates blur or motion smearing
        if variance < 25.0:
            sharpness_score *= 0.35
        elif variance < 45.0:
            sharpness_score *= 0.65

        # 2. Lighting, Exposure & Contrast
        mean_lum = float(gray.mean())
        std_lum = float(gray.std())
        # Penalize over-exposed (> 235) or under-exposed (< 25)
        if 40.0 <= mean_lum <= 215.0 and std_lum >= 15.0:
            exposure_score = 30.0
        elif 25.0 <= mean_lum <= 235.0 and std_lum >= 10.0:
            exposure_score = 15.0
        else:
            exposure_score = 5.0

        # 3. Resolution factor
        resolution_score = min(40.0, (float(w * h) / (1920.0 * 1080.0)) * 40.0)

        # 4. Container & Text Visibility from Detections
        container_score = 0.0
        text_score = 0.0
        bbox_quality_score = 0.0
        has_occlusion_penalty = False

        for d in detections:
            cname = getattr(d, 'class_name', '')
            bbox = getattr(d, 'bbox', None)
            conf = float(getattr(d, 'confidence', 0.8) or 0.8)
            if not bbox or len(bbox) != 4:
                continue
            bw = max(0.0, float(bbox[2] - bbox[0]))
            bh = max(0.0, float(bbox[3] - bbox[1]))
            area_ratio = (bw * bh) / float(w * h)

            # Check if bbox is severely clipped against frame boundaries
            is_clipped = (bbox[0] < 5 or bbox[1] < 5 or bbox[2] > (w - 5) or bbox[3] > (h - 5))

            if cname == 'container':
                # Well-framed container occupying >= 5% of frame area
                if area_ratio >= 0.05 and bw >= 60.0 and bh >= 40.0:
                    c_val = 120.0 * conf * min(1.0, area_ratio * 3.0)
                    if is_clipped and area_ratio < 0.20:
                        c_val *= 0.70  # Edge-clipped sliver penalty
                        has_occlusion_penalty = True
                    container_score = max(container_score, c_val)

                    # Bounding box aspect ratio quality (containers typically 1.2 to 3.5 aspect ratio)
                    aspect = bw / max(1.0, bh)
                    if 0.8 <= aspect <= 4.5:
                        bbox_quality_score = max(bbox_quality_score, 25.0 * conf)
            elif cname == 'container_number':
                if bw >= 35.0 and bh >= 10.0:
                    t_val = 220.0 * conf
                    if not is_clipped:
                        t_val += 30.0  # Fully contained readable text box bonus
                    text_score = max(text_score, t_val)
            elif cname == 'iso_code':
                if bw >= 15.0 and bh >= 8.0:
                    text_score = max(text_score, 100.0 * conf)

        # CRITICAL: A frame is ONLY useful if a container object is actually present in it!
        # Reject frames containing other objects (truck cabs without containers, cars, people, etc.)
        # or empty gate/road backgrounds.
        if container_score <= 0.0 and text_score <= 0.0:
            return 0.0

        # Perspective & Angle relevance bonus
        angle_bonus = 10.0
        role_u = (camera_role or '').upper()
        if role_u in ('FRONT_TOP', 'FRONT'):
            angle_bonus = 25.0
        elif role_u in ('LEFT', 'RIGHT', 'SIDE'):
            angle_bonus = 20.0
        elif role_u in ('REAR', 'BACK'):
            angle_bonus = 22.0

        occlusion_penalty = -15.0 if has_occlusion_penalty else 0.0

        composite = (
            sharpness_score + exposure_score + resolution_score +
            container_score + text_score + bbox_quality_score +
            angle_bonus + occlusion_penalty
        )
        return max(1.0, float(composite))
    except Exception as exc:
        log.debug('quality_eval_exception', extra={'error': str(exc)})
        return 0.0


@dataclass
class CapturedFrameCandidate:
    camera_id: str
    image: np.ndarray
    timestamp: datetime
    quality_score: float
    detections: list[Any] = field(default_factory=list)
    ocr_read: Optional[OCRRead] = None
    role: str = ''
    track_id: Optional[int] = None
    source_id: str = ''
    frame_sequence: int = 0


class ContainerPassageSession:
    """
    Represents one physical container passage on a gate lane.
    Associates all cameras dynamically and buffers top frames per camera.
    Strictly isolated per physical container.
    """

    def __init__(self, session_id: str, gate_id: str, direction: str,
                 cameras: list[str], created_at: datetime,
                 target_frames_per_camera: int = 10,
                 container_slot: str = 'CONTAINER_1',
                 track_id: Optional[int] = None, min_ocr_confidence: float = 0.6):
        self.min_ocr_confidence = min_ocr_confidence
        self.session_id = session_id
        self.gate_id = gate_id
        self.direction = direction
        self.cameras = list(cameras or [])
        self.created_at = created_at
        self.last_seen = created_at
        self.clear_since: Optional[datetime] = None
        self.target_frames_per_camera = target_frames_per_camera
        self.container_slot = container_slot
        self.track_id = track_id
        self.tracks_by_camera: Dict[str, set[int]] = {}

        self.event_id: str = str(uuid4())
        self.finalized: bool = False

        # Per-camera frame candidate buffers: cam_id -> list of CapturedFrameCandidate
        self.captured_frames: Dict[str, List[CapturedFrameCandidate]] = {
            cam_id: [] for cam_id in self.cameras
        }

        # Multi-frame OCR candidates across all cameras
        self.ocr_candidates: List[OCRRead] = []
        self.confirmed_container_number: Optional[str] = None
        self.confirmed_iso_code: Optional[str] = None
        self.confirmed_confidence: float = 0.0
        self.identity_conflict = False
        self.sources_by_camera: Dict[str, str] = {}

        self._lock = threading.Lock()

    def record_camera_track(self, camera_id: str, track_id: Optional[int]):
        if track_id is not None:
            with self._lock:
                self.tracks_by_camera.setdefault(camera_id, set()).add(track_id)

    def add_frame(self, camera_id: str, image: np.ndarray, timestamp: datetime,
                  detections: list[Any] = (), ocr_read: Optional[OCRRead] = None,
                  camera_role: str = '',
                  track_id: Optional[int] = None, source_id: str = '', frame_sequence: int = 0) -> Optional[CapturedFrameCandidate]:
        """
        Record a frame for camera_id only if the container is actually in this camera's view.
        Useless frames without a container are strictly ignored.
        """
        with self._lock:
            # Detections must describe this exact camera capture. Text/size-only
            # observations do not prove that a physical container is visible.
            detections = [d for d in detections if d.camera_id == camera_id and d.frame_timestamp == timestamp]
            if not any(d.class_name in ('container', 'container_object') for d in detections):
                return None
            # If we already have a confirmed container number, verify that incoming OCR
            # does not belong to a different physical container
            if ocr_read and self.confirmed_container_number:
                val = ocr_read if isinstance(ocr_read, Validation) else ContainerValidator().validate(ocr_read)
                val_text = val.normalized_text
                import re
                if val_text and len(val_text) == 11 and re.match(r'^[A-Z]{4}[0-9]{7}$', val_text):
                    if val_text != self.confirmed_container_number:
                        if is_valid_iso_6346(val_text):
                            self.identity_conflict = True
                            # Contradicts confirmed container! Belongs to another physical container.
                            return None

            self.last_seen = max(self.last_seen, timestamp)
            self.clear_since = None
            if camera_id not in self.cameras:
                self.cameras.append(camera_id)

            if track_id is not None:
                self.tracks_by_camera.setdefault(camera_id, set()).add(track_id)

            q_score = evaluate_frame_quality(image, detections, camera_role)
            if q_score <= 0.0:
                # Useless frame: container is not present in this camera's angle
                return None

            cand = CapturedFrameCandidate(
                camera_id=camera_id,
                image=image.copy() if hasattr(image, 'copy') else image,
                timestamp=timestamp,
                quality_score=q_score,
                detections=list(detections),
                ocr_read=ocr_read,
                role=camera_role,
                track_id=track_id if track_id is not None else self.track_id,
                source_id=source_id, frame_sequence=frame_sequence
            )

            buf = self.captured_frames.setdefault(camera_id, [])
            buf.append(cand)

            # Retain highest-scoring candidates
            if len(buf) > self.target_frames_per_camera:
                buf.sort(key=lambda c: (self._identity_read_matches(c), c.quality_score), reverse=True)
                self.captured_frames[camera_id] = buf[:self.target_frames_per_camera]

            if ocr_read:
                self._add_ocr_candidate_unlocked(ocr_read)

            return cand

    def add_ocr_candidate(self, read: Any):
        with self._lock:
            self._add_ocr_candidate_unlocked(read)

    def _add_ocr_candidate_unlocked(self, read: Any):
        if not read:
            return
        validator = ContainerValidator()
        if isinstance(read, Validation):
            val = read
        elif isinstance(read, OCRRead):
            val = validator.validate(read)
        elif isinstance(read, str):
            val = validator.validate(OCRRead(raw_text=read, confidence=0.8))
        else:
            raw = getattr(read, 'raw_text', None) or getattr(read, 'normalized_text', '')
            conf = getattr(read, 'confidence', 0.8)
            val = validator.validate(OCRRead(raw_text=str(raw), confidence=float(conf)))

        if not val or not val.normalized_text or val.confidence < self.min_ocr_confidence:
            return

        # 1. Reject ISO size codes (like 21G7, 45G1) from ever becoming container number
        if getattr(val, 'validation_status', '') == 'VALID_SIZE_CODE' or (val.size_type and val.normalized_text == val.size_type):
            if val.size_type and not self.confirmed_iso_code:
                self.confirmed_iso_code = val.size_type
            return

        # 2. Must match container structure: 4 letters + 7 digits (11 characters)
        import re
        norm = val.normalized_text
        if not norm or len(norm) != 11 or not re.match(r'^[A-Z]{4}[0-9]{7}$', norm):
            if val.size_type and not self.confirmed_iso_code:
                self.confirmed_iso_code = val.size_type
            return

        # 3. Isolation protection: If confirmed container number is already established,
        # reject reads from a completely distinct container (prevents Container B contamination)
        if self.confirmed_container_number and norm != self.confirmed_container_number:
            if is_valid_iso_6346(norm):
                self.identity_conflict = True
                return

        self.ocr_candidates.append(val)
        if val.valid_check_digit and is_valid_iso_6346(norm) and (val.confidence > self.confirmed_confidence or not self.confirmed_container_number):
            self.confirmed_container_number = norm
            self.confirmed_confidence = val.confidence
            if val.size_type:
                self.confirmed_iso_code = val.size_type

    def get_best_frame_per_camera(self) -> Dict[str, CapturedFrameCandidate]:
        """
        Independently evaluate frames for every camera that captured frames during this passage,
        and return the single BEST frame for each camera angle.
        Fully dynamic and flexible:
        - Adapts to ANY number of cameras on the gate (1, 2, 3, 4, 6, 8, etc.)
        - Adapts to ANY number of captured frames per camera
        - Selects the independent best frame for each camera angle.
        """
        with self._lock:
            best_by_camera: Dict[str, CapturedFrameCandidate] = {}
            all_cams = list(dict.fromkeys(self.cameras + list(self.captured_frames.keys())))
            for cam_id in all_cams:
                frames = self.captured_frames.get(cam_id, [])
                if frames:
                    best = max(frames, key=lambda f: (self._identity_read_matches(f), f.quality_score))
                    if best.quality_score > 0.0:
                        has_container = any(
                            getattr(d, 'class_name', '').lower() in ('container', 'cointainer', 'container_object', 'container_number', 'iso_code')
                            for d in (best.detections or [])
                        )
                        if has_container:
                            best_by_camera[cam_id] = best
            return best_by_camera

    def _identity_read_matches(self, candidate) -> bool:
        read = candidate.ocr_read
        if not read or not self.confirmed_container_number:
            return False
        value = read if isinstance(read, Validation) else ContainerValidator().validate(read)
        return bool(value.valid_check_digit and value.normalized_text == self.confirmed_container_number
                    and value.confidence >= self.min_ocr_confidence)

    def get_consensus_container_number(self) -> Tuple[Optional[str], Optional[str], float]:
        """
        Cross-camera, multi-frame OCR consensus across all frames and cameras.
        Returns (container_number, iso_code, confidence).
        STRICT REQUIREMENT: Only returns container_number if it is a verified
        11-char container number (4 letters + 7 digits). Never returns unverified noise.
        """
        import re
        with self._lock:
            if self.identity_conflict:
                return (None, self.confirmed_iso_code, 0.0)
            if self.ocr_candidates:
                # Character-level consensus across all reads
                consensus_engine = CharacterConsensus()
                best_read = consensus_engine.resolve(self.ocr_candidates)
                if best_read and best_read.normalized_text:
                    norm = best_read.normalized_text
                    if best_read.valid_check_digit and is_valid_iso_6346(norm):
                        val = ContainerValidator().validate(best_read)
                        iso = (val.size_type if val else None) or self.confirmed_iso_code
                        conf = max(getattr(best_read, 'confidence', 0.8), self.confirmed_confidence)
                        return (norm, iso, conf)

            # Fallback to any individual read that had a verified container format
            if self.confirmed_container_number:
                norm = self.confirmed_container_number
                if is_valid_iso_6346(norm):
                    return (norm, self.confirmed_iso_code, self.confirmed_confidence)

            return (None, self.confirmed_iso_code, 0.0)

    def is_ready_to_finalize(self, now: datetime, gate_clear_seconds: float = 6.0) -> bool:
        """
        Determine if the session has collected its target evidence and the container has cleared the gate.
        """
        with self._lock:
            if self.finalized:
                return False

            # Zone clearance: container is no longer visible on any camera for >= gate_clear_seconds
            if self.clear_since is not None:
                clear_duration = abs((now - self.clear_since).total_seconds())
                if clear_duration >= gate_clear_seconds:
                    return True

            # If the session has a confirmed container number and evidence captured across cameras,
            # and no new frames have arrived for at least 2.5 seconds, it is ready to finalize.
            if self.confirmed_container_number and len(self.captured_frames) >= 1:
                idle_seconds = abs((now - self.last_seen).total_seconds())
                if idle_seconds >= 2.5:
                    return True

            # Safety timeout: if session has been seen and idle for >= (gate_clear_seconds + 2.0)
            if abs((now - self.last_seen).total_seconds()) >= (gate_clear_seconds + 2.0):
                return True

            return False


class PassageSessionManager:
    """
    Coordinates container passage sessions across Gate IN and Gate OUT lanes.
    Guarantees that 1 physical container passage = 1 tracking session with all cameras
    and multiple frames per camera.
    CRITICAL: Never merges or skips physical containers. Every container maintains
    its own independent session, track ID, OCR consensus, and event ID.
    """

    def __init__(self, settings=None, target_frames_per_camera: Optional[int] = None, outbox=None):
        self.settings = settings
        self.outbox = outbox
        if target_frames_per_camera is not None:
            self.target_frames_per_camera = target_frames_per_camera
        else:
            self.target_frames_per_camera = getattr(settings, 'passage_target_frames_per_camera', 10) if settings else 10
        self.active_sessions: Dict[str, ContainerPassageSession] = {}  # session_key -> session
        self.completed_sessions: Dict[str, ContainerPassageSession] = {}  # event_id -> session
        self.pending_finalizations: List[ContainerPassageSession] = []
        self._lock = threading.Lock()

    def get_or_create_session(self, gate_id: str, direction: str, timestamp: datetime,
                              gate_cameras: list[str],
                              container_slot: str = 'CONTAINER_1',
                              track_id: Optional[int] = None,
                              camera_id: Optional[str] = None, source_id: str = '') -> ContainerPassageSession:
        """
        Retrieve active session for gate_id + container_slot or initialize a new tracking session.
        Guarantees that each physical container (Container A, Container B) has its own independent session.
        """
        key = f"{gate_id}:{container_slot}"
        with self._lock:
            session = self.active_sessions.get(key)
            if session is not None and not session.finalized:
                # Check if this incoming frame represents a NEW physical container:
                # 1. Camera track change: If this specific camera already saw a different track ID
                #    and the session has had some duration (>= 1.5s) or has confirmed OCR/frames.
                new_track_on_camera = False
                if camera_id and track_id is not None:
                    seen_tracks = session.tracks_by_camera.get(camera_id, set())
                    if seen_tracks and track_id not in seen_tracks:
                        duration = abs((timestamp - session.created_at).total_seconds())
                        if duration >= 1.5 or session.confirmed_container_number:
                            new_track_on_camera = True

                # 2. Track ID change on session
                new_session_track = False
                if not camera_id and track_id is not None and session.track_id is not None and track_id != session.track_id:
                    duration = abs((timestamp - session.created_at).total_seconds())
                    if (session.confirmed_container_number or any(len(f) >= 2 for f in session.captured_frames.values())) and duration >= 1.5:
                        new_session_track = True

                # 3. Session clearance timeout
                is_cleared = session.is_ready_to_finalize(
                    timestamp,
                    gate_clear_seconds=getattr(self.settings, 'gate_clear_seconds', 6.0) if self.settings else 6.0
                )

                source_changed = bool(camera_id and source_id and session.sources_by_camera.get(camera_id)
                                      and session.sources_by_camera[camera_id] != source_id)
                if new_track_on_camera or new_session_track or is_cleared or source_changed:
                    # Finalize the prior session immediately into its own event!
                    num, iso, conf = session.get_consensus_container_number()
                    if num:
                        session.confirmed_container_number = num
                    if iso:
                        session.confirmed_iso_code = iso
                    if conf > session.confirmed_confidence:
                        session.confirmed_confidence = conf

                    if self.outbox:
                        self.outbox.save(session)
                    session.finalized = True
                    self.active_sessions.pop(key, None)
                    self.completed_sessions[session.event_id] = session
                    self.pending_finalizations.append(session)
                    log.info('passage_session_preempted_for_new_container', extra={
                        'prior_session_id': session.session_id,
                        'prior_event_id': session.event_id,
                        'new_track_id': track_id,
                        'camera_id': camera_id,
                        'gate_id': gate_id,
                        'container_number': session.confirmed_container_number
                    })
                    session = None

            if session is None or session.finalized:
                session_id = f"TRACK_{uuid4().hex[:8].upper()}"
                session = ContainerPassageSession(
                    session_id=session_id,
                    gate_id=gate_id,
                    direction=direction,
                    cameras=gate_cameras or [],
                    created_at=timestamp,
                    target_frames_per_camera=self.target_frames_per_camera,
                    container_slot=container_slot,
                    track_id=track_id,
                    min_ocr_confidence=getattr(self.settings, "ocr_min_confidence", 0.6)
                )
                if camera_id and track_id is not None:
                    session.record_camera_track(camera_id, track_id)
                self.active_sessions[key] = session
                log.info('passage_session_started', extra={
                    'session_id': session_id, 'gate_id': gate_id,
                    'container_slot': container_slot, 'track_id': track_id,
                    'cameras': gate_cameras,
                    'target_frames_per_camera': self.target_frames_per_camera
                })
            else:
                session.last_seen = max(session.last_seen, timestamp)
                session.clear_since = None
                if track_id is not None and session.track_id is None:
                    session.track_id = track_id
                if camera_id and track_id is not None:
                    session.record_camera_track(camera_id, track_id)
                for c in (gate_cameras or []):
                    if c not in session.cameras:
                        session.cameras.append(c)
                        session.captured_frames.setdefault(c, [])
            if camera_id and source_id:
                session.sources_by_camera[camera_id] = source_id
            return session

    def get_active_session(self, gate_id: str, container_slot: str = 'CONTAINER_1') -> Optional[ContainerPassageSession]:
        key = f"{gate_id}:{container_slot}"
        with self._lock:
            if key in self.active_sessions:
                return self.active_sessions[key]
            return None

    def acknowledge(self, event_id):
        with self._lock:
            self.completed_sessions.pop(event_id, None)

    def get_all_active_sessions(self, gate_id: Optional[str] = None) -> list[ContainerPassageSession]:
        with self._lock:
            if gate_id:
                return [s for k, s in self.active_sessions.items() if k.startswith(f"{gate_id}:") or k == gate_id]
            return list(self.active_sessions.values())

    def mark_clearing(self, gate_id: str, timestamp: datetime, container_slot: Optional[str] = None):
        with self._lock:
            if container_slot:
                key = f"{gate_id}:{container_slot}"
                session = self.active_sessions.get(key)
                if session and not session.clear_since:
                    session.clear_since = timestamp
            else:
                for k, session in self.active_sessions.items():
                    if (k == gate_id or k.startswith(f"{gate_id}:")) and not session.clear_since:
                        session.clear_since = timestamp

    def record_frame(self, camera_id: str, gate_id: str, direction: str,
                     image: np.ndarray, timestamp: datetime,
                     detections: list[Any] = (), ocr_read: Optional[OCRRead] = None,
                     gate_cameras: list[str] = None, camera_role: str = '',
                     container_slot: str = 'CONTAINER_1',
                     track_id: Optional[int] = None, source_id: str = '', frame_sequence: int = 0) -> Optional[CapturedFrameCandidate]:
        """
        Record a frame into the active container passage session for the specified container slot.
        """
        cams = gate_cameras or [camera_id]
        session = self.get_or_create_session(
            gate_id, direction, timestamp, cams,
            container_slot=container_slot, track_id=track_id,
            camera_id=camera_id, source_id=source_id
        )
        if detections:
            session.clear_since = None
        return session.add_frame(
            camera_id, image, timestamp, detections, ocr_read,
            camera_role=camera_role, track_id=track_id, source_id=source_id, frame_sequence=frame_sequence
        )

    def record_ocr_result(self, gate_id: str, read: OCRRead, container_slot: str = 'CONTAINER_1',
                          event_id: str | None = None, proof_job=None):
        with self._lock:
            key = f"{gate_id}:{container_slot}"
            session = self.active_sessions.get(key)
            if event_id and (not session or session.event_id != event_id):
                session = self.completed_sessions.get(event_id)
                if not session and self.outbox:
                    session = self.outbox.get(event_id)
            if session:
                session.add_ocr_candidate(read)
                if proof_job and read:
                    frame = proof_job.frame
                    matching = next((candidate for candidate in session.captured_frames.get(frame.camera_id, [])
                        if candidate.timestamp == frame.timestamp and candidate.frame_sequence == frame.sequence), None)
                    if matching:
                        matching.ocr_read = read
                    else:
                        session.add_frame(frame.camera_id, frame.image, frame.timestamp,
                            list(proof_job.all_detections or ()) + [proof_job.detection], read,
                            camera_role=proof_job.role, track_id=proof_job.track_id,
                            source_id=frame.source_id, frame_sequence=frame.sequence)
                if session.finalized and self.outbox:
                    self.outbox.save(session, replace_existing=True)

    def check_and_finalize(self, now: datetime,
                           gate_clear_seconds: float = 6.0) -> list[ContainerPassageSession]:
        """
        Check all active sessions and finalize those that are complete.
        Each physical container finalizes into its own independent session and event.
        """
        ready_to_finalize = []
        with self._lock:
            # First collect any preempted sessions waiting to be committed
            if self.pending_finalizations:
                ready_to_finalize.extend(self.pending_finalizations)
                self.pending_finalizations.clear()

            for key, session in list(self.active_sessions.items()):
                if session.is_ready_to_finalize(now, gate_clear_seconds=gate_clear_seconds):
                    num, iso, conf = session.get_consensus_container_number()
                    if num:
                        session.confirmed_container_number = num
                    if iso:
                        session.confirmed_iso_code = iso
                    if conf > session.confirmed_confidence:
                        session.confirmed_confidence = conf

                    if self.outbox:
                        self.outbox.save(session)
                    session.finalized = True
                    self.active_sessions.pop(key, None)
                    self.completed_sessions[session.event_id] = session
                    ready_to_finalize.append(session)
                    log.info('passage_session_finalized', extra={
                        'session_id': session.session_id,
                        'event_id': session.event_id,
                        'gate_id': session.gate_id,
                        'container_slot': session.container_slot,
                        'track_id': session.track_id,
                        'frames_collected': {c: len(f) for c, f in session.captured_frames.items()},
                        'container_number': session.confirmed_container_number
                    })
        return ready_to_finalize
