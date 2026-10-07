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
from app.ocr.fast_selector import FastFrameSelector
from app.ocr.validator import ContainerValidator

log = logging.getLogger('gate.passage_session')


def evaluate_frame_quality(image: np.ndarray, detections: list[Any] = (), camera_role: str = '') -> float:
    """
    Comprehensive multi-factor frame quality evaluation.
    Evaluates:
    - Container visibility & bounding box size/area
    - Number/text box visibility and clarity
    - Sharpness via Laplacian variance
    - Lighting, exposure, and contrast
    - Motion blur penalties
    - Camera angle relevance
    Returns a composite score (0.0 to 1000.0+).
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

        # Motion blur penalty: low variance indicates blur
        if variance < 25.0:
            sharpness_score *= 0.35

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

        # 3. Container & Text Visibility from Detections
        container_score = 0.0
        text_score = 0.0

        for d in detections:
            cname = getattr(d, 'class_name', '')
            bbox = getattr(d, 'bbox', None)
            conf = float(getattr(d, 'confidence', 0.8) or 0.8)
            if not bbox or len(bbox) != 4:
                continue
            bw = max(0.0, float(bbox[2] - bbox[0]))
            bh = max(0.0, float(bbox[3] - bbox[1]))
            area_ratio = (bw * bh) / float(w * h)

            if cname == 'container':
                # Well-framed container occupying >= 5% of frame area
                if area_ratio >= 0.05 and bw >= 60.0 and bh >= 40.0:
                    container_score = max(container_score, 120.0 * conf * min(1.0, area_ratio * 3.0))
            elif cname == 'container_number':
                if bw >= 40.0 and bh >= 12.0:
                    text_score = max(text_score, 200.0 * conf)
            elif cname == 'iso_code':
                if bw >= 18.0 and bh >= 10.0:
                    text_score = max(text_score, 100.0 * conf)

        # CRITICAL: A frame is ONLY useful if a container object is actually present in it!
        # Reject frames containing other objects (truck cabs without containers, cars, people, etc.)
        # or empty gate/road backgrounds.
        if container_score <= 0.0 and text_score <= 0.0:
            return 0.0

        # Angle relevance bonus
        angle_bonus = 10.0
        if camera_role in ('FRONT_TOP', 'LEFT', 'RIGHT', 'REAR'):
            angle_bonus = 20.0

        composite = sharpness_score + exposure_score + container_score + text_score + angle_bonus
        return float(composite)
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


class ContainerPassageSession:
    """
    Represents one physical container passage on a gate lane.
    Associates all cameras dynamically and buffers top frames per camera.
    """

    def __init__(self, session_id: str, gate_id: str, direction: str,
                 cameras: list[str], created_at: datetime,
                 target_frames_per_camera: int = 10):
        self.session_id = session_id
        self.gate_id = gate_id
        self.direction = direction
        self.cameras = list(cameras or [])
        self.created_at = created_at
        self.last_seen = created_at
        self.clear_since: Optional[datetime] = None
        self.target_frames_per_camera = target_frames_per_camera

        self.event_id: str = str(uuid4())
        self.finalized: bool = False

        # Per-camera frame candidate buffers
        self.captured_frames: Dict[str, List[CapturedFrameCandidate]] = {
            cam_id: [] for cam_id in self.cameras
        }

        # Multi-frame OCR candidates across all cameras
        self.ocr_candidates: List[OCRRead] = []
        self.confirmed_container_number: Optional[str] = None
        self.confirmed_iso_code: Optional[str] = None
        self.confirmed_confidence: float = 0.0

        self._lock = threading.Lock()

    def add_frame(self, camera_id: str, image: np.ndarray, timestamp: datetime,
                  detections: list[Any] = (), ocr_read: Optional[OCRRead] = None,
                  camera_role: str = '') -> Optional[CapturedFrameCandidate]:
        """
        Record a frame for camera_id only if the container is actually in this camera's view.
        Useless frames without a container are strictly ignored.
        """
        with self._lock:
            self.last_seen = max(self.last_seen, timestamp)
            self.clear_since = None
            if camera_id not in self.cameras:
                self.cameras.append(camera_id)

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
                role=camera_role
            )

            buf = self.captured_frames.setdefault(camera_id, [])
            buf.append(cand)

            # If buffer exceeds target_frames_per_camera capacity, retain highest-scoring candidates
            if len(buf) > self.target_frames_per_camera:
                buf.sort(key=lambda c: c.quality_score, reverse=True)
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

        if not val or not val.normalized_text:
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

        self.ocr_candidates.append(val)
        if val.valid_check_digit and (val.confidence > self.confirmed_confidence or not self.confirmed_container_number):
            self.confirmed_container_number = norm
            self.confirmed_confidence = val.confidence
            if val.size_type:
                self.confirmed_iso_code = val.size_type
        elif not self.confirmed_container_number:
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
        - Adapts to ANY number of captured frames per camera (e.g. 1 frame, 4 frames, 10 frames, 25 frames)
        """
        with self._lock:
            best_by_camera: Dict[str, CapturedFrameCandidate] = {}
            all_cams = list(dict.fromkeys(self.cameras + list(self.captured_frames.keys())))
            for cam_id in all_cams:
                frames = self.captured_frames.get(cam_id, [])
                if frames:
                    best = max(frames, key=lambda f: f.quality_score)
                    if best.quality_score > 0.0:
                        has_container = any(
                            getattr(d, 'class_name', '').lower() in ('container', 'cointainer', 'container_object', 'container_number', 'iso_code')
                            for d in (best.detections or [])
                        ) or best.quality_score > 10.0
                        if has_container:
                            best_by_camera[cam_id] = best
            return best_by_camera

    def get_consensus_container_number(self) -> Tuple[Optional[str], Optional[str], float]:
        """
        Cross-camera, multi-frame OCR consensus across all frames and cameras.
        Returns (container_number, iso_code, confidence).
        STRICT REQUIREMENT: Only returns container_number if it is a verified
        11-char container number (4 letters + 7 digits). Never returns unverified noise.
        """
        import re
        with self._lock:
            if self.ocr_candidates:
                # Character-level consensus across all reads
                consensus_engine = CharacterConsensus()
                best_read = consensus_engine.resolve(self.ocr_candidates)
                if best_read and best_read.normalized_text:
                    norm = best_read.normalized_text
                    if len(norm) == 11 and re.match(r'^[A-Z]{4}[0-9]{7}$', norm):
                        val = ContainerValidator().validate(best_read)
                        iso = (val.size_type if val else None) or self.confirmed_iso_code
                        conf = max(getattr(best_read, 'confidence', 0.8), self.confirmed_confidence)
                        return (norm, iso, conf)

            # Fallback to any individual read that had a verified container format
            if self.confirmed_container_number:
                norm = self.confirmed_container_number
                if len(norm) == 11 and re.match(r'^[A-Z]{4}[0-9]{7}$', norm):
                    return (norm, self.confirmed_iso_code, self.confirmed_confidence)

            return (None, self.confirmed_iso_code, 0.0)

    def is_ready_to_finalize(self, now: datetime, gate_clear_seconds: float = 6.0) -> bool:
        """
        Determine if the session has collected its target evidence and the container has cleared the gate.
        A session MUST NOT finalize while the container is actively present in the camera zone,
        otherwise subsequent frames of the same container will create duplicate sessions and duplicate events.
        """
        with self._lock:
            if self.finalized:
                return False

            # Zone clearance: container is no longer visible on any camera for >= gate_clear_seconds
            if self.clear_since is not None:
                clear_duration = abs((now - self.clear_since).total_seconds())
                if clear_duration >= gate_clear_seconds:
                    return True

            return False


class PassageSessionManager:
    """
    Coordinates container passage sessions across Gate IN and Gate OUT lanes.
    Guarantees that 1 physical container passage = 1 tracking session with 4 cameras
    and multiple frames per camera.
    """

    def __init__(self, settings=None, target_frames_per_camera: Optional[int] = None):
        self.settings = settings
        if target_frames_per_camera is not None:
            self.target_frames_per_camera = target_frames_per_camera
        else:
            self.target_frames_per_camera = getattr(settings, 'passage_target_frames_per_camera', 10) if settings else 10
        self.active_sessions: Dict[str, ContainerPassageSession] = {}  # gate_id -> session
        self.completed_sessions: Dict[str, ContainerPassageSession] = {}  # event_id -> session
        self._lock = threading.Lock()

    def get_or_create_session(self, gate_id: str, direction: str, timestamp: datetime,
                              gate_cameras: list[str]) -> ContainerPassageSession:
        """
        Retrieve active session for gate_id or initialize a new tracking session.
        Fully dynamic: works with any number of cameras configured for that gate.
        """
        with self._lock:
            session = self.active_sessions.get(gate_id)
            if session is None or session.finalized:
                session_id = f"TRACK_{uuid4().hex[:8].upper()}"
                session = ContainerPassageSession(
                    session_id=session_id,
                    gate_id=gate_id,
                    direction=direction,
                    cameras=gate_cameras or [],
                    created_at=timestamp,
                    target_frames_per_camera=self.target_frames_per_camera
                )
                self.active_sessions[gate_id] = session
                log.info('passage_session_started', extra={
                    'session_id': session_id, 'gate_id': gate_id, 'cameras': gate_cameras,
                    'target_frames_per_camera': self.target_frames_per_camera
                })
            else:
                session.last_seen = max(session.last_seen, timestamp)
                session.clear_since = None
                for c in (gate_cameras or []):
                    if c not in session.cameras:
                        session.cameras.append(c)
                        session.captured_frames.setdefault(c, [])
            return session

    def get_active_session(self, gate_id: str) -> Optional[ContainerPassageSession]:
        with self._lock:
            return self.active_sessions.get(gate_id)

    def mark_clearing(self, gate_id: str, timestamp: datetime):
        with self._lock:
            session = self.active_sessions.get(gate_id)
            if session and not session.clear_since:
                session.clear_since = timestamp

    def record_frame(self, camera_id: str, gate_id: str, direction: str,
                     image: np.ndarray, timestamp: datetime,
                     detections: list[Any] = (), ocr_read: Optional[OCRRead] = None,
                     gate_cameras: list[str] = None, camera_role: str = '') -> CapturedFrameCandidate:
        """
        Record a frame into the active container passage session.
        """
        cams = gate_cameras or [camera_id]
        session = self.get_or_create_session(gate_id, direction, timestamp, cams)
        if detections:
            session.clear_since = None
        return session.add_frame(camera_id, image, timestamp, detections, ocr_read, camera_role)

    def record_ocr_result(self, gate_id: str, read: OCRRead):
        with self._lock:
            session = self.active_sessions.get(gate_id)
            if session:
                session.add_ocr_candidate(read)

    def check_and_finalize(self, now: datetime,
                           gate_clear_seconds: float = 6.0) -> list[ContainerPassageSession]:
        """
        Check all active sessions and finalize those that are complete.
        """
        ready_to_finalize = []
        with self._lock:
            for gate_id, session in list(self.active_sessions.items()):
                if session.is_ready_to_finalize(now, gate_clear_seconds=gate_clear_seconds):
                    session.finalized = True
                    num, iso, conf = session.get_consensus_container_number()
                    if num:
                        session.confirmed_container_number = num
                    if iso:
                        session.confirmed_iso_code = iso
                    if conf > session.confirmed_confidence:
                        session.confirmed_confidence = conf

                    self.active_sessions.pop(gate_id, None)
                    self.completed_sessions[session.event_id] = session
                    ready_to_finalize.append(session)
                    log.info('passage_session_finalized', extra={
                        'session_id': session.session_id,
                        'event_id': session.event_id,
                        'gate_id': gate_id,
                        'frames_collected': {c: len(f) for c, f in session.captured_frames.items()},
                        'container_number': session.confirmed_container_number
                    })
        return ready_to_finalize
