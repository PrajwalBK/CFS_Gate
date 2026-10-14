"""Durable finalized passages, independent of event DB and model availability."""
import base64
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import time

import cv2
import numpy as np

from app.domain import Detection, Validation
from app.events.passage_session import ContainerPassageSession, CapturedFrameCandidate


class PassageOutbox:
    def __init__(self, root):
        self.path = Path(root) / 'passage-outbox.sqlite3'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS passages (
                event_id TEXT PRIMARY KEY, payload TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0, retry_at REAL NOT NULL DEFAULT 0,
                error TEXT)''')

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA synchronous=FULL')
        try:
            with db:
                yield db
        finally:
            db.close()

    def save(self, session, replace_existing=False):
        # One SQLite transaction includes metadata and lossless best images.
        # Repeated staging must not reset a persisted retry deadline.
        frames = []
        for candidate in session.get_best_frame_per_camera().values():
            ok, encoded = cv2.imencode('.png', candidate.image)
            if not ok:
                raise OSError('Passage image encoding failed')
            frames.append({
                'camera_id': candidate.camera_id,
                'image': base64.b64encode(encoded).decode('ascii'),
                'timestamp': candidate.timestamp.isoformat(),
                'quality_score': candidate.quality_score,
                'detections': [asdict(d) for d in candidate.detections],
                'role': candidate.role, 'track_id': candidate.track_id,
                'source_id': candidate.source_id, 'frame_sequence': candidate.frame_sequence,
                'ocr_read': asdict(candidate.ocr_read) if isinstance(candidate.ocr_read, Validation) else None,
            })
        payload = {
            'session_id': session.session_id, 'event_id': session.event_id,
            'gate_id': session.gate_id, 'direction': session.direction,
            'cameras': session.cameras, 'created_at': session.created_at.isoformat(),
            'container_slot': session.container_slot, 'track_id': session.track_id,
            'confirmed_container_number': session.confirmed_container_number,
            'confirmed_iso_code': session.confirmed_iso_code,
            'confirmed_confidence': session.confirmed_confidence,
            'identity_conflict': session.identity_conflict, 'min_ocr_confidence': session.min_ocr_confidence,
            'min_ocr_confidence': session.min_ocr_confidence,
            'reads': [asdict(r) for r in session.ocr_candidates], 'frames': frames,
        }
        encoded = json.dumps(payload, default=lambda value: value.isoformat())
        with self.connection() as db:
            db.execute('INSERT OR IGNORE INTO passages(event_id,payload) VALUES(?,?)',
                       (session.event_id, encoded))
            if replace_existing:
                db.execute('UPDATE passages SET payload=? WHERE event_id=?', (encoded, session.event_id))

    def get(self, event_id):
        with self.connection() as db:
            row = db.execute('SELECT payload FROM passages WHERE event_id=?', (event_id,)).fetchone()
        return self.restore(json.loads(row['payload'])) if row else None

    def ready(self):
        with self.connection() as db:
            rows = db.execute('SELECT payload FROM passages WHERE retry_at<=? ORDER BY rowid LIMIT 10',
                              (time.time(),)).fetchall()
        return [self.restore(json.loads(row['payload'])) for row in rows]

    @staticmethod
    def restore(data):
        session = ContainerPassageSession(data['session_id'], data['gate_id'], data['direction'],
            data['cameras'], datetime.fromisoformat(data['created_at']),
            container_slot=data['container_slot'], track_id=data['track_id'],
            min_ocr_confidence=data.get('min_ocr_confidence', 0.6))
        session.event_id = data['event_id']
        session.finalized = True
        session.confirmed_container_number = data['confirmed_container_number']
        session.confirmed_iso_code = data['confirmed_iso_code']
        session.confirmed_confidence = data['confirmed_confidence']
        session.identity_conflict = data.get('identity_conflict', False)
        session.ocr_candidates = [Validation(**r) for r in data['reads']]
        for frame in data['frames']:
            image = cv2.imdecode(np.frombuffer(base64.b64decode(frame['image']), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise OSError('Unreadable passage image')
            detections = [Detection(**{**d, 'bbox': tuple(d['bbox']),
                'frame_timestamp': datetime.fromisoformat(d['frame_timestamp'])}) for d in frame['detections']]
            session.captured_frames[frame['camera_id']] = [CapturedFrameCandidate(
                frame['camera_id'], image, datetime.fromisoformat(frame['timestamp']),
                frame['quality_score'], detections, role=frame['role'], track_id=frame['track_id'],
                source_id=frame.get('source_id', ''), frame_sequence=frame.get('frame_sequence', 0),
                ocr_read=Validation(**frame['ocr_read']) if frame.get('ocr_read') else None)]
        return session

    def retry(self, event_id, error):
        with self.connection() as db:
            row = db.execute('SELECT attempts FROM passages WHERE event_id=?', (event_id,)).fetchone()
            if row:
                attempts = row['attempts'] + 1
                db.execute('UPDATE passages SET attempts=?,retry_at=?,error=? WHERE event_id=?',
                           (attempts, time.time() + min(2 ** min(attempts, 5), 30), error, event_id))

    def complete(self, event_id):
        with self.connection() as db:
            db.execute('DELETE FROM passages WHERE event_id=?', (event_id,))

    def pending(self):
        with self.connection() as db:
            return db.execute('SELECT COUNT(*) FROM passages').fetchone()[0]
