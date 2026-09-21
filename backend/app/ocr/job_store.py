"""Local transactional OCR evidence spool; separate from the event database.

PNG files are flushed and renamed before a metadata-only job is committed.
Completed evidence is retained until an operator applies a retention policy.
"""
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
import json
import hashlib
import os
from pathlib import Path
import sqlite3
import time
from uuid import uuid4

from app.domain import Detection, Frame, Validation


class OCRJobStore:
    def __init__(self, root, max_mb=10240, image_root=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'jobs.sqlite3'
        self.image_root = Path(image_root or self.root / 'images').resolve()
        self.image_root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_mb * 1024 * 1024
        with self.connection() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY, origin TEXT NOT NULL, sequence INTEGER NOT NULL,
                metadata TEXT NOT NULL, crop TEXT NOT NULL, image TEXT NOT NULL,
                payload_bytes INTEGER NOT NULL DEFAULT 0,
                state TEXT NOT NULL DEFAULT 'PENDING', validation TEXT, event_id TEXT,
                attempts INTEGER NOT NULL DEFAULT 0, retry_at REAL NOT NULL DEFAULT 0,
                error_type TEXT, UNIQUE(origin, sequence))''')
            db.execute('CREATE INDEX IF NOT EXISTS pending_jobs ON jobs(state, retry_at, id)')
        self._migrate_blobs()

    def image_path(self, relative):
        path = (self.image_root / relative).resolve()
        if not path.is_relative_to(self.image_root) or path == self.image_root:
            raise ValueError('Invalid OCR image path')
        return path

    def _write_image(self, relative, data):
        path = self.image_path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() == data:
                return
        temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
        try:
            with temporary.open('wb') as file:
                file.write(data)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _save_images(self, origin, sequence, crop, image, folder=None):
        if folder:
            paths = f'{folder}/candidate_{sequence}_crop.png', f'{folder}/candidate_{sequence}_frame.png'
        else:
            folder = hashlib.sha256(json.dumps([origin, sequence]).encode()).hexdigest()
            paths = f'{folder}/crop.png', f'{folder}/frame.png'
        self._write_image(paths[0], crop)
        self._write_image(paths[1], image)
        return paths

    def _migrate_blobs(self):
        # Original database remains recoverable if export fails: all metadata
        # changes roll back. Already exported files are reused on the next run.
        with self.connection() as db:
            columns = {row['name']: row['type'] for row in db.execute('PRAGMA table_info(jobs)')}
            if columns['crop'] != 'BLOB':
                return
            db.execute('BEGIN IMMEDIATE')
            db.execute('ALTER TABLE jobs ADD COLUMN payload_bytes INTEGER NOT NULL DEFAULT 0')
            for row in db.execute('SELECT id,origin,sequence,metadata,crop,image FROM jobs'):
                paths = self._save_images(row['origin'], row['sequence'], row['crop'], row['image'])
                size = len(row['crop']) + len(row['image']) + len(row['metadata'].encode())
                db.execute('UPDATE jobs SET crop=?, image=?, payload_bytes=? WHERE id=?', (*paths, size, row['id']))
            db.execute('''CREATE TABLE jobs_files (
                id INTEGER PRIMARY KEY, origin TEXT NOT NULL, sequence INTEGER NOT NULL,
                metadata TEXT NOT NULL, crop TEXT NOT NULL, image TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'PENDING', validation TEXT, event_id TEXT,
                attempts INTEGER NOT NULL DEFAULT 0, retry_at REAL NOT NULL DEFAULT 0,
                error_type TEXT, payload_bytes INTEGER NOT NULL DEFAULT 0,
                UNIQUE(origin, sequence))''')
            db.execute('INSERT INTO jobs_files SELECT * FROM jobs')
            db.execute('DROP TABLE jobs')
            db.execute('ALTER TABLE jobs_files RENAME TO jobs')
            db.execute('CREATE INDEX pending_jobs ON jobs(state, retry_at, id)')
        # Reclaim pages previously occupied by image blobs, only after commit.
        with self.connection() as db:
            db.execute('VACUUM')

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA synchronous=NORMAL')
        db.execute('''CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY, origin TEXT NOT NULL, sequence INTEGER NOT NULL,
            metadata TEXT NOT NULL, crop TEXT NOT NULL, image TEXT NOT NULL,
            payload_bytes INTEGER NOT NULL DEFAULT 0,
            state TEXT NOT NULL DEFAULT 'PENDING', validation TEXT, event_id TEXT,
            attempts INTEGER NOT NULL DEFAULT 0, retry_at REAL NOT NULL DEFAULT 0,
            error_type TEXT, UNIQUE(origin, sequence))''')
        db.execute('CREATE INDEX IF NOT EXISTS pending_jobs ON jobs(state, retry_at, id)')
        try:
            with db:
                yield db
        finally:
            db.close()

    def save(self, job, crop):
        import cv2
        def encode_img(image):
            ok, data = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if not ok:
                raise OSError('OCR evidence encoding failed')
            return data.tobytes()
        crop_data, image_data = encode_img(crop), encode_img(job.frame.image)
        import math
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        exposure = float(((gray > 10) & (gray < 245)).mean())
        quality = math.log1p(float(cv2.Laplacian(gray, cv2.CV_64F).var())) * exposure
        folder = f"{job.origin_key.replace(':', '_')}_{job.frame.sequence}"
        companion_paths = {}
        companion_bytes = 0
        if getattr(job, 'companion_frames', None):
            for cam_id, comp_img in job.companion_frames.items():
                if comp_img is not None:
                    try:
                        c_data = encode_img(comp_img)
                        c_path = f'{folder}/companion_{cam_id}.png'
                        self._write_image(c_path, c_data)
                        companion_paths[cam_id] = c_path
                        companion_bytes += len(c_data)
                    except Exception:
                        pass
        metadata = {
            'origin_key': job.origin_key, 'gate_id': job.gate_id, 'track_id': job.track_id,
            'event_id': getattr(job, 'event_id', None),
            'proof_quality': quality,
            'direction': job.direction, 'role': job.role, 'started_at': job.started_at.isoformat(),
            'slot_position': getattr(job, 'slot_position', 'FRONT'),
            'is_stationary': getattr(job, 'is_stationary', True),
            'entrance_complete': getattr(job, 'entrance_complete', True),
            'is_twin': getattr(job, 'is_twin', False),
            'roi': job.roi, 'detection': asdict(job.detection),
            'all_detections': [asdict(d) for d in job.all_detections],
            'frame': {'camera_id': job.frame.camera_id, 'timestamp': job.frame.timestamp.isoformat(),
                      'sequence': job.frame.sequence, 'source_id': job.frame.source_id},
            'companion_images': companion_paths}
        encoded = json.dumps(metadata, default=lambda value: value.isoformat())
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            existing = db.execute('SELECT id FROM jobs WHERE origin=? AND sequence=?',
                                  (job.origin_key, job.frame.sequence)).fetchone()
            if existing:
                return existing['id']
            size = db.execute('SELECT COALESCE(SUM(payload_bytes),0) FROM jobs').fetchone()[0]
            payload_bytes = len(crop_data) + len(image_data) + len(encoded.encode()) + companion_bytes
            if size + payload_bytes > self.max_bytes:
                raise OSError('OCR spool capacity reached')
            paths = self._save_images(job.origin_key, job.frame.sequence, crop_data, image_data, folder=folder)
            return db.execute('INSERT INTO jobs(origin,sequence,metadata,crop,image,payload_bytes) VALUES(?,?,?,?,?,?)',
                              (job.origin_key, job.frame.sequence, encoded, *paths, payload_bytes)).lastrowid

    def next_id(self):
        with self.connection() as db:
            row = db.execute("SELECT id FROM jobs WHERE state='PENDING' AND retry_at<=? ORDER BY id LIMIT 1",
                             (time.time(),)).fetchone()
            return row['id'] if row else None

    def load(self, job_id):
        import cv2
        import numpy as np
        from app.runtime import OCRJob
        with self.connection() as db:
            row = db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
        data = json.loads(row['metadata'])
        def detection(value):
            return Detection(**{**value, 'frame_timestamp': datetime.fromisoformat(value['frame_timestamp']),
                                'bbox': tuple(value['bbox'])})
        def decode(relative):
            blob = self.image_path(relative).read_bytes()
            result = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
            if result is None:
                raise ValueError('Unreadable persisted OCR image')
            return result
        comp_frames = {}
        for cam_id, rel in data.get('companion_images', {}).items():
            try:
                comp_frames[cam_id] = decode(rel)
            except Exception:
                pass
        frame = Frame(**{**data['frame'], 'timestamp': datetime.fromisoformat(data['frame']['timestamp']),
                         'image': decode(row['image'])})
        job = OCRJob(data['origin_key'], data['gate_id'], data['track_id'], detection(data['detection']),
                     frame, data['direction'], datetime.fromisoformat(data['started_at']), tuple(data['roi']),
                     tuple(detection(d) for d in data['all_detections']), data['role'],
                     companion_frames=comp_frames, slot_position=data.get('slot_position', 'FRONT'),
                     event_id=data.get('event_id'),
                     is_stationary=data.get('is_stationary', True),
                     entrance_complete=data.get('entrance_complete', True),
                     is_twin=data.get('is_twin', False))
        validation = Validation(**json.loads(row['validation'])) if row['validation'] else None
        return job, decode(row['crop']), validation

    def save_validation(self, job_id, validation):
        with self.connection() as db:
            db.execute('UPDATE jobs SET validation=? WHERE id=?', (json.dumps(asdict(validation)), job_id))

    def best_proof(self, origin, through_id):
        with self.connection() as db:
            rows = db.execute('SELECT id, metadata, validation FROM jobs WHERE origin=? AND id<=? ORDER BY id',
                              (origin, through_id)).fetchall()
        def score(row):
            meta = json.loads(row['metadata']) if row['metadata'] else {}
            val = json.loads(row['validation']) if row['validation'] else {}
            q = float(meta.get('proof_quality', 0))
            if val.get('valid_check_digit'):
                q += 1000.0 + (float(val.get('confidence', 0)) * 100.0)
            return q

        rows.sort(key=score, reverse=True)
        for row in rows:
            try:
                return self.load(row['id'])[0]
            except (OSError, ValueError):
                continue
        return None

    def evidence(self, origin, limit):
        from app.events.event_manager import Evidence
        with self.connection() as db:
            rows = db.execute("SELECT * FROM jobs WHERE origin=? AND state='DONE' ORDER BY id DESC LIMIT ?",
                              (origin, limit)).fetchall()
        rows = list(reversed(rows))
        return Evidence(reads=[Validation(**json.loads(row['validation'])) for row in rows if row['validation']],
                        event_id=next((row['event_id'] for row in rows if row['event_id']), None),
                        last_sequence=rows[-1]['sequence'] if rows else None)

    def complete(self, job_id, event_id=None):
        with self.connection() as db:
            db.execute("UPDATE jobs SET state='DONE', event_id=?, error_type=NULL, retry_at=? WHERE id=?",
                       (event_id, time.time(), job_id))

    def cancel_pending(self, origin, event_id=None):
        with self.connection() as db:
            db.execute("UPDATE jobs SET state='DONE', event_id=?, error_type=NULL, retry_at=? WHERE origin=? AND state='PENDING'",
                       (event_id, time.time(), origin))

    def clean_completed(self, max_age_seconds=10, protected_origins=()):
        cutoff = time.time() - max_age_seconds
        with self.connection() as db:
            rows = db.execute(
                "SELECT id, origin, crop, image, metadata FROM jobs j WHERE state='DONE' AND retry_at<=? "
                "AND NOT EXISTS (SELECT 1 FROM jobs p WHERE p.origin=j.origin AND p.state='PENDING')",
                (cutoff,)
            ).fetchall()
        for row in rows:
            if row['origin'] in protected_origins:
                continue
            try:
                img_p = self.image_path(row['image'])
                crop_p = self.image_path(row['crop'])
                img_p.unlink(missing_ok=True)
                crop_p.unlink(missing_ok=True)
                meta = json.loads(row['metadata']) if row['metadata'] else {}
                for comp_rel in meta.get('companion_images', {}).values():
                    try:
                        self.image_path(comp_rel).unlink(missing_ok=True)
                    except Exception:
                        pass
                parent = img_p.parent
                if parent != self.image_root and parent.is_relative_to(self.image_root):
                    try:
                        parent.rmdir()
                    except OSError:
                        pass
            except Exception:
                pass

    def retry(self, job_id, error_type):
        with self.connection() as db:
            db.execute('UPDATE jobs SET attempts=attempts+1, retry_at=?, error_type=? WHERE id=?',
                       (time.time() + 5, error_type, job_id))

    def status(self):
        with self.connection() as db:
            row = db.execute("SELECT COUNT(*) total, SUM(state='PENDING') pending, SUM(error_type IS NOT NULL) failed FROM jobs").fetchone()
        return {key: row[key] or 0 for key in ('total', 'pending', 'failed')}
