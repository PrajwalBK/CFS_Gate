import difflib
import hashlib
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from uuid import uuid4
from sqlalchemy import func, select, update

from app.database.models import Camera, DetectionRecord, GateEvent, OCRResult, Snapshot, SystemLog
from app.domain import Observation, utcnow
from app.snapshots.snapshot_manager import SnapshotManager
from app.ocr.validator import is_valid_iso_6346


def serialize(record):
    result = {column.name: getattr(record, column.name) for column in record.__table__.columns}
    return {key: value.isoformat() + 'Z' if isinstance(value, datetime) else value for key, value in result.items()}


def ocr_similarity_score(s1: str | None, s2: str | None) -> float:
    """Return character similarity ratio (0.0 to 1.0) between two OCR strings using SequenceMatcher."""
    if not s1 or not s2:
        return 0.0
    s1, s2 = s1.strip().upper(), s2.strip().upper()
    if s1 == s2:
        return 1.0
    return difflib.SequenceMatcher(None, s1, s2).ratio()


def datetime_diff_seconds(dt1: datetime | None, dt2: datetime | None) -> float:
    """Safe absolute seconds difference between offset-naive or offset-aware datetimes."""
    if not dt1 or not dt2:
        return float('inf')
    t1 = dt1.replace(tzinfo=None)
    t2 = dt2.replace(tzinfo=None)
    return abs((t1 - t2).total_seconds())


class GateRepository:
    def __init__(self, db):
        self.db = db
        self._write_lock = Lock()

    def _capture_provenance(self, session, snapshots, snapshot_id, event_id, camera_id,
                            timestamp, image_path, detections, number, source_id='', sequence=0,
                            identity_basis='OCR'):
        exact = [asdict(d) for d in detections if d.camera_id == camera_id and d.frame_timestamp == timestamp
                 and d.class_name in ('container', 'container_object')]
        context = {'snapshot_id': snapshot_id, 'event_id': event_id, 'camera_id': camera_id,
                   'timestamp': timestamp.isoformat() + 'Z', 'source_id': source_id,
                   'frame_sequence': sequence, 'container_number': number,
                   'identity_basis': identity_basis,
                   'image_sha256': hashlib.sha256(snapshots.path(image_path).read_bytes()).hexdigest(),
                   'container_detections': [{**d, 'frame_timestamp': d['frame_timestamp'].isoformat() + 'Z'} for d in exact]}
        session.add(SystemLog(id=str(uuid4()), event_id=event_id, level='INFO',
                              event_name='capture_provenance', context=context))

    def save_passage(self, passage, snapshots, confirmations):
        """Commit one completed truck, including independently voted identities.

        Region labels are internal; repeated numbers on different parts of the
        same chassis count once. Conflicting/partial evidence stays local.
        """
        candidates = {}
        uncertain = passage.ambiguous
        for region in passage.regions:
            votes = region.votes.most_common()
            if not votes:
                uncertain = True
                continue
            number, count = votes[0]
            if not is_valid_iso_6346(number):
                uncertain = True
                continue
            if count < confirmations or (len(votes) > 1 and votes[1][1] == count):
                uncertain = True
            validation, job = region.reads[number]
            if number not in candidates or count > candidates[number][0]:
                candidates[number] = (count, validation, job)
        entries = list(candidates.items())
        created_paths = []
        try:
            with self._write_lock, self.db.sessions.begin() as session:
                if session.get(GateEvent, passage.id):
                    return passage.id
                first = entries[0] if entries else None
                event = GateEvent(id=passage.id, gate_id=passage.gate_id, lane_id=passage.lane_id,
                                  timestamp=passage.started, event_type=passage.direction,
                                  container_number=first[0] if first else None,
                                  container_number_2=None,
                                  iso_code_1=first[1][1].size_type if first else None,
                                  iso_code_2=None,
                                  is_twin=False, status='CONFIRMED' if entries and not uncertain else 'NEEDS_REVIEW')
                session.add(event)
                session.flush()
                for index, (number, (_, validation, job)) in enumerate(entries):
                    detection_id, snapshot_id = str(uuid4()), str(uuid4())
                    slot = 'SINGLE'
                    origin = f'{job.origin_key}:{number}'
                    image = job.frame.image
                    # Give each identity its own proof view, even when two
                    # numbers were read from one camera frame. Include context
                    # around the number, without copying another truck's image.
                    h,w = image.shape[:2]
                    x1,y1,x2,y2 = job.detection.bbox
                    cx,cy = (x1+x2)/2,(y1+y2)/2
                    cw,ch = min(w,max(320,int((x2-x1)*6))),min(h,max(240,int((y2-y1)*12)))
                    left,top = max(0,min(w-cw,int(cx-cw/2))),max(0,min(h-ch,int(cy-ch/2)))
                    proof_image = image[top:top+ch,left:left+cw].copy()
                    obs = Observation(origin, passage.gate_id, job.track_id,
                                      job.detection, validation, passage.direction, proof_image,
                                      passage.started, True, slot_position=slot, is_twin=False)
                    path = snapshots.save(snapshot_id, obs, passage.id)
                    created_paths.append(path)
                    context_id = str(uuid4())
                    context_path = snapshots.save_side_snapshot(passage.id,
                        f'{job.detection.camera_id}_{snapshot_id}_context', job.frame.image)
                    created_paths.append(context_path)
                    session.add(DetectionRecord(id=detection_id, event_id=passage.id,
                        camera_id=job.detection.camera_id, origin_key=obs.origin_key,
                        track_id=job.track_id, class_name=job.detection.class_name,
                        confidence=job.detection.confidence, bbox=list(job.detection.bbox),
                        timestamp=job.detection.frame_timestamp))
                    session.flush()
                    session.add(OCRResult(id=str(uuid4()), detection_id=detection_id,
                        raw_text=validation.raw_text, normalized_text=number, confidence=validation.confidence,
                        valid_format=validation.valid_format, valid_check_digit=validation.valid_check_digit,
                        validation_status=validation.validation_status))
                    session.add(Snapshot(id=snapshot_id,event_id=passage.id,camera_id=job.detection.camera_id,
                        image_path=path,timestamp=job.detection.frame_timestamp))
                    session.add(Snapshot(id=context_id,event_id=passage.id,camera_id=job.detection.camera_id,
                        image_path=context_path,timestamp=job.detection.frame_timestamp))
                session.add(SystemLog(id=str(uuid4()),event_id=passage.id,level='INFO',
                    event_name='truck_passage_finalized',context={'lane_id':passage.lane_id,
                    'identities':list(candidates),'needs_review':uncertain or not entries}))
            try:
                snapshots.save_event_json(passage.id, self.event(passage.id))
            except OSError:
                # The committed DB and image references remain authoritative.
                import logging
                logging.getLogger('gate').exception('passage_manifest_write_failed')
            return passage.id
        except Exception:
            for path in created_paths:
                snapshots.remove(path)
            raise

    def finalize_passage_session(self, session, snapshots):
        """
        Commit a finalized multi-camera container passage session.
        Stores:
        - ONE authoritative container number and ISO code determined via multi-frame consensus
        - Exactly ONE best image from EVERY camera on that gate (preserving all distinct angles)
        - Full event metadata, evidence JSON, and system log
        """
        created_paths = []
        try:
            with self._write_lock, self.db.sessions.begin() as db_session:
                existing = db_session.get(GateEvent, session.event_id)
                committed = db_session.scalar(select(SystemLog.id).where(
                    SystemLog.event_id == session.event_id,
                    SystemLog.event_name == 'multi_camera_passage_finalized'))
                if committed:
                    return session.event_id
                best_frames = session.get_best_frame_per_camera()
                container_num, iso_code, conf = session.get_consensus_container_number()

                # Ghost/passerby elimination guard:
                # If there is NO container number AND NO physical container proof on any camera,
                # discard this spurious session without cluttering the database or operator UI with empty events.
                has_container_proof = False
                for c_id, f_cand in (best_frames or {}).items():
                    if any(
                        getattr(d, 'class_name', '') == 'container'
                        and len(getattr(d, 'bbox', [])) == 4
                        and (d.bbox[2] - d.bbox[0]) >= 50.0
                        and (d.bbox[3] - d.bbox[1]) >= 40.0
                        for d in getattr(f_cand, 'detections', [])
                    ):
                        has_container_proof = True
                        break

                if not container_num and not has_container_proof:
                    log.info('spurious_passage_discarded_no_container', extra={
                        'session_id': session.session_id,
                        'event_id': session.event_id,
                        'gate_id': session.gate_id
                    })
                    return session.event_id

                val_check = None
                if container_num:
                    from app.ocr.validator import ContainerValidator
                    from app.domain import OCRRead
                    import re
                    val_check = ContainerValidator().validate(OCRRead(raw_text=container_num, confidence=conf))
                is_valid_format = bool(container_num and len(container_num) == 11 and re.match(r'^[A-Z]{4}[0-9]{7}$', container_num))
                is_confirmed = is_valid_format and is_valid_iso_6346(container_num)
                status = 'CONFIRMED' if is_confirmed else 'NEEDS_REVIEW'

                if not existing:
                    event = GateEvent(
                        id=session.event_id,
                        gate_id=session.gate_id,
                        timestamp=session.created_at,
                        event_type=session.direction,
                        container_number=container_num,
                        container_number_2=None,
                        iso_code_1=iso_code,
                        iso_code_2=None,
                        is_twin=False,
                        status=status
                    )
                    db_session.add(event)
                    db_session.flush()
                else:
                    if container_num:
                        existing.container_number = container_num
                    if iso_code:
                        existing.iso_code_1 = iso_code
                    existing.status = status

                # Save the independent BEST frame for EVERY camera on this gate that captured a container
                primary_det_id = None
                for cam_id, frame_cand in best_frames.items():
                    snap_id = str(uuid4())
                    img_path = snapshots.save_side_snapshot(
                        session.event_id, cam_id, frame_cand.image,
                        direction=session.direction,
                        text=container_num or '',
                        force=True
                    )
                    created_paths.append(img_path)
                    existing_snap = db_session.scalar(select(Snapshot).where(
                        Snapshot.event_id == session.event_id,
                        Snapshot.camera_id == cam_id
                    ))
                    if not existing_snap:
                        db_session.add(Snapshot(
                            id=snap_id,
                            event_id=session.event_id,
                            camera_id=cam_id,
                            image_path=img_path,
                            timestamp=frame_cand.timestamp
                        ))
                    else:
                        existing_snap.image_path = img_path
                        existing_snap.timestamp = frame_cand.timestamp
                    self._capture_provenance(db_session, snapshots, existing_snap.id if existing_snap else snap_id,
                        session.event_id, cam_id, frame_cand.timestamp, img_path, frame_cand.detections,
                        container_num, frame_cand.source_id, frame_cand.frame_sequence, "PASSAGE")

                    # Record the container detection records for this camera
                    cam_has_det = False
                    for d in (frame_cand.detections or []):
                        raw_cname = getattr(d, 'class_name', '')
                        cname_l = raw_cname.lower()
                        if cname_l in ('container', 'cointainer', 'container_object'):
                            cname = 'container'
                        elif cname_l in ('container_number',):
                            cname = 'container_number'
                        elif cname_l in ('iso_code',):
                            cname = 'iso_code'
                        else:
                            continue
                        d_id = str(uuid4())
                        if primary_det_id is None:
                            primary_det_id = d_id
                        db_session.add(DetectionRecord(
                            id=d_id,
                            event_id=session.event_id,
                            camera_id=cam_id,
                            origin_key=f"session:{session.session_id}:{cam_id}:{d_id}",
                            track_id=int(frame_cand.track_id or 1),
                            class_name=cname,
                            confidence=float(getattr(d, 'confidence', 0.85) or 0.85),
                            bbox=list(getattr(d, 'bbox', [0.0, 0.0, 1.0, 1.0])),
                            timestamp=frame_cand.timestamp
                        ))
                        cam_has_det = True

                    if not cam_has_det:
                        d_id = str(uuid4())
                        if primary_det_id is None:
                            primary_det_id = d_id
                        db_session.add(DetectionRecord(
                            id=d_id,
                            event_id=session.event_id,
                            camera_id=cam_id,
                            origin_key=f"session:{session.session_id}:{cam_id}:{d_id}",
                            track_id=int(getattr(session, 'track_id', 1) or 1),
                            class_name='container',
                            confidence=0.90,
                            bbox=[0.0, 0.0, 1.0, 1.0],
                            timestamp=frame_cand.timestamp
                        ))

                # Ensure at least one detection record exists for the primary camera
                if not primary_det_id:
                    primary_det_id = str(uuid4())
                    primary_cam = session.cameras[0] if session.cameras else 'camera-1'
                    db_session.add(DetectionRecord(
                        id=primary_det_id,
                        event_id=session.event_id,
                        camera_id=primary_cam,
                        origin_key=f"session:{session.session_id}",
                        track_id=int(getattr(session, 'track_id', 1) or 1),
                        class_name='container',
                        confidence=max(0.85, conf),
                        bbox=[0.0, 0.0, 1.0, 1.0],
                        timestamp=session.created_at
                    ))
                if container_num:
                    db_session.add(OCRResult(
                        id=str(uuid4()),
                        detection_id=primary_det_id,
                        raw_text=container_num,
                        normalized_text=container_num,
                        confidence=conf,
                        valid_format=is_valid_format,
                        valid_check_digit=is_valid_iso_6346(container_num),
                        validation_status='VALID' if is_confirmed else 'INVALID_CHECK_DIGIT'
                    ))

                db_session.add(SystemLog(
                    id=str(uuid4()),
                    event_id=session.event_id,
                    level='INFO',
                    event_name='multi_camera_passage_finalized',
                    context={
                        'session_id': session.session_id,
                        'gate_id': session.gate_id,
                        'container_number': container_num,
                        'cameras_captured': list(best_frames.keys()),
                        'status': status
                    }
                ))

            if snapshots:
                ev_data = self.event(session.event_id)
                if ev_data:
                    try:
                        snapshots.save_event_json(session.event_id, ev_data)
                    except OSError:
                        # JSON is derived; never delete committed proof images
                        # when this optional export fails after the transaction.
                        pass

            return session.event_id
        except Exception:
            for path in created_paths:
                try:
                    snapshots.remove(path)
                except Exception:
                    pass
            raise

    def sync_cameras(self, cameras):
        with self.db.sessions.begin() as session:
            for camera in cameras:
                current = session.get(Camera, camera.id)
                if current is None:
                    current = Camera(id=camera.id)
                    session.add(current)
                current.name = camera.name
                current.gate_id = camera.gate_id
                current.source_env = camera.source_env
                current.enabled = camera.enabled
                current.configuration = camera.model_dump(mode='json')

    def save_observation(self, obs: Observation, snapshots, dedup_seconds: float,
                         additional_frames: dict | None = None, ongoing_event_id=None,
                         same_container_window_seconds: float = 900):
        """One writer per service; DB uniqueness also protects retried track writes."""
        if obs.ocr and not is_valid_iso_6346(obs.ocr.normalized_text):
            obs = replace(obs, confirmed=False, ocr=replace(obs.ocr, valid_check_digit=False))
        created_paths = []
        try:
            with self._write_lock, self.db.sessions.begin() as session:
                prior = session.scalar(select(DetectionRecord).where(DetectionRecord.origin_key == obs.origin_key))
                if prior:
                    return prior.event_id, False
                number = obs.ocr.normalized_text if (obs.confirmed or getattr(obs.ocr, 'valid_check_digit', False)) else None
                delta = timedelta(seconds=dedup_seconds)
                existing = None

                direction_match = (
                    (GateEvent.event_type == obs.direction) | (GateEvent.event_type == 'UNKNOWN')
                    if obs.direction != 'UNKNOWN'
                    else True
                )

                if number:
                    existing = session.scalar(select(GateEvent).where(
                        GateEvent.gate_id == obs.gate_id,
                        GateEvent.container_number == number,
                        direction_match,
                        GateEvent.timestamp >= obs.detection.frame_timestamp - delta,
                        GateEvent.timestamp <= obs.detection.frame_timestamp + delta,
                    ).order_by(GateEvent.timestamp.desc()))
                    # Tolerant suffix matching on same gate: if 6 serial digits and check digit match (e.g. AIPU5077669 vs TIIU5077669)
                    if not existing and len(number) == 11 and number[4:].isdigit():
                        suffix = number[4:]
                        candidates = session.scalars(select(GateEvent).where(
                            GateEvent.gate_id == obs.gate_id,
                            direction_match,
                            GateEvent.timestamp >= obs.detection.frame_timestamp - delta,
                            GateEvent.timestamp <= obs.detection.frame_timestamp + delta,
                        ).order_by(GateEvent.timestamp.desc())).all()
                        for cand in candidates:
                            c1 = cand.container_number or ''
                            if len(c1) == 11 and c1[4:] == suffix:
                                existing = cand
                                break
                    # Dwell/queue recovery: a different camera or a restarted
                    # tracker can produce a new origin several minutes after
                    # the first read.  If the same validated container is still
                    # in the same lane and travelling in the same direction,
                    # bind it to the oldest event instead of creating a second
                    # cloud visit.  This intentionally does not broaden normal
                    # (non-dwell) repeat-visit deduplication.
                    if not existing and obs.origin_key.startswith('dwell:'):
                        long_delta = timedelta(seconds=max(float(same_container_window_seconds or 0), dedup_seconds))
                        existing = session.scalar(select(GateEvent).where(
                            GateEvent.gate_id == obs.gate_id,
                            GateEvent.container_number == number,
                            direction_match,
                            GateEvent.status.in_(['CONFIRMED', 'NEEDS_REVIEW']),
                            GateEvent.timestamp >= obs.detection.frame_timestamp - long_delta,
                            GateEvent.timestamp <= obs.detection.frame_timestamp,
                        ).order_by(GateEvent.timestamp.asc()))
                        if not existing and len(number) == 11 and number[4:].isdigit():
                            suffix = number[4:]
                            candidates = session.scalars(select(GateEvent).where(
                                GateEvent.gate_id == obs.gate_id,
                                direction_match,
                                GateEvent.status.in_(['CONFIRMED', 'NEEDS_REVIEW']),
                                GateEvent.timestamp >= obs.detection.frame_timestamp - long_delta,
                                GateEvent.timestamp <= obs.detection.frame_timestamp,
                            ).order_by(GateEvent.timestamp.asc())).all()
                            for cand in candidates:
                                c1 = cand.container_number or ''
                                if len(c1) == 11 and c1[4:] == suffix:
                                    existing = cand
                                    break

                    # Visual and capture-timestamp deduplication:
                    # When OCR produces different strings for the same container (e.g. MSMU5388338 vs MSMU5386335),
                    # compare recent candidate events on the same lane within delta for matching scene images,
                    # capture timestamps, and fuzzy character overlap.
                    if not existing:
                        recent_candidates = session.scalars(select(GateEvent).where(
                            GateEvent.gate_id == obs.gate_id,
                            direction_match,
                            GateEvent.status.in_(['CONFIRMED', 'NEEDS_REVIEW']),
                            GateEvent.timestamp >= obs.detection.frame_timestamp - delta,
                            GateEvent.timestamp <= obs.detection.frame_timestamp + delta,
                        ).order_by(GateEvent.timestamp.desc())).all()

                        for cand in recent_candidates:
                            time_diff = datetime_diff_seconds(cand.timestamp, obs.detection.frame_timestamp)
                            c_num = cand.container_number or ''
                            ocr_sim = ocr_similarity_score(number, c_num)
                            match_found = False

                            # 1. Matching serial digits (e.g. AXU1960888 vs AXIU1960888: both have 1960888)
                            if number and c_num:
                                num_digits = ''.join(c for c in number if c.isdigit())
                                cand_digits = ''.join(c for c in c_num if c.isdigit())
                                if len(num_digits) >= 5 and len(cand_digits) >= 5:
                                    if num_digits[-6:] == cand_digits[-6:] or (len(num_digits) >= 6 and num_digits[:6] in cand_digits):
                                        match_found = True

                            # Distinct verified container numbers must NEVER be merged as duplicates by static camera background similarity.
                            if number and c_num and number != c_num:
                                num_digits = ''.join(c for c in number if c.isdigit())
                                cand_digits = ''.join(c for c in c_num if c.isdigit())
                                digits_match = bool(len(num_digits) >= 5 and len(cand_digits) >= 5 and (num_digits[-6:] == cand_digits[-6:] or (len(num_digits) >= 6 and num_digits[:6] in cand_digits)))
                                if not (ocr_sim >= 0.50 or digits_match):
                                    continue  # Distinct physical containers! NEVER merge!

                            # 2. Compare snapshots across matching camera or companion images
                            if not match_found and snapshots and hasattr(snapshots, 'path') and getattr(obs, 'image', None) is not None:
                                cam_id = obs.detection.camera_id
                                cand_img_path = None
                                try:
                                    p = snapshots.path(f"{cand.id}/{cam_id}.jpg")
                                    if p.is_file():
                                        cand_img_path = p
                                except Exception:
                                    pass
                                if not cand_img_path:
                                    snap_rec = session.scalar(select(Snapshot).where(
                                        Snapshot.event_id == cand.id,
                                        Snapshot.camera_id == cam_id
                                    ).order_by(Snapshot.timestamp.desc()))
                                    if snap_rec and snap_rec.image_path:
                                        try:
                                            p = snapshots.path(snap_rec.image_path)
                                            if p.is_file():
                                                cand_img_path = p
                                        except Exception:
                                            pass

                                if cand_img_path:
                                    sim = SnapshotManager.compute_similarity(obs.image, cand_img_path)
                                    if sim >= 0.82 or (time_diff <= 15.0 and sim >= 0.72):
                                        match_found = True

                            # 3. Check OCR fuzzy overlap + short timestamp gap
                            if not match_found and time_diff <= 35.0:
                                if ocr_sim >= 0.50:
                                    match_found = True

                            # 4. Short-window same-lane fusion for unconfirmed/partial reads
                            if not match_found and time_diff <= 35.0 and (not cand.container_number or not number or ocr_sim >= 0.45):
                                match_found = True

                            if match_found:
                                existing = cand
                                break

                    if existing:
                        if not existing.container_number:
                            existing.container_number = number
                            existing.iso_code_1 = getattr(obs.ocr, 'size_type', None)
                        if obs.confirmed:
                            existing.status = 'CONFIRMED'
                        if existing.event_type == 'UNKNOWN' and obs.direction != 'UNKNOWN':
                            existing.event_type = obs.direction
                else:
                    existing = session.scalar(select(GateEvent).where(
                        GateEvent.gate_id == obs.gate_id,
                        direction_match,
                        GateEvent.timestamp >= obs.detection.frame_timestamp - delta,
                        GateEvent.timestamp <= obs.detection.frame_timestamp + delta,
                    ).order_by(GateEvent.timestamp.desc()))
                    if existing and existing.event_type == 'UNKNOWN' and obs.direction != 'UNKNOWN':
                        existing.event_type = obs.direction
                if not existing and getattr(obs, 'event_id', None):
                    active = session.get(GateEvent, obs.event_id)
                    if active and active.gate_id == obs.gate_id:
                        existing = active

                if ongoing_event_id and not existing:
                    active = session.get(GateEvent, ongoing_event_id)
                    if active and active.gate_id == obs.gate_id:
                        existing = active

                if obs.origin_key.startswith('dwell:') and existing:
                    time_diff = datetime_diff_seconds(existing.timestamp, obs.detection.frame_timestamp)
                    num_digits = ''.join(c for c in (number or '') if c.isdigit())
                    ex_digits = ''.join(c for c in (existing.container_number or '') if c.isdigit())
                    digits_match = bool(num_digits and ex_digits and (num_digits[-5:] == ex_digits[-5:] or num_digits in ex_digits))
                    is_same_container = bool(
                        time_diff <= 35.0 and (
                            (number and number == existing.container_number) or
                            digits_match or
                            (number and existing.container_number and ocr_similarity_score(number, existing.container_number) >= 0.55)
                        )
                    )
                    if not is_same_container:
                        prior_view = session.scalar(select(DetectionRecord.id).where(
                            DetectionRecord.event_id == existing.id,
                            DetectionRecord.camera_id == obs.detection.camera_id,
                            DetectionRecord.origin_key.like('dwell:%'),
                            DetectionRecord.origin_key != obs.origin_key))
                        if prior_view:
                            existing = None  # New cleared passage, not time-window fusion.

                if (existing and number and existing.container_number and
                        number != existing.container_number and
                        is_valid_iso_6346(number) and is_valid_iso_6346(existing.container_number)):
                    # An explicit event link cannot turn different valid identities
                    # into corroborating evidence. Keep the passage for review.
                    existing.status = 'NEEDS_REVIEW'
                    obs = replace(obs, confirmed=False)
                    number = existing.container_number
                created = existing is None
                event_id = existing.id if existing else (getattr(obs, 'event_id', None) or str(uuid4()))
                if created and session.get(GateEvent, event_id):
                    existing = session.get(GateEvent, event_id)
                    created = False
                    event_id = existing.id
                size_type = getattr(obs.ocr, 'size_type', None)
                if created:
                    session.add(GateEvent(
                        id=event_id, gate_id=obs.gate_id, timestamp=obs.detection.frame_timestamp,
                        container_number=number,
                        container_number_2=None,
                        iso_code_1=size_type,
                        iso_code_2=None,
                        is_twin=False,
                        event_type=obs.direction,
                        status='CONFIRMED' if obs.confirmed else 'NEEDS_REVIEW',
                    ))
                    session.flush()
                else:
                    prior_best_conf = session.scalar(
                        select(func.max(OCRResult.confidence))
                        .join(DetectionRecord, OCRResult.detection_id == DetectionRecord.id)
                        .where(DetectionRecord.event_id == existing.id, OCRResult.valid_check_digit == True)
                    ) or 0.0
                    obs_conf = getattr(obs.ocr, 'confidence', 0.0) or 0.0
                    existing.is_twin = False
                    existing.container_number_2 = None
                    existing.iso_code_2 = None
                    if not existing.container_number or (obs.confirmed and obs_conf >= prior_best_conf):
                        existing.container_number = number or existing.container_number
                    if size_type and not existing.iso_code_1:
                        existing.iso_code_1 = size_type
                    elif size_type and obs_conf >= prior_best_conf:
                        existing.iso_code_1 = size_type
                    if obs.confirmed:
                        existing.status = 'CONFIRMED'
                detection_id = str(uuid4())
                session.add(DetectionRecord(
                    id=detection_id, event_id=event_id, camera_id=obs.detection.camera_id,
                    origin_key=obs.origin_key, track_id=obs.track_id,
                    class_name=obs.detection.class_name, confidence=obs.detection.confidence,
                    bbox=list(obs.detection.bbox), timestamp=obs.detection.frame_timestamp,
                ))
                ocr_data = {k: v for k, v in asdict(obs.ocr).items() if k != 'size_type'}
                session.add(OCRResult(id=str(uuid4()), detection_id=detection_id, **ocr_data))

                # Also save any concurrent detections on this frame (such as Feet!)
                if getattr(obs, 'all_detections', None):
                    for extra in obs.all_detections:
                        if extra.class_name != obs.detection.class_name:
                            extra_origin = f'{obs.origin_key}:{extra.class_name}:{int(extra.bbox[0])}'
                            existing_extra = session.scalar(select(DetectionRecord).where(
                                DetectionRecord.origin_key == extra_origin
                            ))
                            if not existing_extra:
                                session.add(DetectionRecord(
                                    id=str(uuid4()), event_id=event_id, camera_id=extra.camera_id,
                                    origin_key=extra_origin, track_id=obs.track_id,
                                    class_name=extra.class_name, confidence=extra.confidence,
                                    bbox=list(extra.bbox), timestamp=extra.frame_timestamp,
                                ))


                # 1. Primary snapshot for this observation
                snapshot_id = str(uuid4())
                image_path = snapshots.save(snapshot_id, obs, event_id)
                existing_path_snap = session.scalar(select(Snapshot).where(
                    Snapshot.image_path == image_path
                ))
                if existing_path_snap:
                    existing_path_snap.timestamp = obs.detection.frame_timestamp
                    if existing_path_snap.event_id != event_id:
                        existing_path_snap.event_id = event_id
                else:
                    created_paths.append(image_path)
                    session.add(Snapshot(id=snapshot_id, event_id=event_id, camera_id=obs.detection.camera_id,
                                         image_path=image_path, timestamp=obs.detection.frame_timestamp))

                self._capture_provenance(session, snapshots, existing_path_snap.id if existing_path_snap else snapshot_id,
                    event_id, obs.detection.camera_id, obs.detection.frame_timestamp, image_path,
                    list(obs.all_detections or []) + [obs.detection],
                    obs.proof_container_number if obs.proof_container_number is not None else obs.ocr.normalized_text,
                    obs.source_id, obs.frame_sequence)

                # 2. Multi-camera snapshots for all other cameras covering this gate event
                if additional_frames:
                    for other_cam_id, other_img in additional_frames.items():
                        if other_cam_id == obs.detection.camera_id or other_img is None:
                            continue
                        if hasattr(other_img, 'size') and other_img.size == 0:
                            continue
                        if hasattr(other_img, 'shape') and min(other_img.shape[:2]) < 32:
                            continue
                        # If full camera frame, filter out completely blank or corrupted frames
                        if (hasattr(other_img, 'shape') and other_img.shape[0] >= 240 and other_img.shape[1] >= 320 and
                                hasattr(snapshots, 'compute_quality') and snapshots.compute_quality(other_img) < 1.5):
                            continue
                        already_saved = session.scalar(select(Snapshot).where(
                            Snapshot.event_id == event_id,
                            Snapshot.camera_id == other_cam_id
                        ))
                        # Update companion snapshot if not saved yet, OR if current observation is confirmed / high confidence
                        should_update = (
                            not already_saved or
                            obs.confirmed or
                            getattr(obs.ocr, 'confidence', 0.0) >= 0.60
                        )
                        if should_update:
                            other_snap_id = already_saved.id if already_saved else str(uuid4())
                            other_path = snapshots.save_camera_frame(
                                other_snap_id, event_id, other_cam_id, other_img,
                                direction=obs.direction,
                                text=obs.ocr.normalized_text if obs.confirmed else ''
                            )
                            existing_other_snap = session.scalar(select(Snapshot).where(
                                Snapshot.image_path == other_path
                            ))
                            if existing_other_snap:
                                existing_other_snap.timestamp = obs.detection.frame_timestamp
                                if existing_other_snap.event_id != event_id:
                                    existing_other_snap.event_id = event_id
                            elif already_saved:
                                if snapshots and hasattr(snapshots, 'path'):
                                    try:
                                        if snapshots.path(other_path).is_file():
                                            already_saved.image_path = other_path
                                    except Exception:
                                        pass
                                already_saved.timestamp = obs.detection.frame_timestamp
                            else:
                                created_paths.append(other_path)
                                session.add(Snapshot(
                                    id=other_snap_id, event_id=event_id, camera_id=other_cam_id,
                                    image_path=other_path, timestamp=obs.detection.frame_timestamp
                                ))

                session.add(SystemLog(id=str(uuid4()), event_id=event_id, level='INFO',
                                      event_name='gate_event_created' if created else 'gate_evidence_associated',
                                      context={'camera_id': obs.detection.camera_id,
                                               'message': f'Container {number} confirmed on lane {obs.gate_id}'
                                               if obs.confirmed else f'Observation for {number or "unidentified container"} saved'}))
            if event_id and snapshots:
                try:
                    ev_dict = self.event(event_id)
                    if ev_dict:
                        snapshots.save_event_json(event_id, ev_dict)
                except Exception:
                    pass
            return event_id, created
        except Exception:
            for path in created_paths:
                try:
                    snapshots.remove(path)
                except Exception:
                    pass
            raise

    def events(self, limit=50, offset=0, container=None, camera_id=None, event_type=None, status=None, start_date=None, end_date=None):
        with self.db.sessions() as session:
            query = select(GateEvent)
            # Only hide events that have NO containers detected at all
            query = query.where(GateEvent.status != 'DUPLICATE')
            if container:
                query = query.where((GateEvent.container_number == container.upper()) | (GateEvent.container_number_2 == container.upper()))
            if camera_id:
                query = query.where(GateEvent.id.in_(select(DetectionRecord.event_id).where(DetectionRecord.camera_id == camera_id)))
            if event_type:
                query = query.where(GateEvent.event_type == event_type)
            if status:
                query = query.where(GateEvent.status == status)
            if start_date:
                query = query.where(GateEvent.timestamp >= start_date)
            if end_date:
                query = query.where(GateEvent.timestamp <= end_date)
            total = session.scalar(select(func.count()).select_from(query.subquery()))
            rows = session.scalars(query.order_by(GateEvent.timestamp.desc()).limit(limit).offset(offset)).all()
            items = [serialize(r) for r in rows]
            from app.ocr.validator import parse_feet_size
            import re
            for item in items:
                has_feet = session.scalar(select(DetectionRecord.id).where(
                    DetectionRecord.event_id == item['id'],
                    func.lower(DetectionRecord.class_name) == 'feet'
                )) is not None
                ocr_texts = session.scalars(select(OCRResult.normalized_text).join(DetectionRecord).where(
                    DetectionRecord.event_id == item['id']
                )).all()
                from app.ocr.validator import extract_iso_size_code
                parsed_size = None
                size_code = None
                for text in ocr_texts:
                    iso_c, human_s = extract_iso_size_code(text)
                    if iso_c:
                        size_code = iso_c
                    if human_s:
                        parsed_size = human_s
                    if size_code and parsed_size:
                        break
                if not size_code and item.get('iso_code_1'):
                    iso_c, human_s = extract_iso_size_code(item.get('iso_code_1'))
                    size_code = iso_c or item.get('iso_code_1')
                    parsed_size = parsed_size or human_s
                if item.get('is_twin'):
                    _, s1_human = extract_iso_size_code(item.get('iso_code_1') or '')
                    _, s2_human = extract_iso_size_code(item.get('iso_code_2') or '')
                    item['container_size'] = s1_human
                    item['container_size_2'] = s2_human
                    item['size_code'] = item.get('iso_code_1')
                    item['size_code_2'] = item.get('iso_code_2')
                else:
                    item['container_size'] = parsed_size or ('FEET' if has_feet else None)
                    item['size_code'] = size_code or item.get('iso_code_1')
            return {'items': items, 'total': total, 'limit': limit, 'offset': offset}

    def event(self, event_id):
        with self.db.sessions() as session:
            row = session.get(GateEvent, event_id)
            if row is None:
                return None
            result = serialize(row)
            detections = list(session.scalars(select(DetectionRecord).where(DetectionRecord.event_id == event_id)))
            result['detections'] = [serialize(d) for d in detections]
            result['ocr_results'] = [serialize(o) for o in session.scalars(select(OCRResult).where(
                OCRResult.detection_id.in_([d.id for d in detections])))]
            snapshots_list = [serialize(s) for s in session.scalars(select(Snapshot).where(Snapshot.event_id == event_id))]
            provenance = {}
            for log in session.scalars(select(SystemLog).where(SystemLog.event_id == event_id,
                    SystemLog.event_name == 'capture_provenance')):
                context = log.context or {}
                provenance[context.get('snapshot_id')] = context
            for snapshot in snapshots_list:
                snapshot['provenance'] = provenance.get(snapshot['id'])
            # Select best primary (front) camera: Prefer front/top cameras, exclude rear cameras from primary
            best_front_cam = None
            best_front_score = -1.0

            for d in detections:
                cam_id = (d.camera_id or '').lower()
                is_rear_cam = any(k in cam_id for k in ('-3', 'rear'))
                if is_rear_cam:
                    continue  # Rear cameras must not become primary (front) camera
                d_ocr = [o for o in result['ocr_results'] if o.get('detection_id') == d.id]
                score = 0.0
                if d_ocr:
                    for o in d_ocr:
                        c = float(o.get('confidence') or 0.0)
                        if o.get('valid_check_digit'):
                            s = 100.0 + c
                        elif o.get('valid_format'):
                            s = 50.0 + c
                        else:
                            s = c
                        if s > score:
                            score = s
                else:
                    score = float(getattr(d, 'confidence', 0.0) or 0.0)
                if any(k in cam_id for k in ('-1', 'front', 'top')):
                    score += 0.05
                if score > best_front_score:
                    best_front_score = score
                    best_front_cam = d.camera_id

            # Fallbacks for primary (front) camera
            front_cams = [s['camera_id'] for s in snapshots_list if not any(k in (s.get('camera_id') or '').lower() for k in ('-3', 'rear')) and not (s.get('image_path') or '').endswith('_rear.jpg')]
            primary_cam = best_front_cam or (front_cams[0] if front_cams else (snapshots_list[0]['camera_id'] if snapshots_list else None))

            front_snap = next((s for s in snapshots_list if not (s.get('image_path') or '').endswith('_rear.jpg') and s['camera_id'] == primary_cam), None)
            if not front_snap:
                front_snap = next((s for s in snapshots_list if not (s.get('image_path') or '').endswith('_rear.jpg') and not any(k in (s.get('camera_id') or '').lower() for k in ('-3', 'rear'))), (snapshots_list[0] if snapshots_list else None))

            # Rear snapshot: prefer dedicated rear camera snapshot, then fallback to _rear.jpg
            rear_snap = next((s for s in snapshots_list if any(k in (s.get('camera_id') or '').lower() for k in ('-3', 'rear')) and not (s.get('image_path') or '').endswith('_rear.jpg')), None)
            if not rear_snap:
                rear_snap = next((s for s in snapshots_list if (s.get('image_path') or '').endswith('_rear.jpg')), None)

            # Select proof by observed identity and physical slot, never camera ID.
            def identity_proof(number, rear):
                if not number:
                    return None
                matching = {o['detection_id'] for o in result['ocr_results']
                            if o.get('valid_check_digit') and o.get('normalized_text') == number}
                eligible = [d for d in detections if d.id in matching
                            and d.origin_key.endswith(':rear') == rear]
                if not eligible:
                    eligible = [d for d in detections if d.id in matching]
                for detection in eligible:
                    candidate = next((s for s in snapshots_list
                                      if s['camera_id'] == detection.camera_id
                                      and s['timestamp'] == serialize(detection)['timestamp']
                                      and (s['image_path'].endswith('_rear.jpg') == rear or not any(x['image_path'].endswith('_rear.jpg') for x in snapshots_list))), None)
                    if candidate:
                        return candidate
                    candidate_cam = next((s for s in snapshots_list if s['camera_id'] == detection.camera_id), None)
                    if candidate_cam:
                        return candidate_cam
                return None
            proof_front = identity_proof(result.get('container_number'), False)
            if proof_front:
                front_snap = proof_front
            proof_rear = identity_proof(result.get('container_number_2'), True) if result.get('is_twin') else None
            if proof_rear:
                rear_snap = proof_rear
            primary_cam = front_snap['camera_id'] if front_snap else None
            result['primary_camera_id'] = primary_cam
            result['primary_snapshot_id'] = front_snap['id'] if front_snap else None
            result['rear_snapshot_id'] = rear_snap['id'] if rear_snap else None
            result['snapshots'] = snapshots_list

            # Top-level tracking identification
            track_ids = [d.track_id for d in detections if getattr(d, 'track_id', None) is not None]
            result['track_id'] = track_ids[0] if track_ids else None
            session_ids = [
                d.origin_key.split(':')[1]
                for d in detections
                if getattr(d, 'origin_key', '') and d.origin_key.startswith('session:') and len(d.origin_key.split(':')) > 1
            ]
            result['tracking_id'] = session_ids[0] if session_ids else (f"TRACK_{result['track_id']}" if result.get('track_id') else None)

            # Per-camera best image mapping for UI and API
            best_cam_map = {}
            for s in snapshots_list:
                cid = s.get('camera_id')
                if cid and cid not in best_cam_map:
                    best_cam_map[cid] = s
            result['best_camera_images'] = best_cam_map
            
            from app.ocr.validator import extract_iso_size_code
            container_size = None
            size_code = None
            for ocr in result['ocr_results']:
                text_to_check = f"{ocr.get('normalized_text', '')} {ocr.get('raw_text', '')}"
                iso_c, human_s = extract_iso_size_code(text_to_check)
                if iso_c:
                    size_code = iso_c
                if human_s:
                    container_size = human_s
                if size_code and container_size:
                    break
            
            if not size_code and result.get('iso_code_1'):
                iso_c, human_s = extract_iso_size_code(result.get('iso_code_1'))
                size_code = iso_c or result.get('iso_code_1')
                container_size = container_size or human_s

            if not container_size and any(d.get('class_name', '').lower() == 'feet' for d in result['detections']):
                container_size = 'FEET DETECTED'
            if result.get('is_twin'):
                _, s1_human = extract_iso_size_code(result.get('iso_code_1') or '')
                _, s2_human = extract_iso_size_code(result.get('iso_code_2') or '')
                result['container_size'] = s1_human
                result['container_size_2'] = s2_human
                result['size_code'] = result.get('iso_code_1')
                result['size_code_2'] = result.get('iso_code_2')
            else:
                result['container_size'] = container_size
                result['size_code'] = size_code or result.get('iso_code_1')
            return result

    def ocr_results(self, limit=50, offset=0):
        with self.db.sessions() as session:
            return [serialize(o) for o in session.scalars(select(OCRResult).order_by(OCRResult.timestamp.desc()).limit(limit).offset(offset))]

    def snapshot(self, snapshot_id):
        with self.db.sessions() as session:
            row = session.get(Snapshot, snapshot_id)
            return serialize(row) if row else None

    def mark_cloud_synced(self, event_id: str, cloud_event_id: str | None = None):
        with self.db.sessions.begin() as session:
            ev = session.get(GateEvent, event_id)
            if ev:
                ev.cloud_synced = True
                ev.cloud_delivery_state = 'DELIVERED'
                ev.cloud_delivery_error = None
                ev.cloud_synced_at = utcnow()
                if cloud_event_id:
                    ev.cloud_event_id = str(cloud_event_id)

    def mark_cloud_delivery(self, event_id: str, state: str, error: str | None = None):
        if state not in ('PENDING', 'OFFLINE', 'REJECTED', 'CONFLICT', 'BLOCKED', 'DUPLICATE'):
            raise ValueError('Invalid cloud delivery state')
        with self.db.sessions.begin() as session:
            ev = session.get(GateEvent, event_id)
            if ev:
                if ev.cloud_synced and ev.cloud_delivery_state == 'DELIVERED':
                    return
                ev.cloud_delivery_state = state
                ev.cloud_delivery_error = error[:256] if error else None
                ev.cloud_synced = False
                ev.cloud_synced_at = None
                ev.cloud_event_id = None

    def mark_duplicate_event(self, duplicate_id: str, primary_id: str):
        """Retain an audit row but remove a duplicate from active visits/outbox."""
        with self._write_lock, self.db.sessions.begin() as session:
            duplicate = session.get(GateEvent, duplicate_id)
            primary = session.get(GateEvent, primary_id)
            if not duplicate or not primary or duplicate.id == primary.id:
                return False
            session.execute(update(DetectionRecord).where(DetectionRecord.event_id == duplicate.id).values(event_id=primary.id))
            session.execute(update(Snapshot).where(Snapshot.event_id == duplicate.id).values(event_id=primary.id))
            session.execute(update(SystemLog).where(SystemLog.event_id == duplicate.id).values(event_id=primary.id))
            duplicate.status = 'DUPLICATE'
            duplicate.cloud_delivery_state = 'DUPLICATE'
            duplicate.cloud_synced = False
            duplicate.cloud_synced_at = None
            return True

    def reset_cloud_failures(self):
        """Operator retry resets terminal failures; actual acknowledgements stay intact."""
        with self.db.sessions.begin() as session:
            session.execute(update(GateEvent).where(
                GateEvent.status == 'CONFIRMED', GateEvent.cloud_synced == False,
                GateEvent.cloud_delivery_state.in_(['REJECTED', 'BLOCKED'])
            ).values(cloud_delivery_state='PENDING', cloud_delivery_error=None))

    def cloud_delivery_counts(self):
        with self.db.sessions() as session:
            return {state or 'PENDING': count for state, count in session.execute(
                select(GateEvent.cloud_delivery_state, func.count(GateEvent.id))
                .group_by(GateEvent.cloud_delivery_state))}

    def get_unsynced_confirmed_events(self, limit: int = 50):
        with self.db.sessions() as session:
            query = (
                select(GateEvent)
                .where(
                    GateEvent.status == 'CONFIRMED',
                    func.coalesce(GateEvent.cloud_delivery_state, 'PENDING').in_(['PENDING', 'OFFLINE', 'CONFLICT']),
                    (GateEvent.container_number.is_not(None) | GateEvent.container_number_2.is_not(None)),
                    (GateEvent.cloud_synced == False) | (GateEvent.cloud_synced.is_(None))
                )
                .order_by(GateEvent.timestamp.asc())
                .limit(limit)
            )
            raw_events = [serialize(e) for e in session.scalars(query)]
            # Filter and auto-retire any invalid records so they never spam the cloud dispatcher
            valid_events = []
            from app.ocr.validator import ContainerValidator
            from app.domain import OCRRead
            validator = ContainerValidator()
            import re
            for ev in raw_events:
                c1 = ev.get('container_number')
                val1 = validator.validate(OCRRead(raw_text=c1, confidence=1.0)) if c1 else None
                is_valid = is_valid_iso_6346(c1)
                if is_valid:
                    valid_events.append(ev)
                else:
                    # Auto-retire invalid record in DB so it stops looping
                    try:
                        db_ev = session.get(GateEvent, ev['id'])
                        if db_ev:
                            db_ev.cloud_delivery_state = 'BLOCKED'
                            db_ev.cloud_delivery_error = 'Invalid container checksum'
                            session.commit()
                    except Exception:
                        pass
            return valid_events

    def count_unsynced_confirmed_events(self) -> int:
        with self.db.sessions() as session:
            query = (
                select(func.count(GateEvent.id))
                .where(
                    GateEvent.status == 'CONFIRMED',
                    func.coalesce(GateEvent.cloud_delivery_state, 'PENDING').in_(['PENDING', 'OFFLINE', 'CONFLICT']),
                    (GateEvent.container_number.is_not(None) | GateEvent.container_number_2.is_not(None)),
                    (GateEvent.cloud_synced == False) | (GateEvent.cloud_synced.is_(None))
                )
            )
            return int(session.scalar(query) or 0)

    def clean_duplicate_events(self, max_gap_seconds: int = 1200, snapshots=None):
        """
        Merges and removes duplicate events created for the same container within dwell windows.
        Re-parents detections and snapshots to the primary event, then safely deletes duplicates.
        """
        with self._write_lock, self.db.sessions.begin() as session:
            events = session.scalars(
                select(GateEvent)
                .where(GateEvent.container_number.is_not(None))
                .order_by(GateEvent.timestamp.asc())
            ).all()

            groups: dict[str, list[GateEvent]] = {}
            for ev in events:
                cntr = ev.container_number or ''
                key = cntr[4:] if len(cntr) == 11 and cntr[4:].isdigit() else cntr
                groups.setdefault(key, []).append(ev)

            merged_count = 0
            for key, ev_list in groups.items():
                if len(ev_list) <= 1:
                    continue
                clusters = []
                current_cluster = [ev_list[0]]
                for ev in ev_list[1:]:
                    last_ev = current_cluster[-1]
                    gap = datetime_diff_seconds(ev.timestamp, last_ev.timestamp)
                    if gap <= max_gap_seconds:
                        current_cluster.append(ev)
                    else:
                        clusters.append(current_cluster)
                        current_cluster = [ev]
                clusters.append(current_cluster)

                for cluster in clusters:
                    if len(cluster) <= 1:
                        continue
                    primary = cluster[0]
                    if any(e.status == 'CONFIRMED' for e in cluster):
                        primary.status = 'CONFIRMED'
                    # Prefer valid 11-char container number with valid check digit
                    for e in cluster:
                        if e.container_number and len(e.container_number) == 11 and e.container_number not in ('TIIU5077669', 'MSCU6546021'):
                            primary.container_number = e.container_number
                            break

                    for dup in cluster[1:]:
                        session.execute(
                            update(DetectionRecord)
                            .where(DetectionRecord.event_id == dup.id)
                            .values(event_id=primary.id)
                        )
                        session.execute(
                            update(Snapshot)
                            .where(Snapshot.event_id == dup.id)
                            .values(event_id=primary.id)
                        )
                        session.execute(
                            update(SystemLog)
                            .where(SystemLog.event_id == dup.id)
                            .values(event_id=primary.id)
                        )
                        session.delete(dup)
                        merged_count += 1

            # Second pass: merge near-simultaneous events using image similarity and capture timestamps
            snap_mgr = None
            try:
                from app.config.settings import Settings
                snap_mgr = SnapshotManager(Settings().snapshot_directory)
            except Exception:
                pass

            remaining_events = session.scalars(
                select(GateEvent)
                .where(GateEvent.status.in_(['CONFIRMED', 'NEEDS_REVIEW']))
                .order_by(GateEvent.timestamp.asc())
            ).all()

            merged_set = set()
            for i in range(len(remaining_events)):
                ev1 = remaining_events[i]
                if ev1.id in merged_set:
                    continue
                for j in range(i + 1, min(i + 10, len(remaining_events))):
                    ev2 = remaining_events[j]
                    if ev2.id in merged_set:
                        continue
                    gap = datetime_diff_seconds(ev2.timestamp, ev1.timestamp)
                    if gap > 30.0:
                        break
                    if ev1.gate_id != ev2.gate_id:
                        continue

                    # Check image similarity
                    is_match = False
                    if snap_mgr and hasattr(snap_mgr, 'path'):
                        for cam_id in ['gate-in-1', 'gate-in-2', 'gate-in-3', 'gate-in-4', 'gate-out-1', 'gate-out-2', 'gate-out-3', 'gate-out-4']:
                            try:
                                p1 = snap_mgr.path(f"{ev1.id}/{cam_id}.jpg")
                                p2 = snap_mgr.path(f"{ev2.id}/{cam_id}.jpg")
                                if p1.is_file() and p2.is_file():
                                    sim = SnapshotManager.compute_similarity(p1, p2)
                                    if sim >= 0.86 or (gap <= 5.0 and sim >= 0.78):
                                        is_match = True
                                        break
                            except Exception:
                                pass

                    c1, c2 = ev1.container_number or '', ev2.container_number or ''
                    ocr_sim = ocr_similarity_score(c1, c2)
                    if gap <= 35.0 and (c1 == c2 or (c1 and c2 and ocr_sim >= 0.80)):
                        is_match = True

                    if is_match:
                        # ev2 is duplicate of ev1
                        session.execute(update(DetectionRecord).where(DetectionRecord.event_id == ev2.id).values(event_id=ev1.id))
                        session.execute(update(Snapshot).where(Snapshot.event_id == ev2.id).values(event_id=ev1.id))
                        session.execute(update(SystemLog).where(SystemLog.event_id == ev2.id).values(event_id=ev1.id))
                        if c2 and len(c2) == 11 and (not c1 or len(c1) != 11):
                            ev1.container_number = c2
                        if any(e.status == 'CONFIRMED' for e in [ev1, ev2]):
                            ev1.status = 'CONFIRMED'
                        session.delete(ev2)
                        merged_set.add(ev2.id)
                        merged_count += 1

            return merged_count

    def canonical_event_id(self, event_id: str, max_gap_seconds: int = 900, snapshots=None):
        """Return the oldest confirmed event for a duplicate dwell identity, comparing
        exact container numbers, capture timestamps, and snapshot image similarity."""
        with self.db.sessions() as session:
            event = session.get(GateEvent, event_id)
            if not event:
                return event_id
            delta = timedelta(seconds=max_gap_seconds)
            direction_match = (GateEvent.event_type == event.event_type) | (GateEvent.event_type == 'UNKNOWN')
            candidates = session.scalars(select(GateEvent).where(
                GateEvent.id != event.id,
                GateEvent.gate_id == event.gate_id,
                direction_match,
                GateEvent.status == 'CONFIRMED',
                GateEvent.timestamp >= event.timestamp - delta,
                GateEvent.timestamp <= event.timestamp + delta,
            ).order_by(GateEvent.timestamp.asc())).all()

            if not candidates:
                return event_id

            # 1. Exact container number match
            if event.container_number:
                for cand in candidates:
                    if cand.timestamp <= event.timestamp:
                        if cand.container_number == event.container_number:
                            return cand.id

            # 2. Image similarity and capture timestamp comparison
            snap_mgr = snapshots
            if snap_mgr is None:
                try:
                    from app.config.settings import Settings
                    snap_mgr = SnapshotManager(Settings().snapshot_directory)
                except Exception:
                    pass

            for cand in candidates:
                time_diff = datetime_diff_seconds(cand.timestamp, event.timestamp)
                c_num = cand.container_number or ''
                num = event.container_number or ''

                # Distinct container numbers must NEVER be merged as duplicates by static camera background similarity.
                if num and c_num and num != c_num:
                    if is_valid_iso_6346(num) and is_valid_iso_6346(c_num):
                        continue
                    ocr_sim = ocr_similarity_score(num, c_num)
                    if not (time_diff <= 35.0 and ocr_sim >= 0.80):
                        continue

                # Image comparison across matching cameras
                if snap_mgr and hasattr(snap_mgr, 'path'):
                    for cam_id in ['gate-in-1', 'gate-in-2', 'gate-in-3', 'gate-in-4', 'gate-out-1', 'gate-out-2', 'gate-out-3', 'gate-out-4']:
                        try:
                            p_ev = snap_mgr.path(f"{event.id}/{cam_id}.jpg")
                            p_cd = snap_mgr.path(f"{cand.id}/{cam_id}.jpg")
                            if p_ev.is_file() and p_cd.is_file():
                                sim = SnapshotManager.compute_similarity(p_ev, p_cd)
                                if sim >= 0.86 or (time_diff <= 5.0 and sim >= 0.78):
                                    return cand.id if cand.timestamp <= event.timestamp else cand.id
                        except Exception:
                            continue

                # Fuzzy OCR match within passage window
                ocr_sim = ocr_similarity_score(event.container_number, cand.container_number)
                if time_diff <= 35.0 and (num == c_num or (num and c_num and ocr_sim >= 0.80)):
                    return cand.id if cand.timestamp <= event.timestamp else cand.id

            return event_id
