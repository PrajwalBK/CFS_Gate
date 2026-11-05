import difflib
import hashlib
import logging
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from uuid import uuid4
from typing import Any
from sqlalchemy import func, select, update

from app.database.models import Camera, DetectionRecord, GateEvent, OCRResult, Snapshot, SystemLog
from app.domain import Observation, Validation, utcnow
from app.snapshots.snapshot_manager import SnapshotManager
from app.ocr.validator import is_valid_iso_6346
from app.ocr.consensus import CharacterConsensus

log = logging.getLogger('gate')


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


def datetime_diff_seconds(dt1: Any, dt2: Any) -> float:
    """Safe absolute seconds difference between offset-naive, offset-aware datetimes, or ISO strings."""
    if not dt1 or not dt2:
        return float('inf')
    try:
        if isinstance(dt1, str):
            dt1 = datetime.fromisoformat(dt1.replace('Z', '+00:00'))
        if isinstance(dt2, str):
            dt2 = datetime.fromisoformat(dt2.replace('Z', '+00:00'))
        t1 = dt1.replace(tzinfo=None)
        t2 = dt2.replace(tzinfo=None)
        return abs((t1 - t2).total_seconds())
    except Exception:
        return float('inf')


COMMON_OCR_CONFUSIONS = {
    ('F', 'I'), ('I', 'F'), ('8', '1'), ('1', '8'), ('8', '0'), ('0', '8'),
    ('8', '3'), ('3', '8'), ('8', 'B'), ('B', '8'), ('0', 'O'), ('O', '0'),
    ('0', 'D'), ('D', '0'), ('0', 'Q'), ('Q', '0'), ('0', 'C'), ('C', '0'),
    ('1', 'I'), ('I', '1'), ('1', '7'), ('7', '1'), ('1', 'T'), ('T', '1'),
    ('1', 'L'), ('L', '1'), ('5', 'S'), ('S', '5'), ('2', 'Z'), ('Z', '2'),
    ('6', 'G'), ('G', '6'), ('6', 'C'), ('C', '6'), ('K', 'R'), ('R', 'K'),
    ('U', 'V'), ('V', 'U'), ('E', 'F'), ('F', 'E'), ('P', 'F'), ('F', 'P'),
    ('P', 'R'), ('R', 'P'), ('C', 'O'), ('O', 'C'), ('A', '4'), ('4', 'A'),
}


def is_distinct_container(num1: str | None, num2: str | None) -> bool:
    """Return True if both are valid ISO 6346 container numbers representing different physical containers.
    
    Returns False when both strings represent the same container (even with common OCR substitutions,
    noisy character reads, or check-digit flips resulting from 1-2 character errors).
    """
    if not num1 or not num2:
        return False
    s1, s2 = num1.strip().upper(), num2.strip().upper()
    if s1 == s2:
        return False
    if is_valid_iso_6346(s1) and is_valid_iso_6346(s2):
        d1 = ''.join(c for c in s1 if c.isdigit())
        d2 = ''.join(c for c in s2 if c.isdigit())
        sim = ocr_similarity_score(s1, s2)
        pref1, pref2 = s1[:4], s2[:4]
        matching_prefix_letters = sum(1 for a, b in zip(pref1, pref2) if a == b)
        matching_serial_digits = sum(1 for a, b in zip(d1[:6], d2[:6]) if a == b) if len(d1) >= 6 and len(d2) >= 6 else 0
        body_diff_pairs = [(s1[i], s2[i]) for i in range(min(10, len(s1))) if s1[i] != s2[i]]
        all_body_confused = bool(body_diff_pairs and all((a, b) in COMMON_OCR_CONFUSIONS for a, b in body_diff_pairs))

        # 1. 6 serial digits match exactly (e.g. TONU7922656 vs TCNU7922652):
        # Same physical container if at least 2 prefix letters match or recognized OCR confusions.
        # Completely different owner codes (e.g. TGHU vs MSCU with coincidental matching digits on twin trailers) are distinct!
        if len(d1) >= 6 and len(d2) >= 6 and (d1[:6] == d2[:6] or (len(s1) >= 10 and len(s2) >= 10 and s1[4:10] == s2[4:10])):
            if matching_prefix_letters >= 2 or all_body_confused:
                return False
            return True

        # 2. At least 5 of 6 serial digits match (e.g. DFSU3084122 vs DISU3014120, EGSU8198223 vs EGSU3198220):
        # Same physical container when 5 serial digits match and prefix has at least 2 matching letters or OCR confusions.
        if matching_serial_digits >= 5:
            if matching_prefix_letters >= 2 or all_body_confused:
                return False

        # 3. Owner prefix matches (first 3 letters): require at least 3 matching serial digits or similarity >= 0.63
        if len(s1) >= 4 and len(s2) >= 4 and s1[:3] == s2[:3]:
            if matching_serial_digits >= 3 or sim >= 0.63:
                return False
            return True

        # 4. Differing owner prefix with high similarity or OCR character confusion:
        # Accommodates 1-2 character substitutions plus resulting check-digit change (up to 3 total character differences)
        char_diffs = sum(1 for a, b in zip(s1, s2) if a != b)
        if char_diffs <= 3:
            if matching_prefix_letters >= 2 and sim >= 0.70:
                return False
            if all_body_confused:
                return False

        if matching_prefix_letters >= 3 and sim >= 0.78:
            return False
        return True
    return False


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
                # Look for existing event on this gate
                existing = db_session.get(GateEvent, session.event_id)
                committed = db_session.scalar(select(SystemLog.id).where(
                    SystemLog.event_id == session.event_id,
                    SystemLog.event_name == 'multi_camera_passage_finalized'))
                if committed and existing:
                    # If already confirmed with container number, nothing to update
                    if existing.status == 'CONFIRMED' and existing.container_number:
                        return session.event_id
                    # If previously NEEDS_REVIEW / unconfirmed, and new consensus container number is now available:
                    container_num, iso_code, conf = session.get_consensus_container_number()
                    if container_num:
                        existing.container_number = container_num
                        if iso_code:
                            existing.iso_code_1 = iso_code
                        if is_valid_iso_6346(container_num):
                            existing.status = 'CONFIRMED'
                        db_session.flush()
                        return session.event_id
                    return session.event_id

                best_frames = session.get_best_frame_per_camera()
                container_num, iso_code, conf = session.get_consensus_container_number()

                # Ghost/passerby elimination guard:
                # If there is NO container number AND NO verified physical container body proof on any camera,
                # discard this spurious session without cluttering the database or operator UI with empty events.
                has_container_proof = False
                for c_id, f_cand in (best_frames or {}).items():
                    for d in getattr(f_cand, 'detections', []):
                        if (getattr(d, 'class_name', '') in ('container', 'container_object')
                            and getattr(d, 'confidence', 0.0) >= 0.40
                            and len(getattr(d, 'bbox', [])) == 4):
                            bw = d.bbox[2] - d.bbox[0]
                            bh = d.bbox[3] - d.bbox[1]
                            if bw >= 60.0 and bh >= 50.0 and (bw * bh) >= 3000.0 and (bw / max(bh, 1.0)) >= 0.50:
                                has_container_proof = True
                                break
                    if has_container_proof:
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

                slot_pos = getattr(session, 'container_slot', 'CONTAINER_1') or 'CONTAINER_1'
                dedup_sec = float(getattr(self, 'dedup_seconds', 35.0) or 35.0)
                delta = timedelta(seconds=max(dedup_sec, 60.0))

                if not existing and slot_pos == 'CONTAINER_1':
                    # 1. Search for existing event on the same gate with the exact same container number within delta
                    if container_num:
                        existing = db_session.scalar(select(GateEvent).where(
                            GateEvent.gate_id == session.gate_id,
                            GateEvent.container_number == container_num,
                            GateEvent.timestamp >= session.created_at - delta,
                            GateEvent.timestamp <= session.created_at + delta,
                        ).order_by(GateEvent.timestamp.desc()))

                        # 2. Suffix / serial match (matching 6 serial digits)
                        if not existing and len(container_num) == 11 and container_num[4:].isdigit():
                            suffix = container_num[4:]
                            num_serial = container_num[4:10]
                            cands = db_session.scalars(select(GateEvent).where(
                                GateEvent.gate_id == session.gate_id,
                                GateEvent.timestamp >= session.created_at - delta,
                                GateEvent.timestamp <= session.created_at + delta,
                            ).order_by(GateEvent.timestamp.desc())).all()
                            for c in cands:
                                c1 = c.container_number or ''
                                if len(c1) == 11 and c1[4:].isdigit():
                                    if c1[4:] == suffix or (c1[4:10] == num_serial and not is_distinct_container(container_num, c1)):
                                        existing = c
                                        break

                    # 3. Absorb any unconfirmed / in-flight draft event created on the lane during this passage
                    if not existing and slot_pos == 'CONTAINER_1':
                        draft_cand = db_session.scalars(select(GateEvent).where(
                            GateEvent.gate_id == session.gate_id,
                            GateEvent.status == 'NEEDS_REVIEW',
                            GateEvent.cloud_synced == False,
                            GateEvent.timestamp >= session.created_at - delta,
                            GateEvent.timestamp <= session.created_at + delta,
                        ).order_by(GateEvent.timestamp.desc())).first()
                        if draft_cand:
                            d_num = draft_cand.container_number or ''
                            if not d_num or not is_valid_iso_6346(d_num) or (container_num and ocr_similarity_score(container_num, d_num) >= 0.40):
                                existing = draft_cand

                target_event_id = existing.id if existing else session.event_id

                if existing and existing.id != session.event_id and snapshots and hasattr(snapshots, 'evidence_manager'):
                    snapshots.evidence_manager.merge_event_folders(session.event_id, target_event_id, container_number=container_num)
                elif snapshots and hasattr(snapshots, 'evidence_manager') and container_num:
                    snapshots.evidence_manager.update_container_number(target_event_id, container_num)

                sess_cloud_ev_id = getattr(session, 'cloud_event_id', None)
                sess_cloud_visit_id = getattr(session, 'cloud_visit_id', None)
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
                        status=status,
                        cloud_event_id=sess_cloud_ev_id,
                        cloud_visit_id=sess_cloud_visit_id,
                        cloud_delivery_state='EVENT_CREATED' if sess_cloud_ev_id else 'PENDING'
                    )
                    db_session.add(event)
                    db_session.flush()
                else:
                    if sess_cloud_ev_id and not existing.cloud_event_id:
                        existing.cloud_event_id = sess_cloud_ev_id
                    elif existing.cloud_event_id and not sess_cloud_ev_id:
                        session.cloud_event_id = existing.cloud_event_id
                    if sess_cloud_visit_id and not existing.cloud_visit_id:
                        existing.cloud_visit_id = sess_cloud_visit_id
                    elif existing.cloud_visit_id and not sess_cloud_visit_id:
                        session.cloud_visit_id = existing.cloud_visit_id
                    if container_num:
                        if not existing.container_number:
                            existing.container_number = container_num
                        elif existing.container_number != container_num:
                            all_cands = list(session.ocr_candidates or [])
                            for ocr_rec in db_session.scalars(
                                select(OCRResult)
                                .join(DetectionRecord, OCRResult.detection_id == DetectionRecord.id)
                                .where(DetectionRecord.event_id == existing.id)
                            ).all():
                                all_cands.append(Validation(
                                    raw_text=getattr(ocr_rec, 'raw_text', '') or '',
                                    normalized_text=getattr(ocr_rec, 'normalized_text', '') or '',
                                    confidence=float(getattr(ocr_rec, 'confidence', 0.0) or 0.0),
                                    valid_format=bool(getattr(ocr_rec, 'valid_format', False)),
                                    valid_check_digit=bool(getattr(ocr_rec, 'valid_check_digit', False)),
                                    validation_status=getattr(ocr_rec, 'validation_status', '') or '',
                                    size_type=getattr(ocr_rec, 'size_type', None)
                                ))
                            best_c = CharacterConsensus.resolve(all_cands)
                            if best_c and best_c.normalized_text and best_c.valid_check_digit and is_valid_iso_6346(best_c.normalized_text):
                                existing.container_number = best_c.normalized_text
                            elif conf > (getattr(existing, 'confidence', 0.0) or 0.0):
                                existing.container_number = container_num
                    if iso_code:
                        existing.iso_code_1 = iso_code
                    if is_confirmed:
                        existing.status = 'CONFIRMED'

                # Save the independent BEST frame for EVERY camera on this gate that captured a container
                primary_det_id = None
                for cam_id, frame_cand in best_frames.items():
                    # Strictly require container detection on this camera: skip building/people/empty frames
                    has_cntr = any(
                        getattr(d, 'class_name', '').lower() in ('container', 'cointainer', 'container_object')
                        for d in (getattr(frame_cand, 'detections', []) or [])
                    )
                    if not has_cntr:
                        continue
                    snap_id = str(uuid4())
                    if hasattr(snapshots, 'evidence_manager'):
                        rel_best, rel_leg = snapshots.evidence_manager.save_best_frame(
                            target_event_id, cam_id, frame_cand.image,
                            direction=session.direction,
                            container_number=container_num or '',
                            quality_score=snapshots.compute_quality(frame_cand.image)
                        )
                        img_path = rel_leg
                    else:
                        img_path = snapshots.save_side_snapshot(
                            target_event_id, cam_id, frame_cand.image,
                            direction=session.direction,
                            text=container_num or '',
                            force=True
                        )
                    created_paths.append(img_path)
                    existing_snap = db_session.scalar(select(Snapshot).where(
                        (Snapshot.image_path == img_path) |
                        ((Snapshot.event_id == target_event_id) & (Snapshot.camera_id == cam_id))
                    ))
                    if not existing_snap:
                        db_session.add(Snapshot(
                            id=snap_id,
                            event_id=target_event_id,
                            camera_id=cam_id,
                            image_path=img_path,
                            timestamp=frame_cand.timestamp
                        ))
                    else:
                        existing_snap.event_id = target_event_id
                        existing_snap.camera_id = cam_id
                        existing_snap.image_path = img_path
                        existing_snap.timestamp = frame_cand.timestamp
                    self._capture_provenance(db_session, snapshots, existing_snap.id if existing_snap else snap_id,
                        target_event_id, cam_id, frame_cand.timestamp, img_path, frame_cand.detections,
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
                            event_id=target_event_id,
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
                            event_id=target_event_id,
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
                        event_id=target_event_id,
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
                    event_id=target_event_id,
                    level='INFO',
                    event_name='multi_camera_passage_finalized',
                    context={
                        'session_id': session.session_id,
                        'gate_id': session.gate_id,
                        'container_number': container_num,
                        'cameras_captured': list(best_frames.keys()),
                        'status': existing.status if existing else status,
                        'merged_from_session_event_id': session.event_id if session.event_id != target_event_id else None
                    }
                ))

            if snapshots:
                ev_data = self.event(target_event_id)
                if ev_data:
                    try:
                        snapshots.save_event_json(target_event_id, ev_data)
                    except OSError:
                        # JSON is derived; never delete committed proof images
                        # when this optional export fails after the transaction.
                        pass

            return target_event_id
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

    def cameras(self):
        with self.db.sessions() as session:
            rows = session.scalars(select(Camera)).all()
            from app.config.settings import CameraConfig
            result = []
            for r in rows:
                if r.configuration:
                    try:
                        result.append(CameraConfig.model_validate(r.configuration))
                        continue
                    except Exception:
                        pass
                result.append(CameraConfig(
                    id=r.id, name=r.name, gate_id=r.gate_id,
                    source_env=r.source_env, enabled=bool(r.enabled)
                ))
            return result

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
                    # Tolerant suffix / serial matching on same gate: if 6 serial digits match (e.g. AIPU5077669 vs TIIU5077669, or TCNU7922652 vs TONU7922656)
                    if not existing and len(number) == 11 and number[4:].isdigit():
                        suffix = number[4:]
                        num_serial = number[4:10]
                        candidates = session.scalars(select(GateEvent).where(
                            GateEvent.gate_id == obs.gate_id,
                            direction_match,
                            GateEvent.timestamp >= obs.detection.frame_timestamp - delta,
                            GateEvent.timestamp <= obs.detection.frame_timestamp + delta,
                        ).order_by(GateEvent.timestamp.desc())).all()
                        for cand in candidates:
                            c1 = cand.container_number or ''
                            if len(c1) == 11 and c1[4:].isdigit():
                                if c1[4:] == suffix or not is_distinct_container(number, c1):
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
                            num_serial = number[4:10]
                            candidates = session.scalars(select(GateEvent).where(
                                GateEvent.gate_id == obs.gate_id,
                                direction_match,
                                GateEvent.status.in_(['CONFIRMED', 'NEEDS_REVIEW']),
                                GateEvent.timestamp >= obs.detection.frame_timestamp - long_delta,
                                GateEvent.timestamp <= obs.detection.frame_timestamp,
                            ).order_by(GateEvent.timestamp.asc())).all()
                            for cand in candidates:
                                c1 = cand.container_number or ''
                                if len(c1) == 11 and c1[4:].isdigit():
                                    if c1[4:] == suffix or not is_distinct_container(number, c1):
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

                            # Distinct verified container numbers must NEVER be merged as duplicates by static camera background similarity.
                            if number and c_num and number != c_num:
                                if is_distinct_container(number, c_num):
                                    continue  # Distinct physical containers! NEVER merge!

                            # 1. Matching serial digits (e.g. AXU1960888 vs AXIU1960888: both have 1960888)
                            if number and c_num:
                                num_digits = ''.join(c for c in number if c.isdigit())
                                cand_digits = ''.join(c for c in c_num if c.isdigit())
                                if len(num_digits) >= 5 and len(cand_digits) >= 5:
                                    if num_digits[:6] == cand_digits[:6] or num_digits[-6:] == cand_digits[-6:] or (len(num_digits) >= 6 and num_digits[:6] in cand_digits):
                                        match_found = True

                            if number and c_num and number != c_num:
                                num_digits = ''.join(c for c in number if c.isdigit())
                                cand_digits = ''.join(c for c in c_num if c.isdigit())
                                digits_match = bool(len(num_digits) >= 5 and len(cand_digits) >= 5 and (num_digits[:6] == cand_digits[:6] or num_digits[-6:] == cand_digits[-6:] or (len(num_digits) >= 6 and num_digits[:6] in cand_digits)))
                                if not (ocr_sim >= 0.50 or digits_match or not is_distinct_container(number, c_num)):
                                    continue  # Distinct physical containers! NEVER merge!

                            slot_pos = getattr(obs, 'slot_position', 'CONTAINER_1') or 'CONTAINER_1'
                            if slot_pos in ('CONTAINER_2', 'CONTAINER_3') and cand.container_number:
                                # Container 2 must never be merged into Container 1's confirmed event
                                continue
                            if slot_pos in ('CONTAINER_2', 'CONTAINER_3'):
                                cand_has_other_track = session.scalar(select(DetectionRecord.id).where(
                                    DetectionRecord.event_id == cand.id,
                                    DetectionRecord.camera_id == obs.detection.camera_id,
                                    DetectionRecord.track_id != obs.track_id
                                ).limit(1))
                                if cand_has_other_track:
                                    continue

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
                            if not match_found and time_diff <= 8.0:
                                same_track = False
                                if getattr(obs, 'track_id', None) is not None:
                                    same_track = bool(session.scalar(select(DetectionRecord.id).where(
                                        DetectionRecord.event_id == cand.id,
                                        DetectionRecord.camera_id == obs.detection.camera_id,
                                        DetectionRecord.track_id == obs.track_id
                                    ).limit(1)))
                                if same_track or ocr_sim >= 0.45 or (not number and time_diff <= 3.0):
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
                is_second_slot = getattr(obs, 'slot_position', 'CONTAINER_1') in ('REAR', 'CONTAINER_2')
                if not existing and getattr(obs, 'event_id', None):
                    active = session.get(GateEvent, obs.event_id)
                    if active and active.gate_id == obs.gate_id:
                        if not is_second_slot and is_distinct_container(active.container_number, number):
                            pass  # Distinct containers! Never merge!
                        else:
                            time_diff = datetime_diff_seconds(active.timestamp, obs.detection.frame_timestamp)
                            if not active.cloud_synced and time_diff <= max(dedup_seconds, 60.0):
                                existing = active

                if ongoing_event_id and not existing:
                    active = session.get(GateEvent, ongoing_event_id)
                    if active and active.gate_id == obs.gate_id:
                        if not is_second_slot and is_distinct_container(active.container_number, number):
                            pass  # Distinct containers! Never merge!
                        else:
                            time_diff = datetime_diff_seconds(active.timestamp, obs.detection.frame_timestamp)
                            if not active.cloud_synced and time_diff <= max(dedup_seconds, 60.0):
                                existing = active

                if obs.origin_key.startswith('dwell:') and existing:
                    time_diff = datetime_diff_seconds(existing.timestamp, obs.detection.frame_timestamp)
                    num_digits = ''.join(c for c in (number or '') if c.isdigit())
                    ex_digits = ''.join(c for c in (existing.container_number or '') if c.isdigit())
                    digits_match = bool(num_digits and ex_digits and (
                        (len(num_digits) >= 6 and len(ex_digits) >= 6 and num_digits[:6] == ex_digits[:6]) or
                        num_digits[-5:] == ex_digits[-5:] or
                        num_digits in ex_digits
                    ))
                    is_same_container = bool(
                        time_diff <= 35.0 and (
                            (number and number == existing.container_number) or
                            digits_match or
                            (number and existing.container_number and ocr_similarity_score(number, existing.container_number) >= 0.55)
                        )
                    )
                    if not is_same_container:
                        # Only separate as a new cleared passage if there is a distinct valid container number
                        # or the time window has genuinely elapsed. Intermediate/unconfirmed frames where
                        # number is None or unconfirmed must fuse into existing rather than spawning ghost duplicates.
                        is_distinct_read = bool(
                            number and existing.container_number and
                            number != existing.container_number and
                            is_distinct_container(existing.container_number, number)
                        )
                        if is_distinct_read or time_diff > max(dedup_seconds, 35.0):
                            prior_view = session.scalar(select(DetectionRecord.id).where(
                                DetectionRecord.event_id == existing.id,
                                DetectionRecord.camera_id == obs.detection.camera_id,
                                DetectionRecord.origin_key.like('dwell:%'),
                                DetectionRecord.origin_key != obs.origin_key))
                            if prior_view:
                                existing = None  # New cleared passage, not time-window fusion.

                if (existing and number and existing.container_number and
                        number != existing.container_number):
                    if not is_second_slot and is_distinct_container(existing.container_number, number):
                        # Genuine distinct identities on different containers!
                        # NEVER merge into existing, and NEVER attach snapshots across different containers!
                        existing = None
                    else:
                        # Check if this is an OCR variation on the same container (fuzzy match or higher confidence read)
                        ocr_sim = ocr_similarity_score(number, existing.container_number)
                        num_digits = ''.join(c for c in number if c.isdigit())
                        ex_digits = ''.join(c for c in existing.container_number if c.isdigit())
                        digits_overlap = bool(num_digits and ex_digits and (
                            (len(num_digits) >= 6 and len(ex_digits) >= 6 and num_digits[:6] == ex_digits[:6]) or
                            num_digits[-6:] == ex_digits[-6:] or 
                            num_digits in ex_digits or 
                            difflib.SequenceMatcher(None, num_digits, ex_digits).ratio() >= 0.50
                        ))
                        obs_conf = getattr(obs.ocr, 'confidence', 0.0) or 0.0
                        prior_best_conf = session.scalar(
                            select(func.max(OCRResult.confidence))
                            .join(DetectionRecord, OCRResult.detection_id == DetectionRecord.id)
                            .where(DetectionRecord.event_id == existing.id, OCRResult.valid_check_digit == True)
                        ) or 0.0

                        if ocr_sim >= 0.45 or digits_overlap:
                            # Same physical container, OCR read improved or varied!
                            if obs_conf >= prior_best_conf:
                                existing.container_number = number
                                if obs.confirmed:
                                    existing.status = 'CONFIRMED'
                            else:
                                number = existing.container_number
                        else:
                            # Genuine distinct identities on different containers
                            if not is_second_slot:
                                existing = None

                created = existing is None
                event_id = existing.id if existing else (getattr(obs, 'event_id', None) or str(uuid4()))
                if created and session.get(GateEvent, event_id):
                    existing_in_db = session.get(GateEvent, event_id)
                    time_diff = datetime_diff_seconds(existing_in_db.timestamp, obs.detection.frame_timestamp)
                    if existing_in_db.cloud_synced or time_diff > max(dedup_seconds, 60.0) or is_distinct_container(existing_in_db.container_number, number):
                        event_id = str(uuid4())
                        existing = None
                    else:
                        existing = existing_in_db
                        created = False
                        event_id = existing.id
                size_type = getattr(obs.ocr, 'size_type', None)
                if created:
                    if not number and not obs.confirmed:
                        cname = getattr(obs.detection, 'class_name', '')
                        bw = (obs.detection.bbox[2] - obs.detection.bbox[0]) if len(getattr(obs.detection, 'bbox', [])) == 4 else 0
                        bh = (obs.detection.bbox[3] - obs.detection.bbox[1]) if len(getattr(obs.detection, 'bbox', [])) == 4 else 0
                        if cname not in ('container', 'container_object') or bw < 60.0 or bh < 50.0:
                            return event_id, False
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
                    existing.is_twin = False
                    existing.container_number_2 = None
                    existing.iso_code_2 = None
                    all_reads = []
                    for ocr_rec in session.scalars(
                        select(OCRResult)
                        .join(DetectionRecord, OCRResult.detection_id == DetectionRecord.id)
                        .where(DetectionRecord.event_id == existing.id)
                    ).all():
                        all_reads.append(Validation(
                            raw_text=getattr(ocr_rec, 'raw_text', '') or '',
                            normalized_text=getattr(ocr_rec, 'normalized_text', '') or '',
                            confidence=float(getattr(ocr_rec, 'confidence', 0.0) or 0.0),
                            valid_format=bool(getattr(ocr_rec, 'valid_format', False)),
                            valid_check_digit=bool(getattr(ocr_rec, 'valid_check_digit', False)),
                            validation_status=getattr(ocr_rec, 'validation_status', '') or '',
                            size_type=getattr(ocr_rec, 'size_type', None)
                        ))
                    if obs.ocr:
                        all_reads.append(obs.ocr)

                    consensus_val = CharacterConsensus.resolve(all_reads)
                    if consensus_val and consensus_val.normalized_text and consensus_val.valid_check_digit and is_valid_iso_6346(consensus_val.normalized_text):
                        existing.container_number = consensus_val.normalized_text
                        existing.status = 'CONFIRMED'
                        if consensus_val.size_type:
                            existing.iso_code_1 = consensus_val.size_type
                    else:
                        prior_best_conf = session.scalar(
                            select(func.max(OCRResult.confidence))
                            .join(DetectionRecord, OCRResult.detection_id == DetectionRecord.id)
                            .where(DetectionRecord.event_id == existing.id, OCRResult.valid_check_digit == True)
                        ) or 0.0
                        obs_conf = getattr(obs.ocr, 'confidence', 0.0) or 0.0
                        is_obs_valid_iso = bool(number and is_valid_iso_6346(number))
                        if (not existing.container_number and is_obs_valid_iso) or (is_obs_valid_iso and obs_conf >= prior_best_conf) or (obs.confirmed and obs_conf >= prior_best_conf):
                            existing.container_number = number or existing.container_number
                            if is_obs_valid_iso:
                                existing.status = 'CONFIRMED'
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
                    if existing_path_snap.event_id == event_id:
                        existing_path_snap.timestamp = obs.detection.frame_timestamp
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
                                if existing_other_snap.event_id == event_id:
                                    existing_other_snap.timestamp = obs.detection.frame_timestamp
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
                if not created and event_id:
                    existing_ev = session.get(GateEvent, event_id)
                    if existing_ev and existing_ev.cloud_event_id:
                        existing_ev.cloud_delivery_state = 'PUT_PENDING'
                        existing_ev.cloud_synced = False
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

    def mark_cloud_synced(self, event_id: str, cloud_event_id: str | None = None, cloud_visit_id: str | None = None):
        with self.db.sessions.begin() as session:
            ev = session.get(GateEvent, event_id)
            if ev:
                ev.cloud_synced = True
                ev.cloud_delivery_state = 'DELIVERED'
                ev.cloud_delivery_error = None
                ev.cloud_synced_at = utcnow()
                if cloud_event_id:
                    ev.cloud_event_id = str(cloud_event_id)
                if cloud_visit_id:
                    ev.cloud_visit_id = str(cloud_visit_id)

    def mark_cloud_delivery(self, event_id: str, state: str, error: str | None = None):
        valid_states = (
            'PENDING', 'OFFLINE', 'REJECTED', 'CONFLICT', 'BLOCKED', 'DUPLICATE',
            'EVENT_CREATED', 'POST_PENDING', 'PUT_PENDING', 'PUT_FAILED', 'DELIVERED', 'NOT_FOUND'
        )
        if state not in valid_states:
            raise ValueError(f'Invalid cloud delivery state: {state}')
        with self.db.sessions.begin() as session:
            ev = session.get(GateEvent, event_id)
            if ev:
                if ev.cloud_synced and ev.cloud_delivery_state == 'DELIVERED' and state != 'DELIVERED':
                    return
                ev.cloud_delivery_state = state
                ev.cloud_delivery_error = error[:256] if error else None
                if state == 'DELIVERED':
                    ev.cloud_synced = True
                    ev.cloud_synced_at = utcnow()
                else:
                    ev.cloud_synced = False
                    ev.cloud_synced_at = None
                    if state in ('REJECTED', 'BLOCKED', 'DUPLICATE'):
                        ev.cloud_event_id = None

    def set_cloud_event_id(self, event_id: str, cloud_event_id: str, state: str = 'EVENT_CREATED', cloud_visit_id: str | None = None):
        with self.db.sessions.begin() as session:
            ev = session.get(GateEvent, event_id)
            if ev:
                ev.cloud_event_id = str(cloud_event_id)
                if cloud_visit_id:
                    ev.cloud_visit_id = str(cloud_visit_id)
                ev.cloud_delivery_state = state
                ev.cloud_delivery_error = None

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
                    func.coalesce(GateEvent.cloud_delivery_state, 'PENDING').in_([
                        'PENDING', 'OFFLINE', 'CONFLICT', 'EVENT_CREATED', 'PUT_PENDING', 'PUT_FAILED', 'POST_PENDING'
                    ]),
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

    def clean_duplicate_events(self, max_gap_seconds: int = 60, snapshots=None):
        """
        Merges and removes duplicate events created for the same container within dwell windows.
        Re-parents detections and snapshots strictly within the passage window, resolving true consensus.
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
                if len(cntr) == 11 and cntr[4:].isdigit():
                    key = f"{ev.gate_id}:{cntr[4:10]}"
                else:
                    key = f"{ev.gate_id}:{cntr}"
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
                    if gap <= max_gap_seconds and not is_distinct_container(ev.container_number, last_ev.container_number):
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

                    # Resolve authoritative container number across all reads in cluster via multi-read consensus
                    all_reads = []
                    for ev_member in cluster:
                        for ocr_rec in session.scalars(
                            select(OCRResult)
                            .join(DetectionRecord, OCRResult.detection_id == DetectionRecord.id)
                            .where(DetectionRecord.event_id == ev_member.id)
                        ).all():
                            all_reads.append(Validation(
                                raw_text=getattr(ocr_rec, 'raw_text', '') or '',
                                normalized_text=getattr(ocr_rec, 'normalized_text', '') or '',
                                confidence=float(getattr(ocr_rec, 'confidence', 0.0) or 0.0),
                                valid_format=bool(getattr(ocr_rec, 'valid_format', False)),
                                valid_check_digit=bool(getattr(ocr_rec, 'valid_check_digit', False)),
                                validation_status=getattr(ocr_rec, 'validation_status', '') or '',
                                size_type=getattr(ocr_rec, 'size_type', None)
                            ))
                    best_c = CharacterConsensus.resolve(all_reads)
                    if best_c and best_c.normalized_text and best_c.valid_check_digit and is_valid_iso_6346(best_c.normalized_text):
                        primary.container_number = best_c.normalized_text
                        primary.status = 'CONFIRMED'
                        if best_c.size_type:
                            primary.iso_code_1 = best_c.size_type

                    for dup in cluster[1:]:
                        session.execute(
                            update(DetectionRecord)
                            .where(DetectionRecord.event_id == dup.id)
                            .values(event_id=primary.id)
                        )
                        # Re-parent only snapshots within the true passage window (<= 25s) to prevent cross-contamination
                        for snap in session.scalars(select(Snapshot).where(Snapshot.event_id == dup.id)).all():
                            if abs(datetime_diff_seconds(snap.timestamp, primary.timestamp)) <= 25.0:
                                snap.event_id = primary.id
                            else:
                                session.delete(snap)
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

                    c1, c2 = ev1.container_number or '', ev2.container_number or ''
                    if is_distinct_container(c1, c2):
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

                    ocr_sim = ocr_similarity_score(c1, c2)
                    c1_digits = ''.join(c for c in c1 if c.isdigit())
                    c2_digits = ''.join(c for c in c2 if c.isdigit())
                    serial_match = bool(len(c1_digits) >= 6 and len(c2_digits) >= 6 and c1_digits[:6] == c2_digits[:6])
                    if gap <= 35.0 and (c1 == c2 or (c1 and c2 and (ocr_sim >= 0.80 or serial_match))):
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
                func.coalesce(GateEvent.cloud_delivery_state, 'PENDING') != 'DUPLICATE',
                GateEvent.timestamp <= event.timestamp,
                GateEvent.timestamp >= event.timestamp - delta,
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
                    if is_distinct_container(num, c_num):
                        continue
                    ocr_sim = ocr_similarity_score(num, c_num)
                    num_digits = ''.join(c for c in num if c.isdigit())
                    c_digits = ''.join(c for c in c_num if c.isdigit())
                    serial_match = bool(len(num_digits) >= 6 and len(c_digits) >= 6 and num_digits[:6] == c_digits[:6])
                    if not (time_diff <= 45.0 and (ocr_sim >= 0.70 or serial_match)):
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
                                    if cand.timestamp <= event.timestamp:
                                        return cand.id
                        except Exception:
                            continue

                # Fuzzy OCR match or serial match within passage window
                ocr_sim = ocr_similarity_score(event.container_number, cand.container_number)
                num_digits = ''.join(c for c in num if c.isdigit())
                c_digits = ''.join(c for c in c_num if c.isdigit())
                serial_match = bool(len(num_digits) >= 6 and len(c_digits) >= 6 and num_digits[:6] == c_digits[:6])
                if time_diff <= 45.0 and (num == c_num or (num and c_num and (ocr_sim >= 0.70 or serial_match))):
                    if cand.timestamp <= event.timestamp:
                        return cand.id

            return event_id
