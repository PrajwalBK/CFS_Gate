from dataclasses import asdict
from datetime import datetime, timedelta
from threading import Lock
from uuid import uuid4
from sqlalchemy import func, select

from app.database.models import Camera, DetectionRecord, GateEvent, OCRResult, Snapshot, SystemLog
from app.domain import Observation


def serialize(record):
    result = {column.name: getattr(record, column.name) for column in record.__table__.columns}
    return {key: value.isoformat() + 'Z' if isinstance(value, datetime) else value for key, value in result.items()}


class GateRepository:
    def __init__(self, db):
        self.db = db
        self._write_lock = Lock()

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

    def save_observation(self, obs: Observation, snapshots, dedup_seconds: float, additional_frames: dict | None = None, ongoing_event_id=None):
        """One writer per service; DB uniqueness also protects retried track writes."""
        created_paths = []
        try:
            with self._write_lock, self.db.sessions.begin() as session:
                prior = session.scalar(select(DetectionRecord).where(DetectionRecord.origin_key == obs.origin_key))
                if prior:
                    return prior.event_id, False
                number = obs.ocr.normalized_text if obs.confirmed else None
                delta = timedelta(seconds=dedup_seconds)
                existing = None

                direction_match = (
                    (GateEvent.event_type == obs.direction) | (GateEvent.event_type == 'UNKNOWN')
                    if obs.direction != 'UNKNOWN'
                    else True
                )

                is_rear_slot = getattr(obs, 'slot_position', 'FRONT') == 'REAR' or obs.origin_key.endswith(':rear')

                if number:
                    existing = session.scalar(select(GateEvent).where(
                        GateEvent.gate_id == obs.gate_id,
                        (GateEvent.container_number == number) |
                        (GateEvent.container_number_2 == number) |
                        (GateEvent.container_number_2.is_(None) if is_rear_slot else GateEvent.container_number.is_(None)),
                        direction_match,
                        GateEvent.timestamp >= obs.detection.frame_timestamp - delta,
                        GateEvent.timestamp <= obs.detection.frame_timestamp + delta,
                    ).order_by(GateEvent.timestamp.desc()))
                    if existing:
                        if is_rear_slot:
                            existing.is_twin = True
                            if (number != existing.container_number or not existing.container_number) and (not existing.container_number_2 or obs.confirmed):
                                existing.container_number_2 = number
                                existing.iso_code_2 = getattr(obs.ocr, 'size_type', None) or '22G1'
                        else:
                            if (number != existing.container_number_2 or not existing.container_number_2) and (not existing.container_number or obs.confirmed):
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

                if obs.origin_key.startswith('dwell:') and existing and not is_rear_slot:
                    prior_view = session.scalar(select(DetectionRecord.id).where(
                        DetectionRecord.event_id == existing.id,
                        DetectionRecord.camera_id == obs.detection.camera_id,
                        DetectionRecord.origin_key.like('dwell:%'),
                        ~DetectionRecord.origin_key.like('%:rear'),
                        DetectionRecord.origin_key != obs.origin_key))
                    if prior_view:
                        existing = None  # New cleared passage, not time-window fusion.

                created = existing is None
                event_id = existing.id if existing else (getattr(obs, 'event_id', None) or str(uuid4()))
                if created and session.get(GateEvent, event_id):
                    existing = session.get(GateEvent, event_id)
                    created = False
                    event_id = existing.id
                size_type = getattr(obs.ocr, 'size_type', None)
                size_str = str(size_type or '')
                is_40ft = size_str.startswith('4') or '40' in size_str or '45' in size_str
                is_twin_event = (is_rear_slot or obs.origin_key.endswith(':front')) and not is_40ft
                if created:
                    session.add(GateEvent(
                        id=event_id, gate_id=obs.gate_id, timestamp=obs.detection.frame_timestamp,
                        container_number=number if not is_rear_slot else None,
                        container_number_2=number if is_rear_slot else None,
                        iso_code_1=getattr(obs.ocr, 'size_type', None) if not is_rear_slot else None,
                        iso_code_2=getattr(obs.ocr, 'size_type', None) if is_rear_slot else None,
                        is_twin=is_twin_event,
                        event_type=obs.direction,
                        status='CONFIRMED' if (obs.confirmed and not is_twin_event) else 'NEEDS_REVIEW',
                    ))
                    session.flush()
                else:
                    prior_best_conf = session.scalar(
                        select(func.max(OCRResult.confidence))
                        .join(DetectionRecord, OCRResult.detection_id == DetectionRecord.id)
                        .where(DetectionRecord.event_id == existing.id, OCRResult.valid_check_digit == True)
                    ) or 0.0
                    obs_conf = getattr(obs.ocr, 'confidence', 0.0) or 0.0
                    if is_40ft:
                        existing.is_twin = False
                        existing.container_number_2 = None
                        existing.iso_code_2 = None
                        if obs.confirmed:
                            if not existing.container_number or obs_conf >= prior_best_conf:
                                existing.container_number = number
                                existing.iso_code_1 = size_type or existing.iso_code_1
                            existing.status = 'CONFIRMED'
                        elif not existing.container_number:
                            existing.container_number = number
                            existing.iso_code_1 = size_type or existing.iso_code_1
                    elif is_rear_slot:
                        existing.is_twin = True
                        if (number != existing.container_number or not existing.container_number) and (not existing.container_number_2 or obs.confirmed or obs_conf >= prior_best_conf):
                            existing.container_number_2 = number
                            existing.iso_code_2 = getattr(obs.ocr, 'size_type', None) or '22G1'
                    else:
                        if (number != existing.container_number_2 or not existing.container_number_2) and (not existing.container_number or (obs.confirmed and obs_conf >= prior_best_conf)):
                            existing.container_number = number
                            existing.iso_code_1 = getattr(obs.ocr, 'size_type', None)
                        if obs.origin_key.endswith(':front'):
                            existing.is_twin = True
                    if existing.is_twin:
                        if existing.container_number and existing.container_number_2:
                            existing.status = 'CONFIRMED'
                    elif obs.confirmed:
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
                    Snapshot.event_id == event_id,
                    Snapshot.image_path == image_path
                ))
                if existing_path_snap:
                    existing_path_snap.timestamp = obs.detection.frame_timestamp
                else:
                    created_paths.append(image_path)
                    session.add(Snapshot(id=snapshot_id, event_id=event_id, camera_id=obs.detection.camera_id,
                                         image_path=image_path, timestamp=obs.detection.frame_timestamp))

                # 2. Multi-camera snapshots for all other cameras covering this gate event
                if additional_frames:
                    for other_cam_id, other_img in additional_frames.items():
                        if other_cam_id == obs.detection.camera_id or other_img is None:
                            continue
                        already_saved = session.scalar(select(Snapshot).where(
                            Snapshot.event_id == event_id,
                            Snapshot.camera_id == other_cam_id
                        ))
                        if not already_saved:
                            other_snap_id = str(uuid4())
                            other_path = snapshots.save_camera_frame(
                                other_snap_id, event_id, other_cam_id, other_img,
                                direction=obs.direction,
                                text=obs.ocr.normalized_text if obs.confirmed else ''
                            )
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
            # Suppress partial twin events: only show twin events when BOTH containers have been read and updated
            query = query.where(~((GateEvent.is_twin == True) & ((GateEvent.container_number.is_(None)) | (GateEvent.container_number_2.is_(None)))))
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
                    item['container_size'] = '20 FT'
                    item['container_size_2'] = '20 FT'
                    item['size_code'] = item.get('iso_code_1') or '22G1'
                    item['size_code_2'] = item.get('iso_code_2') or '22G1'
                else:
                    item['container_size'] = parsed_size or ('40 FT HC' if size_code == '45G1' else ('40 FT' if (size_code or '').startswith('4') else ('20 FT' if (size_code or '').startswith('2') else ('FEET' if has_feet else None))))
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
            # Select best primary camera based on OCR validation quality, check digit, and confidence:
            best_cam = None
            best_score = -1.0
            for d in detections:
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
                if any(k in (d.camera_id or '').lower() for k in ('-3', 'rear', '-2', 'side')):
                    score += 0.05
                if score > best_score:
                    best_score = score
                    best_cam = d.camera_id

            primary_cam = best_cam or (detections[0].camera_id if detections else (snapshots_list[0]['camera_id'] if snapshots_list else None))
            front_snap = next((s for s in snapshots_list if not (s.get('image_path') or '').endswith('_rear.jpg') and s['camera_id'] == primary_cam), (snapshots_list[0] if snapshots_list else None))
            rear_snap = next((s for s in snapshots_list if (s.get('image_path') or '').endswith('_rear.jpg')), None)
            result['primary_camera_id'] = primary_cam
            result['primary_snapshot_id'] = front_snap['id'] if front_snap else None
            result['rear_snapshot_id'] = rear_snap['id'] if rear_snap else None
            result['snapshots'] = snapshots_list
            
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
                result['container_size'] = '20 FT'
                result['container_size_2'] = '20 FT'
                result['size_code'] = result.get('iso_code_1') or '22G1'
                result['size_code_2'] = result.get('iso_code_2') or '22G1'
            else:
                result['container_size'] = container_size or ('40 FT HC' if size_code == '45G1' else ('40 FT' if (size_code or '').startswith('4') else ('20 FT' if (size_code or '').startswith('2') else None)))
                result['size_code'] = size_code or result.get('iso_code_1')
            return result

    def ocr_results(self, limit=50, offset=0):
        with self.db.sessions() as session:
            return [serialize(o) for o in session.scalars(select(OCRResult).order_by(OCRResult.timestamp.desc()).limit(limit).offset(offset))]

    def snapshot(self, snapshot_id):
        with self.db.sessions() as session:
            row = session.get(Snapshot, snapshot_id)
            return serialize(row) if row else None
