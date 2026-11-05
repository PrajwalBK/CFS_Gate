from pathlib import Path
from typing import Optional
import datetime
import math
import os
import cv2


class SnapshotManager:
    def __init__(self, root, evidence_root=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        from app.snapshots.evidence_manager import EvidenceFolderManager
        ev_root = Path(evidence_root).resolve() if evidence_root else (self.root.parent / 'evidence')
        self.evidence_manager = EvidenceFolderManager(root=ev_root, legacy_root=self.root)

    def path(self, relative):
        rel_str = str(relative).replace('\\', '/')
        if rel_str.startswith('/'):
            rel_str = rel_str.lstrip('/')
        # 1. Resolve via EvidenceFolderManager if file exists
        resolved = self.evidence_manager.resolve_path(rel_str)
        if resolved.is_file():
            return resolved

        # 2. If relative starts with GATE_IN or GATE_OUT
        if rel_str.startswith(('GATE_IN/', 'GATE_OUT/')):
            cand = (self.evidence_manager.evidence_root / rel_str).resolve()
            if cand.is_file():
                return cand

        # 3. Default resolution against legacy root
        path = (self.root / rel_str).resolve()
        if not path.is_relative_to(self.root) and not path.is_relative_to(self.evidence_manager.evidence_root):
            raise ValueError('Invalid snapshot reference')
        return path

    def ensure_event_dir(self, event_id: str, gate_id: str = '', direction: str = '', container_number: Optional[str] = None) -> Path:
        return self.evidence_manager.get_or_create_folder(str(event_id), gate_id=gate_id, direction=direction, container_number=container_number)

    @staticmethod
    def compute_quality(image) -> float:
        import cv2, math
        if image is None or image.size == 0:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        exposure = float(((gray > 15) & (gray < 240)).mean())
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return math.log1p(variance) * exposure

    @staticmethod
    def _capture_has_visible_container(snapshot: dict, event_record: dict, image) -> bool:
        """Require a meaningful YOLO container-number box for proof images.

        Whole-frame sharpness is not sufficient proof quality: a sharp frame can
        contain only a tiny sliver of the truck/container.  Match detections by
        camera and capture time and reject boxes that are too small or clipped.
        """
        try:
            h, w = image.shape[:2]
            if h <= 0 or w <= 0:
                return False
            snap_ts = snapshot.get('timestamp')
            def parse(value):
                if not value:
                    return None
                text = str(value).replace('Z', '+00:00')
                try:
                    return datetime.datetime.fromisoformat(text)
                except ValueError:
                    return None
            snap_dt = parse(snap_ts)
            camera = snapshot.get('camera_id')
            all_dets = event_record.get('detections')
            if all_dets is None:
                return True
            prov = snapshot.get('provenance') or {}
            prov_dets = prov.get('container_detections') or []
            detections = [d for d in all_dets if d.get('camera_id') == camera] or prov_dets
            if not detections:
                # If no container detections recorded specifically for this camera, reject it!
                # Never accept empty background, building, or pedestrian frames as container proof.
                return False

            candidates = []
            for detection in detections:
                cname = (detection.get('class_name') or '').lower()
                if cname not in ('container_number', 'iso_code', 'container', 'cointainer', 'container_object'):
                    continue
                det_dt = parse(detection.get('timestamp') if isinstance(detection, dict) else getattr(detection, 'frame_timestamp', None))
                if snap_dt and det_dt:
                    if abs((snap_dt - det_dt).total_seconds()) > 35.0:
                        continue
                bbox = detection.get('bbox') if isinstance(detection, dict) else getattr(detection, 'bbox', [])
                if len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = [float(v) for v in bbox]
                if x2 <= 1.0 and y2 <= 1.0 and w > 1 and h > 1:
                    x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
                bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
                area_ratio = (bw * bh) / float(w * h)

                if cname == 'container_number' and (bw < 30.0 or bh < 10.0 or area_ratio < 0.0003):
                    continue
                if cname == 'iso_code' and (bw < 15.0 or bh < 8.0 or area_ratio < 0.0001):
                    continue
                if cname in ('container', 'cointainer', 'container_object') and (bw < 40.0 or bh < 20.0 or area_ratio < 0.003):
                    continue
                candidates.append((area_ratio, bw, bh))
            return bool(candidates) if detections else True
        except Exception:
            return False

    def build_evidence_manifest(self, event_record: dict, min_quality: float = 2.5) -> dict:
        """Select one usable camera capture per event/camera for UI and cloud."""
        snapshots = event_record.get('snapshots') or []
        by_camera = {}
        for snap in snapshots:
            camera_id, image_path = snap.get('camera_id'), snap.get('image_path')
            if not camera_id or not image_path:
                continue
            try:
                import cv2, numpy as np
                data = self.path(image_path).read_bytes()
                prov = snap.get('provenance') or {}
                if prov.get('image_sha256'):
                    import hashlib
                    if hashlib.sha256(data).hexdigest() != prov['image_sha256']:
                        continue
                decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if decoded is None or decoded.size == 0 or decoded.shape[0] < 240 or decoded.shape[1] < 320:
                    continue
                # Passage window scoping: reject snapshots captured outside the event passage window (> 30s)
                ev_ts = event_record.get('timestamp')
                snap_ts = snap.get('timestamp')
                if ev_ts and snap_ts:
                    def _parse_ts(v):
                        if not v:
                            return None
                        if isinstance(v, datetime.datetime):
                            return v.replace(tzinfo=None)
                        try:
                            clean_str = str(v).replace('Z', '').split('+')[0]
                            return datetime.datetime.fromisoformat(clean_str)
                        except Exception:
                            return None
                    dt1, dt2 = _parse_ts(ev_ts), _parse_ts(snap_ts)
                    if dt1 and dt2 and abs((dt2 - dt1).total_seconds()) > 30.0:
                        continue

                gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
                quality = self.compute_quality(decoded)
                if float(gray.std()) < 8.0 or quality < min_quality:
                    continue
                if not self._capture_has_visible_container(snap, event_record, decoded):
                    continue
                by_camera.setdefault(camera_id, []).append({**snap, 'quality_score': round(float(quality), 3)})
            except Exception:
                continue
        # The event writer may have already selected a frame because it produced
        # the accepted OCR result.  That frame is the proof of record and must
        # not be replaced by a different (possibly sharper but text-invisible)
        # frame from the same camera.  Otherwise the local OCR result and the
        # cloud proof image describe different moments.
        preferred_ids = {
            value for value in (
                event_record.get('primary_snapshot_id'),
                event_record.get('rear_snapshot_id'),
            ) if value
        }
        captures = []
        for items in by_camera.values():
            preferred = [item for item in items if item.get('id') in preferred_ids]
            captures.append(preferred[0] if preferred else max(
                items, key=lambda item: (item['quality_score'], item.get('timestamp') or '')
            ))
        captures.sort(key=lambda item: (item.get('camera_id') or '', item.get('timestamp') or ''))
        ids = {item['id'] for item in captures}
        primary = event_record.get('primary_snapshot_id') if event_record.get('primary_snapshot_id') in ids else (captures[0]['id'] if captures else None)
        rear = event_record.get('rear_snapshot_id') if event_record.get('rear_snapshot_id') in ids else next((item['id'] for item in captures if (item.get('image_path') or '').endswith('_rear.jpg')), None)
        return {'version': 1, 'event_id': event_record.get('id'), 'captures': captures,
                'selected_snapshot_ids': [item['id'] for item in captures],
                'primary_snapshot_id': primary, 'rear_snapshot_id': rear}

    @staticmethod
    def compute_similarity(img1, img2) -> float:
        """
        Fast zero-mean normalized cross-correlation (NCC) of two images (0.0 to 1.0).
        Accepts numpy arrays, file paths (str/Path), or image bytes.
        Downsamples to 64x64 grayscale for fast sub-3ms comparison.
        """
        import cv2, numpy as np
        if isinstance(img1, (str, Path)):
            p = str(img1)
            if not os.path.exists(p):
                return 0.0
            img1 = cv2.imread(p)
        elif isinstance(img1, bytes):
            img1 = cv2.imdecode(np.frombuffer(img1, dtype=np.uint8), cv2.IMREAD_COLOR)

        if isinstance(img2, (str, Path)):
            p = str(img2)
            if not os.path.exists(p):
                return 0.0
            img2 = cv2.imread(p)
        elif isinstance(img2, bytes):
            img2 = cv2.imdecode(np.frombuffer(img2, dtype=np.uint8), cv2.IMREAD_COLOR)

        if img1 is None or img2 is None or getattr(img1, 'size', 0) == 0 or getattr(img2, 'size', 0) == 0:
            return 0.0

        try:
            s1 = cv2.resize(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), (64, 64)).astype(np.float32)
            s2 = cv2.resize(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), (64, 64)).astype(np.float32)
            s1 -= s1.mean()
            s2 -= s2.mean()
            n1 = float(np.linalg.norm(s1))
            n2 = float(np.linalg.norm(s2))
            if n1 <= 1e-6 or n2 <= 1e-6:
                return 0.0
            return float(np.clip((s1 * s2).sum() / (n1 * n2), 0.0, 1.0))
        except Exception:
            return 0.0

    def save_side_snapshot(self, event_id: str, camera_id: str, image, direction: str = '', text: str = '', force: bool = False) -> str:
        """Saves or upgrades the compulsory side snapshot for camera_id under the single evidence folder."""
        folder = self.ensure_event_dir(event_id, direction=direction, container_number=text if text else None)
        relative = f'{event_id}/{camera_id}.jpg'
        final = folder / f'{camera_id}.jpg'
        meta_quality_path = final.with_suffix('.quality')

        new_quality = self.compute_quality(image)
        if not force and final.exists() and meta_quality_path.exists():
            try:
                old_quality = float(meta_quality_path.read_text(encoding='utf-8').strip())
                if new_quality <= old_quality:
                    return relative
            except Exception:
                pass

        ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise OSError('Snapshot encoding failed')

        temp_img = final.with_suffix('.tmp')
        try:
            with temp_img.open('wb') as f:
                f.write(encoded.tobytes())
            os.replace(temp_img, final)
            meta_quality_path.write_text(f'{new_quality:.2f}', encoding='utf-8')
        finally:
            temp_img.unlink(missing_ok=True)
        return relative

    def save_best_crop(self, event_id: str, slot_pos: str, crop, text: str = '', force: bool = False) -> str:
        """Saves or upgrades the highest-clarity OCR proof crop under snapshots/{event_id}/crop_{slot}_best.jpg."""
        import cv2
        self.ensure_event_dir(event_id)
        slot_clean = slot_pos.lower().replace(' ', '_')
        relative = f'{event_id}/crop_{slot_clean}_best.png'
        final = self.path(relative)
        meta_quality_path = final.with_suffix('.quality')

        new_quality = self.compute_quality(crop)
        if not force and final.exists() and meta_quality_path.exists():
            try:
                old_quality = float(meta_quality_path.read_text().strip())
                if new_quality <= old_quality:
                    return relative
            except Exception:
                pass

        crop_img = crop.copy()
        if text:
            cv2.putText(crop_img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 0), 2)
        ok, encoded = cv2.imencode('.jpg', crop_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise OSError('Crop encoding failed')

        temp_crop = final.with_suffix('.tmp')
        try:
            with temp_crop.open('wb') as f:
                f.write(encoded.tobytes())
            os.replace(temp_crop, final)
            meta_quality_path.write_text(f'{new_quality:.2f}', encoding='utf-8')
        finally:
            temp_crop.unlink(missing_ok=True)
        return relative

    def save(self, snapshot_id, observation, event_id=None):
        import cv2
        image = observation.image.copy()

        is_rear = getattr(observation, 'slot_position', 'FRONT') == 'REAR' or observation.origin_key.endswith(':rear')
        slot_name = "REAR (SLOT 2)" if is_rear else "FRONT (SLOT 1)"
        container_num = observation.ocr.normalized_text if (observation.ocr and observation.ocr.normalized_text) else ""
        box_color = (0, 200, 255) if is_rear else (40, 190, 80)

        # Save clean raw camera photo without drawing boxes over text
        # (Overlays over the numbers obscure human verification on the portal)
        ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise OSError('Snapshot encoding failed')
        is_rear = getattr(observation, 'slot_position', 'FRONT') == 'REAR' or observation.origin_key.endswith(':rear')
        slot_suffix = '_rear' if is_rear else ''
        if event_id:
            direction = getattr(observation, 'direction', '')
            gate_id = getattr(observation, 'gate_id', '')
            container_num = observation.ocr.normalized_text if (observation.ocr and observation.ocr.normalized_text) else ""
            folder = self.ensure_event_dir(event_id, gate_id=gate_id, direction=direction, container_number=container_num if getattr(observation, 'confirmed', False) else None)
            relative = f'{event_id}/{observation.detection.camera_id}_{snapshot_id}{slot_suffix}.jpg'
            final = folder / f'{observation.detection.camera_id}_{snapshot_id}{slot_suffix}.jpg'
        else:
            relative = f'{snapshot_id}.jpg'
            final = self.path(relative)
            final.parent.mkdir(parents=True, exist_ok=True)
        quality_path = final.with_suffix('.quality')
        new_quality = self.compute_quality(image)
        if final.exists() and quality_path.exists():
            try:
                if new_quality <= float(quality_path.read_text(encoding='utf-8')):
                    return relative
            except (OSError, ValueError):
                pass
        temporary = final.with_suffix('.tmp')
        try:
            with temporary.open('wb') as file:
                file.write(encoded.tobytes())
            os.replace(temporary, final)
            quality_path.write_text(f'{new_quality:.2f}', encoding='utf-8')
        finally:
            temporary.unlink(missing_ok=True)
        return relative

    def save_camera_frame(self, snapshot_id, event_id, camera_id, image, direction='', text=''):
        return self.save_side_snapshot(event_id, camera_id, image, direction=direction, text=text)

    def save_event_json(self, event_id: str, event_data: dict) -> Path:
        manifest = self.build_evidence_manifest(event_data)
        event_path, _ = self.evidence_manager.save_event_json(
            event_id=event_id,
            event_data=event_data,
            manifest_data=manifest
        )
        return event_path

    def remove(self, relative):
        path = self.path(relative)
        path.unlink(missing_ok=True)
        if path.parent != self.root:
            try:
                path.parent.rmdir()
            except OSError:
                pass
