from pathlib import Path
import os


class SnapshotManager:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise ValueError('Invalid snapshot reference')
        return path

    def ensure_event_dir(self, event_id: str) -> Path:
        event_dir = self.root / str(event_id)
        event_dir.mkdir(parents=True, exist_ok=True)
        return event_dir

    @staticmethod
    def compute_quality(image) -> float:
        import cv2, math
        if image is None or image.size == 0:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        exposure = float(((gray > 15) & (gray < 240)).mean())
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return math.log1p(variance) * exposure

    def save_side_snapshot(self, event_id: str, camera_id: str, image, direction: str = '', text: str = '', force: bool = False) -> str:
        """Saves or upgrades the compulsory side snapshot for camera_id under snapshots/{event_id}/{camera_id}.jpg."""
        import cv2
        self.ensure_event_dir(event_id)
        relative = f'{event_id}/{camera_id}.jpg'
        final = self.path(relative)
        meta_quality_path = final.with_suffix('.quality')

        new_quality = self.compute_quality(image)
        if not force and final.exists() and meta_quality_path.exists():
            try:
                old_quality = float(meta_quality_path.read_text().strip())
                if new_quality <= old_quality:
                    return relative  # Existing snapshot is sharper/clearer
            except Exception:
                pass

        img = image.copy()
        label = f'{camera_id}' + (f' | {direction}' if direction else '') + (f' | {text}' if text else '')
        cv2.putText(img, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .75, (40, 190, 80), 2)
        ok, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
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

        # 1. Draw extra detections on this frame (such as Feet)
        if getattr(observation, 'all_detections', None):
            for d in observation.all_detections:
                if d.class_name != observation.detection.class_name:
                    fx1, fy1, fx2, fy2 = map(int, d.bbox)
                    cv2.rectangle(image, (fx1, fy1), (fx2, fy2), (0, 215, 255), 2)
                    flabel = f'{d.class_name.upper()} ({int(d.confidence * 100)}%)'
                    cv2.putText(image, flabel, (max(5, fx1), max(20, fy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 215, 255), 2)

        # 2. Draw primary container detection cleanly
        x1, y1, x2, y2 = map(int, observation.detection.bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)
        label = f"{observation.detection.camera_id} | {slot_name}" + (f" | {container_num}" if container_num else "")
        # Black outline + white text for high contrast without blocking the image
        text_y = max(24, y1 - 8)
        cv2.putText(image, label, (max(5, x1), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(image, label, (max(5, x1), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # 3. Top-left unobtrusive watermark badge
        top_label = f"PROOF: {slot_name}" + (f" - {container_num}" if container_num else "")
        cv2.putText(image, top_label, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3)
        cv2.putText(image, top_label, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, box_color, 2)

        ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise OSError('Snapshot encoding failed')
        is_rear = getattr(observation, 'slot_position', 'FRONT') == 'REAR' or observation.origin_key.endswith(':rear')
        slot_suffix = '_rear' if is_rear else ''
        relative = f'{event_id}/{observation.detection.camera_id}{slot_suffix}.jpg' if event_id else f'{snapshot_id}.jpg'
        final = self.path(relative)
        final.parent.mkdir(parents=True, exist_ok=True)
        temporary = final.with_suffix('.tmp')
        try:
            with temporary.open('wb') as file:
                file.write(encoded.tobytes())
            os.replace(temporary, final)
        finally:
            temporary.unlink(missing_ok=True)
        return relative

    def save_camera_frame(self, snapshot_id, event_id, camera_id, image, direction='', text=''):
        return self.save_side_snapshot(event_id, camera_id, image, direction=direction, text=text)

    def save_event_json(self, event_id: str, event_data: dict) -> Path:
        import json
        event_dir = self.root / str(event_id)
        event_dir.mkdir(parents=True, exist_ok=True)
        final_path = event_dir / 'event.json'
        temp_path = event_dir / 'event.json.tmp'
        try:
            with temp_path.open('w', encoding='utf-8') as f:
                json.dump(event_data, f, indent=2, default=str)
            os.replace(temp_path, final_path)
        finally:
            temp_path.unlink(missing_ok=True)
        return final_path

    def remove(self, relative):
        path = self.path(relative)
        path.unlink(missing_ok=True)
        if path.parent != self.root:
            try:
                path.parent.rmdir()
            except OSError:
                pass

