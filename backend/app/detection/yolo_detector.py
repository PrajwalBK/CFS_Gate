import math
from app.domain import Detection, Frame


class YoloDetector:
    def __init__(self, settings, model=None):
        self.settings = settings
        if model is None:
            expected = {'pytorch': '.pt', 'onnx': '.onnx', 'tensorrt': '.engine'}[settings.model_backend]
            if settings.model_path.suffix != expected or not settings.model_path.is_file():
                raise ValueError('Model file missing or backend does not match extension')
            from ultralytics import YOLO
            model = YOLO(str(settings.model_path), task='detect')
        self.model = model
    def detect_vehicles(self, frame):
        """Truck and vehicle detection is permanently disabled.
        The system detects and processes ONLY container objects using models/best.pt.
        """
        return []

    def detect(self, frame: Frame) -> list[Detection]:
        results = self.model.predict(source=frame.image, conf=self.settings.confidence_threshold,
                                     imgsz=self.settings.model_image_size, device=self.settings.model_device,
                                     verbose=False, rect=False)
        height, width = frame.image.shape[:2]
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            EXCLUDED_NON_CONTAINER_CLASSES = {
                'person', 'pedestrian', 'human', 'people', 'man', 'woman',
                'truck', 'car', 'vehicle', 'trailer', 'bus', 'motorcycle', 'bicycle',
                'van', 'pickup', 'cab', 'tractor', 'semi', 'wheel', 'tire',
                'license_plate', 'plate', 'driver', 'worker', 'traffic_light', 'stop_sign'
            }
            for box in result.boxes:
                raw_name = str(result.names[int(box.cls.item())]).lower()
                if raw_name in EXCLUDED_NON_CONTAINER_CLASSES:
                    continue
                name = self.settings.class_aliases.get(raw_name, raw_name)
                if name in EXCLUDED_NON_CONTAINER_CLASSES:
                    continue
                if name == 'container_object':
                    name = 'container'
                elif name == 'cointainer':
                    name = 'container_number'
                elif name == 'feet':
                    name = 'iso_code'
                # Strictly detect and process only container objects, container numbers, and ISO codes.
                # Non-container objects (trucks, vehicles, people, trailers, cars, etc.) are ignored.
                if name not in ('container', 'container_number', 'iso_code'):
                    continue
                confidence = float(box.conf.item())
                # Dynamic confidence based on configuration to avoid dropping low-light/distant containers
                cfg_conf = float(getattr(self.settings, 'confidence_threshold', 0.35))
                min_container_conf = min(cfg_conf, 0.35)
                min_text_conf = min(cfg_conf, 0.28)

                if name == 'container' and confidence < min_container_conf:
                    continue
                if name in ('container_number', 'iso_code') and confidence < min_text_conf:
                    continue
                coords = [float(v) for v in box.xyxy[0].tolist()]
                if not all(math.isfinite(v) for v in coords + [confidence]):
                    continue
                x1, y1, x2, y2 = coords
                w = x2 - x1
                h = y2 - y1

                # Physical geometry sanity check:
                # 1. Containers are large structures.
                # 2. Reject vertical slivers (e.g. standing pedestrians or poles falsely classified as containers)
                #    ONLY when the box is fully within the frame. When entering/exiting, the box is clipped
                #    by the camera boundary and will naturally have a narrow aspect ratio.
                is_boundary_clipped = (x1 <= 5.0 or y1 <= 5.0 or x2 >= (width - 5.0) or y2 >= (height - 5.0))
                if name == 'container':
                    if w < 45.0 or h < 35.0 or (w * h) < 1800.0:
                        continue
                    if not is_boundary_clipped and (w / max(h, 1.0)) < 0.50:
                        continue
                elif name == 'container_number':
                    if w < 20.0 or h < 8.0 or (w * h) < 180.0:
                        continue
                elif name == 'iso_code':
                    if w < 12.0 or h < 6.0 or (w * h) < 100.0:
                        continue

                bbox = (max(0., x1), max(0., y1), min(float(width), x2), min(float(height), y2))
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    detections.append(Detection(frame.camera_id, frame.timestamp, name, confidence, bbox))
        return detections
