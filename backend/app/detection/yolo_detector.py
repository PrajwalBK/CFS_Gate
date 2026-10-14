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
        self.vehicle_model = None

    def detect_vehicles(self, frame):
        """Local auxiliary truck model disabled.
        The system detects and processes ONLY container objects.
        """
        # Non-container detection and processing code disabled:
        # if self.vehicle_model is None:
        #     path = self.settings.passage_vehicle_model_path
        #     if not path.is_file():
        #         return []  # No implicit model downloads on the edge device.
        #     from ultralytics import YOLO
        #     self.vehicle_model = YOLO(str(path), task='detect')
        # model = self.vehicle_model
        # classes = [i for i,name in model.names.items() if name in ('truck', 'trailer')]
        # if not classes:
        #     return []
        # results = model.predict(source=frame.image,classes=classes,
        #     conf=self.settings.confidence_threshold,imgsz=self.settings.model_image_size,
        #     device=self.settings.model_device,verbose=False)
        # output=[]
        # for result in results:
        #     if result.boxes is None:
        #         continue
        #     for box in result.boxes:
        #         coords = tuple(float(v) for v in box.xyxy[0].tolist())
        #         if all(math.isfinite(v) for v in coords):
        #             output.append(Detection(frame.camera_id,frame.timestamp,
        #                 str(result.names[int(box.cls.item())]),float(box.conf.item()),coords))
        # return output
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
            for box in result.boxes:
                raw_name = str(result.names[int(box.cls.item())]).lower()
                name = self.settings.class_aliases.get(raw_name, raw_name)
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
                # Require higher confidence for container body to prevent faint noise/pedestrian false triggers
                if name == 'container' and confidence < 0.38:
                    continue
                coords = [float(v) for v in box.xyxy[0].tolist()]
                if not all(math.isfinite(v) for v in coords + [confidence]):
                    continue
                x1, y1, x2, y2 = coords
                w = x2 - x1
                h = y2 - y1

                # Physical geometry sanity check:
                # 1. Containers are horizontal/boxy structures.
                #    A vertical sliver (e.g. standing person, pole) where height > 2.0 * width (aspect < 0.5)
                #    is physically impossible for a shipping container.
                if name == 'container':
                    if w < 50.0 or h < 40.0 or (w * h) < 2500.0:
                        continue
                    if (w / max(h, 1.0)) < 0.50:
                        continue
                elif name == 'container_number':
                    if w < 20.0 or h < 10.0 or (w * h) < 250.0:
                        continue
                elif name == 'iso_code':
                    if w < 15.0 or h < 8.0 or (w * h) < 150.0:
                        continue

                bbox = (max(0., x1), max(0., y1), min(float(width), x2), min(float(height), y2))
                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    detections.append(Detection(frame.camera_id, frame.timestamp, name, confidence, bbox))
        return detections
