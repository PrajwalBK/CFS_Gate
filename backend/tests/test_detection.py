from datetime import timedelta
from types import SimpleNamespace
import numpy as np
import pytest
from app.detection.tracker import TemporalTracker
from app.detection.yolo_detector import YoloDetector
from app.domain import Detection
from app.events.gate_logic import update_direction
from app.ocr.container_ocr import crop_identification


def test_detector_contract_and_aliases(settings, make_job):
    class Box:
        cls = np.array(0)
        conf = np.array(.92)
        xyxy = np.array([[-10, -20, 200, 200]])
    class Model:
        def predict(self, **kwargs):
            assert kwargs['rect'] is False
            return [SimpleNamespace(names={0: 'Cointainer'}, boxes=[Box()])]
    settings.class_aliases = {'cointainer': 'container'}
    frame = make_job().frame
    output = YoloDetector(settings, Model()).detect(frame)
    assert output == [Detection(frame.camera_id, frame.timestamp, 'container', .92, (0, 0, 150, 100))]


def test_no_model_fallback(settings, tmp_path):
    cfg = settings.model_copy(update={'model_path': tmp_path / 'missing.pt'})
    with pytest.raises(ValueError):
        YoloDetector(cfg)


def test_tracking_expires_by_time_and_never_matches_other_classes(make_job):
    tracker = TemporalTracker(ttl_seconds=2)
    detection = make_job().detection
    first = tracker.update([detection], detection.frame_timestamp)[0]
    again = tracker.update([detection], detection.frame_timestamp + timedelta(seconds=.5))[0]
    assert first.id == again.id and again.hits == 2
    truck = Detection(detection.camera_id, detection.frame_timestamp, 'truck', .9, detection.bbox)
    assert tracker.update([truck], detection.frame_timestamp + timedelta(seconds=1))[0].id != first.id
    assert tracker.update([detection], detection.frame_timestamp + timedelta(seconds=5))[0].id != first.id


def test_direction_uses_configuration_not_camera_name(camera, make_job):
    camera.id = 'gate-out-front'
    track = TemporalTracker().update([make_job().detection], make_job().frame.timestamp)[0]
    assert update_direction(camera, track, (100, 150, 3)) == 'UNKNOWN'
    camera.direction = 'EXIT'
    assert update_direction(camera, track, (100, 150, 3)) == 'EXIT'


def test_line_crossing_requires_two_sides(camera, make_job):
    camera.line_axis = 'x'
    camera.line_deadband = .05
    detection = make_job().detection
    track = TemporalTracker().update([detection], detection.frame_timestamp)[0]
    for box, expected in [((5, 10, 35, 60), 'UNKNOWN'), ((35, 10, 65, 60), 'UNKNOWN'), ((65, 10, 95, 60), 'ENTRY')]:
        track.detection = Detection(detection.camera_id, detection.frame_timestamp, 'container', .9, box)
        assert update_direction(camera, track, (100, 100, 3)) == expected


def test_clipped_identification_crop():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    assert crop_identification(image, (-30, -10, 250, 150), (.5, .5, 1, 1)).shape == (50, 100, 3)
    with pytest.raises(ValueError):
        crop_identification(image, (300, 100, 350, 120))


def test_non_container_objects_strictly_filtered_by_yolo_detector(settings, make_job):
    class Box:
        def __init__(self, cls_id):
            self.cls = np.array(cls_id)
            self.conf = np.array(.90)
            self.xyxy = np.array([[10, 10, 100, 80]])

    class MultiClassModel:
        def predict(self, **kwargs):
            # Model outputs truck, car, person, trailer, and container classes
            names = {
                0: 'truck',
                1: 'car',
                2: 'person',
                3: 'trailer',
                4: 'container',
                5: 'container_number',
                6: 'iso_code',
            }
            boxes = [Box(i) for i in range(7)]
            return [SimpleNamespace(names=names, boxes=boxes)]

    frame = make_job().frame
    detector = YoloDetector(settings, MultiClassModel())

    # detect_vehicles is disabled and returns empty list
    assert detector.detect_vehicles(frame) == []

    # detect() strictly discards truck, car, person, trailer; outputs only container classes
    detections = detector.detect(frame)
    detected_classes = [d.class_name for d in detections]
    assert 'truck' not in detected_classes
    assert 'car' not in detected_classes
    assert 'person' not in detected_classes
    assert 'trailer' not in detected_classes
    assert set(detected_classes) == {'container', 'container_number', 'iso_code'}


def test_runtime_processes_only_containers(settings, camera, repository, snapshots, make_job):
    from app.runtime import GateRuntime

    runtime = GateRuntime(settings, [camera], repository, snapshots)

    class ContainerAndNonContainerDetector:
        def detect(self, frame):
            # Mix of container and non-container objects
            return [
                Detection(camera.id, frame.timestamp, 'truck', 0.95, (10, 10, 120, 80)),
                Detection(camera.id, frame.timestamp, 'person', 0.90, (5, 5, 30, 70)),
                Detection(camera.id, frame.timestamp, 'container', 0.92, (20, 20, 140, 90)),
                Detection(camera.id, frame.timestamp, 'container_number', 0.94, (30, 30, 90, 50)),
            ]

    runtime.detector = ContainerAndNonContainerDetector()
    frame = make_job().frame

    runtime.process_frame(camera, frame)

    # Preview detections must contain only container and container_number
    _src, _ts, preview = runtime._preview_detections[camera.id]
    assert len(preview) == 2
    preview_classes = {d.class_name for d in preview}
    assert preview_classes == {'container', 'container_number'}
    assert 'truck' not in preview_classes
    assert 'person' not in preview_classes

    # Active passage session received only container detections
    active_sess = runtime.passage_sessions.get_active_session(camera.gate_id)
    assert active_sess is not None
    assert len(active_sess.captured_frames[camera.id]) >= 1
    recorded_dets = active_sess.captured_frames[camera.id][0].detections
    assert all(d.class_name in ('container', 'container_number', 'iso_code') for d in recorded_dets)


def test_yolo_detector_rejects_person_vertical_geometry(settings, make_job):
    class Box:
        def __init__(self, cls_id, xyxy, conf=0.95):
            self.cls = np.array(cls_id)
            self.conf = np.array(conf)
            self.xyxy = np.array([xyxy])

    class GeometryModel:
        def predict(self, **kwargs):
            names = {
                0: 'person',
                1: 'truck',
                2: 'container',
            }
            boxes = [
                # Person detection (should be rejected by class filter)
                Box(0, [10, 10, 40, 90]),
                # Truck detection (should be rejected by class filter)
                Box(1, [10, 10, 120, 90]),
                # Container with vertical aspect ratio (e.g. person or pole falsely classified as container: w=30, h=80, aspect=0.375)
                Box(2, [10, 10, 40, 90]),
                # Legitimate container box (horizontal box: w=100, h=70, aspect=1.42)
                Box(2, [10, 10, 110, 80]),
            ]
            return [SimpleNamespace(names=names, boxes=boxes)]

    frame = make_job().frame
    detector = YoloDetector(settings, GeometryModel())
    dets = detector.detect(frame)

    assert len(dets) == 1
    assert dets[0].class_name == 'container'
    assert dets[0].bbox == (10.0, 10.0, 110.0, 80.0)


def test_runtime_rejects_truck_and_person_even_with_lanes(settings, camera, repository, snapshots, make_job):
    from app.config.settings import LaneRegion
    from app.runtime import GateRuntime

    # Configure camera with lanes
    camera.lanes = [LaneRegion(id='lane-1', polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])]
    runtime = GateRuntime(settings, [camera], repository, snapshots)

    class MixDetector:
        def detect(self, frame):
            return [
                Detection(camera.id, frame.timestamp, 'truck', 0.99, (10, 10, 120, 80)),
                Detection(camera.id, frame.timestamp, 'trailer', 0.95, (10, 10, 120, 80)),
                Detection(camera.id, frame.timestamp, 'person', 0.98, (5, 5, 30, 70)),
                Detection(camera.id, frame.timestamp, 'container', 0.91, (20, 20, 140, 90)),
            ]

    runtime.detector = MixDetector()
    frame = make_job().frame
    runtime.process_frame(camera, frame)

    _src, _ts, preview = runtime._preview_detections[camera.id]
    preview_classes = {d.class_name for d in preview}
    assert preview_classes == {'container'}
    assert 'truck' not in preview_classes
    assert 'trailer' not in preview_classes
    assert 'person' not in preview_classes

