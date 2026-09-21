import pytest
import numpy as np
from datetime import datetime, timezone

from app.domain import Frame, Detection
from app.events.gate_collector import GateCollector
from app.events.scene_analyzer import SceneAnalyzer


def test_gate_collector_and_scene_analyzer_single_40ft():
    collector = GateCollector(gate_id='lane-in', min_cameras=2, collection_window_seconds=1.5)
    analyzer = SceneAnalyzer()
    now = datetime.now(timezone.utc)
    dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Frame from Rear Camera with container and text
    rear_frame = Frame(camera_id='gate-in-3', timestamp=now, sequence=1, image=dummy_img)
    rear_dets = [
        Detection(camera_id='gate-in-3', frame_timestamp=now, class_name='container', confidence=0.90, bbox=(200.0, 100.0, 1000.0, 600.0)),
        Detection(camera_id='gate-in-3', frame_timestamp=now, class_name='cointainer', confidence=0.88, bbox=(700.0, 200.0, 950.0, 260.0)),
    ]
    snap1, ready1 = collector.add_evidence(
        camera_id='gate-in-3',
        role='REAR',
        frame=rear_frame,
        detections=rear_dets,
        direction='ENTRY',
        stationary=True,
        total_configured_cameras=2,
    )
    assert not ready1  # Needs min_cameras=2

    # Frame from Side / Left Camera
    side_frame = Frame(camera_id='gate-in-2', timestamp=now, sequence=1, image=dummy_img)
    side_dets = [
        Detection(camera_id='gate-in-2', frame_timestamp=now, class_name='container', confidence=0.85, bbox=(100.0, 100.0, 1100.0, 500.0)),
    ]
    snap2, ready2 = collector.add_evidence(
        camera_id='gate-in-2',
        role='LEFT',
        frame=side_frame,
        detections=side_dets,
        direction='ENTRY',
        stationary=True,
        total_configured_cameras=2,
    )
    assert ready2
    assert snap2.status == 'READY'

    result = analyzer.analyze(snap2)
    assert result.scene_type == 'SINGLE_40FT'
    assert not result.is_twin
    assert len(result.containers) == 1
    assert result.containers[0].best_camera_id == 'gate-in-3'  # Rear camera preferred for single container


def test_gate_collector_and_scene_analyzer_twin_20ft():
    collector = GateCollector(gate_id='lane-in', min_cameras=1)
    analyzer = SceneAnalyzer()
    now = datetime.now(timezone.utc)
    dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Frame from Side camera seeing 2 separated container boxes
    side_frame = Frame(camera_id='gate-in-2', timestamp=now, sequence=1, image=dummy_img)
    side_dets = [
        Detection(camera_id='gate-in-2', frame_timestamp=now, class_name='container', confidence=0.89, bbox=(50.0, 100.0, 500.0, 500.0)),
        Detection(camera_id='gate-in-2', frame_timestamp=now, class_name='container', confidence=0.91, bbox=(580.0, 100.0, 1150.0, 500.0)),
        Detection(camera_id='gate-in-2', frame_timestamp=now, class_name='cointainer', confidence=0.85, bbox=(100.0, 150.0, 300.0, 200.0)),
        Detection(camera_id='gate-in-2', frame_timestamp=now, class_name='cointainer', confidence=0.86, bbox=(700.0, 150.0, 900.0, 200.0)),
    ]
    snap, ready = collector.add_evidence(
        camera_id='gate-in-2',
        role='LEFT',
        frame=side_frame,
        detections=side_dets,
        direction='ENTRY',
        stationary=True,
        total_configured_cameras=1,
    )
    assert ready
    result = analyzer.analyze(snap)
    assert result.scene_type == 'TWIN_20FT'
    assert result.is_twin is True
    assert len(result.containers) == 2
    slots = [c.slot for c in result.containers]
    assert 'FRONT' in slots and 'REAR' in slots


def test_gate_collector_and_scene_analyzer_two_40ft():
    collector = GateCollector(gate_id='lane-in', min_cameras=1)
    analyzer = SceneAnalyzer()
    now = datetime.now(timezone.utc)
    dummy_img = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Side camera sees 2 massive wide containers (e.g. tandem 40ft setup)
    side_frame = Frame(camera_id='gate-in-2', timestamp=now, sequence=1, image=dummy_img)
    side_dets = [
        Detection(camera_id='gate-in-2', frame_timestamp=now, class_name='container', confidence=0.92, bbox=(50.0, 100.0, 900.0, 450.0)),
        Detection(camera_id='gate-in-2', frame_timestamp=now, class_name='container', confidence=0.90, bbox=(950.0, 100.0, 1850.0, 450.0)),
    ]
    snap, ready = collector.add_evidence(
        camera_id='gate-in-2',
        role='LEFT',
        frame=side_frame,
        detections=side_dets,
        direction='ENTRY',
        stationary=True,
        total_configured_cameras=1,
    )
    assert ready
    result = analyzer.analyze(snap)
    assert result.scene_type == 'TWO_40FT'
    assert result.is_twin is True
    assert len(result.containers) == 2
    for c in result.containers:
        assert c.expected_size == '40FT'


def test_collector_prefers_stationary_and_text_evidence():
    collector = GateCollector(gate_id='lane-in', min_cameras=2)
    now = datetime.now(timezone.utc)
    dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)

    frame1 = Frame(camera_id='gate-in-1', timestamp=now, sequence=1, image=dummy_img)
    moving_dets = [
        Detection(camera_id='gate-in-1', frame_timestamp=now, class_name='container', confidence=0.70, bbox=(100.0, 100.0, 500.0, 500.0))
    ]
    collector.add_evidence(
        camera_id='gate-in-1', role='FRONT_TOP', frame=frame1, detections=moving_dets, stationary=False
    )
    assert not collector.current_snapshot.camera_frames['gate-in-1'].stationary

    # Update with stationary and text detection
    frame2 = Frame(camera_id='gate-in-1', timestamp=now, sequence=2, image=dummy_img)
    stationary_dets = [
        Detection(camera_id='gate-in-1', frame_timestamp=now, class_name='container', confidence=0.88, bbox=(100.0, 100.0, 500.0, 500.0)),
        Detection(camera_id='gate-in-1', frame_timestamp=now, class_name='cointainer', confidence=0.92, bbox=(200.0, 200.0, 400.0, 250.0)),
    ]
    collector.add_evidence(
        camera_id='gate-in-1', role='FRONT_TOP', frame=frame2, detections=stationary_dets, stationary=True
    )
    ev = collector.current_snapshot.camera_frames['gate-in-1']
    assert ev.stationary is True
    assert len(ev.text_boxes) == 1
    assert ev.best_confidence == 0.92

