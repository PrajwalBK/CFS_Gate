"""View-space occupancy independent of short-lived tracker IDs.

Only measured inference frames advance clearance. Missing camera frames never
count as proof that a lane is clear. One calibrated zone represents one queue.
"""
from dataclasses import dataclass, field
from math import hypot
from uuid import uuid4
from app.detection.tracker import iou


def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def containment(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    min_area = min(box_area(b1), box_area(b2))
    return inter / min_area if min_area > 0.0 else 0.0


def normalized(box, shape):
    h, w = shape[:2]
    return (box[0] / w, box[1] / h, box[2] / w, box[3] / h)


def center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


@dataclass
class ContainerSlot:
    slot_id: str = 'SINGLE'
    origin: str = ''
    bbox: tuple = (0, 0, 1, 1)
    anchor: tuple = (0.5, 0.5)
    first_seen: object = None
    last_measured: object = None
    stationary_since: object = None
    stationary: bool = False
    event_id: str | None = None
    container_number: str | None = None
    closed: bool = False
    clear_since: object = None


@dataclass
class Occupancy:
    source_id: str
    origin: str
    bbox: tuple
    anchor: tuple
    first_seen: object
    last_measured: object
    stationary_since: object
    clear_since: object = None
    stationary: bool = False
    event_id: str | None = None
    container_number: str | None = None
    closed: bool = False
    is_twin: bool = False
    slots: dict = field(default_factory=dict)
    active_slot: str = 'SINGLE'
    provisional_event_id: str | None = None
    provisional_event_ids: dict = field(default_factory=dict)
    entrance_complete: bool = False


class ViewOccupancy:
    def __init__(self, settings):
        self.settings = settings
        self.current = None

    def _past_exit(self, box, camera):
        if camera.gate_exit_axis is None:
            return False
        axis = 0 if camera.gate_exit_axis == 'x' else 1
        edge = box[axis] if camera.gate_exit_positive else box[axis + 2]
        return edge > camera.gate_exit_position if camera.gate_exit_positive else edge < camera.gate_exit_position

    def _update_slot(self, slot, b, now):
        point = center(b)
        drift = hypot(point[0] - slot.anchor[0], point[1] - slot.anchor[1])
        if drift >= self.settings.stationary_drift_threshold:
            slot.anchor, slot.stationary_since, slot.stationary = point, now, False
        else:
            slot.stationary = (now - slot.stationary_since).total_seconds() >= self.settings.stationary_seconds
        slot.bbox = b
    def update(self, camera, frame, detections):
        now = frame.timestamp
        dwell_classes = getattr(self.settings, 'dwell_container_classes', None) or ['container', 'container_number', 'iso_code']
        all_boxes = [(d, normalized(d.bbox, frame.image.shape)) for d in detections
                     if d.class_name in dwell_classes]
        zone = camera.occupancy_roi

        # 1. Filter raw candidates: must intersect occupancy ROI, center within ROI, not past exit, and realistic container/text size
        raw_candidates = []
        for d, b in all_boxes:
            cx, cy = center(b)
            if iou(b, zone) > 0 and (zone[0] <= cx <= zone[2] and zone[1] <= cy <= zone[3]) and not self._past_exit(b, camera):
                min_area = 0.005 if d.class_name == 'container' else 0.0001
                if box_area(b) >= min_area:
                    raw_candidates.append((d, b))

        # 2. Suppress nested sub-boxes (e.g. text strips, decals, doors detected inside container body)
        candidates = []
        for d, b in sorted(raw_candidates, key=lambda item: box_area(item[1]), reverse=True):
            if not any(containment(b, ex[1]) > 0.5 for ex in candidates):
                candidates.append((d, b))

        # Store all validated container candidates in ROI for multi-container tracking
        self.selected_detections = [c[0] for c in candidates]

        state = self.current
        if state and state.source_id != frame.source_id:
            self.current = state = None  # Explicit source/calibration generation change.

        if state:
            gap = (now - state.last_measured).total_seconds()
            if gap <= 0:
                return None  # Duplicate/backwards timestamps cannot advance clearance.
            if gap > self.settings.gate_observation_gap_seconds:
                state.clear_since = None
                state.stationary_since = now
                state.stationary = False
            state.last_measured = now

            # Check if container departed past exit
            if self._past_exit(state.bbox, camera):
                self.current = None
                return None

            matched_dep = max(all_boxes, key=lambda item: iou(state.bbox, item[1]), default=None)
            if matched_dep and iou(state.bbox, matched_dep[1]) > 0.15:
                if not self._past_exit(state.bbox, camera) and self._past_exit(matched_dep[1], camera):
                    self.current = None
                    return None

            if not candidates:
                state.clear_since = state.clear_since or now
                if (now - state.clear_since).total_seconds() >= self.settings.gate_clear_seconds:
                    self.current = None
                return None

            state.clear_since = None

            # Spatial tracking continuity: match candidate to current container by IoU
            matched = max(candidates, key=lambda c: iou(state.bbox, c[1]), default=None)
            if matched and iou(state.bbox, matched[1]) >= 0.15:
                det, b = matched
                front_slot = state.slots.get('FRONT')
                if front_slot:
                    self._update_slot(front_slot, b, now)
                state.bbox = b
                state.anchor = center(b)
                state.stationary = front_slot.stationary if front_slot else False
                state.entrance_complete = state.stationary or (
                    (now - state.first_seen).total_seconds() >= self.settings.gate_observation_gap_seconds
                )
                if state.closed and state.clear_since is not None:
                    return None
                return det
            else:
                # Different container or no IoU match: keep tracking existing without stealing
                if (now - state.last_measured).total_seconds() >= self.settings.gate_clear_seconds:
                    self.current = None
                return None

        # State is None: fresh container arrival
        if not candidates:
            return None

        best_det, best_b = max(candidates, key=lambda c: c[0].confidence * box_area(c[1]))
        base_origin = f'dwell:{camera.id}:{uuid4().hex}'
        prov_id = str(uuid4())
        slots = {
            'FRONT': ContainerSlot('SINGLE', f'{base_origin}:single', best_b, center(best_b), now, now, now, event_id=None)
        }
        self.current = Occupancy(frame.source_id, base_origin, best_b, center(best_b),
                                 now, now, now, event_id=None, is_twin=False, slots=slots, active_slot='SINGLE',
                                 provisional_event_id=prov_id, provisional_event_ids={'CONTAINER_1': prov_id})
        return best_det

    def confirm(self, origin, event_id, container_number=None):
        if not self.current:
            return
        state = self.current
        matched = False
        for s_id, slot in state.slots.items():
            if slot.origin == origin or origin.startswith(state.origin):
                slot.event_id = event_id
                slot.container_number = container_number
                slot.closed = True
                matched = True
                break
        if not matched:
            front = state.slots.get('FRONT')
            if front:
                front.event_id = event_id
                front.container_number = container_number
                front.closed = True
        state.event_id = event_id
        if not state.container_number:
            state.container_number = container_number
        state.closed = True

    def status(self):
        state = self.current
        if not state:
            return {'state': 'CLEAR', 'state_machine': {'state': 'CLEAR', 'next': ['APPROACHING']}, 'event_id': None}
        if state.clear_since is not None:
            machine_state = 'CLEARING'
        elif state.stationary and state.event_id:
            machine_state = 'WAITING'
        elif state.event_id:
            machine_state = 'OCCUPIED_SINGLE'
        elif getattr(state, 'entrance_complete', False):
            machine_state = 'EXITING'
        else:
            machine_state = 'APPROACHING'
        next_states = {
            'CLEAR': ['APPROACHING'], 'APPROACHING': ['OCCUPIED_SINGLE', 'CLEAR'],
            'OCCUPIED_SINGLE': ['WAITING', 'EXITING'],
            'WAITING': ['EXITING', 'CLEARING'], 'EXITING': ['CLEARING', 'WAITING'],
            'CLEARING': ['CLEAR', 'OCCUPIED_SINGLE'],
        }[machine_state]
        return {'state': 'WAITING' if state.stationary and state.event_id else
                'OCCUPIED' if state.event_id else 'TRACKING',
                'state_machine': {'state': machine_state, 'next': next_states},
                'stationary': state.stationary, 'event_id': state.event_id,
                'bbox_normalized': state.bbox, 'clear_pending': state.clear_since is not None,
                'is_twin': False,
                'entrance_complete': getattr(state, 'entrance_complete', False),
                'slots': {key: {'closed': slot.closed, 'container_number': slot.container_number}
                          for key, slot in getattr(state, 'slots', {}).items()}}
