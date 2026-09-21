from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
import logging
from app.domain import Observation

log = logging.getLogger('gate')


@dataclass
class Evidence:
    reads: list = field(default_factory=list)
    event_id: str | None = None
    last_seen: datetime | None = None
    last_sequence: int | None = None
    confirmed: bool = False


class EventManager:
    """Called only by the OCR worker; commit succeeds before marking a track done."""
    def __init__(self, settings, repository, snapshots, dispatcher=None):
        self.settings, self.repository, self.snapshots = settings, repository, snapshots
        self.dispatcher = dispatcher
        self.evidence: dict[str, Evidence] = {}

    def prune(self, active_origins):
        self.evidence = {key: value for key, value in self.evidence.items() if key in active_origins}

    def observe(self, job, validation, additional_frames: dict | None = None, proof_job=None, ongoing_event_id=None):
        evidence = self.evidence.setdefault(job.origin_key, Evidence())
        if evidence.event_id and evidence.confirmed:
            return evidence.event_id, False
        evidence.last_seen = job.detection.frame_timestamp
        # Retrying the same failed DB write is not another independent OCR vote.
        if evidence.last_sequence != job.frame.sequence:
            evidence.reads.append(validation)
            evidence.last_sequence = job.frame.sequence
        evidence.reads = evidence.reads[-self.settings.ocr_max_attempts:]
        good = [r for r in evidence.reads if r and r.valid_check_digit]
        votes = Counter(r.normalized_text for r in good)
        best, count = votes.most_common(1)[0] if votes else (None, 0)
        tied = sum(v == count for v in votes.values()) > 1
        confirmed = count >= self.settings.ocr_confirmations and not tied
        if not confirmed and len(evidence.reads) < self.settings.ocr_max_attempts:
            return None
        if not confirmed and not any(getattr(r, 'raw_text', None) for r in evidence.reads if r):
            return None
        if not confirmed:
            # Let the truck complete its entrance before publishing an unconfirmed / NEEDS_REVIEW event!
            is_entrance_done = getattr(job, 'entrance_complete', True) or getattr(job, 'is_stationary', True)
            if not is_entrance_done:
                return None
        selected = max((r for r in good if r.normalized_text == best), key=lambda r: r.confidence) if confirmed else validation
        proof = proof_job or job
        obs = Observation(job.origin_key, job.gate_id, job.track_id, proof.detection, selected,
                          job.direction, proof.frame.image, job.started_at, confirmed,
                          all_detections=list(getattr(proof, 'all_detections', [])),
                          slot_position=getattr(job, 'slot_position', 'FRONT'),
                          event_id=getattr(job, 'event_id', None))

        target_event_id = ongoing_event_id or evidence.event_id or getattr(job, 'event_id', None)
        options = {'ongoing_event_id': target_event_id} if target_event_id else {}
        event_id, created = self.repository.save_observation(
            obs, self.snapshots, self.settings.dedup_seconds, additional_frames, **options)
        evidence.event_id = event_id
        evidence.confirmed = confirmed
        if self.dispatcher and obs.confirmed:
            self.dispatcher.enqueue(obs, event_id)
        log.info('gate_event_created' if created else 'gate_evidence_associated',
                 extra={'camera_id': job.detection.camera_id, 'event_id': event_id})
        return event_id, created
