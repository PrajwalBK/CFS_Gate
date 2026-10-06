from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
import logging
import time
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
    def __init__(self, settings, repository, snapshots, dispatcher=None, runtime=None):
        self.settings, self.repository, self.snapshots = settings, repository, snapshots
        self.dispatcher = dispatcher
        self.runtime = runtime
        self.evidence: dict[str, Evidence] = {}
        self._dispatched_events: set[str] = set()

    def prune(self, active_origins):
        self.evidence = {key: value for key, value in self.evidence.items() if key in active_origins}

    def observe(self, job, validation, additional_frames: dict | None = None, proof_job=None, ongoing_event_id=None, profiler_record=None):
        evidence = self.evidence.setdefault(job.origin_key, Evidence())
        # A persisted event is already committed. This also covers NEEDS_REVIEW
        # events: replaying the same spool job must not clear its event link or
        # create a second record after a restart.
        if evidence.event_id and evidence.confirmed:
            return evidence.event_id, False
        evidence.last_seen = job.detection.frame_timestamp
        # Retrying the same failed DB write is not another independent OCR vote.
        if evidence.last_sequence != job.frame.sequence:
            evidence.reads.append(validation)
            evidence.last_sequence = job.frame.sequence
        evidence.reads = evidence.reads[-self.settings.ocr_max_attempts:]

        from app.ocr.validator import is_valid_iso_6346

        good = [
            r for r in evidence.reads
            if r and r.valid_check_digit and is_valid_iso_6346(r.normalized_text)
            and getattr(r, 'confidence', 1.0) >= self.settings.ocr_min_confidence
        ]
        votes = Counter(r.normalized_text for r in good)
        best, count = votes.most_common(1)[0] if votes else (None, 0)
        tied = sum(v == count for v in votes.values()) > 1
        confirmed = count >= self.settings.ocr_confirmations and not tied and is_valid_iso_6346(best)

        # Early confirmation: If at least 2 distinct observations agree on a valid check-digit code, confirm immediately
        if not confirmed and count >= 2 and not tied and is_valid_iso_6346(best):
            confirmed = True

        t_cons0 = time.perf_counter()
        consensus_val = None
        if not confirmed and len(evidence.reads) >= 2:
            from app.ocr.consensus import CharacterConsensus
            consensus_val = CharacterConsensus.resolve(evidence.reads, self.settings.ocr_min_confidence)
            if (consensus_val and consensus_val.valid_check_digit and
                is_valid_iso_6346(consensus_val.normalized_text) and
                (consensus_val.validation_status == 'VALID_CONSENSUS' or count >= 2)):
                confirmed = True
                best = consensus_val.normalized_text
        if profiler_record:
            profiler_record.consensus_ms = (time.perf_counter() - t_cons0) * 1000.0

        # NEVER confirm on count == 1 alone without multi-frame or consensus verification.
        # If max attempts reached without confirmation, event stays unconfirmed (NEEDS_REVIEW).
        if not confirmed:
            # Let the truck complete its entrance before publishing an unconfirmed / NEEDS_REVIEW event!
            is_entrance_done = getattr(job, 'entrance_complete', True) or getattr(job, 'is_stationary', True)
            if len(evidence.reads) < self.settings.ocr_max_attempts:
                # If we have a candidate read or passage is in progress, wait for more frames
                if count >= 1 or not is_entrance_done:
                    return None
            # If all attempts yielded no readable text, flag for review with an alert rather than silently dropping
            if not any(getattr(r, 'raw_text', None) for r in evidence.reads if r):
                from app.domain import Validation
                validation = Validation(
                    raw_text='UNREADABLE', normalized_text='', confidence=0.0,
                    valid_format=False, valid_check_digit=False, validation_status='UNREADABLE'
                )
                log.warning('passage_unreadable_flagged_for_review', extra={
                    'origin': job.origin_key, 'gate_id': job.gate_id, 'track_id': job.track_id
                })

        if confirmed and good and any(r.normalized_text == best for r in good):
            selected = max((r for r in good if r.normalized_text == best), key=lambda r: r.confidence)
        elif confirmed and consensus_val and consensus_val.valid_check_digit and is_valid_iso_6346(consensus_val.normalized_text):
            selected = consensus_val
        else:
            # Unconfirmed: Never let an unconfirmed/noisy string or UUID become a container number
            from app.domain import Validation
            selected = Validation(
                raw_text=getattr(validation, 'raw_text', '') or '',
                normalized_text='',
                confidence=0.0,
                valid_format=False,
                valid_check_digit=False,
                validation_status='UNCONFIRMED'
            )
        proof = proof_job or job
        obs = Observation(job.origin_key, job.gate_id, job.track_id, proof.detection, selected,
                          job.direction, proof.frame.image, job.started_at, confirmed,
                          all_detections=list(getattr(proof, 'all_detections', [])),
                          slot_position='SINGLE',
                          event_id=getattr(job, 'event_id', None),
                          is_twin=False)

        target_event_id = ongoing_event_id or evidence.event_id or getattr(job, 'event_id', None)
        options = {'ongoing_event_id': target_event_id} if target_event_id else {}
        t_save0 = time.perf_counter()
        try:
            event_id, created = self.repository.save_observation(
                obs, self.snapshots, self.settings.dedup_seconds, additional_frames,
                same_container_window_seconds=self.settings.same_container_event_window_seconds,
                **options)
        except TypeError as exc:
            # Keep compatibility with small repository test doubles/plugins
            # that still implement the pre-window method signature.
            if 'same_container_window_seconds' not in str(exc):
                raise
            event_id, created = self.repository.save_observation(
                obs, self.snapshots, self.settings.dedup_seconds, additional_frames, **options)
        if profiler_record:
            profiler_record.event_creation_ms = (time.perf_counter() - t_save0) * 1000.0
        evidence.event_id = event_id
        evidence.confirmed = confirmed

        # Only dispatch to cloud and buzz for new events or newly confirmed events that haven't been dispatched yet
        # AND strictly ensure the container number is a valid ISO 6346 code!
        should_dispatch = (
            obs.confirmed and
            is_valid_iso_6346(obs.ocr.normalized_text if obs.ocr else None) and
            (event_id not in self._dispatched_events)
        )
        if self.dispatcher and should_dispatch:
            t_api0 = time.perf_counter()
            self._dispatched_events.add(event_id)
            self.dispatcher.enqueue(
                obs,
                event_id,
                delay_seconds=getattr(self.settings, 'cloud_dispatch_grace_seconds', 0.0),
            )
            if profiler_record:
                profiler_record.api_queue_ms = (time.perf_counter() - t_api0) * 1000.0
        if self.runtime and obs.direction in ('EXIT', 'OUT', 'GATE_OUT') and should_dispatch:
            try:
                cntr = obs.ocr.normalized_text if (obs and obs.ocr and obs.confirmed) else 'Unconfirmed'
                self.runtime.emit_alert(
                    alert_type='GATE_OUT',
                    title='Gate OUT Movement',
                    message=f'Container {cntr} exiting on lane {obs.gate_id}',
                    level='WARNING',
                    container_number=cntr if obs.confirmed else None,
                    event_id=event_id,
                    buzzer=True
                )
            except Exception:
                pass
        log.info('gate_event_created' if created else 'gate_evidence_associated',
                 extra={'camera_id': job.detection.camera_id, 'event_id': event_id})
        return event_id, created
