from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
import logging
import time
from app.domain import Observation
from app.ocr.validator import is_valid_iso_6346

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

        good = [r for r in evidence.reads if r and r.valid_check_digit and is_valid_iso_6346(r.normalized_text)
                and getattr(r, 'confidence', 1.0) >= self.settings.ocr_min_confidence]
        votes = Counter(r.normalized_text for r in good)
        best, count = votes.most_common(1)[0] if votes else (None, 0)
        tied = sum(v == count for v in votes.values()) > 1
        confirmed = count >= self.settings.ocr_confirmations and not tied

        # Early confirmation: If at least 2 distinct observations agree on a valid check-digit code, confirm immediately
        if not confirmed and count >= 2 and not tied:
            confirmed = True

        t_cons0 = time.perf_counter()
        consensus_val = None
        if not confirmed and len(evidence.reads) >= self.settings.ocr_confirmations:
            from app.ocr.consensus import CharacterConsensus
            consensus_val = CharacterConsensus.resolve(evidence.reads, self.settings.ocr_min_confidence)
            if consensus_val and consensus_val.valid_check_digit:
                confirmed = True
                best = consensus_val.normalized_text
        if profiler_record:
            profiler_record.consensus_ms = (time.perf_counter() - t_cons0) * 1000.0

        # A dwell track can end before the configured vote count is reached
        # (camera handoff, truck leaving the ROI, or a short queue passage).
        # If at least two samples were processed and one high-confidence,
        # check-digit-valid read exists, preserve it as the event identity
        # rather than silently dropping the completed track.
        dwell_end_fallback = (
            job.origin_key.startswith('dwell:') and
            getattr(job, 'entrance_complete', False) and
            len(evidence.reads) >= 2 and (count >= 1 or (consensus_val and consensus_val.valid_check_digit)) and not tied
        )
        if dwell_end_fallback:
            confirmed = True
        if not confirmed and len(evidence.reads) >= self.settings.ocr_max_attempts and (count >= 1 or (consensus_val and consensus_val.valid_check_digit)) and not tied:
            confirmed = True
        if not confirmed:
            # While attempts remain, do not prematurely publish an unconfirmed / NEEDS_REVIEW row.
            # Give the system time to read more frames of the passing container.
            if len(evidence.reads) < self.settings.ocr_max_attempts:
                return None

            # Even when attempts are exhausted for this camera/track:
            # If an event is already ongoing or confirmed on this lane, bind evidence to it.
            if not ongoing_event_id and self.runtime:
                slot_pos = getattr(job, 'slot_position', 'CONTAINER_1') or 'CONTAINER_1'
                active_sess = self.runtime.passage_sessions.get_active_session(job.gate_id, container_slot=slot_pos)
                if active_sess:
                    ongoing_event_id = active_sess.event_id
                if not ongoing_event_id:
                    if slot_pos == 'CONTAINER_1':
                        active_lane = self.runtime._active_lane_events.get(f"{job.gate_id}:{slot_pos}") or self.runtime._active_lane_events.get(job.gate_id)
                    else:
                        active_lane = self.runtime._active_lane_events.get(f"{job.gate_id}:{slot_pos}")
                    if active_lane and active_lane.get('event_id') and not active_lane.get('finalized_at'):
                        lane_num = active_lane.get('container_number')
                        val_num = validation.normalized_text if (validation and validation.valid_check_digit) else None
                        from app.database.repositories.gate import is_distinct_container
                        if lane_num and val_num and is_distinct_container(lane_num, val_num):
                            ongoing_event_id = None
                        else:
                            ongoing_event_id = active_lane.get('event_id')

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
        elif confirmed and consensus_val and consensus_val.valid_check_digit:
            selected = consensus_val
        else:
            selected = consensus_val or validation
        proof = proof_job or job
        slot_pos = getattr(job, 'slot_position', 'CONTAINER_1') or 'CONTAINER_1'
        obs = Observation(job.origin_key, job.gate_id, job.track_id, proof.detection, selected,
                          job.direction, proof.frame.image, job.started_at, confirmed,
                          all_detections=list(getattr(proof, 'all_detections', [])),
                          slot_position=slot_pos,
                          event_id=getattr(job, 'event_id', None),
                          is_twin=False, source_id=proof.frame.source_id, frame_sequence=proof.frame.sequence,
                          proof_container_number=validation.normalized_text if validation and validation.valid_check_digit else '')

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

        # Two-stage cloud integration:
        # Step 1: Immediately POST gate event with initial confirmed OCR and any immediately available image.
        # Step 2: Background multi-camera passage session continues and later triggers PUT with final evidence.
        is_ocr_valid = bool(obs.ocr and obs.ocr.valid_check_digit and obs.ocr.valid_format)
        should_dispatch_post = obs.confirmed and is_ocr_valid and (event_id not in self._dispatched_events)
        if self.dispatcher and should_dispatch_post:
            t_api0 = time.perf_counter()
            self._dispatched_events.add(event_id)
            self.dispatcher.enqueue(
                obs,
                event_id,
                stage='POST',
                delay_seconds=getattr(self.settings, 'cloud_dispatch_grace_seconds', 0.0),
            )
            if profiler_record:
                profiler_record.api_queue_ms = (time.perf_counter() - t_api0) * 1000.0
        if self.runtime and obs.direction in ('EXIT', 'OUT', 'GATE_OUT') and should_dispatch_post:
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
