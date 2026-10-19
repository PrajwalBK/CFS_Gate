"""Consensus over observed checksum-valid reads; never invent identity characters."""
from collections import Counter
from dataclasses import replace
from app.ocr.validator import is_valid_iso_6346


class CharacterConsensus:
    @classmethod
    def extract_candidates(cls, reads):
        return [(r.normalized_text, r.confidence, r.size_type) for r in reads
                if r and r.valid_check_digit and is_valid_iso_6346(r.normalized_text)]

    @classmethod
    def resolve(cls, reads, min_confidence=0.40):
        reads = [r for r in reads if r]
        if not reads:
            return None
        valid = [r for r in reads if r.valid_check_digit and
                 is_valid_iso_6346(r.normalized_text) and r.confidence >= min_confidence]
        sizes = [r.size_type for r in reads if r.size_type]
        size = Counter(sizes).most_common(1)[0][0] if sizes else None
        if valid:
            identities = {r.normalized_text for r in valid}
            if len(identities) == 1:
                best = max(valid, key=lambda r: r.confidence)
                return replace(best, size_type=best.size_type or size)

            # Multiple distinct valid identities observed:
            # Group by identity, count votes and track maximum confidence per identity
            by_id = {}
            for r in valid:
                by_id.setdefault(r.normalized_text, []).append(r)

            scored = []
            for ident, r_list in by_id.items():
                count = len(r_list)
                max_conf = max(r.confidence for r in r_list)
                best_r = max(r_list, key=lambda r: r.confidence)
                scored.append((count, max_conf, best_r))

            scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
            winner_count, winner_conf, winner_read = scored[0]
            runner_up_count, runner_up_conf, runner_up_read = scored[1]

            # Clear winner if strictly more votes, or significantly higher confidence (>=0.15)
            if winner_count > runner_up_count or (winner_conf - runner_up_conf) >= 0.15:
                return replace(winner_read, size_type=winner_read.size_type or size)

            # True tie between conflicting valid IDs with equal votes and similar confidence
            return replace(winner_read, normalized_text='',
                           valid_format=False, valid_check_digit=False, validation_status='AMBIGUOUS')

        # Preserve a diagnostic read without converting checksum arithmetic,
        # visual substitutions or positional votes into a fabricated proof.
        best = max(reads, key=lambda r: r.confidence)
        return replace(best, valid_check_digit=False, size_type=best.size_type or size)
