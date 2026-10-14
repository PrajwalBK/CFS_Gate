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
            if len(identities) != 1:
                return replace(max(valid, key=lambda r: r.confidence), normalized_text='',
                               valid_format=False, valid_check_digit=False, validation_status='AMBIGUOUS')
            best = max(valid, key=lambda r: r.confidence)
            return replace(best, size_type=best.size_type or size)
        # Preserve a diagnostic read without converting checksum arithmetic,
        # visual substitutions or positional votes into a fabricated proof.
        best = max(reads, key=lambda r: r.confidence)
        return replace(best, valid_check_digit=False, size_type=best.size_type or size)
