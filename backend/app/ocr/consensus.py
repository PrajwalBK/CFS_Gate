"""Multi-frame character-level consensus and ISO 6346 resolution.

Combines OCR predictions across multiple frames of the same physical container,
performs character-level positional voting with visual confusion matrix correction,
and validates against ISO 6346 mathematical check-digit algorithms.
"""
from collections import Counter, defaultdict
import logging
import re
from typing import List, Optional, Tuple

from app.domain import Validation
from app.ocr.validator import (
    CHECK_DIGIT_CONFUSIONS,
    STRUCTURE,
    check_digit,
    extract_iso_size_code,
    normalize,
)

log = logging.getLogger('gate.ocr.consensus')

# Contextual position character mappings
DIGIT_TO_LETTER = {
    '0': 'O', '1': 'I', '2': 'Z', '3': 'E', '4': 'A',
    '5': 'S', '6': 'G', '7': 'T', '8': 'B', '9': 'P'
}

LETTER_TO_DIGIT = {
    'O': '0', 'D': '0', 'Q': '0',
    'I': '1', 'L': '1', '|': '1', '/': '1', 'T': '1',
    'Z': '2',
    'E': '3',
    'A': '4',
    'S': '5',
    'G': '6', 'C': '0',
    'B': '8',
    'P': '9'
}

# Common optical confusion pairs for container markings
CHARACTER_CONFUSIONS = {
    'B': ['8', 'D', 'P'],
    '8': ['B', '0', '3', '6'],
    '0': ['O', 'D', '8', 'Q', 'U'],
    'O': ['0', 'D', 'Q', 'U'],
    '1': ['I', 'L', '7', '|'],
    'I': ['1', 'L', 'T', '|'],
    '5': ['S', '6'],
    'S': ['5', '8'],
    '6': ['G', '5', '8'],
    'G': ['6', 'C', 'Q'],
    'C': ['G', '0', 'O'],
    '2': ['Z', '7'],
    'Z': ['2', '7'],
    'U': ['0', 'O', 'V', 'J'],
    'V': ['U', 'Y'],
}


class CharacterConsensus:
    """Aggregates multiple OCR readings of the same container into a high-accuracy result."""

    @classmethod
    def extract_candidates(cls, reads: List[Validation]) -> List[Tuple[str, float, Optional[str]]]:
        """Extract normalized candidate container strings with associated confidence and size_types."""
        candidates = []
        for r in reads:
            if not r:
                continue
            text = (r.normalized_text or '').strip()
            raw = (r.raw_text or '').strip()
            conf = getattr(r, 'confidence', 0.8) or 0.8
            size_type = getattr(r, 'size_type', None)

            # 1. Clean container string if already 11 chars
            clean = normalize(text)
            if len(clean) == 11 and clean[:3].isalpha() and clean[4:].isdigit():
                candidates.append((clean, conf, size_type))
                continue

            # 2. Extract from raw text using regex
            matches = re.findall(r'[A-Z]{3}[A-Z0-9][0-9A-Z]{7}', normalize(raw))
            for m in matches:
                candidates.append((m, conf * 0.9, size_type))

            # 3. If normalized text has 10 characters (missing check digit)
            if len(clean) == 10 and clean[:3].isalpha() and clean[4:].isdigit():
                try:
                    pref = clean[:3] + ('U' if clean[3] not in ('U', 'J', 'Z') else clean[3]) + clean[4:10]
                    cd = check_digit(pref)
                    candidates.append((pref + str(cd), conf * 0.85, size_type))
                except Exception:
                    pass

        return candidates

    @classmethod
    def resolve(cls, reads: List[Validation], min_confidence: float = 0.40) -> Optional[Validation]:
        """Produce the consensus container number from multiple frame reads."""
        if not reads:
            return None

        # Collect size codes across all frames
        aggregated_sizes = [
            getattr(r, 'size_type', None) for r in reads if getattr(r, 'size_type', None)
        ]
        best_size = Counter(aggregated_sizes).most_common(1)[0][0] if aggregated_sizes else None

        # 1. Check if any single read is already 100% check-digit valid with high confidence
        valid_reads = [
            r for r in reads
            if r and r.valid_check_digit and (r.confidence or 0.0) >= min_confidence
        ]
        if valid_reads:
            # Group by container number
            counts = Counter(r.normalized_text for r in valid_reads)
            best_num, top_count = counts.most_common(1)[0]
            # Pick the highest-confidence reading matching the consensus number
            best_obj = max(
                (r for r in valid_reads if r.normalized_text == best_num),
                key=lambda r: r.confidence or 0.0
            )
            if not best_obj.size_type and best_size:
                best_obj = Validation(
                    raw_text=best_obj.raw_text,
                    normalized_text=best_obj.normalized_text,
                    confidence=best_obj.confidence,
                    valid_format=best_obj.valid_format,
                    valid_check_digit=best_obj.valid_check_digit,
                    validation_status=best_obj.validation_status,
                    size_type=best_size
                )
            return best_obj

        # 2. Extract candidate strings for character-level voting
        candidates = cls.extract_candidates(reads)
        if not candidates:
            # Fall back to best available partial read
            valid_partial = [r for r in reads if r and r.normalized_text and len(r.normalized_text) >= 4]
            if valid_partial:
                return max(valid_partial, key=lambda r: (r.confidence or 0.0))
            return reads[-1] if reads else None

        # 3. Positional voting across 11 character slots
        position_votes: List[Counter] = [Counter() for _ in range(11)]
        for cand, conf, _ in candidates:
            if len(cand) != 11:
                continue
            for idx, char in enumerate(cand):
                # Contextual normalization per slot:
                if idx in (0, 1, 2):
                    # Owner code: Must be A-Z
                    char = DIGIT_TO_LETTER.get(char, char)
                    if not char.isalpha():
                        continue
                elif idx == 3:
                    # Equipment identifier: Must be U, J, or Z
                    if char not in ('U', 'J', 'Z'):
                        char = 'U'  # Default freight container
                else:
                    # Serial number (4-9) and check digit (10): Must be 0-9
                    char = LETTER_TO_DIGIT.get(char, char)
                    if not char.isdigit():
                        continue
                position_votes[idx][char] += conf

        # 4. Form base consensus string
        consensus_chars = []
        char_confidences = []
        for idx in range(11):
            if position_votes[idx]:
                char, score = position_votes[idx].most_common(1)[0]
                total = sum(position_votes[idx].values())
                consensus_chars.append(char)
                char_confidences.append(score / max(1e-6, total))
            else:
                consensus_chars.append('?' if idx < 4 else '0')
                char_confidences.append(0.0)

        consensus_str = ''.join(consensus_chars)
        avg_conf = sum(char_confidences) / 11.0 if char_confidences else 0.5

        # 5. Check if consensus string is ISO 6346 compliant
        if bool(STRUCTURE.fullmatch(consensus_str)):
            prefix = consensus_str[:10]
            calc_cd = check_digit(prefix)
            read_cd = int(consensus_str[10])

            if calc_cd == read_cd:
                # Direct check digit match from positional voting consensus!
                return Validation(
                    raw_text=consensus_str,
                    normalized_text=consensus_str,
                    confidence=round(min(0.99, max(avg_conf, 0.90)), 3),
                    valid_format=True,
                    valid_check_digit=True,
                    validation_status='VALID_CONSENSUS',
                    size_type=best_size
                )
            # If check digit does not match, DO NOT GUESS OR REPLACE CHARACTERS
            return Validation(
                raw_text=consensus_str,
                normalized_text=consensus_str,
                confidence=round(avg_conf, 3),
                valid_format=True,
                valid_check_digit=False,
                validation_status='INVALID_CHECK_DIGIT',
                size_type=best_size
            )

        # 6. Fallback if check digit cannot be reconciled
        highest_conf_read = max(reads, key=lambda r: (getattr(r, 'confidence', 0.0) or 0.0))
        return Validation(
            raw_text=consensus_str or highest_conf_read.raw_text,
            normalized_text=consensus_str if len(consensus_str) == 11 else highest_conf_read.normalized_text,
            confidence=round(avg_conf, 3),
            valid_format=bool(STRUCTURE.fullmatch(consensus_str)),
            valid_check_digit=False,
            validation_status='CONSENSUS_UNCONFIRMED',
            size_type=best_size or highest_conf_read.size_type
        )
