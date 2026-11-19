"""Consensus over observed checksum-valid reads; never invent identity characters."""
from collections import Counter
from dataclasses import replace
import re
from app.ocr.validator import is_valid_iso_6346, KNOWN_BIC_PREFIXES, STRUCTURE


class CharacterConsensus:
    @classmethod
    def extract_candidates(cls, reads):
        return [(r.normalized_text, r.confidence, r.size_type) for r in reads
                if r and r.valid_check_digit and is_valid_iso_6346(r.normalized_text)]

    @classmethod
    def positional_vote(cls, reads, min_confidence=0.40):
        """Align 11-character container reads position-by-position to resolve isolated frame errors."""
        eligible = []
        for r in reads:
            if not r or r.confidence < min_confidence:
                continue
            txt = (r.normalized_text or '').strip().upper()
            if len(txt) == 11 and re.match(r'^[A-Z0-9]{11}$', txt):
                eligible.append((txt, r.confidence))

        if len(eligible) < 2:
            return None

        pos_votes = [{} for _ in range(11)]
        for txt, conf in eligible:
            for i, ch in enumerate(txt):
                pos_votes[i][ch] = pos_votes[i].get(ch, 0.0) + conf

        winner_chars = []
        for i in range(11):
            if not pos_votes[i]:
                return None
            winner_ch = max(pos_votes[i].items(), key=lambda kv: kv[1])[0]
            winner_chars.append(winner_ch)

        candidate = ''.join(winner_chars)
        if STRUCTURE.fullmatch(candidate) and is_valid_iso_6346(candidate):
            avg_conf = sum(max(pos_votes[i].values()) for i in range(11)) / (11.0 * max(1, len(eligible)))
            return candidate, avg_conf
        return None

    @classmethod
    def resolve(cls, reads, min_confidence=0.40):
        reads = [r for r in reads if r]
        if not reads:
            return None
        valid = [r for r in reads if r.valid_check_digit and
                 is_valid_iso_6346(r.normalized_text) and r.confidence >= min_confidence]
        sizes = [r.size_type for r in reads if r.size_type]
        size = Counter(sizes).most_common(1)[0][0] if sizes else None

        # If multiple conflicting valid identities observed with equal/competing votes,
        # evaluate candidate scoring / ambiguity first before positional synthesis.
        identities = {r.normalized_text for r in valid}
        if len(identities) > 1:
            by_id = {}
            for r in valid:
                by_id.setdefault(r.normalized_text, []).append(r)

            scored = []
            for ident, r_list in by_id.items():
                count = len(r_list)
                max_conf = max(r.confidence for r in r_list)
                best_r = max(r_list, key=lambda r: r.confidence)
                is_known_bic = 1 if (ident and len(ident) >= 4 and ident[:4] in KNOWN_BIC_PREFIXES) else 0
                scored.append((is_known_bic, count, max_conf, best_r))

            scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
            winner_known, winner_count, winner_conf, winner_read = scored[0]
            runner_up_known, runner_up_count, runner_up_conf, runner_up_read = scored[1]

            # Clear winner if verified in BIC registry over unverified candidate,
            # or strictly more votes, or significantly higher confidence (>=0.15)
            if (winner_known > runner_up_known) or winner_count > runner_up_count or (winner_conf - runner_up_conf) >= 0.15:
                return replace(winner_read, size_type=winner_read.size_type or size)

            # True tie between conflicting valid IDs with equal votes and similar confidence
            return replace(winner_read, normalized_text='',
                           valid_format=False, valid_check_digit=False, validation_status='AMBIGUOUS')

        # Positional alignment across multiple frames to recover isolated character dropouts
        pos_result = cls.positional_vote(reads, min_confidence=min_confidence)
        if pos_result:
            pos_text, pos_conf = pos_result
            matching_valid = [r for r in valid if r.normalized_text == pos_text]
            if matching_valid:
                best_m = max(matching_valid, key=lambda r: r.confidence)
                if len(reads) >= 2 and best_m.confidence < 0.95:
                    best_m = replace(best_m, confidence=min(0.99, max(best_m.confidence, pos_conf + 0.10)))
                return replace(best_m, size_type=best_m.size_type or size)
            elif not valid and is_valid_iso_6346(pos_text):
                representative = max(reads, key=lambda r: r.confidence)
                return replace(representative, normalized_text=pos_text, confidence=pos_conf,
                               valid_format=True, valid_check_digit=True, validation_status='VALID',
                               size_type=representative.size_type or size)

        if valid:
            best = max(valid, key=lambda r: r.confidence)
            return replace(best, size_type=best.size_type or size)

        # Preserve a diagnostic read without converting checksum arithmetic,
        # visual substitutions or positional votes into a fabricated proof.
        best = max(reads, key=lambda r: r.confidence)
        return replace(best, valid_check_digit=False, size_type=best.size_type or size)
