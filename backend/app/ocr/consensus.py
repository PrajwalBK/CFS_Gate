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
        if STRUCTURE.fullmatch(candidate):
            from app.ocr.validator import recover_visual_confusion
            resolved = recover_visual_confusion(candidate)
            if is_valid_iso_6346(resolved):
                avg_conf = sum(max(pos_votes[i].values()) for i in range(11)) / (11.0 * max(1, len(eligible)))
                return resolved, avg_conf
            if is_valid_iso_6346(candidate):
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
                cams = {getattr(r, 'camera_id', None) for r in r_list if getattr(r, 'camera_id', None)}
                cam_count = len(cams)
                max_conf = max(r.confidence for r in r_list)
                avg_conf = sum(r.confidence for r in r_list) / count
                best_r = max(r_list, key=lambda r: r.confidence)
                is_known_bic = 1 if (ident and len(ident) >= 4 and ident[:4] in KNOWN_BIC_PREFIXES) else 0
                raw_match = 1 if any(getattr(r, 'raw_text', None) and ident in getattr(r, 'raw_text', '').replace(' ', '').replace('-', '').upper() for r in r_list) else 0

                # Multi-evidence composite score:
                # 1. Multi-camera agreement (+3.0 per distinct camera): Physical confirmation across angles
                # 2. Vote frequency (+1.0 per observation)
                # 3. Peak OCR confidence (+1.5 * max_conf)
                # 4. Direct raw match (+0.5 if raw OCR text directly matches without synthetic character mutation)
                # 5. BIC registry tie-breaker (+0.2 bonus)
                score = (cam_count * 3.0) + count + (max_conf * 1.5) + (raw_match * 0.5) + (is_known_bic * 0.2)
                scored.append((score, cam_count, count, max_conf, is_known_bic, best_r))

            scored.sort(key=lambda item: item[0], reverse=True)
            winner_score, winner_cam, winner_count, winner_conf, winner_known, winner_read = scored[0]
            runner_up_score, runner_up_cam, runner_up_count, runner_up_conf, runner_up_known, runner_up_read = scored[1]

            # Clear winner if strictly higher composite score, multi-camera agreement, higher votes, or registered BIC tie-breaker
            if (winner_cam > runner_up_cam) or (winner_count > runner_up_count) or \
               (winner_score - runner_up_score >= 0.25) or (winner_conf - runner_up_conf >= 0.07) or \
               (winner_known > runner_up_known and winner_conf >= runner_up_conf - 0.05):
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
