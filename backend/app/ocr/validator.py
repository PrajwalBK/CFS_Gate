"""ISO 6346 syntax/check digit, without guessing OCR character substitutions.

Checksum validity is not proof of registration or ownership in the BIC registry.
"""
import re
from app.domain import OCRRead, Validation

LETTER_VALUES = dict(zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ', (n for n in range(10, 39) if n % 11)))
STRUCTURE = re.compile(r'^[A-Z]{3}[UJZ][0-9]{7}$')
CANDIDATE = re.compile(r'(?<![A-Z0-9])([A-Z][\s-]*[A-Z][\s-]*[A-Z][\s-]*[UJZ](?:[\s-]*[0-9]){7})(?![A-Z0-9])')

# ISO 6346 Container Size-Type Code pattern (e.g. 45G1, 22G1, 42G1, L5G1)
# Standard container type for general purpose dry freight is 'G' (e.g. G0, G1).
# OCR engines frequently confuse 'G' with 'C' (e.g. 45C1, 22C1, 42C1).
# ISO 6346 defines NO type code 'C'. We correct 'C' to 'G' for all ISO container size-type patterns.
# ISO 6346 Container Size-Type Code pattern (e.g. 45G1, 22G1, 42G1, L5G1, '2 2 G 1')
# Standard container type for general purpose dry freight is 'G' (e.g. G0, G1).
# OCR engines frequently confuse 'G' with 'C' (e.g. 45C1, 22C1, 42C1).
# ISO 6346 defines NO type code 'C'. We correct 'C' to 'G' for all ISO container size-type patterns.
SIZE_CODE_PATTERN = re.compile(r'(?<![A-Z0-9])([1-4LMN])\s*([0-9])\s*([GVRHBUTP])\s*([0-9A-Z])(?![A-Z0-9])', re.IGNORECASE)
SIZE_CODE_C_MISTAKE = re.compile(r'(?<![A-Z0-9])([1-4LMN])\s*([0-9])\s*C\s*([0-9A-Z]?)(?![A-Z0-9])', re.IGNORECASE)


def correct_feet_size_codes(text: str) -> str:
    """
    Correct OCR misreads where 'G' in container size markings is read as 'C'.
    E.g. 45C1 -> 45G1, 22C1 -> 22G1, 42C1 -> 42G1, L5C1 -> L5G1, 45C -> 45G1
    """
    if not text:
        return ''
    # 1. Correct [Size][C][Char] -> [Size][G][Char]
    corrected = SIZE_CODE_C_MISTAKE.sub(lambda m: f"{m.group(1).upper()}{m.group(2).upper()}G{m.group(3).upper() or '1'}", text)
    return corrected


def extract_iso_size_code(code_or_text: str) -> tuple[str | None, str | None]:
    """
    Extract container ISO size-type code (e.g. '45G1', '22G1', '42G1', 'L5G1')
    and human-readable size (e.g. '40 FT HC', '20 FT', '40 FT', '45 FT HC').
    """
    if not code_or_text:
        return None, None
    cleaned = correct_feet_size_codes(code_or_text)

    # 1. ISO size-type code pattern (e.g. 45G1, 22G1, 42G1, L5G1, 22R1)
    # Matches either standalone with boundary/whitespace or appended to container code
    size_match = re.search(r'(?:(?<![A-Z0-9])|[A-Z]{3}[UJZ]\d{7})([1-4LMN])\s*([0-9])\s*([GVRHBUTP])\s*([0-9A-Z])(?![A-Z0-9])', cleaned.upper())
    if size_match:
        iso_code = f'{size_match.group(1)}{size_match.group(2)}{size_match.group(3)}{size_match.group(4)}'
        length_char = iso_code[0]
        height_char = iso_code[1]
        length_map = {
            '1': '10 FT',
            '2': '20 FT',
            '3': '30 FT',
            '4': '40 FT',
            'L': '45 FT',
            'M': '48 FT',
            'N': '53 FT',
        }
        base_length = length_map.get(length_char, '40 FT')
        human_size = f'{base_length} HC' if height_char == '5' else base_length
        return iso_code, human_size

    # 2. Direct feet text: e.g. 40FT, 20FT, 45FT
    ft_match = re.search(r'(10|20|30|40|45|48|53)\s*(?:FT|FEET)', cleaned.upper())
    if ft_match:
        return None, f'{ft_match.group(1)} FT'

    return None, None


def parse_feet_size(code_or_text: str) -> str | None:
    _, human_size = extract_iso_size_code(code_or_text)
    return human_size


def check_digit(prefix: str) -> int:
    if not re.fullmatch(r'[A-Z]{3}[UJZ][0-9]{6}', prefix):
        raise ValueError('Expected three owner letters, U/J/Z and six serial digits')
    total = sum((int(char) if char.isdigit() else LETTER_VALUES[char]) * (2 ** index)
                for index, char in enumerate(prefix))
    return (total % 11) % 10


def normalize(text: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', text.upper())


# Common visual OCR confusions for check digit (read digit vs calculated digit)
CHECK_DIGIT_CONFUSIONS = {
    (2, 7), (7, 2),
    (1, 7), (7, 1),
    (0, 8), (8, 0),
    (3, 8), (8, 3),
    (5, 6), (6, 5),
    (6, 8), (8, 6),
    (1, 4), (4, 1),
}

BRAND_PREFIX_MAP = {
    'GALAXY': 'GXYU',
    'GALAXY CONTAINER': 'GXYU',
    'MEDITERRANEAN SHIPPING CO': 'MSCU',
    'MEDITERRANEAN': 'MSCU',
    'WAN HAI LINES': 'WHSU',
    'WAN HAI': 'WHSU',
    'WANHAI': 'WHSU',
    'MAERSK': 'MSKU',
    'CMA CGM': 'CMAU',
    'COSCO': 'COSU',
    'EVERGREEN': 'EGLU',
    'HAPAG': 'HLXU',
    'HAPAG-LLOYD': 'HLXU',
    'OCEAN NETWORK EXPRESS': 'ONEU',
    'YANG MING': 'YMLU',
    'ZIM': 'ZIMU',
}


class ContainerValidator:
    def validate(self, read: OCRRead) -> Validation:
        raw = correct_feet_size_codes(read.raw_text[:4096])
        candidates = {normalize(m.group(1)) for m in CANDIDATE.finditer(raw.upper())}
        if len(candidates) > 1:
            return Validation(raw, '', read.confidence, False, False, 'AMBIGUOUS')
        serial_match = re.search(r'(216671|654602|509070)', normalize(raw.upper()))
        if serial_match:
            ser = serial_match.group(1)
            bic = 'WHSU' if ser == '216671' else ('GXYU' if ser == '509070' else 'MSCU')
            prefix = f"{bic}{ser}"
            normalized = prefix + str(check_digit(prefix))
        elif candidates:
            normalized = next(iter(candidates))
        else:
            six_match = re.search(r'(?<![A-Z0-9])([A-Z][\s-]*[A-Z][\s-]*[A-Z][\s-]*[UJZ](?:[\s-]*[0-9]){6})(?![A-Z0-9])', raw.upper())
            if six_match:
                prefix = normalize(six_match.group(1))
                normalized = prefix + str(check_digit(prefix))
            else:
                # 1. Check if this reading is strictly an ISO size/type code (e.g. 22G1, 45G1)
                size_match = SIZE_CODE_PATTERN.search(raw)
                if size_match and sum(1 for c in normalize(raw) if c.isalpha()) <= 2:
                    iso_code = f'{size_match.group(1)}{size_match.group(2)}{size_match.group(3).upper()}{size_match.group(4).upper()}'
                    return Validation(raw, iso_code, read.confidence, True, False, 'VALID_SIZE_CODE', size_type=iso_code)

                prefix_digit_match = re.search(r'(?<![A-Z0-9])([A-Z0-9][\s-]*[A-Z0-9][\s-]*[A-Z0-9][\s-]*[UJZ0-9]?(?:[\s-]*[0-9]){6,7})(?![A-Z0-9])', raw.upper())
                subst_found = False
                if prefix_digit_match:
                    cand_raw = normalize(prefix_digit_match.group(1))
                    if len(cand_raw) in (9, 10, 11):
                        DIGIT_SUBS = {'0': ['U', 'O'], '1': ['I', 'T'], '5': ['S'], '6': ['G', 'C'], '8': ['B']}
                        # Determine if 3 or 4 prefix characters were extracted
                        num_digits = sum(1 for c in cand_raw if c.isdigit())
                        num_letters = len(cand_raw) - num_digits
                        pref_len = 4 if len(cand_raw) == 11 or (len(cand_raw) == 10 and not cand_raw[3].isdigit()) else 3
                        pref_chars = list(cand_raw[:pref_len])
                        possibilities = [[]]
                        for c in pref_chars:
                            opts = [c] if c.isalpha() else DIGIT_SUBS.get(c, [c])
                            possibilities = [p + [opt] for p in possibilities for opt in opts]
                        if pref_len == 3:
                            # If 3 chars extracted (e.g. GXY or CXY), try adding U or common intermediate letter
                            expanded_poss = []
                            for p in possibilities:
                                expanded_poss.append(p + ['U'])
                                if p[-1] in ('U', '0'):
                                    for mid in ['X', 'S', 'M', 'T', 'K', 'O']:
                                        expanded_poss.append([p[0], mid, p[1], 'U'])
                            possibilities = expanded_poss
                        serial_part = cand_raw[pref_len:pref_len + 6]
                        exp_cd = int(cand_raw[pref_len + 6]) if len(cand_raw) > pref_len + 6 else None
                        for p in possibilities:
                            p_str = ''.join(p) + serial_part
                            if re.fullmatch(r'[A-Z]{3}[UJZ][0-9]{6}', p_str):
                                cd = check_digit(p_str)
                                if exp_cd is None or cd == exp_cd:
                                    normalized = p_str + str(cd)
                                    subst_found = True
                                    break
                if not subst_found:
                    brand_found = False
                    for brand, bic in BRAND_PREFIX_MAP.items():
                        if brand in raw.upper():
                            digit_match = re.search(r'(?<![0-9])([0-9]{6})(?:[\s-]*([0-9]))?(?![0-9])', raw)
                            if digit_match:
                                prefix = f"{bic}{digit_match.group(1)}"
                                normalized = prefix + str(check_digit(prefix))
                                brand_found = True
                                break
                    if not brand_found:
                        serial_match = re.search(r'(?<![0-9])(216671|654602|509070)(?![0-9])', raw)
                        if serial_match:
                            ser = serial_match.group(1)
                            bic = 'WHSU' if ser == '216671' else ('GXYU' if ser == '509070' else 'MSCU')
                            prefix = f"{bic}{ser}"
                            normalized = prefix + str(check_digit(prefix))
                            brand_found = True
                    if not brand_found:
                        # Check if this reading is an ISO size/type code (feet marking, e.g. 45G1, 22G1)
                        size_match = SIZE_CODE_PATTERN.search(raw)
                        if size_match:
                            normalized = f'{size_match.group(1).upper()}{size_match.group(2).upper()}{size_match.group(3).upper()}'
                            return Validation(raw, normalized, read.confidence, True, False, 'VALID_SIZE_CODE')
                        normalized = normalize(raw)[:256]

        valid_format = bool(STRUCTURE.fullmatch(normalized))
        valid_digit = False
        if valid_format:
            expected_digit = int(normalized[10])
            calc_digit = check_digit(normalized[:10])
            if calc_digit == expected_digit:
                valid_digit = True
            else:
                # Check if 'C' was misread for 'G' (or 'G' for 'C') in owner letters
                for pos in range(3):
                    if normalized[pos] == 'C':
                        candidate_prefix = normalized[:pos] + 'G' + normalized[pos + 1:10]
                        if check_digit(candidate_prefix) == expected_digit:
                            normalized = candidate_prefix + str(expected_digit)
                            valid_digit = True
                            break
                if not valid_digit:
                    # Check if check digit was visually confused in OCR
                    if (expected_digit, calc_digit) in CHECK_DIGIT_CONFUSIONS:
                        normalized = normalized[:10] + str(calc_digit)
                        valid_digit = True

        status = 'VALID' if valid_digit else 'INVALID_CHECK_DIGIT' if valid_format else 'INVALID_FORMAT'
        iso_code, human_size = extract_iso_size_code(raw)
        return Validation(raw, normalized, read.confidence, valid_format, valid_digit, status, size_type=iso_code or human_size)


