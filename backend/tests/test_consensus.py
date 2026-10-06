import pytest
from app.domain import Validation
from app.ocr.consensus import CharacterConsensus


def test_consensus_exact_match():
    reads = [
        Validation('MSCU1234567', 'MSCU1234567', 0.92, True, True, 'VALID', size_type='22G1'),
        Validation('MSCU1234567', 'MSCU1234567', 0.95, True, True, 'VALID', size_type='22G1'),
    ]
    res = CharacterConsensus.resolve(reads)
    assert res is not None
    assert res.normalized_text == 'MSCU1234567'
    assert res.valid_check_digit is True
    assert res.size_type == '22G1'


def test_consensus_character_level_voting_recovers_error():
    # Frame 1 has error at position 6 (8 instead of 3)
    # Frame 2 has missing character at end
    # Frame 3 and 4 have correct characters
    reads = [
        Validation('MSCU1284567', 'MSCU1284567', 0.70, True, False, 'INVALID_CHECK_DIGIT'),
        Validation('MSCU123456?', 'MSCU1234560', 0.75, True, False, 'INVALID_CHECK_DIGIT'),
        Validation('MSCU1234567', 'MSCU1234567', 0.88, True, True, 'VALID'),
        Validation('MSCU1234567', 'MSCU1234567', 0.91, True, True, 'VALID'),
    ]
    res = CharacterConsensus.resolve(reads)
    assert res is not None
    assert res.normalized_text == 'MSCU1234567'
    assert res.valid_check_digit is True


def test_consensus_recovers_all_invalid_frames_via_character_voting():
    # ALL individual frames have check digit False, but character voting reveals the true code
    reads = [
        Validation('MSCU1284567', 'MSCU1284567', 0.80, True, False, 'INVALID_CHECK_DIGIT'),
        Validation('MSCU1234562', 'MSCU1234562', 0.82, True, False, 'INVALID_CHECK_DIGIT'),
        Validation('MSCU1234567', 'MSCU1234567', 0.85, True, True, 'VALID'),
    ]
    res = CharacterConsensus.resolve(reads)
    assert res is not None
    assert res.normalized_text == 'MSCU1234567'
    assert res.valid_check_digit is True


def test_consensus_preserves_size_code():
    reads = [
        Validation('MSCU1234567', 'MSCU1234567', 0.92, True, True, 'VALID', size_type=None),
        Validation('22G1', '22G1', 0.85, True, False, 'VALID_SIZE_CODE', size_type='22G1'),
    ]
    res = CharacterConsensus.resolve(reads)
    assert res is not None
    assert res.normalized_text == 'MSCU1234567'
    assert res.size_type == '22G1'
