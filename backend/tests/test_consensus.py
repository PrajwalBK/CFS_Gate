import pytest
from app.domain import Validation
from app.ocr.consensus import CharacterConsensus


def test_consensus_exact_match():
    reads = [
        Validation('MSCU1234566', 'MSCU1234566', 0.92, True, True, 'VALID', size_type='22G1'),
        Validation('MSCU1234566', 'MSCU1234566', 0.95, True, True, 'VALID', size_type='22G1'),
    ]
    res = CharacterConsensus.resolve(reads)
    assert res is not None
    assert res.normalized_text == 'MSCU1234566'
    assert res.valid_check_digit is True
    assert res.size_type == '22G1'


def test_consensus_character_level_voting_recovers_error():
    # Frame 1 has error at position 6 (8 instead of 3)
    # Frame 2 has missing character at end
    # Frame 3 and 4 have correct characters
    reads = [
        Validation('MSCU1284567', 'MSCU1284567', 0.70, True, False, 'INVALID_CHECK_DIGIT'),
        Validation('MSCU123456?', 'MSCU1234560', 0.75, True, False, 'INVALID_CHECK_DIGIT'),
        Validation('MSCU1234566', 'MSCU1234566', 0.88, True, True, 'VALID'),
        Validation('MSCU1234566', 'MSCU1234566', 0.91, True, True, 'VALID'),
    ]
    res = CharacterConsensus.resolve(reads)
    assert res is not None
    assert res.normalized_text == 'MSCU1234566'
    assert res.valid_check_digit is True


def test_consensus_recovers_all_invalid_frames_via_character_voting():
    # ALL individual frames have check digit False, but character voting reveals the true code
    reads = [
        Validation('MSCU1284567', 'MSCU1284567', 0.80, True, False, 'INVALID_CHECK_DIGIT'),
        Validation('MSCU1234562', 'MSCU1234562', 0.82, True, False, 'INVALID_CHECK_DIGIT'),
        Validation('MSCU1234566', 'MSCU1234566', 0.85, True, True, 'VALID'),
    ]
    res = CharacterConsensus.resolve(reads)
    assert res is not None
    assert res.normalized_text == 'MSCU1234566'
    assert res.valid_check_digit is True


def test_consensus_preserves_size_code():
    reads = [
        Validation('MSCU1234566', 'MSCU1234566', 0.92, True, True, 'VALID', size_type=None),
        Validation('22G1', '22G1', 0.85, True, False, 'VALID_SIZE_CODE', size_type='22G1'),
    ]
    res = CharacterConsensus.resolve(reads)
    assert res is not None
    assert res.normalized_text == 'MSCU1234566'
    assert res.size_type == '22G1'


def test_consensus_does_not_swallow_size_code_or_hallucinate():
    # Raw reads with size code appended to container markings
    reads = [
        Validation('6 E S U 1 1 3 2 1 B 22G1', '6ESU11321B', 0.90, False, False, 'INVALID_FORMAT', size_type='22G1'),
        Validation('GESU1273128', 'GESU1273128', 0.91, True, True, 'VALID', size_type='22G1'),
    ]
    res = CharacterConsensus.resolve(reads)
    assert res is not None
    # Must correctly resolve to the valid container number, NOT swallow '22' to make 'ESU11321B22' -> 'EBUU1321822'
    assert res.normalized_text == 'GESU1273128'
    assert res.size_type == '22G1'


@pytest.mark.parametrize('text', ['TIIU507766', 'MSCU1234567', 'CA1U2528405'])
def test_consensus_never_repairs_or_completes_identity(text):
    read = Validation(text, text, .99, len(text) == 11, False, 'INVALID_CHECK_DIGIT')
    result = CharacterConsensus.resolve([read, read, read])
    assert not result.valid_check_digit
    assert result.normalized_text == text


def test_conflicting_observed_valid_ids_are_ambiguous():
    result = CharacterConsensus.resolve([
        Validation('MSCU1234566', 'MSCU1234566', .98, True, True, 'VALID'),
        Validation('MSKU1234565', 'MSKU1234565', .99, True, True, 'VALID')])
    assert result.validation_status == 'AMBIGUOUS'
    assert not result.valid_check_digit


def test_consensus_majority_and_confidence_overcomes_single_noisy_checksum_match():
    # Coincidental valid check digit on a blurry frame (e.g. ADMU6211047 conf 0.679)
    # followed by sharp reads of the true container (ADMU2111477 conf 0.985)
    noisy = Validation('ADMU6211047', 'ADMU6211047', 0.679, True, True, 'VALID')
    sharp1 = Validation('ADMU2111477', 'ADMU2111477', 0.985, True, True, 'VALID')
    sharp2 = Validation('ADMU2111477', 'ADMU2111477', 0.989, True, True, 'VALID')
    result = CharacterConsensus.resolve([noisy, sharp1, sharp2])
    assert result.normalized_text == 'ADMU2111477'
    assert result.valid_check_digit is True
    assert result.validation_status == 'VALID'

