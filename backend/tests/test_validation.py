import pytest
from app.domain import OCRRead
from app.ocr.validator import ContainerValidator, check_digit, normalize


@pytest.mark.parametrize('text', ['CSQU3054383', 'csqu 305438 3', 'CSQU-305438-3', 'ID: CSQU\n305438\n3'])
def test_valid_container(text):
    value = ContainerValidator().validate(OCRRead(text, .94))
    assert value.raw_text == text
    assert value.normalized_text == 'CSQU3054383'
    assert value.valid_format and value.valid_check_digit


def test_six_digit_prefix_derives_check_digit():
    val = ContainerValidator().validate(OCRRead('GXYU 509070', .95))
    assert val.normalized_text == 'GXYU5090704'
    assert val.valid_format and val.valid_check_digit and val.validation_status == 'VALID'


@pytest.mark.parametrize('text,status', [('CSQU3054384', 'INVALID_CHECK_DIGIT'), ('CSQA3054383', 'INVALID_FORMAT'),
                                       ('CSQU3O54383', 'INVALID_FORMAT'), ('CSQU30543833', 'INVALID_FORMAT'),
                                       ('No number visible', 'INVALID_FORMAT'), ('', 'INVALID_FORMAT')])
def test_invalid_ocr_is_not_trusted(text, status):
    result = ContainerValidator().validate(OCRRead(text, .99))
    assert result.validation_status == status
    assert not result.valid_check_digit


def test_ambiguous_ids_and_checksum_algorithm():
    assert check_digit('CSQU305438') == 3
    assert check_digit('MSCU663987') == 0
    result = ContainerValidator().validate(OCRRead('CSQU3054383 MSCU6639870', .9))
    assert result.validation_status == 'AMBIGUOUS'
    assert normalize(' abcu-123456-7 ') == 'ABCU1234567'
    with pytest.raises(ValueError):
        check_digit('INVALID')


def test_feet_size_code_c_to_g_correction():
    from app.ocr.validator import correct_feet_size_codes, parse_feet_size
    assert correct_feet_size_codes('45C1') == '45G1'
    assert correct_feet_size_codes('22C1') == '22G1'
    assert correct_feet_size_codes('42C1') == '42G1'
    assert correct_feet_size_codes('L5C1') == 'L5G1'
    assert correct_feet_size_codes('45C') == '45G1'
    assert correct_feet_size_codes('42 C 1') == '42G1'

    assert parse_feet_size('45C1') == '40 FT HC'
    assert parse_feet_size('45G1') == '40 FT HC'
    assert parse_feet_size('42G1') == '40 FT'
    assert parse_feet_size('22G1') == '20 FT'
    assert parse_feet_size('L5G1') == '45 FT HC'
    assert parse_feet_size('40 FT') == '40 FT'
    assert parse_feet_size('20FEET') == '20 FT'

    # Validator normalizes 45C1 directly to 45G1
    val = ContainerValidator().validate(OCRRead('45C1', .95))
    assert val.normalized_text == '45G1'
    assert val.validation_status == 'VALID_SIZE_CODE'


def test_delimiter_tolerant_container_read():
    raw_trhu = (
        'TRITON\n'
        'TRHU1 1545895\n'
        '22G1\n'
        'MAXGROS 30.480 KGS\n'
        '67.200 LBS'
    )
    val = ContainerValidator().validate(OCRRead(raw_trhu, .95))
    assert val.normalized_text == 'TRHU1545895'
    assert val.valid_format and val.valid_check_digit
    assert val.validation_status == 'VALID'
    assert val.size_type == '22G1'


def test_uuid_and_tracker_ids_are_strictly_rejected():
    from app.ocr.validator import is_uuid, is_valid_iso_6346
    uuid_str = '3c780c27-e351-408d-b642-bdcf8e046d90'
    assert is_uuid(uuid_str)
    assert not is_valid_iso_6346(uuid_str)
    val = ContainerValidator().validate(OCRRead(uuid_str, 0.99))
    assert val.normalized_text == ''
    assert not val.valid_format
    assert not val.valid_check_digit
    assert val.validation_status == 'INVALID_FORMAT'


def test_iso_6346_check_digit_strictness():
    from app.ocr.validator import is_valid_iso_6346
    # Valid ISO containers with mathematically correct check digits
    assert is_valid_iso_6346('MSCU1234566')  # check digit 6
    assert is_valid_iso_6346('CSQU3054383')  # check digit 3
    assert is_valid_iso_6346('TRHU1545895')  # check digit 5
    assert is_valid_iso_6346('MSCU6639870')  # check digit 0
    
    # Check digits must match exactly (no fuzzy substitutions)
    assert not is_valid_iso_6346('MSCU1234567')  # wrong check digit (expected 6)
    assert not is_valid_iso_6346('MSCU1234568')  # wrong check digit
    assert not is_valid_iso_6346('3c780c27-e351-408d-b642-bdcf8e046d90')  # UUID
    assert not is_valid_iso_6346('UNKNOWN123')
    assert not is_valid_iso_6346('MSCU12345')  # too short
    assert not is_valid_iso_6346('MSCU12345678')  # too long


def test_user_reported_suspicious_values_handling():
    from app.ocr.validator import is_valid_iso_6346, is_uuid
    validator = ContainerValidator()

    # 1. UUID must be completely rejected as an invalid container number
    uuid_val = '3c780c27-e351-408d-b642-bdcf8e046d90'
    assert is_uuid(uuid_val)
    assert not is_valid_iso_6346(uuid_val)
    res = validator.validate(OCRRead(uuid_val, 0.99))
    assert res.normalized_text == ''
    assert not res.valid_format
    assert not res.valid_check_digit

    # 2. Mathematical check digit validation for the reported examples
    for candidate in ['IHEU9036531', 'CLLU0041015', 'NLMU2345670']:
        calc = check_digit(candidate[:10])
        assert calc == int(candidate[10])
        assert is_valid_iso_6346(candidate)
        
        # Tampered or single noisy misread digit must be strictly rejected
        tampered = candidate[:10] + str((calc + 1) % 10)
        assert not is_valid_iso_6346(tampered)
        t_res = validator.validate(OCRRead(tampered, 0.95))
        assert not t_res.valid_check_digit



