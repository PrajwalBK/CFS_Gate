from pathlib import Path
import pytest
from unittest.mock import patch, Mock

from app.config.settings import Settings
from app.ocr.ocr_engine import PaddleOCREngine
from app.ocr.ocr_engine import EasyOCREngine
from app.domain import OCRRead
import numpy as np


def test_small_crop_retry_uses_observed_valid_text_without_digit_repair():
    engine = EasyOCREngine.__new__(EasyOCREngine)
    engine.min_confidence = .6
    engine._read_image = Mock(side_effect=[OCRRead('MRSU977196', .95), OCRRead('MRSU9771962', .75)])
    crop = np.zeros((120, 220, 3), dtype=np.uint8)
    result = engine.read(crop)
    assert result == OCRRead('MRSU9771962', .75)
    assert engine._read_image.call_args.args[0].shape == (360, 660, 3)


def test_small_crop_retry_cannot_choose_between_conflicting_valid_identities():
    engine = EasyOCREngine.__new__(EasyOCREngine)
    engine.min_confidence = .6
    engine._read_image = Mock(side_effect=[OCRRead('MSKU1234565', .4), OCRRead('MSCU1234566', .9)])
    assert engine.read(np.zeros((100, 200, 3), dtype=np.uint8)) == OCRRead('', 0)


def test_valid_confident_read_does_not_need_enlargement():
    engine = EasyOCREngine.__new__(EasyOCREngine)
    engine._read_image = Mock(return_value=OCRRead('MSKU1234565', .9))
    assert engine.read(np.zeros((100, 200, 3), dtype=np.uint8)).raw_text == 'MSKU1234565'
    engine._read_image.assert_called_once()


def test_initialization_failure_honors_easyocr_fallback():
    from app.ocr.ocr_engine import create_ocr_engine
    settings = Settings(_env_file=None, ocr_engine='paddleocr', ocr_fallback_engine='easyocr')
    fallback = Mock()
    with patch('app.ocr.ocr_engine.PaddleOCREngine', side_effect=RuntimeError('blocked DLL')), \
         patch('app.ocr.ocr_engine.EasyOCREngine', return_value=fallback), \
         patch('app.ocr.ocr_engine.OlmOCREngine') as unconfigured:
        assert create_ocr_engine(settings) is fallback
        unconfigured.assert_not_called()


def test_initialization_failure_does_not_override_disabled_fallback():
    from app.ocr.ocr_engine import create_ocr_engine
    settings = Settings(_env_file=None, ocr_engine='paddleocr', ocr_fallback_engine='none')
    with patch('app.ocr.ocr_engine.PaddleOCREngine', side_effect=RuntimeError('blocked DLL')), \
         patch('app.ocr.ocr_engine.OlmOCREngine') as unconfigured:
        with pytest.raises(RuntimeError, match='blocked DLL'):
            create_ocr_engine(settings)
        unconfigured.assert_not_called()


def test_paddle_worker_reads_saved_container_snapshot():
    root = Path(__file__).resolve().parents[1]
    image_path = root / 'snapshots' / '195f1efc-b762-4bca-8780-98b510461d19' / 'gate-in-1_9a560466-f173-4b27-8f41-3cf476cff414.jpg'
    if not image_path.is_file():
        pytest.skip("Real OCR acceptance image is unavailable")
    import cv2
    # The parent is the real GateRuntime: it already has PyTorch/YOLO loaded.
    # The Paddle child must still start independently on Windows.
    import torch
    assert torch.cuda.is_available()
    settings = Settings(ocr_engine='paddleocr', ocr_gpu=True, ocr_fallback_engine='none')
    engine = PaddleOCREngine(settings)
    try:
        result = engine.read(cv2.imread(str(image_path)))
    finally:
        engine.close()
    assert 'MSMU' in result.raw_text
    assert result.confidence > 0
