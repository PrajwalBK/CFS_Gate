from pathlib import Path

from app.config.settings import Settings
from app.ocr.ocr_engine import PaddleOCREngine


def test_paddle_worker_reads_saved_container_snapshot():
    root = Path(__file__).resolve().parents[1]
    image_path = root / 'snapshots' / '195f1efc-b762-4bca-8780-98b510461d19' / 'gate-in-1_9a560466-f173-4b27-8f41-3cf476cff414.jpg'
    if not image_path.is_file():
        return
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
