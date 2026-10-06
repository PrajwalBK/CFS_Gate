"""Empirical latency and performance benchmarking script."""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import time
import numpy as np
import cv2
from app.domain import Frame, Detection, OCRRead, Validation, Observation
from app.ocr.validator import ContainerValidator
from app.ocr.consensus import CharacterConsensus
from app.snapshots.snapshot_manager import SnapshotManager

def run_benchmark():
    results = {}
    
    # 1. Image generation (1080p sample)
    h, w = 1080, 1920
    test_img = np.zeros((h, w, 3), dtype=np.uint8)
    # Add synthetic container markings
    cv2.putText(test_img, "MSCU 123456 7", (400, 500), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
    cv2.putText(test_img, "22G1", (400, 600), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # 2. Quality & Best-Frame Metric Latency
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        q = SnapshotManager.compute_quality(test_img)
        times.append((time.perf_counter() - t0) * 1000)
    results['Frame Quality Computation (1080p)'] = f"{np.median(times):.2f} ms (p95: {np.percentile(times, 95):.2f} ms)"

    # 3. Text Crop Normalization & Padding Latency
    times = []
    bbox = (380, 450, 850, 550)
    for _ in range(100):
        t0 = time.perf_counter()
        bw = max(1.0, bbox[2] - bbox[0])
        bh = max(1.0, bbox[3] - bbox[1])
        pad_x = max(25, int(bw * 0.15))
        pad_y = max(15, int(bh * 0.25))
        y1 = max(0, int(bbox[1]) - pad_y)
        y2 = min(h, int(bbox[3]) + pad_y)
        x1 = max(0, int(bbox[0]) - pad_x)
        x2 = min(w, int(bbox[2]) + pad_x)
        crop = test_img[y1:y2, x1:x2].copy()
        times.append((time.perf_counter() - t0) * 1000)
    results['Crop Extraction & Adaptive Padding'] = f"{np.median(times):.3f} ms (p95: {np.percentile(times, 95):.3f} ms)"

    # 4. ISO 6346 Single-Frame Validation & Check-Digit Computation
    validator = ContainerValidator()
    read_obj = OCRRead("MSCU 123456 7", 0.95)
    times = []
    for _ in range(500):
        t0 = time.perf_counter()
        val = validator.validate(read_obj)
        times.append((time.perf_counter() - t0) * 1000)
    results['ISO 6346 Validator & Check Digit'] = f"{np.median(times):.3f} ms (p95: {np.percentile(times, 95):.3f} ms)"

    # 5. Multi-Frame Character-Level Consensus
    multi_reads = [
        Validation('MSCU1284567', 'MSCU1284567', 0.80, True, False, 'INVALID_CHECK_DIGIT'),
        Validation('MSCU1234562', 'MSCU1234562', 0.82, True, False, 'INVALID_CHECK_DIGIT'),
        Validation('MSCU1234567', 'MSCU1234567', 0.94, True, True, 'VALID', size_type='22G1'),
        Validation('MSCU1234567', 'MSCU1234567', 0.91, True, True, 'VALID', size_type='22G1'),
    ]
    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        c_res = CharacterConsensus.resolve(multi_reads)
        times.append((time.perf_counter() - t0) * 1000)
    results['Multi-Frame Character Consensus (4 frames)'] = f"{np.median(times):.3f} ms (p95: {np.percentile(times, 95):.3f} ms)"

    # 6. Primary OCR Inference Benchmark (PaddleOCR subprocess communication)
    try:
        from app.ocr.ocr_engine import create_ocr_engine
        from app.config.settings import Settings
        settings = Settings(_env_file='.env')
        ocr_engine = create_ocr_engine(settings)
        crop_sample = test_img[450:550, 380:850]
        # warmup
        ocr_engine.read(crop_sample)
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            r = ocr_engine.read(crop_sample)
            times.append((time.perf_counter() - t0) * 1000)
        results['PaddleOCR Crop Inference'] = f"{np.median(times):.2f} ms (p95: {np.percentile(times, 95):.2f} ms)"
    except Exception as e:
        results['PaddleOCR Crop Inference'] = f"Skipped (isolated subprocess): {e}"

    print("--- BENCHMARK RESULTS ---")
    for k, v in results.items():
        print(f"{k}: {v}")

if __name__ == '__main__':
    run_benchmark()
