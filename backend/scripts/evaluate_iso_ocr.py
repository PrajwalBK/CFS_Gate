"""Batch-evaluate ISO size/type OCR on saved images.

This is intentionally an offline diagnostic: it never creates gate events.
Run from backend with the project virtual environment active.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config.settings import Settings
from app.ocr.ocr_engine import EasyOCREngine, OlmOCREngine
from app.ocr.validator import extract_iso_size_code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('snapshots'))
    parser.add_argument('--limit', type=int, default=250)
    parser.add_argument('--engine', choices=('easyocr', 'olmocr'), default='easyocr')
    parser.add_argument('--report', type=Path, default=Path('data/iso_ocr_evaluation.json'))
    args = parser.parse_args()
    settings = Settings(ocr_engine=args.engine)
    engine = EasyOCREngine(settings) if args.engine == 'easyocr' else OlmOCREngine(settings)
    paths = sorted(p for p in args.root.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'})[:args.limit]
    results = []
    started = time.perf_counter()
    for index, path in enumerate(paths, 1):
        image = cv2.imread(str(path))
        if image is None:
            continue
        item_started = time.perf_counter()
        read = engine.read_iso(image)
        code, size = extract_iso_size_code(read.raw_text)
        results.append({
            'file': str(path), 'raw_text': read.raw_text, 'iso_code': code,
            'container_size': size, 'confidence': read.confidence,
            'latency_ms': round((time.perf_counter() - item_started) * 1000, 1),
        })
        if index % 25 == 0:
            print(f'processed {index}/{len(paths)}')
    latencies = sorted(item['latency_ms'] for item in results)
    codes = [item['iso_code'] for item in results if item['iso_code']]
    report = {
        'engine': args.engine, 'images_seen': len(paths), 'images_processed': len(results),
        'iso_reads': len(codes), 'iso_read_rate': round(len(codes) / len(results), 4) if results else 0,
        'unique_iso_codes': sorted(set(codes)),
        'latency_ms': {
            'total': round((time.perf_counter() - started) * 1000, 1),
            'p50': latencies[len(latencies)//2] if latencies else None,
            'p95': latencies[max(0, int(len(latencies)*.95)-1)] if latencies else None,
        },
        'results': results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('engine', 'images_seen', 'images_processed', 'iso_reads', 'iso_read_rate', 'unique_iso_codes', 'latency_ms')}, indent=2))


if __name__ == '__main__':
    main()
