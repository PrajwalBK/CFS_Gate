"""High-resolution latency measurement and benchmarking for the real-time container pipeline.

Measures runtime latency for each stage:
- Detection latency
- Tracking latency
- Frame quality filtering latency
- OCR latency
- OCR aggregation latency
- ISO validation latency
- Event creation latency
- API queue latency
- Total processing latency
"""
import logging
import time
from dataclasses import dataclass, field

log = logging.getLogger('gate.pipeline')


@dataclass
class LatencyRecord:
    origin_key: str
    t_start: float = field(default_factory=time.perf_counter)
    detection_ms: float = 0.0
    tracking_ms: float = 0.0
    filtering_ms: float = 0.0
    ocr_ms: float = 0.0
    consensus_ms: float = 0.0
    iso_validation_ms: float = 0.0
    event_creation_ms: float = 0.0
    api_queue_ms: float = 0.0
    total_ms: float = 0.0

    def finalize(self) -> "LatencyRecord":
        self.total_ms = (time.perf_counter() - self.t_start) * 1000.0
        return self

    def summary(self) -> str:
        return (
            f"\n{'='*20} PIPELINE LATENCY BENCHMARK {'='*20}\n"
            f"Detection:         {self.detection_ms:7.2f} ms\n"
            f"Tracking:          {self.tracking_ms:7.2f} ms\n"
            f"Frame Filtering:   {self.filtering_ms:7.2f} ms\n"
            f"OCR (Parallel):    {self.ocr_ms:7.2f} ms\n"
            f"Consensus:         {self.consensus_ms:7.3f} ms\n"
            f"ISO Validation:    {self.iso_validation_ms:7.3f} ms\n"
            f"Event Creation:    {self.event_creation_ms:7.2f} ms\n"
            f"API Queue:         {self.api_queue_ms:7.3f} ms\n"
            f"{'-'*68}\n"
            f"Total Processing:  {self.total_ms:7.2f} ms\n"
            f"{'='*68}"
        )


class PipelineProfiler:
    """Singleton/thread-safe profiler recording real latency across pipeline components."""
    
    def __init__(self, metrics=None):
        self.metrics = metrics

    def new_record(self, origin_key: str) -> LatencyRecord:
        return LatencyRecord(origin_key=origin_key)

    def record_and_log(self, record: LatencyRecord, container_number: str = ""):
        rec = record.finalize()
        log.info(
            "pipeline_latency_benchmark",
            extra={
                "origin": rec.origin_key,
                "container_number": container_number,
                "detection_ms": round(rec.detection_ms, 2),
                "tracking_ms": round(rec.tracking_ms, 2),
                "filtering_ms": round(rec.filtering_ms, 2),
                "ocr_ms": round(rec.ocr_ms, 2),
                "consensus_ms": round(rec.consensus_ms, 3),
                "iso_validation_ms": round(rec.iso_validation_ms, 3),
                "event_creation_ms": round(rec.event_creation_ms, 2),
                "api_queue_ms": round(rec.api_queue_ms, 3),
                "total_ms": round(rec.total_ms, 2),
            }
        )
        if self.metrics:
            self.metrics.latency("pipeline_detection", rec.detection_ms / 1000.0)
            self.metrics.latency("pipeline_ocr", rec.ocr_ms / 1000.0)
            self.metrics.latency("pipeline_event", rec.event_creation_ms / 1000.0)
            self.metrics.latency("pipeline_total", rec.total_ms / 1000.0)
        return rec
