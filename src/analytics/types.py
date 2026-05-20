"""
Type definitions, Protocols, and Enums for Sports Analytics.

PEP 484 type hints, Protocol interfaces, and magic string enums to improve
type safety, IDE support, and code clarity across the analytics pipeline.
"""

from enum import Enum
from typing import Protocol, runtime_checkable, Callable, Any, TypeVar, Optional
from dataclasses import dataclass
from pathlib import Path
import time
import logging
import contextlib
from functools import wraps

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# MAGIC STRING ENUMS (replacing hardcoded string literals)
# ─────────────────────────────────────────────────────────────────────────────

class YoloModelSize(str, Enum):
    """Valid YOLO model sizes for pose estimation."""
    NANO = "n"
    SMALL = "s"
    MEDIUM = "m"
    LARGE = "l"
    XLARGE = "x"

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if a string is a valid YOLO model size."""
        return value in [member.value for member in cls]

    @classmethod
    def get_default(cls) -> "YoloModelSize":
        return cls.MEDIUM


class VideoCodec(str, Enum):
    """Supported video codec formats for output writing."""
    MP4V = "mp4v"
    MJPEG = "MJPG"
    X264 = "X264"
    DIVX = "DIVX"


class AnalysisMode(str, Enum):
    """Analysis execution modes."""
    INTERACTIVE = "interactive"
    AUTOMATIC = "automatic"
    BATCH = "batch"


class BiomechanicsBackend(str, Enum):
    """Available biomechanics angle calculation backends."""
    SPORTS2D = "sports2d"
    SCIPY = "scipy"
    NUMPY = "numpy"

    def __str__(self) -> str:
        return self.value


class RiskLevel(str, Enum):
    """Risk score classification levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

    def to_threshold_range(self) -> tuple[float, float]:
        """Convert risk level to score threshold range (0-100)."""
        ranges = {
            RiskLevel.LOW: (0.0, 25.0),
            RiskLevel.MODERATE: (25.0, 50.0),
            RiskLevel.HIGH: (50.0, 75.0),
            RiskLevel.CRITICAL: (75.0, 100.0),
        }
        return ranges[self]

    @classmethod
    def from_score(cls, score: float) -> "RiskLevel":
        """Get risk level from numerical score (0-100)."""
        if score < 25:
            return cls.LOW
        elif score < 50:
            return cls.MODERATE
        elif score < 75:
            return cls.HIGH
        else:
            return cls.CRITICAL


class ExportFormat(str, Enum):
    """Supported unified export formats (TRC/MOT are produced by Sports2D, not here)."""
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"

    @property
    def file_extension(self) -> str:
        """Get file extension including dot."""
        return f".{self.value}"


# ─────────────────────────────────────────────────────────────────────────────
# TYPE PROTOCOLS
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class PoseFrame(Protocol):
    frame_idx: int
    timestamp: float
    keypoints: Any
    confidences: Any


@runtime_checkable
class AnalyzerLike(Protocol):
    def estimate(self, frame: Any) -> Any: ...
    def set_confidence_threshold(self, threshold: float) -> None: ...


@runtime_checkable
class ExportWriter(Protocol):
    def write(self, data: Any, output_path: str) -> None: ...
    def validate(self, data: Any) -> bool: ...


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE BENCHMARKING
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    name: str
    elapsed_ms: float
    iteration_count: int = 1

    @property
    def avg_ms(self) -> float:
        return self.elapsed_ms / self.iteration_count

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "elapsed_ms": self.elapsed_ms,
            "iteration_count": self.iteration_count,
            "avg_ms": self.avg_ms,
        }

    def __str__(self) -> str:
        if self.iteration_count == 1:
            return f"{self.name}: {self.elapsed_ms:.2f}ms"
        return f"{self.name}: {self.elapsed_ms:.2f}ms ({self.iteration_count} iter, avg {self.avg_ms:.2f}ms)"


class PerformanceTimer:
    def __init__(self, name: str, log_level: int = logging.DEBUG):
        self.name = name
        self.log_level = log_level
        self.start_time: Optional[float] = None
        self.result: Optional[BenchmarkResult] = None

    def __enter__(self) -> "PerformanceTimer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.start_time is None:
            return
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        self.result = BenchmarkResult(self.name, elapsed_ms)
        logger.log(self.log_level, f"⏱️  {self.result}")


def benchmark_method(threshold_ms: float = 100.0) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                func_name = f"{func.__module__}.{func.__qualname__}"
                if threshold_ms > 0 and elapsed_ms > threshold_ms:
                    logger.warning(f"⚠️  {func_name} exceeded: {elapsed_ms:.2f}ms > {threshold_ms:.2f}ms")
                else:
                    logger.debug(f"⏱️  {func_name}: {elapsed_ms:.2f}ms")
        return wrapper
    return decorator


class PipelineTimer:
    def __init__(self, name: str):
        self.name = name
        self.stages: dict[str, list[float]] = {}

    def record(self, stage_name: str, elapsed_ms: float) -> None:
        if stage_name not in self.stages:
            self.stages[stage_name] = []
        self.stages[stage_name].append(elapsed_ms)

    @contextlib.contextmanager
    def stage(self, stage_name: str):
        """Context manager for timing a pipeline stage."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.record(stage_name, elapsed_ms)

    def report(self) -> str:
        lines = [f"\n{'='*70}", f"Pipeline Report: {self.name}", f"{'='*70}"]
        total_ms = 0.0
        for stage_name in sorted(self.stages.keys()):
            times = self.stages[stage_name]
            total_ms_stage = sum(times)
            total_ms += total_ms_stage
            lines.append(
                f"  {stage_name:.<40} "
                f"avg={total_ms_stage/len(times):7.2f}ms  n={len(times)}"
            )
        lines.append(f"  {'TOTAL':.<40} {total_ms:7.2f}ms")
        lines.append(f"{'='*70}\n")
        return "\n".join(lines)

    def log_report(self, level: int = logging.INFO) -> None:
        logger.log(level, self.report())

    def reset(self) -> None:
        self.stages.clear()


# ─────────────────────────────────────────────────────────────────────────────
# TYPE ALIASES
# ─────────────────────────────────────────────────────────────────────────────

T = TypeVar("T")
PixelCoordinate = tuple[int, int]
NormalizedCoordinate = tuple[float, float]
FrameArray = Any
KeypointArray = Any
ConfidenceArray = Any


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Enums
    "YoloModelSize",
    "VideoCodec",
    "AnalysisMode",
    "BiomechanicsBackend",
    "RiskLevel",
    "ExportFormat",
    # Protocols
    "PoseFrame",
    "AnalyzerLike",
    "ExportWriter",
    # Benchmarking
    "BenchmarkResult",
    "PerformanceTimer",
    "benchmark_method",
    "PipelineTimer",
    # Type aliases
    "T",
    "PixelCoordinate",
    "NormalizedCoordinate",
    "FrameArray",
    "KeypointArray",
    "ConfidenceArray",
]