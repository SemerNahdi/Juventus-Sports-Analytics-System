"""Type definitions, Protocols, and Enums for Sports Analytics.

PEP 484 type hints, Protocol interfaces, and magic string enums to improve
type safety, IDE support, and code clarity across the analytics pipeline.
"""

from enum import Enum, auto
from typing import Protocol, runtime_checkable, Callable, Any, TypeVar, Optional
from dataclasses import dataclass
import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# MAGIC STRING ENUMS (replacing hardcoded string literals)
# ─────────────────────────────────────────────────────────────────────────────

class YoloModelSize(str, Enum):
    """Valid YOLO model sizes for pose estimation.
    
    Order from smallest to largest (inference speed vs accuracy tradeoff):
    - NANO ("n"): Fastest, least accurate
    - SMALL ("s"): Fast, balanced
    - MEDIUM ("m"): Default, good accuracy
    - LARGE ("l"): Slower, high accuracy
    - XLARGE ("x"): Slowest, best accuracy
    """
    NANO = "n"
    SMALL = "s"
    MEDIUM = "m"
    LARGE = "l"
    XLARGE = "x"

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if a string is a valid YOLO model size."""
        return value in cls.__members__.values()

    @classmethod
    def get_default(cls) -> "YoloModelSize":
        """Return the default YOLO model size (MEDIUM)."""
        return cls.MEDIUM


class VideoCodec(str, Enum):
    """Supported video codec formats for output writing."""
    MP4V = "mp4v"      # Modern H.264 variant
    MJPEG = "MJPG"     # Motion JPEG (universal compatibility)
    X264 = "X264"      # H.264 codec
    DIVX = "DIVX"      # DivX codec


class AnalysisMode(str, Enum):
    """Analysis execution modes."""
    INTERACTIVE = "interactive"  # Interactive player selection
    AUTOMATIC = "automatic"      # Auto-detect primary player
    BATCH = "batch"              # Batch processing mode


class BiomechanicsBackend(str, Enum):
    """Available biomechanics angle calculation backends."""
    SPORTS2D = "sports2d"       # Native Sports2D OpenSim integration
    SCIPY = "scipy"             # SciPy-based angle calculation
    NUMPY = "numpy"             # NumPy-based fallback


class RiskLevel(str, Enum):
    """Risk score classification levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

    def to_threshold_range(self) -> tuple[float, float]:
        """Convert risk level to score threshold range (0-100)."""
        ranges = {
            self.LOW: (0.0, 25.0),
            self.MODERATE: (25.0, 50.0),
            self.HIGH: (50.0, 75.0),
            self.CRITICAL: (75.0, 100.0),
        }
        return ranges[self]


class ExportFormat(str, Enum):
    """Supported data export formats."""
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    TRC = "trc"          # OpenSim track format
    MOT = "mot"          # OpenSim motion format


# ─────────────────────────────────────────────────────────────────────────────
# TYPE PROTOCOLS (structural typing for duck-typed components)
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class PoseFrame(Protocol):
    """Protocol for pose estimation frame data."""
    
    frame_idx: int
    timestamp: float
    keypoints: Any  # 2D array of (x, y) positions
    confidences: Any  # 1D array of detection confidences


@runtime_checkable
class AnalyzerLike(Protocol):
    """Protocol for pose analyzer implementations."""
    
    def estimate(self, frame: Any) -> Any:
        """Estimate pose from video frame."""
        ...
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Set detection confidence threshold."""
        ...


@runtime_checkable
class ExportWriter(Protocol):
    """Protocol for export format writers."""
    
    def write(self, data: Any, output_path: str) -> None:
        """Write analysis data to output file."""
        ...
    
    def validate(self, data: Any) -> bool:
        """Validate data before writing."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE BENCHMARKING INSTRUMENTATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    """Result of a performance benchmark."""
    name: str
    elapsed_ms: float
    iteration_count: int = 1
    
    @property
    def avg_ms(self) -> float:
        """Average time per iteration in milliseconds."""
        return self.elapsed_ms / self.iteration_count
    
    def __str__(self) -> str:
        if self.iteration_count == 1:
            return f"{self.name}: {self.elapsed_ms:.2f}ms"
        return f"{self.name}: {self.elapsed_ms:.2f}ms ({self.iteration_count} iterations, avg {self.avg_ms:.2f}ms)"


class PerformanceTimer:
    """Context manager for measuring code execution time with automatic logging."""
    
    def __init__(self, name: str, log_level: int = logging.DEBUG) -> None:
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
        
        elapsed_s = time.perf_counter() - self.start_time
        elapsed_ms = elapsed_s * 1000
        self.result = BenchmarkResult(self.name, elapsed_ms)
        
        logger.log(self.log_level, f"⏱️  {self.result}")


def benchmark_method(threshold_ms: float = 100.0) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to benchmark method execution time.
    
    Logs WARNING if execution exceeds threshold_ms.
    
    Args:
        threshold_ms: Log threshold in milliseconds. If 0, always logs at DEBUG.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                func_name = f"{func.__module__}.{func.__qualname__}"
                
                if threshold_ms > 0 and elapsed_ms > threshold_ms:
                    logger.warning(f"⚠️  {func_name} exceeded threshold: {elapsed_ms:.2f}ms > {threshold_ms:.2f}ms")
                else:
                    logger.debug(f"⏱️  {func_name}: {elapsed_ms:.2f}ms")
        
        return wrapper
    return decorator


class PipelineTimer:
    """Accumulator for tracking pipeline stage timings."""
    
    def __init__(self, name: str) -> None:
        self.name = name
        self.stages: dict[str, list[float]] = {}
    
    def record(self, stage_name: str, elapsed_ms: float) -> None:
        """Record timing for a pipeline stage."""
        if stage_name not in self.stages:
            self.stages[stage_name] = []
        self.stages[stage_name].append(elapsed_ms)
    
    def report(self) -> str:
        """Generate performance report for all stages."""
        lines = [f"\n{'='*70}"]
        lines.append(f"Pipeline Report: {self.name}")
        lines.append(f"{'='*70}")
        
        total_ms = 0.0
        for stage_name in sorted(self.stages.keys()):
            times = self.stages[stage_name]
            count = len(times)
            total_ms_stage = sum(times)
            avg_ms = total_ms_stage / count if count > 0 else 0
            min_ms = min(times)
            max_ms = max(times)
            
            total_ms += total_ms_stage
            lines.append(
                f"  {stage_name:.<40} "
                f"avg={avg_ms:7.2f}ms  min={min_ms:7.2f}ms  max={max_ms:7.2f}ms  (n={count})"
            )
        
        lines.append(f"  {'TOTAL':.<40} {total_ms:7.2f}ms")
        lines.append(f"{'='*70}\n")
        
        return "\n".join(lines)
    
    def log_report(self, level: int = logging.INFO) -> None:
        """Log the performance report."""
        logger.log(level, self.report())


# ─────────────────────────────────────────────────────────────────────────────
# TYPE ALIASES (semantic naming for complex types)
# ─────────────────────────────────────────────────────────────────────────────

T = TypeVar("T")
"""Generic type variable."""

PixelCoordinate = tuple[int, int]
"""2D pixel coordinate as (x, y) integers."""

NormalizedCoordinate = tuple[float, float]
"""Normalized coordinate in [0, 1] range."""

FrameArray = Any  # numpy.ndarray shape (H, W, 3) for BGR frames
"""OpenCV video frame (BGR color, H×W×3 array)."""

KeypointArray = Any  # numpy.ndarray shape (N, 2)
"""Array of 2D keypoint positions (N×2)."""

ConfidenceArray = Any  # numpy.ndarray shape (N,)
"""Array of detection confidences (N,)."""
