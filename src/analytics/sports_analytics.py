"""
Sports Analytics Package - Modular Biomechanical Analysis System
Production-grade SDK facade.
"""

__version__ = "2.0.0"
API_VERSION = 2


# ─────────────────────────────────────────────
# Lazy loader (prevents circular imports)
# ─────────────────────────────────────────────
def _lazy(module, name):
    import importlib
    return getattr(importlib.import_module(f".{module}", package=__package__), name)


# ─────────────────────────────────────────────
# Public API (lazy-loaded)
# ─────────────────────────────────────────────
SportsAnalyzer = _lazy("analysis_engine", "SportsAnalyzer")
AnalyticsPlotter = _lazy("visualization", "AnalyticsPlotter")
Sports2DRunner = _lazy("sports2d_runner", "Sports2DRunner")
ProtocolHandler = _lazy("analysis_engine", "ProtocolHandler")


# ─────────────────────────────────────────────
# Models (safe direct export)
# ─────────────────────────────────────────────
from .models import (
    PoseKeypoints,
    PoseFrame,
    FrameMetrics,
    PlayerSummary,
    BioFrame,
    JOINT_NAMES,
)

# ─────────────────────────────────────────────
# Flags
# ─────────────────────────────────────────────
from .core import HAS_SPORTS2D, HAS_SCIPY, HAS_YOLO, HAS_MPL


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────
from .player_picker import pick_player_interactive, select_primary_player
from .reporting import generate_report

from .types import BenchmarkResult, PerformanceTimer, RiskLevel


# ─────────────────────────────────────────────
# Public API surface
# ─────────────────────────────────────────────
__all__ = [
    "SportsAnalyzer",
    "AnalyticsPlotter",
    "Sports2DRunner",
    "ProtocolHandler",

    "PoseKeypoints",
    "PoseFrame",
    "FrameMetrics",
    "PlayerSummary",
    "BioFrame",
    "JOINT_NAMES",

    "HAS_SPORTS2D",
    "HAS_SCIPY",
    "HAS_YOLO",
    "HAS_MPL",

    "pick_player_interactive",
    "select_primary_player",
    "generate_report",

    "BenchmarkResult",
    "PerformanceTimer",
    "RiskLevel",
]


def __dir__():
    return sorted(set(__all__) | set(globals().keys()))


def __getattr__(name):
    deprecated = {
        "SportsAnalyst": "SportsAnalyzer",
        "Analyzer": "SportsAnalyzer",
    }

    import warnings

    if name in deprecated:
        warnings.warn(
            f"{name} is deprecated. Use {deprecated[name]} instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return globals()[deprecated[name]]

    raise AttributeError(
        f"'{__name__}' has no attribute '{name}'. API v{__version__}"
    )