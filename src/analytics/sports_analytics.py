"""
Backward-compatible facade for the modular analytics package.

Legacy imports such as:
    from src.analytics.sports_analytics import SportsAnalyzer, AnalyticsPlotter, HAS_SPORTS2D
continue to work after refactoring into smaller modules.
"""

from .core import *  # noqa: F401,F403
from .sports2d_runner import Sports2DRunner
from .output_manager import OpenSimFileWriter
from .visualization import AnalyticsPlotter
from .analysis_engine import SportsAnalyzer

__all__ = [
    # Primary public API
    "SportsAnalyzer",
    "AnalyticsPlotter",
    "Sports2DRunner",
    "OpenSimFileWriter",
    # Compatibility flags from core
    "HAS_SPORTS2D",
    "HAS_SCIPY",
    "HAS_MPL",
    "HAS_YOLO",
]
