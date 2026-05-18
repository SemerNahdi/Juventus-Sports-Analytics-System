"""
Backward-compatible facade for the modular analytics package.

Legacy imports such as:
    from src.analytics.sports_analytics import SportsAnalyzer, AnalyticsPlotter, HAS_SPORTS2D
continue to work after refactoring into smaller modules.
"""

# Re-export core symbols
from .core import (
    # Types & Libraries
    cv2, math, json, os, threading, np, pd, deque,
    dataclass, asdict, Optional, List, Tuple,
    # Models & Dataclasses
    PoseKeypoints, PoseFrame, FrameMetrics, PlayerSummary, BioFrame,
    JOINT_NAMES, BIO_ANGLE_FIELDS, BONE_DEFS,
    # Math & Utils
    midpoint, angle_3pts, dist2d, smooth_arr, clamp01, 
    lerp_color, risk_color, bbox_iou, bbox_centre, 
    crop_hist, hist_sim, estimate_player_orientation,
    s2d_joint_angle, s2d_seg_angle, clean_nans,
    HAS_SCIPY, HAS_MPL, HAS_SPORTS2D, SPORTS2D_PROCESS, HAS_FAIRMOTION,
    # Tracking & Detection
    KalmanTrack, ByteTracker, TargetLock, SceneChangeDetector,
    DetectionLayer, get_detection_layer, HAS_YOLO,
    # Pose & Rendering
    HybridPoseEstimator, JointKalman, PoseKalmanSmoother,
    draw_gradient_bone, draw_glow_joint, render_skeleton,
    # Biomechanics
    BiomechanicsEngine, MATEventKPIs, MATSummary, MATEventDetector, MATGridCalibrator,
    # Player Selection
    pick_player_interactive, select_primary_player,
    # Scoring & Risk
    RiskScorer,
    # Rendering & Reporting
    annotate_frame, draw_player_aura, generate_report,
    # System Utilities
    preload_all_models,
)

# Re-export type system and performance monitoring
from .types import (
    # Enums for magic strings
    YoloModelSize,
    VideoCodec,
    AnalysisMode,
    BiomechanicsBackend,
    RiskLevel,
    ExportFormat,
    # Protocols (structural typing)
    PoseFrame as PoseFrameProtocol,
    AnalyzerLike,
    ExportWriter,
    # Performance benchmarking
    BenchmarkResult,
    PerformanceTimer,
    benchmark_method,
    PipelineTimer,
    # Type aliases
    PixelCoordinate,
    NormalizedCoordinate,
    KeypointArray,
    ConfidenceArray,
)

from .sports2d_runner import Sports2DRunner
from .output_manager import OpenSimFileWriter
from .visualization import AnalyticsPlotter
from .analysis_engine import SportsAnalyzer, ProtocolHandler

__all__ = [
    # Primary public API
    "SportsAnalyzer",
    "AnalyticsPlotter",
    "Sports2DRunner",
    "OpenSimFileWriter",
    "ProtocolHandler",
    # Compatibility flags from core
    "HAS_SPORTS2D",
    "HAS_SCIPY",
    "HAS_MPL",
    "HAS_YOLO",
]
