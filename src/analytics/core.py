"""
Sports Analytics System - Core Shim
======================================
This file now serves as a compatibility shim, re-exporting symbols 
from the modularized analytics components.
"""

import cv2
import math
import json
import os
import threading
import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple

from .models import *
from .math_utils import *
from .tracking import *
from .pose import *
from .biomechanics import *
from .player_picker import *
from .scoring import *
from .rendering import *
from .reporting import *

# Re-exporting flags specifically for clarity and compatibility
# Some of these are already in the * imports above
__all__ = [
    # Types & Libraries
    "cv2", "math", "json", "os", "threading", "np", "pd", "deque",
    "dataclass", "asdict", "Optional", "List", "Tuple",
    
    # Models & Dataclasses
    "PoseKeypoints", "PoseFrame", "FrameMetrics", "PlayerSummary", "BioFrame",
    "JOINT_NAMES", "BIO_ANGLE_FIELDS", "BONE_DEFS",
    
    # Math & Utils
    "midpoint", "angle_3pts", "dist2d", "smooth_arr", "clamp01", 
    "lerp_color", "risk_color", "bbox_iou", "bbox_centre", 
    "crop_hist", "hist_sim", "estimate_player_orientation",
    "s2d_joint_angle", "s2d_seg_angle", "clean_nans",
    "HAS_SCIPY", "HAS_MPL", "HAS_SPORTS2D", "SPORTS2D_PROCESS",
    
    # Tracking & Detection
    "KalmanTrack", "ByteTracker", "TargetLock", "SceneChangeDetector",
    "DetectionLayer", "get_detection_layer", "HAS_YOLO",
    
    # Pose & Rendering
    "HybridPoseEstimator", "JointKalman", "PoseKalmanSmoother",
    "draw_gradient_bone", "draw_glow_joint", "render_skeleton",
    
    # Biomechanics
    "BiomechanicsEngine",
    
    # Player Selection
    "pick_player_interactive", "select_primary_player",

    # Scoring & Risk
    "RiskScorer",

    # Rendering & Reporting
    "annotate_frame", "draw_player_aura", "generate_report",

    # MAT (Movement Assessment Tool)
    "MATEventKPIs", "MATSummary", "MATEventDetector", "MATGridCalibrator",
]