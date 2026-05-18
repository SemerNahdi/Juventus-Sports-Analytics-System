"""
Sports Analytics System - Core Shim
======================================
This file now serves as a compatibility shim, re-exporting symbols 
from the modularized analytics components.
"""

import math
import json
import os
import threading
import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple

from .cv_wrapper import cv2


from .models import *
from .math_utils import *
from .tracking import *
from .pose import *
from .biomechanics import *
from .player_picker import *
from .scoring import *
from .rendering import *
from .reporting import *
# from .mesh_generator import get_smpl_model

try:
    from .mesh_generator import get_smpl_model
except ImportError:
    def get_smpl_model(): return None

try:
    import fairmotion
    HAS_FAIRMOTION = True
except ImportError:
    HAS_FAIRMOTION = False

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
    "HAS_SCIPY", "HAS_MPL", "HAS_SPORTS2D", "SPORTS2D_PROCESS", "HAS_FAIRMOTION",
    
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
    "preload_all_models"
]


def preload_all_models():
    """
    Utility to load all heavy AI models into memory at startup.
    Avoids 3-4 minute delay during the first analysis request.
    """
    print("\n" + "═" * 50)
    print(" MITUS AI: PRE-LOADING NEURAL NETWORKS")
    print("-" * 50)
    
    # 1. YOLO Models
    try:
        from .tracking import preload_yolo_models
        # Preload the default sizes used in the system
        preload_yolo_models(sizes=["n", "m"])
    except Exception as e:
        print(f" [!] YOLO Preload Failed: {e}")

    # 2. SMPL Mesh Model
    try:
        print(" [SMPL] Pre-loading 3D Mesh model...")
        get_smpl_model()
    except Exception as e:
        print(f" [!] SMPL Preload Failed: {e}")

    # 3. Sports2D Warm-up
    if HAS_SPORTS2D:
        try:
            print(" [S2D] Warming up Sports2D pipeline...")
            # Just importing it already happened, but we can check if it's responsive
            if SPORTS2D_PROCESS:
                print(" [S2D] Sports2D API resolved.")
        except Exception as e:
            print(f" [!] S2D Warmup Failed: {e}")
        
    print("-" * 50)
    print(" ALL MODELS WARM AND READY")
    print("═" * 50 + "\n")