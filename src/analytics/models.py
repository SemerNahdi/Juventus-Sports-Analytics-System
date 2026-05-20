import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Tuple

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

JOINT_NAMES = [
    "head", "neck", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_foot", "right_foot",
    "hip_center", "shoulder_center",
]

# COCO keypoint indices for YOLO pose model
_COCO = {
    "nose": 0, "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8, "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12, "left_knee": 13,
    "right_knee": 14, "left_ankle": 15, "right_ankle": 16,
}

BIO_ANGLE_FIELDS = [
    "left_knee_flexion",    "right_knee_flexion",
    "left_hip_flexion",     "right_hip_flexion",
    "left_ankle_dorsiflexion", "right_ankle_dorsiflexion",
    "left_elbow_flexion",   "right_elbow_flexion",
    "trunk_lateral_lean",   "trunk_sagittal_lean",
    "pelvis_obliquity",     "pelvis_rotation",
    "left_thigh_angle",     "right_thigh_angle",
    "left_shank_angle",     "right_shank_angle",
    "trunk_segment_angle",
    "left_valgus_clinical", "right_valgus_clinical",
    "left_arm_swing",       "right_arm_swing",
]

# Rendering Colors
_W  = (240, 240, 240)
_L  = (255, 200, 0)
_R  = (0, 140, 255)
_S  = (180, 240, 180)

BONE_DEFS = [
    ("head",            "neck",            _W, _W, 4),
    ("neck",            "shoulder_center", _W, _S, 4),
    ("shoulder_center", "hip_center",      _S, _S, 5),
    ("left_shoulder",   "left_elbow",      _L, _L, 5),
    ("left_elbow",      "left_wrist",      _L, _L, 4),
    ("right_shoulder",  "right_elbow",     _R, _R, 5),
    ("right_elbow",     "right_wrist",     _R, _R, 4),
    ("left_shoulder",   "right_shoulder",  _L, _R, 4),
    ("left_hip",        "right_hip",       _L, _R, 5),
    ("left_hip",        "left_knee",       _L, _L, 7),
    ("left_knee",       "left_ankle",      _L, _L, 6),
    ("left_ankle",      "left_foot",       _L, _L, 4),
    ("right_hip",       "right_knee",      _R, _R, 7),
    ("right_knee",      "right_ankle",     _R, _R, 6),
    ("right_ankle",     "right_foot",      _R, _R, 4),
]

# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PoseKeypoints:
    head:            Tuple[float, float] = (0., 0.)
    neck:            Tuple[float, float] = (0., 0.)
    left_shoulder:   Tuple[float, float] = (0., 0.)
    right_shoulder:  Tuple[float, float] = (0., 0.)
    left_elbow:      Tuple[float, float] = (0., 0.)
    right_elbow:     Tuple[float, float] = (0., 0.)
    left_wrist:      Tuple[float, float] = (0., 0.)
    right_wrist:     Tuple[float, float] = (0., 0.)
    left_hip:        Tuple[float, float] = (0., 0.)
    right_hip:       Tuple[float, float] = (0., 0.)
    left_knee:       Tuple[float, float] = (0., 0.)
    right_knee:      Tuple[float, float] = (0., 0.)
    left_ankle:      Tuple[float, float] = (0., 0.)
    right_ankle:     Tuple[float, float] = (0., 0.)
    left_foot:       Tuple[float, float] = (0., 0.)
    right_foot:      Tuple[float, float] = (0., 0.)
    hip_center:      Tuple[float, float] = (0., 0.)
    shoulder_center: Tuple[float, float] = (0., 0.)


@dataclass
class PoseFrame:
    frame_idx: int
    timestamp: float
    bbox: Tuple[int, int, int, int]
    kp: PoseKeypoints


@dataclass
class FrameMetrics:
    frame_idx: int = 0
    timestamp: float = 0.
    speed: float = 0.
    acceleration: float = 0.
    stride_length: float = 0.
    step_time: float = 0.
    cadence: float = 0.
    flight_time: float = 0.
    left_knee_angle: float = 0.
    right_knee_angle: float = 0.
    left_hip_angle: float = 0.
    right_hip_angle: float = 0.
    trunk_lateral_lean: float = 0.
    trunk_sagittal_lean: float = 0.
    direction_change: bool = False
    energy_expenditure: float = 0.
    gait_symmetry: float = 100.
    stride_variability: float = 0.
    fall_risk: float = 0.
    injury_risk: float = 0.
    joint_stress: float = 0.
    fatigue_index: float = 0.
    body_center_disp: float = 0.
    l_valgus: float = 0.
    r_valgus: float = 0.
    risk_score: float = 0.
    l_valgus_clinical: float = 0.
    r_valgus_clinical: float = 0.
    perspective_confidence: float = 1.0


@dataclass
class PlayerSummary:
    player_id: int = 1
    total_frames: int = 0
    duration_seconds: float = 0.
    avg_speed: float = 0.
    max_speed: float = 0.
    avg_stride_length: float = 0.
    avg_step_time: float = 0.
    avg_cadence: float = 0.
    avg_flight_time: float = 0.
    direction_change_freq: float = 0.
    estimated_energy_kcal_hr: float = 0.
    gait_symmetry_pct: float = 0.
    stride_variability_pct: float = 0.
    total_distance_m: float = 0.
    peak_risk_score: float = 0.
    fall_risk_label: str = "Low"
    injury_risk_label: str = "Low"
    injury_risk_detail: str = ""
    body_stress_label: str = "Low"
    fatigue_label: str = "Low"
    double_support_pct: float = 0.
    avg_pelvic_rotation: float = 0.


@dataclass
class BioFrame:
    frame_idx: int = 0
    timestamp: float = 0.
    left_knee_flexion: float = 0.
    right_knee_flexion: float = 0.
    left_hip_flexion: float = 0.
    right_hip_flexion: float = 0.
    left_ankle_dorsiflexion: float = 0.
    right_ankle_dorsiflexion: float = 0.
    left_elbow_flexion: float = 0.
    right_elbow_flexion: float = 0.
    trunk_lateral_lean: float = 0.
    trunk_sagittal_lean: float = 0.
    pelvis_obliquity: float = 0.
    pelvis_rotation: float = 0.
    left_thigh_angle: float = 0.
    right_thigh_angle: float = 0.
    left_shank_angle: float = 0.
    right_shank_angle: float = 0.
    trunk_segment_angle: float = 0.
    left_valgus_clinical: float = 0.
    right_valgus_clinical: float = 0.
    left_arm_swing: float = 0.
    right_arm_swing: float = 0.
    arm_swing_asymmetry: float = 0.
    left_knee_ang_vel: float = 0.
    right_knee_ang_vel: float = 0.
    left_hip_ang_vel: float = 0.
    right_hip_ang_vel: float = 0.
    left_heel_strike: bool = False
    right_heel_strike: bool = False
    left_toe_off: bool = False
    right_toe_off: bool = False
    stance_left: bool = False
    stance_right: bool = False
    double_support: bool = False
    step_width: float = 0.
@dataclass
class MATEventKPIs:
    """Key Performance Indicators for one MAT event."""
    event_type: str  # "single_hop_left", "drop_vertical_jump", "sebt_reach_right"
    flight_time: float = 0.
    landing_valgus_left: float = 0.
    landing_valgus_right: float = 0.
    peak_knee_flexion_landing: float = 0.
    time_to_stabilization: float = 0.
    hop_distance_m: float = 0.
    balance_score: float = 0.
    takeoff_idx: int = 0
    landing_idx: int = 0
    stabilized_idx: int = 0


@dataclass
class MATSummary:
    """Session summary for MAT protocol testing."""
    protocol_id: str
    participant_id: int
    limb_symmetry_index: float
    events: List[MATEventKPIs] = field(default_factory=list)
    average_landing_valgus: float = 0.
    worst_landing_valgus: float = 0.
    stability_trend: str = "stable"
