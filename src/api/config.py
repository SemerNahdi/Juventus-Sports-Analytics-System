import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Tuple, Dict

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://your-url.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "your-service-role-key")
BUCKET_NAME = os.getenv("SUPABASE_BUCKET", "Sports Analytics")

# Cloudinary configuration
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

# Analysis defaults
DEFAULT_YOLO_SIZE = os.getenv("YOLO_SIZE_DEFAULT", "n")
DEFAULT_STRIDE = int(os.getenv("ANALYSIS_STRIDE", "2"))
DEFAULT_TARGET_HEIGHT = int(os.getenv("ANALYSIS_TARGET_HEIGHT", "640"))

# SMTP configuration
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").replace(" ", "").strip()

# Application
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
PORT = int(os.environ.get("PORT", 8000))

# Upload settings
SUPABASE_UPLOAD_RETRIES = int(os.getenv("SUPABASE_UPLOAD_RETRIES", "3"))
SUPABASE_UPLOAD_RETRY_DELAY = float(os.getenv("SUPABASE_UPLOAD_RETRY_DELAY", "0.5"))
SUPABASE_UPLOAD_WORKERS = int(os.getenv("SUPABASE_UPLOAD_WORKERS", "1"))
    
# Static files
STATIC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
    "dashboard"
)

# ══════════════════════════════════════════════════════════════════════════════
#  POSE ESTIMATION CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PoseVerticalProportions:
    """Vertical position proportions for procedural pose skeleton (relative to bbox height)."""
    head: float = 0.04
    neck: float = 0.11
    shoulder: float = 0.20
    elbow: float = 0.34
    wrist: float = 0.46
    hip: float = 0.54
    knee: float = 0.73
    ankle: float = 0.91
    foot: float = 0.99


@dataclass
class PoseBodyWidthConfig:
    """Body width calculation factors for procedural skeleton."""
    shoulder_factor: float = 0.29        # dsh = bw * 0.29
    hip_factor: float = 0.17             # dh = bw * 0.17
    shoulder_min_ratio: float = 0.18     # min shoulder width as ratio of bw
    shoulder_max_ratio: float = 0.42     # max shoulder width as ratio of bw
    hip_min_ratio: float = 0.10          # min hip width as ratio of bw
    hip_max_ratio: float = 0.32          # max hip width as ratio of bw
    shoulder_sample_start: float = 0.15  # start of shoulder sample (% of height)
    shoulder_sample_end: float = 0.40    # end of shoulder sample (% of height)
    hip_sample_start: float = 0.48       # start of hip sample (% of height)
    hip_sample_end: float = 0.68         # end of hip sample (% of height)
    min_crop_width: int = 5              # minimum crop width for analysis


@dataclass
class PoseMotionConfig:
    """Motion amplitude and timing parameters for procedural skeleton."""
    arm_swing_factor: float = 0.10       # arm_sw = swing * 0.10 * w
    leg_swing_factor: float = 0.08       # leg_sw = swing * 0.08 * w
    knee_lift_factor: float = 0.08       # k_lift = swing * 0.08 * h
    leg_offset_factor: float = 0.45      # ankle offset from knee = leg_sw * 0.45
    knee_lift_attenuation: float = 0.5   # ankle lift attenuation = k_lift * 0.5
    speed_swing_threshold: float = 9.0   # speed threshold for max swing amplitude
    displacement_phase_scale: float = 0.18 * 4.0  # phase = (ds / (w * 0.18)) * pi


@dataclass
class KalmanFilterConfig:
    """Kalman filter parameters for per-joint smoothing."""
    process_noise_pos: float = 1.5       # pn: position process noise
    process_noise_vel_factor: float = 2.0  # velocity process noise = pn * factor
    observation_noise: float = 8.0       # on: observation noise
    initial_position_variance: float = 100.0
    initial_velocity_variance: float = 100.0


@dataclass
class VisualizationConfig:
    """Rendering parameters for skeleton visualization."""
    # Joint colors (BGR)
    color_left: Tuple[int, int, int] = (100, 200, 255)      # Cyan for left
    color_right: Tuple[int, int, int] = (255, 100, 100)     # Blue for right
    color_center: Tuple[int, int, int] = (200, 200, 200)    # White for center
    color_risk: Tuple[int, int, int] = (0, 0, 220)          # Red for risk
    color_joint_outline: Tuple[int, int, int] = (255, 255, 255)  # White outline
    
    # Joint sizes (radius in pixels)
    joint_size_head: int = 4
    joint_size_neck: int = 3
    joint_size_shoulder: int = 4
    joint_size_elbow: int = 3
    joint_size_wrist: int = 3
    joint_size_hip: int = 5
    joint_size_knee: int = 6
    joint_size_ankle: int = 5
    joint_size_foot: int = 3
    
    # Bone rendering
    bone_thickness: int = 2
    glow_radius_offset: int = 6
    glow_opacity: float = 0.225          # 0.45 * 0.5
    risk_tint_factor: float = 0.6        # risk_tint * 0.6
    risk_joint_tint: float = 0.5         # risk_tint * 0.5


@dataclass
class BioMechanicsConfig:
    """Biomechanics and scaling parameters."""
    pix_to_m_default: float = 0.002      # Default pixel-to-meter conversion
    pix_to_m_leg_length: float = 0.90    # Reference leg length in meters
    pix_to_m_leg_length_px: int = 45     # Reference leg length in pixels (~0.90/0.002)


@dataclass
class AnalysisConfig:
    """Top-level analysis algorithm parameters."""
    min_pose_frames_mat: int = 5         # Minimum frames for MAT event detection
    min_flight_time: float = 0.05        # Minimum flight phase duration (seconds)
    expected_flight_time: float = 0.6    # Expected flight time for scoring (seconds)
    expected_knee_flexion: float = 120.0  # Expected peak knee flexion (degrees)


# Singleton instances with environment variable overrides
POSE_VERTICAL_PROPS = PoseVerticalProportions(
    head=float(os.getenv("POSE_VP_HEAD", "0.04")),
    neck=float(os.getenv("POSE_VP_NECK", "0.11")),
    shoulder=float(os.getenv("POSE_VP_SHOULDER", "0.20")),
)

POSE_BODY_WIDTH = PoseBodyWidthConfig(
    shoulder_factor=float(os.getenv("POSE_SHOULDER_FACTOR", "0.29")),
    hip_factor=float(os.getenv("POSE_HIP_FACTOR", "0.17")),
)

POSE_MOTION = PoseMotionConfig(
    arm_swing_factor=float(os.getenv("POSE_ARM_SWING", "0.10")),
    leg_swing_factor=float(os.getenv("POSE_LEG_SWING", "0.08")),
)

KALMAN_FILTER = KalmanFilterConfig(
    process_noise_pos=float(os.getenv("KALMAN_PROC_NOISE_POS", "1.5")),
    observation_noise=float(os.getenv("KALMAN_OBS_NOISE", "8.0")),
)

VISUALIZATION = VisualizationConfig()

BIOMECHANICS = BioMechanicsConfig(
    pix_to_m_default=float(os.getenv("PIX_TO_M_DEFAULT", "0.002")),
)

ANALYSIS = AnalysisConfig(
    min_pose_frames_mat=int(os.getenv("MIN_POSE_FRAMES_MAT", "5")),
    min_flight_time=float(os.getenv("MIN_FLIGHT_TIME", "0.05")),
)

@classmethod
def fix_ssl_cert(cls):
    """Fix SSL_CERT_FILE issue if it's set to non-existent path"""
    if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
        os.environ.pop("SSL_CERT_FILE", None)