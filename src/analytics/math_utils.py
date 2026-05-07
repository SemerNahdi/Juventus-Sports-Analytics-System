import cv2
import math
import numpy as np
from typing import Optional, List, Tuple
from .models import PoseKeypoints

try:
    from scipy.signal import find_peaks, butter, filtfilt
    from scipy.ndimage import uniform_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Optional Sports2D / Pose2Sim ───────────────────────────────────────────────
HAS_SPORTS2D = False
_s2d_angle   = None
_s2d_seg     = None
SPORTS2D_PROCESS = None

try:
    try:
        from Sports2D import Sports2D as _Sports2DModule
        SPORTS2D_PROCESS = getattr(_Sports2DModule, "process", None)
    except Exception:
        _Sports2DModule = None

    if SPORTS2D_PROCESS is None:
        try:
            import sports2d as _sports2d_mod
            SPORTS2D_PROCESS = getattr(_sports2d_mod, "process", None)
        except Exception:
            _sports2d_mod = None

    HAS_SPORTS2D = SPORTS2D_PROCESS is not None
    try:
        from Pose2Sim.common import points_to_angles as _pta
        def _s2d_angle(p1, p2, p3):
            v = _pta([p1, p2, p3])
            return [float(v)] if isinstance(v, (float, int, np.number)) else list(v)
        def _s2d_seg(p_from, p_to):
            v = _pta([p_from, p_to])
            return [float(v)] if isinstance(v, (float, int, np.number)) else list(v)
    except Exception:
        pass
except ImportError:
    pass

def s2d_joint_angle(p_prox, p_vertex, p_dist) -> float:
    if _s2d_angle is not None:
        try:
            pp  = np.array(p_prox,   dtype=float).reshape(1, 2)
            pv  = np.array(p_vertex, dtype=float).reshape(1, 2)
            pd_ = np.array(p_dist,   dtype=float).reshape(1, 2)
            return float(_s2d_angle(pp, pv, pd_)[0])
        except Exception:
            pass
    return angle_3pts(p_prox, p_vertex, p_dist)

def s2d_seg_angle(p_from, p_to) -> float:
    if _s2d_seg is not None:
        try:
            pf = np.array(p_from, dtype=float).reshape(1, 2)
            pt = np.array(p_to,   dtype=float).reshape(1, 2)
            return float(_s2d_seg(pf, pt)[0])
        except Exception:
            pass
    dx = p_to[0] - p_from[0]
    dy = p_to[1] - p_from[1]
    return float(math.degrees(math.atan2(dx, abs(dy) + 1e-9)))

# ══════════════════════════════════════════════════════════════════════════════
#  MATH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def midpoint(a, b) -> Tuple[float, float]:
    return ((a[0] + b[0]) / 2., (a[1] + b[1]) / 2.)

def angle_3pts(a, b, c) -> float:
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    n = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / n, -1, 1))))

def dist2d(p1, p2) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def smooth_arr(arr, w=5) -> np.ndarray:
    a = np.array(arr, dtype=float)
    if HAS_SCIPY:
        return uniform_filter1d(a, size=w)
    pad = w // 2
    a_pad = np.pad(a, pad, mode='edge')
    return np.convolve(a_pad, np.ones(w) / w, mode='valid')[:len(a)]

def clamp01(x) -> float:
    return float(np.clip(x, 0., 1.))

def lerp_color(c1, c2, t):
    t = clamp01(t)
    return tuple(int(c1[i] * (1 - t) + c2[i] * t) for i in range(3))

def risk_color(s):
    t = clamp01(s / 100.)
    if t < 0.5:
        return lerp_color((0, 200, 0), (0, 200, 255), t * 2)
    return lerp_color((0, 200, 255), (0, 0, 230), (t - .5) * 2)

def _size_sim(a_w, a_h, b_w, b_h) -> float:
    a_area = a_w * a_h
    b_area = b_w * b_h
    return min(a_area, b_area) / (max(a_area, b_area) + 1e-6)

def bbox_iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter + 1e-6)

def bbox_centre(bbox):
    x, y, w, h = bbox
    return (x + w / 2., y + h / 2.)

def crop_hist(frame, bbox):
    bx, by, bw, bh = [int(v) for v in bbox]
    H, W = frame.shape[:2]
    bx, by = max(0, bx), max(0, by)
    bw, bh = min(bw, W - bx), min(bh, H - by)
    if bw < 5 or bh < 5:
        return None
    hsv = cv2.cvtColor(frame[by:by + bh, bx:bx + bw], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist

def hist_sim(h1, h2) -> float:
    if h1 is None or h2 is None:
        return 0.
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))

def estimate_player_orientation(kp: PoseKeypoints) -> float:
    sh_w = abs(kp.left_shoulder[0] - kp.right_shoulder[0])
    hp_w = abs(kp.left_hip[0] - kp.right_hip[0])
    body_h = abs(kp.head[1] - kp.left_ankle[1]) + 1e-6
    expected_sh = 0.22 * body_h
    expected_hp = 0.18 * body_h
    conf_sh = clamp01(sh_w / (expected_sh + 1e-6))
    conf_hp = clamp01(hp_w / (expected_hp + 1e-6))
    return float((conf_sh + conf_hp) / 2.0)

def clean_nans(obj):
    """Recursively convert NaN/Inf into standard JSON null."""
    if isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean_nans(v) for v in obj]
    elif isinstance(obj, (float, np.floating)):
        fv = float(obj)
        return None if math.isnan(fv) or math.isinf(fv) else fv
    return obj
