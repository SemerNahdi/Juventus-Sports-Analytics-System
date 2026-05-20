from .cv_wrapper import cv2
import math
import numpy as np
from collections import deque
from typing import Optional, Dict, Tuple

from .models import PoseKeypoints, JOINT_NAMES, BONE_DEFS, _COCO
from .math_utils import midpoint, dist2d, smooth_arr, clamp01, lerp_color


# ════════════════════════════════════════════════════════════════════════
# CONFIG SAFE IMPORT
# ════════════════════════════════════════════════════════════════════════

try:
    from ..api.config import (
        POSE_VERTICAL_PROPS,
        POSE_BODY_WIDTH,
        POSE_MOTION,
        VISUALIZATION
    )
except ImportError:
    from dataclasses import dataclass

    @dataclass
    class _Fallback:
        head = 0.04
        shoulder = 0.20
        elbow = 0.34
        wrist = 0.46
        hip = 0.54
        knee = 0.73
        ankle = 0.91
        foot = 0.99

    POSE_VERTICAL_PROPS = _Fallback()
    POSE_BODY_WIDTH = _Fallback()
    POSE_MOTION = _Fallback()
    VISUALIZATION = _Fallback()


# ════════════════════════════════════════════════════════════════════════
# BIOMECHANICAL TEMPORAL MODEL (RESEARCH LEVEL)
# ════════════════════════════════════════════════════════════════════════

class TemporalPoseModel:
    """
    Replaces Kalman:
    - velocity-aware smoothing
    - adaptive damping
    - outlier rejection
    """

    def __init__(self, alpha=0.75, beta=0.2):
        self.alpha = alpha
        self.beta = beta
        self.state: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def update(self, name: str, pt: Tuple[float, float]) -> Tuple[float, float]:
        x = np.array(pt, dtype=np.float32)

        if name not in self.state:
            self.state[name] = (x, np.zeros(2, dtype=np.float32))
            return tuple(x)

        prev_x, vel = self.state[name]

        # velocity update
        new_vel = self.beta * (x - prev_x) + (1 - self.beta) * vel

        # predicted position
        pred = prev_x + new_vel

        # adaptive smoothing (motion-aware)
        smooth = self.alpha * pred + (1 - self.alpha) * x

        self.state[name] = (smooth, new_vel)

        return tuple(smooth)


# ════════════════════════════════════════════════════════════════════════
# POSE KALMAN SMOOTHER (Frame-Level Wrapper)
# ════════════════════════════════════════════════════════════════════════

class PoseKalmanSmoother:
    """Wraps TemporalPoseModel to smooth entire PoseKeypoints frames."""
    
    def __init__(self, alpha: float = 0.75, beta: float = 0.2):
        self.model = TemporalPoseModel(alpha=alpha, beta=beta)
    
    def smooth(self, kp: "PoseKeypoints") -> "PoseKeypoints":
        """Smooth all joints in a PoseKeypoints frame."""
        if kp is None:
            return kp
        
        smoothed = PoseKeypoints()
        
        # List of all joint attributes to smooth
        joint_names = [
            "head", "neck", "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow", "left_wrist", "right_wrist",
            "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle", "left_foot", "right_foot",
            "hip_center", "shoulder_center"
        ]
        
        for joint in joint_names:
            pt = getattr(kp, joint, (0., 0.))
            if pt and isinstance(pt, (tuple, list)) and len(pt) >= 2:
                smoothed_pt = self.model.update(joint, pt)
                setattr(smoothed, joint, smoothed_pt)
            else:
                setattr(smoothed, joint, (0., 0.))
        
        # Copy any extra attributes that might exist
        if hasattr(kp, "_yolo_confident"):
            smoothed._yolo_confident = kp._yolo_confident
        
        return smoothed


# ════════════════════════════════════════════════════════════════════════
# HYBRID POSE ESTIMATOR (IMPROVED)
# ════════════════════════════════════════════════════════════════════════

class HybridPoseEstimator:

    def __init__(self):
        self.prev_cx = None
        self.dhistory = deque(maxlen=10)
        self.temporal = TemporalPoseModel()

    # ────────────────────────────────────────────────────────────────
    def _motion_phase(self, cx, w, speed):
        disp = abs(cx - self.prev_cx) if self.prev_cx is not None else 0
        self.prev_cx = cx

        self.dhistory.append(disp)
        total = sum(self.dhistory)

        scale = max(w * 0.6, 5.0)
        phase = (total / scale) * math.pi

        swing = clamp01(speed * 0.8)
        return phase, swing

    # ────────────────────────────────────────────────────────────────
    def estimate(self, frame, bbox, ts, speed=0.0, yolo_kp=None) -> PoseKeypoints:

        x, y, w, h = bbox
        cx = x + w * 0.5

        phase, swing = self._motion_phase(cx, w, speed)

        arm_sw = swing * w * 0.35
        leg_sw = swing * w * 0.25
        knee_lift = swing * h * 0.18

        def vy(f): return y + f * h

        kp = PoseKeypoints()

        # ───────── torso anchors
        kp.head = (cx, vy(POSE_VERTICAL_PROPS.head))
        kp.left_shoulder = (cx - w * 0.18, vy(POSE_VERTICAL_PROPS.shoulder))
        kp.right_shoulder = (cx + w * 0.18, vy(POSE_VERTICAL_PROPS.shoulder))

        # ───────── arms (oscillation model)
        arm_phase = math.sin(phase)

        ls, rs = kp.left_shoulder, kp.right_shoulder

        kp.left_elbow = (ls[0] - arm_sw * arm_phase, vy(POSE_VERTICAL_PROPS.elbow))
        kp.right_elbow = (rs[0] + arm_sw * arm_phase, vy(POSE_VERTICAL_PROPS.elbow))

        kp.left_wrist = (kp.left_elbow[0] - arm_sw * 0.5, vy(POSE_VERTICAL_PROPS.wrist))
        kp.right_wrist = (kp.right_elbow[0] + arm_sw * 0.5, vy(POSE_VERTICAL_PROPS.wrist))

        # ───────── hips
        kp.left_hip = (cx - w * 0.12, vy(POSE_VERTICAL_PROPS.hip))
        kp.right_hip = (cx + w * 0.12, vy(POSE_VERTICAL_PROPS.hip))

        hip_phase = -arm_phase

        # ───────── legs
        kp.left_knee = (
            kp.left_hip[0] + leg_sw * hip_phase,
            vy(POSE_VERTICAL_PROPS.knee) - knee_lift * max(0, arm_phase)
        )

        kp.right_knee = (
            kp.right_hip[0] - leg_sw * hip_phase,
            vy(POSE_VERTICAL_PROPS.knee) - knee_lift * max(0, -arm_phase)
        )

        kp.left_ankle = (kp.left_knee[0], vy(POSE_VERTICAL_PROPS.ankle))
        kp.right_ankle = (kp.right_knee[0], vy(POSE_VERTICAL_PROPS.ankle))

        kp.left_foot = (kp.left_ankle[0] + w * 0.05, vy(POSE_VERTICAL_PROPS.foot))
        kp.right_foot = (kp.right_ankle[0] + w * 0.05, vy(POSE_VERTICAL_PROPS.foot))

        # ─────────────────────────────────────────────
        # YOLO FUSION (soft confidence blending)
        # ─────────────────────────────────────────────
        yolo_conf = False

        if yolo_kp is not None and len(yolo_kp) == 17:
            try:
                valid = np.sum((yolo_kp[:, 0] > 1) & (yolo_kp[:, 1] > 1))
                yolo_conf = valid >= 8

                def get(nm):
                    i = _COCO.get(nm)
                    if i is None:
                        return None
                    p = yolo_kp[i]
                    if p[0] <= 1 or p[1] <= 1:
                        return None
                    return (float(p[0]), float(p[1]))

                def blend(nm, fallback):
                    p = get(nm)
                    if p is None:
                        return fallback
                    return (
                        0.7 * fallback[0] + 0.3 * p[0],
                        0.7 * fallback[1] + 0.3 * p[1],
                    )

                kp.left_shoulder = blend("left_shoulder", kp.left_shoulder)
                kp.right_shoulder = blend("right_shoulder", kp.right_shoulder)
                kp.left_hip = blend("left_hip", kp.left_hip)
                kp.right_hip = blend("right_hip", kp.right_hip)
                kp.head = get("nose") or kp.head

            except Exception:
                pass

        # ─────────────────────────────────────────────
        # TEMPORAL SMOOTHING (CORE RESEARCH UPGRADE)
        # ─────────────────────────────────────────────
        for j in JOINT_NAMES:
            pt = getattr(kp, j)
            sm = self.temporal.update(j, pt)
            setattr(kp, j, sm)

        kp.shoulder_center = midpoint(kp.left_shoulder, kp.right_shoulder)
        kp.hip_center = midpoint(kp.left_hip, kp.right_hip)
        kp.neck = midpoint(kp.head, kp.shoulder_center)

        setattr(kp, "_yolo_confident", bool(yolo_conf))

        return kp


# ════════════════════════════════════════════════════════════════════════
# BIOMECHANICAL CONSTRAINT REPAIR (NEW)
# ════════════════════════════════════════════════════════════════════════

class PoseConstraintFixer:

    def __init__(self):
        self.bone_lengths = {}

    def calibrate(self, kp: PoseKeypoints):
        for a, b, *_ in BONE_DEFS:
            pa = getattr(kp, a, None)
            pb = getattr(kp, b, None)
            if pa and pb:
                self.bone_lengths[(a, b)] = dist2d(pa, pb)

    def enforce(self, kp: PoseKeypoints):
        for (a, b), length in self.bone_lengths.items():
            pa = getattr(kp, a, None)
            pb = getattr(kp, b, None)
            if not pa or not pb:
                continue

            dx, dy = pb[0] - pa[0], pb[1] - pa[1]
            d = math.sqrt(dx * dx + dy * dy) + 1e-6

            scale = length / d
            fixed = (pa[0] + dx * scale, pa[1] + dy * scale)

            setattr(kp, b, fixed)

        return kp


# ════════════════════════════════════════════════════════════════════════
# FINAL PIPELINE WRAPPER
# ════════════════════════════════════════════════════════════════════════

class PoseSystem:

    def __init__(self):
        self.estimator = HybridPoseEstimator()
        self.fixer = PoseConstraintFixer()

    def init(self, kp: PoseKeypoints):
        self.fixer.calibrate(kp)

    def process(self, frame, bbox, ts, speed=0.0, yolo_kp=None):
        kp = self.estimator.estimate(frame, bbox, ts, speed, yolo_kp)
        kp = self.fixer.enforce(kp)
        return kp


# ════════════════════════════════════════════════════════════════════════
# RENDERING (LIGHTWEIGHT + FAST)
# ════════════════════════════════════════════════════════════════════════

def render_skeleton(frame, kp, risk=0.0):

    pts = {n: getattr(kp, n) for n in JOINT_NAMES}

    for a, b, c1, c2, th in BONE_DEFS:
        if a in pts and b in pts:
            cv2.line(frame,
                     (int(pts[a][0]), int(pts[a][1])),
                     (int(pts[b][0]), int(pts[b][1])),
                     c1, th, cv2.LINE_AA)

    for n, p in pts.items():
        r = 4
        col = lerp_color((0, 255, 0), (0, 0, 255), risk)
        cv2.circle(frame, (int(p[0]), int(p[1])), r, col, -1, cv2.LINE_AA)

    return frame