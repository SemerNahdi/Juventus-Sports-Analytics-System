from .cv_wrapper import cv2
import math
import numpy as np
from collections import deque
from typing import Optional, List, Tuple
from .models import PoseKeypoints, JOINT_NAMES, _COCO, BONE_DEFS, _L, _R, _W
from .math_utils import midpoint, dist2d, smooth_arr, clamp01, lerp_color

# Import configuration
try:
    from ..api.config import POSE_VERTICAL_PROPS, POSE_BODY_WIDTH, POSE_MOTION, KALMAN_FILTER, VISUALIZATION
except ImportError:
    # Fallback defaults if config not available (for standalone testing)
    from dataclasses import dataclass
    @dataclass
    class _FallbackConfig:
        head = 0.04
        neck = 0.11
        shoulder = 0.20
        elbow = 0.34
        wrist = 0.46
        hip = 0.54
        knee = 0.73
        ankle = 0.91
        foot = 0.99
    POSE_VERTICAL_PROPS = _FallbackConfig()

# ══════════════════════════════════════════════════════════════════════════════
#  HYBRID POSE ESTIMATOR
# ══════════════════════════════════════════════════════════════════════════════

class HybridPoseEstimator:
    def __init__(self):
        self._prev_cx = None
        self._dh = deque(maxlen=8)

    def estimate(self, frame, bbox, ts, spd=0., yolo_kp=None) -> PoseKeypoints:
        x, y, w, h = bbox
        cx = x + w / 2.
        disp = abs(cx - self._prev_cx) if self._prev_cx is not None else 0.
        self._prev_cx = cx
        self._dh.append(disp)
        ds = sum(self._dh)
        phase = (ds / max(w * POSE_MOTION.displacement_phase_scale, 4.)) * math.pi
        swing = clamp01(spd / POSE_MOTION.speed_swing_threshold)
        arm_sw = swing * POSE_MOTION.arm_swing_factor * w
        leg_sw = swing * POSE_MOTION.leg_swing_factor * w
        k_lift = swing * POSE_MOTION.knee_lift_factor * h
        cw = self._cwidths(frame, bbox)
        sh, hh = self._bwidths(cw, w, h)

        def vy(f): return y + f * h

        kp = PoseKeypoints()
        kp.head = (cx, vy(POSE_VERTICAL_PROPS.head))
        ls = (cx - sh, vy(POSE_VERTICAL_PROPS.shoulder))
        rs = (cx + sh, vy(POSE_VERTICAL_PROPS.shoulder))
        kp.left_shoulder  = ls
        kp.right_shoulder = rs
        aoff = arm_sw * math.sin(phase)
        le = (ls[0] - aoff, vy(POSE_VERTICAL_PROPS.elbow))
        re = (rs[0] + aoff, vy(POSE_VERTICAL_PROPS.elbow))
        kp.left_elbow  = le
        kp.right_elbow = re
        kp.left_wrist  = (le[0] - aoff * .55, vy(POSE_VERTICAL_PROPS.wrist))
        kp.right_wrist = (re[0] + aoff * .55, vy(POSE_VERTICAL_PROPS.wrist))
        lh = (cx - hh, vy(POSE_VERTICAL_PROPS.hip))
        rh = (cx + hh, vy(POSE_VERTICAL_PROPS.hip))
        kp.left_hip   = lh
        kp.right_hip  = rh
        loff = leg_sw * math.sin(phase)
        roff = -loff
        ll = k_lift * max(0., math.sin(phase))
        rl = k_lift * max(0., -math.sin(phase))
        kp.left_knee   = (lh[0] + loff, vy(POSE_VERTICAL_PROPS.knee) - ll)
        kp.right_knee  = (rh[0] + roff, vy(POSE_VERTICAL_PROPS.knee) - rl)
        kp.left_ankle  = (lh[0] + loff * POSE_MOTION.leg_offset_factor, vy(POSE_VERTICAL_PROPS.ankle) - ll * POSE_MOTION.knee_lift_attenuation)
        kp.right_ankle = (rh[0] + roff * POSE_MOTION.leg_offset_factor, vy(POSE_VERTICAL_PROPS.ankle) - rl * POSE_MOTION.knee_lift_attenuation)
        kp.left_foot   = (kp.left_ankle[0] + w * .07,  vy(POSE_VERTICAL_PROPS.foot))
        kp.right_foot  = (kp.right_ankle[0] + w * .07, vy(POSE_VERTICAL_PROPS.foot))

        yolo_confident = False
        if yolo_kp is not None and len(yolo_kp) == 17:
            try:
                valid = int(np.sum((yolo_kp[:, 0] > 1) & (yolo_kp[:, 1] > 1)))
                yolo_confident = valid >= 8
            except Exception:
                yolo_confident = False
            def g(nm):
                i = _COCO.get(nm)
                if i is None:
                    return None
                pt = yolo_kp[i]
                return (float(pt[0]), float(pt[1])) if (pt[0] > 1 or pt[1] > 1) else None

            def gxy(nm, df):
                p = g(nm)
                return p if p is not None else df

            kp.left_shoulder  = gxy("left_shoulder",  kp.left_shoulder)
            kp.right_shoulder = gxy("right_shoulder", kp.right_shoulder)
            kp.left_elbow     = gxy("left_elbow",     kp.left_elbow)
            kp.right_elbow    = gxy("right_elbow",    kp.right_elbow)
            kp.left_wrist     = gxy("left_wrist",     kp.left_wrist)
            kp.right_wrist    = gxy("right_wrist",    kp.right_wrist)
            kp.left_hip       = gxy("left_hip",       kp.left_hip)
            kp.right_hip      = gxy("right_hip",      kp.right_hip)
            kp.left_knee      = gxy("left_knee",      kp.left_knee)
            kp.right_knee     = gxy("right_knee",     kp.right_knee)
            kp.left_ankle     = gxy("left_ankle",     kp.left_ankle)
            kp.right_ankle    = gxy("right_ankle",    kp.right_ankle)
            nose = g("nose")
            if nose:
                kp.head = nose
            kp.shoulder_center = midpoint(kp.left_shoulder, kp.right_shoulder)
            kp.hip_center = midpoint(kp.left_hip, kp.right_hip)
            kp.neck = midpoint(kp.shoulder_center, kp.head)
            for side in ("left", "right"):
                ank = getattr(kp, f"{side}_ankle")
                object.__setattr__(kp, f"{side}_foot", (ank[0] + w * .04, ank[1] + h * .03))
        if kp.shoulder_center == (0., 0.):
            kp.shoulder_center = midpoint(kp.left_shoulder, kp.right_shoulder)
        if kp.hip_center == (0., 0.):
            kp.hip_center = midpoint(kp.left_hip, kp.right_hip)
        if kp.neck == (0., 0.):
            kp.neck = midpoint(kp.shoulder_center, kp.head)

        object.__setattr__(kp, "_yolo_confident", bool(yolo_confident))
        return kp

    def _cwidths(self, frame, bbox):
        bx, by, bw, bh = bbox
        H, W = frame.shape[:2]
        bx2, by2 = min(bx + bw, W), min(by + bh, H)
        bx, by = max(0, bx), max(0, by)
        if bx2 - bx < POSE_BODY_WIDTH.min_crop_width or by2 - by < POSE_BODY_WIDTH.min_crop_width:
            return None
        crop = frame[by:by2, bx:bx2]
        _, mask = cv2.threshold(
            cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        ws = np.array([np.sum(mask[r] > 0) for r in range(mask.shape[0])], dtype=float)
        return smooth_arr(ws, w=max(3, bh // 20)) if len(ws) > 5 else None

    def _bwidths(self, cw, bw, bh):
        dsh = bw * POSE_BODY_WIDTH.shoulder_factor
        dh  = bw * POSE_BODY_WIDTH.hip_factor
        if cw is None or len(cw) < 10:
            return dsh, dh
        n = len(cw)
        u = cw[int(n * POSE_BODY_WIDTH.shoulder_sample_start):int(n * POSE_BODY_WIDTH.shoulder_sample_end)]
        l = cw[int(n * POSE_BODY_WIDTH.hip_sample_start):int(n * POSE_BODY_WIDTH.hip_sample_end)]
        sh = float(np.max(u)) / 2. if len(u) else dsh
        hh = float(np.max(l)) / 2. if len(l) else dh
        return (float(np.clip(sh, bw * POSE_BODY_WIDTH.shoulder_min_ratio, bw * POSE_BODY_WIDTH.shoulder_max_ratio)), 
                float(np.clip(hh, bw * POSE_BODY_WIDTH.hip_min_ratio, bw * POSE_BODY_WIDTH.hip_max_ratio)))


# ══════════════════════════════════════════════════════════════════════════════
#  KALMAN JOINT SMOOTHER
# ══════════════════════════════════════════════════════════════════════════════

class JointKalman:
    def __init__(self, pn=None, on=None):
        if pn is None:
            pn = KALMAN_FILTER.process_noise_pos
        if on is None:
            on = KALMAN_FILTER.observation_noise
        self.x = None
        self.v = 0.
        self.P = np.array([[KALMAN_FILTER.initial_position_variance, 0.], [0., KALMAN_FILTER.initial_position_variance]])
        self.Q = np.diag([pn, pn * KALMAN_FILTER.process_noise_vel_factor])
        self.R = on
        self.F = np.array([[1., 1.], [0., 1.]])
        self.H = np.array([[1., 0.]])

    def update(self, z):
        # Validate input: reject NaN/Inf to prevent state corruption
        if np.isnan(z) or np.isinf(z):
            # Return previous state if available, otherwise return input
            return self.x if self.x is not None else z
        
        if self.x is None:
            self.x = float(z)  # Ensure scalar float, not NaN
            return self.x
        
        st = self.F @ np.array([self.x, self.v])
        Pp = self.F @ self.P @ self.F.T + self.Q
        y  = z - (self.H @ st)[0]
        S  = (self.H @ Pp @ self.H.T)[0, 0] + self.R
        K  = Pp @ self.H.T / S
        st = st + (K * y).flatten()
        self.P = (np.eye(2) - np.outer(K.flatten(), self.H)) @ Pp
        self.x, self.v = float(st[0]), float(st[1])
        return self.x


class PoseKalmanSmoother:
    def __init__(self):
        self._k: dict = {}

    def smooth(self, kp) -> PoseKeypoints:
        out = PoseKeypoints()
        for nm in JOINT_NAMES:
            raw = getattr(kp, nm)
            if nm not in self._k:
                self._k[nm] = (JointKalman(), JointKalman())
            kx, ky = self._k[nm]
            setattr(out, nm, (kx.update(raw[0]), ky.update(raw[1])))
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  SKELETON RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def draw_gradient_bone(img, p1, p2, c1, c2, th, rt=0.):
    s = max(8, int(dist2d(p1, p2) / 4))
    for i in range(s):
        t  = i / max(s - 1, 1)
        t2 = (i + 1) / max(s - 1, 1)
        col = lerp_color(lerp_color(c1, c2, t), VISUALIZATION.color_risk, rt * VISUALIZATION.risk_tint_factor)
        cv2.line(img,
                 (int(p1[0] + t  * (p2[0] - p1[0])), int(p1[1] + t  * (p2[1] - p1[1]))),
                 (int(p1[0] + t2 * (p2[0] - p1[0])), int(p1[1] + t2 * (p2[1] - p1[1]))),
                 col, th, cv2.LINE_AA)


def draw_glow_joint(img, pt, r, col, ga=None):
    if ga is None:
        ga = VISUALIZATION.glow_opacity / 0.45  # Normalize back to original scale
    px, py = int(pt[0]), int(pt[1])
    for rr in range(r + VISUALIZATION.glow_radius_offset, r, -2):  
        cv2.circle(img, (px, py), rr, col, -1, cv2.LINE_AA)


def render_skeleton(frame, kp, risk_tint=0.):
    kpd = {n: getattr(kp, n) for n in JOINT_NAMES}
    for a, b, c1, c2, th in BONE_DEFS:
        if a in kpd and b in kpd:
            draw_gradient_bone(frame, kpd[a], kpd[b], c1, c2, th, risk_tint)
    glow = np.zeros_like(frame)
    sz = {
        "head": VISUALIZATION.joint_size_head,
        "neck": VISUALIZATION.joint_size_neck,
        "left_shoulder": VISUALIZATION.joint_size_shoulder,
        "right_shoulder": VISUALIZATION.joint_size_shoulder,
        "left_elbow": VISUALIZATION.joint_size_elbow,
        "right_elbow": VISUALIZATION.joint_size_elbow,
        "left_wrist": VISUALIZATION.joint_size_wrist,
        "right_wrist": VISUALIZATION.joint_size_wrist,
        "left_hip": VISUALIZATION.joint_size_hip,
        "right_hip": VISUALIZATION.joint_size_hip,
        "left_knee": VISUALIZATION.joint_size_knee,
        "right_knee": VISUALIZATION.joint_size_knee,
        "left_ankle": VISUALIZATION.joint_size_ankle,
        "right_ankle": VISUALIZATION.joint_size_ankle,
        "left_foot": VISUALIZATION.joint_size_foot,
        "right_foot": VISUALIZATION.joint_size_foot,
    }
    for nm, r in sz.items():
        if nm in kpd:
            col = lerp_color(
                VISUALIZATION.color_left if "left" in nm else VISUALIZATION.color_right if "right" in nm else VISUALIZATION.color_center,
                VISUALIZATION.color_risk, risk_tint * VISUALIZATION.risk_joint_tint,
            )
            draw_glow_joint(glow, kpd[nm], r, col)
            px, py = int(kpd[nm][0]), int(kpd[nm][1])
            cv2.circle(frame, (px, py), r,          VISUALIZATION.color_joint_outline, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), max(1, r-2), col,            -1, cv2.LINE_AA)
    cv2.addWeighted(frame, 1.0, glow, VISUALIZATION.glow_opacity, 0, frame)
