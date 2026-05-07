import cv2
import math
import numpy as np
from collections import deque
from typing import Optional, List, Tuple
from .models import PoseKeypoints, JOINT_NAMES, _COCO, BONE_DEFS, _L, _R, _W
from .math_utils import midpoint, dist2d, smooth_arr, clamp01, lerp_color

# ══════════════════════════════════════════════════════════════════════════════
#  HYBRID POSE ESTIMATOR
# ══════════════════════════════════════════════════════════════════════════════

class HybridPoseEstimator:
    _VP = dict(head=0.04, neck=0.11, shoulder=0.20, elbow=0.34, wrist=0.46,
               hip=0.54, knee=0.73, ankle=0.91, foot=0.99)

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
        phase = (ds / max(w * 0.18, 4.)) * math.pi
        swing = clamp01(spd / 9.)
        arm_sw = swing * 0.10 * w
        leg_sw = swing * 0.08 * w
        k_lift = swing * 0.08 * h
        cw = self._cwidths(frame, bbox)
        sh, hh = self._bwidths(cw, w, h)

        def vy(f): return y + f * h

        kp = PoseKeypoints()
        kp.head = (cx, vy(self._VP["head"]))
        ls = (cx - sh, vy(self._VP["shoulder"]))
        rs = (cx + sh, vy(self._VP["shoulder"]))
        kp.left_shoulder  = ls
        kp.right_shoulder = rs
        aoff = arm_sw * math.sin(phase)
        le = (ls[0] - aoff, vy(self._VP["elbow"]))
        re = (rs[0] + aoff, vy(self._VP["elbow"]))
        kp.left_elbow  = le
        kp.right_elbow = re
        kp.left_wrist  = (le[0] - aoff * .55, vy(self._VP["wrist"]))
        kp.right_wrist = (re[0] + aoff * .55, vy(self._VP["wrist"]))
        lh = (cx - hh, vy(self._VP["hip"]))
        rh = (cx + hh, vy(self._VP["hip"]))
        kp.left_hip   = lh
        kp.right_hip  = rh
        loff = leg_sw * math.sin(phase)
        roff = -loff
        ll = k_lift * max(0., math.sin(phase))
        rl = k_lift * max(0., -math.sin(phase))
        kp.left_knee   = (lh[0] + loff, vy(self._VP["knee"]) - ll)
        kp.right_knee  = (rh[0] + roff, vy(self._VP["knee"]) - rl)
        kp.left_ankle  = (lh[0] + loff * .45, vy(self._VP["ankle"]) - ll * .5)
        kp.right_ankle = (rh[0] + roff * .45, vy(self._VP["ankle"]) - rl * .5)
        kp.left_foot   = (kp.left_ankle[0] + w * .07,  vy(self._VP["foot"]))
        kp.right_foot  = (kp.right_ankle[0] + w * .07, vy(self._VP["foot"]))

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
        if bx2 - bx < 5 or by2 - by < 5:
            return None
        crop = frame[by:by2, bx:bx2]
        _, mask = cv2.threshold(
            cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        ws = np.array([np.sum(mask[r] > 0) for r in range(mask.shape[0])], dtype=float)
        return smooth_arr(ws, w=max(3, bh // 20)) if len(ws) > 5 else None

    def _bwidths(self, cw, bw, bh):
        dsh = bw * .29
        dh  = bw * .17
        if cw is None or len(cw) < 10:
            return dsh, dh
        n = len(cw)
        u = cw[int(n * .15):int(n * .40)]
        l = cw[int(n * .48):int(n * .68)]
        sh = float(np.max(u)) / 2. if len(u) else dsh
        hh = float(np.max(l)) / 2. if len(l) else dh
        return float(np.clip(sh, bw * .18, bw * .42)), float(np.clip(hh, bw * .10, bw * .32))


# ══════════════════════════════════════════════════════════════════════════════
#  KALMAN JOINT SMOOTHER
# ══════════════════════════════════════════════════════════════════════════════

class JointKalman:
    def __init__(self, pn=1.5, on=8.0):
        self.x = None
        self.v = 0.
        self.P = np.array([[100., 0.], [0., 100.]])
        self.Q = np.diag([pn, pn * 2])
        self.R = on
        self.F = np.array([[1., 1.], [0., 1.]])
        self.H = np.array([[1., 0.]])

    def update(self, z):
        if self.x is None:
            self.x = z
            return z
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
        col = lerp_color(lerp_color(c1, c2, t), (0, 0, 220), rt * .6)
        cv2.line(img,
                 (int(p1[0] + t  * (p2[0] - p1[0])), int(p1[1] + t  * (p2[1] - p1[1]))),
                 (int(p1[0] + t2 * (p2[0] - p1[0])), int(p1[1] + t2 * (p2[1] - p1[1]))),
                 col, th, cv2.LINE_AA)


def draw_glow_joint(img, pt, r, col, ga=0.45):
    px, py = int(pt[0]), int(pt[1])
    for rr in range(r + 6, r, -2):  
        cv2.circle(img, (px, py), rr, col, -1, cv2.LINE_AA)


def render_skeleton(frame, kp, risk_tint=0.):
    kpd = {n: getattr(kp, n) for n in JOINT_NAMES}
    for a, b, c1, c2, th in BONE_DEFS:
        if a in kpd and b in kpd:
            draw_gradient_bone(frame, kpd[a], kpd[b], c1, c2, th, risk_tint)
    glow = np.zeros_like(frame)
    sz = {
        "head": 4, "neck": 3,
        "left_shoulder": 4, "right_shoulder": 4,
        "left_elbow": 3,    "right_elbow": 3,
        "left_wrist": 3,    "right_wrist": 3,
        "left_hip": 5,      "right_hip": 5,
        "left_knee": 6,     "right_knee": 6,
        "left_ankle": 5,    "right_ankle": 5,
        "left_foot": 3,     "right_foot": 3,
    }
    for nm, r in sz.items():
        if nm in kpd:
            col = lerp_color(
                _L if "left" in nm else _R if "right" in nm else _W,
                (0, 0, 220), risk_tint * .5,
            )
            draw_glow_joint(glow, kpd[nm], r, col)
            px, py = int(kpd[nm][0]), int(kpd[nm][1])
            cv2.circle(frame, (px, py), r,          (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), max(1, r-2), col,            -1, cv2.LINE_AA)
    cv2.addWeighted(frame, 1.0, glow, 0.45 * 0.5, 0, frame)
