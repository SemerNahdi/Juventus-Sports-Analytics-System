"""
Sports Analytics System  v6
======================================
Refactored for data integrity, clean video output, and OpenSim compatibility.

Install:
    pip install ultralytics opencv-python numpy pandas scipy matplotlib
    pip install sports2d pose2sim          # for Sports2D pipeline
    # For OpenSim IK: conda install -c opensim-org opensim
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

try:
    from scipy.signal import find_peaks, butter, filtfilt
    from scipy.ndimage import uniform_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib
    # NOTE: Do NOT call matplotlib.use("Agg") here globally.
    # Sports2D needs an interactive backend to show its native graphs.
    # We set the backend lazily only when our own plotter saves files.
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── YOLO pose detection ────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO as _YOLO
    HAS_YOLO = True
except (ImportError, Exception):
    HAS_YOLO = False

# ── Optional Sports2D / Pose2Sim ───────────────────────────────────────────────
HAS_SPORTS2D = False
_s2d_angle   = None
_s2d_seg     = None
SPORTS2D_PROCESS = None

try:
    # Sports2D's Python API import path varies by distribution/version.
    # We support multiple import styles and normalize to a single callable.
    try:
        from Sports2D import Sports2D as _Sports2DModule  # common in some installs
        SPORTS2D_PROCESS = getattr(_Sports2DModule, "process", None)
    except Exception:
        _Sports2DModule = None

    if SPORTS2D_PROCESS is None:
        try:
            # Alternative: some versions expose a top-level module API.
            import sports2d as _sports2d_mod  # type: ignore
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
#  DATA STRUCTURES
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
    trunk_lean: float = 0.
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
    foot_progression_angle: float = 0.


# ══════════════════════════════════════════════════════════════════════════════
#  MATH HELPERS
# ══════════════════════════════════════════════════════════════════════════════

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
    return np.convolve(a, np.ones(w) / w, mode='same')


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
    # Slightly higher binning improves discriminative power for kit colors
    # without making the histogram too sparse/noisy.
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


# ══════════════════════════════════════════════════════════════════════════════
#  KALMAN TRACK
# ══════════════════════════════════════════════════════════════════════════════

class KalmanTrack:
    _next_id = 1
    F = np.array([
        [1,0,0,0,1,0,0,0], [0,1,0,0,0,1,0,0], [0,0,1,0,0,0,1,0], [0,0,0,1,0,0,0,1],
        [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1],
    ], dtype=float)
    H = np.eye(4, 8)

    def __init__(self, bbox, frame, conf=1.0):
        self.id = KalmanTrack._next_id
        KalmanTrack._next_id += 1
        cx, cy = bbox_centre(bbox)
        w, h = bbox[2], bbox[3]
        self.x = np.array([cx, cy, w, h, 0., 0., 0., 0.], dtype=float)
        self.P = np.diag([10., 10., 10., 10., 100., 100., 10., 10.])
        self.Q = np.diag([1., 1., 1., 1., .5, .5, .2, .2])
        self.R = np.diag([4., 4., 10., 10.])
        self.conf = conf
        self.hit_streak = 1
        self.missed = 0
        self.age = 1
        self.ref_hist = crop_hist(frame, bbox)
        self.last_bbox = bbox
        self.trajectory = deque(maxlen=30)
        self.trajectory.append(bbox_centre(bbox))
        self._yolo_kp = None

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.x[2] = max(1., self.x[2])
        self.x[3] = max(1., self.x[3])
        self.age += 1
        self.missed += 1
        return self.get_bbox()

    def update(self, bbox, frame, conf=1.0):
        cx, cy = bbox_centre(bbox)
        z = np.array([cx, cy, bbox[2], bbox[3]], dtype=float)
        S = self.H @ self.P @ self.H.T + self.R
        K = np.linalg.solve(S.T, (self.P @ self.H.T).T).T  # numerically stable vs inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P
        self.conf = conf
        self.hit_streak += 1
        self.missed = 0
        self.last_bbox = bbox
        self.trajectory.append(bbox_centre(bbox))
        nh = crop_hist(frame, bbox)
        if nh is not None and self.ref_hist is not None:
            self.ref_hist = (0.92 * self.ref_hist + 0.08 * nh).astype(np.float32)
            cv2.normalize(self.ref_hist, self.ref_hist)
        elif nh is not None:
            self.ref_hist = nh

    def get_bbox(self) -> Tuple[int, int, int, int]:
        cx, cy, w, h = self.x[:4]
        return (int(cx - w / 2), int(cy - h / 2), int(w), int(h))

    def reactivate(self, bbox, frame):
        cx, cy = bbox_centre(bbox)
        self.x[:4] = [cx, cy, bbox[2], bbox[3]]
        # Reset covariance on reactivation to avoid over-trusting stale state
        # after long occlusions (reduces jitter/lag on reacquire).
        self.P = np.diag([10., 10., 10., 10., 100., 100., 10., 10.])
        self.x[4:] = 0.0
        self.missed = 0
        self.hit_streak = 1
        self.last_bbox = bbox
        nh = crop_hist(frame, bbox)
        if nh is not None:
            self.ref_hist = nh


# ══════════════════════════════════════════════════════════════════════════════
#  DETECTION LAYER
# ══════════════════════════════════════════════════════════════════════════════

class DetectionLayer:
    def __init__(self, model_size="m", shared_yolo=None):
        self._yolo = shared_yolo
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=400, varThreshold=40, detectShadows=False
        )
        self._mode = "yolo" if self._yolo is not None else "blob"
        if self._yolo is None and HAS_YOLO:
            try:
                self._yolo = _YOLO(_resolve_yolo_model_path(model_size))
                self._mode = "yolo"
            except Exception:
                pass

    def reset_bg(self):
        """Reset background model (useful on scene cuts)."""
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=400, varThreshold=40, detectShadows=False
        )

    @property
    def mode(self):
        return self._mode

    def detect(self, frame) -> List[dict]:
        return self._yolo_detect(frame) if self._mode == "yolo" else self._blob_detect(frame)

    def _yolo_detect(self, frame) -> List[dict]:
        res = self._yolo(frame, verbose=False, conf=0.25)[0]
        dets = []
        if res.boxes is None or len(res.boxes) == 0:
            return dets
        for i, box in enumerate(res.boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            bw, bh = x2 - x1, y2 - y1
            if bh < bw * 0.8:
                continue
            bbox = (int(x1), int(y1), int(bw), int(bh))
            conf = float(box.conf[0].cpu())
            kp = None
            if res.keypoints is not None and i < len(res.keypoints.xy):
                kpxy = res.keypoints.xy[i].cpu().numpy()
                kpc = res.keypoints.conf[i].cpu().numpy()
                kpxy[kpc < 0.3] = 0.
                kp = kpxy
            dets.append({'bbox': bbox, 'conf': conf, 'kp': kp})
        return dets

    def _blob_detect(self, frame) -> List[dict]:
        mask = self._bg.apply(frame)
        k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cands = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if not (2000 <= area <= 90000):
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if not (1.3 <= bh / (bw + 1e-6) <= 5.0):
                continue
            fill = area / (bw * bh + 1e-6)
            if fill < 0.25:
                continue
            cands.append({'bbox': (bx, by, bw, bh), 'conf': fill, 'kp': None, 'area': area})
        cands.sort(key=lambda c: c['area'], reverse=True)
        kept, sup = [], set()
        for i, ci in enumerate(cands):
            if i in sup:
                continue
            kept.append(ci)
            for j, cj in enumerate(cands):
                if j <= i or j in sup:
                    continue
                if bbox_iou(ci['bbox'], cj['bbox']) > 0.40:
                    sup.add(j)
        return kept


# ══════════════════════════════════════════════════════════════════════════════
#  SCENE CHANGE DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

class SceneChangeDetector:
    def __init__(self, threshold=0.45):
        self._prev = None
        self._thr = threshold

    def is_cut(self, frame) -> bool:
        h, w = frame.shape[:2]
        # Use a centre-crop (80% of frame) to reduce sensitivity to edge banners / overlays
        cy, cx = h // 2, w // 2
        ch, cw = int(h * 0.4), int(w * 0.4)
        crop = frame[cy - ch:cy + ch, cx - cw:cx + cw]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)
        if self._prev is None:
            self._prev = hist
            return False
        score = float(cv2.compareHist(self._prev, hist, cv2.HISTCMP_CORREL))
        self._prev = hist
        return score < self._thr


# ══════════════════════════════════════════════════════════════════════════════
#  BYTETRACKER
# ══════════════════════════════════════════════════════════════════════════════

class ByteTracker:
    HIGH_THRESH = 0.50
    LOW_THRESH  = 0.20
    IOU_HIGH    = 0.30
    IOU_LOW     = 0.15
    IOU_LOST    = 0.20
    MIN_HITS    = 2
    LOST_TTL    = 60

    def __init__(self):
        self.active_tracks: List[KalmanTrack] = []
        self.lost_tracks:   List[KalmanTrack] = []

    def update(self, detections: List[dict], frame) -> List[KalmanTrack]:
        for t in self.active_tracks + self.lost_tracks:
            t.predict()
        high = [d for d in detections if d['conf'] >= self.HIGH_THRESH]
        low  = [d for d in detections if self.LOW_THRESH <= d['conf'] < self.HIGH_THRESH]

        unm_t, unm_h = self._associate(self.active_tracks, high, frame, self.IOU_HIGH)
        still_unm, _ = self._associate(unm_t, low, frame, self.IOU_LOW)
        self._associate(self.lost_tracks, low, frame, self.IOU_LOST, reactivate=True)

        for t in still_unm:
            t.hit_streak = 0
            if t not in self.lost_tracks:
                self.lost_tracks.append(t)
            if t in self.active_tracks:
                self.active_tracks.remove(t)

        for d in unm_h:
            self.active_tracks.append(KalmanTrack(d['bbox'], frame, d['conf']))

        self.lost_tracks = [t for t in self.lost_tracks if t.missed <= self.LOST_TTL]
        return [t for t in self.active_tracks if t.hit_streak >= self.MIN_HITS]

    def _associate(self, tracks, dets, frame, iou_thr, reactivate=False):
        if not tracks or not dets:
            return list(tracks), list(dets)
        
        # We work with a static copy for matching to avoid IndexError if the original 
        # list (e.g. self.lost_tracks) is modified during loop iteration (reactivation).
        tracks_copy = list(tracks)
        cost = np.zeros((len(tracks_copy), len(dets)), dtype=float)
        for ti, t in enumerate(tracks_copy):
            tb = t.get_bbox()
            th = t.ref_hist
            for di, d in enumerate(dets):
                iou = bbox_iou(tb, d['bbox'])
                hs  = hist_sim(th, crop_hist(frame, d['bbox']))
                cost[ti, di] = 1.0 - (iou * 0.60 + hs * 0.40)

        # Use Hungarian matching when scipy is available
        mt, md = set(), set()
        if HAS_SCIPY:
            try:
                from scipy.optimize import linear_sum_assignment
                row_ind, col_ind = linear_sum_assignment(cost)
                for ti, di in zip(row_ind, col_ind):
                    if cost[ti, di] < 1.0 - iou_thr:
                        mt.add(ti)
                        md.add(di)
                        t = tracks_copy[ti]
                        d = dets[di]
                        if reactivate:
                            t.reactivate(d['bbox'], frame)
                            if t in self.lost_tracks:
                                self.lost_tracks.remove(t)
                            if t not in self.active_tracks:
                                self.active_tracks.append(t)
                        else:
                            t.update(d['bbox'], frame, d['conf'])
                        if d.get('kp') is not None:
                            t._yolo_kp = d['kp']
            except ImportError:
                pass

        if not mt:  # fallback greedy
            while True:
                avail = [(ti, di) for ti in range(len(tracks_copy)) for di in range(len(dets))
                         if ti not in mt and di not in md]
                if not avail:
                    break
                ti, di = min(avail, key=lambda p: cost[p[0], p[1]])
                if cost[ti, di] >= 1.0 - iou_thr:
                    break
                mt.add(ti)
                md.add(di)
                t = tracks_copy[ti]
                d = dets[di]
                if reactivate:
                    t.reactivate(d['bbox'], frame)
                    if t in self.lost_tracks:
                        self.lost_tracks.remove(t)
                    if t not in self.active_tracks:
                        self.active_tracks.append(t)
                else:
                    t.update(d['bbox'], frame, d['conf'])
                if d.get('kp') is not None:
                    t._yolo_kp = d['kp']

        return (
            [tracks[i] for i in range(len(tracks)) if i not in mt],
            [dets[i]   for i in range(len(dets))   if i not in md],
        )

    def reset(self):
        for t in self.active_tracks + self.lost_tracks:
            t.x[4:] = 0.


# ══════════════════════════════════════════════════════════════════════════════
#  TARGET LOCK
# ══════════════════════════════════════════════════════════════════════════════

class TargetLock:
    def __init__(self, seed_bbox, seed_hist, seed_frame_idx, yolo_size="m"):
        self._seed_bbox    = seed_bbox
        self._ref_hist     = seed_hist
        self._seed_fi      = seed_frame_idx
        self._target_id    = None
        self._last_bbox    = None
        self._smooth_box   = None
        self._alpha        = 0.35
        self._state        = "searching"
        self._lost_frames  = 0
        self._fi           = 0
        self.bt    = ByteTracker()
        self.scene = SceneChangeDetector()
        self._det_layer = get_detection_layer(yolo_size)

    @property
    def state(self):
        return self._state

    @property
    def lost_count(self):
        return self._lost_frames

    def update(self, frame) -> Optional[Tuple]:
        if self.scene.is_cut(frame) and self._fi > 10:
            self.bt.reset()
            self._target_id = None
            self._state = "searching"
            # Scene cut: reset blob background model to avoid drift.
            if self._det_layer is not None:
                try:
                    self._det_layer.reset_bg()
                except Exception:
                    pass

        dets   = self._det_layer.detect(frame)
        tracks = self.bt.update(dets, frame)
        self._fi += 1

        if self._target_id is None:
            if self._fi >= self._seed_fi:
                self._target_id = self._choose(tracks, frame)
                if self._target_id is not None:
                    self._state = "tracking"
            return None

        target = next((t for t in tracks if t.id == self._target_id), None)
        if target is not None:
            target = self._resolve_overlap(target, tracks)

        if target is None:
            self._lost_frames += 1
            self._state = "lost"
            target = self._reacquire(tracks, strict=self._lost_frames <= 5)
            if target is None:
                target = self._reacquire(self.bt.lost_tracks, strict=False)
        else:
            self._lost_frames = 0
            self._state = "tracking"

        if target is None:
            return None

        self._target_id = target.id
        self._last_bbox = target.get_bbox()
        return self._emit(self._last_bbox)

    def _choose(self, tracks, frame) -> Optional[int]:
        if not tracks:
            return None
        best, bid = -1., None
        for t in tracks:
            iou = bbox_iou(t.get_bbox(), self._seed_bbox)
            hs  = hist_sim(t.ref_hist, self._ref_hist)
            sw, sh = self._seed_bbox[2], self._seed_bbox[3]
            tw, th = t.get_bbox()[2], t.get_bbox()[3]
            ss = min(sw * sh, tw * th) / (max(sw * sh, tw * th) + 1e-6)
            sc = iou * 0.45 + hs * 0.40 + ss * 0.15
            if sc > best:
                best, bid = sc, t.id
        return bid

    def _reacquire(self, tracks, strict=True) -> Optional[KalmanTrack]:
        if not tracks:
            return None
        thr = 0.35 if strict else 0.18
        best, bt = -1., None
        for t in tracks:
            hs = hist_sim(t.ref_hist, self._ref_hist)
            if hs < thr:
                continue
            if self._last_bbox is not None:
                lw, lh = self._last_bbox[2], self._last_bbox[3]
                tw, th = t.get_bbox()[2], t.get_bbox()[3]
                ss = min(lw * lh, tw * th) / (max(lw * lh, tw * th) + 1e-6)
                if ss < 0.25:
                    continue
                sc = hs * 0.65 + ss * 0.35
            else:
                sc = hs
            if sc > best:
                best, bt = sc, t
        if bt is not None:
            self._target_id = bt.id
            self._state = "tracking"
        return bt

    def _resolve_overlap(self, target, tracks) -> KalmanTrack:
        tb = target.get_bbox()
        for other in tracks:
            if other.id == target.id:
                continue
            if bbox_iou(tb, other.get_bbox()) > 0.55:
                ts = hist_sim(target.ref_hist, self._ref_hist)
                os = hist_sim(other.ref_hist, self._ref_hist)
                if os > ts + 0.12:
                    self._target_id = other.id
                    return other
        return target

    def _emit(self, bbox) -> Tuple:
        arr = np.array(bbox, dtype=float)
        if self._smooth_box is None:
            self._smooth_box = arr
        else:
            self._smooth_box = self._alpha * arr + (1 - self._alpha) * self._smooth_box
        return tuple(int(v) for v in self._smooth_box)


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE-LEVEL DETECTION SINGLETON
# ══════════════════════════════════════════════════════════════════════════════

_yolo_models: dict[str, object] = {}
_yolo_models_lock = threading.RLock()


def _resolve_yolo_model_path(model_size: str) -> str:
    # Prefer YOLOv8 pose checkpoints (faster / widely available),
    # but keep yolo11 pose as a fallback for existing installs.
    # Supports absolute paths via env var for deploys.
    env_model = os.getenv("YOLO_MODEL_PATH") or os.getenv("YOLO_POSE_MODEL")
    model_candidates = []
    if env_model:
        model_candidates.append(env_model)
    model_candidates.extend([
        f"yolov8{model_size}-pose.pt",
        f"yolo11{model_size}-pose.pt",
    ])

    potential_paths = []
    for mn in model_candidates:
        potential_paths.extend([
            mn,
            os.path.join("models", mn),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", mn),  # ../../../models/
            os.path.join(os.path.dirname(__file__), "..", "..", "models", mn),  # ../../models/
        ])

    for p in potential_paths:
        if os.path.exists(p):
            return p
    # Let ultralytics resolve/download if none found locally.
    return model_candidates[0]


def _get_or_load_yolo_model(model_size="m"):
    if not HAS_YOLO:
        return None
    with _yolo_models_lock:
        model = _yolo_models.get(model_size)
        if model is None:
            model = _YOLO(_resolve_yolo_model_path(model_size))
            _yolo_models[model_size] = model
        return model


def get_detection_layer(model_size="m") -> DetectionLayer:
    """
    Return an isolated detection layer instance per caller.
    Mutable detection state (e.g., background model) is no longer shared
    across jobs, while YOLO weights are reused safely.
    """
    shared_yolo = _get_or_load_yolo_model(model_size)
    return DetectionLayer(model_size=model_size, shared_yolo=shared_yolo)


# ══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE PLAYER PICKER
# ══════════════════════════════════════════════════════════════════════════════

def pick_player_interactive(video_path: str) -> Optional[dict]:
    det = get_detection_layer()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    WARMUP = min(90, total // 3)
    cands = []
    for fi in range(WARMUP):
        ret, frame = cap.read()
        if not ret:
            break
        dets = det.detect(frame)
        if dets:
            cands.append((frame.copy(), dets, fi))
    cap.release()
    if not cands:
        return select_primary_player(video_path)

    best_frame, best_dets, best_fi = max(cands, key=lambda c: len(c[1]))
    display = cv2.addWeighted(best_frame.copy(), 0.65, np.zeros_like(best_frame), 0.35, 0)
    COLORS = [
        (0,255,180),(0,140,255),(255,215,0),(0,200,255),
        (180,0,255),(0,255,80),(255,80,80),(80,255,255),
    ]
    blobs = [d['bbox'] for d in best_dets]
    for i, (bx, by, bw, bh) in enumerate(blobs):
        col = COLORS[i % len(COLORS)]
        cv2.rectangle(display, (bx, by), (bx + bw, by + bh), col, 3, cv2.LINE_AA)
        lbl = str(i + 1)
        lw, _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        bxb, byb = bx + bw // 2 - lw // 2 - 6, max(0, by - 34)
        cv2.rectangle(display, (bxb, byb), (bxb + lw + 12, byb + 28), col, -1)
        cv2.putText(display, lbl, (bxb + 6, byb + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    BH = 52
    banner = np.full((BH, W, 3), 15, np.uint8)
    cv2.putText(banner, "CLICK player to track  |  ESC=auto", (W // 2 - 200, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 215, 0), 1, cv2.LINE_AA)
    display = np.vstack([banner, display])
    chosen = [None]

    def on_click(ev, cx, cy, fl, p):
        if ev != cv2.EVENT_LBUTTONDOWN:
            return
        ay = cy - BH
        if ay < 0:
            return
        for b in blobs:
            bx, by, bw, bh = b
            if bx <= cx <= bx + bw and by <= ay <= by + bh:
                chosen[0] = b
                return
        chosen[0] = min(blobs, key=lambda b: math.hypot(
            cx - (b[0] + b[2] / 2), ay - (b[1] + b[3] / 2)))

    cv2.namedWindow("Select Player", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Player", min(W, 1280), min(H + BH, 800))
    cv2.setMouseCallback("Select Player", on_click)
    while True:
        cv2.imshow("Select Player", display)
        if chosen[0] is not None or (cv2.waitKey(20) & 0xFF) == 27:
            break
    cv2.destroyAllWindows()
    if chosen[0] is None:
        return select_primary_player(video_path)
    blob = chosen[0]
    bx, by, bw, bh = blob
    return {'hist': crop_hist(best_frame, blob), 'size': (float(bw), float(bh)),
            'seed_bbox': blob, 'seed_frame': best_fi}


# ══════════════════════════════════════════════════════════════════════════════
#  AUTO PRE-SCAN
# ══════════════════════════════════════════════════════════════════════════════

def select_primary_player(video_path: str, sample_step: int = 6) -> Optional[dict]:
    det = get_detection_layer()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tracks: List[dict] = []
    MAX_GAP = max(sample_step * 5, 30)
    fi = 0
    while fi < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            break
        for d in det.detect(frame):
            blob = d['bbox']
            bx, by, bw, bh = blob
            matched = False
            for tr in tracks:
                if fi - tr["lf"] > MAX_GAP:
                    continue
                iou = bbox_iou(blob, tr["lb"])
                rw, rh = tr["ms"]
                ss = min(bw * bh, rw * rh) / (max(bw * bh, rw * rh) + 1e-6)
                if iou * 0.7 + ss * 0.3 > 0.15 and (iou > 0.10 or ss > 0.55):
                    h = crop_hist(frame, blob)
                    tr["n"] += 1
                    if h is not None:
                        tr["hs"].append(h)
                    n = tr["n"]
                    pw, ph = tr["ms"]
                    tr["ms"] = ((pw * (n - 1) + bw) / n, (ph * (n - 1) + bh) / n)
                    tr["lb"] = blob
                    tr["lf"] = fi
                    matched = True
                    break
            if not matched:
                h = crop_hist(frame, blob)
                tracks.append({
                    "n": 1,
                    "hs": [h] if h is not None else [],
                    "ms": (float(bw), float(bh)),
                    "lb": blob, "lf": fi, "sb": blob, "sf": fi,
                })
        fi += sample_step
    cap.release()
    if not tracks:
        return None
    best = max(tracks, key=lambda t: t["n"])
    mh = None
    if best["hs"]:
        stacked = np.mean(best["hs"], axis=0).astype(np.float32)
        cv2.normalize(stacked, stacked)
        mh = stacked
    return {'hist': mh, 'size': best["ms"], 'seed_bbox': best["sb"], 'seed_frame': best["sf"]}



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
        kp.neck = (cx, vy(self._VP["neck"]))
        ls = (cx - sh, vy(self._VP["shoulder"]))
        rs = (cx + sh, vy(self._VP["shoulder"]))
        kp.left_shoulder  = ls
        kp.right_shoulder = rs
        kp.shoulder_center = ((ls[0] + rs[0]) / 2., (ls[1] + rs[1]) / 2.)
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
        kp.hip_center = ((lh[0] + rh[0]) / 2., (lh[1] + rh[1]) / 2.)
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

        # Confidence hint: if YOLO keypoints are extremely sparse, downstream
        # angle/risk metrics are likely to be physically meaningless.
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
            kp.shoulder_center = (
                (kp.left_shoulder[0] + kp.right_shoulder[0]) / 2.,
                (kp.left_shoulder[1] + kp.right_shoulder[1]) / 2.,
            )
            kp.hip_center = (
                (kp.left_hip[0] + kp.right_hip[0]) / 2.,
                (kp.left_hip[1] + kp.right_hip[1]) / 2.,
            )
            kp.neck = (
                (kp.shoulder_center[0] + kp.head[0]) / 2.,
                (kp.shoulder_center[1] + kp.head[1]) / 2.,
            )
            for side in ("left", "right"):
                ank = getattr(kp, f"{side}_ankle")
                object.__setattr__(kp, f"{side}_foot", (ank[0] + w * .04, ank[1] + h * .03))
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
        self._kx = {}
        self._ky = {}

    def smooth(self, kp) -> PoseKeypoints:
        out = PoseKeypoints()
        for nm in JOINT_NAMES:
            raw = getattr(kp, nm)
            if nm not in self._kx:
                self._kx[nm] = JointKalman()
                self._ky[nm] = JointKalman()
            setattr(out, nm, (self._kx[nm].update(raw[0]), self._ky[nm].update(raw[1])))
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  SKELETON RENDERER
# ══════════════════════════════════════════════════════════════════════════════

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
    # Glow is drawn on a per-frame overlay to avoid copying the full frame
    # for every joint.
    ov = img  # overlay buffer
    for rr in range(r + 6, r, -2):
        cv2.circle(ov, (px, py), rr, col, -1, cv2.LINE_AA)


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


# ══════════════════════════════════════════════════════════════════════════════
#  BIOMECHANICS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class BiomechanicsEngine:
    FILTER_HZ = 6.0

    def __init__(self, fps: float = 25.0, pix_to_m: float = 0.002):
        self.fps       = fps
        self.pix_to_m  = pix_to_m
        self.frames:   List[BioFrame] = []
        self._ah:      dict = {}
        self._la_y:    List[float] = []
        self._ra_y:    List[float] = []
        self._lf_x:    List[float] = []
        self._rf_x:    List[float] = []
        self.lhs:      List[int] = []
        self.rhs:      List[int] = []
        self.lto:      List[int] = []
        self.rto:      List[int] = []

    def process_frame(self, fi: int, ts: float, kp: PoseKeypoints) -> BioFrame:
        bf = BioFrame(frame_idx=fi, timestamp=ts)

        bf.left_knee_flexion         = s2d_joint_angle(kp.left_hip,  kp.left_knee,  kp.left_ankle)
        bf.right_knee_flexion        = s2d_joint_angle(kp.right_hip, kp.right_knee, kp.right_ankle)
        bf.left_hip_flexion          = s2d_joint_angle(kp.shoulder_center, kp.left_hip,  kp.left_knee)
        bf.right_hip_flexion         = s2d_joint_angle(kp.shoulder_center, kp.right_hip, kp.right_knee)
        bf.left_ankle_dorsiflexion   = s2d_joint_angle(kp.left_knee,  kp.left_ankle,  kp.left_foot)
        bf.right_ankle_dorsiflexion  = s2d_joint_angle(kp.right_knee, kp.right_ankle, kp.right_foot)
        bf.left_elbow_flexion        = s2d_joint_angle(kp.left_shoulder,  kp.left_elbow,  kp.left_wrist)
        bf.right_elbow_flexion       = s2d_joint_angle(kp.right_shoulder, kp.right_elbow, kp.right_wrist)

        bf.left_thigh_angle    = s2d_seg_angle(kp.left_hip,   kp.left_knee)
        bf.right_thigh_angle   = s2d_seg_angle(kp.right_hip,  kp.right_knee)
        bf.left_shank_angle    = s2d_seg_angle(kp.left_knee,  kp.left_ankle)
        bf.right_shank_angle   = s2d_seg_angle(kp.right_knee, kp.right_ankle)
        bf.trunk_segment_angle = s2d_seg_angle(kp.hip_center, kp.shoulder_center)

        dx = kp.shoulder_center[0] - kp.hip_center[0]
        dy = kp.shoulder_center[1] - kp.hip_center[1]
        bf.trunk_lateral_lean  = math.degrees(math.atan2(dx,       abs(dy) + 1e-9))
        bf.trunk_sagittal_lean = math.degrees(math.atan2(abs(dx),  abs(dy) + 1e-9))

        hd = kp.left_hip[1] - kp.right_hip[1]
        hw = dist2d(kp.left_hip, kp.right_hip) + 1e-9
        bf.pelvis_obliquity = math.degrees(math.atan2(abs(hd), hw))
        # Pelvis rotation: estimated from anterior/posterior hip offset (X only)
        hdx = kp.left_hip[0] - kp.right_hip[0]
        bf.pelvis_rotation  = math.degrees(math.atan2(abs(hdx), hw))

        bf.left_valgus_clinical  = self._clinical_valgus(kp.left_hip,  kp.left_knee,  kp.left_ankle)
        bf.right_valgus_clinical = self._clinical_valgus(kp.right_hip, kp.right_knee, kp.right_ankle)

        bf.left_arm_swing      = abs(self._seg_to_vert(kp.left_shoulder,  kp.left_elbow))
        bf.right_arm_swing     = abs(self._seg_to_vert(kp.right_shoulder, kp.right_elbow))
        bf.arm_swing_asymmetry = abs(bf.left_arm_swing - bf.right_arm_swing)

        bf.left_knee_ang_vel   = self._angvel("lk", bf.left_knee_flexion)
        bf.right_knee_ang_vel  = self._angvel("rk", bf.right_knee_flexion)
        bf.left_hip_ang_vel    = self._angvel("lh", bf.left_hip_flexion)
        bf.right_hip_ang_vel   = self._angvel("rh", bf.right_hip_flexion)

        bf.step_width = abs(kp.left_foot[0] - kp.right_foot[0]) * self.pix_to_m

        la = math.degrees(math.atan2(kp.left_foot[0]  - kp.left_ankle[0],
                                     abs(kp.left_foot[1]  - kp.left_ankle[1])  + 1e-9))
        ra = math.degrees(math.atan2(kp.right_foot[0] - kp.right_ankle[0],
                                     abs(kp.right_foot[1] - kp.right_ankle[1]) + 1e-9))
        bf.foot_progression_angle = (abs(la) + abs(ra)) / 2.

        self._la_y.append(kp.left_ankle[1])
        self._ra_y.append(kp.right_ankle[1])
        self._lf_x.append(kp.left_foot[0])
        self._rf_x.append(kp.right_foot[0])
        self.frames.append(bf)
        return bf

    def post_process(self):
        if len(self.frames) < 8:
            return
        for field in [
            "left_knee_flexion", "right_knee_flexion",
            "left_hip_flexion", "right_hip_flexion",
            "left_ankle_dorsiflexion", "right_ankle_dorsiflexion",
            "trunk_lateral_lean", "trunk_sagittal_lean",
            "left_valgus_clinical", "right_valgus_clinical",
        ]:
            raw = np.array([getattr(f, field) for f in self.frames], dtype=float)
            sm  = self._smooth(raw)
            for i, bf in enumerate(self.frames):
                setattr(bf, field, float(sm[i]))

        md = max(4, int(self.fps * 0.18))
        la = np.array(self._la_y)
        ra = np.array(self._ra_y)
        self.lhs = self._peaks( la, md)
        self.rhs = self._peaks( ra, md)
        self.lto = self._peaks(-la, md)
        self.rto = self._peaks(-ra, md)

        lhs_s = set(self.lhs)
        rhs_s = set(self.rhs)
        lto_s = set(self.lto)
        rto_s = set(self.rto)
        sl = self._stance_mask(self.lhs, self.lto, len(self.frames))
        sr = self._stance_mask(self.rhs, self.rto, len(self.frames))

        lf_x = np.array(self._lf_x)
        rf_x = np.array(self._rf_x)
        for i, bf in enumerate(self.frames):
            bf.left_heel_strike  = i in lhs_s
            bf.right_heel_strike = i in rhs_s
            bf.left_toe_off      = i in lto_s
            bf.right_toe_off     = i in rto_s
            bf.stance_left   = sl[i]
            bf.stance_right  = sr[i]
            bf.double_support = sl[i] and sr[i]
            if bf.left_heel_strike or bf.right_heel_strike:
                bf.step_width = abs(lf_x[i] - rf_x[i]) * self.pix_to_m

    def summary_dict(self) -> dict:
        if not self.frames:
            return {}
        skip = {
            "frame_idx", "timestamp",
            "left_heel_strike", "right_heel_strike",
            "left_toe_off", "right_toe_off",
            "stance_left", "stance_right", "double_support",
        }
        out = {}
        for f in BioFrame.__dataclass_fields__:
            if f in skip:
                continue
            v = np.array([getattr(x, f) for x in self.frames], dtype=float)
            out[f"{f}_mean"] = float(np.mean(v))
            out[f"{f}_max"]  = float(np.max(v))
            out[f"{f}_std"]  = float(np.std(v))
        out["lhs_count"]          = len(self.lhs)
        out["rhs_count"]          = len(self.rhs)
        out["double_support_pct"] = 100. * sum(
            1 for x in self.frames if x.double_support
        ) / max(len(self.frames), 1)
        out["valgus_asymmetry"]   = abs(
            out.get("left_valgus_clinical_mean", 0) - out.get("right_valgus_clinical_mean", 0)
        )
        return out

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(f) for f in self.frames])

    @staticmethod
    def _seg_to_vert(p, d) -> float:
        dx = d[0] - p[0]
        dy = d[1] - p[1]
        return float(math.degrees(math.atan2(dx, abs(dy) + 1e-9)))

    @staticmethod
    def _clinical_valgus(hip, knee, ankle) -> float:
        ha  = np.array([ankle[0] - hip[0], ankle[1] - hip[1]], dtype=float)
        hk  = np.array([knee[0]  - hip[0], knee[1]  - hip[1]], dtype=float)
        dev = float(np.cross(ha, hk)) / (np.linalg.norm(ha) + 1e-9)
        return float(math.degrees(math.atan2(dev, np.linalg.norm(hk) + 1e-9)))

    def _angvel(self, key: str, ang: float) -> float:
        prev = self._ah.get(key, ang)
        self._ah[key] = ang
        return (ang - prev) * self.fps

    def _smooth(self, arr: np.ndarray) -> np.ndarray:
        if HAS_SCIPY:
            try:
                nyq = self.fps / 2.
                b, a = butter(4, min(self.FILTER_HZ, nyq * .9) / nyq, btype="low")
                return filtfilt(b, a, arr)
            except Exception:
                pass
        w = max(3, int(self.fps * 0.12))
        return smooth_arr(arr, w=w)

    def _peaks(self, sig: np.ndarray, md: int) -> List[int]:
        if HAS_SCIPY:
            try:
                pk, _ = find_peaks(sig, distance=md, prominence=2.)
                return [int(p) for p in pk]
            except Exception:
                pass
        pks = []
        for i in range(1, len(sig) - 1):
            if sig[i] >= sig[i - 1] and sig[i] >= sig[i + 1]:
                if not pks or i - pks[-1] >= md:
                    pks.append(i)
        return pks

    @staticmethod
    def _stance_mask(hs: List[int], to: List[int], n: int) -> List[bool]:
        m = [False] * n
        for h in hs:
            nxt = [t for t in to if t > h]
            end = min(nxt) if nxt else min(h + 20, n - 1)
            for i in range(h, min(end + 1, n)):
                m[i] = True
        return m


# ══════════════════════════════════════════════════════════════════════════════
#  SPORTS2D RUNNER
# ══════════════════════════════════════════════════════════════════════════════
