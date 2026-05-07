import math
import numpy as np
import pandas as pd
from dataclasses import asdict
from typing import Optional, List, Tuple
from .models import BioFrame, PoseKeypoints, BIO_ANGLE_FIELDS
from .math_utils import (
    s2d_joint_angle, s2d_seg_angle, dist2d, 
    HAS_SCIPY
)

try:
    from scipy.signal import find_peaks, butter, filtfilt
except ImportError:
    pass

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
        bf.trunk_lateral_lean   = math.degrees(math.atan2(dx, abs(dy) + 1e-9))
        bf.trunk_sagittal_lean  = s2d_seg_angle(kp.hip_center, kp.shoulder_center)

        hd = kp.left_hip[1] - kp.right_hip[1]
        hw = dist2d(kp.left_hip, kp.right_hip) + 1e-9
        bf.pelvis_obliquity = math.degrees(math.atan2(abs(hd), hw))
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
        angle_fields = set(BIO_ANGLE_FIELDS)
        missing = angle_fields - set(BioFrame.__dataclass_fields__)
        if missing:
            raise RuntimeError(f"BioFrame missing expected fields: {missing}")

        for field in angle_fields:
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
        cross = float(np.cross(ha, hk))
        dot   = float(np.dot(ha, hk))
        return float(math.degrees(math.atan2(cross, dot + 1e-9)))

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
        pad = w // 2
        a_pad = np.pad(arr, pad, mode='edge')
        return np.convolve(a_pad, np.ones(w) / w, mode='valid')[:len(arr)]

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
                local_min = min(sig[max(0, i - md):i + md + 1])
                if sig[i] - local_min < 2.0:
                    continue
                if not pks or i - pks[-1] >= md:
                    pks.append(i)
        return pks

    @staticmethod
    def _stance_mask(hs: List[int], to: List[int], n: int) -> List[bool]:
        import bisect
        m = [False] * n
        to_sorted = sorted(to)
        for h in hs:
            idx = bisect.bisect_right(to_sorted, h)
            end = to_sorted[idx] if idx < len(to_sorted) else min(h + 20, n - 1)
            for i in range(h, min(end + 1, n)):
                m[i] = True
        return m
