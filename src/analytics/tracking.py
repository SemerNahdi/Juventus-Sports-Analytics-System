from .cv_wrapper import cv2
import numpy as np
import os
import threading
from collections import deque
from typing import Optional, List, Tuple
from .math_utils import bbox_centre, crop_hist, bbox_iou, hist_sim, _size_sim, HAS_SCIPY

try:
    from ultralytics import YOLO as _YOLO
    HAS_YOLO = True
except (ImportError, Exception):
    HAS_YOLO = False

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
    _P0 = np.diag([10., 10., 10., 10., 100., 100., 10., 10.])

    def __init__(self, bbox, frame, conf=1.0):
        self.id = KalmanTrack._next_id
        KalmanTrack._next_id += 1
        cx, cy = bbox_centre(bbox)
        w, h = bbox[2], bbox[3]
        self.x = np.array([cx, cy, w, h, 0., 0., 0., 0.], dtype=float)
        self.P = self._P0.copy()
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
        K = np.linalg.solve(S.T, (self.P @ self.H.T).T).T
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
        self.P = self._P0.copy()
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
        self.active_tracks: dict[int, KalmanTrack] = {}
        self.lost_tracks:   dict[int, KalmanTrack] = {}

    def update(self, detections: List[dict], frame) -> List[KalmanTrack]:
        for d in detections:
            if d.get("_hist") is None:
                d["_hist"] = crop_hist(frame, d["bbox"])

        for t in list(self.active_tracks.values()) + list(self.lost_tracks.values()):
            t.predict()
        high = [d for d in detections if d['conf'] >= self.HIGH_THRESH]
        low  = [d for d in detections if self.LOW_THRESH <= d['conf'] < self.HIGH_THRESH]

        active_list = list(self.active_tracks.values())
        lost_list   = list(self.lost_tracks.values())

        unm_t, unm_h = self._associate(active_list, high, frame, self.IOU_HIGH)
        still_unm, _ = self._associate(unm_t, low, frame, self.IOU_LOW)
        self._associate(lost_list, low, frame, self.IOU_LOST, reactivate=True)

        for t in still_unm:
            t.hit_streak = 0
            self.lost_tracks[t.id] = t
            self.active_tracks.pop(t.id, None)

        for d in unm_h:
            nt = KalmanTrack(d['bbox'], frame, d['conf'])
            self.active_tracks[nt.id] = nt

        self.lost_tracks = {tid: t for tid, t in self.lost_tracks.items() if t.missed <= self.LOST_TTL}
        return [t for t in self.active_tracks.values() if t.hit_streak >= self.MIN_HITS]

    def _associate(self, tracks, dets, frame, iou_thr, reactivate=False):
        if not tracks or not dets:
            return list(tracks), list(dets)

        tracks_copy = list(tracks)
        cost = np.zeros((len(tracks_copy), len(dets)), dtype=float)
        for ti, t in enumerate(tracks_copy):
            tb = t.get_bbox()
            th = t.ref_hist
            for di, d in enumerate(dets):
                iou = bbox_iou(tb, d['bbox'])
                hs  = hist_sim(th, d.get("_hist"))
                cost[ti, di] = 1.0 - (iou * 0.60 + hs * 0.40)

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
                            self.lost_tracks.pop(t.id, None)
                            self.active_tracks[t.id] = t
                        else:
                            t.update(d['bbox'], frame, d['conf'])
                        if d.get('kp') is not None:
                            t._yolo_kp = d['kp']
            except ImportError:
                pass

        if not mt:
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
                    self.lost_tracks.pop(t.id, None)
                    self.active_tracks[t.id] = t
                else:
                    t.update(d['bbox'], frame, d['conf'])
                if d.get('kp') is not None:
                    t._yolo_kp = d['kp']

        return (
            [tracks_copy[i] for i in range(len(tracks_copy)) if i not in mt],
            [dets[i]        for i in range(len(dets))        if i not in md],
        )

    def reset(self):
        for t in list(self.active_tracks.values()) + list(self.lost_tracks.values()):
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
                target = self._reacquire(list(self.bt.lost_tracks.values()), strict=False)
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
            ss = _size_sim(sw, sh, tw, th)
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
                ss = _size_sim(lw, lh, tw, th)
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
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", mn),
            os.path.join(os.path.dirname(__file__), "..", "..", "models", mn),
        ])
    
    seen = set()
    unique_paths = []
    for p in potential_paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)

    for p in unique_paths:
        if os.path.exists(p):
            return p
    
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

def preload_yolo_models(sizes: Optional[List[str]] = None):
    """Pre-load YOLO models into memory for faster first-request response."""
    if not HAS_YOLO:
        return
    if sizes is None:
        sizes = ["n", "m"] # Preload common sizes
    for sz in sizes:
        try:
            print(f"[TRACKING] Pre-loading YOLOv8/11 {sz} model...")
            _get_or_load_yolo_model(sz)
        except Exception as e:
            print(f"[TRACKING] Failed to pre-load YOLO {sz}: {e}")

def get_detection_layer(model_size="m") -> DetectionLayer:
    shared_yolo = _get_or_load_yolo_model(model_size)
    return DetectionLayer(model_size=model_size, shared_yolo=shared_yolo)
