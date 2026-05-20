from .cv_wrapper import cv2
import numpy as np
import os
import threading
import logging
from collections import deque
from typing import Optional, List, Tuple, Dict, Set

from .math_utils import (
    bbox_centre,
    crop_hist,
    bbox_iou,
    hist_sim,
    _size_sim,
)

try:
    from ultralytics import YOLO as _YOLO
    HAS_YOLO = True
except Exception:
    HAS_YOLO = False

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────

MAX_MISSED_ACTIVE = 5
MAX_MISSED_LOST   = 30
MIN_HIT_STREAK    = 2

OCCLUSION_IOU_THR = 0.3
EMA_ALPHA_OCCLUDED = 0.7
EMA_ALPHA_NORMAL   = 0.15


# ─────────────────────────────────────────────────────────────
#  REID EMBEDDER
# ─────────────────────────────────────────────────────────────

class ReIDEmbedder:
    """
    ONNX ReID + histogram fallback.
    """

    SIZE = (128, 256)

    def __init__(self, model_path: Optional[str] = None):
        self.session = None

        if model_path and HAS_ORT and os.path.isfile(model_path):
            try:
                self.session = ort.InferenceSession(model_path)
                self.in_name = self.session.get_inputs()[0].name
                self.out_name = self.session.get_outputs()[0].name
            except Exception:
                self.session = None

    def embed(self, frame, bbox) -> Optional[np.ndarray]:
        if self.session is None:
            return None

        x, y, w, h = bbox
        h_img, w_img = frame.shape[:2]

        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img, x + w), min(h_img, y + h)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, self.SIZE).astype(np.float32) / 255.0
        crop = crop.transpose(2, 0, 1)[None]

        try:
            out = self.session.run([self.out_name], {self.in_name: crop})[0]
            vec = out[0]
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 1e-6 else None
        except Exception:
            return None

    def similarity(self, ea, eb, ha, hb) -> float:
        if ea is not None and eb is not None:
            return float(np.clip(np.dot(ea, eb), 0, 1))
        if ha is not None and hb is not None:
            return hist_sim(ha, hb)
        return 0.0


# ─────────────────────────────────────────────────────────────
#  KALMAN TRACK
# ─────────────────────────────────────────────────────────────

class KalmanTrack:
    _id = 1
    _lock = threading.Lock()

    F = np.array([
        [1,0,0,0,1,0,0,0],
        [0,1,0,0,0,1,0,0],
        [0,0,1,0,0,0,1,0],
        [0,0,0,1,0,0,0,1],
        [0,0,0,0,1,0,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,1],
    ], float)

    H = np.eye(4, 8)

    def __init__(self, bbox, frame, conf=1.0, embedder=None):
        with KalmanTrack._lock:
            self.id = KalmanTrack._id
            KalmanTrack._id += 1

        cx, cy = bbox_centre(bbox)
        w, h = bbox[2], bbox[3]

        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], float)
        self.P = np.diag([10,10,10,10,100,100,10,10])

        self.Q = np.diag([1,1,1,1,.5,.5,.2,.2])
        self.R = np.diag([4,4,10,10])

        self.conf = conf
        self.missed = 0
        self.hit = 1

        self.ref_hist = crop_hist(frame, bbox)
        self.ref_emb  = embedder.embed(frame, bbox) if embedder else None

        self._smooth = bbox
        self.is_occluded = False

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.missed += 1
        return self.get_bbox()

    def update(self, bbox, frame, conf=1.0, embedder=None):
        cx, cy = bbox_centre(bbox)
        z = np.array([cx, cy, bbox[2], bbox[3]])

        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = np.linalg.solve(S.T, (self.P @ self.H.T).T).T
        except np.linalg.LinAlgError:
            K = self.P @ self.H.T @ np.linalg.pinv(S)

        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P

        self.missed = 0
        self.hit += 1
        self.conf = conf

        nh = crop_hist(frame, bbox)
        if nh is not None:
            self.ref_hist = nh if self.ref_hist is None else 0.9*self.ref_hist + 0.1*nh

        if embedder:
            ne = embedder.embed(frame, bbox)
            if ne is not None:
                self.ref_emb = ne if self.ref_emb is None else 0.9*self.ref_emb + 0.1*ne

    def get_bbox(self):
        cx, cy, w, h = self.x[:4]
        return (int(cx-w/2), int(cy-h/2), int(w), int(h))

    def smooth_bbox(self, new_bbox, occluded=False):
        a = EMA_ALPHA_OCCLUDED if occluded else EMA_ALPHA_NORMAL
        sx, sy, sw, sh = self._smooth
        x,y,w,h = new_bbox

        self._smooth = (
            int(a*sx + (1-a)*x),
            int(a*sy + (1-a)*y),
            int(a*sw + (1-a)*w),
            int(a*sh + (1-a)*h),
        )

        return self._smooth


# ─────────────────────────────────────────────────────────────
#  DETECTION LAYER
# ─────────────────────────────────────────────────────────────

class DetectionLayer:
    def __init__(self, model=None):
        self.model = model
        self.bg = cv2.createBackgroundSubtractorMOG2(400, 40, False)

    @property
    def mode(self) -> str:
        """Returns the detection mode: 'YOLO' or 'MOG2'."""
        return "YOLO" if self.model else "MOG2"

    def detect(self, frame):
        if self.model:
            res = self.model(frame, verbose=False)[0]
            out = []
            for b in res.boxes:
                x1,y1,x2,y2 = b.xyxy[0].cpu().numpy()
                out.append({
                    "bbox": (int(x1),int(y1),int(x2-x1),int(y2-y1)),
                    "conf": float(b.conf[0].cpu()),
                    "_hist": crop_hist(frame, (int(x1),int(y1),int(x2-x1),int(y2-y1)))
                })
            return out

        mask = self.bg.apply(frame)
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        dets = []
        for c in cnts:
            if cv2.contourArea(c) < 2000:
                continue
            x,y,w,h = cv2.boundingRect(c)
            dets.append({"bbox":(x,y,w,h),"conf":0.5,"_hist":crop_hist(frame,(x,y,w,h))})
        return dets


# ─────────────────────────────────────────────────────────────
#  BYTE TRACKER (FIXED + REID + OCCLUSION)
# ─────────────────────────────────────────────────────────────

class ByteTracker:
    def __init__(self, embedder=None):
        self.active = {}
        self.lost = {}
        self.embedder = embedder

    def update(self, dets, frame):
        for t in list(self.active.values()):
            t.predict()

        unmatched = dets.copy()

        for t in list(self.active.values()):
            best, bi = 0, None

            for i,d in enumerate(unmatched):
                iou = bbox_iou(t.get_bbox(), d["bbox"])
                sim = self.embedder.similarity(
                    t.ref_emb, None,
                    t.ref_hist, d["_hist"]
                ) if self.embedder else hist_sim(t.ref_hist, d["_hist"])

                score = 0.6*iou + 0.4*sim

                if score > best:
                    best, bi = score, i

            if bi is not None and best > 0.4:
                d = unmatched.pop(bi)
                t.update(d["bbox"], frame, d["conf"], self.embedder)
            else:
                t.missed += 1
                self.lost[t.id] = t
                self.active.pop(t.id, None)

        for d in unmatched:
            t = KalmanTrack(d["bbox"], frame, d["conf"], self.embedder)
            self.active[t.id] = t

        return list(self.active.values())


# ─────────────────────────────────────────────────────────────
#  TARGET LOCK
# ─────────────────────────────────────────────────────────────

class TargetLock:
    def __init__(self, seed_bbox, seed_hist, seed_frame_idx=0, yolo_size="m"):
        self.seed = seed_bbox
        self.seed_hist = seed_hist
        self.seed_frame_idx = seed_frame_idx
        self._target_id = None
        
        # Load YOLO model based on yolo_size
        model = None
        if HAS_YOLO:
            try:
                model_map = {
                    "n": "yolo11n-pose.pt",
                    "s": "yolo11n-pose.pt",
                    "m": "yolo11m-pose.pt",
                    "l": "yolo11l-pose.pt",
                    "x": "yolo11l-pose.pt",
                }
                model_file = model_map.get(yolo_size, "yolo11m-pose.pt")
                model_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "models", model_file))
                model = _YOLO(model_path)
            except Exception as e:
                logging.warning(f"Failed to load YOLO model in TargetLock: {e}")
        
        self.det = DetectionLayer(model)
        self.embedder = ReIDEmbedder()
        self.bt = ByteTracker(self.embedder)
        self.target = None

    def update(self, frame, frame_idx: Optional[int] = None):
        dets = self.det.detect(frame)
        tracks = self.bt.update(dets, frame)

        if frame_idx is not None and frame_idx < self.seed_frame_idx:
            return None

        if self.target is None:
            best = -1
            best_track = None
            for t in tracks:
                score = bbox_iou(t.get_bbox(), self.seed)
                if score > best:
                    best = score
                    best_track = t
            
            if best_track is not None:
                self.target = best_track.id
                self._target_id = best_track.id
                return best_track.smooth_bbox(best_track.get_bbox())
            return None

        for t in tracks:
            if t.id == self.target:
                self._target_id = t.id
                return t.smooth_bbox(t.get_bbox())

        return None
def get_detection_layer(model_size="m", shared_yolo=None):
    """
    Backward-compatible helper used by older modules.
    Loads YOLO model and returns DetectionLayer instance.
    """
    if shared_yolo is not None:
        return DetectionLayer(model=shared_yolo)
    
    if not HAS_YOLO:
        return DetectionLayer(model=None)
    
    # Map model_size to YOLO model file (only available models: n, m, l)
    model_map = {
        "n": "yolo11n-pose.pt",
        "s": "yolo11n-pose.pt",  # Fall back to nano
        "m": "yolo11m-pose.pt",
        "l": "yolo11l-pose.pt",
        "x": "yolo11l-pose.pt",  # Fall back to large
    }
    
    model_file = model_map.get(model_size, "yolo11m-pose.pt")
    # Path: src/analytics/tracking.py -> ../.. -> root/models/
    model_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "models", model_file))
    
    try:
        yolo_model = _YOLO(model_path)
        return DetectionLayer(model=yolo_model)
    except Exception as e:
        logging.warning(f"Failed to load YOLO model from {model_path}: {e}")
        return DetectionLayer(model=None)