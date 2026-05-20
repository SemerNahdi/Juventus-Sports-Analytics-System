"""Player selection preview and seed DTO for UI → TargetLock handoff."""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from .cv_wrapper import cv2
from .math_utils import crop_hist
from .tracking import get_detection_layer

logger = logging.getLogger(__name__)

BBoxXYWH = Tuple[int, int, int, int]
BBoxFormat = Literal["xywh", "xyxy"]

_PREVIEW_COLORS = [
    (0, 255, 180), (0, 140, 255), (255, 215, 0), (0, 200, 255),
    (180, 0, 255), (0, 255, 80), (255, 80, 80), (80, 255, 255),
]


@dataclass(frozen=True)
class DetectionCandidate:
    index: int
    bbox: BBoxXYWH
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "bbox": list(self.bbox),
            "confidence": self.confidence,
        }


@dataclass
class PlayerSeed:
    seed_bbox: BBoxXYWH
    seed_frame_idx: int = 0
    hist: Optional[np.ndarray] = None
    source: str = "auto"
    candidate_index: Optional[int] = None

    def to_lock_dict(self) -> Dict[str, Any]:
        return {
            "seed_bbox": self.seed_bbox,
            "seed_frame": self.seed_frame_idx,
            "hist": self.hist,
        }


@dataclass
class PreviewResult:
    frame_idx: int
    frame_count: int
    width: int
    height: int
    candidates: List[DetectionCandidate]
    image_jpeg_base64: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "candidates": [c.to_dict() for c in self.candidates],
            "image_jpeg_base64": self.image_jpeg_base64,
        }


def xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> BBoxXYWH:
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))


def normalize_bbox(
    bbox: Tuple[int, ...],
    fmt: BBoxFormat,
    width: int,
    height: int,
) -> BBoxXYWH:
    if len(bbox) != 4:
        raise ValueError("bbox must have 4 values")

    if fmt == "xyxy":
        x, y, w, h = xyxy_to_xywh(*bbox)
    else:
        x, y, w, h = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))

    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return (x, y, w, h)


def read_video_frame(video_path: str, frame_idx: int) -> Tuple[Optional[np.ndarray], int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if frame_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            h = height or 0
            w = width or 0
            return None, w, h
        h = height if height else frame.shape[0]
        w = width if width else frame.shape[1]
        return frame, w, h
    finally:
        cap.release()


def find_best_preview_frame(
    video_path: str,
    max_scan: int = 90,
    sample_step: int = 15,
    yolo_size: str = "n",
) -> int:
    """Pick frame index with the most detections in an early-window scan."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    det = get_detection_layer(yolo_size)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    limit = min(max_scan, max(total // 3, 1)) if total > 0 else max_scan

    best_idx = 0
    best_count = -1

    try:
        fi = 0
        while fi < limit:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            count = len(det.detect(frame))
            if count > best_count:
                best_count = count
                best_idx = fi
            fi += sample_step
    finally:
        cap.release()

    return best_idx


def preview_detections(
    video_path: str,
    frame_idx: Optional[int] = None,
    yolo_size: str = "n",
    auto_frame: bool = True,
) -> PreviewResult:
    """Run detector on one frame and return annotated preview + candidates."""
    if auto_frame and frame_idx is None:
        frame_idx = find_best_preview_frame(video_path, yolo_size=yolo_size)
    frame_idx = max(0, int(frame_idx or 0))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"Could not read frame {frame_idx} from video")

        height, width = frame.shape[:2]
        det = get_detection_layer(yolo_size)
        raw = det.detect(frame)

        candidates: List[DetectionCandidate] = []
        display = frame.copy()
        for i, d in enumerate(raw):
            bx, by, bw, bh = d["bbox"]
            conf = float(d.get("conf", 0.5))
            candidates.append(DetectionCandidate(i, (bx, by, bw, bh), conf))
            col = _PREVIEW_COLORS[i % len(_PREVIEW_COLORS)]
            cv2.rectangle(display, (bx, by), (bx + bw, by + bh), col, 2, cv2.LINE_AA)
            label = str(i + 1)
            cv2.putText(
                display, label, (bx + 4, max(by + 22, 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA,
            )

        banner_h = 40
        banner = np.full((banner_h, width, 3), 20, np.uint8)
        msg = f"Click a player (1-{len(candidates)}) or use buttons below  |  Frame {frame_idx}"
        cv2.putText(
            banner, msg, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 215, 0), 1, cv2.LINE_AA,
        )
        display = np.vstack([banner, display])

        ok_enc, buf = cv2.imencode(".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok_enc:
            raise RuntimeError("Failed to encode preview JPEG")
        image_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        return PreviewResult(
            frame_idx=frame_idx,
            frame_count=total,
            width=width,
            height=height,
            candidates=candidates,
            image_jpeg_base64=image_b64,
        )
    finally:
        cap.release()


def build_player_seed(
    video_path: str,
    bbox: BBoxXYWH,
    frame_idx: int = 0,
    yolo_size: str = "n",
    source: str = "ui",
    candidate_index: Optional[int] = None,
    hist: Optional[np.ndarray] = None,
) -> PlayerSeed:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    try:
        if frame_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError(f"Could not read frame {frame_idx} for seed")

        H, W = frame.shape[:2]
        seed_bbox = normalize_bbox(bbox, "xywh", W, H)
        if hist is None:
            hist = crop_hist(frame, seed_bbox)

        return PlayerSeed(
            seed_bbox=seed_bbox,
            seed_frame_idx=max(0, int(frame_idx)),
            hist=hist,
            source=source,
            candidate_index=candidate_index,
        )
    finally:
        cap.release()


def candidate_at_point(
    candidates: List[DetectionCandidate],
    x: int,
    y: int,
    banner_offset: int = 40,
) -> Optional[int]:
    """Map display click (with banner) to candidate index."""
    ay = y - banner_offset
    if ay < 0:
        return None
    for c in candidates:
        bx, by, bw, bh = c.bbox
        if bx <= x <= bx + bw and by <= ay <= by + bh:
            return c.index
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda c: (
            (x - (c.bbox[0] + c.bbox[2] / 2)) ** 2
            + (ay - (c.bbox[1] + c.bbox[3] / 2)) ** 2
        ),
    ).index


def seed_from_candidate(
    video_path: str,
    preview: PreviewResult,
    candidate_index: int,
    source: str = "ui_index",
) -> PlayerSeed:
    if candidate_index < 0 or candidate_index >= len(preview.candidates):
        raise IndexError(f"Invalid candidate index {candidate_index}")
    c = preview.candidates[candidate_index]
    return build_player_seed(
        video_path=video_path,
        bbox=c.bbox,
        frame_idx=preview.frame_idx,
        source=source,
        candidate_index=candidate_index,
    )
