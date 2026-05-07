import cv2
import math
import numpy as np
from typing import Optional, List, Tuple
from .math_utils import crop_hist, bbox_iou, _size_sim
from .tracking import get_detection_layer

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
                ss = _size_sim(bw, bh, rw, rh)
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
