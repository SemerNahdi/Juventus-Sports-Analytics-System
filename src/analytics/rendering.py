from .cv_wrapper import cv2
import numpy as np
from typing import Optional, List, Tuple
from .models import PoseKeypoints, PoseFrame, FrameMetrics
from .math_utils import clamp01, lerp_color
from .pose import render_skeleton

def annotate_frame(frame: np.ndarray, pf: PoseFrame, fm: FrameMetrics, player_id: int, draw_badge: bool = True) -> np.ndarray:
    kp = pf.kp
    rt = clamp01(fm.risk_score / 100.)
    render_skeleton(frame, kp, risk_tint=rt)

    # Knee angle labels
    for kpt, ang in [(kp.left_knee, fm.left_knee_angle),
                     (kp.right_knee, fm.right_knee_angle)]:
        kx, ky = int(kpt[0]), int(kpt[1])
        ac = (0, 220, 0) if ang > 145 else (0, 140, 255) if ang > 120 else (0, 0, 220)
        cv2.putText(frame, f"{ang:.0f}°", (kx + 12, ky - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, ac, 1, cv2.LINE_AA)

    if draw_badge:
        hx       = int(kp.head[0])
        head_y   = int(kp.head[1]) - 35
        badge    = f"  #{player_id}  "
        (tw, _), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        bx0 = hx - tw // 2
        x1, y1, x2, y2 = bx0, head_y - 22, bx0 + tw, head_y + 6
        H, W = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 > x1 and y2 > y1:
            roi = frame[y1:y2, x1:x2]
            rect = np.zeros_like(roi)
            cv2.addWeighted(roi, 0.4, rect, 0.6, 0, roi)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 215, 0), 1)
        cv2.putText(frame, badge, (bx0 + 2, head_y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    # Speed overlay
    bx, by, bw, bh = pf.bbox
    spd_txt = f"{fm.speed:.1f} m/s"
    cv2.putText(frame, spd_txt, (bx, by - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)

    return frame

def annotate_mat_events(frame: np.ndarray, frame_idx: int, takeoff_idx: int, landing_idx: int, stabilized_idx: int = -1) -> np.ndarray:
    """Draw MAT event badges (Takeoff, Landing, Stabilized) on the frame."""
    H, W = frame.shape[:2]
    
    label = None
    color = (0, 0, 0)
    
    if frame_idx == takeoff_idx:
        label = "TAKEOFF"
        color = (0, 255, 255) # Yellow
    elif frame_idx == landing_idx:
        label = "LANDING"
        color = (0, 0, 255) # Red
    elif stabilized_idx > 0 and frame_idx == stabilized_idx:
        label = "STABILIZED"
        color = (0, 255, 0) # Green
        
    if label:
        # Draw a prominent badge in the top-center
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 1.2
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
        
        tx = (W - tw) // 2
        ty = 100
        
        # Background box
        cv2.rectangle(frame, (tx - 10, ty - th - 10), (tx + tw + 10, ty + 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (tx - 10, ty - th - 10), (tx + tw + 10, ty + 10), color, 2)
        
        # Text
        cv2.putText(frame, label, (tx, ty), font, scale, color, thickness, cv2.LINE_AA)
        
    return frame

def draw_player_aura(frame: np.ndarray, kp: PoseKeypoints, fm: FrameMetrics, bbox: Tuple[int, int, int, int], accel_burst: int = 0) -> np.ndarray:
    if fm.speed < .5:
        return frame
    hx, hy = int(kp.hip_center[0]), int(kp.hip_center[1])
    bx, by, bw, bh = bbox
    rx  = max(12, bw // 2 + 6)
    ry  = max(20, bh // 2 + 10)
    col = lerp_color((0, 180, 60), (0, 60, 255), clamp01(fm.speed / 8.))
    
    H, W = frame.shape[:2]

    if accel_burst > 0:
        br = int(rx * 1.6 + accel_burst * 3)
        ba = accel_burst / 8. * .4
        
        # ROI for burst
        x1, y1 = max(0, hx - br - 5), max(0, hy - int(br * 1.4) - 5)
        x2, y2 = min(W, hx + br + 5), min(H, hy + int(br * 1.4) + 5)
        
        if x2 > x1 and y2 > y1:
            roi = frame[y1:y2, x1:x2].copy()
            cv2.ellipse(roi, (hx - x1, hy - y1), (br, int(br * 1.4)), 0, 0, 360,
                        (0, 200, 255), 3, cv2.LINE_AA)
            frame[y1:y2, x1:x2] = cv2.addWeighted(roi, ba, frame[y1:y2, x1:x2], 1 - ba, 0)

    for exp, a in [(14, .12), (6, .20)]:
        # ROI for aura
        ex_rx, ex_ry = rx + exp, ry + exp
        x1, y1 = max(0, hx - ex_rx - 2), max(0, hy - ex_ry - 2)
        x2, y2 = min(W, hx + ex_rx + 2), min(H, hy + ex_ry + 2)
        
        if x2 > x1 and y2 > y1:
            roi = frame[y1:y2, x1:x2].copy()
            cv2.ellipse(roi, (hx - x1, hy - y1), (ex_rx, ex_ry), 0, 0, 360, col, -1, cv2.LINE_AA)
            frame[y1:y2, x1:x2] = cv2.addWeighted(roi, a, frame[y1:y2, x1:x2], 1 - a, 0)
            
    return frame
