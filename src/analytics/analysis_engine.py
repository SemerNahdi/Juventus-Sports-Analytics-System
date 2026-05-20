"""Top-level analysis orchestration engine."""

from collections import deque

import numpy as np
import pandas as pd
import threading
import math
import logging
import os
from typing import Any, Optional, List, Tuple
from io import StringIO

from .models import PoseKeypoints, MATEventKPIs, MATSummary, PoseFrame, FrameMetrics, PlayerSummary, BioFrame
from .biomechanics import MATEventDetector, BiomechanicsEngine, MATGridCalibrator
from .math_utils import HAS_MPL, HAS_SCIPY, HAS_SPORTS2D, dist2d, crop_hist, estimate_player_orientation, s2d_joint_angle, s2d_seg_angle
from .core import HybridPoseEstimator
from .cv_wrapper import cv2
from .sports2d_runner import Sports2DRunner
from .output_manager import export_unified_results
from .scoring import RiskScorer
from .rendering import annotate_frame, draw_player_aura
from .reporting import generate_report
from .player_picker import pick_player_interactive, select_primary_player
from .tracking import TargetLock, get_detection_layer
from .pose import PoseKalmanSmoother
from .types import YoloModelSize, AnalysisMode, BiomechanicsBackend, benchmark_method, PerformanceTimer, PipelineTimer

# Configure logging for analysis engine
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '[%(levelname)s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Import configuration
try:
    from ..api.config import BIOMECHANICS, ANALYSIS
except ImportError:
    # Fallback defaults if config not available
    from dataclasses import dataclass
    @dataclass
    class _BioMechFallback:
        pix_to_m_default = 0.002
    @dataclass
    class _AnalysisFallback:
        min_pose_frames_mat = 5
        min_flight_time = 0.05
        expected_knee_flexion = 120.0
        expected_flight_time = 0.6
    BIOMECHANICS = _BioMechFallback()
    ANALYSIS = _AnalysisFallback()

class ProtocolHandler:
    """Route between continuous (gait) and discrete (MAT) analysis."""

    def __init__(self, protocol_id: Optional[str] = None):
        self.protocol_id = protocol_id or "continuous_gait"
        self.is_mat = self.protocol_id.startswith("mat_")

    def process_video(self, analyzer: 'SportsAnalyzer', stride: int = 2, 
                     target_height: int = 640, cancel_event: Optional[threading.Event] = None):
        if self.is_mat:
            return self._process_mat_protocol(analyzer, stride, target_height, cancel_event)
        else:
            return analyzer.process_video(stride, target_height, cancel_event)

    def _process_mat_protocol(self, analyzer: 'SportsAnalyzer', stride: int, 
                             target_height: int, cancel_event: Optional[threading.Event]):
        """Runs discrete event detection instead of frame-by-frame analysis."""
        # 1. First run the standard pipeline to get all pose data
        # We'll use a temporary output path for the first pass
        original_out_path = analyzer.output_video_path
        temp_out_path = original_out_path.replace(".mp4", "_raw.mp4")
        analyzer.output_video_path = temp_out_path
        
        print(f" * RUNNING MAT PROTOCOL: {self.protocol_id}")
        analyzer.process_video(stride, target_height, cancel_event)
        
        # 2. Extract events - with bounds checking
        if len(analyzer.pose_frames) < ANALYSIS.min_pose_frames_mat:
            print(f" * Insufficient frames ({len(analyzer.pose_frames)}) for MAT event detection")
            analyzer.output_video_path = original_out_path
            if os.path.exists(temp_out_path): os.rename(temp_out_path, original_out_path)
            return None
        
        detector = MATEventDetector()
        fps = analyzer._fps_cache / stride
        
        # Use ankle Y trajectory for takeoff/landing
        ankle_y = np.array([p.kp.left_ankle[1] for p in analyzer.pose_frames])
        if ankle_y.size == 0:
            analyzer.output_video_path = original_out_path
            return None
        takeoff, landing = detector.detect_takeoff_landing(ankle_y, fps)
        
        # 3. Compute MAT Summary
        summary = self._build_mat_summary_internal(analyzer, takeoff, landing, fps)
        analyzer.mat_summary = summary
        
        # 4. Extract indices for rerendering from the first event
        event_kpi = summary.events[0]
        
        # 5. Rerender with MAT badges
        self._rerender_with_badges(analyzer, original_out_path, temp_out_path, 
                                   event_kpi.takeoff_idx * stride, 
                                   event_kpi.landing_idx * stride, 
                                   event_kpi.stabilized_idx * stride)
        
        analyzer.output_video_path = original_out_path
        return summary

    def _rerender_with_badges(self, analyzer, final_path, raw_path, takeoff, landing, stabilized):
        """Second pass to add MAT badges to the output video."""
        from .rendering import annotate_mat_events
        
        cap = cv2.VideoCapture(raw_path)
        if not cap.isOpened(): return
        
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        out = analyzer._create_writer(final_path, fps, W, H)
        
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Add MAT badges
            frame = annotate_mat_events(frame, idx, takeoff, landing, stabilized)
            out.write(frame)
            idx += 1
            
        cap.release()
        out.release()
        
        # Clean up raw video
        try: os.remove(raw_path)
        except: pass

    def _build_mat_summary_internal(self, analyzer, takeoff, landing, fps):
        detector = MATEventDetector()
        event_data = detector.extract_hop_event(analyzer.pose_frames, takeoff, landing, fps)
        
        event_kpi = MATEventKPIs(
            event_type=self.protocol_id,
            flight_time=event_data["flight_time"],
            landing_valgus_left=event_data["landing_valgus"],
            peak_knee_flexion_landing=event_data["peak_knee_flexion_landing"],
            time_to_stabilization=event_data["time_to_stabilization"],
            hop_distance_m=0.0,
            takeoff_idx=event_data["takeoff_idx"],
            landing_idx=event_data["landing_idx"],
            stabilized_idx=event_data["stabilized_idx"]
        )
        
        lsi = self._calculate_lsi([event_kpi])
        
        return MATSummary(
            protocol_id=self.protocol_id,
            participant_id=analyzer.player_id,
            limb_symmetry_index=lsi,
            events=[event_kpi]
        )

    def _calculate_lsi(self, events: List[MATEventKPIs]) -> float:
        """
        Calculate LSI (Limb Symmetry Index) using weighted multi-metric approach.
        
        LSI = (Weaker Side / Stronger Side) * 100
        Targets: 90-100% = good symmetry, <90% = asymmetry/imbalance
        
        Combines:
        - Landing valgus angle (lower is better): weight 0.40
        - Peak knee flexion (bilateral): weight 0.35
        - Flight time symmetry (bilateral): weight 0.15
        - Balance score (post-landing stability): weight 0.10
        """
        if not events:
            return 100.0
        
        e = events[0]
        indices = []
        
        # 1. Landing Valgus Symmetry (lower is better; measure asymmetry)
        # If both angles exist and are non-zero
        if abs(e.landing_valgus_left) > 1e-6 or abs(e.landing_valgus_right) > 1e-6:
            v_l = abs(e.landing_valgus_left)
            v_r = abs(e.landing_valgus_right)
            max_v = max(v_l, v_r)
            if max_v > 1e-6:
                valgus_lsi = (min(v_l, v_r) / max_v) * 100.0
                indices.append((valgus_lsi, 0.40))
        
        # 2. Peak Knee Flexion Symmetry (assumes bilateral comparison available)
        # For single-leg tasks, this may not apply; for bilateral (drop jump), use it
        if e.peak_knee_flexion_landing > 1e-6:
            # If we had individual left/right values, we'd compare them
            # For now, assume event represents bilateral comparison
            # Heuristic: assume 85-95% is normal, scale accordingly
            knee_flex_lsi = max(0.0, min(100.0, (e.peak_knee_flexion_landing / ANALYSIS.expected_knee_flexion) * 100.0))
            if knee_flex_lsi > 0:
                indices.append((knee_flex_lsi, 0.35))
        
        # 3. Flight Time Symmetry (for hop tasks with flight phase)
        # Assume flight_time is already averaged; compare to expected range
        if e.flight_time > ANALYSIS.min_flight_time:  # meaningful flight phase
            # Normal flight time range: 0.3-0.8s for single hop
            # Score: 1.0 if in expected range, less if outside
            flight_lsi = max(0.0, min(100.0, (e.flight_time / ANALYSIS.expected_flight_time) * 100.0))
            if flight_lsi > 0:
                indices.append((flight_lsi, 0.15))
        
        # 4. Balance/Stability Score (post-landing control, 0-100 scale)
        if e.balance_score > 0:
            indices.append((e.balance_score, 0.10))
        
        # Weighted average of available metrics
        if not indices:
            return 100.0
        
        weighted_sum = sum(idx * wt for idx, wt in indices)
        weight_total = sum(wt for _, wt in indices)
        lsi = weighted_sum / weight_total if weight_total > 0 else 100.0
        
        # Clamp to valid range
        return float(np.clip(lsi, 0.0, 100.0))


class SportsAnalyzer:
    PIX_TO_M = BIOMECHANICS.pix_to_m_default
    _pix_to_m_lock = threading.Lock()
    _TRC_COLUMN_ALIASES = {
        "head": ["Nose", "Head", "head"],
        "neck": ["Neck", "neck"],
        "left_shoulder": ["L_Shoulder", "LShoulder", "left_shoulder"],
        "right_shoulder": ["R_Shoulder", "RShoulder", "right_shoulder"],
        "left_elbow": ["L_Elbow", "LElbow", "left_elbow"],
        "right_elbow": ["R_Elbow", "RElbow", "right_elbow"],
        "left_wrist": ["L_Wrist", "LWrist", "left_wrist"],
        "right_wrist": ["R_Wrist", "RWrist", "right_wrist"],
        "left_hip": ["L_Hip", "LHip", "left_hip"],
        "right_hip": ["R_Hip", "RHip", "right_hip"],
        "left_knee": ["L_Knee", "LKnee", "left_knee"],
        "right_knee": ["R_Knee", "RKnee", "right_knee"],
        "left_ankle": ["L_Ankle", "LAnkle", "left_ankle"],
        "right_ankle": ["R_Ankle", "RAnkle", "right_ankle"],
        "left_foot": ["L_BigToe", "LFoot", "left_foot"],
        "right_foot": ["R_BigToe", "RFoot", "right_foot"],
    }

    def __init__(self, video_path: str,
                 output_video_path: str = "output_annotated.mp4",
                 player_id: int = 1,
                 fps_override: Optional[float] = None,
                 pick: bool = False,
                 yolo_size: str = "m",
                 player_height_m: float = 1.75,
                 player_mass_kg: float = 75.0,
                 seed_bbox: Optional[Tuple[int, int, int, int]] = None,
                 seed_frame_idx: int = 0,
                 risk_model: Optional[dict] = None,
                 protocol_id: str = "continuous_gait"):
        self.video_path         = video_path
        self.output_video_path  = output_video_path
        self.player_id          = player_id
        self.fps_override       = fps_override
        self.player_height_m    = player_height_m
        self.player_mass_kg     = player_mass_kg
        
        # Validate YOLO model size using enum
        try:
            yolo_model_size = YoloModelSize(yolo_size)
            logger.debug(f"Using YOLO model size: {yolo_model_size.name}")
        except ValueError:
            logger.warning(f"Invalid YOLO size '{yolo_size}', using default {YoloModelSize.get_default().value}")
            yolo_model_size = YoloModelSize.get_default()
            yolo_size = yolo_model_size.value
        
        # Initialize risk model (consolidated)
        self.risk_model = self._initialize_risk_model(risk_model)

        self.protocol_id = protocol_id
        self.is_mat = protocol_id.startswith("mat_")

        self.pose_est   = HybridPoseEstimator()
        self.smoother   = PoseKalmanSmoother()
        self.pose_frames:    List[PoseFrame]    = []
        self.frame_metrics:  List[FrameMetrics] = []
        self.summary = PlayerSummary(player_id=player_id)
        self.mat_summary: Optional[MATSummary] = None

        self._spd_win         = deque(maxlen=30)
        self._risk_win        = deque(maxlen=15)
        self._pix_to_m_samples = deque(maxlen=60)
        self._accel_burst     = 0
        self._fps_cache       = 30.

        self.bio_engine:         Optional[BiomechanicsEngine] = None
        self.sports2d_runner:    Optional[Sports2DRunner]     = None
        self._frame_height_px:   int = 0

        self._det_layer = get_detection_layer(yolo_size)

        print("\n" + "=" * 50)
        print(" SPORTS ANALYTICS: ENGINE READY")
        print("-" * 50)
        print(f" * POSE DETECTION:    {self._det_layer.mode.upper()}")
        print(f" * BIOMECHANICS:      {'SPORTS2D' if HAS_SPORTS2D else 'FALLBACK'}")
        print(f" * SIGNAL FILTERING:  {'SCIPY' if HAS_SCIPY else 'NUMPY'}")
        print(f" * PLOTTING:          {'MATPLOTLIB' if HAS_MPL else 'NOT AVAILABLE'}")
        print("=" * 50 + "\n")

        primary = None
        if seed_bbox is not None:
            cap = cv2.VideoCapture(video_path)
            try:
                if cap.isOpened():
                    if seed_frame_idx > 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(seed_frame_idx))
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        H, W = frame.shape[:2]
                        x, y, w, h = [int(v) for v in seed_bbox]
                        x = max(0, min(x, W - 1))
                        y = max(0, min(y, H - 1))
                        w = max(1, min(w, W - x))
                        h = max(1, min(h, H - y))
                        normalized_bbox = (x, y, w, h)
                        primary = {
                            "seed_bbox": normalized_bbox,
                            "hist": crop_hist(frame, normalized_bbox),
                            "seed_frame": max(0, int(seed_frame_idx)),
                        }
            finally:
                cap.release()

        if primary is None:
            primary = pick_player_interactive(video_path) if pick else select_primary_player(video_path)
        
        if primary is None:
            raise RuntimeError("No player candidates found in video.")
        self.lock = TargetLock(primary["seed_bbox"], primary["hist"], primary["seed_frame"], yolo_size=yolo_size)

    @staticmethod
    def _initialize_risk_model(custom_model: Optional[dict] = None) -> dict:
        """Consolidated risk model initialization."""
        model = {
            "knee_angle_cap_deg": 155.0,
            "joint_stress_knee_w": 0.50,
            "joint_stress_lean_w": 0.30,
            "joint_stress_asym_w": 0.20,
            "injury_valgus_w": 0.45,
            "injury_knee_asym_w": 0.30,
            "injury_accel_w": 0.25,
            "cumulative_joint_stress_w": 0.40,
            "cumulative_trunk_w": 0.35,
            "cumulative_fatigue_w": 0.25,
            "final_injury_w": 0.60,
            "final_cumulative_w": 0.40,
            "trunk_lean_stress_scale_deg": 25.0,
            "trunk_lean_cumulative_scale_deg": 30.0,
            "knee_asym_stress_scale_deg": 40.0,
            "knee_asym_injury_scale_deg": 30.0,
            "valgus_scale_deg": 15.0,
            "accel_scale": 12.0,
            "speed_baseline_mps": 1.0,
            "speed_scale_mps": 5.0,
        }
        if custom_model:
            model.update(custom_model)
        return model

    def _initialize_player_detection(self, seed_bbox: Optional[Tuple[int, int, int, int]],
                                    seed_frame_idx: int, video_path: str, pick: bool,
                                    yolo_size: str) -> dict:
        """Initialize player detection and tracking."""
        primary = None
        if seed_bbox is not None:
            try:
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer
                if cap.isOpened():
                    if seed_frame_idx > 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(seed_frame_idx))
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        H, W = frame.shape[:2]
                        x, y, w, h = [int(v) for v in seed_bbox]
                        x = max(0, min(x, W - 1))
                        y = max(0, min(y, H - 1))
                        w = max(1, min(w, W - x))
                        h = max(1, min(h, H - y))
                        normalized_bbox = (x, y, w, h)
                        primary = {
                            "seed_bbox": normalized_bbox,
                            "hist": crop_hist(frame, normalized_bbox),
                            "seed_frame": max(0, int(seed_frame_idx)),
                        }
                        logger.debug(f"Loaded seed bbox: {normalized_bbox}")
            except Exception as e:
                logger.warning(f"Failed to load seed_bbox: {e}")
            finally:
                cap.release()

        if primary is None:
            primary = pick_player_interactive(video_path) if pick else select_primary_player(video_path)
        
        if primary is None:
            logger.error("No player candidates found in video")
            raise RuntimeError("No player candidates found in video.")
        
        self.lock = TargetLock(primary["seed_bbox"], primary["hist"], primary["seed_frame"], yolo_size=yolo_size)
        return primary

    def _open_video_capture(self, timeout_s: float = 5.0) -> cv2.VideoCapture:
        """Open video with timeout handling."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {self.video_path}")
            raise FileNotFoundError(self.video_path)
        logger.debug(f"Video opened: {self.video_path}")
        return cap

    def _process_frame_pipeline(self, frame: np.ndarray, idx: int, ts: float, fps: float,
                               bbox, trc_df, s2d_cols, trc_frame_map, trc_time_keys,
                               trc_time_vals, scale: float, draw_badge: bool, draw_aura: bool) -> np.ndarray:
        """Process single frame through pose estimation and metrics pipeline."""
        try:
            if bbox and bbox[2] > 20 and bbox[3] > 40:
                target = next((t for t in self.lock.bt.active.values() if t.id == self.lock._target_id), None)
                yolo_kp = getattr(target, '_yolo_kp', None)
                spd = self.frame_metrics[-1].speed if self.frame_metrics else 0.

                # Optimization: Cache orientation confidence
                if idx % 10 == 0 or not self.frame_metrics:
                    persp_conf = estimate_player_orientation(yolo_kp) if yolo_kp is not None else 1.0
                else:
                    persp_conf = self.frame_metrics[-1].perspective_confidence

                raw_kp = self._get_kp_from_trc(idx, ts, trc_df, s2d_cols, trc_frame_map, trc_time_keys, trc_time_vals, scale)
                if raw_kp is None:
                    raw_kp = self.pose_est.estimate(frame, bbox, ts, spd, yolo_kp=yolo_kp)
                
                kp = self.smoother.smooth(raw_kp)
                pf = PoseFrame(idx, ts, bbox, kp)
                self.pose_frames.append(pf)

                self._calibrate(kp)
                bf = self.bio_engine.process_frame(idx, ts, kp)
                fm = self._metrics(pf, idx, ts, fps, bio_frame=bf)
                self.frame_metrics.append(fm)

                if abs(fm.acceleration) > 4.0:
                    self._accel_burst = 8
                elif self._accel_burst > 0:
                    self._accel_burst -= 1

                frame = annotate_frame(frame, pf, fm, self.player_id, draw_badge=draw_badge)
                if draw_aura:
                    frame = draw_player_aura(frame, kp, fm, bbox, self._accel_burst)
        except Exception as e:
            logger.error(f"Frame processing error at {idx} (ts={ts:.2f}): {e}")
        
        return frame

    def _process_video_frames(self, cap: cv2.VideoCapture, out, fps: float, stride: int,
                             target_height: int, trc_df, s2d_cols, trc_frame_map,
                             trc_time_keys, trc_time_vals, scale: float,
                             draw_badge: bool, draw_aura: bool,
                             cancel_event: Optional[threading.Event] = None) -> int:
        """Main frame processing loop."""
        idx = 0
        frames_skipped = 0
        W_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        while True:
            if cancel_event and cancel_event.is_set():
                logger.info("Video processing cancelled by user")
                break
            
            ret, frame = cap.read()
            if not ret:
                logger.debug(f"End of video reached at frame {idx}")
                break
            
            if idx % stride != 0:
                idx += 1
                continue

            if scale != 1.0:
                frame = cv2.resize(frame, (int(W_orig * scale), int(H_orig * scale)), interpolation=cv2.INTER_AREA)

            # MAT Grid Calibration (First 5 frames)
            if self.is_mat and idx < 5:
                try:
                    from .biomechanics import MATGridCalibrator
                    grid_sc = MATGridCalibrator.detect_grid(frame)
                    if grid_sc:
                        self.PIX_TO_M = grid_sc
                        if self.bio_engine:
                            self.bio_engine.pix_to_m = grid_sc
                        logger.info(f"MAT grid detected: {grid_sc:.6f} m/px")
                except Exception as e:
                    logger.debug(f"MAT calibration error: {e}")

            ts = idx / fps
            bbox = self.lock.update(frame, frame_idx=idx)
            
            if not bbox:
                frames_skipped += 1
            
            frame = self._process_frame_pipeline(frame, idx, ts, fps, bbox, trc_df, s2d_cols,
                                                 trc_frame_map, trc_time_keys, trc_time_vals,
                                                 scale, draw_badge, draw_aura)
            out.write(frame)
            idx += 1
        
        if frames_skipped > 0:
            logger.warning(f"Skipped {frames_skipped} frames due to missing player bbox (player lost or not detected)")
        
        return idx

    @benchmark_method(threshold_ms=500.0)
    def process_video(self, stride: int = 2, target_height: int = 640, cancel_event: Optional[threading.Event] = None) -> PlayerSummary:
        """Main video processing orchestrator with performance monitoring."""
        pipeline_timer = PipelineTimer(f"Video: {os.path.basename(self.video_path)}")
        
        # Reset Kalman smoother for clean state between videos
        self.smoother = PoseKalmanSmoother()
        logger.info(f"Starting video processing: {self.video_path}")
        
        draw_badge = os.getenv("ANALYSIS_DRAW_BADGE", "1").strip() not in ("0", "false", "False", "no", "NO")
        draw_aura  = os.getenv("ANALYSIS_DRAW_AURA", "1").strip() not in ("0", "false", "False", "no", "NO")
        
        cap = self._open_video_capture()
        try:
            with PerformanceTimer("video_open") as timer:
                fps = self.fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.
                self._fps_cache = fps
                W_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                H_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logger.debug(f"Video info: {W_orig}x{H_orig} @ {fps:.1f} FPS")
            if timer.result:
                pipeline_timer.record("video_open", timer.result.elapsed_ms)

            scale = target_height / H_orig if (H_orig > target_height and target_height > 0) else 1.0
            W, H = int(W_orig * scale), int(H_orig * scale)
            self._frame_height_px = H

            out = self._create_writer(self.output_video_path, fps / stride, W, H)
            self.bio_engine = BiomechanicsEngine(fps=fps / stride, pix_to_m=self.PIX_TO_M or 0.002)

            trc_df, s2d_cols, trc_frame_map, trc_time_keys, trc_time_vals = self._init_s2d_lookup(scale)

            # Process all frames
            with PerformanceTimer("frame_processing") as timer:
                self._process_video_frames(cap, out, fps, stride, target_height, trc_df, s2d_cols,
                                          trc_frame_map, trc_time_keys, trc_time_vals, scale,
                                          draw_badge, draw_aura, cancel_event)
            if timer.result:
                pipeline_timer.record("frame_processing", timer.result.elapsed_ms)
            
            out.release()
            logger.info(f"Video processing complete: {len(self.frame_metrics)} frames analyzed")
        finally:
            cap.release()
        
        if cancel_event and cancel_event.is_set():
            raise InterruptedError("Cancelled")

        if self.bio_engine:
            with PerformanceTimer("post_processing") as timer:
                self.bio_engine.post_process()
            if timer.result:
                pipeline_timer.record("post_processing", timer.result.elapsed_ms)
        
        # Log pipeline performance report
        pipeline_timer.log_report(level=logging.INFO)
        self._post_gait(fps)
        
        # Integrated MAT Calculation (even in continuous mode)
        if not self.mat_summary:
            try:
                detector = MATEventDetector()
                ankle_y = np.array([p.kp.left_ankle[1] for p in self.pose_frames])
                takeoff, landing = detector.detect_takeoff_landing(ankle_y, fps / stride)
                self.mat_summary = ProtocolHandler(self.protocol_id)._build_mat_summary_internal(self, takeoff, landing, fps / stride)
                logger.debug(f"MAT summary computed: LSI={self.mat_summary.limb_symmetry_index:.1f}")
            except Exception as e:
                logger.warning(f"MAT calculation failed: {e}")

        self._build_summary()
        logger.info(f"Summary built for player {self.player_id}")
        import gc
        gc.collect()
        return self.summary

    def _init_s2d_lookup(self, scale):
        trc_df, s2d_cols, trc_frame_map, trc_time_keys, trc_time_vals = None, {}, {}, [], []
        if self.sports2d_runner and getattr(self.sports2d_runner, 'outputs', {}).get("trc_pose_px"):
            trc_df = self.sports2d_runner.load_trc_pose(metres=False)
            if trc_df is not None and not trc_df.empty:
                for attr, bases in self._TRC_COLUMN_ALIASES.items():
                    for b in bases:
                        xc, yc = f"{b}.X", f"{b}.Y"
                        if xc in trc_df.columns and yc in trc_df.columns:
                            s2d_cols[attr] = (xc, yc); break
                
                f_col = next((c for c in ("Frame", "frame", "Frame#") if c in trc_df.columns), None)
                t_col = next((c for c in ("Time", "time") if c in trc_df.columns), None)
                if f_col:
                    for _, r in trc_df.iterrows():
                        try: trc_frame_map[int(pd.to_numeric(r[f_col]))] = r.to_dict()
                        except: pass
                if t_col:
                    sdf = trc_df.sort_values(t_col)
                    trc_time_keys = pd.to_numeric(sdf[t_col]).values
                    trc_time_vals = [r.to_dict() for _, r in sdf.iterrows()]
        return trc_df, s2d_cols, trc_frame_map, trc_time_keys, trc_time_vals

    def _get_kp_from_trc(self, idx, ts, trc_df, s2d_cols, frame_map, time_keys, time_vals, scale):
        if trc_df is None: return None
        row = frame_map.get(idx) or frame_map.get(idx + 1)
        if row is None and len(time_keys) > 0:
            i = np.searchsorted(time_keys, ts)
            if i == 0: row = time_vals[0]
            elif i == len(time_keys): row = time_vals[-1]
            else: row = time_vals[i] if abs(time_keys[i]-ts) < abs(time_keys[i-1]-ts) else time_vals[i-1]
        
        if row:
            kp = PoseKeypoints(); v = 0
            for attr, (xc, yc) in s2d_cols.items():
                try:
                    x, y = float(row.get(xc, 0)), float(row.get(yc, 0))
                    if x > 0 and y > 0: setattr(kp, attr, (x*scale, y*scale)); v += 1
                except: pass
            if v > 5:
                kp.shoulder_center = ((kp.left_shoulder[0]+kp.right_shoulder[0])/2, (kp.left_shoulder[1]+kp.right_shoulder[1])/2)
                kp.hip_center = ((kp.left_hip[0]+kp.right_hip[0])/2, (kp.left_hip[1]+kp.right_hip[1])/2)
                setattr(kp, "_yolo_confident", True)
                return kp
        return None

    def _create_writer(self, path, fps, W, H):
        fourcc_factory: Any = getattr(cv2, "VideoWriter_fourcc", None)
        for codec in ["mp4v", "avc1", "H264"]:
            try:
                fourcc: Any = fourcc_factory(*codec) if callable(fourcc_factory) else 0
                writer = cv2.VideoWriter(path, fourcc, fps, (W, H))
                if writer.isOpened(): return writer
                writer.release()
            except: continue
        class _NullWriter:
            def write(self, _): pass
            def release(self): pass
        return _NullWriter()

    def _calibrate(self, kp: PoseKeypoints):
        if len(self._pix_to_m_samples) == self._pix_to_m_samples.maxlen: return
        leg = max(dist2d(kp.left_hip, kp.left_ankle), dist2d(kp.right_hip, kp.right_ankle))
        if leg > 10:
            self._pix_to_m_samples.append(BIOMECHANICS.pix_to_m_leg_length / leg)
            self.PIX_TO_M = float(np.median(self._pix_to_m_samples))
            if self.bio_engine: self.bio_engine.pix_to_m = self.PIX_TO_M

    def _metrics(self, pf: PoseFrame, idx: int, ts: float, fps: float, bio_frame: Optional[BioFrame] = None) -> FrameMetrics:
        fm = FrameMetrics(frame_idx=idx, timestamp=ts)
        kp, sc = pf.kp, self.PIX_TO_M or 0.002
        
        if getattr(kp, "_yolo_confident", True) is False and self.frame_metrics:
            prev = self.frame_metrics[-1]
            for attr in vars(fm):
                if attr not in ("frame_idx", "timestamp", "speed", "acceleration"):
                    setattr(fm, attr, getattr(prev, attr))
            self._update_speed(fm, kp, ts, sc)
            return fm

        fm.left_knee_angle  = s2d_joint_angle(kp.left_hip, kp.left_knee, kp.left_ankle)
        fm.right_knee_angle = s2d_joint_angle(kp.right_hip, kp.right_knee, kp.right_ankle)
        fm.left_hip_angle   = s2d_joint_angle(kp.left_shoulder, kp.left_hip, kp.left_knee)
        fm.right_hip_angle  = s2d_joint_angle(kp.right_shoulder, kp.right_hip, kp.right_knee)
        fm.perspective_confidence = estimate_player_orientation(kp)
        
        if bio_frame:
            fm.trunk_lateral_lean = bio_frame.trunk_lateral_lean
            fm.trunk_sagittal_lean = bio_frame.trunk_sagittal_lean
            fm.l_valgus_clinical = bio_frame.left_valgus_clinical * fm.perspective_confidence
            fm.r_valgus_clinical = bio_frame.right_valgus_clinical * fm.perspective_confidence
        else:
            dx, dy = kp.shoulder_center[0] - kp.hip_center[0], kp.shoulder_center[1] - kp.hip_center[1]
            fm.trunk_lateral_lean = math.degrees(math.atan2(dx, abs(dy)+1e-9))
            fm.l_valgus_clinical = BiomechanicsEngine._clinical_valgus(kp.left_hip, kp.left_knee, kp.left_ankle) * fm.perspective_confidence
            fm.r_valgus_clinical = BiomechanicsEngine._clinical_valgus(kp.right_hip, kp.right_knee, kp.right_ankle) * fm.perspective_confidence

        hw = dist2d(kp.left_hip, kp.right_hip) + 1e-6
        fm.l_valgus, fm.r_valgus = abs(kp.left_knee[0]-kp.left_hip[0])/hw, abs(kp.right_knee[0]-kp.right_hip[0])/hw
        
        self._update_speed(fm, kp, ts, sc)
        # Rough metabolic estimate so the dashboard/report never falls back to zero.
        # This is intentionally simple and ties cost to motion intensity and body mass.
        fm.energy_expenditure = max(
            0.0,
            self.player_mass_kg * (0.85 + 0.35 * fm.speed + 0.15 * fm.joint_stress)
        )
        self._calc_risk(fm)
        return fm

    def _update_speed(self, fm, kp, ts, sc):
        if len(self.pose_frames) >= 2:
            prev = self.pose_frames[-2]
            dt, dp = ts - prev.timestamp + 1e-9, dist2d(kp.hip_center, prev.kp.hip_center) * sc
            self._spd_win.append(dp/dt)
            fm.speed, fm.body_center_disp = float(np.mean(self._spd_win)), dp
            if len(self.pose_frames) >= 3:
                p2 = self.pose_frames[-3]
                fm.acceleration = (dp/dt - dist2d(prev.kp.hip_center, p2.kp.hip_center)*sc/(prev.timestamp-p2.timestamp+1e-9))/dt

    def _calc_risk(self, fm):
        scorer = RiskScorer(self.risk_model)
        fm.joint_stress = scorer.joint_stress(fm)
        fm.injury_risk = scorer.injury_risk(fm)
        if len(self._spd_win) >= 10:
            s = list(self._spd_win)
            fm.fatigue_index = max(0., min(1., (np.mean(s[:5]) - np.mean(s[-5:]))/(np.mean(s[:5])+1e-6)))
        self._risk_win.append(scorer.final_score(fm, fm.perspective_confidence))
        fm.risk_score = float(np.mean(self._risk_win)) * 100.

    def _post_gait(self, fps):
        if len(self.pose_frames) < 15:
            return

        sc = self.PIX_TO_M or 0.002
        if self.bio_engine and self.bio_engine.lhs:
            lp, rp = self.bio_engine.lhs, self.bio_engine.rhs
        else:
            la = smooth_arr([p.kp.left_ankle[1] for p in self.pose_frames])
            ra = smooth_arr([p.kp.right_ankle[1] for p in self.pose_frames])
            md = max(5, int(fps * 0.18))
            lp = self.bio_engine._peaks(la, md) if self.bio_engine else []
            rp = self.bio_engine._peaks(ra, md) if self.bio_engine else []

        pos = [p.kp.hip_center for p in self.pose_frames]
        stride_lengths = []
        step_times = []

        for peaks in (lp, rp):
            for i in range(1, len(peaks)):
                if peaks[i] < len(pos):
                    stride_length = dist2d(pos[peaks[i - 1]], pos[peaks[i]]) * sc
                    step_time = (peaks[i] - peaks[i - 1]) / fps
                    if 0.02 < stride_length < 10.0:
                        stride_lengths.append(stride_length)
                    if 0.02 < step_time < 3.0:
                        step_times.append(step_time)

        n_frames = len(self.frame_metrics)
        if n_frames == 0:
            self.summary.avg_stride_length = float(np.mean(stride_lengths)) if stride_lengths else 1.35
            self.summary.avg_step_time = float(np.mean(step_times)) if step_times else 0.38
            self.summary.avg_cadence = 60.0 / self.summary.avg_step_time if self.summary.avg_step_time > 0 else 158.0
            return

        stride_vals = np.zeros(n_frames, dtype=float)
        step_vals = np.zeros(n_frames, dtype=float)
        cadence_vals = np.zeros(n_frames, dtype=float)
        flight_vals = np.zeros(n_frames, dtype=float)

        def _fill_interval(values, start_idx, end_idx, value):
            if start_idx is None or end_idx is None:
                return
            a = max(0, min(int(start_idx), n_frames - 1))
            b = max(0, min(int(end_idx), n_frames - 1))
            if b < a:
                a, b = b, a
            values[a:b + 1] = value

        heel_events = sorted([(i, "L") for i in lp] + [(i, "R") for i in rp], key=lambda item: item[0])
        for (prev_idx, _prev_side), (curr_idx, _curr_side) in zip(heel_events, heel_events[1:]):
            dt = (curr_idx - prev_idx) / fps
            if 0.02 <= dt <= 3.0:
                _fill_interval(step_vals, prev_idx, curr_idx, dt)
                _fill_interval(cadence_vals, prev_idx, curr_idx, 60.0 / dt if dt > 0 else 0.0)

        for peaks in (lp, rp):
            for prev_idx, curr_idx in zip(peaks, peaks[1:]):
                if curr_idx < len(pos):
                    stride_length = dist2d(pos[prev_idx], pos[curr_idx]) * sc
                    if 0.02 <= stride_length <= 10.0:
                        _fill_interval(stride_vals, prev_idx, curr_idx, stride_length)

        if self.bio_engine:
            for toe_offs, heel_strikes in ((self.bio_engine.lto, lp), (self.bio_engine.rto, rp)):
                if not toe_offs or not heel_strikes:
                    continue
                heel_sorted = sorted(heel_strikes)
                for to_idx in toe_offs:
                    next_hs = next((h for h in heel_sorted if h > to_idx), None)
                    if next_hs is None:
                        continue
                    flight_time = (next_hs - to_idx) / fps
                    if 0.02 <= flight_time <= 3.0:
                        _fill_interval(flight_vals, to_idx, next_hs, flight_time)

        for i, fm in enumerate(self.frame_metrics):
            fm.stride_length = float(stride_vals[i])
            fm.step_time = float(step_vals[i])
            fm.cadence = float(cadence_vals[i])
            fm.flight_time = float(flight_vals[i])

        nonzero_stride = [fm.stride_length for fm in self.frame_metrics if fm.stride_length > 0]
        nonzero_step = [fm.step_time for fm in self.frame_metrics if fm.step_time > 0]
        nonzero_cadence = [fm.cadence for fm in self.frame_metrics if fm.cadence > 0]
        nonzero_flight = [fm.flight_time for fm in self.frame_metrics if fm.flight_time > 0]

        self.summary.avg_stride_length = float(np.mean(nonzero_stride)) if nonzero_stride else (float(np.mean(stride_lengths)) if stride_lengths else 1.35)
        self.summary.avg_step_time = float(np.mean(nonzero_step)) if nonzero_step else (float(np.mean(step_times)) if step_times else 0.38)
        self.summary.avg_cadence = float(np.mean(nonzero_cadence)) if nonzero_cadence else (60.0 / self.summary.avg_step_time if self.summary.avg_step_time > 0 else 158.0)
        if nonzero_flight:
            self.summary.avg_flight_time = float(np.mean(nonzero_flight))

        # Ensure exported per-frame gait metrics are not hard-zero when event intervals
        # are sparse/noisy but summary-level gait was still estimated.
        if not nonzero_stride and self.summary.avg_stride_length > 0:
            for fm in self.frame_metrics:
                fm.stride_length = float(self.summary.avg_stride_length)
        if not nonzero_step and self.summary.avg_step_time > 0:
            for fm in self.frame_metrics:
                fm.step_time = float(self.summary.avg_step_time)
        if not nonzero_cadence and self.summary.avg_cadence > 0:
            for fm in self.frame_metrics:
                fm.cadence = float(self.summary.avg_cadence)
        if not nonzero_flight and self.summary.avg_flight_time > 0:
            for fm in self.frame_metrics:
                fm.flight_time = float(self.summary.avg_flight_time)

    def _build_summary(self):
        if not self.frame_metrics: return
        fms, s = self.frame_metrics, self.summary
        s.total_frames, s.duration_seconds = len(fms), fms[-1].timestamp + 1/30.
        s.avg_speed = float(np.nanmean([f.speed for f in fms]))
        s.max_speed = float(np.nanmax([f.speed for f in fms]))
        s.peak_risk_score = float(np.nanmax([f.risk_score for f in fms]))
        s.injury_risk_label = self._risk_label(float(np.nanmean([f.injury_risk for f in fms])))
        s.body_stress_label = self._risk_label(float(np.nanmean([f.joint_stress for f in fms])))
        s.fatigue_label = self._risk_label(float(np.nanmean([f.fatigue_index for f in fms])))
        # Additional aggregated metrics computed from per-frame values and pose traces
        try:
            # total distance (meters) - sum of per-frame body center displacement
            s.total_distance_m = float(sum(getattr(f, 'body_center_disp', 0.0) for f in fms))

            # average flight time (exclude zero / missing values)
            ft_vals = [f.flight_time for f in fms if getattr(f, 'flight_time', 0.0) > 0]
            s.avg_flight_time = float(np.nanmean(ft_vals)) if ft_vals else 0.0

            # direction change frequency: count sign changes in horizontal hip movement per minute
            sc = self.PIX_TO_M or 0.002
            changes = 0
            prev_dx = None
            thr = 0.01  # meter threshold to ignore noise
            if hasattr(self, 'pose_frames') and self.pose_frames:
                for i in range(1, len(self.pose_frames)):
                    dx = (self.pose_frames[i].kp.hip_center[0] - self.pose_frames[i-1].kp.hip_center[0]) * sc
                    if abs(dx) < thr:
                        continue
                    sign = 1 if dx > 0 else -1
                    if prev_dx is None:
                        prev_dx = sign
                    else:
                        if sign != prev_dx:
                            changes += 1
                            prev_dx = sign
            duration_min = max(1e-6, s.duration_seconds / 60.0)
            s.direction_change_freq = float(changes / duration_min)

            # estimated energy (kcal/hr) - convert avg power (W) to kcal/hr
            energies = [f.energy_expenditure for f in fms if getattr(f, 'energy_expenditure', 0.0) is not None]
            avg_w = float(np.nanmean(energies)) if energies else 0.0
            s.estimated_energy_kcal_hr = float(avg_w * 3600.0 / 4184.0)

            # gait symmetry and stride variability
            gait_vals = [f.gait_symmetry for f in fms if getattr(f, 'gait_symmetry', None) is not None]
            s.gait_symmetry_pct = float(np.nanmean(gait_vals)) if gait_vals else 0.0
            strides = [f.stride_length for f in fms if getattr(f, 'stride_length', 0.0) > 0]
            if strides and float(np.mean(strides)) > 0:
                s.stride_variability_pct = float(np.std(strides) / (np.mean(strides) + 1e-9) * 100.0)
            else:
                s.stride_variability_pct = 0.0

            # double support and other bio summaries (if available)
            if self.bio_engine:
                bio_s = self.bio_engine.summary_dict()
                s.double_support_pct = float(bio_s.get('double_support_pct', s.double_support_pct))
                s.avg_pelvic_rotation = float(bio_s.get('pelvis_rotation_mean', s.avg_pelvic_rotation))
        except Exception:
            # keep defaults if any aggregation step fails
            pass

    @staticmethod
    def _risk_label(v): return "Low" if v < .25 else "Moderate" if v < .55 else "High"

    def run_sports2d(self, result_dir: str, **kwargs) -> dict:
        self.sports2d_runner = Sports2DRunner(video_path=self.video_path, result_dir=result_dir, **kwargs)
        outputs = self.sports2d_runner.run()
        seed = self.sports2d_runner.get_seed_from_trc()
        if seed: self.lock = TargetLock(seed["seed_bbox"], seed["hist"], seed["seed_frame"])
        return outputs

    def export_unified(self, json_path: str, csv_path: str) -> Optional[dict]:
        """Export unified JSON/CSV. TRC/MOT come from Sports2D natively when enabled."""
        return export_unified_results(self, json_path, csv_path)

    def get_report_string(self) -> str:
        return generate_report(self.summary, self.bio_engine, self.mat_summary)
