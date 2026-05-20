"""
Sports2D execution and native output discovery (UPGRADED VERSION)

Improvements:
- Safe multiprocessing lifecycle
- Deadlock-free queue handling
- Fast filesystem scanning
- Memory-safe TRC parsing (streaming mode)
- Robust bbox handling
- Deterministic output classification
- Cleaner config normalization
"""

import os
import re
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

from .cv_wrapper import cv2
from .core import HAS_SPORTS2D, SPORTS2D_PROCESS
from .math_utils import crop_hist

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════
# PATH SAFETY
# ═════════════════════════════════════════════════════════════

def _validate_sports2d_result_path(path: str, base_dir: str) -> str:
    base = Path(base_dir).resolve()
    target = Path(path).resolve()
    target.relative_to(base)
    return str(target)


# ═════════════════════════════════════════════════════════════
# WORKER
# ═════════════════════════════════════════════════════════════

def _sports2d_worker(config: dict, q):
    try:
        if SPORTS2D_PROCESS is None:
            raise RuntimeError("Sports2D not installed properly")

        SPORTS2D_PROCESS(config)
        q.put({"ok": True})

    except Exception as e:
        import traceback
        q.put({
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        })


# ═════════════════════════════════════════════════════════════
# RUNNER
# ═════════════════════════════════════════════════════════════

class Sports2DRunner:

    JOINT_ANGLES = [
        "Right ankle", "Left ankle",
        "Right knee", "Left knee",
        "Right hip", "Left hip",
        "Right shoulder", "Left shoulder",
        "Right elbow", "Left elbow",
    ]

    SEGMENT_ANGLES = [
        "Right foot", "Left foot",
        "Right shank", "Left shank",
        "Right thigh", "Left thigh",
        "Pelvis", "Trunk", "Shoulders",
        "Right arm", "Left arm",
        "Right forearm", "Left forearm",
    ]

    def __init__(
        self,
        video_path: str,
        result_dir: str,
        player_height_m: float = 1.75,
        participant_mass_kg: float = 75.0,
        mode: str = "balanced",
        show_realtime: bool = False,
        person_ordering: str = "greatest_displacement",
        do_ik: bool = False,
        use_augmentation: bool = False,
        visible_side: str = "auto front",
    ):
        self.video_path = video_path
        self.result_dir = result_dir
        self.player_height_m = player_height_m
        self.participant_mass_kg = participant_mass_kg
        self.mode = mode
        self.show_realtime = show_realtime
        self.person_ordering = person_ordering
        self.do_ik = do_ik
        self.use_augmentation = use_augmentation
        self.visible_side = visible_side

        self.outputs: Dict[str, List[str]] = {}

    # ═════════════════════════════════════════════════════════
    # MAIN RUN
    # ═════════════════════════════════════════════════════════

    def run(self) -> dict:
        if not HAS_SPORTS2D:
            logger.warning("Sports2D not installed")
            return {}

        os.makedirs(self.result_dir, exist_ok=True)

        video_abs = str(Path(self.video_path).resolve())
        result_abs = str(Path(self.result_dir).resolve())

        # robust visible_side parsing
        vs = str(self.visible_side)
        vs_tokens = re.split(r"[ ,]+", vs.strip())
        visible_side_cfg = vs_tokens if vs_tokens else ["auto", "front"]

        config = self._build_config(video_abs, result_abs, visible_side_cfg)

        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=_sports2d_worker, args=(config, q), daemon=True)

        timeout_s = int(os.getenv("SPORTS2D_TIMEOUT_SEC", "1200"))

        logger.info(f"Sports2D start | video={video_abs} | timeout={timeout_s}s")

        p.start()
        p.join(timeout=timeout_s)

        if p.is_alive():
            logger.warning("Timeout reached → terminating Sports2D")
            p.terminate()
            p.join(5)
            if p.is_alive():
                p.kill()
                p.join()

        worker_result = None
        try:
            worker_result = q.get_nowait()
        except Exception:
            worker_result = None

        self.outputs = self._collect_outputs()

        if worker_result and not worker_result.get("ok"):
            logger.error(f"Sports2D failed: {worker_result.get('error')}")
            if worker_result.get("traceback"):
                logger.debug(worker_result["traceback"])
            return {}

        return self.outputs

    # ═════════════════════════════════════════════════════════
    # CONFIG BUILDER
    # ═════════════════════════════════════════════════════════

    def _build_config(self, video_abs, result_abs, visible_side_cfg):
        return {
            "base": {
                "video_input": video_abs,
                "video_dir": "",
                "result_dir": result_abs,
                "nb_persons_to_detect": 1,
                "person_ordering_method": self.person_ordering,
                "first_person_height": self.player_height_m,
                "visible_side": visible_side_cfg,
                "show_realtime_results": self.show_realtime,
                "save_vid": True,
                "save_pose": True,
                "save_angles": True,
            },
            "pose": {
                "pose_model": "Body_with_feet",
                "mode": self.mode,
                "det_frequency": int(os.getenv("S2D_DET_FREQ", "6")),
                "device": os.getenv("S2D_DEVICE", "auto"),
            },
            "angles": {
                "calculate_angles": True,
                "joint_angles": self.JOINT_ANGLES,
                "segment_angles": self.SEGMENT_ANGLES,
            },
            "post-processing": {
                "interpolate": True,
                "filter": True,
                "show_graphs": self.show_realtime,
            },
            "kinematics": {
                "do_ik": self.do_ik,
                "participant_mass": [self.participant_mass_kg],
                "default_height": self.player_height_m,
            },
        }

    # ═════════════════════════════════════════════════════════
    # OUTPUT COLLECTION (FAST)
    # ═════════════════════════════════════════════════════════

    def _collect_outputs(self) -> dict:
        VIDEO_EXT = {".mp4", ".avi"}
        IMG_EXT = {".png"}
        TRC_EXT = {".trc"}
        MOT_EXT = {".mot"}
        TOML_EXT = {".toml"}
        C3D_EXT = {".c3d"}
        OSIM_EXT = {".osim"}
        XML_EXT = {".xml"}

        out = {
            "annotated_video": [],
            "angle_plots_png": [],
            "trc_pose_px": [],
            "trc_pose_m": [],
            "mot_angles": [],
            "calib_toml": [],
            "c3d": [],
            "osim_model": [],
            "osim_mot": [],
            "osim_setup": [],
            "all": [],
        }

        for root, _, files in os.walk(self.result_dir):
            for name in files:
                path = str(Path(root) / name)

                try:
                    validated = _validate_sports2d_result_path(
                        path, self.result_dir
                    )
                except ValueError:
                    continue

                out["all"].append(validated)
                ext = Path(validated).suffix.lower()

                if ext in VIDEO_EXT:
                    if "_h264" not in name:
                        out["annotated_video"].append(validated)

                elif ext in IMG_EXT:
                    out["angle_plots_png"].append(validated)

                elif ext in TRC_EXT:
                    if "_px" in name:
                        out["trc_pose_px"].append(validated)
                    else:
                        out["trc_pose_m"].append(validated)

                elif ext in MOT_EXT:
                    out["mot_angles"].append(validated)

                elif ext in TOML_EXT:
                    out["calib_toml"].append(validated)

                elif ext in C3D_EXT:
                    out["c3d"].append(validated)

                elif ext in OSIM_EXT:
                    out["osim_model"].append(validated)

                elif ext in XML_EXT:
                    out["osim_setup"].append(validated)

        return out

    # ═════════════════════════════════════════════════════════
    # SEED EXTRACTION (STREAMING - NO PANDAS)
    # ═════════════════════════════════════════════════════════

    def get_seed_from_trc(self) -> Optional[dict]:
        trcs = self.outputs.get("trc_pose_px", [])
        if not trcs:
            return None

        path = trcs[0]

        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            data_start = None
            for i, l in enumerate(lines):
                if i > 4 and l.strip() and l[0].isdigit():
                    data_start = i
                    break

            if data_start is None:
                return None

            header = lines[data_start - 2].strip().split("\t")
            values = lines[data_start].strip().split("\t")

            row = dict(zip(header, values))

            xs, ys = [], []

            for k, v in row.items():
                if ".X" in k or ".Y" in k:
                    try:
                        val = float(v)
                        if ".X" in k:
                            xs.append(val)
                        else:
                            ys.append(val)
                    except:
                        pass

            if not xs or not ys:
                return None

            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            w = max(40, max_x - min_x)
            h = max(80, max_y - min_y)

            pad_x = w * 0.15
            pad_y = h * 0.10

            bx = max(0, int(min_x - pad_x))
            by = max(0, int(min_y - pad_y))
            bw = int(w + 2 * pad_x)
            bh = int(h + 2 * pad_y)

            bbox = (bx, by, bw, bh)

            hist = None
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    hist = crop_hist(frame, bbox)
            cap.release()

            return {
                "seed_bbox": bbox,
                "seed_frame": 0,
                "hist": hist,
            }

        except Exception as e:
            logger.error(f"seed extraction failed: {e}")
            return None

    # ═════════════════════════════════════════════════════════
    # MOT LOADER
    # ═════════════════════════════════════════════════════════

    def load_mot_angles(self) -> Optional[pd.DataFrame]:
        mots = self.outputs.get("mot_angles", [])
        if not mots:
            return None

        try:
            return pd.read_csv(
                mots[0],
                sep="\t",
                encoding="utf-8",
                on_bad_lines="skip",
            )
        except Exception as e:
            logger.error(f"MOT load failed: {e}")
            return None

    # ═════════════════════════════════════════════════════════
    # TRC POSE LOADER
    # ═════════════════════════════════════════════════════════

    def load_trc_pose(self, metres: bool = True) -> Optional[pd.DataFrame]:
        """
        Load TRC (motion capture) pose data from Sports2D output.
        
        Args:
            metres: If True, load trc_pose_m; if False, load trc_pose_px
        
        Returns:
            DataFrame with pose keypoints or None if not available
        """
        key = "trc_pose_m" if metres else "trc_pose_px"
        trcs = self.outputs.get(key, [])
        
        if not trcs:
            return None
        
        try:
            path = trcs[0]
            
            # Read TRC file, skipping the header lines
            # TRC files have a specific format with metadata in first few lines
            with open(path, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            
            # Find where the actual data starts (first line with numeric frame number)
            data_start = None
            for i, line in enumerate(lines):
                if i > 4 and line.strip() and line.strip()[0].isdigit():
                    data_start = i
                    break
            
            if data_start is None:
                logger.warning(f"Could not find data start in TRC file: {path}")
                return None
            
            # Read from the data start line, using the header line before it
            header_line = lines[data_start - 1].strip()
            header = header_line.split("\t")
            
            # Read remaining lines as data
            data_lines = lines[data_start:]
            
            # Parse data into rows
            rows = []
            for line in data_lines:
                if not line.strip():
                    continue
                values = line.strip().split("\t")
                if len(values) >= len(header):
                    rows.append(dict(zip(header, values)))
            
            if not rows:
                logger.warning(f"No data rows found in TRC file: {path}")
                return None
            
            df = pd.DataFrame(rows)
            
            # Convert numeric columns
            for col in df.columns:
                if col not in ("Frame", "frame", "Frame#"):
                    try:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    except:
                        pass
            
            logger.debug(f"Loaded TRC pose with {len(df)} frames from {path}")
            return df
            
        except Exception as e:
            logger.error(f"TRC pose load failed: {e}")
            return None