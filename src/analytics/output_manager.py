"""OpenSim/TRC/MOT output writing utilities."""

import json
import os
import logging
import csv as _csv
from pathlib import Path
from typing import List, Optional
from dataclasses import asdict
from io import StringIO

import pandas as pd
import numpy as np

from .models import PoseFrame, BioFrame, BIO_ANGLE_FIELDS
from .math_utils import clean_nans
from .core import HAS_SPORTS2D, HAS_SCIPY

logger = logging.getLogger(__name__)


# TO DO 
# Remove the Opensim caalcultion as i get them in sports2S natively 

def _validate_output_path(path: str, base_dir: Optional[str] = None) -> str:
    """
    Validate that output path is safe and create parent directories if needed.
    If base_dir is provided, ensures path is within it.
    """
    try:
        target = Path(path).resolve()
        
        if base_dir:
            base = Path(base_dir).resolve()
            # Ensure target is within base directory
            target.relative_to(base)
            
        # Ensure parent directory exists
        target.parent.mkdir(parents=True, exist_ok=True)
        return str(target)
    except (ValueError, RuntimeError, OSError) as e:
        if base_dir:
            raise ValueError(
                f"Invalid output path '{path}': attempts to escape base directory '{base_dir}'"
            ) from e
        raise ValueError(f"Invalid output path '{path}': {e}") from e

class OpenSimFileWriter:
    """
    Generates valid OpenSim input files from tracked pose data.

    TRC format:
        Standard OpenSim Marker Trajectory (marker positions in metres, 3-D).
        We set Z=0 for all markers (monocular video → 2-D plane).
        Coordinate system: X = horizontal (right), Y = vertical (up, image Y inverted),
        Z = depth (out of plane, zero). This matches the standard OpenSim convention
        used by Sports2D / Pose2Sim.

    MOT format:
        OpenSim Motion file (tab-separated, header block).
        Stores joint angles in degrees, same convention as Sports2D.
    """

    # Subset of our joint names that map to standard OpenSim marker labels
    OPENSIM_MARKERS = [
        "head", "neck",
        "left_shoulder", "right_shoulder",
        "left_elbow",    "right_elbow",
        "left_wrist",    "right_wrist",
        "left_hip",      "right_hip",
        "left_knee",     "right_knee",
        "left_ankle",    "right_ankle",
        "left_foot",     "right_foot",
        "hip_center",    "shoulder_center",
    ]

    # Canonical OpenSim marker label mapping
    _LABEL_MAP = {
        "head":             "Head",
        "neck":             "Neck",
        "left_shoulder":    "L_Shoulder",
        "right_shoulder":   "R_Shoulder",
        "left_elbow":       "L_Elbow",
        "right_elbow":      "R_Elbow",
        "left_wrist":       "L_Wrist",
        "right_wrist":      "R_Wrist",
        "left_hip":         "L_Hip",
        "right_hip":        "R_Hip",
        "left_knee":        "L_Knee",
        "right_knee":       "R_Knee",
        "left_ankle":       "L_Ankle",
        "right_ankle":      "R_Ankle",
        "left_foot":        "L_Foot",
        "right_foot":       "R_Foot",
        "hip_center":       "Hip_Center",
        "shoulder_center":  "Shoulder_Center",
    }

    def write_trc(self, pose_frames: List[PoseFrame], path: str,
                  fps: float, pix_to_m: float, frame_height_px: int,
                  base_output_dir: Optional[str] = None) -> bool:
        """
        Write a .trc file with 3-D marker trajectories.

        Coordinate conversion from image (px) to OpenSim (m):
            X_osim =  x_px * pix_to_m          (right is positive)
            Y_osim =  (H - y_px) * pix_to_m    (Y flipped: up is positive)
            Z_osim =  0.0                       (monocular — no depth)
        """
        n_frames  = len(pose_frames)
        n_markers = len(self.OPENSIM_MARKERS)
        H         = frame_height_px

        if n_frames == 0:
            print("[TRC] No pose frames — skipping TRC export.")
            return False

        try:
            # Validate path to prevent traversal attacks
            validated_path = _validate_output_path(path, base_output_dir)
            with open(validated_path, "w", newline="\r\n") as f:
                # ── Header ────────────────────────────────────────────────────
                # Line 0: file-type header
                f.write(f"PathFileType\t4\t(X/Y/Z)\t{os.path.basename(path)}\n")
                # Line 1: field names
                f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\t"
                        "Units\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
                # Line 2: values
                f.write(f"{fps:.6f}\t{fps:.6f}\t{n_frames}\t{n_markers}\t"
                        f"m\t{fps:.6f}\t1\t{n_frames}\n")
                # Line 3: marker labels — Frame# Time M1 '' '' M2 '' '' ...
                labels_row = "Frame#\tTime"
                for nm in self.OPENSIM_MARKERS:
                    lbl = self._LABEL_MAP[nm]
                    labels_row += f"\t{lbl}\t\t"  # label + 2 empty for Y Z
                f.write(labels_row + "\n")
                # Line 4: X/Y/Z sub-headers
                xyz_row = "\t"
                for _ in self.OPENSIM_MARKERS:
                    xyz_row += "\tX\tY\tZ"
                f.write(xyz_row + "\n")

                # ── Data rows ─────────────────────────────────────────────────
                for i, pf in enumerate(pose_frames):
                    # Frame index is 1-based sequential counter (OpenSim expects this)
                    row = f"{i + 1}\t{pf.timestamp:.6f}"
                    for nm in self.OPENSIM_MARKERS:
                        px, py = getattr(pf.kp, nm)
                        x =  px * pix_to_m
                        y = (H - py) * pix_to_m  # flip Y: image Y↓ → OpenSim Y↑
                        z = 0.0
                        row += f"\t{x:.6f}\t{y:.6f}\t{z:.6f}"
                    f.write(row + "\n")
            print(f"[TRC] Written: {validated_path}  ({n_frames} frames, {n_markers} markers)")
            return True
        except ValueError as e:
            print(f"[TRC] Security error: {e}")
            return False
        except Exception as e:
            print(f"[TRC] Failed to write {path}: {e}")
            return False

    def write_mot(self, bio_frames: List[BioFrame], path: str, fps: float) -> bool:
        """
        Write a .mot (OpenSim Motion) file containing joint angles (degrees).

        The column ordering matches the standard Sports2D MOT output so the
        file can be loaded directly in OpenSim's Motion Visualizer or used
        as input to Inverse Kinematics.
        """
        if not bio_frames:
            print("[MOT] No biomechanics frames — skipping MOT export.")
            return False

        # Use shared fields list for consistency with BiomechanicsEngine
        angle_fields = BIO_ANGLE_FIELDS

        n_rows = len(bio_frames)
        n_cols = 1 + len(angle_fields)  # time + angles

        try:
            with open(path, "w", newline="\r\n") as f:
                # ── OpenSim MOT header ────────────────────────────────────────
                f.write(f"{os.path.basename(path)}\n")
                f.write("version=1\n")
                f.write(f"nRows={n_rows}\n")
                f.write(f"nColumns={n_cols}\n")
                f.write("inDegrees=yes\n")
                f.write("endheader\n")

                # ── Column header row ─────────────────────────────────────────
                header = "time\t" + "\t".join(angle_fields)
                f.write(header + "\n")

                # ── Data rows ─────────────────────────────────────────────────
                for bf in bio_frames:
                    row = f"{bf.timestamp:.6f}"
                    for field in angle_fields:
                        row += f"\t{getattr(bf, field):.6f}"
                    f.write(row + "\n")
            print(f"[MOT] Written: {path}  ({n_rows} rows, {len(angle_fields)} angles)")
            return True
        except Exception as e:
            print(f"[MOT] Failed to write {path}: {e}")
            return False


def export_unified_results(analyzer, json_path: str, csv_path: str,
                           trc_path: Optional[str] = None,
                           mot_path: Optional[str] = None):
    """Standalone result exporter for SportsAnalyzer results."""
    import csv as _csv
    import json
    from pathlib import Path
    from dataclasses import asdict
    from .math_utils import clean_nans

    export_jsonl = os.getenv("ANALYTICS_EXPORT_JSONL", "0").strip() in ("1", "true", "True", "yes", "YES")

    bio_by_frame: dict = {}
    if analyzer.bio_engine and analyzer.bio_engine.frames:
        for bf in analyzer.bio_engine.frames:
            bio_by_frame[bf.frame_idx] = asdict(bf)

    s2d_angle_summary: dict = {}
    s2d_pose_summary:  dict = {}
    s2d_mot_df: Optional[pd.DataFrame] = None
    
    if analyzer.sports2d_runner:
        s2d_mot_df = analyzer.sports2d_runner.load_mot_angles()
        if s2d_mot_df is not None and not s2d_mot_df.empty:
            angle_cols = [c for c in s2d_mot_df.columns if c.lower() != "time"]
            for col in angle_cols:
                try:
                    vals = pd.to_numeric(s2d_mot_df[col], errors="coerce").dropna()
                    if len(vals):
                        s2d_angle_summary[col] = {
                            "mean": float(vals.mean()),
                            "max":  float(vals.max()),
                            "min":  float(vals.min()),
                            "std":  float(vals.std()),
                        }
                except (ValueError, TypeError, KeyError):
                    pass
        trc_df = analyzer.sports2d_runner.load_trc_pose(metres=True)
        if trc_df is not None and not trc_df.empty:
            s2d_pose_summary["trc_shape"] = list(trc_df.shape)
            s2d_pose_summary["trc_columns"] = list(trc_df.columns)

    payload = {
        "metadata": {
            "player_id":   analyzer.player_id,
            "video_path":  analyzer.video_path,
            "fps":         analyzer._fps_cache,
            "pix_to_m":    analyzer.PIX_TO_M,
            "total_frames": len(analyzer.frame_metrics),
            "angle_backend": "sports2d" if HAS_SPORTS2D else "scipy" if HAS_SCIPY else "numpy",
        },
        "player_summary":   asdict(analyzer.summary),
        "mat_summary":      asdict(analyzer.mat_summary) if analyzer.mat_summary else None,
        "biomechanics_summary": analyzer.bio_engine.summary_dict() if analyzer.bio_engine else {},
        "sports2d_angle_summary": s2d_angle_summary,
        "sports2d_pose_summary":  s2d_pose_summary,
        "sports2d_output_files":  analyzer.sports2d_runner.outputs if analyzer.sports2d_runner else {},
    }

    payload = clean_nans(payload)

    mot_by_ts: dict[float, dict] = {}
    if s2d_mot_df is not None and not s2d_mot_df.empty and "time" in s2d_mot_df.columns:
        try:
            sdf = s2d_mot_df.rename(columns={"time": "timestamp"})
            sdf["timestamp"] = pd.to_numeric(sdf["timestamp"], errors="coerce").round(3)
            angle_cols = [c for c in sdf.columns if c != "timestamp"]
            for _, r in sdf.iterrows():
                ts = r.get("timestamp")
                if ts is None or (isinstance(ts, float) and np.isnan(ts)):
                    continue
                row = {}
                for c in angle_cols:
                    row[f"s2d_{c}"] = r.get(c)
                mot_by_ts[float(ts)] = row
        except (ValueError, TypeError, KeyError):
            mot_by_ts = {}

    n_frames = 0
    csv_f = None
    jf = None
    csv_writer = None
    csv_buffer = StringIO()  # Buffered CSV writing for better performance

    jsonl_path = str(Path(json_path).with_suffix(".jsonl"))
    out_json_path = jsonl_path if export_jsonl else json_path

    try:
        jf = open(out_json_path, "w", encoding="utf-8", newline="")  # Explicit newline for Windows compat
        csv_f = open(csv_path, "w", encoding="utf-8", newline="")    # Explicit newline for Windows compat

        if export_jsonl:
            jf.write(json.dumps(payload, default=str) + "\n")
        else:
            jf.write("{\n")
            keys = list(payload.keys())
            for k in keys:
                if k == "frames":
                    continue
                jf.write(json.dumps(k) + ": " + json.dumps(payload[k], indent=2, default=str) + ",\n")
            jf.write("\"frames\": [\n")

        for fm in analyzer.frame_metrics:
            record = asdict(fm)
            bio = bio_by_frame.get(fm.frame_idx, {})
            for k, v in bio.items():
                if k not in ("frame_idx", "timestamp"):
                    record[f"bio_{k}"] = v

            ts_key = round(float(record.get("timestamp", 0.0)), 3)
            extra = mot_by_ts.get(ts_key)
            if extra:
                record.update(extra)

            record = clean_nans(record)

            if csv_writer is None:
                csv_writer = _csv.DictWriter(csv_buffer, fieldnames=list(record.keys()), extrasaction="ignore")
                csv_writer.writeheader()
            csv_writer.writerow(record)

            if export_jsonl:
                jf.write(json.dumps(record, default=str) + "\n")
            else:
                if n_frames > 0:
                    jf.write(",\n")
                jf.write(json.dumps(record, default=str))

            n_frames += 1
        
        # Flush buffered CSV to actual file with consistent line endings
        if csv_writer is not None:
            csv_content = csv_buffer.getvalue()
            csv_f.write(csv_content)
            logger.info(f"Exported {n_frames} frames to {out_json_path} and {csv_path}")


        if export_jsonl:
            logger.info(f"Exported {n_frames} frames to {out_json_path}")
        else:
            jf.write("\n]\n}\n")
            print(f"[EXPORT] data_output.json → {out_json_path}  ({n_frames} frames)")

    finally:
        if jf is not None:
            try:
                jf.close()
            except Exception:
                pass
        if csv_f is not None:
            try:
                csv_f.close()
            except Exception:
                pass

    if (trc_path and analyzer.pose_frames) or (mot_path and analyzer.bio_engine and analyzer.bio_engine.frames):
        writer = OpenSimFileWriter()
        if trc_path and analyzer.pose_frames:
            writer.write_trc(
                pose_frames     = analyzer.pose_frames,
                path            = trc_path,
                fps             = analyzer._fps_cache,
                pix_to_m        = analyzer.PIX_TO_M or 0.002,
                frame_height_px = analyzer._frame_height_px or 720,
            )
        if mot_path and analyzer.bio_engine and analyzer.bio_engine.frames:
            writer.write_mot(
                bio_frames = analyzer.bio_engine.frames,
                path       = mot_path,
                fps        = analyzer._fps_cache,
            )

    return payload


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYTICS PLOTTER  — saves all plots to /results
# ══════════════════════════════════════════════════════════════════════════════
