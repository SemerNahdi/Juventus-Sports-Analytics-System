import csv
import gzip
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

from .core import HAS_SCIPY, HAS_SPORTS2D
from .math_utils import clean_nans

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# PATH VALIDATION
# ════════════════════════════════════════════════════════════════════════

def validate_output_path(path: str) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


# ════════════════════════════════════════════════════════════════════════
# SPORTS2D HELPERS
# ════════════════════════════════════════════════════════════════════════

def build_sports2d_summaries(analyzer: Any) -> Tuple[Dict, Dict, Optional[pd.DataFrame]]:
    angle_summary, pose_summary, mot_df = {}, {}, None

    if not analyzer.sports2d_runner:
        return angle_summary, pose_summary, mot_df

    try:
        mot_df = analyzer.sports2d_runner.load_mot_angles()

        if mot_df is not None and not mot_df.empty:
            angle_cols = [c for c in mot_df.columns if c.lower() != "time"]

            for col in angle_cols:
                vals = pd.to_numeric(mot_df[col], errors="coerce").dropna()
                if len(vals):
                    angle_summary[col] = {
                        "mean": float(vals.mean()),
                        "max": float(vals.max()),
                        "min": float(vals.min()),
                        "std": float(vals.std()),
                    }

    except Exception as e:
        logger.warning(f"MOT load failed: {e}")

    try:
        trc_df = analyzer.sports2d_runner.load_trc_pose(metres=True)
        if trc_df is not None and not trc_df.empty:
            pose_summary["trc_shape"] = list(trc_df.shape)
            pose_summary["trc_columns"] = list(trc_df.columns)
    except Exception as e:
        logger.warning(f"TRC load failed: {e}")

    return angle_summary, pose_summary, mot_df


# ════════════════════════════════════════════════════════════════════════
# MOT LOOKUP
# ════════════════════════════════════════════════════════════════════════

def build_mot_lookup(mot_df: Optional[pd.DataFrame]) -> Dict[int, Dict[str, Any]]:
    if mot_df is None or mot_df.empty:
        return {}

    try:
        df = mot_df
        if "frame_idx" not in df.columns:
            df = df.copy()
            df["frame_idx"] = np.arange(len(df))

        angle_cols = [c for c in df.columns if c not in ("time", "timestamp", "frame_idx")]

        lookup = {}
        for row in df.itertuples(index=False):
            lookup[int(row.frame_idx)] = {
                f"s2d_{c}": getattr(row, c) for c in angle_cols
            }

        return lookup

    except Exception as e:
        logger.warning(f"MOT lookup failed: {e}")
        return {}


# ════════════════════════════════════════════════════════════════════════
# MAIN EXPORT
# ════════════════════════════════════════════════════════════════════════
def export_unified_results(
    analyzer: Any,
    json_path: str,
    csv_path: str,
    jsonl: bool = False,
    compress_csv: bool = False,
    progress_interval: int = 1000,
) -> Optional[Dict]:

    jsonl = os.getenv("ANALYTICS_EXPORT_JSONL", str(jsonl)).lower() in ("1", "true", "yes")
    compress_csv = os.getenv("ANALYTICS_COMPRESS_CSV", str(compress_csv)).lower() in ("1", "true", "yes")

    if not analyzer.frame_metrics:
        logger.warning("No frame metrics - returning empty payload")
        # Return a minimal payload instead of None
        return {
            "metadata": {
                "player_id": analyzer.player_id,
                "video_path": analyzer.video_path,
                "fps": analyzer._fps_cache,
                "pix_to_m": analyzer.PIX_TO_M,
                "total_frames": 0,
                "angle_backend": "sports2d" if HAS_SPORTS2D else "scipy" if HAS_SCIPY else "numpy",
            },
            "player_summary": vars(analyzer.summary),
            "mat_summary": vars(analyzer.mat_summary) if analyzer.mat_summary else None,
            "biomechanics_summary": analyzer.bio_engine.summary_dict() if analyzer.bio_engine else {},
            "sports2d_angle_summary": {},
            "sports2d_pose_summary": {},
            "sports2d_output_files": analyzer.sports2d_runner.outputs if analyzer.sports2d_runner else {},
            "frames": [],
        }

    total_frames = len(analyzer.frame_metrics)
    logger.info(f"Exporting {total_frames} frames...")

    # ─────────────────────────────
    # BIO LOOKUP
    # ─────────────────────────────
    bio_lookup = {
        bf.frame_idx: vars(bf).copy()
        for bf in (analyzer.bio_engine.frames if analyzer.bio_engine else [])
    }

    # ─────────────────────────────
    # SPORTS2D
    # ─────────────────────────────
    s2d_angle_summary, s2d_pose_summary, mot_df = build_sports2d_summaries(analyzer)
    mot_lookup = build_mot_lookup(mot_df)

    # ─────────────────────────────
    # PAYLOAD
    # ─────────────────────────────
    payload = clean_nans({
        "metadata": {
            "player_id": analyzer.player_id,
            "video_path": analyzer.video_path,
            "fps": analyzer._fps_cache,
            "pix_to_m": analyzer.PIX_TO_M,
            "total_frames": total_frames,
            "angle_backend": "sports2d" if HAS_SPORTS2D else "scipy" if HAS_SCIPY else "numpy",
        },
        "player_summary": vars(analyzer.summary),
        "mat_summary": vars(analyzer.mat_summary) if analyzer.mat_summary else None,
        "biomechanics_summary": analyzer.bio_engine.summary_dict() if analyzer.bio_engine else {},
        "sports2d_angle_summary": s2d_angle_summary,
        "sports2d_pose_summary": s2d_pose_summary,
        "sports2d_output_files": analyzer.sports2d_runner.outputs if analyzer.sports2d_runner else {},
    })

    # ─────────────────────────────
    # OUTPUT PATHS
    # ─────────────────────────────
    json_target = validate_output_path(json_path)
    if jsonl and not json_target.suffix == ".jsonl":
        json_target = json_target.with_suffix(".jsonl")

    csv_target = validate_output_path(csv_path)
    if compress_csv:
        csv_target = csv_target.with_suffix(csv_target.suffix + ".gz")

    opener = gzip.open if compress_csv else open

    # ─────────────────────────────
    # STREAMING EXPORT (NO MEMORY BLOAT)
    # ─────────────────────────────

    logger.info(f"Writing JSON → {json_target}")
    logger.info(f"Writing CSV → {csv_target}")

    bio_engine = analyzer.bio_engine
    frame_metrics = analyzer.frame_metrics

    csv_writer = None
    fieldnames = None

    with open(json_target, "w", encoding="utf-8") as jf, \
         opener(csv_target, "wt", encoding="utf-8", newline="") as cf:

        # JSON header
        if jsonl:
            jf.write(json.dumps(payload, default=str, ensure_ascii=False) + "\n")
        else:
            jf.write(json.dumps({**payload, "frames": []}, default=str, ensure_ascii=False)[:-2] + "\n")

        for i, fm in enumerate(frame_metrics, 1):

            record = vars(fm).copy()

            # BIO
            bio = bio_lookup.get(fm.frame_idx)
            if bio:
                for k, v in bio.items():
                    if k not in ("frame_idx", "timestamp"):
                        record[f"bio_{k}"] = v

            # MOT
            extra = mot_lookup.get(fm.frame_idx)
            if extra:
                record.update(extra)

            record = clean_nans(record)

            # INIT CSV schema from FIRST FRAME ONLY
            if csv_writer is None:
                fieldnames = list(record.keys())   # no sorting, stable order
                csv_writer = csv.DictWriter(cf, fieldnames=fieldnames, extrasaction="ignore")
                csv_writer.writeheader()

            csv_writer.writerow(record)

            # JSON streaming
            if jsonl:
                jf.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
            else:
                if i > 1:
                    jf.write(",\n")
                jf.write(json.dumps(record, default=str, ensure_ascii=False))

            # progress log
            if i % progress_interval == 0:
                logger.info(f"{i}/{total_frames} frames exported")

        if not jsonl:
            jf.write("\n]\n}\n")

    logger.info(f"Export done: {total_frames} frames")

    return payload