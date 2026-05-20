"""Shared helpers for starting analysis jobs from API routes."""

from __future__ import annotations

import os
import tempfile
import uuid
from typing import List, Optional

from fastapi import BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse

from .database import create_job_record, supabase
from .jobs.analysis_job import run_full_analysis_job
from .utils.email_utils import send_analysis_email


def _upload_dir() -> str:
    base = os.path.join(tempfile.gettempdir(), "mitus_uploads")
    os.makedirs(base, exist_ok=True)
    return base


def queue_analysis_job(
    background_tasks: BackgroundTasks,
    video_path: str,
    original_filename: str,
    *,
    player_id: int,
    yolo_size: str,
    player_height: float,
    mass_kg: float,
    session_tags: str,
    run_sports2d: bool,
    email: Optional[str],
    stride: int,
    target_height: int,
    seed_bbox: Optional[List[int]],
    seed_frame_idx: int,
    protocol_id: str,
    mat_grid_spacing_cm: float,
) -> JSONResponse:
    if supabase is None:
        msg = "Supabase client not configured. Set SUPABASE_URL and SUPABASE_KEY in .env."
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": msg},
        )

    job_id = str(uuid.uuid4())

    try:
        create_job_record(job_id, player_id, session_tags)
    except RuntimeError as e:
        error_str = str(e)
        if "SCHEMA ERROR" in error_str:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "SCHEMA ERROR", "detail": error_str},
            )
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Database Error", "detail": error_str},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Database Error", "detail": str(e)},
        )

    dest_path = os.path.join(_upload_dir(), f"{job_id}_{original_filename}")
    if os.path.abspath(video_path) != os.path.abspath(dest_path):
        import shutil
        shutil.copy2(video_path, dest_path)
        work_path = dest_path
    else:
        work_path = video_path

    if email:
        try:
            send_analysis_email(
                to_email=email,
                job_id=job_id,
                player_id=player_id,
                video_url="pending",
                risk_score=0.0,
            )
        except Exception:
            pass

    background_tasks.add_task(
        run_full_analysis_job,
        job_id,
        work_path,
        player_id,
        yolo_size,
        player_height,
        mass_kg,
        session_tags,
        run_sports2d,
        original_filename,
        email,
        stride,
        target_height,
        seed_bbox,
        seed_frame_idx,
        protocol_id,
        mat_grid_spacing_cm,
    )

    return JSONResponse(
        status_code=202,
        content={
            "status": "accepted",
            "job_id": job_id,
            "message": "Analysis queued.",
        },
    )
