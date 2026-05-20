import os
import uuid
import json
import shutil
import tempfile
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Import from refactored modules
from .config import STATIC_DIR, DEFAULT_YOLO_SIZE, DEFAULT_STRIDE, DEFAULT_TARGET_HEIGHT, PORT, BASE_URL
from .database import (
    supabase,
    create_job_record,
    get_job_status,
    list_analyses as db_list_analyses,
    get_latest_analysis as db_get_latest,
    get_analysis_by_id
)
from .database.operations import safe_supabase_update, get_first_row
from .jobs.job_manager import get_active_job_event
from .analysis_routes import queue_analysis_job
from .selection_sessions import selection_store
from .utils.decorators import require_supabase
from .utils.email_utils import send_analysis_email

from src.analytics.player_selection import preview_detections, seed_from_candidate

# Create FastAPI app
app = FastAPI(
    title="Sports Analytics API",
    description="Advanced Biomechanics & Tracking backend with Supabase Storage integration."
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== API ENDPOINTS ====================

@app.get("/")
async def root() -> FileResponse:
    """Serve root index page."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/dashboard.html")
async def dashboard() -> FileResponse:
    """Serve analytics dashboard."""
    return FileResponse(os.path.join(STATIC_DIR, "dashboard.html"))


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check for port binding detection."""
    return {"status": "ok", "service": "Mitus AI Sports Analytics System"}


@app.post("/analyze/preview")
async def preview_video_for_selection(
    file: UploadFile = File(...),
    yolo_size: str = Form(DEFAULT_YOLO_SIZE),
    frame_idx: Optional[int] = Form(None),
) -> JSONResponse:
    """
    Step 1: Upload video and receive an annotated preview frame + player candidates.
    Does not require Supabase.
    """
    content_type = file.content_type or ""
    if not content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video.")

    original_filename = file.filename or "upload.mp4"
    session_id = str(uuid.uuid4())
    temp_dir = os.path.join(tempfile.gettempdir(), "mitus_uploads", "preview")
    os.makedirs(temp_dir, exist_ok=True)
    video_path = os.path.join(temp_dir, f"{session_id}_{original_filename}")

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        preview = preview_detections(
            video_path,
            frame_idx=frame_idx,
            yolo_size=yolo_size,
            auto_frame=frame_idx is None,
        )
    except Exception as e:
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except OSError:
                pass
        raise HTTPException(status_code=400, detail=f"Preview failed: {e}") from e

    session = selection_store.create(
        video_path=video_path,
        original_filename=original_filename,
        preview=preview,
        yolo_size=yolo_size,
    )

    payload = preview.to_dict()
    payload.update({
        "status": "ok",
        "session_id": session.session_id,
        "message": (
            f"{len(preview.candidates)} player(s) detected. Select one to start analysis."
            if preview.candidates
            else "No players detected on this frame. Try another frame."
        ),
    })
    return JSONResponse(status_code=200, content=payload)


@app.post("/analyze/preview/refresh")
async def refresh_preview_frame(
    session_id: str = Form(...),
    frame_idx: int = Form(0),
) -> JSONResponse:
    """Re-run detection on a different frame for an existing preview session."""
    session = selection_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Selection session expired or not found.")

    try:
        preview = preview_detections(
            session.video_path,
            frame_idx=frame_idx,
            yolo_size=session.yolo_size,
            auto_frame=False,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preview refresh failed: {e}") from e

    session.preview = preview
    payload = preview.to_dict()
    payload.update({
        "status": "ok",
        "session_id": session.session_id,
        "message": (
            f"{len(preview.candidates)} player(s) on frame {frame_idx}."
            if preview.candidates
            else "No players on this frame."
        ),
    })
    return JSONResponse(status_code=200, content=payload)


@app.post("/analyze/confirm")
async def confirm_selection_and_analyze(
    background_tasks: BackgroundTasks,
    session_id: str = Form(...),
    candidate_index: int = Form(...),
    player_id: int = Form(1),
    yolo_size: str = Form(DEFAULT_YOLO_SIZE),
    player_height: float = Form(1.75),
    mass_kg: float = Form(75.0),
    session_tags: str = Form("performance-match"),
    run_sports2d: bool = Form(False),
    email: Optional[str] = Form(None),
    stride: int = Form(DEFAULT_STRIDE),
    target_height: int = Form(DEFAULT_TARGET_HEIGHT),
    protocol_id: str = Form("continuous_gait"),
    mat_grid_spacing_cm: float = Form(10.0),
) -> JSONResponse:
    """
    Step 2: User selected a player from preview → queue full production analysis.
    """
    session = selection_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Selection session expired or not found.")

    if not session.candidates:
        raise HTTPException(status_code=400, detail="No players to select on preview frame.")

    if candidate_index < 0 or candidate_index >= len(session.candidates):
        raise HTTPException(
            status_code=400,
            detail=f"candidate_index must be 0..{len(session.candidates) - 1}",
        )

    seed = seed_from_candidate(session.video_path, session.preview, candidate_index)
    seed_bbox = list(seed.seed_bbox)
    seed_frame_idx = seed.seed_frame_idx

    response = queue_analysis_job(
        background_tasks,
        session.video_path,
        session.original_filename,
        player_id=player_id,
        yolo_size=yolo_size or session.yolo_size,
        player_height=player_height,
        mass_kg=mass_kg,
        session_tags=session_tags,
        run_sports2d=run_sports2d,
        email=email,
        stride=stride,
        target_height=target_height,
        seed_bbox=seed_bbox,
        seed_frame_idx=seed_frame_idx,
        protocol_id=protocol_id,
        mat_grid_spacing_cm=mat_grid_spacing_cm,
    )

    selection_store.delete(session_id)
    try:
        if os.path.exists(session.video_path):
            os.remove(session.video_path)
    except OSError:
        pass
    return response


@app.post("/analyze")
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    player_id: int = Form(1),
    yolo_size: str = Form(DEFAULT_YOLO_SIZE),
    player_height: float = Form(1.75),
    mass_kg: float = Form(75.0),
    session_tags: str = Form("performance-match"),
    run_sports2d: bool = Form(False),
    email: Optional[str] = Form(None),
    stride: int = Form(DEFAULT_STRIDE),
    target_height: int = Form(DEFAULT_TARGET_HEIGHT),
    seed_bbox: Optional[str] = Form(None),
    seed_frame_idx: int = Form(0),
    protocol_id: str = Form("continuous_gait"),
    mat_grid_spacing_cm: float = Form(10.0)
) -> JSONResponse:
    print(f"[API] New Analysis Request: email={email}")
    """
    Async analysis endpoint:
    1. Verify database connection + schema.
    2. Return 202 Accepted with job_id immediately.
    3. Process AI analysis in background.
    """
    if supabase is None:
        msg = "Supabase client not configured. Set SUPABASE_URL and SUPABASE_KEY in .env or environment."
        print(f"[Analyze Error] {msg}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": msg, "detail": "See template.env for variables."}
        )

    # Validate file is video
    content_type = file.content_type or ""
    if not content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video.")

    # Parse seed_bbox if provided
    parsed_seed_bbox: Optional[List[int]] = None
    if seed_bbox:
        try:
            raw = seed_bbox.strip()
            if raw.startswith("["):
                vals = json.loads(raw)
            else:
                vals = [v.strip() for v in raw.split(",")]
            parsed_seed_bbox = [int(float(v)) for v in vals]
            if len(parsed_seed_bbox) != 4:
                raise ValueError("seed_bbox must have exactly 4 values: x,y,w,h")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid seed_bbox format: {e}")

    original_filename = file.filename or "upload.mp4"
    temp_dir_base = os.path.join(tempfile.gettempdir(), "mitus_uploads")
    os.makedirs(temp_dir_base, exist_ok=True)
    temp_input_path = os.path.join(temp_dir_base, f"preview_{uuid.uuid4()}_{original_filename}")

    with open(temp_input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return queue_analysis_job(
        background_tasks,
        temp_input_path,
        original_filename,
        player_id=player_id,
        yolo_size=yolo_size,
        player_height=player_height,
        mass_kg=mass_kg,
        session_tags=session_tags,
        run_sports2d=run_sports2d,
        email=email,
        stride=stride,
        target_height=target_height,
        seed_bbox=parsed_seed_bbox,
        seed_frame_idx=seed_frame_idx,
        protocol_id=protocol_id,
        mat_grid_spacing_cm=mat_grid_spacing_cm,
    )


@app.post("/analyses/{job_id}/cancel", response_model=None)
async def cancel_analysis(job_id: str) -> JSONResponse:
    """Cancel running or queued job."""
    if supabase is None:
        return JSONResponse(status_code=503, content={"status": "error", "message": "Supabase not configured."})

    job_event = get_active_job_event(job_id)
    if job_event is not None:
        print(f"[API] Signaling cancellation for job {job_id[:8]}...")
        job_event.set()
        safe_supabase_update(job_id, {
            "status": "cancelling",
            "error": "User requested cancellation."
        })
        return JSONResponse(status_code=200, content={"status": "success", "message": "Cancellation signal sent."})

    # Check database for pending/processing jobs
    if supabase is not None:
        try:
            status = get_job_status(job_id)
            if status in ["processing", "pending"]:
                supabase.table("analyses").update({"status": "cancelled"}).eq("id", job_id).execute()
                return JSONResponse(status_code=200, content={"status": "success", "message": "Job marked as cancelled in database."})
        except Exception as e:
            print(f"[Cancel Error] {job_id}: {e}")

    return JSONResponse(
        status_code=404,
        content={"status": "error", "message": "Job not found or not active."}
    )


@app.get("/analyses", response_model=None)
@require_supabase
async def list_analyses(limit: int = 20) -> Any:
    """List recent analyses from Supabase."""
    try:
        return db_list_analyses(limit)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analyses/latest", response_model=None)
@require_supabase
async def get_latest_analysis() -> Any:
    """Fetch single most recent analysis."""
    try:
        result = db_get_latest()
        if result is None:
            raise HTTPException(status_code=404, detail="No analyses found.")
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analyses/{job_id}")
async def get_analysis(job_id: str) -> JSONResponse:
    """Fetch details for specific analysis job."""
    if supabase is None:
        return JSONResponse(status_code=503, content={"status": "error", "message": "Supabase not configured."})

    try:
        result = get_analysis_by_id(job_id)

        if result is None:
            return JSONResponse(
                status_code=404,
                content={"status": "error", "message": f"Job {job_id} not found in database."}
            )

        return JSONResponse(status_code=200, content=result)

    except Exception as e:
        print(f"[Supabase DEBUG] Error fetching {job_id}: {str(e)}")
        detail = str(e)
        if "getaddrinfo failed" in detail or "Name or service not known" in detail or "Temporary failure in name resolution" in detail:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "message": "Database unavailable",
                    "detail": "Supabase host could not be resolved. Check SUPABASE_URL and network connectivity."
                }
            )
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Database query failed",
                "detail": detail
            }
        )


@app.post("/analyses/{job_id}/email")
async def email_analysis_results(job_id: str, email: str) -> JSONResponse:
    """Send analysis results to specified email."""
    if supabase is None:
        return JSONResponse(status_code=503, content={"status": "error", "message": "Supabase not configured."})
    
    from .utils.email_utils import send_analysis_email
    
    try:
        result = get_analysis_by_id(job_id)
        if not result:
            return JSONResponse(status_code=404, content={"status": "error", "message": "Job not found."})
            
        summary = result.get("summary", {})
        if isinstance(summary, str):
            summary = json.loads(summary)
            
        player_summary = summary.get("player_summary", {})
        risk_score = player_summary.get("peak_risk_score", 0.0)
        player_id = result.get("player_id", 1)
        video_url = result.get("video_url", "")
        
        send_analysis_email(email, job_id, player_id, video_url, risk_score=risk_score)
        
        return JSONResponse(status_code=200, content={"status": "success", "message": f"Results sent to {email}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


# Final Catch-all for Static Assets
app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    print(f"Starting Mitus AI Sports Analytics API on port {PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=PORT)