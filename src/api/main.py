import os
import uuid
import json
import shutil
import tempfile
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Import from refactored modules
from .config import STATIC_DIR, DEFAULT_YOLO_SIZE, DEFAULT_STRIDE, DEFAULT_TARGET_HEIGHT, PORT
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
from .jobs.analysis_job import run_full_analysis_job
from .utils.decorators import require_supabase

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
    job_id = str(uuid.uuid4())
    print(f"\n[JOB {job_id[:8]}] Attempting to initialize job record...")

    # Critical: fast database verification
    try:
        create_job_record(job_id, player_id, session_tags)
        print(f"[JOB {job_id[:8]}] DB record initialized.")
    except RuntimeError as e:
        error_str = str(e)
        print(f"[JOB {job_id[:8]}] DB Initialization FAILED: {error_str}")
        
        if "SCHEMA ERROR" in error_str:
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "SCHEMA ERROR: Your Supabase 'analyses' table is missing required columns.",
                    "detail": error_str
                }
            )
        
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Database Error", "detail": error_str}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Database Error", "detail": str(e)}
        )

    # Save temp copy for background worker
    temp_dir_base = tempfile.gettempdir()
    os.makedirs(os.path.join(temp_dir_base, "mitus_uploads"), exist_ok=True)
    temp_input_path = os.path.join(temp_dir_base, "mitus_uploads", f"{job_id}_{original_filename}")

    with open(temp_input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Queue heavy work
    background_tasks.add_task(
        run_full_analysis_job,
        job_id,
        temp_input_path,
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
        parsed_seed_bbox,
        seed_frame_idx,
        protocol_id,
        mat_grid_spacing_cm
    )

    return JSONResponse(
        status_code=202,
        content={
            "status": "processing",
            "job_id": job_id,
            "message": "Analysis queued successfully."
        }
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