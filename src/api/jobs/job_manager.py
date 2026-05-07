import threading
from typing import Optional, Dict

# --- GLOBAL STATE FOR JOB TRACKING ---
active_jobs: Dict[str, threading.Event] = {}
active_jobs_lock = threading.RLock()


def register_active_job(job_id: str, cancel_event: threading.Event) -> None:
    """Register active job with cancellation event."""
    with active_jobs_lock:
        active_jobs[job_id] = cancel_event


def get_active_job_event(job_id: str) -> Optional[threading.Event]:
    """Retrieve cancellation event for active job."""
    with active_jobs_lock:
        return active_jobs.get(job_id)


def unregister_active_job(job_id: str) -> None:
    """Unregister completed or cancelled job."""
    with active_jobs_lock:
        active_jobs.pop(job_id, None)


def is_job_active(job_id: str) -> bool:
    """Check if job is currently running."""
    with active_jobs_lock:
        return job_id in active_jobs