import datetime
import threading
from typing import Optional, List
from ..database import safe_supabase_update


def check_cancel(cancel_event: Optional[threading.Event]) -> bool:
    """Check if cancellation was requested."""
    return cancel_event is not None and cancel_event.is_set()


def log_step(
    job_id: str,
    msg: str,
    logs: List[str],
    cancel_event: Optional[threading.Event] = None
) -> None:
    """Log analysis step and update database. Raise if cancelled."""
    if cancel_event and cancel_event.is_set():
        raise InterruptedError("Job cancelled by user.")
    
    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
    print(f"[JOB {job_id[:8]}] {msg}")
    
    import psutil
    mem = psutil.virtual_memory()
    print(f"[MEM] {mem.percent}% used | Available: {mem.available // (1024*1024)}MB")
    
    logs.append(f"{timestamp} - {msg}")
    safe_supabase_update(job_id, {"logs": logs})