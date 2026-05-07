import os
import time
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from ..database.supabase_client import supabase
from ..config import BUCKET_NAME, SUPABASE_UPLOAD_RETRIES, SUPABASE_UPLOAD_RETRY_DELAY, SUPABASE_UPLOAD_WORKERS
from ..utils.file_utils import get_content_type
from ..jobs.job_helpers import check_cancel
import threading

def upload_file_to_supabase(local_path: str, remote_path: str, cancel_event: Optional['threading.Event'] = None) -> str:
    """Upload single file to Supabase and return signed URL."""
    def should_retry(exc: Exception) -> bool:
        message = str(exc)
        if isinstance(exc, OSError):
            if getattr(exc, "winerror", None) in (10035, 10054, 10060, 10061):
                return True
        retry_markers = (
            "10035", "temporarily unavailable", "timed out", "timeout",
            "connection reset", "connection aborted", "connection closed",
            "non-blocking socket operation could not be completed immediately",
        )
        return any(marker in message.lower() for marker in retry_markers)

    if check_cancel(cancel_event):
        return ""

    if supabase is None:
        print("Supabase client unavailable: skipping upload to Supabase storage.")
        return ""

    content_type = get_content_type(local_path)
    max_attempts = SUPABASE_UPLOAD_RETRIES
    base_delay = SUPABASE_UPLOAD_RETRY_DELAY

    for attempt in range(1, max_attempts + 1):
        if check_cancel(cancel_event):
            return ""

        try:
            with open(local_path, "rb") as f:
                if check_cancel(cancel_event):
                    return ""
                supabase.storage.from_(BUCKET_NAME).upload(
                    path=remote_path,
                    file=f,
                    file_options={"content-type": content_type, "upsert": "true"}
                )

            # Get signed URL (10 years expiry)
            try:
                if check_cancel(cancel_event):
                    return ""
                signed_url_res = supabase.storage.from_(BUCKET_NAME).create_signed_url(remote_path, 315360000)
                if isinstance(signed_url_res, dict) and "signedURL" in signed_url_res:
                    return signed_url_res["signedURL"]
                elif hasattr(signed_url_res, "signed_url"):
                    return signed_url_res.signed_url
                return supabase.storage.from_(BUCKET_NAME).get_public_url(remote_path)
            except Exception:
                return supabase.storage.from_(BUCKET_NAME).get_public_url(remote_path)

        except Exception as e:
            if check_cancel(cancel_event):
                return ""
            if attempt < max_attempts and should_retry(e):
                print(f"[Supabase Upload Retry] {remote_path} attempt {attempt}/{max_attempts}: {e}")
                time.sleep(base_delay * attempt)
                continue
            print(f"[Supabase Upload Error] {local_path} -> {remote_path}: {e}")
            return ""

    return ""


def upload_directory_to_supabase(directory: str, prefix: str, cancel_event: Optional[threading.Event] = None) -> Dict[str, str]:
    """Recursively upload directory to Supabase in parallel."""
    urls = {}
    tasks = {}
    max_workers = SUPABASE_UPLOAD_WORKERS
    executor = ThreadPoolExecutor(max_workers=max_workers)

    try:
        if check_cancel(cancel_event):
            return urls

        for root, _, files in os.walk(directory):
            if check_cancel(cancel_event):
                break
            for file in files:
                if check_cancel(cancel_event):
                    break
                local_path = os.path.join(root, file)
                rel_path = os.path.relpath(local_path, directory).replace("\\", "/")
                remote_path = f"{prefix}/{rel_path}"
                tasks[executor.submit(upload_file_to_supabase, local_path, remote_path, cancel_event)] = rel_path

        pending = set(tasks.keys())
        while pending:
            if check_cancel(cancel_event):
                executor.shutdown(wait=False, cancel_futures=True)
                return urls

            done, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
            for future in done:
                rel_path = tasks[future]
                try:
                    url = future.result()
                    if url:
                        urls[rel_path] = url
                except Exception as e:
                    print(f"[Upload Error] {rel_path}: {e}")
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return urls