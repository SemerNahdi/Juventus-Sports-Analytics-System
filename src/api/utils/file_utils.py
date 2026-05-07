import os
from typing import Optional


def get_content_type(filename: str) -> str:
    """Guess content type based on file extension."""
    ext = os.path.splitext(filename)[1].lower()
    mapping = {
        ".mp4": "video/mp4",
        ".json": "application/json",
        ".csv": "text/csv",
        ".txt": "text/plain",
        ".png": "image/png",
        ".svg": "image/svg+xml",
        ".trc": "text/plain",
        ".mot": "text/plain",
    }
    return mapping.get(ext, "application/octet-stream")


def is_video_file(filename: str) -> bool:
    """Check if file is a video based on content type."""
    content_type = get_content_type(filename)
    return content_type.startswith("video/")


def get_file_size_kb(path: str) -> Optional[int]:
    """Get file size in kilobytes."""
    try:
        return os.path.getsize(path) // 1024
    except Exception:
        return None