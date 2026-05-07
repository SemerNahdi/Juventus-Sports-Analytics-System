from typing import Optional
import cloudinary
import cloudinary.uploader
from ..config import CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET, CLOUDINARY_CLOUD_NAME
import threading
def initialize_cloudinary() -> None:
    """Initialize Cloudinary configuration."""
    cloudinary.config(
        cloud_name=CLOUDINARY_CLOUD_NAME,
        api_key=CLOUDINARY_API_KEY,
        api_secret=CLOUDINARY_API_SECRET,
        secure=True
    )


# Initialize on import
initialize_cloudinary()


def upload_video_to_cloudinary(local_path: str, public_id: str, cancel_event: Optional[threading.Event] = None) -> str:
    """Upload video to Cloudinary with auto transcoding."""
    from ..jobs.job_helpers import check_cancel
    
    try:
        if check_cancel(cancel_event):
            return ""
        print(f"[Cloudinary] Uploading video: {local_path}...")
        response = cloudinary.uploader.upload(
            local_path,
            public_id=public_id,
            resource_type="video",
            overwrite=True,
            transformation=[{'fetch_format': "auto", 'quality': "auto"}]
        )
        return response.get("secure_url", "")
    except Exception as e:
        print(f"[Cloudinary Error] {e}")
        return ""