import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://your-url.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "your-service-role-key")
BUCKET_NAME = os.getenv("SUPABASE_BUCKET", "Sports Analytics")

# Cloudinary configuration
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

# Analysis defaults
DEFAULT_YOLO_SIZE = os.getenv("YOLO_SIZE_DEFAULT", "n")
DEFAULT_STRIDE = int(os.getenv("ANALYSIS_STRIDE", "2"))
DEFAULT_TARGET_HEIGHT = int(os.getenv("ANALYSIS_TARGET_HEIGHT", "640"))

# SMTP configuration
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").replace(" ", "").strip()

# Application
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
PORT = int(os.environ.get("PORT", 8000))

# Upload settings
SUPABASE_UPLOAD_RETRIES = int(os.getenv("SUPABASE_UPLOAD_RETRIES", "3"))
SUPABASE_UPLOAD_RETRY_DELAY = float(os.getenv("SUPABASE_UPLOAD_RETRY_DELAY", "0.5"))
SUPABASE_UPLOAD_WORKERS = int(os.getenv("SUPABASE_UPLOAD_WORKERS", "1"))
    
# Static files
STATIC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
    "dashboard"
)

@classmethod
def fix_ssl_cert(cls):
    """Fix SSL_CERT_FILE issue if it's set to non-existent path"""
    if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
        os.environ.pop("SSL_CERT_FILE", None)