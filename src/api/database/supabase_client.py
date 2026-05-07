import os
from typing import Optional
from supabase import create_client, Client

from src.api.config import SUPABASE_KEY, SUPABASE_URL

# Fix for SSL_CERT_FILE issue: if it's set to a non-existent path, httpx (used by supabase) will fail.
if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    os.environ.pop("SSL_CERT_FILE", None)


def initialize_supabase() -> Optional[Client]:
    """Initialize Supabase client."""
    if not SUPABASE_URL or "your-url" in SUPABASE_URL:
        print("Supabase configuration missing or using placeholder URL. Set SUPABASE_URL in .env or env vars.")
        return None
    
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print("Failed to initialize Supabase client:", e)
        return None


# Global client instance
supabase: Optional[Client] = initialize_supabase()