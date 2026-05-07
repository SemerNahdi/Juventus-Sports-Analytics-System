from typing import Optional, Dict, Any
from .supabase_client import supabase


def get_first_row(query_result: Any) -> Optional[Dict[str, Any]]:
    """Extract first row from Supabase query result safely."""
    rows = query_result.data or []
    return rows[0] if rows and isinstance(rows[0], dict) else None


def safe_supabase_update(job_id: str, data: Dict[str, Any], table: str = "analyses") -> None:
    """Attempt Supabase update without crashing on failure.
    
    If a PGRST204 (unknown column) error is returned, automatically retries
    the update with only the keys that Supabase already knows about.
    """
    if supabase is None:
        return

    remaining = dict(data)
    for _attempt in range(len(remaining) + 1):
        if not remaining:
            break
        try:
            supabase.table(table).update(remaining).eq("id", job_id).execute()
            return  # success
        except Exception as e:
            err_str = str(e)
            # PGRST204: column does not exist — find and drop it, then retry
            if "PGRST204" in err_str and remaining:
                # Try to extract the bad column name from the error message
                import re
                m = re.search(r"find the '(\w+)' column", err_str)
                bad_key = m.group(1) if m else None
                if bad_key and bad_key in remaining:
                    print(f"[DB Warn] Column '{bad_key}' not in Supabase schema — skipping field.")
                    remaining.pop(bad_key)
                    continue
            print(f"[DB Update Error] {table} {job_id}: {e}")
            return


def create_job_record(job_id: str, player_id: int, session_tags: str) -> bool:
    """Create new analysis job record in database. Returns True on success."""
    if supabase is None:
        return False
    
    try:
        supabase.table("analyses").insert({
            "id": job_id,
            "player_id": player_id,
            "status": "processing",
            "session_tags": session_tags
        }).execute()
        return True
    except Exception as e:
        error_str = str(e)
        print(f"[DB Initialization Error] {job_id}: {error_str}")
        
        if "PGRST204" in error_str or "status" in error_str.lower():
            raise RuntimeError(
                "SCHEMA ERROR: Your Supabase 'analyses' table is missing the 'status' column. "
                "Run: ALTER TABLE public.analyses ADD COLUMN status TEXT DEFAULT 'processing';"
            )
        raise RuntimeError(f"Database Error: {error_str}")


def get_job_status(job_id: str) -> Optional[str]:
    """Fetch job status from database."""
    if supabase is None:
        return None
    
    try:
        res = supabase.table("analyses").select("status").eq("id", job_id).execute()
        row = get_first_row(res)
        return row.get("status") if row else None
    except Exception as e:
        print(f"[DB Query Error] {job_id}: {e}")
        return None


def list_analyses(limit: int = 20) -> Any:
    """List recent analyses from database."""
    if supabase is None:
        return []
    
    try:
        response = supabase.table("analyses").select("*").order("created_at", desc=True).limit(limit).execute()
        return response.data
    except Exception as e:
        raise RuntimeError(f"Failed to fetch analyses: {str(e)}")


def get_latest_analysis() -> Any:
    """Fetch most recent analysis."""
    if supabase is None:
        return None
    
    try:
        response = supabase.table("analyses").select("*").order("created_at", desc=True).limit(1).execute()
        if not response.data:
            return None
        return response.data[0]
    except Exception as e:
        raise RuntimeError(f"Failed to fetch latest analysis: {str(e)}")


def get_analysis_by_id(job_id: str) -> Optional[Dict[str, Any]]:
    """Fetch specific analysis by job ID."""
    if supabase is None:
        return None
    
    try:
        res = supabase.table("analyses").select("*").eq("id", job_id).execute()
        return get_first_row(res)
    except Exception as e:
        print(f"[Supabase Query Error] {job_id}: {str(e)}")
        return None