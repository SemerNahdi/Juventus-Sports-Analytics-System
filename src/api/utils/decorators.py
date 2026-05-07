from functools import wraps
from typing import Any, Callable
from fastapi import HTTPException
from ..database.supabase_client import supabase


def require_supabase(func: Callable) -> Callable:
    """Decorator: Enforce Supabase availability before endpoint execution."""
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        if supabase is None:
            raise HTTPException(status_code=503, detail="Supabase not configured.")
        return await func(*args, **kwargs)
    return wrapper