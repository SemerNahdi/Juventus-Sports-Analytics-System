"""Per-request player selection sessions (no module-level global preview state)."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.analytics.player_selection import DetectionCandidate, PreviewResult


@dataclass
class SelectionSession:
    session_id: str
    video_path: str
    original_filename: str
    yolo_size: str
    preview: PreviewResult
    created_at: float = field(default_factory=time.time)

    @property
    def frame_idx(self) -> int:
        return self.preview.frame_idx

    @property
    def candidates(self) -> List[DetectionCandidate]:
        return self.preview.candidates


class SelectionSessionStore:
    """Thread-safe in-memory store. Use Redis for multi-worker deployments."""

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._ttl = ttl_seconds
        self._sessions: Dict[str, SelectionSession] = {}
        self._lock = threading.RLock()

    def create(
        self,
        video_path: str,
        original_filename: str,
        preview: PreviewResult,
        yolo_size: str,
    ) -> SelectionSession:
        self.cleanup_expired()
        session_id = str(uuid.uuid4())
        session = SelectionSession(
            session_id=session_id,
            video_path=video_path,
            original_filename=original_filename,
            yolo_size=yolo_size,
            preview=preview,
        )
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Optional[SelectionSession]:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            return None
        if time.time() - session.created_at > self._ttl:
            self.delete(session_id)
            return None
        return session

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def cleanup_expired(self) -> int:
        now = time.time()
        expired: List[str] = []
        with self._lock:
            for sid, sess in self._sessions.items():
                if now - sess.created_at > self._ttl:
                    expired.append(sid)
            for sid in expired:
                self._sessions.pop(sid, None)
        return len(expired)


# Process-local singleton for the API layer
selection_store = SelectionSessionStore()
