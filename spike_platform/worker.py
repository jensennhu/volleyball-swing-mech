"""
Simple threading-based background job runner.
One job at a time — no Celery/Redis dependency.
"""

import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Optional


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.PENDING
    progress_pct: float = 0.0
    message: str = ""
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class BackgroundWorker:
    """Single-threaded background job runner."""

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._current_thread: Optional[threading.Thread] = None

    def submit(self, fn: Callable, *args, **kwargs) -> str:
        """Submit a background job. Returns job_id."""
        job_id = str(uuid.uuid4())[:8]
        job = Job(id=job_id)

        with self._lock:
            self._jobs[job_id] = job

        def _run():
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(timezone.utc)
            try:
                fn(*args, progress_callback=self._make_progress_cb(job_id), **kwargs)
                job.status = JobStatus.COMPLETED
                job.progress_pct = 100.0
            except Exception as e:
                job.status = JobStatus.FAILED
                job.error = traceback.format_exc()
                job.message = str(e)
            finally:
                job.completed_at = datetime.now(timezone.utc)

        thread = threading.Thread(target=_run, daemon=True)
        self._current_thread = thread
        thread.start()
        return job_id

    def _make_progress_cb(self, job_id: str):
        """Create a progress callback for a specific job."""
        def callback(pct: float, message: str = ""):
            with self._lock:
                job = self._jobs.get(job_id)
                if job:
                    job.progress_pct = pct
                    if message:
                        job.message = message
        return callback

    def get_status(self, job_id: str) -> Optional[dict]:
        """Get job status as a dict."""
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            return {
                "id": job.id,
                "status": job.status.value,
                "progress_pct": job.progress_pct,
                "message": job.message,
                "error": job.error,
            }

    @property
    def is_busy(self) -> bool:
        """Check if a job is currently running."""
        return (
            self._current_thread is not None
            and self._current_thread.is_alive()
        )


# Singleton worker instance
worker = BackgroundWorker()
