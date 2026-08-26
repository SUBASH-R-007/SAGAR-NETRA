"""In-process job registry for survey uploads.

Jobs run on worker threads (the pipeline is synchronous NumPy/torch);
the WebSocket layer reads job state and pushes deltas to the browser.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class Job:
    id: str
    filename: str
    status: str = "queued"  # queued | running | done | error
    stage: str = ""
    fraction: float = 0.0
    message: str = ""
    survey: str | None = None
    n_contacts: int = 0
    error: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(tz=UTC).isoformat(timespec="seconds")
    )
    finished_at: str | None = None
    version: int = 0  # bumped on every update so pollers can detect change

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


class JobRegistry:
    """Thread-safe registry; updates bump a version for cheap change detection."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, filename: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], filename=filename)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in fields.items():
                setattr(job, key, value)
            if job.status in ("done", "error") and job.finished_at is None:
                job.finished_at = datetime.now(tz=UTC).isoformat(timespec="seconds")
            job.version += 1

    def all(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
