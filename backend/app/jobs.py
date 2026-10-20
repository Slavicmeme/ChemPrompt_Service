"""In-process background runner for training jobs (prototype).

Job state lives in PostgreSQL (training_jobs / workflows); the executor only runs work. On startup,
jobs left RUNNING by a previous process are marked FAILED and QUEUED jobs are resubmitted.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from sqlalchemy import select

from app.config import settings
from app.db import session_scope
from app.models import TrainingJob
from app.services import training_service, workflow_service
from app.services.cancellation import JobCancelled

log = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=settings.training_workers, thread_name_prefix="training")
_cancel_events: dict[int, threading.Event] = {}


def _execute(job_id: int, workflow_id: int | None) -> None:
    on_step = (lambda code, result: workflow_service.complete_system_step(workflow_id, code, result)) if workflow_id else None
    event = _cancel_events.setdefault(job_id, threading.Event())
    try:
        training_service.run_job(job_id, on_step=on_step, should_cancel=event.is_set)
    except JobCancelled:
        if workflow_id:
            workflow_service.mark_cancelled(workflow_id)
    except Exception as exc:  # noqa: BLE001
        if workflow_id:
            workflow_service.fail(workflow_id, f"training job {job_id} failed: {exc}")
    finally:
        _cancel_events.pop(job_id, None)


def submit(job_id: int, workflow_id: int | None = None):
    _cancel_events.setdefault(job_id, threading.Event())
    return _executor.submit(_execute, job_id, workflow_id)


def request_cancel(job_id: int) -> str:
    """Ask a QUEUED/RUNNING job to stop. Returns the resulting state for the API response."""
    event = _cancel_events.get(job_id)
    if event is not None:          # handled by this process: the worker stops at the next check
        event.set()
        return "CANCELLING"
    with session_scope() as db:   # not tracked here (e.g. orphaned): mark it directly
        job = db.get(TrainingJob, job_id)
        if job is None or job.status not in ("QUEUED", "RUNNING"):
            return job.status if job else "NOT_FOUND"
        job.status, job.stage, job.error = "CANCELLED", "cancelled", "사용자 요청으로 학습이 중단됨"
        job.finished_at = datetime.now(timezone.utc)
        wf_id = job.workflow_id
    if wf_id:
        workflow_service.mark_cancelled(wf_id)
    return "CANCELLED"


def recover() -> None:
    with session_scope() as db:
        for job in db.scalars(select(TrainingJob).where(TrainingJob.status == "RUNNING")):
            job.status, job.stage, job.error = "FAILED", "failed", "interrupted by server restart"
            job.finished_at = datetime.now(timezone.utc)
            if job.workflow_id:
                from app.models import Workflow

                wf = db.get(Workflow, job.workflow_id)
                wf.status, wf.error = "FAILED", "training interrupted by server restart"
        queued = [(j.id, j.workflow_id) for j in db.scalars(select(TrainingJob).where(TrainingJob.status == "QUEUED"))]
    for job_id, wf_id in queued:
        log.info("resubmitting queued training job %s", job_id)
        submit(job_id, wf_id)
