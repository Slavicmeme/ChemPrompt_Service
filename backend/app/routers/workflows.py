from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import jobs
from app.db import SessionLocal, get_db
from app.deps import current_user
from app.models import TrainingJob, User, Workflow
from app.routers.common import SSE_HEADERS, sse
from app.services import deletion_service
from app.services import workflow_service as wfs
from app.services.dataset_service import DatasetError
from app.services.training_service import TrainingError

router = APIRouter(prefix="/api/workflows", tags=["workflows"])


def _get(db: Session, user: User, workflow_id: int) -> Workflow:
    wf = db.get(Workflow, workflow_id)
    if wf is None or (wf.owner_id != user.id and not user.is_admin):
        raise HTTPException(404, "workflow not found")
    return wf


@router.post("")
def create(db: Session = Depends(get_db), user: User = Depends(current_user)):
    wf = wfs.create(db, user)
    db.commit()
    return wfs.to_dict(wf)


@router.get("")
def list_workflows(db: Session = Depends(get_db), user: User = Depends(current_user)):
    rows = db.scalars(select(Workflow).where(Workflow.owner_id == user.id).order_by(Workflow.id.desc()).limit(50)).all()
    return {"workflows": [wfs.to_dict(w) for w in rows]}


@router.get("/{workflow_id}")
def get_workflow(workflow_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    return wfs.to_dict(_get(db, user, workflow_id))


@router.post("/{workflow_id}/steps/{step_no}")
def run_step(workflow_id: int, step_no: int, body: dict | None = None, db: Session = Depends(get_db),
             user: User = Depends(current_user)):
    try:
        wf = wfs.run_user_step(db, user, workflow_id, step_no, body or {})
    except wfs.StepFailure as exc:
        db.commit()  # keep e.g. the validation report; current_step is NOT advanced
        return JSONResponse(status_code=422, content={"detail": str(exc), "report": exc.detail})
    except (wfs.WorkflowError, DatasetError, TrainingError, ValueError) as exc:
        db.rollback()
        raise HTTPException(409 if "not allowed" in str(exc) else 400, str(exc))
    db.commit()
    if wf.status == "RUNNING" and wf.context.get("training_job_id"):
        jobs.submit(wf.context["training_job_id"], wf.id)
    db.refresh(wf)
    return wfs.to_dict(wf)


@router.post("/{workflow_id}/cancel")
def cancel(workflow_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    """Stop training (RUNNING) or abandon a workflow that has not started training (ACTIVE)."""
    wf = _get(db, user, workflow_id)
    if wf.status == "RUNNING" and wf.context.get("training_job_id"):
        state = jobs.request_cancel(wf.context["training_job_id"])
        return {"workflow_id": wf.id, "status": state}
    if wf.status == "ACTIVE":
        wf.status, wf.error = "CANCELLED", "사용자가 workflow를 중단함"
        db.commit()
        return {"workflow_id": wf.id, "status": "CANCELLED"}
    raise HTTPException(409, f"이미 끝난 workflow임 ({wf.status})")


@router.delete("/{workflow_id}")
def delete_workflow(workflow_id: int, delete_model: bool = False, db: Session = Depends(get_db),
                    user: User = Depends(current_user)):
    wf = _get(db, user, workflow_id)
    try:
        result = deletion_service.delete_workflow(db, wf, delete_model=delete_model)
    except deletion_service.DeletionError as exc:
        db.rollback()
        raise HTTPException(409, str(exc))
    db.commit()
    return deletion_service.remove_files(result)


@router.get("/{workflow_id}/events")
async def events(workflow_id: int, request: Request, db: Session = Depends(get_db), user: User = Depends(current_user)):
    """SSE: emits only DB-observed state changes (workflow_step / status), never synthetic progress."""
    _get(db, user, workflow_id)

    async def gen():
        last_step, last_stage, last_status = None, None, None
        while True:
            if await request.is_disconnected():
                return
            with SessionLocal() as s:
                wf = s.get(Workflow, workflow_id)
                d = wfs.to_dict(wf)
                job = s.get(TrainingJob, wf.context.get("training_job_id")) if wf.context.get("training_job_id") else None
                stage = (job.status, job.stage, (job.progress or {}).get("folds_trained")) if job else None
            if d["current_step"] != last_step:
                last_step = d["current_step"]
                yield sse("workflow_step", {"current_step": d["current_step"], "steps": d["steps"], "status": d["status"]})
            if stage != last_stage and job is not None:
                last_stage = stage
                yield sse("status", {"training_job_id": job.id, "job_status": stage[0], "stage": stage[1], "progress": job.progress})
            if d["status"] != last_status:
                last_status = d["status"]
                if d["status"] in ("COMPLETED", "FAILED", "CANCELLED"):
                    if d["status"] == "FAILED":
                        yield sse("error", {"error": d["error"]})
                    yield sse("done", {"status": d["status"], "context": d["context"]})
                    return
            await asyncio.sleep(1.0)

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)
