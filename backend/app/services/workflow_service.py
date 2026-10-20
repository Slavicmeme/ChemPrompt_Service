"""Workflow engine (docs/11_WORKFLOW_ENGINE.md).

* PostgreSQL is the source of truth: ``current_step = N`` means steps 1..N are completed.
* Only step ``current_step + 1`` may run; anything else is rejected (also for the Agent).
* A step's result row and the ``current_step`` transition are written in ONE transaction while the
  workflow row is locked (SELECT ... FOR UPDATE).
"""
from __future__ import annotations

from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import session_scope
from app.models import DatasetVersion, User, Workflow, WorkflowDefinition, WorkflowStep
from app.services import dataset_service, training_service
from app.services.split_service import STRATEGIES

CUSTOM_TRAINING = "CUSTOM_TRAINING"


class WorkflowError(ValueError):
    pass


def create(db: Session, user: User, code: str = CUSTOM_TRAINING) -> Workflow:
    d = db.scalar(select(WorkflowDefinition).where(WorkflowDefinition.code == code))
    if d is None:
        raise WorkflowError(f"unknown workflow {code}")
    wf = Workflow(owner_id=user.id, workflow_definition_id=d.id, current_step=0, status="ACTIVE", context={})
    db.add(wf)
    db.flush()
    return wf


def _lock(db: Session, workflow_id: int, owner_id: int | None) -> Workflow:
    wf = db.scalar(select(Workflow).where(Workflow.id == workflow_id).with_for_update())
    if wf is None or (owner_id is not None and wf.owner_id != owner_id):
        raise WorkflowError("workflow not found")
    return wf


def _commit_step(db: Session, wf: Workflow, step_no: int, code: str, inp: dict | None, result: dict, ctx: dict) -> None:
    db.add(WorkflowStep(workflow_id=wf.id, step_no=step_no, code=code, status="COMPLETED", input=inp, result=result))
    wf.current_step = step_no
    wf.context = {**(wf.context or {}), **ctx}
    db.flush()


# ---- USER steps -------------------------------------------------------------------------------
def _dv(db: Session, wf: Workflow) -> DatasetVersion:
    dv = db.get(DatasetVersion, wf.context.get("dataset_version_id"))
    if dv is None:
        raise WorkflowError("dataset version missing from workflow context")
    return dv


def _s1(db, wf, user, inp):
    dv = db.get(DatasetVersion, int(inp.get("dataset_version_id", 0)))
    if dv is None or dv.dataset.owner_id != user.id:
        raise WorkflowError("dataset version not found")
    return {"dataset_version_id": dv.id, "columns": dv.columns, "row_count": dv.row_count,
            "suggested_mapping": dataset_service.suggest_mapping(dv)}, {"dataset_version_id": dv.id}


def _s2(db, wf, user, inp):
    dv = dataset_service.set_mapping(db, _dv(db, wf), inp.get("smiles_column", ""), list(inp.get("target_columns") or []))
    return {"smiles_column": dv.smiles_column, "target_columns": dv.target_columns}, {}


def _s3(db, wf, user, inp):
    dv = _dv(db, wf)
    report = dataset_service.validate(db, dv)
    if dv.status != "VALIDATED":
        raise StepFailure("validation failed: " + "; ".join(report["errors"]), report)
    return report, {}


def _s4(db, wf, user, inp):
    dv = dataset_service.set_task_type(db, _dv(db, wf), inp.get("task_type") or _dv(db, wf).task_type)
    name = (inp.get("model_name") or "").strip()[:200] or f"{dv.dataset.name}"
    return {"task_type": dv.task_type, "model_name": name}, {"task_type": dv.task_type, "model_name": name}


def _s5(db, wf, user, inp):
    mode = str(inp.get("descriptor_mode", "")).upper()
    if mode not in training_service.DESCRIPTOR_MODES:
        raise WorkflowError(f"descriptor_mode must be one of {training_service.DESCRIPTOR_MODES}")
    return {"descriptor_mode": mode}, {"descriptor_mode": mode}


def _s6(db, wf, user, inp):
    strategy = str(inp.get("split_strategy", "")).upper()
    if strategy not in STRATEGIES:
        raise WorkflowError(f"split_strategy must be one of {STRATEGIES}")
    dv = _dv(db, wf)
    job = training_service.create_job(db, owner_id=user.id, dv=dv, descriptor_mode=wf.context["descriptor_mode"],
                                      split_strategy=strategy, model_name=wf.context.get("model_name"), workflow_id=wf.id)
    wf.status = "RUNNING"
    return ({"split_strategy": strategy, "split_config_id": job.split_config_id, "training_job_id": job.id,
             "split_seeds": job.config["split_seeds"]},
            {"split_strategy": strategy, "training_job_id": job.id})


USER_STEPS: dict[str, Callable] = {"DATASET_UPLOAD": _s1, "COLUMN_MAPPING": _s2, "VALIDATION": _s3,
                                   "TASK_CONFIGURATION": _s4, "DESCRIPTOR_MODE": _s5, "SPLIT_STRATEGY": _s6}


def run_user_step(db: Session, user: User, workflow_id: int, step_no: int, inp: dict[str, Any]) -> Workflow:
    wf = _lock(db, workflow_id, user.id)
    if wf.status not in ("ACTIVE",):
        raise WorkflowError(f"workflow is {wf.status}")
    if step_no != wf.current_step + 1:
        raise WorkflowError(f"step {step_no} is not allowed: current_step={wf.current_step}, next allowed={wf.current_step + 1}")
    sd = next((s for s in wf.definition.steps if s.step_no == step_no), None)
    if sd is None or sd.executor != "USER":
        raise WorkflowError(f"step {step_no} is executed by the system")
    result, ctx = USER_STEPS[sd.code](db, wf, user, inp or {})
    _commit_step(db, wf, step_no, sd.code, inp, result, ctx)
    return wf


# ---- SYSTEM steps (called by the training runner) -----------------------------------------------
def complete_system_step(workflow_id: int, code: str, result: dict) -> None:
    with session_scope() as db:
        wf = _lock(db, workflow_id, None)
        sd = next(s for s in wf.definition.steps if s.code == code)
        if sd.step_no != wf.current_step + 1:
            raise WorkflowError(f"system step {code} out of order (current_step={wf.current_step})")
        _commit_step(db, wf, sd.step_no, code, None, result, {})
        if sd.step_no == len(wf.definition.steps):
            wf.status = "COMPLETED"
            wf.context = {**wf.context, "model_version_id": result.get("model_version_id")}


def mark_cancelled(workflow_id: int) -> None:
    with session_scope() as db:
        wf = _lock(db, workflow_id, None)
        if wf.status in ("RUNNING", "ACTIVE"):
            wf.status, wf.error = "CANCELLED", "사용자 요청으로 학습이 중단됨"


def fail(workflow_id: int, error: str) -> None:
    with session_scope() as db:
        wf = _lock(db, workflow_id, None)
        wf.status, wf.error = "FAILED", error[:4000]


def to_dict(wf: Workflow) -> dict[str, Any]:
    done = {s.step_no: s for s in wf.steps}
    return {
        "workflow_id": wf.id, "definition": wf.definition.code, "status": wf.status, "current_step": wf.current_step,
        "next_step": wf.current_step + 1 if wf.current_step < len(wf.definition.steps) else None,
        "context": wf.context, "error": wf.error, "created_at": wf.created_at, "updated_at": wf.updated_at,
        "steps": [{"step_no": sd.step_no, "code": sd.code, "name": sd.name, "executor": sd.executor,
                   "status": "COMPLETED" if sd.step_no in done else ("NEXT" if sd.step_no == wf.current_step + 1 else "PENDING"),
                   "result": done[sd.step_no].result if sd.step_no in done else None,
                   "completed_at": done[sd.step_no].completed_at if sd.step_no in done else None}
                  for sd in wf.definition.steps],
    }


class StepFailure(WorkflowError):
    def __init__(self, msg: str, detail: Any = None):
        super().__init__(msg)
        self.detail = detail


__all__ = ["create", "run_user_step", "complete_system_step", "fail", "to_dict", "WorkflowError"]
