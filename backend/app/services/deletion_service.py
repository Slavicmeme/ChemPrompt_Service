"""Deleting user workflows and models (DB rows first, then storage files).

FK order: predictions -> model versions (artifacts cascade) -> training jobs (fold/GA results cascade)
-> datasets (versions, splits, fold assignments, representations cascade).
Storage prefixes are removed only after the DB transaction commits.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.models import (Dataset, DatasetVersion, Model, ModelVersion, Prediction, RepresentationArtifact,
                        TrainingJob, Workflow)
from app.services.storage import storage


class DeletionError(ValueError):
    pass


def _model_version_delete(db: Session, mv: ModelVersion, files: list[str]) -> int:
    """Delete one model version (+ its predictions); the logical model too if no versions remain."""
    n_pred = db.execute(delete(Prediction).where(Prediction.model_version_id == mv.id)).rowcount
    files.append(mv.storage_prefix)
    model = mv.model
    db.delete(mv)
    db.flush()
    if not db.scalar(select(ModelVersion.id).where(ModelVersion.model_id == model.id)):
        db.delete(model)
    return n_pred


def delete_model_version(db: Session, mv: ModelVersion, *, by_admin: bool = False) -> dict[str, Any]:
    if mv.model.scope != "USER":
        raise DeletionError("built-in 모델은 여기서 삭제할 수 없음 (scripts.reset_builtin 사용)")
    if mv.status == "SERVING" and not by_admin:
        raise DeletionError("Admin이 승인해 서비스 중(SERVING)인 모델은 삭제할 수 없음, Admin에게 Archive를 요청하세요")
    files: list[str] = []
    n_pred = _model_version_delete(db, mv, files)
    return {"deleted_model_versions": 1, "deleted_predictions": n_pred, "_files": files}


def delete_workflow(db: Session, wf: Workflow, *, delete_model: bool) -> dict[str, Any]:
    if wf.status == "RUNNING":
        raise DeletionError("학습이 진행 중인 workflow는 삭제할 수 없음, 먼저 학습을 중단하세요")
    files: list[str] = []
    out = {"deleted_workflow": wf.id, "deleted_model_versions": 0, "deleted_predictions": 0,
           "deleted_training_jobs": 0, "deleted_datasets": 0, "kept_model_version_id": None}

    jobs = db.scalars(select(TrainingJob).where(TrainingJob.workflow_id == wf.id)).all()
    for job in jobs:
        if job.status in ("QUEUED", "RUNNING"):
            raise DeletionError("학습이 아직 진행 중임, 중단이 끝난 뒤 다시 시도하세요")
        mv = db.scalar(select(ModelVersion).where(ModelVersion.training_job_id == job.id))
        if mv is not None:
            if delete_model:
                if mv.status == "SERVING":
                    raise DeletionError("이 workflow의 모델이 서비스 중(SERVING)이라 함께 삭제할 수 없음, "
                                        "모델은 남기고 삭제하거나 Admin에게 Archive를 요청하세요")
                out["deleted_predictions"] += _model_version_delete(db, mv, files)
                out["deleted_model_versions"] += 1
            else:
                out["kept_model_version_id"] = mv.id  # model artifacts are self-contained in its own storage prefix
        db.delete(job)
        out["deleted_training_jobs"] += 1
    db.flush()

    # the dataset uploaded in step 1 belongs to this workflow
    dv_id = (wf.context or {}).get("dataset_version_id")
    dv = db.get(DatasetVersion, dv_id) if dv_id else None
    if dv is not None and not dv.dataset.is_builtin and dv.dataset.owner_id == wf.owner_id:
        still_used = db.scalar(select(TrainingJob.id).where(TrainingJob.dataset_version_id == dv.id))
        if not still_used:
            ds = dv.dataset
            for v in ds.versions:
                files.append(v.storage_key.rsplit("/", 1)[0])
                files += [r.storage_key.rsplit("/", 1)[0] for r in
                          db.scalars(select(RepresentationArtifact).where(RepresentationArtifact.dataset_version_id == v.id))]
            if not db.scalar(select(TrainingJob.id).join(DatasetVersion).where(DatasetVersion.dataset_id == ds.id)):
                db.execute(delete(Dataset).where(Dataset.id == ds.id))
                out["deleted_datasets"] = 1

    db.delete(wf)  # workflow_steps cascade
    db.flush()
    out["_files"] = files
    return out


def remove_files(result: dict[str, Any]) -> dict[str, Any]:
    """Call after commit: deletes the storage prefixes collected during the DB deletion."""
    for prefix in sorted(set(result.pop("_files", []))):
        storage.delete_prefix(prefix)
    return result
