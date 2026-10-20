from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import current_user
from app.models import GAExperiment, GAFoldResult, TrainingJob, User

router = APIRouter(prefix="/api/training", tags=["training"])


def job_to_dict(j: TrainingJob, detail: bool = False, db: Session | None = None) -> dict:
    d = {"training_job_id": j.id, "status": j.status, "stage": j.stage, "progress": j.progress,
         "task_type": j.task_type, "descriptor_mode": j.descriptor_mode, "sklearn_model": j.sklearn_model,
         "model_name": j.model_name, "dataset_version_id": j.dataset_version_id, "split_config_id": j.split_config_id,
         "config": j.config, "metrics": j.metrics, "error": j.error, "workflow_id": j.workflow_id,
         "created_at": j.created_at, "started_at": j.started_at, "finished_at": j.finished_at}
    if detail:
        d["folds"] = [{"fold": f.fold_no, "split_seed": f.split_seed, "n_train": f.n_train, "n_valid": f.n_valid,
                       "metrics": f.metrics, "n_selected": len(f.selected_descriptors),
                       "selected_descriptors": f.selected_descriptors} for f in j.fold_results]
        if db is not None:
            ga = db.scalar(select(GAExperiment).where(GAExperiment.training_job_id == j.id))
            if ga is not None:
                d["ga"] = {"provider": ga.provider, "status": ga.status, "candidate_k": ga.candidate_k,
                           "folds": [{"fold": g.fold_no, "flags": g.flags, "n_selected": len(g.selected_descriptors)}
                                     for g in db.scalars(select(GAFoldResult).where(GAFoldResult.ga_experiment_id == ga.id)
                                                         .order_by(GAFoldResult.fold_no))]}
    return d


@router.get("/jobs")
def list_jobs(db: Session = Depends(get_db), user: User = Depends(current_user)):
    rows = db.scalars(select(TrainingJob).where(TrainingJob.owner_id == user.id).order_by(TrainingJob.id.desc()).limit(100)).all()
    return {"jobs": [job_to_dict(j) for j in rows]}


@router.get("/jobs/{job_id}")
def get_job(job_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    j = db.get(TrainingJob, job_id)
    if j is None or (j.owner_id != user.id and not user.is_admin):
        raise HTTPException(404, "training job not found")
    return job_to_dict(j, detail=True, db=db)
