"""Admin API (docs/13_ADMIN.md). Every endpoint depends on ``require_admin`` (server-side role check).
Admins never provide storage paths — packages/CSVs are uploaded and Storage Service assigns keys."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import jobs
from app.db import get_db
from app.deps import require_admin
from app.models import Dataset, DatasetVersion, Model, ModelVersion, Property, TrainingJob, User
from app.routers.datasets import read_upload
from app.routers.training import job_to_dict
from app.services import dataset_service as ds
from app.services import registry_service as reg
from app.services import training_service
from app.services.representation import current_info, get_backend

router = APIRouter(prefix="/api/admin", tags=["admin"])


def _mv(db: Session, mv_id: int) -> ModelVersion:
    mv = db.get(ModelVersion, mv_id)
    if mv is None:
        raise HTTPException(404, "model version not found")
    return mv


class Review(BaseModel):
    note: str | None = None


@router.get("/overview")
def overview(db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    info = current_info()
    health = None
    backend = get_backend()
    if hasattr(backend, "health"):
        try:
            health = backend.health()
        except Exception as exc:  # noqa: BLE001
            health = {"status": "unreachable", "error": str(exc)}
    counts = {s: db.query(ModelVersion).filter(ModelVersion.status == s).count()
              for s in ("UPLOADED", "VALIDATING", "READY", "SERVING", "ARCHIVED", "INVALID")}
    return {"representation": {"backend": info.backend, "model": info.model, "label": info.label, "health": health},
            "model_version_counts": counts,
            "users": db.query(User).count(),
            "training_jobs": {s: db.query(TrainingJob).filter(TrainingJob.status == s).count()
                              for s in ("QUEUED", "RUNNING", "SUCCEEDED", "FAILED")}}


@router.get("/model-versions")
def model_versions(status: str | None = None, scope: str | None = None, db: Session = Depends(get_db),
                   admin: User = Depends(require_admin)):
    q = select(ModelVersion).join(Model)
    if status:
        q = q.where(ModelVersion.status == status.upper())
    if scope:
        q = q.where(Model.scope == scope.upper())
    return {"model_versions": [reg.version_to_dict(mv) for mv in db.scalars(q.order_by(ModelVersion.id.desc()).limit(200))]}


@router.get("/model-versions/{mv_id}")
def model_version(mv_id: int, db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    return reg.version_to_dict(_mv(db, mv_id), include_metadata=True)


@router.post("/model-versions/{mv_id}/validate")
def validate(mv_id: int, db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    mv = _mv(db, mv_id)
    try:
        if mv.status in ("READY", "INVALID", "UPLOADED"):
            report = reg.validate_version(db, mv)
        else:
            raise reg.RegistryError(f"cannot validate a {mv.status} version")
    except reg.RegistryError as exc:
        raise HTTPException(409, str(exc))
    db.commit()
    return {"status": mv.status, "report": report}


@router.post("/model-versions/{mv_id}/serve")
def serve(mv_id: int, body: Review, db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    """Approve + switch serving (also used to approve a user's custom model)."""
    try:
        mv = reg.set_serving(db, _mv(db, mv_id), admin, body.note)
    except reg.RegistryError as exc:
        raise HTTPException(409, str(exc))
    db.commit()
    return reg.version_to_dict(mv)


@router.post("/model-versions/{mv_id}/archive")
def archive(mv_id: int, body: Review, db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    try:
        mv = reg.archive(db, _mv(db, mv_id), admin, body.note)
    except reg.RegistryError as exc:
        raise HTTPException(409, str(exc))
    db.commit()
    return reg.version_to_dict(mv)


@router.post("/models/upload")
async def upload_model(property_code: str = Form(...), model_name: str = Form(...), file: UploadFile = File(...),
                       db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    prop = db.scalar(select(Property).where(Property.code == property_code))
    if prop is None:
        raise HTTPException(404, "unknown property")
    data = await read_upload(file)
    try:
        mv = reg.upload_package(db, admin=admin, property_id=prop.id, model_name=model_name, data=data)
        db.commit()
        reg.validate_version(db, mv)
    except reg.RegistryError as exc:
        db.rollback()
        raise HTTPException(400, str(exc))
    db.commit()
    return reg.version_to_dict(mv, include_metadata=True)


# ---- built-in datasets & training -----------------------------------------------------------------
@router.post("/builtin-datasets")
async def upload_builtin_dataset(property_code: str = Form(...), smiles_column: str = Form("smiles"),
                                 file: UploadFile = File(...), db: Session = Depends(get_db),
                                 admin: User = Depends(require_admin)):
    prop = db.scalar(select(Property).where(Property.code == property_code))
    if prop is None:
        raise HTTPException(404, "unknown property")
    data = await read_upload(file)
    try:
        dv = create_builtin_version(db, prop, file.filename or "data.csv", data, smiles_column)
    except ds.DatasetError as exc:
        db.rollback()
        raise HTTPException(400, str(exc))
    db.commit()
    return ds.dataset_to_dict(dv)


def create_builtin_version(db: Session, prop: Property, filename: str, data: bytes, smiles_column: str) -> DatasetVersion:
    dataset = db.get(Dataset, prop.dataset_id) if prop.dataset_id else None
    if dataset is None:
        dv = ds.create_dataset_upload(db, owner_id=None, name=prop.code, description=prop.name, filename=filename,
                                      data=data, is_builtin=True)
        prop.dataset_id = dv.dataset_id
    else:
        dv = ds.add_version(db, dataset, filename=filename, data=data)
    ds.set_mapping(db, dv, smiles_column, list(prop.target_columns))
    ds.validate(db, dv)
    if dv.status == "VALIDATED":
        ds.set_task_type(db, dv, prop.task_type)
    return dv


@router.get("/builtin-datasets")
def builtin_datasets(db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    rows = db.scalars(select(DatasetVersion).join(Dataset).where(Dataset.is_builtin).order_by(DatasetVersion.id.desc())).all()
    return {"datasets": [ds.dataset_to_dict(dv) for dv in rows]}


class BuiltinTrain(BaseModel):
    property_code: str
    dataset_version_id: int
    descriptor_mode: str = "FULL"
    split_strategy: str = "RANDOM"
    model_name: str | None = None


@router.post("/builtin-training")
def builtin_training(body: BuiltinTrain, db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    prop = db.scalar(select(Property).where(Property.code == body.property_code))
    dv = db.get(DatasetVersion, body.dataset_version_id)
    if prop is None or dv is None or not dv.dataset.is_builtin or prop.dataset_id != dv.dataset_id:
        raise HTTPException(400, "built-in dataset version of this property가 필요합니다.")
    try:
        job = training_service.create_job(db, owner_id=None, dv=dv, descriptor_mode=body.descriptor_mode,
                                          split_strategy=body.split_strategy, property_id=prop.id,
                                          model_name=body.model_name or f"{prop.code} ChemPrompt {body.descriptor_mode.upper()}")
    except (training_service.TrainingError, ValueError) as exc:
        raise HTTPException(400, str(exc))
    db.commit()
    jobs.submit(job.id)
    return job_to_dict(job)


@router.post("/training-jobs/{job_id}/cancel")
def cancel_training_job(job_id: int, db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    job = db.get(TrainingJob, job_id)
    if job is None:
        raise HTTPException(404, "training job not found")
    if job.status not in ("QUEUED", "RUNNING"):
        raise HTTPException(409, f"이미 끝난 학습임 ({job.status})")
    return {"training_job_id": job_id, "status": jobs.request_cancel(job_id)}


@router.get("/training-jobs")
def training_jobs(db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    rows = db.scalars(select(TrainingJob).order_by(TrainingJob.id.desc()).limit(200)).all()
    return {"jobs": [job_to_dict(j) | {"owner_id": j.owner_id} for j in rows]}


@router.get("/datasets")
def all_datasets(db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    rows = db.scalars(select(DatasetVersion).order_by(DatasetVersion.id.desc()).limit(200)).all()
    return {"datasets": [ds.dataset_to_dict(dv) | {"owner_id": dv.dataset.owner_id} for dv in rows]}


@router.get("/users")
def users(db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    return {"users": [{"id": u.id, "username": u.username, "role": u.role, "is_active": u.is_active,
                       "created_at": u.created_at} for u in db.scalars(select(User).order_by(User.id))]}
