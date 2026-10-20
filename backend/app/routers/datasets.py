from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.deps import current_user
from app.models import Dataset, DatasetVersion, User
from app.services import dataset_service as ds

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


async def read_upload(file: UploadFile) -> bytes:
    data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"파일은 {settings.max_upload_mb}MB 이하여야 합니다.")
    return data


@router.post("")
async def upload(name: str = Form(...), description: str = Form(""), file: UploadFile = File(...),
                 db: Session = Depends(get_db), user: User = Depends(current_user)):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(400, "CSV 파일만 업로드할 수 있습니다.")
    data = await read_upload(file)
    try:
        dv = ds.create_dataset_upload(db, owner_id=user.id, name=name, description=description or None,
                                      filename=file.filename, data=data)
    except ds.DatasetError as exc:
        raise HTTPException(400, str(exc))
    db.commit()
    return ds.dataset_to_dict(dv) | {"suggested_mapping": ds.suggest_mapping(dv)}


@router.get("")
def list_datasets(db: Session = Depends(get_db), user: User = Depends(current_user)):
    rows = db.scalars(select(DatasetVersion).join(Dataset).where(Dataset.owner_id == user.id)
                      .order_by(DatasetVersion.id.desc())).all()
    return {"datasets": [ds.dataset_to_dict(dv) for dv in rows]}


@router.get("/versions/{dataset_version_id}")
def get_version(dataset_version_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    dv = db.get(DatasetVersion, dataset_version_id)
    if dv is None or (dv.dataset.owner_id != user.id and not user.is_admin):
        raise HTTPException(404, "dataset not found")
    return ds.dataset_to_dict(dv) | {"suggested_mapping": ds.suggest_mapping(dv)}
