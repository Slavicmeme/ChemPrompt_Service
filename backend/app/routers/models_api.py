from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import current_user
from app.models import Model, ModelVersion, User
from app.services.prediction_service import visible_versions_query
from app.services import deletion_service
from app.services.registry_service import version_to_dict

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("")
def list_models(db: Session = Depends(get_db), user: User = Depends(current_user)):
    """Usable versions (built-in SERVING + own READY/SERVING) and all of the user's own versions."""
    rows = db.scalars(visible_versions_query(user).order_by(ModelVersion.id.desc())).all()
    mine = db.scalars(select(ModelVersion).join(Model).where(Model.owner_id == user.id).order_by(ModelVersion.id.desc())).all()
    seen, out = set(), []
    for mv in [*rows, *mine]:
        if mv.id not in seen:
            seen.add(mv.id)
            out.append(version_to_dict(mv) | {"usable": mv.status == "SERVING" or (mv.model.owner_id == user.id and mv.status == "READY")})
    return {"model_versions": out}


@router.get("/versions/{mv_id}")
def get_version(mv_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    mv = db.get(ModelVersion, mv_id)
    if mv is None or (mv.model.scope == "USER" and mv.model.owner_id != user.id and not user.is_admin):
        raise HTTPException(404, "model version not found")
    return version_to_dict(mv, include_metadata=True)


@router.delete("/versions/{mv_id}")
def delete_version(mv_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    """Delete one of the user's own custom model versions (and the predictions made with it)."""
    mv = db.get(ModelVersion, mv_id)
    if mv is None or mv.model.owner_id != user.id:
        raise HTTPException(404, "model version not found")
    try:
        result = deletion_service.delete_model_version(db, mv)
    except deletion_service.DeletionError as exc:
        db.rollback()
        raise HTTPException(409, str(exc))
    db.commit()
    return deletion_service.remove_files(result)
