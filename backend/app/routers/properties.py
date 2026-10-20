from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import current_user
from app.models import Model, ModelVersion, Property, User

router = APIRouter(prefix="/api/properties", tags=["properties"])


def list_properties(db: Session) -> list[dict]:
    out = []
    for p in db.scalars(select(Property).order_by(Property.id)):
        mv = db.scalar(select(ModelVersion).join(Model).where(Model.property_id == p.id, Model.scope == "BUILTIN",
                                                              ModelVersion.status == "SERVING").order_by(ModelVersion.id.desc()))
        out.append({
            "code": p.code, "name": p.name, "name_ko": p.name_ko, "description": p.description,
            "task_type": p.task_type, "unit": p.unit, "target_columns": p.target_columns, "target_info": p.target_info,
            "serving_model": None if mv is None else {
                "model_version_id": mv.id, "label": mv.label, "descriptor_mode": mv.descriptor_mode,
                "split_strategy": mv.split_strategy, "representation_model": mv.representation_model,
                "metrics": {k: v.get("mean") for k, v in (mv.metrics or {}).items()},
            },
        })
    return out


@router.get("")
def properties(db: Session = Depends(get_db), user: User = Depends(current_user)):
    return {"properties": list_properties(db)}
