from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.deps import current_user
from app.models import Prediction, User
from app.services import prediction_service as ps

router = APIRouter(prefix="/api/predictions", tags=["predictions"])


class PredictRequest(BaseModel):
    smiles: list[str]
    property: str | None = None
    model_version_id: int | None = None


@router.post("")
def predict(body: PredictRequest, db: Session = Depends(get_db), user: User = Depends(current_user)):
    smiles = [s.strip() for s in body.smiles if s and s.strip()]
    if not smiles:
        raise HTTPException(400, "SMILES를 1개 이상 입력하세요.")
    if len(smiles) > settings.max_prediction_batch:
        raise HTTPException(400, f"한 번에 최대 {settings.max_prediction_batch}개까지 예측할 수 있습니다.")
    try:
        mv = ps.resolve_model_version(db, user, model_version_id=body.model_version_id, property_code=body.property)
    except ps.PredictionError as exc:
        raise HTTPException(404, str(exc))
    preds = [ps.predict(db, user, s, mv, source="WEB") for s in smiles]
    db.commit()
    return {"predictions": [ps.prediction_to_dict(p, with_folds=True) for p in preds]}


@router.get("")
def history(limit: int = 50, offset: int = 0, db: Session = Depends(get_db), user: User = Depends(current_user)):
    rows = db.scalars(select(Prediction).where(Prediction.user_id == user.id).order_by(Prediction.id.desc())
                      .limit(min(limit, 200)).offset(offset)).all()
    return {"predictions": [ps.prediction_to_dict(p) for p in rows]}


@router.get("/{prediction_id}")
def get_prediction(prediction_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    p = db.get(Prediction, prediction_id)
    if p is None or (p.user_id != user.id and not user.is_admin):
        raise HTTPException(404, "prediction not found")
    return ps.prediction_to_dict(p, with_folds=True)
