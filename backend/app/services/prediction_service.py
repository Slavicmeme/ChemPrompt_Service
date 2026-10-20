"""Prediction (docs/05_PREDICTION.md).

SMILES -> RDKit descriptors -> fold descriptor flag -> ChemPrompt prompt -> representation
       -> fold-specific sklearn model -> aggregation

Regression:      prediction  = mean(fold_predictions)
Classification:  probability = mean(fold_probabilities), label = probability >= threshold

The prediction value is always computed here from the fold models. The chat agent only calls this
service through a tool and never produces values itself.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Model, ModelVersion, Prediction, PredictionFoldResult, Property, User
from app.services import registry_service
from app.services.descriptor_service import canonicalize, flag_to_str
from app.services.representation import embed_prompts


class PredictionError(ValueError):
    pass


def _positive_proba(model, X: np.ndarray) -> np.ndarray:
    """Probability of class 1 per task — same extraction as ``ScikitLearnModel._prediction_scores``."""
    proba = model.predict_proba(X)
    if isinstance(proba, list):  # MultiOutputClassifier
        return np.asarray([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in proba]).T
    if proba.ndim == 2 and proba.shape[1] == 2:
        return proba[:, 1:2]
    return proba


def predict_with_bundle(bundle: registry_service.ModelBundle, smiles: str) -> dict[str, Any]:
    meta = bundle.metadata
    task = meta["task_type"]
    targets = meta.get("target_columns") or ["label"]
    # embed once per distinct flag (FULL mode: one embedding shared by all folds)
    by_flag: dict[str, np.ndarray] = {}
    for f in bundle.folds:
        key = flag_to_str(f["flag"])
        if key not in by_flag:
            prompt = bundle.engine.prompt(smiles, f["flag"])
            by_flag[key] = embed_prompts([prompt], expected=bundle.representation)
    fold_outputs = []
    for f in bundle.folds:
        X = by_flag[flag_to_str(f["flag"])]
        if task == "regression":
            v = float(np.asarray(f["model"].predict(X)).ravel()[0])
            fold_outputs.append({"fold": f["fold"], "value": v})
        else:
            p = _positive_proba(f["model"], X)[0].ravel().tolist()
            fold_outputs.append({"fold": f["fold"], "probabilities": dict(zip(targets, map(float, p)))})

    if task == "regression":
        vals = np.array([o["value"] for o in fold_outputs])
        result = {"value": float(vals.mean()), "fold_std": float(vals.std()), "target": targets[0]}
    else:
        thr = meta.get("classification_threshold") or settings.classification_threshold
        tasks = []
        for t in targets:
            ps = np.array([o["probabilities"][t] for o in fold_outputs])
            prob = float(ps.mean())
            tasks.append({"name": t, "probability": prob, "label": int(prob >= thr), "fold_std": float(ps.std())})
        result = {"tasks": tasks, "threshold": thr}
    return {"result": result, "fold_outputs": fold_outputs}


def visible_versions_query(user: User):
    """Built-in SERVING models for everyone; a user's own READY/SERVING custom models for the owner."""
    q = select(ModelVersion).join(Model)
    return q.where(or_(
        ModelVersion.status == "SERVING",
        (Model.owner_id == user.id) & ModelVersion.status.in_(("READY", "SERVING")),
    ))


def resolve_model_version(db: Session, user: User, *, model_version_id: int | None = None,
                          property_code: str | None = None) -> ModelVersion:
    if model_version_id is not None:
        mv = db.get(ModelVersion, model_version_id)
        if mv is None:
            raise PredictionError("model version not found")
        is_owner = mv.model.owner_id == user.id
        # SERVING (admin-approved) models are public; READY models only for their owner and admins
        if mv.status == "SERVING" or (mv.status == "READY" and (is_owner or user.is_admin)):
            return mv
        if mv.model.scope == "USER" and not is_owner and not user.is_admin:
            raise PredictionError("model version not found")
        raise PredictionError(f"model version {mv.id} is not available for prediction (status {mv.status})")
    if property_code:
        mv = db.scalar(
            select(ModelVersion).join(Model).join(Property, Model.property_id == Property.id)
            .where(Property.code == property_code, Model.scope == "BUILTIN", ModelVersion.status == "SERVING")
            .order_by(ModelVersion.id.desc())
        )
        if mv is None:
            raise PredictionError(f"'{property_code}' 에 대해 serving 중인 모델이 없습니다.")
        return mv
    raise PredictionError("model_version_id 또는 property를 지정하세요.")


def predict(db: Session, user: User, smiles: str, mv: ModelVersion, *, source: str = "WEB",
            conversation_id: int | None = None) -> Prediction:
    smiles = (smiles or "").strip()
    canonical = canonicalize(smiles) if smiles else None
    pred = Prediction(
        user_id=user.id, model_version_id=mv.id, smiles=smiles[:2000], task_type=mv.task_type,
        model_version_label=mv.label, split_strategy=mv.split_strategy, n_folds=mv.n_folds,
        split_config_id=mv.split_config_id, split_seeds=mv.split_seeds, descriptor_mode=mv.descriptor_mode,
        representation_model=mv.representation_model, prompt_version=mv.prompt_version,
        aggregation=registry_service.aggregation_name(mv.task_type), source=source, conversation_id=conversation_id,
        status="FAILED",
    )
    if canonical is None:
        pred.error = "RDKit이 SMILES를 해석하지 못했습니다."
        db.add(pred)
        db.flush()
        return pred
    try:
        bundle = registry_service.load_bundle(db, mv)
        out = predict_with_bundle(bundle, smiles)
        pred.status, pred.result = "SUCCEEDED", out["result"] | {"canonical_smiles": canonical}
        db.add(pred)
        db.flush()
        for fo in out["fold_outputs"]:
            db.add(PredictionFoldResult(prediction_id=pred.id, fold_no=fo["fold"], output=fo))
    except Exception as exc:  # noqa: BLE001 — failure is recorded, not hidden
        pred.error = str(exc)[:2000]
        db.add(pred)
    db.flush()
    return pred


def prediction_to_dict(p: Prediction, with_folds: bool = False) -> dict[str, Any]:
    d = {
        "prediction_id": p.id, "smiles": p.smiles, "status": p.status, "task_type": p.task_type, "result": p.result,
        "error": p.error, "model_version_id": p.model_version_id, "model_version": p.model_version_label,
        "split_strategy": p.split_strategy, "n_folds": p.n_folds, "split_config_id": p.split_config_id,
        "split_seeds": p.split_seeds, "descriptor_mode": p.descriptor_mode,
        "representation_model": p.representation_model, "prompt_version": p.prompt_version,
        "aggregation": p.aggregation, "source": p.source, "created_at": p.created_at,
    }
    prop = p.model_version.model.prop if p.model_version is not None else None
    d["property"] = None if prop is None else {"code": prop.code, "name_ko": prop.name_ko, "unit": prop.unit,
                                              "target_info": prop.target_info}
    if with_folds:
        d["fold_results"] = [fr.output for fr in p.fold_results]
    return d
