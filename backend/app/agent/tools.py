"""Agent tools (docs/12_AGENT_MODE.md). Each tool runs with the requesting user's permissions in its
own DB session. Prediction values only ever come from ``prediction_service``."""
from __future__ import annotations

import json
from typing import Any, Callable

from sqlalchemy import select

from app.db import session_scope
from app.models import Prediction, TrainingJob, User, Workflow
from app.routers.properties import list_properties
from app.routers.training import job_to_dict
from app.services import prediction_service as ps
from app.services import workflow_service as wfs


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required, "additionalProperties": False}}}


TOOL_SPECS = [
    _fn("list_properties", "예측 가능한 built-in property 목록과 serving 모델 정보를 조회한다.", {}, []),
    _fn("get_property_info", "특정 property의 설명, 단위, task type, serving 모델의 검증 성능을 조회한다.",
        {"property": {"type": "string", "description": "FreeSolv | ESOL | Lipo | ClinTox"}}, ["property"]),
    _fn("predict_property", "SMILES 분자에 대해 ChemPrompt 5-fold ensemble 모델로 property를 예측한다. 예측 값은 반드시 이 tool로만 얻는다.",
        {"smiles": {"type": "string"}, "property": {"type": "string", "description": "built-in property code"},
         "model_version_id": {"type": "integer", "description": "사용자 custom 모델 사용 시"}}, ["smiles"]),
    _fn("get_prediction_result", "prediction id로 저장된 예측 결과와 fold별 결과를 조회한다.",
        {"prediction_id": {"type": "integer"}}, ["prediction_id"]),
    _fn("get_prediction_history", "사용자의 최근 예측 기록을 조회한다.", {"limit": {"type": "integer"}}, []),
    _fn("get_training_status", "custom training job의 상태/단계/fold별 성능을 조회한다. id가 없으면 최근 job들.",
        {"training_job_id": {"type": "integer"}}, []),
    _fn("get_workflow_status", "custom training workflow의 현재 step과 다음에 가능한 step을 조회한다. id가 없으면 최근 workflow들.",
        {"workflow_id": {"type": "integer"}}, []),
]


def _list_properties(user: User, args: dict, ctx: dict) -> Any:
    with session_scope() as db:
        return {"properties": list_properties(db)}


def _get_property_info(user: User, args: dict, ctx: dict) -> Any:
    with session_scope() as db:
        for p in list_properties(db):
            if p["code"].lower() == str(args.get("property", "")).lower():
                return p
    return {"error": f"unknown property {args.get('property')}"}


def _predict(user: User, args: dict, ctx: dict) -> Any:
    with session_scope() as db:
        u = db.get(User, user.id)
        try:
            mv = ps.resolve_model_version(db, u, model_version_id=args.get("model_version_id"), property_code=args.get("property"))
        except ps.PredictionError as exc:
            return {"error": str(exc)}
        p = ps.predict(db, u, str(args.get("smiles", "")), mv, source="AGENT", conversation_id=ctx.get("conversation_id"))
        db.flush()
        out = ps.prediction_to_dict(p, with_folds=True)
        out["unit"] = mv.model.prop.unit if mv.model.prop else None
        out["validation_metrics"] = {k: v.get("mean") for k, v in (mv.metrics or {}).items()}
        return out


def _get_prediction(user: User, args: dict, ctx: dict) -> Any:
    with session_scope() as db:
        p = db.get(Prediction, int(args.get("prediction_id", 0)))
        if p is None or p.user_id != user.id:
            return {"error": "prediction not found"}
        return ps.prediction_to_dict(p, with_folds=True)


def _history(user: User, args: dict, ctx: dict) -> Any:
    limit = max(1, min(int(args.get("limit") or 10), 50))
    with session_scope() as db:
        rows = db.scalars(select(Prediction).where(Prediction.user_id == user.id).order_by(Prediction.id.desc()).limit(limit))
        return {"predictions": [ps.prediction_to_dict(p) for p in rows]}


def _training(user: User, args: dict, ctx: dict) -> Any:
    with session_scope() as db:
        if args.get("training_job_id"):
            j = db.get(TrainingJob, int(args["training_job_id"]))
            if j is None or j.owner_id != user.id:
                return {"error": "training job not found"}
            return job_to_dict(j, detail=True, db=db)
        rows = db.scalars(select(TrainingJob).where(TrainingJob.owner_id == user.id).order_by(TrainingJob.id.desc()).limit(5))
        return {"jobs": [job_to_dict(j) for j in rows]}


def _workflow(user: User, args: dict, ctx: dict) -> Any:
    with session_scope() as db:
        if args.get("workflow_id"):
            wf = db.get(Workflow, int(args["workflow_id"]))
            if wf is None or wf.owner_id != user.id:
                return {"error": "workflow not found"}
            return wfs.to_dict(wf)
        rows = db.scalars(select(Workflow).where(Workflow.owner_id == user.id).order_by(Workflow.id.desc()).limit(5))
        return {"workflows": [wfs.to_dict(w) for w in rows]}


TOOLS: dict[str, Callable[[User, dict, dict], Any]] = {
    "list_properties": _list_properties, "get_property_info": _get_property_info, "predict_property": _predict,
    "get_prediction_result": _get_prediction, "get_prediction_history": _history,
    "get_training_status": _training, "get_workflow_status": _workflow,
}


def run_tool(user: User, name: str, raw_args: str, ctx: dict) -> dict:
    if name not in TOOLS:
        return {"error": f"unknown tool {name}"}
    try:
        args = json.loads(raw_args or "{}")
    except json.JSONDecodeError:
        return {"error": "invalid JSON arguments"}
    try:
        return TOOLS[name](user, args, ctx)
    except Exception as exc:  # noqa: BLE001 — surface tool failure to the model, never invent results
        return {"error": f"{type(exc).__name__}: {exc}"}
