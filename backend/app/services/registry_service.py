"""Model registry (docs/08_MODEL_REGISTRY.md, docs/09_STORAGE_POLICY.md, docs/13_ADMIN.md).

State machine: UPLOADED -> VALIDATING -> READY -> SERVING -> ARCHIVED   (failed validation -> INVALID)
Upload != Serving: only an Admin can move a READY version to SERVING.
"""
from __future__ import annotations

import io
import json
import re
import secrets
import zipfile
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import sklearn
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Model, ModelArtifact, ModelVersion, Property, TrainingJob, User
from app.services.descriptor_service import DescriptorEngine, str_to_flag
from app.services.representation import RepresentationInfo, current_info
from app.services.storage import StorageError, storage

N_FOLDS = 5
REQUIRED_METADATA = ("task_type", "descriptor_method", "split_strategy", "n_folds", "split_seeds",
                     "representation", "prompt_version", "sklearn_model", "fold_artifacts")
TRANSITIONS = {
    "UPLOADED": {"VALIDATING"},
    "VALIDATING": {"READY", "INVALID"},
    "READY": {"SERVING", "ARCHIVED", "VALIDATING"},
    "SERVING": {"ARCHIVED", "READY"},
    "INVALID": {"VALIDATING", "ARCHIVED"},
    "ARCHIVED": {"READY"},
}


class RegistryError(ValueError):
    pass


def _transition(mv: ModelVersion, target: str) -> None:
    if target not in TRANSITIONS.get(mv.status, set()):
        raise RegistryError(f"cannot move model version from {mv.status} to {target}")
    mv.status = target


def _next_version(db: Session, model_id: int) -> int:
    return (db.scalar(select(func.max(ModelVersion.version)).where(ModelVersion.model_id == model_id)) or 0) + 1


def _record(db: Session, mv: ModelVersion, artifact_type: str, meta: dict, fold_no: int | None = None) -> dict:
    db.add(ModelArtifact(model_version_id=mv.id, artifact_type=artifact_type, fold_no=fold_no,
                         storage_key=meta["storage_key"], checksum=meta["checksum"], size_bytes=meta["size_bytes"]))
    return meta


def get_or_create_model(db: Session, *, owner_id: int | None, property_id: int | None, scope: str, name: str) -> Model:
    m = db.scalar(select(Model).where(Model.owner_id.is_(owner_id) if owner_id is None else Model.owner_id == owner_id,
                                      Model.scope == scope, Model.name == name))
    if m is None:
        m = Model(owner_id=owner_id, property_id=property_id, scope=scope, name=name)
        db.add(m)
        db.flush()
    return m


def register_from_training(db: Session, job: TrainingJob, *, catalog: dict, fold_payloads: list[dict], summary: dict,
                           sklearn_version: str) -> ModelVersion:
    scope = "BUILTIN" if job.owner_id is None else "USER"
    prop = db.get(Property, job.property_id) if job.property_id else None
    model = get_or_create_model(db, owner_id=job.owner_id, property_id=job.property_id, scope=scope, name=job.model_name)
    version = _next_version(db, model.id)
    prefix = storage.model_prefix(scope=scope, model_id=model.id, version=version, user_id=job.owner_id,
                                  property_code=prop.code if prop else None)
    if storage.list_keys(prefix):
        raise RegistryError(f"storage prefix already used: {prefix}")
    rep = job.config["representation"]
    dv = job.dataset_version_id
    from app.models import DatasetVersion, SplitConfig  # local import to avoid cycles in type checkers

    dvo = db.get(DatasetVersion, dv)
    sc = db.get(SplitConfig, job.split_config_id)
    mv = ModelVersion(
        model_id=model.id, version=version, training_job_id=job.id, split_config_id=sc.id, status="UPLOADED",
        task_type=job.task_type, target_columns=dvo.target_columns, descriptor_mode=job.descriptor_mode,
        split_strategy=sc.strategy, n_folds=sc.n_folds, split_seeds=sc.seeds,
        representation_model=RepresentationInfo(rep["backend"], rep["model"]).label,
        prompt_version=job.config["prompt_version"], sklearn_model=job.sklearn_model, sklearn_version=sklearn_version,
        storage_prefix=prefix, metadata_={}, metrics=summary,
    )
    db.add(mv)
    db.flush()

    _record(db, mv, "DESCRIPTOR_CATALOG", storage.put_json(f"{prefix}/descriptor_catalog.json", catalog))
    _record(db, mv, "SPLIT", storage.put_json(f"{prefix}/split.json", {
        "split_config_id": sc.id, "strategy": sc.strategy, "n_folds": sc.n_folds, "seed": sc.seed, "seeds": sc.seeds,
        "config": sc.config, "config_checksum": sc.checksum, "assignment_checksum": sc.assignment_checksum,
    }))
    fold_artifacts = []
    for p in fold_payloads:
        k = p["fold_no"]
        m_meta = _record(db, mv, "FOLD_MODEL", storage.put_bytes(f"{prefix}/fold_{k}/model.joblib", p["model_bytes"]), k)
        s_meta = _record(db, mv, "SELECTED_DESCRIPTORS", storage.put_json(f"{prefix}/fold_{k}/selected_descriptors.json", {
            "fold": k, "flag": p["flag"], "selected_descriptors": p["selected"], "n_selected": len(p["selected"]),
        }), k)
        entry = {"fold": k, "split_seed": p["seed"], "n_train": p["n_train"], "n_valid": p["n_valid"],
                 "metrics": p["metrics"], "model": {"key": f"fold_{k}/model.joblib", "sha256": m_meta["checksum"]},
                 "selected_descriptors": {"key": f"fold_{k}/selected_descriptors.json", "sha256": s_meta["checksum"]}}
        if p.get("ga"):
            g_meta = _record(db, mv, "GA_FLAGS", storage.put_json(f"{prefix}/fold_{k}/ga_flags.json", p["ga"]), k)
            entry["ga_flags"] = {"key": f"fold_{k}/ga_flags.json", "sha256": g_meta["checksum"]}
        fold_artifacts.append(entry)

    owner = db.get(User, job.owner_id) if job.owner_id else None
    metadata = {
        "schema_version": 1,
        "model_id": model.id, "model_name": model.name, "version": version, "scope": scope,
        "owner": owner.username if owner else "builtin", "owner_id": job.owner_id,
        "dataset": {"dataset_id": dvo.dataset_id, "dataset_version_id": dvo.id, "version": dvo.version, "checksum": dvo.checksum},
        "property": prop.code if prop else None,
        "task_type": job.task_type, "target_columns": dvo.target_columns,
        "descriptor_method": job.descriptor_mode,
        "feature_selection": job.config.get("feature_selection"),
        "ga_provider": job.config.get("ga_provider"),
        "split_strategy": sc.strategy, "n_folds": sc.n_folds, "split_seeds": sc.seeds,
        "split_config": {"id": sc.id, "checksum": sc.checksum, "assignment_checksum": sc.assignment_checksum},
        "representation": rep, "prompt_version": job.config["prompt_version"],
        "preprocessing_version": settings.preprocessing_version,
        "sklearn_model": job.sklearn_model, "sklearn_version": sklearn_version,
        "aggregation": aggregation_name(job.task_type),
        "classification_threshold": settings.classification_threshold if job.task_type != "regression" else None,
        "fold_artifacts": fold_artifacts,
        "descriptor_catalog": "descriptor_catalog.json",
        "metrics": summary,
        "training_job_id": job.id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _record(db, mv, "METADATA", storage.put_json(f"{prefix}/metadata.json", metadata))
    mv.metadata_ = metadata
    db.flush()
    return mv


def aggregation_name(task_type: str) -> str:
    return "mean(fold_predictions)" if task_type == "regression" else "mean(fold_probabilities)+threshold"


# ---------------------------------------------------------------------------------------------------
# Runtime bundle (used by prediction + validation)
# ---------------------------------------------------------------------------------------------------
class ModelBundle:
    def __init__(self, mv_id: int, metadata: dict, engine: DescriptorEngine, folds: list[dict]):
        self.mv_id, self.metadata, self.engine, self.folds = mv_id, metadata, engine, folds

    @property
    def representation(self) -> RepresentationInfo:
        r = self.metadata["representation"]
        return RepresentationInfo(r["backend"], r["model"])


def _checked(prefix: str, rel: str, sha: str) -> bytes:
    data = storage.get_bytes(f"{prefix}/{rel}")
    from app.services.storage import sha256_bytes

    if sha256_bytes(data) != sha:
        raise RegistryError(f"checksum mismatch: {rel}")
    return data


@lru_cache(maxsize=32)
def _load_bundle_cached(mv_id: int, prefix: str, metadata_sha: str) -> ModelBundle:
    meta_bytes = storage.get_bytes(f"{prefix}/metadata.json")
    from app.services.storage import sha256_bytes

    if sha256_bytes(meta_bytes) != metadata_sha:
        raise RegistryError("metadata.json checksum mismatch")
    metadata = json.loads(meta_bytes)
    engine = DescriptorEngine.from_catalog(storage.get_json(f"{prefix}/{metadata['descriptor_catalog']}"))
    folds = []
    for fa in sorted(metadata["fold_artifacts"], key=lambda x: x["fold"]):
        model = joblib.load(io.BytesIO(_checked(prefix, fa["model"]["key"], fa["model"]["sha256"])))
        sel = json.loads(_checked(prefix, fa["selected_descriptors"]["key"], fa["selected_descriptors"]["sha256"]))
        flag = str_to_flag(sel["flag"])
        if len(flag) != len(engine.descriptor_names):
            raise RegistryError(f"fold {fa['fold']}: flag length {len(flag)} != {len(engine.descriptor_names)} descriptors")
        folds.append({"fold": fa["fold"], "model": model, "flag": flag})
    return ModelBundle(mv_id, metadata, engine, folds)


def load_bundle(db: Session, mv: ModelVersion) -> ModelBundle:
    art = db.scalar(select(ModelArtifact).where(ModelArtifact.model_version_id == mv.id, ModelArtifact.artifact_type == "METADATA"))
    if art is None:
        raise RegistryError("metadata artifact missing")
    return _load_bundle_cached(mv.id, mv.storage_prefix, art.checksum)


def validate_version(db: Session, mv: ModelVersion) -> dict[str, Any]:
    """Structural + checksum + smoke-inference validation. Ends in READY or INVALID."""
    from app.services.prediction_service import predict_with_bundle

    _transition(mv, "VALIDATING")
    db.flush()
    report: dict[str, Any] = {"checks": [], "warnings": [], "errors": []}
    try:
        for a in mv.artifacts:
            if storage.checksum(a.storage_key) != a.checksum:
                report["errors"].append(f"checksum mismatch: {a.storage_key}")
        report["checks"].append(f"{len(mv.artifacts)} artifacts checksum-verified")
        meta = storage.get_json(f"{mv.storage_prefix}/metadata.json")
        missing = [k for k in REQUIRED_METADATA if k not in meta]
        if missing:
            report["errors"].append(f"metadata missing fields: {missing}")
        if meta.get("n_folds") != N_FOLDS or len(meta.get("fold_artifacts", [])) != N_FOLDS:
            report["errors"].append("V1 serving requires exactly 5 fold models")
        if meta.get("sklearn_version") and meta["sklearn_version"] != sklearn.__version__:
            report["warnings"].append(f"trained with sklearn {meta['sklearn_version']}, runtime {sklearn.__version__}")
        if not report["errors"]:
            bundle = load_bundle(db, mv)
            report["checks"].append(f"{len(bundle.folds)} fold models loaded; {len(bundle.engine.descriptor_names)} descriptors restored")
            live = current_info()
            if (live.backend, live.model) != (bundle.representation.backend, bundle.representation.model):
                report["errors"].append(f"representation mismatch: model={bundle.representation.label}, service={live.label}")
            else:
                out = predict_with_bundle(bundle, "CCO")
                report["checks"].append(f"smoke prediction OK (CCO -> {json.dumps(out['result'])[:120]})")
    except (StorageError, RegistryError, KeyError, ValueError) as exc:
        report["errors"].append(str(exc))
    _transition(mv, "INVALID" if report["errors"] else "READY")
    mv.validation_report = report
    db.flush()
    return report


def set_serving(db: Session, mv: ModelVersion, admin: User | None, note: str | None = None) -> ModelVersion:
    """admin=None: system action (e.g. built-in bootstrap before any admin account exists)."""
    if mv.status != "READY":
        raise RegistryError("only READY versions can be served")
    current = db.scalar(select(ModelVersion).where(ModelVersion.model_id == mv.model_id, ModelVersion.status == "SERVING"))
    if current is not None:
        _transition(current, "READY")
        db.flush()
    _transition(mv, "SERVING")
    mv.reviewed_by, mv.reviewed_at, mv.review_note = (admin.id if admin else None), datetime.now(timezone.utc), note
    db.flush()
    return mv


def archive(db: Session, mv: ModelVersion, admin: User, note: str | None = None) -> ModelVersion:
    _transition(mv, "ARCHIVED")
    mv.reviewed_by, mv.reviewed_at, mv.review_note = admin.id, datetime.now(timezone.utc), note
    db.flush()
    return mv


# ---------------------------------------------------------------------------------------------------
# Admin package upload (zip). Paths inside the zip are whitelisted; storage keys are generated here.
# ---------------------------------------------------------------------------------------------------
_ALLOWED = re.compile(r"^(metadata\.json|descriptor_catalog\.json|split\.json|"
                      r"fold_[1-5]/(model\.joblib|selected_descriptors\.json|ga_flags\.json|preprocessing\.joblib))$")
_TYPES = {"metadata.json": "METADATA", "descriptor_catalog.json": "DESCRIPTOR_CATALOG", "split.json": "SPLIT",
          "model.joblib": "FOLD_MODEL", "selected_descriptors.json": "SELECTED_DESCRIPTORS",
          "ga_flags.json": "GA_FLAGS", "preprocessing.joblib": "PREPROCESSING"}


def upload_package(db: Session, *, admin: User, property_id: int, model_name: str, data: bytes) -> ModelVersion:
    prop = db.get(Property, property_id)
    if prop is None:
        raise RegistryError("unknown property")
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise RegistryError("not a zip file") from exc
    entries = [i for i in zf.infolist() if not i.is_dir()]
    tops = {i.filename.split("/", 1)[0] for i in entries}
    # allow a single wrapping directory (e.g. "my_model/metadata.json")
    strip = len(tops) == 1 and all("/" in i.filename for i in entries) and not next(iter(tops)).startswith("fold_")
    files: dict[str, bytes] = {}
    for info in entries:
        name = info.filename.split("/", 1)[1] if strip else info.filename
        if not _ALLOWED.match(name):
            raise RegistryError(f"unexpected file in package: {info.filename}")
        if info.file_size > 512 * 1024 * 1024:
            raise RegistryError(f"file too large: {info.filename}")
        files[name] = zf.read(info)
    if "metadata.json" not in files or "descriptor_catalog.json" not in files:
        raise RegistryError("package must contain metadata.json and descriptor_catalog.json")
    meta = json.loads(files["metadata.json"])
    missing = [k for k in REQUIRED_METADATA if k not in meta]
    if missing:
        raise RegistryError(f"metadata missing fields: {missing}")
    if meta["task_type"] != prop.task_type:
        raise RegistryError(f"task_type {meta['task_type']} != property task {prop.task_type}")

    model = get_or_create_model(db, owner_id=None, property_id=prop.id, scope="BUILTIN", name=model_name)
    version = _next_version(db, model.id)
    prefix = storage.model_prefix(scope="BUILTIN", model_id=model.id, version=version, property_code=prop.code)
    rep = meta["representation"]
    mv = ModelVersion(
        model_id=model.id, version=version, status="UPLOADED", task_type=meta["task_type"],
        target_columns=meta.get("target_columns") or prop.target_columns, descriptor_mode=meta["descriptor_method"],
        split_strategy=meta["split_strategy"], n_folds=meta["n_folds"], split_seeds=meta["split_seeds"],
        representation_model=RepresentationInfo(rep["backend"], rep["model"]).label, prompt_version=meta["prompt_version"],
        sklearn_model=meta["sklearn_model"], sklearn_version=meta.get("sklearn_version"), storage_prefix=prefix,
        metadata_=meta | {"uploaded_by": admin.username, "upload_token": secrets.token_hex(4)}, metrics=meta.get("metrics"),
    )
    db.add(mv)
    db.flush()
    for name, blob in sorted(files.items()):
        fold = int(name[5]) if name.startswith("fold_") else None
        _record(db, mv, _TYPES[name.rsplit("/", 1)[-1]], storage.put_bytes(f"{prefix}/{name}", blob), fold)
    db.flush()
    return mv


def version_to_dict(mv: ModelVersion, include_metadata: bool = False) -> dict[str, Any]:
    d = {
        "model_version_id": mv.id, "model_id": mv.model_id, "model_name": mv.model.name, "version": mv.version,
        "label": mv.label, "scope": mv.model.scope, "owner_id": mv.model.owner_id,
        "property": mv.model.prop.code if mv.model.prop else None, "status": mv.status, "task_type": mv.task_type,
        "target_columns": mv.target_columns, "descriptor_mode": mv.descriptor_mode, "split_strategy": mv.split_strategy,
        "n_folds": mv.n_folds, "split_seeds": mv.split_seeds, "representation_model": mv.representation_model,
        "prompt_version": mv.prompt_version, "sklearn_model": mv.sklearn_model, "sklearn_version": mv.sklearn_version,
        "metrics": mv.metrics, "validation_report": mv.validation_report, "training_job_id": mv.training_job_id,
        "review_note": mv.review_note, "reviewed_at": mv.reviewed_at, "created_at": mv.created_at,
    }
    if include_metadata:
        d["metadata"] = mv.metadata_
        d["artifacts"] = [{"type": a.artifact_type, "fold": a.fold_no, "key": a.storage_key, "sha256": a.checksum,
                           "size": a.size_bytes} for a in mv.artifacts]
    return d


__all__ = ["ModelVersion", "np"]
