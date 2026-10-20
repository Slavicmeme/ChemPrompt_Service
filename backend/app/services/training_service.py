"""5-fold training (docs/03, docs/04, docs/06).

For each fold of the common Split Service:
    fold train (~80%) -> [Top-K feature selection] -> [GA] -> descriptor flag
                      -> ChemPrompt prompts -> representation -> sklearn fit
    fold valid (~20%) -> evaluation only (never used for selection/GA/fit)

sklearn estimators and metrics come from the original ``chemprompt_core.sklearn_model.ScikitLearnModel``
(Ridge / LogisticRegression / MultiOutputClassifier(LogisticRegression)) and
``chemprompt_core.metrics`` exactly as ``ScikitLearnModel.fit_and_evaluate`` uses them.
"""
from __future__ import annotations

import io
import logging
import traceback
from datetime import datetime, timezone
from typing import Any, Callable

import joblib
import numpy as np
import sklearn
from sqlalchemy.orm import Session

from chemprompt_core.descriptor_selection import descriptor_selection
from chemprompt_core.sklearn_model import ScikitLearnModel
from app.config import settings
from app.db import session_scope
from app.models import (DatasetVersion, GAExperiment, GAFoldResult, SplitConfig, TrainingFoldResult, TrainingJob)
from app.services import registry_service
from app.services.dataset_service import load_validated
from app.services.descriptor_service import DescriptorEngine, flag_to_str
from app.services.cancellation import JobCancelled, cancel_scope, check_cancelled
from app.services.ga_service import get_provider, k_bits_to_full_flag
from app.services.representation import current_info, get_or_create_dataset_representation
from app.services.split_service import get_or_create_split, load_folds
from app.services.storage import sha256_bytes, storage

log = logging.getLogger(__name__)

DESCRIPTOR_MODES = ("FULL", "FEATURE_SELECTION", "GENETIC")
SKLEARN_MODEL_NAMES = {"regression": "Ridge", "classification": "LogisticRegression",
                       "multilabel": "MultiOutputClassifier(LogisticRegression)"}


class TrainingError(RuntimeError):
    pass


def new_sklearn_wrapper(task_type: str) -> ScikitLearnModel:
    workdir = storage.path("tmp/sklearn")
    workdir.mkdir(parents=True, exist_ok=True)
    return ScikitLearnModel(model_type=task_type, save_dir=str(workdir))


def create_job(db: Session, *, owner_id: int | None, dv: DatasetVersion, descriptor_mode: str, split_strategy: str,
               model_name: str | None = None, property_id: int | None = None, workflow_id: int | None = None) -> TrainingJob:
    descriptor_mode = descriptor_mode.upper()
    if descriptor_mode not in DESCRIPTOR_MODES:
        raise TrainingError(f"descriptor_mode must be one of {DESCRIPTOR_MODES}")
    if dv.status != "VALIDATED" or not dv.task_type:
        raise TrainingError("dataset version must be VALIDATED with a task type")
    smiles, y = load_validated(dv)
    sc, _ = get_or_create_split(db, dv, split_strategy, smiles, y)
    info = current_info()
    job = TrainingJob(
        owner_id=owner_id, dataset_version_id=dv.id, split_config_id=sc.id, property_id=property_id, workflow_id=workflow_id,
        task_type=dv.task_type, descriptor_mode=descriptor_mode, sklearn_model=SKLEARN_MODEL_NAMES[dv.task_type],
        model_name=model_name or f"{dv.dataset.name} {descriptor_mode} {sc.strategy}",
        config={
            "split_strategy": sc.strategy, "n_folds": sc.n_folds, "split_seeds": sc.seeds,
            "feature_selection": {"k": settings.feature_selection_k, "method": settings.feature_selection_method}
            if descriptor_mode != "FULL" else None,
            "ga_provider": get_provider().name if descriptor_mode == "GENETIC" else None,
            "representation": {"backend": info.backend, "model": info.model},
            "prompt_version": settings.prompt_version,
        },
        status="QUEUED", stage="queued", progress={"folds_done": 0, "n_folds": sc.n_folds},
    )
    db.add(job)
    db.flush()
    return job


def _set_stage(job_id: int, stage: str, **progress: Any) -> None:
    check_cancelled()  # stage boundary = cancellation point
    with session_scope() as db:
        job = db.get(TrainingJob, job_id)
        job.stage = stage
        if progress:
            job.progress = {**(job.progress or {}), **progress}


def _fold_flag(mode: str, engine: DescriptorEngine, train_smiles: list[str], y_train: np.ndarray) -> tuple[list[bool], list[int] | None]:
    if mode == "FULL":
        return engine.full_flag(), None
    X_train_df = engine.descriptor_frame(train_smiles)           # fold train rows only
    selector = descriptor_selection()
    selector.fit_feature_selector(X_train_df, y_train, k=settings.feature_selection_k, method=settings.feature_selection_method)
    idx = [int(i) for i in selector.selected_indices]
    return engine.flag_from_indices(idx), idx


def run_job(job_id: int, on_step: Callable[[str, dict], None] | None = None,
            should_cancel: Callable[[], bool] | None = None) -> None:
    """Executes a training job. Every stage is committed so status always reflects real backend work.

    ``on_step(step_code, result)`` is called after the phases that map onto the custom-training
    workflow steps: REPRESENTATION, TRAINING, EVALUATION, MODEL_REGISTRATION.
    """
    step = on_step or (lambda *_: None)
    try:
        with session_scope() as db:
            job = db.get(TrainingJob, job_id)
            if job.status != "QUEUED":
                raise TrainingError(f"job {job_id} is {job.status}")
            job.status, job.started_at, job.stage = "RUNNING", datetime.now(timezone.utc), "loading_dataset"
        with cancel_scope(should_cancel):
            check_cancelled()
            _run(job_id, step)
    except JobCancelled as exc:
        log.info("training job %s cancelled", job_id)
        with session_scope() as db:
            job = db.get(TrainingJob, job_id)
            job.status, job.stage, job.error = "CANCELLED", "cancelled", str(exc)
            job.finished_at = datetime.now(timezone.utc)
            ga = db.query(GAExperiment).filter_by(training_job_id=job_id).one_or_none()
            if ga is not None:
                ga.status = "CANCELLED"
        raise
    except Exception as exc:  # noqa: BLE001 — persist failure
        log.exception("training job %s failed", job_id)
        with session_scope() as db:
            job = db.get(TrainingJob, job_id)
            job.status, job.error, job.finished_at = "FAILED", str(exc)[:4000], datetime.now(timezone.utc)
            job.stage = "failed"
            ga = db.query(GAExperiment).filter_by(training_job_id=job_id).one_or_none()
            if ga is not None:
                ga.status = "FAILED"
        raise


def _run(job_id: int, step: Callable[[str, dict], None]) -> None:
    with session_scope() as db:
        job = db.get(TrainingJob, job_id)
        dv = db.get(DatasetVersion, job.dataset_version_id)
        sc = db.get(SplitConfig, job.split_config_id)
        smiles, y = load_validated(dv)
        folds = load_folds(db, sc)
        mode, task_type, owner_id = job.descriptor_mode, job.task_type, job.owner_id
        dv_id, dv_checksum, strategy = dv.id, dv.checksum, sc.strategy
        ga_exp_id = None
        if mode == "GENETIC":
            provider = get_provider()
            ga = GAExperiment(training_job_id=job.id, provider=provider.name, candidate_k=settings.feature_selection_k,
                              candidate_method=settings.feature_selection_method,
                              config={"params": settings.ga_params if provider.name == "GENETIC" else {}})
            db.add(ga)
            db.flush()
            ga_exp_id = ga.id

    # Global, label-free descriptor filtering on the dataset SMILES — identical to
    # "Step 3. Global Descriptor Filtering (One-time)" in Optimization_of_Chemprompt.ipynb.
    _set_stage(job_id, "descriptor_filtering")
    engine = DescriptorEngine.fit(smiles)
    catalog = engine.catalog()

    # ---- Phase A: per-fold descriptor configuration + representation (workflow step 7) ----------
    prepared: list[dict[str, Any]] = []
    for f in folds:
        tr = f.train_idx
        train_smiles = [smiles[i] for i in tr]
        y_tr = y[tr]
        _set_stage(job_id, f"fold_{f.fold_no}_descriptor_selection")
        flag, candidates = _fold_flag(mode, engine, train_smiles, y_tr)
        ga_payload = None
        if mode == "GENETIC":
            _set_stage(job_id, f"fold_{f.fold_no}_genetic")
            res = get_provider().select(fold_no=f.fold_no, split_strategy=strategy, split_seed=f.seed,
                                        train_smiles=train_smiles, y_train=y_tr, task_type=task_type,
                                        engine=engine, candidate_indices=candidates)
            flag = k_bits_to_full_flag(engine, candidates, res.flags)
            ga_payload = {"candidate_descriptors": [engine.property_names[i] for i in candidates],
                          "candidate_indices": candidates, "flags": flag_to_str(res.flags),
                          "full_flag": flag_to_str(flag), "fitness": res.fitness, "config": res.config}
            with session_scope() as db:
                db.add(GAFoldResult(ga_experiment_id=ga_exp_id, fold_no=f.fold_no,
                                    candidate_descriptors=ga_payload["candidate_descriptors"], flags=ga_payload["flags"],
                                    full_flag=ga_payload["full_flag"], selected_descriptors=engine.selected_names(flag),
                                    fitness=res.fitness))
        if not any(flag):
            raise TrainingError(f"fold {f.fold_no}: empty descriptor selection")

        _set_stage(job_id, f"fold_{f.fold_no}_representation")
        with session_scope() as db:
            rep, X = get_or_create_dataset_representation(
                db, owner_id=owner_id, dataset_version_id=dv_id, dataset_checksum=dv_checksum, descriptor_mode=mode,
                descriptor_config={"descriptor_names": engine.descriptor_names, "flag": flag_to_str(flag)},
                prompts_fn=lambda fl=flag: engine.prompts(smiles, fl),
            )
            rep_id, rep_dim = rep.id, rep.dim
        prepared.append({"fold": f, "flag": flag, "X": X, "rep_id": rep_id, "ga": ga_payload})
    step("REPRESENTATION", {"representation_ids": sorted({p["rep_id"] for p in prepared}), "dim": rep_dim,
                            "n_descriptors_filtered": len(engine.descriptor_names),
                            "selected_per_fold": [int(sum(p["flag"])) for p in prepared]})

    # ---- Phase B: fit one sklearn model per fold on fold-train rows only (workflow step 8) -------
    fold_payloads: list[dict[str, Any]] = []
    for p in prepared:
        f, X, flag = p["fold"], p["X"], p["flag"]
        tr, va = f.train_idx, f.valid_idx
        _set_stage(job_id, f"fold_{f.fold_no}_training")
        wrapper = new_sklearn_wrapper(task_type)
        wrapper._maybe_set_multilabel(y[tr])
        wrapper.model = wrapper._default_model(wrapper.model_type)
        wrapper.model.fit(X[tr], wrapper._prepare_y(y[tr]))
        buf = io.BytesIO()
        joblib.dump(wrapper.model, buf)
        fold_payloads.append({"fold_no": f.fold_no, "seed": f.seed, "n_train": int(len(tr)), "n_valid": int(len(va)),
                              "flag": flag_to_str(flag), "selected": engine.selected_names(flag), "wrapper": wrapper,
                              "model_bytes": buf.getvalue(), "ga": p["ga"], "representation_id": p["rep_id"]})
        with session_scope() as db:
            job = db.get(TrainingJob, job_id)
            job.progress = {**job.progress, "folds_trained": f.fold_no}
    step("TRAINING", {"folds_trained": len(fold_payloads), "sklearn_model": SKLEARN_MODEL_NAMES[task_type]})

    # ---- Phase C: evaluate each fold model on its held-out validation rows (workflow step 9) -----
    _set_stage(job_id, "evaluation")
    for p, fp in zip(prepared, fold_payloads):
        f, X, wrapper = p["fold"], p["X"], fp.pop("wrapper")
        va = f.valid_idx
        y_pred = wrapper.model.predict(X[va])
        y_score = None if wrapper.model_type == "regression" else wrapper._prediction_scores(X[va])
        # metrics in the target's original units (no z-score standardisation via y_reference)
        fp["metrics"] = {k: _f(v) for k, v in wrapper._score_metrics(y[va], y_pred, y_score).items()}
        with session_scope() as db:
            db.add(TrainingFoldResult(training_job_id=job_id, fold_no=f.fold_no, split_seed=f.seed, n_train=fp["n_train"],
                                      n_valid=fp["n_valid"], metrics=fp["metrics"], selected_descriptors=fp["selected"],
                                      representation_id=fp["representation_id"]))
    summary = summarize([fp["metrics"] for fp in fold_payloads])
    with session_scope() as db:
        db.get(TrainingJob, job_id).metrics = summary
    step("EVALUATION", {"metrics": {k: {"mean": v["mean"], "std": v["std"]} for k, v in summary.items()}})

    # ---- Phase D: register + validate the 5-fold model version (workflow step 10) --------------
    _set_stage(job_id, "registration")
    with session_scope() as db:
        job = db.get(TrainingJob, job_id)
        if ga_exp_id:
            db.get(GAExperiment, ga_exp_id).status = "SUCCEEDED"
        mv = registry_service.register_from_training(db, job, catalog=catalog, fold_payloads=fold_payloads, summary=summary,
                                                     sklearn_version=sklearn.__version__)
        mv_id = mv.id
    with session_scope() as db:
        mv = db.get(registry_service.ModelVersion, mv_id)
        report = registry_service.validate_version(db, mv)
        status = mv.status
    with session_scope() as db:
        job = db.get(TrainingJob, job_id)
        job.status, job.stage, job.finished_at = "SUCCEEDED", "done", datetime.now(timezone.utc)
        job.progress = {**job.progress, "model_version_id": mv_id}
    step("MODEL_REGISTRATION", {"model_version_id": mv_id, "status": status, "validation_errors": report["errors"]})


def _f(v: Any) -> float | None:
    v = float(v)
    return None if not np.isfinite(v) else v


def summarize(fold_metrics: list[dict[str, float | None]]) -> dict[str, dict[str, float | None]]:
    out = {}
    for name in fold_metrics[0]:
        vals = np.array([m[name] if m[name] is not None else np.nan for m in fold_metrics], dtype=float)
        out[name] = {"mean": _f(np.nanmean(vals)) if np.isfinite(vals).any() else None,
                     "std": _f(np.nanstd(vals)) if np.isfinite(vals).any() else None,
                     "folds": [None if not np.isfinite(v) else float(v) for v in vals]}
    return out


__all__ = ["create_job", "run_job", "summarize", "sha256_bytes", "TrainingError"]
