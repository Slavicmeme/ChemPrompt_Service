"""Datasets & versions (docs/06_CUSTOM_TRAINING.md steps 1-4)."""
from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Dataset, DatasetVersion
from app.services.descriptor_service import is_valid_smiles
from app.services.storage import sha256_bytes, storage

TASK_TYPES = ("regression", "classification", "multilabel")
MIN_ROWS = 25  # 5 folds x >= 5 validation rows


class DatasetError(ValueError):
    pass


def infer_task_type(y: np.ndarray) -> str:
    """Same rule as ``infer_task_type`` in the ChemPrompt notebooks / ``genetic._infer_model_type``."""
    y_arr = np.asarray(y)
    if y_arr.ndim == 2 and y_arr.shape[1] > 1:
        return "multilabel"
    flat = y_arr.ravel()
    flat = flat[~pd.isna(flat)]
    unique = set(np.unique(flat).tolist())
    if unique and unique.issubset({0, 1, 0.0, 1.0}):
        return "classification"
    return "regression"


def _read_csv(data: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(data))
    except Exception as exc:  # pandas raises many types
        raise DatasetError(f"CSV를 읽을 수 없습니다: {exc}") from exc


def create_dataset_upload(db: Session, *, owner_id: int | None, name: str, description: str | None,
                          filename: str, data: bytes, is_builtin: bool = False) -> DatasetVersion:
    df = _read_csv(data)
    if df.shape[1] < 2:
        raise DatasetError("SMILES와 target을 포함해 최소 2개 column이 필요합니다.")
    ds = Dataset(owner_id=owner_id, name=name.strip()[:200] or "dataset", description=description, is_builtin=is_builtin)
    db.add(ds)
    db.flush()
    return add_version(db, ds, filename=filename, data=data, df=df)


def add_version(db: Session, ds: Dataset, *, filename: str, data: bytes, df: pd.DataFrame | None = None) -> DatasetVersion:
    df = _read_csv(data) if df is None else df
    version = (db.scalar(select(func.max(DatasetVersion.version)).where(DatasetVersion.dataset_id == ds.id)) or 0) + 1
    prefix = storage.dataset_prefix(None if ds.is_builtin else ds.owner_id, ds.id, version)
    meta = storage.put_bytes(f"{prefix}/raw.csv", data)
    dv = DatasetVersion(
        dataset_id=ds.id, version=version, storage_key=meta["storage_key"], raw_checksum=meta["checksum"],
        original_filename=filename, columns=[str(c) for c in df.columns], row_count=int(len(df)), status="UPLOADED",
    )
    db.add(dv)
    db.flush()
    return dv


PROFILE_ROWS = 200       # rows read to profile columns
SMILES_SAMPLE = 50       # values checked with RDKit per column
TARGET_NAME_HINTS = ("target", "label", "y", "value", "activity", "property")


def profile_columns(dv: DatasetVersion) -> list[dict[str, Any]]:
    """Per-column content profile (example values, numeric ratio, RDKit-parsable ratio) on the first rows."""
    df = _read_csv(storage.get_bytes(dv.storage_key)).head(PROFILE_ROWS)
    out = []
    for c in df.columns:
        col = df[c].dropna()
        numeric_ratio = float(pd.to_numeric(col, errors="coerce").notna().mean()) if len(col) else 0.0
        sample = col.astype(str).head(SMILES_SAMPLE)
        # purely numeric columns are never SMILES (e.g. "95.82")
        smiles_ratio = float(sample.map(is_valid_smiles).mean()) if len(sample) and numeric_ratio < 0.5 else 0.0
        out.append({"name": str(c), "examples": [str(v)[:60] for v in col.head(3)],
                    "numeric_ratio": round(numeric_ratio, 3), "smiles_ratio": round(smiles_ratio, 3)})
    return out


def suggest_mapping(dv: DatasetVersion) -> dict[str, Any]:
    """Suggest the SMILES column by content (RDKit-parsable values) and a numeric target column."""
    prof = profile_columns(dv)
    smiles_cands = [p for p in prof if p["smiles_ratio"] >= 0.8]
    smiles_cands.sort(key=lambda p: (p["smiles_ratio"], "smile" in p["name"].lower()), reverse=True)
    smiles_col = smiles_cands[0]["name"] if smiles_cands else None
    numeric = [p["name"] for p in prof if p["name"] != smiles_col and p["numeric_ratio"] >= 0.9
               and p["name"].lower() not in ("num", "name")]
    hinted = [c for c in numeric if c.lower() in TARGET_NAME_HINTS]
    target = (hinted or numeric or [None])[0]
    return {"smiles_column": smiles_col, "target_columns": [target] if target else [], "columns_profile": prof}


def set_mapping(db: Session, dv: DatasetVersion, smiles_column: str, target_columns: list[str]) -> DatasetVersion:
    if smiles_column not in dv.columns:
        raise DatasetError(f"SMILES column '{smiles_column}' 이(가) 없습니다.")
    if not target_columns:
        raise DatasetError("target column을 1개 이상 선택하세요.")
    missing = [c for c in target_columns if c not in dv.columns]
    if missing:
        raise DatasetError(f"target column {missing} 이(가) 없습니다.")
    if smiles_column in target_columns:
        raise DatasetError("SMILES column은 target이 될 수 없습니다.")
    prof = {p["name"]: p for p in profile_columns(dv)}
    sp = prof[smiles_column]
    if sp["smiles_ratio"] < 0.5:
        raise DatasetError(
            f"'{smiles_column}' 열은 SMILES가 아닌 것 같습니다 (앞쪽 값 중 RDKit이 해석한 비율 {sp['smiles_ratio']:.0%}, "
            f"예시: {', '.join(sp['examples'])}). SMILES가 들어 있는 열을 선택하세요.")
    for t in target_columns:
        if prof[t]["numeric_ratio"] < 0.5:
            raise DatasetError(
                f"'{t}' 열은 숫자가 아닌 값이 대부분입니다 (예시: {', '.join(prof[t]['examples'])}). 예측할 숫자 값이 든 열을 선택하세요.")
    dv.smiles_column, dv.target_columns = smiles_column, list(target_columns)
    dv.status, dv.checksum, dv.validation_report, dv.task_type = "MAPPED", None, None, None
    db.flush()
    return dv


def validate(db: Session, dv: DatasetVersion) -> dict[str, Any]:
    """Drop rows with missing values / invalid SMILES (RDKit), write the normalized ``data.csv``."""
    if dv.status not in ("MAPPED", "VALIDATED", "INVALID") or not dv.smiles_column:
        raise DatasetError("column mapping을 먼저 완료하세요.")
    df = _read_csv(storage.get_bytes(dv.storage_key))
    cols = [dv.smiles_column, *dv.target_columns]
    sub = df[cols].copy()
    sub.insert(0, "source_row", np.arange(len(sub)))
    issues: dict[str, Any] = {}

    for t in dv.target_columns:
        sub[t] = pd.to_numeric(sub[t], errors="coerce")
    missing_mask = sub[cols].isna().any(axis=1)
    issues["missing_or_non_numeric_rows"] = sub.loc[missing_mask, "source_row"].astype(int).tolist()[:200]
    sub = sub.loc[~missing_mask]

    valid_mask = sub[dv.smiles_column].astype(str).map(is_valid_smiles)
    issues["invalid_smiles_rows"] = sub.loc[~valid_mask, "source_row"].astype(int).tolist()[:200]
    sub = sub.loc[valid_mask].reset_index(drop=True)

    dup = int(sub[dv.smiles_column].duplicated().sum())
    y = sub[dv.target_columns].to_numpy()
    task = infer_task_type(y) if len(sub) else None
    errors = []
    if len(sub) == 0:
        errors.append(f"유효한 row가 없습니다. SMILES 열('{dv.smiles_column}')과 target 열 선택이 맞는지 확인하세요.")
    elif len(sub) < MIN_ROWS:
        errors.append(f"유효한 row가 {len(sub)}개입니다. 5-fold 학습에는 최소 {MIN_ROWS}개가 필요합니다.")
    if task in ("classification", "multilabel"):
        for t in dv.target_columns:
            counts = sub[t].value_counts().to_dict()
            if len(counts) < 2 or min(counts.values()) < 5:
                errors.append(f"'{t}' class 분포가 너무 치우쳐 있습니다: {counts}")

    report = {
        "rows_total": int(dv.row_count or len(df)),
        "rows_valid": int(len(sub)),
        "rows_removed_missing": int(missing_mask.sum()),
        "rows_removed_invalid_smiles": int((~valid_mask).sum()),
        "duplicate_smiles": dup,
        "inferred_task_type": task,
        "target_stats": {t: _target_stats(sub[t]) for t in dv.target_columns},
        "issues": issues,
        "errors": errors,
    }
    out = sub.rename(columns={dv.smiles_column: "smiles"})
    out["smiles"] = out["smiles"].astype(str)
    data = out.to_csv(index=False).encode()
    key = dv.storage_key.rsplit("/", 1)[0] + "/data.csv"
    storage.put_bytes(key, data)

    dv.validation_report = report
    dv.valid_row_count = int(len(sub))
    dv.checksum = sha256_bytes(data) if not errors else None
    dv.status = "VALIDATED" if not errors else "INVALID"
    dv.task_type = task if not errors else None
    db.flush()
    return report


def _target_stats(s: pd.Series) -> dict[str, Any]:
    if s.empty:  # no valid rows: avoid NaN (not valid JSON for PostgreSQL)
        return {"min": None, "max": None, "mean": None, "n_unique": 0}
    return {"min": float(s.min()), "max": float(s.max()), "mean": float(s.mean()), "n_unique": int(s.nunique())}


def set_task_type(db: Session, dv: DatasetVersion, task_type: str) -> DatasetVersion:
    if dv.status != "VALIDATED":
        raise DatasetError("validation을 통과한 dataset version만 설정할 수 있습니다.")
    inferred = dv.validation_report["inferred_task_type"]
    if task_type not in TASK_TYPES:
        raise DatasetError(f"task_type은 {TASK_TYPES} 중 하나여야 합니다.")
    if task_type != inferred and not (inferred == "classification" and task_type == "regression"):
        raise DatasetError(f"target 값으로 판단한 task는 '{inferred}' 입니다. '{task_type}'로 학습할 수 없습니다.")
    dv.task_type = task_type
    db.flush()
    return dv


def load_validated(dv: DatasetVersion) -> tuple[list[str], np.ndarray]:
    if dv.status != "VALIDATED" or not dv.checksum:
        raise DatasetError("dataset version이 VALIDATED 상태가 아닙니다.")
    key = dv.storage_key.rsplit("/", 1)[0] + "/data.csv"
    data = storage.get_bytes(key)
    if sha256_bytes(data) != dv.checksum:
        raise DatasetError("dataset checksum mismatch")
    df = pd.read_csv(io.BytesIO(data))
    y = df[dv.target_columns].to_numpy()
    y = y.astype(float) if dv.task_type == "regression" else y.astype(int)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    return df["smiles"].astype(str).tolist(), y


def dataset_to_dict(dv: DatasetVersion) -> dict[str, Any]:
    ds = dv.dataset
    return {
        "dataset_id": ds.id, "dataset_version_id": dv.id, "name": ds.name, "description": ds.description,
        "version": dv.version, "is_builtin": ds.is_builtin, "status": dv.status, "filename": dv.original_filename,
        "columns": dv.columns, "smiles_column": dv.smiles_column, "target_columns": dv.target_columns,
        "task_type": dv.task_type, "row_count": dv.row_count, "valid_row_count": dv.valid_row_count,
        "checksum": dv.checksum, "validation_report": dv.validation_report, "created_at": dv.created_at,
    }
