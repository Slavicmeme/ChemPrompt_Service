"""Common 5-fold Split Service (docs/03_SPLIT_AND_EVALUATION.md).

Every training path (FULL / FEATURE_SELECTION / GENETIC, built-in and custom) obtains its folds
from here. The split itself is delegated to the original ``chemprompt_core.data_splitter.DataSplitter``:

* RANDOM   -> ``split_method="kfold"`` with ``RANDOM_SEED=42`` (StratifiedKFold for single-label
             classification, KFold otherwise — existing DataSplitter behaviour).
* SCAFFOLD -> ``split_method="scaffold"`` (chemprop ``scaffold_balanced``, 80/20 per fold) with an
             explicit seed list. The original code used 3 scaffold folds with seeds ``42 + i``
             (42, 43, 44); the same rule extended to 5 gives (42, 43, 44, 45, 46).
             Each scaffold fold is an independent scaffold-balanced 80/20 split, so the same
             scaffold never appears in both train and validation of one fold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from chemprompt_core.data_splitter import DataSplitter
from app.config import settings
from app.models import DatasetVersion, SplitConfig, SplitFoldAssignment
from app.services.storage import sha256_json

SPLITTER_VERSION = "chemprompt-datasplitter-v1"
STRATEGIES = ("RANDOM", "SCAFFOLD")


@dataclass
class Fold:
    fold_no: int
    seed: int
    train_idx: np.ndarray
    valid_idx: np.ndarray


def split_spec(strategy: str, task_type: str) -> dict:
    strategy = strategy.upper()
    if strategy not in STRATEGIES:
        raise ValueError(f"unsupported split strategy {strategy}")
    pol = settings.split
    if strategy == "RANDOM":
        method = "stratified_kfold" if task_type == "classification" else "kfold"
        seeds = [pol.random_seed] * pol.n_folds
        seed = pol.random_seed
    else:
        method = "scaffold_balanced"
        seeds = list(pol.scaffold_seeds)
        seed = seeds[0]
    if len(seeds) != pol.n_folds:
        raise ValueError("seed list length must equal n_folds")
    return {
        "strategy": strategy,
        "n_folds": pol.n_folds,
        "seed": seed,
        "seeds": seeds,
        "config": {
            "splitter": SPLITTER_VERSION,
            "datasplitter_method": "scaffold" if strategy == "SCAFFOLD" else "kfold",
            "effective_method": method,
            "task_type": task_type,
            "train_fraction": 0.8,
            "valid_fraction": 0.2,
        },
    }


def compute_folds(smiles: list[str], y: np.ndarray, strategy: str, task_type: str) -> list[Fold]:
    spec = split_spec(strategy, task_type)
    n_folds = spec["n_folds"]
    if spec["strategy"] == "SCAFFOLD":
        splitter = DataSplitter(list(smiles), y, num_folds=n_folds, random_seed=spec["seeds"])
        it = splitter.get_split_indices(split_method="scaffold", task_type=task_type)
    else:
        # notebooks pass np.arange(len(x)) as splitter input for kfold
        splitter = DataSplitter(np.arange(len(smiles)), y, num_folds=n_folds, random_seed=spec["seed"])
        it = splitter.get_split_indices(split_method="kfold", task_type=task_type)
    folds = []
    for i, (tr, va) in enumerate(it, start=1):
        folds.append(Fold(i, int(spec["seeds"][i - 1]), np.sort(np.asarray(tr, dtype=int)), np.sort(np.asarray(va, dtype=int))))
    if len(folds) != n_folds:
        raise RuntimeError(f"splitter returned {len(folds)} folds, expected {n_folds}")
    for f in folds:
        if np.intersect1d(f.train_idx, f.valid_idx).size:
            raise RuntimeError(f"fold {f.fold_no}: train/valid overlap")
    return folds


def assignment_checksum(folds: list[Fold]) -> str:
    return sha256_json([[f.fold_no, f.seed, sorted(f.train_idx.tolist()), sorted(f.valid_idx.tolist())] for f in folds])


def get_or_create_split(db: Session, dv: DatasetVersion, strategy: str, smiles: list[str], y: np.ndarray) -> tuple[SplitConfig, list[Fold]]:
    """Persist (or reuse) the split config + fold assignment for a dataset version."""
    spec = split_spec(strategy, dv.task_type)
    checksum = sha256_json({k: spec[k] for k in ("strategy", "n_folds", "seed", "seeds", "config")} | {"dataset_checksum": dv.checksum})
    existing = db.scalar(select(SplitConfig).where(SplitConfig.dataset_version_id == dv.id, SplitConfig.checksum == checksum))
    if existing is not None:
        return existing, load_folds(db, existing)

    folds = compute_folds(smiles, y, spec["strategy"], dv.task_type)
    sc = SplitConfig(
        dataset_version_id=dv.id, strategy=spec["strategy"], n_folds=spec["n_folds"], seed=spec["seed"],
        seeds=spec["seeds"], config=spec["config"] | {"dataset_checksum": dv.checksum,
                                                      "fold_sizes": [[len(f.train_idx), len(f.valid_idx)] for f in folds]},
        checksum=checksum, assignment_checksum=assignment_checksum(folds),
    )
    db.add(sc)
    db.flush()
    rows = []
    for f in folds:
        rows += [{"split_config_id": sc.id, "fold_no": f.fold_no, "row_index": int(i), "partition": "TRAIN"} for i in f.train_idx]
        rows += [{"split_config_id": sc.id, "fold_no": f.fold_no, "row_index": int(i), "partition": "VALID"} for i in f.valid_idx]
    db.execute(SplitFoldAssignment.__table__.insert(), rows)
    db.flush()
    return sc, folds


def load_folds(db: Session, sc: SplitConfig) -> list[Fold]:
    rows = db.execute(
        select(SplitFoldAssignment.fold_no, SplitFoldAssignment.row_index, SplitFoldAssignment.partition)
        .where(SplitFoldAssignment.split_config_id == sc.id)
        .order_by(SplitFoldAssignment.fold_no, SplitFoldAssignment.row_index)
    ).all()
    folds = []
    for k in range(1, sc.n_folds + 1):
        tr = np.array([r.row_index for r in rows if r.fold_no == k and r.partition == "TRAIN"], dtype=int)
        va = np.array([r.row_index for r in rows if r.fold_no == k and r.partition == "VALID"], dtype=int)
        folds.append(Fold(k, int(sc.seeds[k - 1]), tr, va))
    if sc.assignment_checksum and assignment_checksum(folds) != sc.assignment_checksum:
        raise RuntimeError(f"split_config {sc.id}: stored fold assignment checksum mismatch")
    return folds
