"""Delete ALL built-in models and built-in datasets (DB rows + storage files) so they can be retrained.

    bin/compose.sh exec api chemprompt scripts.reset_builtin          # dry run: only shows what would be deleted
    bin/compose.sh exec api chemprompt scripts.reset_builtin --yes    # actually delete

Users, users' own datasets/models and agent conversations are NOT touched. Predictions made with
built-in models are deleted (they reference the deleted model versions).
"""
from __future__ import annotations

import argparse

from sqlalchemy import delete, func, select

from app.db import session_scope
from app.models import Dataset, DatasetVersion, Model, ModelVersion, Prediction, TrainingJob
from app.services.storage import storage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="actually delete (default: dry run)")
    a = ap.parse_args()

    with session_scope() as db:
        mv_ids = db.scalars(select(ModelVersion.id).join(Model).where(Model.scope == "BUILTIN")).all()
        model_ids = db.scalars(select(Model.id).where(Model.scope == "BUILTIN")).all()
        n_pred = db.scalar(select(func.count()).select_from(Prediction).where(Prediction.model_version_id.in_(mv_ids))) if mv_ids else 0
        job_ids = db.scalars(select(TrainingJob.id).where(TrainingJob.owner_id.is_(None))).all()
        ds_ids = db.scalars(select(Dataset.id).where(Dataset.is_builtin)).all()
        n_dv = db.scalar(select(func.count()).select_from(DatasetVersion).where(DatasetVersion.dataset_id.in_(ds_ids))) if ds_ids else 0

        print("built-in models          :", len(model_ids), f"(versions {len(mv_ids)})")
        print("predictions using them   :", n_pred)
        print("built-in training jobs   :", len(job_ids))
        print("built-in datasets        :", len(ds_ids), f"(versions {n_dv}; splits/representations cascade)")
        print("storage prefix           :", storage.path("builtin"))
        if not a.yes:
            print("\ndry run — nothing deleted. Re-run with --yes to delete.")
            return

        # order matters: predictions -> models(versions, artifacts cascade) -> jobs -> datasets(cascade)
        if mv_ids:
            db.execute(delete(Prediction).where(Prediction.model_version_id.in_(mv_ids)))
        if model_ids:
            db.execute(delete(Model).where(Model.id.in_(model_ids)))
        if job_ids:
            db.execute(delete(TrainingJob).where(TrainingJob.id.in_(job_ids)))
        if ds_ids:
            db.execute(delete(Dataset).where(Dataset.id.in_(ds_ids)))  # properties.dataset_id -> NULL
    storage.delete_prefix("builtin")
    print("\ndeleted. Retrain with: bin/compose.sh exec api chemprompt scripts.bootstrap_builtin --properties ... --serve")


if __name__ == "__main__":
    main()
