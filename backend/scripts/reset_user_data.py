"""Delete all users' custom data (workflows, uploaded datasets, custom models) — e.g. before publishing.

    bin/compose.sh exec api chemprompt scripts.reset_user_data          # dry run: only shows what would be deleted
    bin/compose.sh exec api chemprompt scripts.reset_user_data --yes    # actually delete

Deleted: every workflow (+ training jobs), every user-uploaded dataset (+ splits, embeddings), every
USER-scope model (+ predictions made with it), and the storage files under users/ and tmp/.
Kept: user accounts, built-in datasets/models, predictions made with built-in models, agent conversations.
"""
from __future__ import annotations

import argparse

from sqlalchemy import delete, func, select

from app.db import session_scope
from app.models import Dataset, DatasetVersion, Model, ModelVersion, Prediction, TrainingJob, Workflow
from app.services import deletion_service
from app.services.storage import storage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="actually delete (default: dry run)")
    a = ap.parse_args()

    with session_scope() as db:
        wfs = db.scalars(select(Workflow)).all()
        running = [w.id for w in wfs if w.status == "RUNNING"]
        mvs = db.scalars(select(ModelVersion).join(Model).where(Model.scope == "USER")).all()
        n_pred = db.scalar(select(func.count()).select_from(Prediction)
                           .where(Prediction.model_version_id.in_([m.id for m in mvs]))) if mvs else 0
        ds_ids = db.scalars(select(Dataset.id).where(Dataset.is_builtin.is_(False))).all()
        print("workflows            :", len(wfs), f"(running: {running})" if running else "")
        print("user models          :", len(mvs), f"(predictions made with them: {n_pred})")
        print("user datasets        :", len(ds_ids))
        print("storage prefixes     :", storage.path("users"), "and", storage.path("tmp"))
        if running:
            raise SystemExit("cancel running trainings first (Custom 학습 화면 또는 Admin 화면에서 중단)")
        if not a.yes:
            print("\ndry run — nothing deleted. Re-run with --yes to delete.")
            return

        for wf in wfs:
            deletion_service.delete_workflow(db, wf, delete_model=True)
        db.flush()
        for mv in db.scalars(select(ModelVersion).join(Model).where(Model.scope == "USER")).all():
            deletion_service.delete_model_version(db, mv, by_admin=True)
        db.flush()
        # datasets uploaded outside a workflow (or kept after the workflow was deleted)
        user_dv_ids = select(DatasetVersion.id).join(Dataset).where(Dataset.is_builtin.is_(False))
        db.execute(delete(TrainingJob).where(TrainingJob.dataset_version_id.in_(user_dv_ids)))
        db.execute(delete(Dataset).where(Dataset.is_builtin.is_(False)))
    storage.delete_prefix("users")
    storage.delete_prefix("tmp")
    print("\ndeleted.")


if __name__ == "__main__":
    main()
