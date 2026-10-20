"""Create built-in datasets from the ORIGINAL ChemPrompt DataLoader (DeepChem MolNet) and train/serve
built-in 5-fold models. Runs on the admin workstation (CPU is fine; Aya via REPRESENTATION_MODE=http).

    python -m scripts.bootstrap_builtin --properties FreeSolv ESOL Lipo ClinTox \
        --mode FULL --split RANDOM [--serve] [--admin admin]
"""
from __future__ import annotations

import argparse
import io

import numpy as np
import pandas as pd
from sqlalchemy import select

from app.db import session_scope
from app.models import Model, ModelVersion, Property, TrainingJob, User
from app.routers.admin import create_builtin_version
from app.services import registry_service, training_service


def _stub_pytdc_if_missing() -> None:
    """DataLoader imports ``tdc.single_pred.ADME`` at module level but only uses it for Caco2_Wang,
    which is not a service property. PyTDC 1.1.15 conflicts with transformers 4.56.2, so it is not
    installed; register a stub that fails loudly if Caco2_Wang is ever requested."""
    try:
        import tdc.single_pred  # noqa: F401
    except ImportError:
        import sys
        import types

        class ADME:  # noqa: D401
            def __init__(self, *_, **__):
                raise ImportError("PyTDC is not installed; Caco2_Wang (PyTDC ADME) is unavailable in this environment")

        single_pred = types.ModuleType("tdc.single_pred")
        single_pred.ADME = ADME
        tdc = types.ModuleType("tdc")
        tdc.single_pred = single_pred
        sys.modules.setdefault("tdc", tdc)
        sys.modules.setdefault("tdc.single_pred", single_pred)


FREESOLV_RAW_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv"


def freesolv_to_kcal(df: pd.DataFrame) -> pd.DataFrame:
    """Replace FreeSolv labels with experimental hydration free energies in kcal/mol.

    DeepChem's ``freesolv.csv.gz`` (used by the original DataLoader) stores ``y`` already z-scored:
    ``y == (expt - mean(expt)) / std(expt, ddof=0)`` over MoleculeNet's ``SAMPL.csv`` (verified
    for all 642 molecules, max abs diff ~1e-15). Rows/SMILES still come from the original DataLoader;
    only the label is swapped, and the alignment is re-checked before returning.
    """
    import os

    import deepchem as dc

    data_dir = dc.utils.data_utils.get_data_dir()
    path = os.path.join(data_dir, "SAMPL.csv")
    if not os.path.exists(path):
        dc.utils.data_utils.download_url(url=FREESOLV_RAW_URL, dest_dir=data_dir)
    raw = pd.read_csv(path)[["smiles", "expt"]]
    out = df.merge(raw, on="smiles", how="left", validate="many_to_one")
    if out["expt"].isna().any():
        raise SystemExit(f"FreeSolv: {int(out['expt'].isna().sum())} molecules missing in SAMPL.csv")
    z = (out["expt"] - raw["expt"].mean()) / raw["expt"].std(ddof=0)
    if not np.allclose(z.to_numpy(), out["label"].to_numpy(), atol=1e-6):
        raise SystemExit("FreeSolv: SAMPL.csv expt does not match DeepChem z-scored labels — refusing to convert")
    out["label"] = out["expt"]
    print(f"[FreeSolv] labels converted to kcal/mol (range {out['label'].min():.2f} .. {out['label'].max():.2f})")
    return out.drop(columns=["expt"])


def load_with_chemprompt(code: str) -> pd.DataFrame:
    _stub_pytdc_if_missing()
    from chemprompt_core.data_loader import DataLoader

    loader = DataLoader()
    df = loader.supported_datasets[code]()
    df = loader._filter_valid_smiles(df, code)
    if code == "FreeSolv":
        df = freesolv_to_kcal(df)
    return df


def serving_version(db, property_id: int) -> ModelVersion | None:
    """SERVING built-in version usable with the current representation (a mock-trained model does not count
    once REPRESENTATION_MODE=http, so it gets retrained)."""
    from app.services.representation import current_info

    return db.scalar(select(ModelVersion).join(Model).where(
        Model.property_id == property_id, Model.scope == "BUILTIN", ModelVersion.status == "SERVING",
        ModelVersion.representation_model == current_info().label))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--properties", nargs="+", default=["FreeSolv", "ESOL", "Lipo", "ClinTox"])
    ap.add_argument("--mode", default="FULL", choices=training_service.DESCRIPTOR_MODES)
    ap.add_argument("--split", default="RANDOM", choices=["RANDOM", "SCAFFOLD"])
    ap.add_argument("--csv", help="use this CSV instead of the DeepChem download (single property only)")
    ap.add_argument("--serve", action="store_true", help="switch the new READY version to SERVING")
    ap.add_argument("--admin", default=None, help="admin username recorded as reviewer when --serve")
    ap.add_argument("--missing-only", action="store_true", help="skip properties that already have a SERVING built-in model")
    a = ap.parse_args()

    for code in a.properties:
        with session_scope() as db:
            prop = db.scalar(select(Property).where(Property.code == code))
            if prop is None:
                raise SystemExit(f"unknown property {code}")
            if a.missing_only and serving_version(db, prop.id) is not None:
                print(f"[{code}] already has a SERVING built-in model — skipped")
                continue
        if a.csv:
            raw = open(a.csv, "rb").read()
        else:
            df = load_with_chemprompt(code)
            raw = df.to_csv(index=False).encode()
        with session_scope() as db:
            prop = db.scalar(select(Property).where(Property.code == code))
            dv = create_builtin_version(db, prop, f"{code}.csv", raw, "smiles")
            print(f"[{code}] dataset_version={dv.id} status={dv.status} rows={dv.valid_row_count} task={dv.task_type}")
            if dv.status != "VALIDATED":
                print(dv.validation_report)
                continue
            job = training_service.create_job(db, owner_id=None, dv=dv, descriptor_mode=a.mode, split_strategy=a.split,
                                              property_id=prop.id, model_name=f"{code} ChemPrompt {a.mode}")
            job_id = job.id
        training_service.run_job(job_id, on_step=lambda code_, r: print(f"   step {code_}: {str(r)[:160]}"))
        with session_scope() as db:
            job = db.get(TrainingJob, job_id)
            mv = db.get(ModelVersion, job.progress["model_version_id"])
            print(f"[{code}] model_version={mv.id} status={mv.status} metrics=" +
                  ", ".join(f"{k}={v['mean']:.3f}" for k, v in job.metrics.items() if v["mean"] is not None))
            if a.serve and mv.status == "READY":
                admin = db.scalar(select(User).where(User.username == a.admin)) if a.admin else \
                    db.scalar(select(User).where(User.role == "ADMIN"))
                # no admin yet (e.g. first Docker start): recorded as a system action
                registry_service.set_serving(db, mv, admin, "bootstrap" if admin else "bootstrap (system, no admin yet)")
                print(f"[{code}] now SERVING")


if __name__ == "__main__":
    main()
