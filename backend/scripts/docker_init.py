"""One-shot initialisation for `docker compose up` (service `init`). Idempotent.

1. apply sql/*.sql (also picks up files added after the DB volume was first created)
2. wait for the Aya service when REPRESENTATION_MODE=http
3. create the web admin from ADMIN_USERNAME / ADMIN_PASSWORD if no admin exists (skipped when empty)
4. train + serve built-in models for properties without a SERVING model (BUILTIN_DESCRIPTOR_MODE / BUILTIN_SPLIT)
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import httpx
from sqlalchemy import func, select

from app.config import settings
from app.db import session_scope
from app.models import Model, ModelVersion, Property, User
from app.security import hash_password


def run(*args: str) -> None:
    print("$ python -m", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-m", *args], check=True)


def wait_for_representation(timeout_s: int = 7200) -> None:  # first start may download the LLM
    if settings.representation_mode != "http":
        return
    url = settings.representation_service_url.rstrip("/") + "/health"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            if httpx.get(url, timeout=5).json().get("loaded"):
                print("Aya representation service ready", flush=True)
                return
        except Exception:  # noqa: BLE001 — not up yet
            pass
        time.sleep(10)
    raise SystemExit(f"Aya service not ready after {timeout_s}s: {url}")


def ensure_admin() -> None:
    with session_scope() as db:
        if db.scalar(select(func.count()).select_from(User).where(User.role == "ADMIN")):
            print("admin: exists", flush=True)
            return
        name = os.environ.get("ADMIN_USERNAME", "admin").strip() or "admin"
        pw = os.environ.get("ADMIN_PASSWORD", "")
        if len(pw) < 8:
            print("admin: not created (ADMIN_PASSWORD empty or < 8 chars) -> "
                  f"docker compose exec api python -m scripts.create_admin {name}", flush=True)
            return
        db.add(User(username=name, password_hash=hash_password(pw), role="ADMIN"))
        print(f"admin: created '{name}'", flush=True)


def missing_builtin() -> list[str]:
    """Properties without a SERVING built-in model for the current representation (mock vs Aya)."""
    from app.services.representation import current_info

    label = current_info().label
    with session_scope() as db:
        return [p.code for p in db.scalars(select(Property).order_by(Property.id))
                if db.scalar(select(ModelVersion.id).join(Model).where(
                    Model.property_id == p.id, Model.scope == "BUILTIN", ModelVersion.status == "SERVING",
                    ModelVersion.representation_model == label)) is None]


def main() -> None:
    run("scripts.apply_sql")
    wait_for_representation()
    ensure_admin()
    missing = missing_builtin()
    if not missing:
        print("built-in models: all serving", flush=True)
    else:
        run("scripts.bootstrap_builtin", "--properties", *missing,
            "--mode", os.environ.get("BUILTIN_DESCRIPTOR_MODE", "FULL"),
            "--split", os.environ.get("BUILTIN_SPLIT", "RANDOM"), "--serve", "--missing-only")
    print("init: done", flush=True)


if __name__ == "__main__":
    main()
