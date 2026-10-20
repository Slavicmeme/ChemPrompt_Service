"""Artifact storage (docs/09_STORAGE_POLICY.md).

Users/Admins never pass filesystem paths. This service builds canonical object keys and is the only
component that maps a key to a real path under ``MODEL_STORAGE_ROOT``.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from app.config import settings

_KEY_PART = re.compile(r"^[A-Za-z0-9_.\-]+$")


class StorageError(Exception):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode())


class StorageService:
    def __init__(self, root: Path | None = None):
        self.root = (root or settings.storage_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- canonical keys ---------------------------------------------------
    @staticmethod
    def _join(*parts: Any) -> str:
        out = []
        for p in parts:
            s = str(p)
            if not _KEY_PART.match(s) or s in {".", ".."}:
                raise StorageError(f"invalid key component: {s!r}")
            out.append(s)
        return "/".join(out)

    def dataset_prefix(self, user_id: int | None, dataset_id: int, version: int) -> str:
        if user_id is None:
            return self._join("builtin", "datasets", dataset_id, version)
        return self._join("users", user_id, "datasets", dataset_id, version)

    def representation_prefix(self, user_id: int | None, cache_key: str) -> str:
        owner = ("builtin",) if user_id is None else ("users", user_id)
        return self._join(*owner, "representations", cache_key)

    def model_prefix(self, *, scope: str, model_id: int, version: int, user_id: int | None = None,
                     property_code: str | None = None) -> str:
        if scope == "BUILTIN":
            return self._join("builtin", property_code or "unassigned", "models", model_id, version)
        return self._join("users", user_id, "models", model_id, version)

    def staging_prefix(self, token: str) -> str:
        return self._join("staging", token)

    # ---- io ----------------------------------------------------------------
    def path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root not in p.parents and p != self.root:
            raise StorageError("key escapes storage root")
        return p

    def put_bytes(self, key: str, data: bytes) -> dict[str, Any]:
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)
        return {"storage_key": key, "checksum": sha256_bytes(data), "size_bytes": len(data)}

    def put_json(self, key: str, obj: Any) -> dict[str, Any]:
        return self.put_bytes(key, json.dumps(obj, indent=2, sort_keys=True, default=str).encode())

    def put_npy(self, key: str, arr: np.ndarray) -> dict[str, Any]:
        buf = io.BytesIO()
        np.save(buf, np.asarray(arr), allow_pickle=False)
        return self.put_bytes(key, buf.getvalue())

    def get_bytes(self, key: str) -> bytes:
        p = self.path(key)
        if not p.is_file():
            raise StorageError(f"object not found: {key}")
        return p.read_bytes()

    def get_json(self, key: str) -> Any:
        return json.loads(self.get_bytes(key))

    def get_npy(self, key: str) -> np.ndarray:
        return np.load(io.BytesIO(self.get_bytes(key)), allow_pickle=False)

    def exists(self, key: str) -> bool:
        return self.path(key).is_file()

    def checksum(self, key: str) -> str:
        return sha256_bytes(self.get_bytes(key))

    def list_keys(self, prefix: str) -> list[str]:
        base = self.path(prefix)
        if not base.is_dir():
            return []
        return sorted(str(p.relative_to(self.root)) for p in base.rglob("*") if p.is_file())

    def move_prefix(self, src: str, dst: str) -> None:
        s, d = self.path(src), self.path(dst)
        if d.exists():
            raise StorageError(f"destination exists: {dst}")
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(s), str(d))

    def delete_prefix(self, prefix: str) -> None:
        p = self.path(prefix)
        if p.is_dir():
            shutil.rmtree(p)


storage = StorageService()
