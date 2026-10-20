"""Representation client (docs/07_REPRESENTATION_SERVICE.md).

The web/sklearn side never loads Aya. It sends ChemPrompt prompts to the Aya Representation Service
(GPU host) and receives embeddings. ``REPRESENTATION_MODE=mock`` swaps in a deterministic, CPU-only
hashing embedding so the whole pipeline can be developed/tested without a GPU. Mock embeddings are
recorded as backend ``mock`` everywhere (cache identity, model metadata, predictions) and must never
be confused with Aya representations.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Sequence

import httpx
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import RepresentationArtifact
from app.services.cancellation import check_cancelled
from app.services.storage import sha256_json, storage


class RepresentationError(RuntimeError):
    pass


@dataclass(frozen=True)
class RepresentationInfo:
    backend: str  # aya | mock
    model: str

    @property
    def label(self) -> str:
        return self.model if self.backend == "aya" else f"mock:{self.model}"


class MockBackend:
    """Deterministic signed feature hashing of prompt parts (descriptor name -> scaled value) and
    SMILES character 3-grams. Only for development; carries real descriptor signal so that the
    sklearn pipeline yields meaningful (non-random) metrics."""

    _num = re.compile(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$")

    def __init__(self, dim: int):
        self.dim = dim
        self.info = RepresentationInfo("mock", f"hashing-{dim}")

    def _h(self, token: str) -> tuple[int, float]:
        d = hashlib.blake2b(token.encode(), digest_size=8).digest()
        idx = int.from_bytes(d[:4], "little") % self.dim
        return idx, (1.0 if d[4] & 1 else -1.0)

    def embed_one(self, prompt: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        parts = prompt.split(" | ")
        smiles = parts[0].removeprefix("SMILES: ")
        padded = f"^{smiles}$"
        for i in range(len(padded) - 2):
            idx, sign = self._h("ng:" + padded[i:i + 3])
            v[idx] += 0.25 * sign
        for part in parts[1:]:
            name, _, raw = part.partition(": ")
            if not self._num.match(raw.strip()):
                continue
            x = float(raw)
            scaled = math.copysign(math.log1p(abs(x)), x)
            idx, sign = self._h("d:" + name)
            v[idx] += sign * scaled
        return v

    def embed(self, prompts: Sequence[str]) -> np.ndarray:
        out = []
        for i, p in enumerate(prompts):
            if i % 256 == 0:
                check_cancelled()
            out.append(self.embed_one(p))
        return np.stack(out) if out else np.zeros((0, self.dim), np.float32)


class HttpBackend:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.info = RepresentationInfo("aya", model)

    def embed(self, prompts: Sequence[str]) -> np.ndarray:
        out = []
        bs = settings.representation_batch_size
        with httpx.Client(timeout=settings.representation_timeout) as client:
            for i in range(0, len(prompts), bs):
                check_cancelled()  # between Aya batches
                batch = list(prompts[i:i + bs])
                try:
                    r = client.post(f"{self.base_url}/v1/embeddings", json={"prompts": batch, "model": self.info.model})
                    r.raise_for_status()
                except httpx.HTTPError as exc:
                    raise RepresentationError(f"Representation service error: {exc}") from exc
                body = r.json()
                if body.get("model") != self.info.model:
                    raise RepresentationError(f"representation model mismatch: {body.get('model')} != {self.info.model}")
                out.append(np.asarray(body["embeddings"], dtype=np.float32))
        return np.concatenate(out) if out else np.zeros((0, 0), np.float32)

    def health(self) -> dict:
        r = httpx.get(f"{self.base_url}/health", timeout=10)
        r.raise_for_status()
        return r.json()


def get_backend():
    if settings.representation_mode == "http":
        return HttpBackend(settings.representation_service_url, settings.aya_model_id)
    if settings.representation_mode == "mock":
        return MockBackend(settings.mock_embedding_dim)
    raise RepresentationError(f"unknown REPRESENTATION_MODE {settings.representation_mode}")


def current_info() -> RepresentationInfo:
    return get_backend().info


def embed_prompts(prompts: Sequence[str], expected: RepresentationInfo | None = None) -> np.ndarray:
    backend = get_backend()
    if expected is not None and (backend.info.backend, backend.info.model) != (expected.backend, expected.model):
        raise RepresentationError(
            f"model was trained with representation {expected.label} but the service provides {backend.info.label}"
        )
    return backend.embed(prompts)


def get_or_create_dataset_representation(
    db: Session, *, owner_id: int | None, dataset_version_id: int, dataset_checksum: str,
    descriptor_mode: str, descriptor_config: dict, prompts_fn,
) -> tuple[RepresentationArtifact, np.ndarray]:
    """Cached embeddings for one (dataset version, descriptor config) pair. ``prompts_fn`` is only
    called on a cache miss."""
    info = current_info()
    config_hash = sha256_json(descriptor_config)
    cache_key = sha256_json({
        "dataset_checksum": dataset_checksum, "backend": info.backend, "model": info.model,
        "prompt_version": settings.prompt_version, "descriptor_mode": descriptor_mode,
        "descriptor_config_hash": config_hash, "preprocessing_version": settings.preprocessing_version,
    })
    art = db.scalar(select(RepresentationArtifact).where(RepresentationArtifact.cache_key == cache_key))
    if art is not None and storage.exists(art.storage_key) and storage.checksum(art.storage_key) == art.checksum:
        return art, storage.get_npy(art.storage_key)

    emb = embed_prompts(prompts_fn())
    key = storage.representation_prefix(owner_id, cache_key) + "/embeddings.npy"
    meta = storage.put_npy(key, emb)
    storage.put_json(storage.representation_prefix(owner_id, cache_key) + "/meta.json", {
        "cache_key": cache_key, "representation": info.label, "descriptor_mode": descriptor_mode,
        "descriptor_config": descriptor_config, "prompt_version": settings.prompt_version,
    })
    if art is None:
        art = RepresentationArtifact(
            owner_id=owner_id, dataset_version_id=dataset_version_id, cache_key=cache_key,
            representation_model=info.model, representation_backend=info.backend,
            prompt_version=settings.prompt_version, descriptor_mode=descriptor_mode,
            descriptor_config_hash=config_hash, preprocessing_version=settings.preprocessing_version,
            storage_key=key, checksum=meta["checksum"], n_rows=int(emb.shape[0]), dim=int(emb.shape[1]),
        )
        db.add(art)
    else:
        art.storage_key, art.checksum = key, meta["checksum"]
    db.flush()
    return art, emb
