"""Aya Representation Service (docs/07_REPRESENTATION_SERVICE.md).

Internal Docker service (``http://aya:8100``, not published). Aya (CohereLabs/aya-expanse-8b) is used
ONLY to turn ChemPrompt prompts into embeddings — it is not a chat agent. The embedding itself is the
original ``LLMModel._embed_prompt`` (``chemprompt_core.llm_model``, mean-pooled last hidden state), so
representations are identical to the research code.

Multi-GPU: the GPUs given to the container (``.env`` ``AYA_GPUS``, e.g. ``0,1``) are all used. One model
replica is loaded per visible GPU (an 8B model fits on one GPU) and the prompts of a request are split
across the replicas in contiguous chunks, so the output order is preserved.

Model files live in ``HF_HOME`` (``<DATA_DIR>/llm`` in Docker) and are downloaded on the first start
when missing (``HF_TOKEN`` is needed for the gated Aya repository).

Prompts are built on the web/CPU side; this service receives prompt strings only.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# chemprompt_core lives in ../backend (the Docker image also sets PYTHONPATH=/app/backend)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

MODEL_ID = os.environ.get("AYA_MODEL_ID", "CohereLabs/aya-expanse-8b")
# Weight dtype. Original LLMModel accepts "half" (-> torch.float16 for Aya) or anything else (-> float32).
AYA_DTYPE = os.environ.get("AYA_DTYPE", "float16").strip().lower()
_DTYPE_ARG = {"float16": "half", "fp16": "half", "half": "half", "float32": "float32", "fp32": "float32"}
if AYA_DTYPE not in _DTYPE_ARG:
    raise RuntimeError(f"AYA_DTYPE must be float16 or float32, got {AYA_DTYPE!r}")
EXPECTED_TORCH_DTYPE = "torch.float16" if _DTYPE_ARG[AYA_DTYPE] == "half" else "torch.float32"
MAX_BATCH = int(os.environ.get("AYA_MAX_BATCH", "256"))


class Replica:
    """One LLMModel on one device; forwards on the same GPU are serialised."""

    def __init__(self, model, device: str):
        self.model, self.device = model, device
        self.lock = threading.Lock()

    def embed(self, prompts: Sequence[str]) -> list[np.ndarray]:
        with self.lock:
            return [np.asarray(self.model._embed_prompt(p), dtype=np.float32).ravel() for p in prompts]


def split_chunks(n_items: int, n_parts: int) -> list[tuple[int, int]]:
    """Contiguous (start, end) ranges covering n_items over at most n_parts parts (order preserved)."""
    n_parts = max(1, min(n_parts, n_items))
    base, extra = divmod(n_items, n_parts)
    out, start = [], 0
    for i in range(n_parts):
        end = start + base + (1 if i < extra else 0)
        out.append((start, end))
        start = end
    return out


def embed_parallel(replicas: Sequence[Replica], prompts: Sequence[str], pool: ThreadPoolExecutor) -> np.ndarray:
    chunks = split_chunks(len(prompts), len(replicas))
    futures = [pool.submit(replicas[i].embed, prompts[s:e]) for i, (s, e) in enumerate(chunks)]
    vecs = [v for f in futures for v in f.result()]
    return np.stack(vecs)


def visible_devices() -> list[str]:
    import torch

    n = torch.cuda.device_count()
    if n == 0:
        raise RuntimeError("no GPU visible to the aya service (check AYA_GPUS and the NVIDIA Container Toolkit)")
    return [f"cuda:{i}" for i in range(n)]


app = FastAPI(title="ChemPrompt Aya Representation Service")
_replicas: list[Replica] = []
_load_lock = threading.Lock()
_pool: ThreadPoolExecutor | None = None


def get_replicas() -> list[Replica]:
    global _pool
    if not _replicas:
        with _load_lock:
            if not _replicas:
                from chemprompt_core.llm_model import LLMModel

                devices = visible_devices()
                for dev in devices:  # sequential: the first load also downloads the model into HF_HOME
                    m = LLMModel(model_repo=MODEL_ID, dtype=_DTYPE_ARG[AYA_DTYPE], device=dev)
                    actual = str(next(m.model.parameters()).dtype)
                    if actual != EXPECTED_TORCH_DTYPE:  # e.g. bf16 would change embeddings vs. the expected setting
                        raise RuntimeError(f"Aya weights loaded as {actual}, expected {EXPECTED_TORCH_DTYPE} (AYA_DTYPE={AYA_DTYPE})")
                    _replicas.append(Replica(m, dev))
                    print(f"Aya replica ready on {dev}", flush=True)
                _pool = ThreadPoolExecutor(max_workers=len(_replicas), thread_name_prefix="aya")
    return _replicas


class EmbedRequest(BaseModel):
    prompts: list[str] = Field(min_length=1)
    model: str


class EmbedResponse(BaseModel):
    model: str
    dim: int
    embeddings: list[list[float]]
    elapsed_ms: int


@app.on_event("startup")
def _warmup() -> None:
    if os.environ.get("AYA_PRELOAD", "true").lower() == "true":
        get_replicas()


@app.get("/health")
def health() -> dict:
    loaded = bool(_replicas)
    dtype = str(next(_replicas[0].model.model.parameters()).dtype) if loaded else None
    return {"status": "ok", "model": MODEL_ID, "devices": [r.device for r in _replicas], "dtype": dtype,
            "expected_dtype": EXPECTED_TORCH_DTYPE, "loaded": loaded}


@app.post("/v1/embeddings", response_model=EmbedResponse)
def embeddings(req: EmbedRequest) -> EmbedResponse:
    if req.model != MODEL_ID:
        raise HTTPException(400, f"this service serves {MODEL_ID}, requested {req.model}")
    if len(req.prompts) > MAX_BATCH:
        raise HTTPException(413, f"max {MAX_BATCH} prompts per request")
    replicas = get_replicas()
    t = time.time()
    arr = embed_parallel(replicas, req.prompts, _pool)
    return EmbedResponse(model=MODEL_ID, dim=int(arr.shape[1]), embeddings=arr.tolist(), elapsed_ms=int((time.time() - t) * 1000))
