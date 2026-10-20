"""Aya service multi-GPU logic with fake replicas (no GPU / model needed)."""
import importlib.util
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("aya_app", Path(__file__).resolve().parents[2] / "aya_service" / "app.py")
aya = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(aya)


class FakeModel:
    def __init__(self, tag):
        self.tag, self.threads = tag, set()

    def _embed_prompt(self, prompt):
        self.threads.add(threading.get_ident())
        return np.array([float(prompt.split("-")[1]), self.tag], dtype=np.float32)


def test_split_chunks_cover_all_in_order():
    assert aya.split_chunks(5, 2) == [(0, 3), (3, 5)]
    assert aya.split_chunks(2, 4) == [(0, 1), (1, 2)]       # never more parts than items
    assert aya.split_chunks(7, 1) == [(0, 7)]


def test_embed_parallel_uses_every_replica_and_keeps_order():
    models = [FakeModel(0), FakeModel(1), FakeModel(2)]
    replicas = [aya.Replica(m, f"cuda:{i}") for i, m in enumerate(models)]
    prompts = [f"p-{i}" for i in range(10)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        out = aya.embed_parallel(replicas, prompts, pool)
    assert out.shape == (10, 2)
    assert out[:, 0].tolist() == list(range(10))            # order preserved
    assert sorted(set(out[:, 1].tolist())) == [0.0, 1.0, 2.0]  # all GPUs used
