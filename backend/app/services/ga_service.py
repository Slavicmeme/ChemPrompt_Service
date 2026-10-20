"""GENETIC descriptor mode (docs/04_GA_AND_DESCRIPTOR_MODES.md).

Per outer fold, using ONLY that fold's training rows:
  1. Top-K (K=50) candidates via ``descriptor_selection.fit_feature_selector(method="f_regression")``
  2. a GA provider chooses a K-bit flag over the candidates
  3. the K-bit flag is mapped back to a full flag over all filtered descriptors (as ``genetic.run``)

Providers share one interface so the V1 mock can be replaced by the real GA without any change to
storage/API/schema:
  * ``MockGAProvider``     — deterministic pseudo-random flags (GA_MODE=mock, V1 default)
  * ``GeneticGAProvider``  — the original ChemPrompt ``genetic`` class (``chemprompt_core.genetic``), with inner
                             CV restricted to the outer fold's train rows (GA_MODE=genetic)
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np

from app.config import settings
from app.services.descriptor_service import DescriptorEngine
from app.services.representation import embed_prompts


@dataclass
class GAResult:
    flags: list[bool]                    # K-bit, aligned with candidate_indices
    fitness: dict[str, Any] | None = None
    config: dict[str, Any] = field(default_factory=dict)


class GAProvider(Protocol):
    name: str

    def select(self, *, fold_no: int, split_strategy: str, split_seed: int, train_smiles: Sequence[str],
               y_train: np.ndarray, task_type: str, engine: DescriptorEngine,
               candidate_indices: Sequence[int]) -> GAResult: ...


class MockGAProvider:
    name = "MOCK"

    def select(self, *, fold_no, split_strategy, split_seed, train_smiles, y_train, task_type, engine, candidate_indices):
        material = f"{split_strategy}|{split_seed}|{fold_no}|{','.join(map(str, candidate_indices))}"
        seed = int.from_bytes(hashlib.sha256(material.encode()).digest()[:4], "little")
        rng = np.random.RandomState(seed)
        flags = (rng.rand(len(candidate_indices)) < 0.5).tolist()
        if not any(flags):
            flags[int(rng.randint(len(flags)))] = True
        return GAResult(flags=[bool(f) for f in flags], fitness=None,
                        config={"provider": "MOCK", "mock_seed": seed, "note": "V1 mock flags; not GA-optimized"})


class _RepresentationLLMAdapter:
    """Duck-types the parts of ``LLMModel`` that ``genetic`` uses, backed by the Representation
    Service (prompts built on CPU, embeddings from Aya)."""

    device = "representation-service"

    def __init__(self, engine: DescriptorEngine):
        self._engine = engine
        self.property_names = engine.property_names

    def get_embeddings_for_ga(self, smiles_list, flag, save_dir="./embedding"):
        return embed_prompts(self._engine.prompts(smiles_list, flag))


class GeneticGAProvider:
    name = "GENETIC"

    def __init__(self, params: dict | None = None):
        self.params = params or dict(settings.ga_params)

    def select(self, *, fold_no, split_strategy, split_seed, train_smiles, y_train, task_type, engine, candidate_indices):
        from chemprompt_core.genetic import genetic  # imported lazily: only needed for GA_MODE=genetic

        method = "scaffold" if split_strategy == "SCAFFOLD" else "kfold"
        ga = genetic(
            smiles_list=list(train_smiles), y=y_train, llm_model=_RepresentationLLMAdapter(engine),
            dataset_name=f"service_{uuid.uuid4().hex[:8]}_fold_{fold_no}",  # genetic writes logs to ./evolutionary_process/<name>
            selected_indices=list(candidate_indices),
            num_generations=self.params["num_generations"], pop_size=self.params["pop_size"],
            top_k=self.params["top_k"], mutation_rate=self.params["mutation_rate"],
            split_method=method, num_folds=self.params["inner_folds"], random_seed=settings.split.random_seed,
            model_type=task_type,
        )
        best_individual, _full_flag = ga.run()
        fitness = ga.individual_metrics.get(tuple(best_individual))
        return GAResult(flags=[bool(b) for b in best_individual], fitness=fitness,
                        config={"provider": "GENETIC", **self.params, "inner_split": method})


def get_provider() -> GAProvider:
    if settings.ga_mode == "genetic":
        return GeneticGAProvider()
    return MockGAProvider()


def k_bits_to_full_flag(engine: DescriptorEngine, candidate_indices: Sequence[int], flags: Sequence[bool]) -> list[bool]:
    """Same mapping as the end of ``genetic.run``."""
    full_flag = [False] * len(engine.descriptor_names)
    for j, desc_idx in enumerate(candidate_indices):
        if flags[j]:
            full_flag[desc_idx] = True
    return full_flag
