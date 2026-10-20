"""Descriptor calculation & ChemPrompt prompt construction.

All of this is the original ChemPrompt ``LLMModel`` code (``chemprompt_core.llm_model``)
(the module used by Optimization_of_Chemprompt.ipynb). We instantiate ``LLMModel`` *without*
``__init__`` so that no LLM weights are loaded on the CPU/web side, and call its descriptor methods:

* ``_filter_descriptors``   -> dataset-level descriptor filtering (std > 1 or unique > 10)
* ``get_descriptor_values`` -> descriptor matrix (used for Top-K feature selection)
* ``_create_prompt``        -> "SMILES: ... | name: value | ..." prompt

The filter is data dependent, so the resulting descriptor list is persisted with every model
(``descriptor_catalog.json``). To rebuild the same functions at inference time without the training
data, ``_candidate_descriptor_funcs`` mirrors the candidate dictionary that ``_filter_descriptors``
builds before filtering; ``tests/test_descriptor_service.py`` asserts that a restored engine produces
exactly the same names and prompts as the original method.
"""
from __future__ import annotations

import inspect
from collections import OrderedDict
from typing import Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Crippen, Descriptors, Fragments, Lipinski, rdMolDescriptors

from chemprompt_core.llm_model import LLMModel
from app.services.cancellation import check_cancelled
from app.config import settings

RDLogger.DisableLog("rdApp.*")


def _candidate_descriptor_funcs() -> "OrderedDict[str, callable]":
    """Mirror of the candidate dict ``d_all`` inside ``LLMModel._filter_descriptors``."""

    def _count_if_iterable(val):
        try:
            if isinstance(val, (list, tuple)):
                return len(val)
        except Exception:
            pass
        return val

    d_all: OrderedDict[str, callable] = OrderedDict()
    for name, fn in Descriptors.descList:
        d_all[name] = fn
    for name in ("MolLogP", "MolMR"):
        d_all[name] = getattr(Crippen, name)
    for name, func in inspect.getmembers(Lipinski, inspect.isfunction):
        def make_safe_func(f):
            return lambda m: _count_if_iterable(f(m))
        d_all[f"Lipinski_{name}"] = make_safe_func(func)
    for name, fn in inspect.getmembers(rdMolDescriptors, inspect.isfunction):
        if "fingerprint" in name.lower():
            continue
        if len(inspect.signature(fn).parameters) == 1:
            d_all[f"rdMol_{name}"] = fn
    for name, fn in inspect.getmembers(Fragments, inspect.isfunction):
        d_all[name] = fn
    d_all["QED"] = QED.qed
    return d_all


def _bare_llm() -> LLMModel:
    llm = LLMModel.__new__(LLMModel)  # no tokenizer / weights
    llm._descriptor_funcs = OrderedDict()
    llm.property_names = []
    llm.property_funcs = []
    return llm


class DescriptorEngine:
    """Holds a filtered descriptor list and exposes the original ChemPrompt descriptor/prompt methods."""

    def __init__(self, llm: LLMModel):
        self._llm = llm

    # ---- construction -------------------------------------------------------
    @classmethod
    def fit(cls, smiles: Sequence[str]) -> "DescriptorEngine":
        llm = _bare_llm()
        llm._filter_descriptors(list(smiles))  # original implementation
        return cls(llm)

    @classmethod
    def from_names(cls, descriptor_names: Sequence[str]) -> "DescriptorEngine":
        cand = _candidate_descriptor_funcs()
        missing = [n for n in descriptor_names if n not in cand]
        if missing:
            raise ValueError(f"descriptor(s) not available in this RDKit build: {missing[:5]}")
        llm = _bare_llm()
        llm._descriptor_funcs = OrderedDict((n, cand[n]) for n in descriptor_names)
        # identical to the tail of LLMModel._filter_descriptors
        llm.property_names = [name.replace("_", " ") for name in llm._descriptor_funcs.keys()]
        llm.property_funcs = []
        for fn in llm._descriptor_funcs.values():
            def wrapper(f):
                return lambda m: f(m)
            llm.property_funcs.append(wrapper(fn))
        return cls(llm)

    @classmethod
    def from_catalog(cls, catalog: dict) -> "DescriptorEngine":
        return cls.from_names(catalog["descriptor_names"])

    # ---- accessors ------------------------------------------------------------
    @property
    def descriptor_names(self) -> list[str]:
        """Internal function names (persisted)."""
        return list(self._llm._descriptor_funcs.keys())

    @property
    def property_names(self) -> list[str]:
        """Display names used inside prompts (underscores replaced by spaces)."""
        return list(self._llm.property_names)

    def catalog(self) -> dict:
        import rdkit

        return {
            "descriptor_names": self.descriptor_names,
            "property_names": self.property_names,
            "n_descriptors": len(self.descriptor_names),
            "filter_rule": "std > 1 or unique > 10 (LLMModel._filter_descriptors)",
            "preprocessing_version": settings.preprocessing_version,
            "rdkit_version": rdkit.__version__,
        }

    def full_flag(self) -> list[bool]:
        return [True] * len(self.descriptor_names)

    def flag_from_indices(self, indices: Sequence[int]) -> list[bool]:
        flag = [False] * len(self.descriptor_names)
        for i in indices:
            flag[int(i)] = True
        return flag

    def selected_names(self, flag: Sequence[bool]) -> list[str]:
        return [n for keep, n in zip(flag, self.property_names) if keep]

    # ---- original ChemPrompt computations ----------------------------------
    def descriptor_frame(self, smiles: Sequence[str]) -> pd.DataFrame:
        values = self._llm.get_descriptor_values(list(smiles))
        return pd.DataFrame(values, columns=self.property_names)

    def prompt(self, smiles: str, flag: Sequence[bool]) -> str:
        return self._llm._create_prompt(smiles, flag)

    def prompts(self, smiles: Sequence[str], flag: Sequence[bool]) -> list[str]:
        if len(flag) != len(self.descriptor_names):
            raise ValueError("flag length mismatch with descriptor count")
        out = []
        for i, s in enumerate(smiles):
            if i % 256 == 0:
                check_cancelled()
            out.append(self.prompt(s, flag))
        return out


def canonicalize(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    return None if mol is None else Chem.MolToSmiles(mol)


def is_valid_smiles(smiles: str) -> bool:
    """Same validity criterion as ``DataLoader._filter_valid_smiles`` (RDKit MolFromSmiles is not None)."""
    return Chem.MolFromSmiles(str(smiles)) is not None


def flag_to_str(flag: Sequence[bool]) -> str:
    return "".join("1" if b else "0" for b in flag)


def str_to_flag(s: str) -> list[bool]:
    return [c == "1" for c in s]


__all__ = ["DescriptorEngine", "canonicalize", "is_valid_smiles", "flag_to_str", "str_to_flag", "np"]
