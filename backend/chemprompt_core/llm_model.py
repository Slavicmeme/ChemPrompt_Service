from __future__ import annotations
import os
import inspect
import ast
from collections import OrderedDict
from typing import List, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError


import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from rdkit import Chem
from rdkit.Chem import (
    Descriptors, Crippen, Lipinski, rdMolDescriptors,
    Fragments, QED,
)

from chemprompt_core.custom_model import get_model_repo

class LLMModel:
    EXAONE_COMPAT_REVISION = "17b70148e344c28f54a542a030a805b8c96be8c3"

    def __init__(self, model_repo: str, dtype: str = "half", device: str = "cuda", model_family: str = "auto"):
        model_info = get_model_repo(model_repo)
        repo, name = model_info["repo"], model_info["name"]
        ckpt = f"{repo}/{name}"

        self.device = torch.device(device)
        self.is_exaone = "exaone" in ckpt.lower()
        self.is_gemma = "gemma" in ckpt.lower()
        self.model_family = self._resolve_model_family(model_family, ckpt)
        self.dtype = self._resolve_dtype(dtype)
        load_path = self._resolve_model_path(ckpt)

        self.tokenizer = AutoTokenizer.from_pretrained(
            load_path,
            trust_remote_code=True,
            legacy=False,
        )
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model_cls = AutoModelForCausalLM if self.model_family == "causal" else AutoModel
        self.model = model_cls.from_pretrained(
            load_path,
            trust_remote_code=True,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.resize_token_embeddings(len(self.tokenizer))

        
        self._descriptor_funcs: OrderedDict[str, callable] = OrderedDict()
        self.property_names: list[str] = []
        self.property_funcs: list[callable] = []

    def _resolve_model_path(self, model_path: str) -> str:
        if not self.is_exaone:
            return model_path
        try:
            return snapshot_download(
                repo_id=model_path,
                revision=self.EXAONE_COMPAT_REVISION,
                local_files_only=True,
            )
        except LocalEntryNotFoundError:
            return snapshot_download(
                repo_id=model_path,
                revision=self.EXAONE_COMPAT_REVISION,
            )

    @staticmethod
    def _resolve_model_family(model_family: str, model_path: str) -> str:
        normalized = (model_family or "auto").lower().replace("-", "_")
        if normalized == "auto":
            model_name = model_path.lower().replace("-", "_")
            return "gpt_oss" if "gpt_oss" in model_name else "default"
        aliases = {
            "default": "default",
            "base": "default",
            "auto_model": "default",
            "encoder": "default",
            "casual": "causal",
            "causal": "causal",
            "causal_lm": "causal",
            "gptoss": "gpt_oss",
            "gpt_oss": "gpt_oss",
        }
        if normalized not in aliases:
            raise ValueError("model_family must be one of: auto, default, causal, gpt_oss")
        return aliases[normalized]

    def _resolve_dtype(self, dtype: str):
        if dtype == "half":
            if self.is_exaone or self.is_gemma:
                return torch.bfloat16
            return torch.bfloat16 if self.model_family == "gpt_oss" else torch.float16
        return torch.float32

    def _embed_prompt(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            if self.model_family == "causal":
                outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
                vec = outputs.hidden_states[-1].mean(dim=1).squeeze().float().cpu().numpy()
            elif self.model_family == "gpt_oss":
                outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                masked_hidden = last_hidden * attention_mask
                sum_hidden = masked_hidden.sum(dim=1)
                lengths = attention_mask.sum(dim=1)
                vec = (sum_hidden / lengths).float().cpu().numpy().flatten()
            else:
                outputs = self.model(**inputs)
                vec = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
        return vec

    def _filter_descriptors(self, smiles_list: list[str]):
        def _safe_len(x):
            try:
                return len(x)
            except:
                return None

        def _count_if_iterable(val):
            try:
                if isinstance(val, (list, tuple)):
                    return len(val)
            except:
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
    
        # filtering
        names, funcs = list(d_all.keys()), list(d_all.values())
        rows = []
        for smi in tqdm(smiles_list, desc="Filtering descriptors"):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                rows.append([None] * len(funcs))
                continue
            row = []
            for fn in funcs:
                try:
                    row.append(fn(mol))
                except:
                    row.append(None)
            rows.append(row)
    
        df = pd.DataFrame(rows, columns=names)
        df_stats = pd.DataFrame({
            "name": df.columns,
            "std": df.std(),
            "unique": df.nunique()
        })
    
        selected_names = df_stats.query("std > 1 or unique > 10")["name"]
        self._descriptor_funcs = OrderedDict((n, d_all[n]) for n in selected_names if n in d_all)
        #self.property_names = list(self._descriptor_funcs.keys())
        self.property_names = [name.replace("_", " ") for name in self._descriptor_funcs.keys()]
    
        self.property_funcs = []
        for fn in self._descriptor_funcs.values():
            def wrapper(f):
                return lambda m: f(m)
            self.property_funcs.append(wrapper(fn))

    def _calculate_properties(self, smiles: str, flag: Sequence[bool]):
        if len(flag) != len(self.property_names):
            raise ValueError("flag length mismatch with descriptor count")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [None] * sum(flag)

        selected = [fn for keep, fn in zip(flag, self.property_funcs) if keep]
        out = []
        for fn in selected:
            try:
                out.append(fn(mol))
            except Exception:
                out.append(None)
        return out

    def _create_prompt(self, smiles: str, flag: Sequence[bool]):
        if not any(flag):
            return f"SMILES: {smiles}"

        props = self._calculate_properties(smiles, flag)
        names = [n for keep, n in zip(flag, self.property_names) if keep]
        parts = [f"{n}: {v:.5f}" if isinstance(v, (float, int)) else f"{n}: {v}"
                 for n, v in zip(names, props) if v is not None]

        prompt = f"SMILES: {smiles} | " + " | ".join(parts).replace('_', ' ') if parts else f"SMILES: {smiles}"
        # print(prompt)
        return prompt
        
    def get_descriptor_values(self, smiles_list: Sequence[str]) -> np.ndarray:
        """
        Return a numeric matrix of descriptor values for the given SMILES list.
        Uses only the filtered descriptor functions.
        """
        if not self.property_funcs:
            raise RuntimeError("Descriptors not filtered yet. Run _filter_descriptors() first.")
    
        records = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                records.append([None] * len(self.property_funcs))
                continue
    
            vals = []
            for fn in self.property_funcs:
                try:
                    vals.append(fn(mol))
                except Exception:
                    vals.append(None)
            records.append(vals)
    
        df = pd.DataFrame(records, columns=self.property_names)
        return df.fillna(df.mean()).to_numpy()

    def get_embeddings(self, smiles_list: Sequence[str], flag: Sequence[bool] = None):
        # Perform descriptor filtering on the first call
        if not self.property_names:
            self._filter_descriptors(smiles_list)

        if flag is None:
            flag = [True] * len(self.property_names)
        if len(flag) != len(self.property_names):
            print(len(self.property_names))
            raise ValueError("flag length mismatch with descriptor count")

        vecs: list[np.ndarray] = []
        for smi in tqdm(smiles_list, desc="Embedding SMILES"):
            prompt = self._create_prompt(smi, flag)
            vecs.append(self._embed_prompt(prompt))
        return np.asarray(vecs)

    def get_embeddings_for_ga(
        self,
        smiles_list: Sequence[str],
        flag: Sequence[bool],
        save_dir: str = "./embedding"
    ):
    
        if len(flag) != len(self.property_names):
            raise ValueError("flag length mismatch with descriptor count")
    
        flag_str = ''.join(['1' if b else '0' for b in flag])
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"flag_{flag_str}.csv")
    
        # cache load
        if os.path.exists(save_path):
            df = pd.read_csv(save_path)
            embeddings = np.array(df['embedding'].apply(ast.literal_eval).to_list())
            print(f"[Cached] Loaded from {save_path}")
            return embeddings
    
        vecs: list[np.ndarray] = []
        rows = []
    
        for smi in tqdm(smiles_list, desc=f"Embedding flag {flag_str}"):
            prompt = self._create_prompt(smi, flag)
            vec = self._embed_prompt(prompt)
            vecs.append(vec)
            rows.append({'smiles': smi, 'embedding': vec.tolist()})
    
        # save embeddings
        df_save = pd.DataFrame(rows)
        df_save.to_csv(save_path, index=False)
    
        # print(f"[Saved] {save_path}")
        return np.array(vecs)
