import numpy as np

from app.services.descriptor_service import DescriptorEngine
from app.services.ga_service import MockGAProvider, k_bits_to_full_flag
from app.services.split_service import compute_folds
from app.services.training_service import _fold_flag


def test_feature_selection_ignores_validation_rows(smiles, y_reg):
    """Changing the validation targets must not change the fold's selected descriptors."""
    engine = DescriptorEngine.fit(smiles)
    f = compute_folds(smiles, y_reg, "RANDOM", "regression")[0]
    tr = f.train_idx
    flag_a, _ = _fold_flag("FEATURE_SELECTION", engine, [smiles[i] for i in tr], y_reg[tr])
    y_perturbed = y_reg.copy()
    y_perturbed[f.valid_idx] = 1e6
    flag_b, _ = _fold_flag("FEATURE_SELECTION", engine, [smiles[i] for i in tr], y_perturbed[tr])
    assert flag_a == flag_b
    assert sum(flag_a) == min(50, len(engine.descriptor_names))


def test_mock_ga_is_deterministic_and_maps_to_full_flag(smiles, y_reg):
    engine = DescriptorEngine.fit(smiles)
    cands = list(range(0, min(50, len(engine.descriptor_names))))
    kw = dict(fold_no=1, split_strategy="RANDOM", split_seed=42, train_smiles=smiles, y_train=y_reg,
              task_type="regression", engine=engine, candidate_indices=cands)
    a, b = MockGAProvider().select(**kw), MockGAProvider().select(**kw)
    assert a.flags == b.flags and any(a.flags) and len(a.flags) == len(cands)
    c = MockGAProvider().select(**(kw | {"fold_no": 2}))
    assert c.flags != a.flags                                   # fold-specific flags
    full = k_bits_to_full_flag(engine, cands, a.flags)
    assert len(full) == len(engine.descriptor_names)
    assert [full[i] for i in cands] == a.flags and sum(full) == sum(a.flags)
