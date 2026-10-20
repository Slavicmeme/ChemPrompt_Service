import numpy as np

from chemprompt_core.data_splitter import DataSplitter
from app.config import settings
from app.services.split_service import compute_folds, split_spec


def test_scaffold_seed_policy_extends_existing_rule():
    # original rule: base 42 + i; original scaffold run used 3 folds -> (42, 43, 44)
    original = DataSplitter(["C"], [0], num_folds=3, random_seed=42)._scaffold_seeds()
    assert original == [42, 43, 44]
    assert list(settings.split.scaffold_seeds[:3]) == original           # existing behaviour preserved
    assert list(settings.split.scaffold_seeds) == [42, 43, 44, 45, 46]   # extended to 5
    assert DataSplitter(["C"], [0], num_folds=5, random_seed=42)._scaffold_seeds() == list(settings.split.scaffold_seeds)


def test_random_5fold_is_a_partition(smiles, y_reg):
    folds = compute_folds(smiles, y_reg, "RANDOM", "regression")
    assert len(folds) == 5
    valid = np.concatenate([f.valid_idx for f in folds])
    assert sorted(valid.tolist()) == list(range(len(smiles)))     # every row validated exactly once
    for f in folds:
        assert len(np.intersect1d(f.train_idx, f.valid_idx)) == 0
        assert len(f.train_idx) + len(f.valid_idx) == len(smiles)
        assert abs(len(f.valid_idx) / len(smiles) - 0.2) < 0.05


def test_random_split_is_reproducible(smiles, y_reg):
    a = compute_folds(smiles, y_reg, "RANDOM", "regression")
    b = compute_folds(smiles, y_reg, "RANDOM", "regression")
    assert all((x.valid_idx == y.valid_idx).all() for x, y in zip(a, b))


def test_classification_random_uses_stratified(smiles):
    y = np.array([[i % 2] for i in range(len(smiles))])
    spec = split_spec("RANDOM", "classification")
    assert spec["config"]["effective_method"] == "stratified_kfold"
    for f in compute_folds(smiles, y, "RANDOM", "classification"):
        assert 0 < y[f.valid_idx].mean() < 1


def test_scaffold_folds_do_not_share_scaffolds(smiles, y_reg):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    def scaf(s):
        return MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s), includeChirality=False)

    folds = compute_folds(smiles, y_reg, "SCAFFOLD", "regression")
    assert [f.seed for f in folds] == [42, 43, 44, 45, 46]
    for f in folds:
        tr = {scaf(smiles[i]) for i in f.train_idx}
        va = {scaf(smiles[i]) for i in f.valid_idx}
        assert not (tr & va), f"fold {f.fold_no} leaks scaffolds"
        assert len(f.train_idx) + len(f.valid_idx) == len(smiles)
