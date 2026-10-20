from collections import OrderedDict

import numpy as np
try:
    from chemprop import data as chemprop_data
except ImportError:
    chemprop_data = None
from sklearn.model_selection import KFold, StratifiedKFold


class DataSplitter:
    def __init__(self, x, y, num_folds=5, random_seed=42):
        self.x = np.array(x)
        self.y = np.array(y)
        self.num_folds = num_folds
        self.random_seed = random_seed

    def get_splits(self):
        for train_index, test_index in self.get_kfold_indices():
            yield self._build_split(train_index, test_index)

    def get_kfold_indices(self):
        cv = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.random_seed)
        for train_index, test_index in cv.split(self.x):
            yield train_index, test_index

    def get_stratified_splits(self):
        yield from self.get_stratified_kfold_splits()

    def get_stratified_indices(self):
        yield from self.get_stratified_kfold_indices()

    def get_stratified_kfold_splits(self):
        for train_index, test_index in self.get_stratified_kfold_indices():
            yield self._build_split(train_index, test_index)

    def get_stratified_kfold_indices(self):
        y = self._single_label_y()
        cv = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.random_seed)
        for train_index, test_index in cv.split(self.x, y):
            yield train_index, test_index

    def get_scaffold_splits(self):
        for train_index, test_index in self.get_scaffold_indices():
            yield self._build_split(train_index, test_index)

    def get_scaffold_indices(self):
        test_size = 1.0 / self.num_folds
        sizes = (1.0 - test_size, 0.0, test_size)
        for seed in self._scaffold_seeds():
            train_replicates, _, test_replicates = self._chemprop_scaffold_balanced_indices(
                self.x,
                sizes=sizes,
                seed=seed,
                num_replicates=1,
            )
            train_index = np.array(train_replicates[0], dtype=int)
            test_index = np.array(test_replicates[0], dtype=int)
            yield train_index, test_index

    def get_split_indices(self, split_method="kfold", task_type="regression"):
        split_method = split_method.lower().replace("-", "_")
        if split_method in {"scaffold", "scaffold_balanced", "chemprop_scaffold"}:
            yield from self.get_scaffold_indices()
        elif split_method in {"stratified", "stratified_kfold", "skfold"}:
            yield from self.get_stratified_kfold_indices()
        elif split_method == "kfold":
            if task_type == "classification":
                yield from self.get_stratified_kfold_indices()
            else:
                yield from self.get_kfold_indices()
        else:
            raise ValueError(
                f"Unsupported split_method {split_method}. "
                "Available: kfold, stratified, stratified_kfold, scaffold"
            )

    def get_splits_by_method(self, split_method="kfold", task_type="regression"):
        for train_index, test_index in self.get_split_indices(split_method, task_type):
            yield self._build_split(train_index, test_index)

    def _build_split(self, train_index, test_index):
        x_train, x_test = self.x[train_index], self.x[test_index]
        y_train, y_test = self.y[train_index], self.y[test_index]
        return x_train, x_test, y_train, y_test, train_index, test_index

    def _single_label_y(self):
        y = np.asarray(self.y)
        if y.ndim == 2 and y.shape[1] == 1:
            return y.ravel()
        if y.ndim == 1:
            return y
        raise ValueError(
            "Stratified split supports single-label classification only. "
            "Use split_method=scaffold or split_method=kfold for multilabel data."
        )

    def _scaffold_seeds(self):
        if isinstance(self.random_seed, (list, tuple, np.ndarray)):
            seeds = [int(seed) for seed in self.random_seed]
            if len(seeds) != self.num_folds:
                raise ValueError(
                    "When random_seed is a sequence for scaffold splitting, "
                    "its length must match num_folds."
                )
            return seeds
        base_seed = 42 if self.random_seed is None else int(self.random_seed)
        return [base_seed + i for i in range(self.num_folds)]

    @staticmethod
    def _chemprop_scaffold_balanced_indices(smiles, sizes, seed, num_replicates=1):
        if chemprop_data is None:
            raise ImportError(
                "Chemprop is required for scaffold-balanced splitting. "
                "Install Chemprop, e.g. pip install chemprop."
            )

        if hasattr(chemprop_data, "make_split_indices"):
            datapoints = [chemprop_data.MoleculeDatapoint.from_smi(str(smi)) for smi in smiles]
            mols = [datapoint.mol for datapoint in datapoints]
            return chemprop_data.make_split_indices(
                mols,
                split="scaffold_balanced",
                sizes=sizes,
                seed=seed,
                num_replicates=num_replicates,
            )

        if not hasattr(chemprop_data, "split_data"):
            raise ImportError("Installed Chemprop does not provide a scaffold_balanced split API.")

        train_replicates, val_replicates, test_replicates = [], [], []
        for replicate in range(num_replicates):
            split_seed = seed + replicate
            datapoints = [
                chemprop_data.MoleculeDatapoint(
                    smiles=[str(smi)],
                    targets=[0.0],
                    row=OrderedDict([("_chemprompt_index", str(idx))]),
                )
                for idx, smi in enumerate(smiles)
            ]
            dataset = chemprop_data.MoleculeDataset(datapoints)
            train_data, val_data, test_data = chemprop_data.split_data(
                dataset,
                split_type="scaffold_balanced",
                sizes=sizes,
                seed=split_seed,
            )
            train_replicates.append(DataSplitter._indices_from_chemprop_dataset(train_data))
            val_replicates.append(DataSplitter._indices_from_chemprop_dataset(val_data))
            test_replicates.append(DataSplitter._indices_from_chemprop_dataset(test_data))

        return train_replicates, val_replicates, test_replicates

    @staticmethod
    def _indices_from_chemprop_dataset(dataset):
        return [int(datapoint.row["_chemprompt_index"]) for datapoint in dataset]
