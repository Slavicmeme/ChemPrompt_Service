import numpy as np

from app.services.descriptor_service import DescriptorEngine
from app.services.prediction_service import predict_with_bundle
from app.services.registry_service import ModelBundle
from app.services.representation import current_info


class Const:
    def __init__(self, v):
        self.v = v

    def predict(self, X):
        return np.full((len(X), 1), self.v)


class Proba:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):  # MultiOutputClassifier-style list output (2 tasks)
        return [np.array([[1 - self.p, self.p]] * len(X)), np.array([[1 - self.p / 2, self.p / 2]] * len(X))]


def _bundle(task, models, smiles):
    e = DescriptorEngine.fit(smiles)
    info = current_info()
    meta = {"task_type": task, "target_columns": ["label"] if task == "regression" else ["A", "B"],
            "representation": {"backend": info.backend, "model": info.model}, "classification_threshold": 0.5}
    return ModelBundle(1, meta, e, [{"fold": i + 1, "model": m, "flag": e.full_flag()} for i, m in enumerate(models)])


def test_regression_mean_of_folds(smiles):
    out = predict_with_bundle(_bundle("regression", [Const(v) for v in (1, 2, 3, 4, 5)], smiles), "CCO")
    assert out["result"]["value"] == 3.0
    assert len(out["fold_outputs"]) == 5


def test_classification_mean_probability_then_threshold(smiles):
    out = predict_with_bundle(_bundle("multilabel", [Proba(p) for p in (0.4, 0.6, 0.7, 0.5, 0.6)], smiles), "CCO")
    a, b = out["result"]["tasks"]
    assert abs(a["probability"] - 0.56) < 1e-9 and a["label"] == 1
    assert abs(b["probability"] - 0.28) < 1e-9 and b["label"] == 0
