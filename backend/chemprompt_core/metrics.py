import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    r2_score,
)


def _flatten(y):
    return np.asarray(y).ravel()


def zscore_regression_targets(y_true, y_pred, y_reference):
    """Standardize regression targets using the provided reference targets.

    Predictions remain untouched outside metric calculation; this helper is
    intended to put fold-level regression metrics, especially RMSE, on a
    consistent z-score scale.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    reference = np.asarray(y_reference, dtype=float)
    mean = float(np.nanmean(reference))
    std = float(np.nanstd(reference))
    if not np.isfinite(std) or std <= 0:
        return y_true, y_pred
    return (y_true - mean) / std, (y_pred - mean) / std


def _safe_auc(metric_func, y_true, y_score, average="macro"):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    try:
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            return metric_func(_flatten(y_true), _flatten(y_score))
        return metric_func(y_true, y_score, average=average)
    except ValueError:
        return np.nan


def _accuracy_macro(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
        return accuracy_score(_flatten(y_true), _flatten(y_pred))
    return float(np.mean([accuracy_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]))


def _f1_micro_macro(y_true, y_pred):
    micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return float((micro + macro) / 2.0)


def get_metric_score(metric_name, y_true, y_pred, y_score=None):
    """Calculate a metric score.

    ``y_score`` is optional and is used for ROC-AUC / PR-AUC when available.
    For classification metrics that only need labels, ``y_pred`` is used.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    score_input = y_pred if y_score is None else np.asarray(y_score)

    metric_mapping = {
        # Classification metrics
        "accuracy": lambda: accuracy_score(y_true, y_pred),
        "subset_accuracy": lambda: accuracy_score(y_true, y_pred),
        "accuracy_micro": lambda: accuracy_score(_flatten(y_true), _flatten(y_pred)),
        "accuracy_macro": lambda: _accuracy_macro(y_true, y_pred),
        "balanced_accuracy": lambda: balanced_accuracy_score(_flatten(y_true), _flatten(y_pred)),
        "mcc": lambda: matthews_corrcoef(_flatten(y_true), _flatten(y_pred)),
        "hamming_loss": lambda: hamming_loss(y_true, y_pred),
        "jaccard_micro": lambda: jaccard_score(y_true, y_pred, average="micro", zero_division=0),
        "jaccard_macro": lambda: jaccard_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_micro": lambda: f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": lambda: f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_micro_macro": lambda: _f1_micro_macro(y_true, y_pred),
        "f1_weighted": lambda: f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_micro": lambda: precision_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_macro": lambda: precision_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": lambda: precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_micro": lambda: recall_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_macro": lambda: recall_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_weighted": lambda: recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "roc_auc": lambda: _safe_auc(roc_auc_score, y_true, score_input),
        "roc_auc_micro": lambda: _safe_auc(roc_auc_score, y_true, score_input, average="micro"),
        "roc_auc_macro": lambda: _safe_auc(roc_auc_score, y_true, score_input, average="macro"),
        "pr_auc": lambda: _safe_auc(average_precision_score, y_true, score_input),
        "pr_auc_micro": lambda: _safe_auc(average_precision_score, y_true, score_input, average="micro"),
        "pr_auc_macro": lambda: _safe_auc(average_precision_score, y_true, score_input, average="macro"),

        # Regression metrics
        "rmse": lambda: np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": lambda: r2_score(y_true, y_pred),
        "mae": lambda: mean_absolute_error(y_true, y_pred),
        "mse": lambda: mean_squared_error(y_true, y_pred),

        # Correlation metrics
        "pearson": lambda: pearsonr(_flatten(y_true), _flatten(y_pred))[0],
        "spearman": lambda: spearmanr(_flatten(y_true), _flatten(y_pred))[0],
    }

    if metric_name not in metric_mapping:
        raise ValueError(
            f"Unsupported metric '{metric_name}'. Available metrics: {list(metric_mapping.keys())}"
        )
    return metric_mapping[metric_name]()


def get_metric_scores(metric_names, y_true, y_pred, y_score=None, task_type=None, y_reference=None):
    """Calculate multiple metrics through the shared metric utility.

    For regression, when ``y_reference`` is provided, targets and predictions
    are standardized with those reference-target statistics before computing
    metrics. Passing the fold test targets gives test-set z-score RMSE.
    """
    if task_type == "regression" and y_reference is not None:
        y_true, y_pred = zscore_regression_targets(y_true, y_pred, y_reference)

    return {
        metric_name: get_metric_score(metric_name, y_true, y_pred, y_score)
        for metric_name in metric_names
    }


def summarize_metric_results(metric_results):
    """Convert metric result lists/scalars into a flat scalar dictionary.

    Model evaluators in this repository commonly return ``{metric: [values]}``
    for compatibility with fold-level evaluation. Notebook runners should use
    this helper instead of calculating means directly, so metric post-processing
    stays centralized in this module.
    """
    summary = {}
    for metric_name, values in metric_results.items():
        arr = np.asarray(values, dtype=float)
        summary[metric_name] = float(np.nanmean(arr)) if arr.size else float("nan")
    return summary


def get_metric_function(metric_name):
    """Returns a two-argument metric function for backward compatibility."""
    return lambda y_true, y_pred: get_metric_score(metric_name, y_true, y_pred)
