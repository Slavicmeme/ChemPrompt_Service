import pandas as pd
import numpy as np
import inspect
from collections import OrderedDict
from typing import Sequence, Callable

from rdkit import Chem
from rdkit.Chem import (
    Descriptors, Crippen, Lipinski, rdMolDescriptors,
    Fragments, QED,
)

from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    f_regression,
    mutual_info_regression,
    r_regression,
)
from sklearn.linear_model import Lasso, LinearRegression


class descriptor_selection:
    """
    Perform feature selection on pre-computed descriptor matrices.
    Designed to:
    - Receive already-filtered descriptor DataFrame (e.g., from LLMModel)
    - Fit feature selector on training data only (to prevent leakage)
    - Transform any dataset (train/test) to same selected feature subset
    """
    def __init__(self, y=None):
        self.y = y
        self.selector = None
        self.selected_columns = None
        self.selected_indices = None
        self.selected_mask = None

    def fit_feature_selector(
        self,
        X_train: pd.DataFrame,
        y_train,
        k=50,
        method="f_regression",
        **method_params,
    ):
        """Fit a feature selector using only training data.

        Supported methods are ``f_regression``, ``r_regression``,
        ``mutual_info``, ``lasso``, ``rfe``, ``random_search``, and
        ``correlation``.
        Method-specific parameters are optional; scikit-learn defaults are
        used for LASSO and RFE, and random search uses ``random_state=42`` by
        default. RFE uses ``LinearRegression`` as its default estimator.
        """
        if not isinstance(X_train, pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if X_train.shape[1] == 0:
            raise ValueError("X_train must contain at least one feature.")
        if k <= 0:
            raise ValueError("k must be a positive integer.")

        method = self._normalize_method_name(method)
        if method in {"lasso", "rfe", "random_search", "correlation"}:
            return self._fit_ranked_selector(
                X_train,
                y_train,
                k=k,
                method=method,
                method_params=method_params,
            )
        if method_params:
            raise ValueError(
                f"Method-specific parameters are not supported for {method}: "
                f"{sorted(method_params)}"
            )

        method_func = self._get_method(method)
        k = min(k, X_train.shape[1])
        y_train = np.asarray(y_train)

        if y_train.ndim == 2 and y_train.shape[1] > 1:
            score_rows = []
            for task_idx in range(y_train.shape[1]):
                scores = method_func(X_train, y_train[:, task_idx])
                if isinstance(scores, tuple):
                    scores = scores[0]
                score_rows.append(np.asarray(scores, dtype=float))

            mean_scores = np.nanmean(np.vstack(score_rows), axis=0)
            mean_scores = np.nan_to_num(mean_scores, nan=-np.inf)
            selected_indices = np.argsort(mean_scores)[-k:]
            selected_indices = np.sort(selected_indices)

            mask = np.zeros(X_train.shape[1], dtype=bool)
            mask[selected_indices] = True

            self.selector = None
            self.selected_columns = X_train.columns[mask]
            self.selected_mask = mask
            self.selected_indices = selected_indices.tolist()
        else:
            selector = SelectKBest(score_func=method_func, k=k)
            selector.fit(X_train, y_train.ravel())

            self.selector = selector
            self.selected_columns = X_train.columns[selector.get_support()]
            self.selected_mask = selector.get_support()
            self.selected_indices = np.where(selector.get_support())[0].tolist()

        print(f"[Feature Selection] {len(self.selected_columns)} features selected using {method}.")
        return self

    def _fit_ranked_selector(self, X_train, y_train, k, method, method_params):
        k = min(int(k), X_train.shape[1])
        X_values = np.asarray(X_train, dtype=float)
        y_values = np.asarray(y_train, dtype=float)
        if y_values.ndim == 1:
            y_values = y_values.reshape(-1, 1)

        if len(X_values) != len(y_values):
            raise ValueError("X_train and y_train must contain the same number of samples.")

        if method == "lasso":
            task_scores = []
            for task_idx in range(y_values.shape[1]):
                model = Lasso(**method_params)
                model.fit(X_values, y_values[:, task_idx])
                task_scores.append(np.abs(np.asarray(model.coef_, dtype=float)))
            scores = np.mean(np.vstack(task_scores), axis=0)
        elif method == "rfe":
            params = dict(method_params)
            estimator = params.pop("estimator", None)
            if "n_features_to_select" in params:
                raise ValueError(
                    "Use the k argument instead of n_features_to_select for RFE."
                )

            task_rankings = []
            for task_idx in range(y_values.shape[1]):
                task_estimator = estimator if estimator is not None else LinearRegression()
                selector = RFE(
                    estimator=task_estimator,
                    n_features_to_select=k,
                    **params,
                )
                selector.fit(X_values, y_values[:, task_idx])
                task_rankings.append(np.asarray(selector.ranking_, dtype=float))

            # Lower RFE ranking is better. Averaging rankings provides one
            # stable ranking when y contains multiple label columns.
            mean_rankings = np.mean(np.vstack(task_rankings), axis=0)
            selected_indices = np.argsort(mean_rankings, kind="stable")[:k]
            selected_indices = np.sort(selected_indices)
            return self._set_selected_features(X_train, selected_indices, method)
        elif method == "random_search":
            params = dict(method_params)
            random_state = params.pop("random_state", params.pop("seed", 42))
            if params:
                raise ValueError(
                    "Unknown random_search parameters: "
                    f"{sorted(params)}. Use random_state (or seed)."
                )
            rng = np.random.RandomState(random_state)
            selected_indices = np.sort(
                rng.choice(X_train.shape[1], size=k, replace=False)
            )
            return self._set_selected_features(X_train, selected_indices, method)
        else:
            if method_params:
                raise ValueError(
                    "correlation does not accept method-specific parameters: "
                    f"{sorted(method_params)}"
                )
            task_scores = []
            for task_idx in range(y_values.shape[1]):
                target = y_values[:, task_idx]
                centered_x = X_values - np.nanmean(X_values, axis=0)
                centered_y = target - np.nanmean(target)
                numerator = np.nansum(centered_x * centered_y[:, None], axis=0)
                denominator = np.sqrt(
                    np.nansum(centered_x ** 2, axis=0)
                    * np.nansum(centered_y ** 2)
                )
                correlations = np.divide(
                    numerator,
                    denominator,
                    out=np.zeros_like(numerator, dtype=float),
                    where=denominator != 0,
                )
                task_scores.append(np.abs(correlations))
            scores = np.mean(np.vstack(task_scores), axis=0)

        scores = np.nan_to_num(scores, nan=-np.inf)
        selected_indices = np.argsort(-scores, kind="stable")[:k]
        selected_indices = np.sort(selected_indices)
        return self._set_selected_features(X_train, selected_indices, method)

    def _set_selected_features(self, X_train, selected_indices, method):
        selected_indices = np.asarray(selected_indices, dtype=int)
        mask = np.zeros(X_train.shape[1], dtype=bool)
        mask[selected_indices] = True

        self.selector = None
        self.selected_columns = X_train.columns[mask]
        self.selected_mask = mask
        self.selected_indices = selected_indices.tolist()

        print(f"[Feature Selection] {len(self.selected_columns)} features selected using {method}.")
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted selector to another dataset (train/test)"""
        if self.selected_columns is None:
            raise RuntimeError("Feature selector not fitted. Run fit_feature_selector() first.")
        missing = [c for c in self.selected_columns if c not in X.columns]
        if missing:
            raise ValueError(f"Input missing {len(missing)} columns: {missing[:5]}")
        return X[self.selected_columns]

    @staticmethod
    def _get_method(name: str):
        methods = {
            "f_regression": f_regression,
            "r_regression": r_regression,
            "mutual_info": mutual_info_regression,
        }
        if name not in methods:
            raise ValueError(f"Unknown feature selection method: {name}")
        return methods[name]

    @staticmethod
    def _normalize_method_name(name: str):
        normalized = str(name).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "random": "random_search",
            "randomsearch": "random_search",
            "correlation_top_k": "correlation",
            "correlation_topk": "correlation",
            "top_k_correlation": "correlation",
            "recursive_feature_elimination": "rfe",
        }
        return aliases.get(normalized, normalized)
