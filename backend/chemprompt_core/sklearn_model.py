import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multioutput import MultiOutputClassifier
from chemprompt_core.data_splitter import DataSplitter
from chemprompt_core.metrics import get_metric_scores
from chemprompt_core.base_model import BaseModel


class ScikitLearnModel(BaseModel):
    def __init__(
        self,
        model_type="classification",
        fold=5,
        metric_names=None,
        save_dir=None,
        max_iter=500,
        split_method="kfold",
        random_seed=42,
    ):
        """Initialize a scikit-learn model with cross-validation support.

        Args:
            model_type (str): 'classification', 'multilabel', or 'regression'.
            fold (int): Number of folds for cross-validation.
            metric_names (list, optional): Metrics to calculate.
            save_dir (str, optional): Directory to save results.
            split_method (str): 'kfold' or 'scaffold' for fold evaluation.
        """
        super(ScikitLearnModel, self).__init__()
        self.model_type = model_type
        self.fold = fold
        self.max_iter = max_iter
        self.split_method = split_method
        self.random_seed = random_seed
        self._metric_names_provided = metric_names is not None

        self.model = self._default_model(model_type)
        self.metric_names = metric_names or self._default_metrics()
        self.save_dir = save_dir or f"./results/{self.model_type}"
        os.makedirs(self.save_dir, exist_ok=True)

    def _default_model(self, model_type):
        if model_type == "classification":
            return LogisticRegression(max_iter=self.max_iter, random_state=2025)
        if model_type == "multilabel":
            base = LogisticRegression(
                max_iter=self.max_iter,
                solver="lbfgs",
                penalty="l2",
                random_state=2025,
            )
            return MultiOutputClassifier(base)
        if model_type == "regression":
            return Ridge(max_iter=self.max_iter, random_state=2025)
        raise ValueError(f"Unsupported model type: {model_type}")

    def _default_metrics(self):
        if self.model_type == "classification":
            return ["accuracy", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc"]
        if self.model_type == "multilabel":
            return ["accuracy_micro", "accuracy_macro", "f1_micro", "f1_macro", "f1_micro_macro", "precision_micro", "precision_macro", "recall_micro", "recall_macro", "hamming_loss", "roc_auc_micro", "roc_auc_macro", "pr_auc_micro", "pr_auc_macro"]
        if self.model_type == "regression":
            return ["rmse", "r2", "pearson", "spearman"]
        raise ValueError(f"Unsupported model type: {self.model_type}")

    def _maybe_set_multilabel(self, y):
        y = np.asarray(y)
        if len(y.shape) == 2 and y.shape[1] > 1 and self.model_type != "multilabel":
            self.model_type = "multilabel"
            self.model = self._default_model(self.model_type)
            if not self._metric_names_provided:
                self.metric_names = self._default_metrics()

    def _prepare_y(self, y):
        y = np.asarray(y)
        if self.model_type == "classification" and y.ndim == 2 and y.shape[1] == 1:
            return y.ravel()
        return y

    def _prediction_scores(self, x):
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x)
            if isinstance(proba, list):
                return np.asarray([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in proba]).T
            if proba.ndim == 2 and proba.shape[1] == 2:
                return proba[:, 1]
            return proba
        if hasattr(self.model, "decision_function"):
            return self.model.decision_function(x)
        return None

    def _score_metrics(self, y_true, y_pred, y_score=None, y_reference=None):
        return get_metric_scores(
            self.metric_names,
            y_true,
            y_pred,
            y_score=y_score,
            task_type=self.model_type,
            y_reference=y_reference,
        )

    def _fold_indices(self, data_list, x, y):
        splitter_x = data_list if self.split_method == "scaffold" else x
        splitter = DataSplitter(splitter_x, y, num_folds=self.fold, random_seed=self.random_seed)
        yield from splitter.get_split_indices(
            split_method=self.split_method,
            task_type=self.model_type,
        )

    def fit_and_evaluate(self, data_list, train_x, train_y, test_x, test_y):
        metric_results = {metric: [] for metric in self.metric_names}

        train_x = np.asarray(train_x)
        train_y = np.asarray(train_y)
        test_x = np.asarray(test_x)
        test_y = np.asarray(test_y)

        self._maybe_set_multilabel(train_y)
        metric_results = {metric: [] for metric in self.metric_names}
        self.model = self._default_model(self.model_type)

        train_y_fit = self._prepare_y(train_y)
        self.model.fit(train_x, train_y_fit)
        y_pred = self.model.predict(test_x)
        y_score = None if self.model_type == "regression" else self._prediction_scores(test_x)

        scores = self._score_metrics(
            test_y, y_pred, y_score, y_reference=test_y
        )
        for metric_name, metric_score in scores.items():
            metric_results[metric_name].append(metric_score)

        all_predictions = []
        for idx, pred in enumerate(y_pred):
            try:
                prediction = list(pred)
            except TypeError:
                prediction = pred
            all_predictions.append({
                "SMILES": data_list[idx],
                "GroundTruth": test_y[idx],
                "Prediction": prediction,
            })

        self._save_predictions(all_predictions)
        self._save_combined_results(metric_results)

        print("\n[RESULTS]")
        for k, v in metric_results.items():
            print(f"{k}: {v[0]:.3f}")

        return metric_results

    def fit_and_evaluate_fold(self, data_list, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        self._maybe_set_multilabel(y)

        all_predictions = []
        metric_results = {metric: [] for metric in self.metric_names}

        for fold_idx, (train_idx, test_idx) in enumerate(self._fold_indices(data_list, x, y), start=1):
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            self.model = self._default_model(self.model_type)
            self.model.fit(x_train, self._prepare_y(y_train))
            y_pred = self.model.predict(x_test)
            y_score = None if self.model_type == "regression" else self._prediction_scores(x_test)

            scores = self._score_metrics(
                y_test, y_pred, y_score, y_reference=y_test
            )
            for metric_name, metric_score in scores.items():
                metric_results[metric_name].append(metric_score)

            for idx, pred in zip(test_idx, y_pred):
                ground_truth = y[idx]
                try:
                    prediction = list(pred)
                except TypeError:
                    prediction = pred
                all_predictions.append({
                    "Fold": fold_idx,
                    "SMILES": data_list[idx],
                    "GroundTruth": ground_truth,
                    "Prediction": prediction,
                })

        self._save_predictions(all_predictions)
        self._save_combined_results(metric_results)

        summary_results = {}
        for metric_name, scores in metric_results.items():
            summary_results[f"{metric_name}_mean"] = np.nanmean(scores)
            summary_results[f"{metric_name}_std"] = np.nanstd(scores)

        return summary_results

    def _save_predictions(self, predictions):
        if self.save_dir is None:
            save_path = os.path.join("./results/", f"{self.model_type}_Predictions.csv")
        else:
            save_path = os.path.join(self.save_dir, f"{self.model_type}_Predictions.csv")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        prediction_df = pd.DataFrame(predictions, columns=["Fold", "SMILES", "GroundTruth", "Prediction"])
        prediction_df.to_csv(save_path, index=False)
        print(f"Predictions saved to {save_path}")

    def _save_combined_results(self, metric_results):
        combined_results = {
            "Metric": [],
            "Mean": [],
            "Standard Deviation": [],
        }

        for metric_name in self.metric_names:
            scores = metric_results[metric_name]
            combined_results["Metric"].append(metric_name)
            combined_results["Mean"].append(np.nanmean(scores))
            combined_results["Standard Deviation"].append(np.nanstd(scores))

        combined_df = pd.DataFrame(combined_results)
        combined_results_path = os.path.join(self.save_dir, f"{self.model_type}_CombinedResults.csv")
        combined_df.to_csv(combined_results_path, index=False)
        print(f"Combined results saved to {combined_results_path}")

    def predict(self, x, y_true=None, smiles_list=None):
        y_pred = self.model.predict(x)

        if y_true is not None:
            save_path = os.path.join(self.save_dir, f"{self.model_type}_Predictions.csv")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            records = []
            for i, pred in enumerate(y_pred):
                try:
                    pred_val = list(pred)
                except TypeError:
                    pred_val = pred
                record = {
                    "GroundTruth": y_true[i],
                    "Prediction": pred_val,
                }
                if smiles_list:
                    record["SMILES"] = smiles_list[i]
                records.append(record)

            df = pd.DataFrame(records)
            df.to_csv(save_path, index=False)
            print(f"[INFO] Test predictions saved to {save_path}")

        return y_pred
