import os
import random
import numpy as np
import pandas as pd

from chemprompt_core.sklearn_model import ScikitLearnModel
from chemprompt_core.data_splitter import DataSplitter


class genetic:
    """
    Genetic Algorithm for descriptor subset optimization.

    The constructor remains backward-compatible with the previous notebooks,
    while the fitness evaluation can now use kfold/scaffold splits and
    regression/classification/multilabel models.
    """

    def __init__(
        self,
        smiles_list,
        y,
        llm_model,
        dataset_name,
        selected_indices,
        num_generations=10,
        pop_size=10,
        top_k=3,
        mutation_rate=0.01,
        save_predictions=False,
        split_method="kfold",
        num_folds=3,
        random_seed=42,
        model_class=None,
        model_type="regression",
        model_params=None,
        metric_names=None,
        selection_metric=None,
    ):
        self.smiles_list = list(smiles_list)
        self.y = np.asarray(y)
        self.llm = llm_model
        self.dataset_name = dataset_name
        self.selected_indices = selected_indices
        self.num_features = len(selected_indices)

        self.num_generations = num_generations
        self.pop_size = pop_size
        self.top_k = top_k
        self.mutation_rate = mutation_rate
        self.save_predictions = save_predictions

        self.split_method = split_method
        self.num_folds = num_folds
        self.random_seed = random_seed
        self.model_class = model_class or ScikitLearnModel
        self.model_type = self._infer_model_type(model_type, self.y)
        self.model_params = model_params or {}
        self.metric_names = metric_names or self._default_metrics(self.model_type)
        self.selection_metric = selection_metric or self._default_selection_metric(self.model_type)

        self.device = llm_model.device
        self.population = self._init_population()
        self.seen_individuals = set(tuple(ind) for ind in self.population)

        self.individual_metrics = {}
        self.metrics_dir = os.path.join("./evolutionary_process", self.dataset_name)
        os.makedirs(self.metrics_dir, exist_ok=True)
        self.population_history = []
        self.best_history = []

    @staticmethod
    def _infer_model_type(model_type, y):
        if model_type not in {"auto", None}:
            return model_type
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] > 1:
            return "multilabel"
        flat = y.ravel()
        if np.issubdtype(flat.dtype, np.number):
            flat = flat[~pd.isna(flat)]
        unique = set(np.unique(flat).tolist())
        if unique and unique.issubset({0, 1, 0.0, 1.0}):
            return "classification"
        return "regression"

    @staticmethod
    def _default_metrics(model_type):
        if model_type == "classification":
            return ["accuracy", "balanced_accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc"]
        if model_type == "multilabel":
            return ["accuracy_micro", "accuracy_macro", "f1_micro", "f1_macro", "f1_micro_macro", "precision_micro", "precision_macro", "recall_micro", "recall_macro", "hamming_loss", "roc_auc_micro", "roc_auc_macro", "pr_auc_micro", "pr_auc_macro"]
        return ["rmse", "r2", "pearson", "spearman"]

    @staticmethod
    def _default_selection_metric(model_type):
        if model_type == "multilabel":
            return "f1_micro_macro"
        if model_type == "classification":
            return "f1_macro"
        return "r2"

    @staticmethod
    def _jsonable_value(value):
        arr = np.asarray(value)
        if arr.ndim == 0:
            return arr.item()
        return arr.tolist()

    def _init_population(self):
        pop = []
        while len(pop) < self.pop_size:
            pop.append([random.choice([True, False]) for _ in range(self.num_features)])
        return pop

    def _mutate(self, individual):
        if random.random() > self.mutation_rate:
            return individual

        indices_on = [i for i, v in enumerate(individual) if v]
        indices_off = [i for i, v in enumerate(individual) if not v]

        if indices_on and indices_off:
            i_off = random.choice(indices_off)
            i_on = random.choice(indices_on)
            individual[i_off] = True
            individual[i_on] = False

        return individual

    def _crossover(self, parents):
        children = []
        while len(parents) + len(children) < self.pop_size:
            p1, p2 = random.sample(parents, 2)
            split_point = random.randint(1, self.num_features - 1)
            child = p1[:split_point] + p2[split_point:]

            if random.random() < self.mutation_rate:
                child = self._mutate(child)

            key = tuple(child)
            if key not in self.seen_individuals:
                self.seen_individuals.add(key)
                children.append(child)

        return parents + children

    def _fold_indices(self, X):
        scaffold_names = {"scaffold", "scaffold_balanced", "chemprop_scaffold"}
        split_name = self.split_method.lower().replace("-", "_")
        splitter_x = self.smiles_list if split_name in scaffold_names else X
        splitter = DataSplitter(splitter_x, self.y, num_folds=self.num_folds, random_seed=self.random_seed)
        yield from splitter.get_split_indices(split_method=self.split_method, task_type=self.model_type)

    def _new_eval_model(self):
        return self.model_class(
            model_type=self.model_type,
            fold=self.num_folds,
            metric_names=self.metric_names,
            save_dir=None,
            split_method=self.split_method,
            random_seed=self.random_seed,
            **self.model_params,
        )

    def _evaluate_population(self, embeddings_list):
        scores = []

        for idx, individual in enumerate(self.population):
            key = tuple(individual)
            if key not in self.individual_metrics:
                X = np.asarray(embeddings_list[idx])
                metric_results = {metric: [] for metric in self.metric_names}

                for fold_idx, (tr, te) in enumerate(self._fold_indices(X), 1):
                    X_tr, X_te = X[tr], X[te]
                    y_tr, y_te = self.y[tr], self.y[te]

                    eval_model = self._new_eval_model()
                    eval_model._maybe_set_multilabel(y_tr)
                    eval_model.model = eval_model._default_model(eval_model.model_type)
                    eval_model.model.fit(X_tr, eval_model._prepare_y(y_tr))
                    y_pred = eval_model.model.predict(X_te)
                    y_score = None if eval_model.model_type == "regression" else eval_model._prediction_scores(X_te)

                    fold_scores = eval_model._score_metrics(y_te, y_pred, y_score)
                    for metric_name, metric_score in fold_scores.items():
                        metric_results[metric_name].append(metric_score)

                    if self.save_predictions:
                        pred_dir = os.path.join(self.metrics_dir, f"generation_{self.current_gen}", f"ind_{idx + 1}")
                        os.makedirs(pred_dir, exist_ok=True)
                        rows = []
                        for local_idx, pred in enumerate(y_pred):
                            rows.append({
                                "fold": fold_idx,
                                "SMILES": self.smiles_list[int(te[local_idx])],
                                "GroundTruth": self._jsonable_value(y_te[local_idx]),
                                "Prediction": self._jsonable_value(pred),
                            })
                        pd.DataFrame(rows).to_csv(os.path.join(pred_dir, f"fold_{fold_idx}.csv"), index=False)

                self.individual_metrics[key] = {
                    metric: float(np.nanmean(values)) for metric, values in metric_results.items()
                }

            m = self.individual_metrics[key]
            flag_str = "".join("1" if b else "0" for b in individual)
            metric_value = m.get(self.selection_metric, np.nan)
            print(f"Ind {idx + 1}: {self.selection_metric}={metric_value:.3f} | flag={flag_str}")
            scores.append(metric_value)

        return scores

    def _save_generation_metrics(self):
        gen_records = []
        for idx, ind in enumerate(self.population):
            fstr = "".join("1" if b else "0" for b in ind)
            m = self.individual_metrics[tuple(ind)]
            gen_records.append({
                "generation": self.current_gen,
                "ind_idx": idx + 1,
                "flag": fstr,
                "selection_metric": self.selection_metric,
                **m,
            })
        df_gen = pd.DataFrame(gen_records)
        df_gen.to_csv(os.path.join(self.metrics_dir, f"population_gen_{self.current_gen}.csv"), index=False)
        self.population_history.append(df_gen)

        best_row = df_gen.loc[df_gen[self.selection_metric].idxmax()].to_dict()
        pd.DataFrame([best_row]).to_csv(os.path.join(self.metrics_dir, f"best_gen_{self.current_gen}.csv"), index=False)
        self.best_history.append(best_row)

    def _save_summary(self):
        if self.population_history:
            pd.concat(self.population_history, ignore_index=True).to_csv(
                os.path.join(self.metrics_dir, "population_all_generations.csv"), index=False
            )
        if self.best_history:
            pd.DataFrame(self.best_history).to_csv(
                os.path.join(self.metrics_dir, "best_each_generation.csv"), index=False
            )

    def run(self):
        best_individual, best_score = None, -np.inf
        no_improve = 0
        desc_count = len(self.llm.property_names)

        for gen in range(1, self.num_generations + 1):
            self.current_gen = gen
            print(f"\n=== Generation {gen}/{self.num_generations} ===")

            embeddings_list = []
            for i, ind in enumerate(self.population):
                full_flag = [False] * desc_count
                for j, desc_idx in enumerate(self.selected_indices):
                    if ind[j]:
                        full_flag[desc_idx] = True

                embed = self.llm.get_embeddings_for_ga(
                    self.smiles_list,
                    flag=full_flag,
                    save_dir=os.path.join(self.metrics_dir, f"generation_{gen}", "embeddings"),
                )
                embeddings_list.append(embed)

            scores = self._evaluate_population(embeddings_list)
            self._save_generation_metrics()

            gen_best_idx = int(np.nanargmax(scores))
            gen_best_score = scores[gen_best_idx]
            if gen_best_score > best_score:
                best_score = gen_best_score
                best_individual = self.population[gen_best_idx]
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 10:
                    print("[EarlyStop] 10 generations without improvement")
                    break

            top_k_idx = np.argsort(scores)[-self.top_k:]
            parents = [self.population[i] for i in top_k_idx]
            self.population = self._crossover(parents)

        self._save_summary()

        full_flag = [False] * desc_count
        for j, desc_idx in enumerate(self.selected_indices):
            if best_individual[j]:
                full_flag[desc_idx] = True

        best_flag_str = "".join("1" if b else "0" for b in best_individual)
        full_flag_str = "".join("1" if b else "0" for b in full_flag)
        best_metrics = self.individual_metrics[tuple(best_individual)]

        pd.DataFrame([{
            "flag": best_flag_str,
            "full_flag": full_flag_str,
            "selection_metric": self.selection_metric,
            **best_metrics,
        }]).to_csv(os.path.join(self.metrics_dir, "best_overall_metrics.csv"), index=False)

        print(f"\nBest overall {self.selection_metric}={best_score:.3f} | flag={best_flag_str}")
        self.best_flag_str = best_flag_str

        return best_individual, full_flag
