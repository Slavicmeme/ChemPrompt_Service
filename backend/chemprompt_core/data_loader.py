import os
import pandas as pd
import deepchem as dc
from tdc.single_pred import ADME
from rdkit import Chem

class DataLoader:
    def __init__(self):
        self.supported_datasets = {
            # --- DeepChem MolNet ---
            "FreeSolv": self._load_freesolv,
            "ESOL": self._load_esol,
            "Lipo": self._load_lipo,
            "Caco2_Wang": self._load_caco2_wang,
            "ClinTox": self._load_clintox,
        }
        self.featurizer = dc.feat.CircularFingerprint(radius=2, size=2048)

    # --- Public API ---

    def load_dataset(self, dataset_name):
        if dataset_name not in self.supported_datasets:
            raise ValueError(f"Dataset '{dataset_name}' is not supported. Available datasets: {list(self.supported_datasets.keys())}")

        df = self.supported_datasets[dataset_name]()
        df = self._filter_valid_smiles(df, dataset_name)
        x = df['smiles'].tolist()
        y = df.drop(columns=['smiles']).values.tolist()
        return x, y

    def load_custom_csv(self, csv_path, smiles_column=None, label_columns=None):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file '{csv_path}' not found.")
        df = pd.read_csv(csv_path)

        if smiles_column:
            if smiles_column not in df.columns:
                raise ValueError(f"Column '{smiles_column}' not found in CSV.")
        else:
            smiles_column = next((col for col in df.columns if 'smiles' in col.lower()), None)
            if smiles_column is None:
                raise ValueError("CSV must contain a column with 'SMILES' in its name.")

        if label_columns:
            missing_labels = [label for label in label_columns if label not in df.columns]
            if missing_labels:
                raise ValueError(f"Label columns {missing_labels} not found in CSV.")
        else:
            label_columns = [col for col in df.columns if col != smiles_column and col.lower() not in ['num', 'name']]

        if df[label_columns[0]].astype(str).str.contains(',').any():
            df['labels'] = df[label_columns[0]].apply(lambda x: list(map(int, str(x).split(','))))
        else:
            df['labels'] = df[label_columns].values.tolist()

        df = df.rename(columns={smiles_column: 'smiles'})
        df = df.dropna().reset_index(drop=True)
        df = self._filter_valid_smiles(df, "Custom CSV")

        x = df['smiles'].tolist()
        y = df['labels'].tolist()
        return x, y

    # --- Utility ---

    def flatten_labels(self, y):
        return [row[0] if isinstance(row, list) else row for row in y]

    def _filter_valid_smiles(self, df, dataset_name="Dataset"):
        before = len(df)
        valid_mask = []
        invalid_rows = []

        for row_idx, smiles in df["smiles"].items():
            mol = Chem.MolFromSmiles(str(smiles))
            is_valid = mol is not None
            valid_mask.append(is_valid)
            if not is_valid:
                invalid_rows.append(row_idx)

        filtered = df.loc[valid_mask].reset_index(drop=True)
        removed = before - len(filtered)
        print(
            f"[SMILES validation] {dataset_name}: removed {removed} invalid rows "
            f"at row indices {invalid_rows}. {before} -> {len(filtered)}"
        )
        return filtered

    # --- DeepChem loaders ---
    def _load_freesolv(self):
        tasks, datasets, _ = dc.molnet.load_freesolv(featurizer=self.featurizer, splitter=None, transformers=[], reload=True)
        dataset = datasets[0]
        df = pd.DataFrame({"smiles": dataset.ids, "label": dataset.y[:, 0]}).dropna()
        print(df.shape)
        return df

    def _load_esol(self):
        tasks, datasets, _ = dc.molnet.load_delaney(featurizer=self.featurizer, splitter=None, transformers=[], reload=True)
        dataset = datasets[0]
        df = pd.DataFrame({"smiles": dataset.ids, "label": dataset.y[:, 0]}).dropna()
        print(df.shape)
        return df
        
    def _load_lipo(self):
        tasks, datasets, _ = dc.molnet.load_lipo(featurizer=self.featurizer, splitter=None, transformers=[], reload=True)
        dataset = datasets[0]
        df = pd.DataFrame({"smiles": dataset.ids, "label": dataset.y[:, 0]}).dropna()
        print(df.shape)
        return df
        
    def _load_clintox(self):
        tasks, datasets, _ = dc.molnet.load_clintox(featurizer=self.featurizer, splitter=None, transformers=[], reload=True)
        dataset = datasets[0]
        labels = pd.DataFrame(dataset.y, columns=tasks)
        labels = labels.apply(pd.to_numeric, errors="coerce")
        df = pd.concat([pd.Series(dataset.ids, name="smiles"), labels], axis=1)
        df = df.dropna().reset_index(drop=True)
        for task in tasks:
            df[task] = df[task].astype(int)
        print(("ClinTox", df.shape))
        return df

    # --- PyTDC ADME loaders ---

    def _load_caco2_wang(self):
        raw = ADME(name="Caco2_Wang").get_data()

        df = raw[["Drug", "Y"]].rename(
            columns={"Drug": "smiles", "Y": "label"}
        )
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        df = df.dropna()

        df = (
            df.groupby("smiles", as_index=False)
            .agg(label=("label", "mean"))
            .reset_index(drop=True)
        )

        print(("Caco2_Wang (unique, averaged)", df.shape))
        return df