"""ChemPrompt core — the parts of the ChemPrompt research code that this service uses.

Extracted from the original repository ``Slavicmeme/Chemprompt`` (MIT License, Copyright (c) 2026 APC Lab,
see ``LICENSE`` in this folder). The scientific logic is copied unchanged; only the import paths were
changed from ``Chemprompt.<subpackage>.<module>`` to ``chemprompt_core.<module>``.

| this package               | original file                                         | used for                                  |
|----------------------------|-------------------------------------------------------|-------------------------------------------|
| llm_model.py               | Chemprompt/embeddings/full_descriptors_embedding.py   | RDKit descriptors, prompts, Aya embedding |
| custom_model.py            | Chemprompt/config/custom_model.py                     | model repo name parsing (llm_model)       |
| data_splitter.py           | Chemprompt/utils/data_splitter.py                     | 5-fold random / scaffold splits           |
| metrics.py                 | Chemprompt/utils/metrics.py                           | evaluation metrics                        |
| descriptor_selection.py    | Chemprompt/feature_selection/descriptor_selection.py  | Top-K (f_regression) feature selection    |
| base_model.py              | Chemprompt/models/base_model.py                       | base class of sklearn_model               |
| sklearn_model.py           | Chemprompt/models/sklearn_model.py                    | Ridge / LogisticRegression wrappers       |
| genetic.py                 | Chemprompt/genetic/genetic.py                         | genetic algorithm (GA_MODE=genetic)       |
| data_loader.py             | Chemprompt/data/data_loader.py                        | built-in MolNet datasets (bootstrap)      |
"""
