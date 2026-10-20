# 05. Prediction

```text
SMILES
 -> RDKit descriptor
 -> Descriptor Mode
 -> ChemPrompt prompt
 -> Aya Representation
 -> fold-specific sklearn model
 -> aggregation
```

Regression:
`prediction = mean(fold_predictions)`

Classification:
`probability = mean(fold_probabilities)` 후 threshold 적용을 기본으로 한다.

Genetic mode에서는 fold별 GA descriptor configuration과 해당 fold model이 짝을 이뤄야 한다.

Prediction record에는 최소:
- model_version
- split_strategy
- fold count
- seed/config reference
- descriptor mode
- representation version
을 남긴다.
