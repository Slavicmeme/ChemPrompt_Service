# 04. Descriptor Modes & GA

## Full
기존 ChemPrompt descriptor 전체 사용.

## Feature Selection
Top K=50.

## Genetic
Top-50 후보를 기반으로 GA-selected descriptor subset을 사용한다.

## GA + 5-fold

GA도 일반 inference model과 동일한 5-fold split contract를 따른다.

```text
Dataset
  |
5-fold split
  |
  +-- Fold 1 Train 80% -> Feature Selection / GA -> Model 1
  |         Validation 20% -----------------------> Evaluation 1
  |
  +-- Fold 2 Train 80% -> Feature Selection / GA -> Model 2
  |         Validation 20% -----------------------> Evaluation 2
  |
  ...
  |
  `-- Fold 5
```

GA/feature selection이 해당 fold의 outer validation 정보를 보고 descriptor를 선택하면 안 된다.

GA descriptor flags는 fold마다 달라질 수 있으므로 fold-level artifact로 저장할 수 있어야 한다.

예:
```text
fold_1/
  ga_flags.json
  selected_descriptors.json
  model.joblib

fold_2/
  ga_flags.json
  selected_descriptors.json
  model.joblib
...
```

V1에서 GA 계산 자체는 mock flags를 사용해도 된다. 단 storage/API/schema는 실제 GA 교체가 가능해야 한다.

## Source First
`./Chemprompt`의 feature_selection, genetic, experiments 및 관련 utility를 먼저 조사한다.
