# 08. Model Registry

## Model Version
하나의 serving 가능한 논리 모델 버전.

기본 artifact:
- metadata.json
- fold_1.joblib ... fold_5.joblib

필요 시 fold별:
- selected_descriptors.json
- ga_flags.json
- preprocessing artifact

향후 optional:
- full_fit.joblib

## State
UPLOADED -> VALIDATING -> READY -> SERVING -> ARCHIVED

Upload != Serving.

## metadata.json
최소:
- owner
- dataset/version
- property/task
- descriptor_method
- split_strategy
- n_folds=5
- split seed/config
- representation model/version
- prompt version
- sklearn model type/version
- fold artifacts/checksums
- metrics
