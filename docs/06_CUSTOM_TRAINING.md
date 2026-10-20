# 06. Custom Dataset Training

## Workflow
1. Dataset Upload
2. Column Mapping
3. Validation
4. Task Configuration
5. Descriptor Mode
6. Split Strategy
7. Representation 생성/조회
8. 5-Fold Training
9. Evaluation
10. Model Registration

## Dataset
최소:
- SMILES column
- target column

## Split
Custom dataset도 built-in dataset과 동일한 split engine을 사용한다.

지원:
- RANDOM 5-fold
- SCAFFOLD 5-fold

각 fold model은 약 80% train data를 사용한다.

Scaffold split을 선택한 경우 seed 정책은 `03_SPLIT_AND_EVALUATION.md`를 따른다.

GA custom training도 별도 임의 split을 만들지 않고 동일한 Split Service 결과를 사용한다.
