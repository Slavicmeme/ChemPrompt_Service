# 03. Split & Evaluation — 핵심 규칙

이 문서는 모든 학습 경로의 공통 split contract다.

## 공통 5-fold 원칙

GA 여부와 관계없이 모델 학습/평가는 5-fold를 기본으로 한다.

각 fold는 개념적으로:
- Train ≈ 80%
- Validation ≈ 20%

각 fold마다 validation partition이 달라진다. 전체 데이터의 20%를 영구적으로 버리는 방식이 아니다.

```text
Fold 1: train ~80% / validation ~20%
Fold 2: train ~80% / validation ~20%
Fold 3: train ~80% / validation ~20%
Fold 4: train ~80% / validation ~20%
Fold 5: train ~80% / validation ~20%
```

## Split Strategy

### RANDOM
일반 5-fold split. 재현 가능한 random seed를 사용한다.

### SCAFFOLD
분자를 scaffold group으로 묶은 뒤 scaffold 단위로 train/validation을 분리한다.

동일 scaffold group이 같은 fold의 train과 validation 양쪽에 섞이지 않도록 한다.

## Scaffold Seed Policy

기존 `./Chemprompt` 소스에서 scaffold split/seed 관련 구현을 먼저 조사한다.

기존 코드가 3개의 seed를 사용하는 구조라면 **5개의 seed로 확장한다.**

중요:
- seed 값 자체를 Agent가 임의로 새로 정하지 않는다.
- 기존 3개 seed 값/생성 규칙을 먼저 확인한다.
- 기존 규칙을 유지하면서 2개를 추가해 총 5개가 되도록 구현한다.
- 최종 seed list는 configuration으로 명시한다.
- dataset/model metadata에 사용한 seed를 저장한다.
- fold assignment도 재현 가능하게 저장하거나 checksum/version으로 추적한다.

즉 요구사항은 단순히 `n_splits=5`만 의미하지 않는다.
Scaffold 평가에서 사용하는 seed set도 기존 3개에서 5개로 확장한다.

## Leakage 금지

Outer validation data는 해당 fold의 feature selection/GA/model fitting에 사용하면 안 된다.

모든 preprocessing/selection은 가능한 한 fold train 영역에서 fit한다.
