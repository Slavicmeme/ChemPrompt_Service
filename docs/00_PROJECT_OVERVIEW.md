# 00. Project Overview

## 사용자
AI 개발자가 아니라 화학·생물·약학 등 자연과학 연구자가 주 사용자다.

## Built-in Properties
FreeSolv, ESOL, Lipo, ClinTox.

## Descriptor Modes
- FULL
- FEATURE_SELECTION (Top K=50)
- GENETIC

Genetic은 ChemPrompt의 핵심 descriptor optimization mode다. V1에서는 실제 GA 실행 결과를 mock flags로 대체할 수 있지만 API/DB/artifact는 실제 GA 결과를 수용하도록 만든다.

## Custom Dataset
사용자가 CSV를 업로드하고 SMILES/target을 지정하여 자신의 sklearn downstream model을 학습할 수 있다.

## Source Reuse
scientific logic 구현 전에 반드시 `./Chemprompt`를 조사한다. 경로의 대소문자는 실제 checkout을 기준으로 한다. repository에 기존 코드가 있으면 이를 adapter/service로 감싸 재사용한다.
