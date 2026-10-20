# 16. ChemPrompt Source Reuse Policy

## 절대 규칙

새 scientific code를 작성하기 전에 프로젝트의 `./Chemprompt`를 먼저 조사한다.

사용자가 지정한 기준 경로는 `./Chemprompt`다.
다만 checkout의 실제 directory/package casing이 다를 수 있으므로 실제 filesystem을 확인한다.

원본 GitHub repository는 `Slavicmeme/Chemprompt`다.

repository에서 확인해야 할 기존 영역:
- analysis
- config
- data
- embeddings
- experiments
- feature_selection
- genetic
- models
- utils

특히 아래는 새로 추측해 작성하지 않는다.
- descriptor calculation
- prompt construction
- Aya embedding
- feature selection
- GA selection/flags
- model wrappers
- dataset handling
- split/evaluation logic

## Split 변경 작업

기존 코드에서 split/scaffold/seed 관련 구현을 검색한다.

기존 scaffold evaluation이 3개 seed를 사용하는 구조라면:
1. 기존 seed 목록 또는 seed 생성 규칙 확인
2. 기존 동작 보존
3. 동일 규칙으로 총 5개 seed가 되도록 확장
4. configuration으로 이동/명시
5. 5-fold split과 GA/일반 model 모두 공통 Split Service를 사용
6. regression/classification/scaffold 동작에 대한 test 작성

검색 결과가 없다고 바로 새 코드를 작성하지 말고 관련 experiment/notebook/config까지 확인한다.
