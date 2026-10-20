# ChemPrompt 시스템 구조와 권한

- 서비스 내부 동작(구성, 예측·학습 흐름, 저장소, DB, workflow, Agent)과 사용자/운영자 권한을 정리한 문서
- 설치와 운영은 [README](../README.md), 사용법은 [USER_GUIDE.md](USER_GUIDE.md) 참고

## 목차
1. [시스템 로직](#1-시스템-로직)
2. [사용자 권한과 운영자 권한](#2-사용자-권한과-운영자-권한)

---

## 1. 시스템 로직

### 1.1 구성
```
브라우저 ──HTTP/SSE──▶ <WEB_HOST>:<WEB_PORT> ──포워딩──▶ nginx 컨테이너
                                                        ├─ 정적 화면 (frontend/)
                                                        └─ /api ──▶ api 컨테이너 (FastAPI, CPU)
                         api ├─ Auth / 예측 / Custom 학습(workflow) / Model Registry / Admin / Agent API
                             ├─ 학습 작업 실행기 (서버 프로세스 안의 스레드 2개)
                             ├─▶ db 컨테이너 (PostgreSQL, DB chemprompt) — 모든 상태의 기준
                             ├─▶ Artifact storage (<DATA_DIR>/storage) — 데이터셋, embedding, 모델 파일
                             ├─▶ aya 컨테이너 (내부 http://aya:8100, AYA_GPUS의 GPU마다 Aya 1개, 모델은 <DATA_DIR>/llm) — REPRESENTATION_MODE=http일 때
                             └─▶ OpenAI Responses API (Agent) / Langfuse (선택)
     init 컨테이너 (실행 때마다 1회): SQL 적용 → 관리자 생성 → built-in 모델 학습
```
- 과학 로직(descriptor, prompt, 분할, Top-50 선택, GA, sklearn 모델, 지표)은 모두 원본 ChemPrompt 코드(`backend/chemprompt_core`)를 그대로 불러 사용
- Aya는 embedding 전용이고 대화용 Agent와 별개, Agent는 OpenAI 모델 담당
- sklearn 학습/예측은 CPU에서 실행, GPU는 Aya 전용
- Aya는 `.env`의 `AYA_GPUS`(예: `0,1`)에 적은 GPU만 사용, GPU마다 모델을 하나씩 올려 요청 분산 처리, 모델 파일은 `<DATA_DIR>/llm`에 없으면 첫 시작 때 다운로드

### 1.2 예측 흐름
```
SMILES → RDKit 검증 → 모델에 저장된 descriptor 목록으로 descriptor 계산
       → Model k별 descriptor 선택(flag)으로 prompt 생성 "SMILES: ... | MolWt: ... | ..."
       → representation (Aya 4,096차원 또는 mock 256차원)
       → Model 1~5 (sklearn) 각각 예측
       → 회귀: 평균 / 분류: 확률 평균 후 0.5 기준 판정
       → 예측 기록 저장 (모델 버전, 분할 방식, seed, descriptor 방식, representation, prompt 버전)
```
- descriptor 선택이 같은 모델들은 embedding 1회만 계산 (FULL이면 1회)

### 1.3 학습 흐름
| 단계 | 내용 | workflow 단계 |
|---|---|---|
| ① 데이터 | 검증된 데이터셋 로드 (SMILES + target) | 1~4 |
| ② descriptor 필터 | 데이터셋 전체 SMILES로 RDKit descriptor 필터 (std > 1 또는 고유값 > 10, 라벨 미사용, 원본 규칙) | 7 |
| ③ 분할 | 공통 Split Service로 Model 1~5용 80/20 분할 | 6에서 생성 |
| ④ descriptor 선택 | FULL = 전체, FEATURE_SELECTION = 학습용 80%에서 f_regression Top-50, GENETIC = Top-50 중 GA 선택 | 7 |
| ⑤ representation | 전체 데이터의 prompt → embedding, 설정이 같으면 캐시 재사용 | 7 |
| ⑥ 학습 | Model k를 자기 학습용 80%로 fit (Ridge / LogisticRegression / MultiOutputClassifier) | 8 |
| ⑦ 평가 | Model k를 자기 검증용 20%로 평가, 원래 단위 | 9 |
| ⑧ 등록 | 파일 저장 → metadata.json → 자동 검증 → READY | 10 |
- 검증용 20%는 descriptor 선택, GA, 학습에 절대 미사용 (leakage 방지)
- 각 단계 사이와 embedding 묶음 사이마다 중단 요청 확인

### 1.4 데이터 분할과 seed
- RANDOM: KFold(분류는 StratifiedKFold), seed 42, Model 1~5의 검증 데이터 중복 없음
- SCAFFOLD: 원본 규칙 `42 + i`(기존 3개: 42, 43, 44)를 5개로 확장한 **[42, 43, 44, 45, 46]**, 모델마다 독립적인 scaffold-balanced 80/20 분할 (같은 골격이 학습과 검증에 동시에 포함되지 않음)
- 분할 설정, seed, 분할 결과(+checksum)는 `split_configs`, `split_fold_assignments`, 모델 `metadata.json`, 예측 기록에 저장
- 같은 데이터셋 + 같은 설정이면 같은 분할 재현

### 1.5 Representation 캐시
- 캐시 키: 데이터셋 checksum + representation 모델 + prompt 버전 + descriptor 방식 + descriptor 선택(flag) + 전처리 버전
- mock과 Aya는 키가 달라 혼용 없음

### 1.6 모델 레지스트리와 상태
```
UPLOADED → VALIDATING → READY → SERVING → ARCHIVED
                      ↘ INVALID
```
- 자동 검증: checksum, 5개 모델 로드, descriptor 복원, representation 일치, 시험 예측(CCO)
- READY: 본인(및 Admin)만 예측에 사용
- SERVING: Admin 승인, 모든 사용자가 사용 (같은 논리 모델 안에서 SERVING은 1개, 새로 승인하면 기존 SERVING은 READY로 강등)
- ARCHIVED: 예측 미사용

### 1.7 저장소 구조 (`<DATA_DIR>/storage`)
```
builtin/
  datasets/{dataset_id}/{version}/raw.csv, data.csv
  {property}/models/{model_id}/{version}/
      metadata.json, descriptor_catalog.json, split.json
      fold_1/ model.joblib, selected_descriptors.json, (GA: ga_flags.json)
      ... fold_5/
  representations/{cache_key}/embeddings.npy, meta.json
users/{user_id}/
  datasets/..., representations/..., models/{model_id}/{version}/...
```
- 경로는 서버가 결정, 사용자/Admin의 파일 경로 입력 기능 없음
- DB에는 경로(key), checksum, 소유자, fold 번호 기록, 불러올 때 checksum 확인

### 1.8 DB (`sql/001`~`010`, 순번 적용, 재실행 안전)
| 파일 | 내용 |
|---|---|
| 001 | users, user_sessions |
| 002 | datasets, dataset_versions, properties (+ built-in 4개) |
| 003 | split_configs, split_fold_assignments, representation_artifacts |
| 004 | training_jobs, training_fold_results, ga_experiments, ga_fold_results |
| 005 | models, model_versions, model_artifacts |
| 006 | predictions, prediction_fold_results |
| 007 | workflow_definitions, workflow_step_definitions, workflows, workflow_steps (+ Custom 학습 10단계 정의) |
| 008 | agent_conversations, agent_messages |
| 009 | properties 국문 이름, target 설명 |
| 010 | 학습 중단 상태(CANCELLED) |

### 1.9 Workflow 엔진
- PostgreSQL의 workflow 상태가 기준, `current_step = N`이면 1~N단계 완료 상태
- N+1단계만 실행 가능, 건너뛰면 409로 거부 (Agent도 동일)
- 단계 결과 저장과 `current_step` 변경은 한 transaction으로 처리 (workflow 행 잠금)
- 진행 상황 화면은 DB에 기록된 실제 단계만 SSE로 전달 (가짜 진행률 없음)

### 1.10 Agent
- LangGraph: `agent`(OpenAI 호출, 스트리밍) ⇄ `tools`(tool 실행) 반복, 최대 6회
- OpenAI Responses API 사용, reasoning 모델은 `OPENAI_REASONING_EFFORT` 적용
- tool: `list_properties`, `get_property_info`, `predict_property`, `get_prediction_result`, `get_prediction_history`, `get_training_status`, `get_workflow_status`
- tool은 요청한 사용자 권한으로 실행 (다른 사용자 데이터 접근 불가)
- 대화와 tool 결과는 DB(`agent_messages`)에 저장
- Langfuse는 key가 있을 때만 활성, 장애 시에도 예측/학습에 영향 없음

### 1.11 과학 로직 재사용 (`backend/chemprompt_core`)
- 원본 ChemPrompt 연구 코드(`Slavicmeme/Chemprompt`, MIT License, Copyright (c) 2026 APC Lab) 중 서비스가 쓰는 파일만 옮긴 패키지
- 과학 로직은 원본 그대로, import 경로만 변경, 라이선스는 `backend/chemprompt_core/LICENSE`

| 모듈 | 원본 파일 | 쓰는 곳 |
|---|---|---|
| `llm_model.py` | `embeddings/full_descriptors_embedding.py` | descriptor 필터/계산, prompt 생성 (weight 없이 사용), Aya embedding (`_embed_prompt`) |
| `custom_model.py` | `config/custom_model.py` | 모델 이름 해석 (llm_model 내부) |
| `data_splitter.py` | `utils/data_splitter.py` | 분할 (kfold / chemprop scaffold_balanced) |
| `metrics.py` | `utils/metrics.py` | 평가 지표 |
| `descriptor_selection.py` | `feature_selection/descriptor_selection.py` | Top-50 (f_regression) |
| `base_model.py`, `sklearn_model.py` | `models/` | Ridge / LogisticRegression / MultiOutputClassifier |
| `genetic.py` | `genetic/genetic.py` | GA (`GA_MODE=genetic`, 기본), 개발·테스트용 mock flag도 지원 |
| `data_loader.py` | `data/data_loader.py` | built-in 데이터 (FreeSolv는 `SAMPL.csv` 실험값 사용) |

### 1.12 파일 구성
```
bin/                     compose.sh(Docker 실행), docker-entrypoint.sh·chemprompt.sh(컨테이너 내부)
config/                  docker-compose.yml, environment.yml(conda env, 검증된 버전 고정)
sql/                     001~010 PostgreSQL DDL
backend/app/             FastAPI (routers/, services/, agent/, models.py, config.py)
backend/chemprompt_core/ 원본 ChemPrompt 핵심 모듈 (MIT)
backend/scripts/         apply_sql, create_admin, bootstrap_builtin, reset_builtin, reset_user_data, docker_init
backend/tests/           pytest 19개
aya_service/             GPU Aya Representation 서비스
frontend/                정적 화면 (vendor/: marked, DOMPurify)
sample/hppb.csv          Custom 학습 예시 데이터
docs/                    원본 명세(00~18), USER_GUIDE.md, SYSTEM.md
HANDOVER.md              개발 인수인계 문서
docker/Dockerfile        micromamba + config/environment.yml 이미지 (api, init, aya 공용)
nginx/nginx.conf         Docker의 nginx 설정
```

### 1.13 개발 기준 (원본 명세 요약)
- 과학 로직은 원본 `Slavicmeme/Chemprompt` 코드를 재사용(`backend/chemprompt_core`), 임의 재작성 금지
- 스택: FastAPI + PostgreSQL + SQLAlchemy 2.x (스키마는 `sql/*.sql`), Static HTML/CSS/JS, OpenAI Agent + LangGraph, Langfuse, SSE
- Aya는 representation 전용, GPU 서버에서 분리 serving, sklearn은 CPU에서도 실행 가능
- Full / Feature Selection Top-50 / Genetic 지원, Genetic V1은 mock 가능하되 실제 GA로 교체 가능한 구조
- Built-in: FreeSolv, ESOL, Lipo, ClinTox, Custom Dataset Training 지원
- 모든 학습(일반/GA)은 같은 5-fold 분할 규칙 사용, serving 기본은 5-fold 앙상블 (화면에서는 Model 1~5로 표기)

---

## 2. 사용자 권한과 운영자 권한

### 2.1 역할
- **비로그인**: 로그인/회원가입 화면만 접근 가능
- **USER**: 회원가입 시 자동으로 USER
- **ADMIN**: 웹에서 가입·승격 불가, 서버에서 `bin/compose.sh exec api chemprompt scripts.create_admin <이름>`으로만 생성·승격
- 권한은 모든 요청마다 서버에서 확인 (화면의 버튼 숨김과 별개로 API가 403/404로 차단)

### 2.2 기능별 권한
| 기능 | USER | ADMIN |
|---|---|---|
| built-in 모델(SERVING)로 예측 | O | O |
| 내 Custom 모델(READY)로 예측 | O (본인 모델만) | O (모든 READY 모델) |
| Admin이 승인한 사용자 모델(SERVING)로 예측 | O | O |
| Agent 대화 | O (본인 대화만) | O (본인 대화만) |
| 예측 기록 보기 | 본인 기록만 | 본인 기록 + API로 다른 사용자 예측 상세 조회 |
| Custom 학습 (workflow 생성·진행) | O (본인 것만) | O |
| 학습 중단 | 본인 workflow | 모든 학습 작업 (built-in 포함, Admin 화면) |
| Workflow 삭제 | 본인 것, 진행 중이 아닐 때 | 본인 것 (API로는 다른 사용자 것도 가능, 화면 목록에는 본인 것만 표시) |
| 내 모델 삭제 | 본인 모델, SERVING이 아닐 때 | 본인 모델 |
| 모델 검증 / 승인·Serving / Archive | X | O |
| 모델 패키지 업로드 | X | O |
| built-in 데이터 등록 / built-in 학습 | X | O |
| 모든 모델·학습·데이터셋·사용자 목록 | X | O |
| built-in 모델 전체 삭제 (`scripts.reset_builtin`) | X | 서버 터미널에서만 |

### 2.3 모델 상태별 사용 가능자
| 상태 | 예측에 쓸 수 있는 사람 |
|---|---|
| READY | 모델 소유자, ADMIN |
| SERVING | 모든 로그인 사용자 |
| UPLOADED / VALIDATING / INVALID / ARCHIVED | 사용 불가 |

### 2.4 보안
- 비밀번호는 Argon2id 해시로만 저장
- 로그인 세션은 서버 DB에 저장되는 무작위 토큰, 브라우저에는 HttpOnly cookie로만 전달 (JavaScript에서 읽기 불가), 유효기간 24시간
- `SECRET_KEY`를 설정하면 세션 토큰 해시에 함께 사용
- OpenAI / Langfuse key는 서버 `.env`에만 보관, 브라우저로 미전달
- Agent 답변(Markdown)은 DOMPurify로 위험한 HTML/스크립트를 제거한 뒤 표시
- 사용자/Admin 모두 서버 파일 경로 지정 불가, 업로드 zip도 허용된 파일 이름만 허용
- 모델 파일은 불러올 때마다 checksum 확인
