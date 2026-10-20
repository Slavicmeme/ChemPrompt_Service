# ChemPrompt Web Service

- 분자 구조(SMILES)를 넣으면 물성을 예측해 주는 웹 서비스임
- ChemPrompt 연구 코드(LLM 기반 분자 표현 + 5개 모델 앙상블)를 화학·생물·약학 연구자가 바로 쓸 수 있게 만든 것임
- 주요 기능: built-in 물성 예측(FreeSolv, ESOL, Lipo, ClinTox), 자연어 Agent, 내 CSV로 모델 학습(Custom 학습), 운영자 모델 관리
- 스택: FastAPI, PostgreSQL, SQLAlchemy, LangGraph + OpenAI, RDKit, scikit-learn, Aya(aya-expanse-8b) representation, Docker compose

## 문서
| 문서 | 내용 |
|---|---|
| README (이 문서) | 설치와 운영 (Docker) |
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | 기능별 사용법, 결과 해석 |
| [docs/SYSTEM.md](docs/SYSTEM.md) | 시스템 로직, 사용자/운영자 권한 |
| [docs/](docs/) `00`~`18` | 원본 구현 명세 |
| [HANDOVER.md](HANDOVER.md) | 개발 인수인계 |

## 폴더 구성
| 폴더 | 내용 |
|---|---|
| `bin/` | `compose.sh`(Docker 실행), `docker-entrypoint.sh`·`chemprompt.sh`(컨테이너 내부용) |
| `config/` | `docker-compose.yml`, `environment.yml`(이미지의 conda 패키지 버전) |
| `docker/` | `Dockerfile` |
| `nginx/` | nginx 설정 |
| `sql/` | DB 스키마 (`001`~`010`) |
| `backend/`, `frontend/`, `aya_service/` | 서버, 화면, Aya representation 서비스 |
| `backend/chemprompt_core/` | 원본 ChemPrompt 연구 코드(`Slavicmeme/Chemprompt`, MIT) 중 서비스가 쓰는 모듈만 옮긴 패키지 |
| `docs/`, `sample/` | 문서, 예시 데이터 |

## 목차
1. [설치](#1-설치)
2. [운영](#2-운영)
3. [문제 해결](#3-문제-해결)

---

## 1. 설치

### 1.1 필요한 것
- Linux(Ubuntu 기준), Docker, Docker compose v2
- NVIDIA GPU(GPU당 약 18GB 여유) + 호스트 NVIDIA 드라이버: `nvidia-smi`가 동작해야 함
- NVIDIA Container Toolkit: Docker 컨테이너가 GPU를 쓰게 해 줌, 설치는 [1.2 처음 실행](#12-처음-실행)의 1~7번
- Hugging Face token (`HF_TOKEN`): Aya 모델을 처음 내려받을 때 필요 (`<DATA_DIR>/llm`에 이미 있으면 불필요)
  - https://huggingface.co/CohereLabs/aya-expanse-8b 에서 로그인 후 라이선스 동의 → Settings → Access Tokens에서 Read token 발급
- 호스트에 conda나 PostgreSQL은 설치하지 않음, 모두 컨테이너 안에서 돎

### 1.2 처음 실행
- 아무것도 없는 서버에서 GPU 연결 → 모델 받기 → 실행까지 순서대로 함
- 1~7번(GPU 연결)은 서버에 한 번만 하면 됨, 관리자 권한(sudo) 필요

**GPU 연결 (NVIDIA Container Toolkit)**
```bash
# 0) 호스트 드라이버 확인 (GPU 목록과 메모리 사용량이 나와야 함)
nvidia-smi

# 1) NVIDIA 저장소 키 등록
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# 2) NVIDIA 저장소 등록
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 3) NVIDIA Container Toolkit 설치
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# 4) Docker에 NVIDIA runtime 등록
sudo nvidia-ctk runtime configure --runtime=docker

# 5) GPU 장치 목록(CDI) 생성
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# 6) Docker 재시작 (실행 중인 다른 컨테이너도 재시작되므로 작업 중인 것이 없을 때 함)
sudo systemctl restart docker

# 7) 확인: "GPU 0: ..."처럼 GPU 목록이 나오면 성공
docker run --rm --gpus device=0 nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi -L
```
- 0번 `nvidia-smi`에서 메모리가 비어 있는 GPU 번호를 기억해 둠 (10번의 `AYA_GPUS`에 씀)
- 7번 확인용 이미지는 `docker image rm nvidia/cuda:12.6.0-base-ubuntu22.04`로 지워도 됨

**Hugging Face token 준비**
- Aya 모델(`CohereLabs/aya-expanse-8b`)은 라이선스 동의가 필요한 모델이라 token이 있어야 받을 수 있음
- https://huggingface.co/CohereLabs/aya-expanse-8b 에서 로그인 후 라이선스 동의
- Hugging Face → Settings → Access Tokens에서 **Read** token 발급 (`hf_...`), 10번에서 `.env`의 `HF_TOKEN`에 넣음

**설치와 실행**
```bash
# 8) 받기
git clone <저장소 주소> && cd <프로젝트 폴더>

# 9) 설정 파일 만들기
cp .env.example .env

# 10) .env 편집
#     필수: DATA_DIR, WEB_HOST / WEB_PORT, POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_DB (1.4 참고)
#     GPU:  AYA_GPUS (0번에서 고른 GPU 번호, 여러 개면 0,1), HF_TOKEN (위에서 발급한 token)
#     선택: OPENAI_API_KEY (Agent), ADMIN_USERNAME / ADMIN_PASSWORD (관리자 자동 생성), SECRET_KEY (2.2 참고)

# 11) 이미지 빌드 (conda 환경 설치, 처음에는 오래 걸림)
bin/compose.sh build

# 12) Aya 모델 받기 + GPU에 올리기 (약 15GB를 <DATA_DIR>/llm에 받음, 이미 있으면 바로 올림) 
# "GET /health HTTP/1.1" 200 OK 문구가 출려되면 "Ctrl + c"를 입력하여 종료
bin/compose.sh up -d aya
bin/compose.sh logs -f aya
#     "Aya replica ready on cuda:N"이 AYA_GPUS의 GPU 수만큼 나오면 Ctrl+C (로그 보기만 종료, aya는 계속 돎)

# 13) 나머지 실행 (db → init → api → nginx), 이 명령으로 서버가 실행됨 (백그라운드)
bin/compose.sh up -d

# 14) init 진행 확인: built-in 모델 4개 학습 → "init: done"이 나오면 Ctrl+C
bin/compose.sh logs -f init

# 15) 관리자 만들기 (비밀번호 두 번 입력), 실행 중인 서비스에 계정만 추가함
bin/compose.sh exec api chemprompt scripts.create_admin admin

# 16) 확인
bin/compose.sh ps                                  # db·aya·api healthy, nginx Up, init Exited (0)
curl http://<WEB_HOST>:<WEB_PORT>/api/health       # "representation_mode":"http", "ga_mode":"genetic"
nvidia-smi                                         # AYA_GPUS의 GPU에 약 18GB 사용 중
```
- **따로 실행하는 단계는 없음**: 13번에서 이미 실행됐고, 14번에서 "init: done"이 나오면 바로 접속할 수 있음, 15번 후 재시작도 필요 없음
- 15번은 `.env`에 `ADMIN_PASSWORD`(8자 이상)를 넣어 두었으면 **하지 않아도 됨**, init이 13번에서 자동으로 만듦 (아이디는 `ADMIN_USERNAME`, 비우면 `admin`)
- 끝나면 `http://<WEB_HOST>:<WEB_PORT>`로 접속해 관리자 계정으로 로그인함, 일반 사용자는 화면의 **회원가입**으로 가입함
- 11~12번 대신 `bin/compose.sh up -d --build` 한 번으로 해도 같음 (빌드 → aya 모델 받기 → 나머지 실행, init은 aya 준비가 끝날 때까지 기다림)
- 모델을 받은 뒤에는 `HF_TOKEN`이 없어도 됨, 이후 실행은 `<DATA_DIR>/llm`의 파일을 그대로 씀
- 데이터 폴더(`<DATA_DIR>/storage`, `<DATA_DIR>/llm`)는 `bin/compose.sh`가 `.env`의 `DATA_DIR`을 보고 실행한 계정 소유로 만들어 줌, 직접 만들 필요 없음
- `bin/compose.sh`는 `docker compose`와 같고, compose 파일(`config/docker-compose.yml`)과 프로젝트 폴더를 대신 지정하고 `AYA_GPUS`의 GPU를 aya에 연결해 줌 (뒤에 붙는 인자는 `docker compose`와 동일)

### 1.3 삭제하고 다시 실행
```bash
cd <프로젝트 폴더>

# 1) 컨테이너·네트워크·직접 빌드한 이미지 삭제
bin/compose.sh down --rmi local

# 2) 데이터 삭제 (DB, 모델, 업로드 데이터, Aya 모델 모두, 되돌릴 수 없음)
#    postgres 폴더는 컨테이너 계정 소유라 컨테이너로 지움
docker run --rm -v "$(realpath <DATA_DIR>)":/d postgres:16 rm -rf /d/postgres /d/storage /d/llm
```
- 이후 [1.2 처음 실행](#12-처음-실행)의 11번(빌드)부터 다시 함 (GPU 연결과 `.env`는 남아 있음, 데이터 폴더는 자동으로 다시 만들어짐)
- Aya 모델(약 15GB)을 다시 받지 않으려면 2번에서 `/d/llm`을 빼고 지움
- 데이터는 남기고 컨테이너만 다시 띄우려면 1번 대신 `bin/compose.sh down`, 2번을 건너뛰고 `bin/compose.sh up -d --build`만 실행함 (계정과 모델이 그대로 이어짐)
- 받아 온 기반 이미지까지 지우려면 `docker image rm postgres:16 nginx:1.27-alpine`
- GPU 연결까지 지우고 1.2의 0번부터 다시 하려면 (관리자 권한 필요)
```bash
sudo apt-get purge -y nvidia-container-toolkit nvidia-container-toolkit-base libnvidia-container-tools libnvidia-container1
sudo rm -f /etc/cdi/nvidia.yaml /etc/apt/sources.list.d/nvidia-container-toolkit.list /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
sudo nano /etc/docker/daemon.json     # "runtimes"의 "nvidia" 항목 삭제 (다른 설정이 없으면 파일째 삭제)
sudo systemctl restart docker
```

### 1.4 `.env` 필수 항목
| 항목 | 설명 | 예시 |
|---|---|---|
| `DATA_DIR` | **데이터 저장 위치**, 호스트 폴더가 컨테이너에 연결됨 (`postgres/`, `storage/`, `llm/`이 생김) | `./data` |
| `WEB_HOST` / `WEB_PORT` | **외부 접속**, 이 컴퓨터의 IP와 포트를 nginx로 포워딩 (`0.0.0.0` = 모든 네트워크) | `0.0.0.0` / `8889` |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | **DB 계정**, compose 안의 PostgreSQL이 이 계정으로 만들어지고 api도 같은 값으로 접속함 | `chemprompt` / … / `chemprompt` |
- 그 밖의 설정은 [2.2 선택 설정](#22-선택-설정) 참고 (`.env`에 빈 값으로 들어 있음)

### 1.5 구성
```
호스트 WEB_HOST:WEB_PORT ──포워딩──▶ nginx :80 ──▶ api :8000 ──▶ db :5432
                                                    └──▶ aya :8100 (REPRESENTATION_MODE=http일 때, GPU)
                                  └────── compose 내부 네트워크 (외부 비공개) ──────┘
```
| 서비스 | 이미지 | 역할 |
|---|---|---|
| `db` | postgres:16 | PostgreSQL, 데이터는 `<DATA_DIR>/postgres` |
| `init` | chemprompt-service | 실행할 때마다 한 번 돌고 종료: SQL 적용 → 관리자 생성(`ADMIN_PASSWORD`가 있을 때) → SERVING 모델이 없는 built-in property 학습, 이미 된 것은 건너뜀 |
| `api` | chemprompt-service | FastAPI 서버, 모델·데이터 파일은 `<DATA_DIR>/storage` |
| `nginx` | nginx:1.27-alpine | 화면(`frontend/`) 서빙, `/api`를 api로 연결, 호스트 포트 공개 |
| `aya` | chemprompt-service | Aya representation 내부 서버 (`http://aya:8100`, 외부 비공개), `REPRESENTATION_MODE=http`이면 자동으로 함께 뜸, `AYA_GPUS`의 GPU마다 모델 하나씩 |
- `chemprompt-service` 이미지는 micromamba로 `config/environment.yml`을 설치한 이미지 하나이고 api, init, aya가 같이 씀 (`docker/Dockerfile`)
- 외부에 열리는 포트는 `WEB_PORT` 하나뿐이고, api와 db는 compose 내부 네트워크에서 컨테이너 이름으로만 통신함
- `.env`는 이미지에 들어가지 않음 (`.dockerignore`), compose가 필요한 값만 컨테이너에 전달함
- api/init 컨테이너는 `<DATA_DIR>/storage` 폴더의 소유자 계정으로 실행됨, 그래서 생기는 파일도 그 계정 소유임

### 1.6 데이터 저장 위치 (`DATA_DIR`)
```
<DATA_DIR>/
├── postgres/   DB 데이터 (PostgreSQL 컨테이너가 관리, 직접 수정하지 않음)
├── storage/    모델·데이터셋·embedding 파일
│   ├── builtin/              built-in 데이터셋과 모델
│   └── users/<user_id>/      사용자가 올린 데이터셋, 학습한 모델
└── llm/        Aya LLM 파일 (Hugging Face cache 구조, 처음 실행 때 자동으로 내려받음, 약 15GB)
```
- DB에는 파일 위치가 `storage/` 기준 상대 경로(예: `users/3/models/6/1/fold_1/model.joblib`)로 저장됨, 그래서 `DATA_DIR`을 다른 곳으로 옮겨도 `.env` 값만 바꾸면 그대로 씀
- 백업은 서비스를 멈춘 뒤(`bin/compose.sh down`) `DATA_DIR` 폴더를 통째로 복사하면 됨

---

## 2. 운영

### 2.1 운영 명령
```bash
bin/compose.sh ps                         # 상태
bin/compose.sh logs -f api                # 서버 로그 (Ctrl+C로 종료)
bin/compose.sh up -d                      # .env 변경 후 반영
bin/compose.sh up -d --build              # 코드 변경 후 다시 빌드·실행 (init이 다시 돌며 새 SQL 적용)
bin/compose.sh restart api                # 서버만 재시작
bin/compose.sh down                       # 중지 (DATA_DIR의 데이터는 그대로 남음)

# 관리 스크립트: api 컨테이너 안에서 chemprompt <모듈> (storage 소유자 계정으로 실행됨)
bin/compose.sh exec api chemprompt scripts.create_admin <이름>   # 관리자 생성 / 비밀번호 재설정
bin/compose.sh exec api chemprompt scripts.apply_sql             # SQL 적용 (init도 매번 적용함)
bin/compose.sh exec api chemprompt pytest -q -p no:cacheprovider tests   # 테스트
bin/compose.sh exec api chemprompt scripts.reset_user_data        # 사용자 데이터만 지우기 (확인 → --yes로 삭제, 계정과 built-in 모델은 남음)

bin/compose.sh build --no-cache && bin/compose.sh up -d          # 이미지만 처음부터 다시 빌드
```
- 전부 지우고 다시 설치하려면 [1.3 삭제하고 다시 실행](#13-삭제하고-다시-실행)
- 컨테이너는 `restart: unless-stopped`라 Docker 서비스가 켜지면(호스트 재부팅 후 포함) 자동으로 다시 뜸

### 2.2 선택 설정
- `.env`(와 `.env.example`)에 빈 값으로 들어 있음, 비워 두면 `config/docker-compose.yml`의 기본값을 씀
- 바꾸려면 `.env`에 값을 넣고 `bin/compose.sh up -d`

| 항목 | 설명 | 기본값 |
|---|---|---|
| `OPENAI_API_KEY` | Agent용 OpenAI key (비우면 Agent 꺼짐) | (빈 값) |
| `OPENAI_AGENT_MODEL` | Agent 모델 | `gpt-4o-mini` |
| `OPENAI_REASONING_EFFORT` | reasoning 모델용 (`none`/`minimal`/`low`/`medium`/`high`), 비우면 보내지 않음 | (빈 값) |
| `ADMIN_USERNAME` / `ADMIN_PASSWORD` | init이 자동으로 만들 관리자 (비우면 만들지 않음) | `admin` / (빈 값) |
| `SECRET_KEY` | 세션 토큰 해시용 비밀값 (비어 있으면 경고), 바꾸면 기존 로그인이 끊김 | (빈 값) |
| `REPRESENTATION_MODE` | `http`(Aya GPU, 실전) / `mock`(개발·테스트용 CPU embedding) | `http` |
| `GA_MODE` | `genetic`(원본 GA, 실전) / `mock`(개발·테스트용) | `genetic` |
| `BUILTIN_DESCRIPTOR_MODE` / `BUILTIN_SPLIT` | init의 built-in 모델 학습 설정 | `FULL` / `RANDOM` |
| `TRAINING_WORKERS` | 동시에 돌 수 있는 학습 수 | `2` |
| `AYA_MODEL_ID` | Aya 모델 | `CohereLabs/aya-expanse-8b` |
| `AYA_GPUS` | Aya가 쓸 GPU 번호, 여러 개면 쉼표 (예: `0,1`), GPU마다 모델을 하나씩 올려 나눠 처리함 | `0` |
| `HF_TOKEN` | Hugging Face token, Aya 모델을 처음 내려받을 때 필요 (라이선스 동의가 필요한 모델) | (빈 값) |
| `AYA_DTYPE`, `HF_HUB_OFFLINE` | Aya dtype, 오프라인 여부 (`0` = 모델이 없으면 내려받음) | `float16`, `0` |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL` | tracing (비우면 꺼짐) | (빈 값) |

### 2.3 Built-in 모델 학습
- 첫 실행 때 init이 4개(FreeSolv, ESOL, Lipo, ClinTox)를 `BUILTIN_DESCRIPTOR_MODE` / `BUILTIN_SPLIT`으로 학습해 SERVING으로 올림
- 다른 설정으로 직접 학습하려면
```bash
# 4개를 한 번에
bin/compose.sh exec api chemprompt scripts.bootstrap_builtin --properties FreeSolv ESOL Lipo ClinTox --mode FULL --split RANDOM --serve

# 하나만
bin/compose.sh exec api chemprompt scripts.bootstrap_builtin --properties ESOL --mode GENETIC --split SCAFFOLD --serve

# 기존 built-in 모델을 지우고 다시 학습
bin/compose.sh exec api chemprompt scripts.reset_builtin          # 지워질 내용 확인 (아무것도 지우지 않음)
bin/compose.sh exec api chemprompt scripts.reset_builtin --yes    # 실제 삭제: built-in 모델, 그 모델로 한 예측 기록, 학습 기록, 데이터셋
```
- `--mode`: `FULL` / `FEATURE_SELECTION` / `GENETIC`, `--split`: `RANDOM` / `SCAFFOLD`
- `--serve`를 빼면 READY로만 등록됨, Admin 화면에서 "승인·Serving"을 누르면 SERVING이 됨
- Admin 화면의 "Built-in 데이터 · 학습"에서도 학습할 수 있음
- 같은 property에 SERVING 모델이 여러 개면 예측은 가장 최근에 SERVING된 모델을 씀, 하나만 남기려면 나머지를 Archive함
- GENETIC 모드는 원본 GA(50세대 × 개체 40)를 실제 Aya embedding으로 돌리므로 property 하나에 수 시간~수 일이 걸릴 수 있음 (빠른 확인이 필요하면 `GA_MODE=mock`)

### 2.4 Aya(GPU) 사용
- Aya는 compose 안의 내부 서버(`aya`)로 돌고, api·init이 `http://aya:8100`으로 연결함 (외부에 열리지 않음)
- 호스트에 NVIDIA Container Toolkit이 필요함, 설치는 [1.2 처음 실행](#12-처음-실행)의 GPU 연결, 없으면 `aya` 컨테이너가 시작되지 않음
- `.env` 설정
  ```
  REPRESENTATION_MODE=http
  AYA_GPUS=0          # 쓸 GPU 번호, 여러 개면 0,1 처럼 쉼표로
  HF_TOKEN=<Hugging Face token>   # 모델을 처음 내려받을 때만 필요
  ```
- 실행은 똑같이 `bin/compose.sh up -d --build`, `REPRESENTATION_MODE=http`면 `bin/compose.sh`가 aya를 자동으로 함께 띄우고 `AYA_GPUS`의 GPU를 연결함
- 모델 파일은 `<DATA_DIR>/llm`에 저장됨, 없으면 aya가 처음 시작할 때 Hugging Face에서 내려받음 (약 15GB, init은 aya 준비가 끝난 뒤 시작함)
- GPU마다 모델을 하나씩 올림 (GPU당 약 18GB, float16), 요청은 GPU 수만큼 나눠 동시에 처리함
- built-in 모델은 지금 representation과 맞는 SERVING 모델이 없으면 init이 다시 학습함 (예: mock으로 학습한 모델만 있으면 Aya로 재학습)
- `REPRESENTATION_MODE=mock`은 GPU 없이 화면·흐름을 확인하는 개발용이고, 이때는 aya 컨테이너를 띄우지 않음

---

## 3. 문제 해결

| 증상 / 질문 | 원인과 해결 |
|---|---|
| 관리자로 로그인이 안 됨 | 관리자 계정이 없을 수 있음, `bin/compose.sh exec api chemprompt scripts.create_admin admin`으로 만들거나 비밀번호를 다시 설정함 |
| 관리자 비밀번호를 잊음 | 위와 같은 명령 실행 후 새 비밀번호를 두 번 입력함 |
| 운영자 화면은 어디? | 같은 홈페이지에서 ADMIN 계정으로 로그인 → 상단 **Admin** 탭 |
| `bin/compose.sh up`에서 `set POSTGRES_USER in .env` 오류 | `.env`의 필수 항목이 비어 있음, [1.4](#14-env-필수-항목) 참고 |
| 접속이 안 됨 | `bin/compose.sh ps`로 상태 확인, init이 아직 학습 중이면 `bin/compose.sh logs -f init`, api가 unhealthy면 `bin/compose.sh logs api` |
| 포트가 이미 사용 중 | 다른 프로그램이 쓰고 있음, `.env`의 `WEB_PORT`를 바꿈, 확인: `ss -ltnp \| grep :<포트>` |
| `DATA_DIR`의 파일을 지우거나 옮길 수 없음 | `postgres/`는 DB 컨테이너 계정 소유임, 지울 때는 [1.3](#13-삭제하고-다시-실행)의 2번처럼 컨테이너로 지움 (`storage/`, `llm/`은 `bin/compose.sh`가 내 계정 소유로 만듦) |
| Agent `Error code: 400 ... reasoning_effort ...` | 모델에 맞게 `OPENAI_REASONING_EFFORT`를 설정하고 `bin/compose.sh up -d` |
| 예측 실패 "representation mismatch" | 모델을 학습할 때와 지금의 representation(Aya / mock)이 다름, built-in은 `bin/compose.sh up -d`로 init이 다시 학습함, Custom 모델은 다시 학습함 |
| `aya` 컨테이너가 안 뜸 (`no known GPU vendor`, `could not select device driver`) | 호스트에 NVIDIA Container Toolkit이 없음, [1.2](#12-처음-실행)의 1~7번 후 다시 실행함 |
| aya 로그 `PermissionError at /data/llm/hub` | `<DATA_DIR>/storage`와 `<DATA_DIR>/llm`의 소유자가 다름 (`bin/compose.sh` 대신 `docker compose`로 직접 띄워 Docker가 root로 만든 경우 등), `bin/compose.sh up -d --build`로 다시 띄우면 llm 폴더를 storage 소유자에게 맞춤 |
| `nvidia-ctk: command not found` | NVIDIA Container Toolkit이 설치되지 않음, [1.2](#12-처음-실행)의 1~3번부터 다시 함 |
| `nvidia-smi`가 안 됨 | 호스트 NVIDIA 드라이버 문제임, 드라이버를 먼저 설치함 (Docker와 무관) |
| 빌드 중 `failed to receive status ... EOF` | 빌드 도중 Docker가 재시작됨, 같은 `bin/compose.sh up -d --build`를 다시 실행함 (끝난 단계는 캐시로 건너뜀) |
| aya 로그 `CUDA out of memory` | 그 GPU를 다른 작업이 쓰고 있음, `nvidia-smi`로 비어 있는 GPU를 골라 `AYA_GPUS`를 바꾸고 `bin/compose.sh up -d` |
| init이 오래 멈춘 것 같음 | aya가 모델을 받거나 올리는 중임, `bin/compose.sh logs -f aya`로 확인함 (init은 aya가 준비된 뒤 시작) |
| aya가 모델을 못 받음 (401/403, gated repo) | `.env`의 `HF_TOKEN`이 비었거나 Hugging Face에서 Aya 라이선스에 동의하지 않은 계정임 |
| `AYA_GPUS must be GPU numbers` | `AYA_GPUS`는 `0` 또는 `0,1`처럼 숫자와 쉼표로만 씀 |
| 서버 로그 `SECRET_KEY is empty` | `.env`의 `SECRET_KEY`에 임의의 긴 문자열을 넣고 `bin/compose.sh up -d` (기존 로그인은 끊김) |
| 화면이 예전 그대로임 | 브라우저 캐시임, Ctrl+F5로 새로고침 |
