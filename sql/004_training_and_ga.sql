-- 004. Training jobs & GA experiments (docs/04, docs/06)

CREATE TABLE IF NOT EXISTS training_jobs (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    owner_id            BIGINT       REFERENCES users(id) ON DELETE CASCADE,
    dataset_version_id  BIGINT       NOT NULL REFERENCES dataset_versions(id) ON DELETE CASCADE,
    split_config_id     BIGINT       NOT NULL REFERENCES split_configs(id),
    property_id         BIGINT       REFERENCES properties(id) ON DELETE SET NULL,
    workflow_id         BIGINT,      -- FK는 007에서 추가
    task_type           VARCHAR(16)  NOT NULL CHECK (task_type IN ('regression', 'classification', 'multilabel')),
    descriptor_mode     VARCHAR(32)  NOT NULL CHECK (descriptor_mode IN ('FULL', 'FEATURE_SELECTION', 'GENETIC')),
    sklearn_model       VARCHAR(64)  NOT NULL,
    model_name          VARCHAR(200),
    config              JSONB        NOT NULL DEFAULT '{}'::jsonb,
    status              VARCHAR(16)  NOT NULL DEFAULT 'QUEUED'
                        CHECK (status IN ('QUEUED', 'RUNNING', 'SUCCEEDED', 'FAILED')),
    stage               VARCHAR(64),  -- 실제 수행 중인 backend 단계
    progress            JSONB        NOT NULL DEFAULT '{}'::jsonb,
    metrics             JSONB,        -- fold 평균/표준편차
    error               TEXT,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT now(),
    started_at          TIMESTAMPTZ,
    finished_at         TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS ix_training_jobs_owner ON training_jobs(owner_id);

CREATE TABLE IF NOT EXISTS training_fold_results (
    id                      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    training_job_id         BIGINT       NOT NULL REFERENCES training_jobs(id) ON DELETE CASCADE,
    fold_no                 SMALLINT     NOT NULL CHECK (fold_no BETWEEN 1 AND 5),
    split_seed              INTEGER      NOT NULL,
    n_train                 INTEGER      NOT NULL,
    n_valid                 INTEGER      NOT NULL,
    metrics                 JSONB        NOT NULL,
    selected_descriptors    JSONB        NOT NULL,     -- 이 fold의 prompt에 들어간 descriptor 이름
    representation_id       BIGINT       REFERENCES representation_artifacts(id) ON DELETE SET NULL,
    created_at              TIMESTAMPTZ  NOT NULL DEFAULT now(),
    UNIQUE (training_job_id, fold_no)
);

CREATE TABLE IF NOT EXISTS ga_experiments (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    training_job_id     BIGINT       NOT NULL UNIQUE REFERENCES training_jobs(id) ON DELETE CASCADE,
    provider            VARCHAR(16)  NOT NULL CHECK (provider IN ('MOCK', 'GENETIC')),
    candidate_k         INTEGER      NOT NULL DEFAULT 50,
    candidate_method    VARCHAR(32)  NOT NULL DEFAULT 'f_regression',
    config              JSONB        NOT NULL DEFAULT '{}'::jsonb,
    status              VARCHAR(16)  NOT NULL DEFAULT 'RUNNING'
                        CHECK (status IN ('RUNNING', 'SUCCEEDED', 'FAILED')),
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- GA flags는 fold마다 다를 수 있다. outer validation은 GA에 사용하지 않는다.
CREATE TABLE IF NOT EXISTS ga_fold_results (
    id                      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ga_experiment_id        BIGINT       NOT NULL REFERENCES ga_experiments(id) ON DELETE CASCADE,
    fold_no                 SMALLINT     NOT NULL CHECK (fold_no BETWEEN 1 AND 5),
    candidate_descriptors   JSONB        NOT NULL,   -- Top-K 후보 (fold train에서 fit)
    flags                   VARCHAR(256) NOT NULL,   -- K-bit flag (e.g. '0110...')
    full_flag               TEXT         NOT NULL,   -- 전체 filtered descriptor 기준 flag
    selected_descriptors    JSONB        NOT NULL,
    fitness                 JSONB,
    created_at              TIMESTAMPTZ  NOT NULL DEFAULT now(),
    UNIQUE (ga_experiment_id, fold_no)
);
