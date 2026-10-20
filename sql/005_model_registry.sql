-- 005. Model registry (docs/08, docs/09, docs/13)
-- State: UPLOADED -> VALIDATING -> READY -> SERVING -> ARCHIVED  (Upload != Serving)

CREATE TABLE IF NOT EXISTS models (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    owner_id        BIGINT       REFERENCES users(id) ON DELETE CASCADE,  -- NULL = built-in
    property_id     BIGINT       REFERENCES properties(id) ON DELETE SET NULL,
    scope           VARCHAR(16)  NOT NULL CHECK (scope IN ('BUILTIN', 'USER')),
    name            VARCHAR(200) NOT NULL,
    description     TEXT,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_versions (
    id                      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_id                BIGINT       NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    version                 INTEGER      NOT NULL,
    training_job_id         BIGINT       REFERENCES training_jobs(id) ON DELETE SET NULL,
    split_config_id         BIGINT       REFERENCES split_configs(id) ON DELETE SET NULL,
    status                  VARCHAR(16)  NOT NULL DEFAULT 'UPLOADED'
                            CHECK (status IN ('UPLOADED', 'VALIDATING', 'READY', 'SERVING', 'ARCHIVED', 'INVALID')),
    task_type               VARCHAR(16)  NOT NULL,
    target_columns          JSONB        NOT NULL DEFAULT '[]'::jsonb,
    descriptor_mode         VARCHAR(32)  NOT NULL,
    split_strategy          VARCHAR(16)  NOT NULL,
    n_folds                 INTEGER      NOT NULL DEFAULT 5,
    split_seeds             JSONB        NOT NULL,
    representation_model    TEXT         NOT NULL,
    prompt_version          VARCHAR(32)  NOT NULL,
    sklearn_model           VARCHAR(64)  NOT NULL,
    sklearn_version         VARCHAR(32),
    storage_prefix          TEXT         NOT NULL,
    metadata                JSONB        NOT NULL,     -- metadata.json 내용
    metrics                 JSONB,
    validation_report       JSONB,
    reviewed_by             BIGINT       REFERENCES users(id) ON DELETE SET NULL,
    reviewed_at             TIMESTAMPTZ,
    review_note             TEXT,
    created_at              TIMESTAMPTZ  NOT NULL DEFAULT now(),
    UNIQUE (model_id, version)
);
-- 한 논리 모델에서 SERVING 버전은 최대 1개
CREATE UNIQUE INDEX IF NOT EXISTS ux_model_versions_serving ON model_versions(model_id) WHERE status = 'SERVING';

CREATE TABLE IF NOT EXISTS model_artifacts (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_version_id    BIGINT       NOT NULL REFERENCES model_versions(id) ON DELETE CASCADE,
    artifact_type       VARCHAR(32)  NOT NULL CHECK (artifact_type IN (
                            'METADATA', 'DESCRIPTOR_CATALOG', 'FOLD_MODEL', 'SELECTED_DESCRIPTORS',
                            'GA_FLAGS', 'SPLIT', 'PREPROCESSING', 'FULL_FIT')),
    fold_no             SMALLINT     CHECK (fold_no BETWEEN 1 AND 5),
    storage_key         TEXT         NOT NULL,
    checksum            CHAR(64)     NOT NULL,
    size_bytes          BIGINT       NOT NULL,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT now(),
    UNIQUE (model_version_id, storage_key)
);
