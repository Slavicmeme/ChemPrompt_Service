-- 003. Split configs & Representation artifacts (docs/03, docs/07, docs/10)

-- 동일 dataset_version + config(checksum) => 동일 split 재현.
CREATE TABLE IF NOT EXISTS split_configs (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_version_id  BIGINT       NOT NULL REFERENCES dataset_versions(id) ON DELETE CASCADE,
    strategy            VARCHAR(16)  NOT NULL CHECK (strategy IN ('RANDOM', 'SCAFFOLD')),
    n_folds             INTEGER      NOT NULL DEFAULT 5 CHECK (n_folds = 5),
    seed                INTEGER      NOT NULL,              -- RANDOM: KFold seed / SCAFFOLD: 첫 번째 seed
    seeds               JSONB        NOT NULL,              -- 실제 사용한 seed list (SCAFFOLD: fold별 seed 5개)
    config              JSONB        NOT NULL DEFAULT '{}'::jsonb, -- splitter/method/task_type/version 등
    checksum            CHAR(64)     NOT NULL,              -- config identity checksum
    assignment_checksum CHAR(64),                           -- fold assignment 전체의 checksum
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT now(),
    UNIQUE (dataset_version_id, checksum)
);

-- Fold assignment 저장 (row_index = validated dataset의 0-based row 위치)
CREATE TABLE IF NOT EXISTS split_fold_assignments (
    split_config_id     BIGINT       NOT NULL REFERENCES split_configs(id) ON DELETE CASCADE,
    fold_no             SMALLINT     NOT NULL CHECK (fold_no BETWEEN 1 AND 5),
    row_index           INTEGER      NOT NULL,
    partition           CHAR(5)      NOT NULL CHECK (partition IN ('TRAIN', 'VALID')),
    PRIMARY KEY (split_config_id, fold_no, row_index)
);

-- Representation cache identity: dataset checksum + Aya model + prompt version
--   + descriptor mode + descriptor config(fold/GA flags) + preprocessing version
CREATE TABLE IF NOT EXISTS representation_artifacts (
    id                      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    owner_id                BIGINT       REFERENCES users(id) ON DELETE CASCADE,
    dataset_version_id      BIGINT       REFERENCES dataset_versions(id) ON DELETE CASCADE,
    cache_key               CHAR(64)     NOT NULL UNIQUE,
    representation_model    TEXT         NOT NULL,
    representation_backend  VARCHAR(16)  NOT NULL CHECK (representation_backend IN ('aya', 'mock')),
    prompt_version          VARCHAR(32)  NOT NULL,
    descriptor_mode         VARCHAR(32)  NOT NULL CHECK (descriptor_mode IN ('FULL', 'FEATURE_SELECTION', 'GENETIC')),
    descriptor_config_hash  CHAR(64)     NOT NULL,
    preprocessing_version   VARCHAR(32)  NOT NULL,
    storage_key             TEXT         NOT NULL,
    checksum                CHAR(64)     NOT NULL,
    n_rows                  INTEGER      NOT NULL,
    dim                     INTEGER      NOT NULL,
    created_at              TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_repr_dataset_version ON representation_artifacts(dataset_version_id);
