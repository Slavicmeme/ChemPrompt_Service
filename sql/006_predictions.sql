-- 006. Predictions (docs/05)
-- LLM(Agent)은 prediction 값을 생성하지 않는다. 값은 항상 fold sklearn model 결과의 aggregation이다.

CREATE TABLE IF NOT EXISTS predictions (
    id                      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id                 BIGINT       NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    model_version_id        BIGINT       NOT NULL REFERENCES model_versions(id),
    smiles                  TEXT         NOT NULL,
    status                  VARCHAR(16)  NOT NULL CHECK (status IN ('SUCCEEDED', 'FAILED')),
    task_type               VARCHAR(16)  NOT NULL,
    result                  JSONB,       -- {"value":..} / {"tasks":[{"name","probability","label"}]}
    error                   TEXT,
    -- reproducibility metadata (docs/05 최소 항목)
    model_version_label     TEXT         NOT NULL,
    split_strategy          VARCHAR(16)  NOT NULL,
    n_folds                 INTEGER      NOT NULL,
    split_config_id         BIGINT       REFERENCES split_configs(id) ON DELETE SET NULL,
    split_seeds             JSONB        NOT NULL,
    descriptor_mode         VARCHAR(32)  NOT NULL,
    representation_model    TEXT         NOT NULL,
    prompt_version          VARCHAR(32)  NOT NULL,
    aggregation             VARCHAR(64)  NOT NULL,
    source                  VARCHAR(16)  NOT NULL DEFAULT 'WEB' CHECK (source IN ('WEB', 'AGENT', 'API')),
    conversation_id         BIGINT,
    created_at              TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_predictions_user_created ON predictions(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS prediction_fold_results (
    prediction_id       BIGINT       NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    fold_no             SMALLINT     NOT NULL CHECK (fold_no BETWEEN 1 AND 5),
    output              JSONB        NOT NULL,
    PRIMARY KEY (prediction_id, fold_no)
);
