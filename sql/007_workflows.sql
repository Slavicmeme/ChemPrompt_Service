-- 007. Workflow engine (docs/11)
-- PostgreSQL workflow state가 Source of Truth. current_step=N 이면 1..N 완료.

CREATE TABLE IF NOT EXISTS workflow_definitions (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    code            VARCHAR(64)  NOT NULL UNIQUE,
    name            VARCHAR(200) NOT NULL,
    description     TEXT
);

CREATE TABLE IF NOT EXISTS workflow_step_definitions (
    id                      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    workflow_definition_id  BIGINT       NOT NULL REFERENCES workflow_definitions(id) ON DELETE CASCADE,
    step_no                 SMALLINT     NOT NULL,
    code                    VARCHAR(64)  NOT NULL,
    name                    VARCHAR(200) NOT NULL,
    executor                VARCHAR(16)  NOT NULL CHECK (executor IN ('USER', 'SYSTEM')),
    UNIQUE (workflow_definition_id, step_no),
    UNIQUE (workflow_definition_id, code)
);

CREATE TABLE IF NOT EXISTS workflows (
    id                      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    owner_id                BIGINT       NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    workflow_definition_id  BIGINT       NOT NULL REFERENCES workflow_definitions(id),
    current_step            SMALLINT     NOT NULL DEFAULT 0,
    status                  VARCHAR(16)  NOT NULL DEFAULT 'ACTIVE'
                            CHECK (status IN ('ACTIVE', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED')),
    context                 JSONB        NOT NULL DEFAULT '{}'::jsonb,
    error                   TEXT,
    created_at              TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_workflows_owner ON workflows(owner_id);

CREATE TABLE IF NOT EXISTS workflow_steps (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    workflow_id     BIGINT       NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
    step_no         SMALLINT     NOT NULL,
    code            VARCHAR(64)  NOT NULL,
    status          VARCHAR(16)  NOT NULL CHECK (status IN ('COMPLETED', 'FAILED')),
    input           JSONB,
    result          JSONB,
    error           TEXT,
    completed_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
    UNIQUE (workflow_id, step_no)
);

ALTER TABLE training_jobs DROP CONSTRAINT IF EXISTS fk_training_jobs_workflow;
ALTER TABLE training_jobs ADD CONSTRAINT fk_training_jobs_workflow
    FOREIGN KEY (workflow_id) REFERENCES workflows(id) ON DELETE SET NULL;

-- Custom Dataset Training workflow (docs/06 Workflow 1..10)
INSERT INTO workflow_definitions (code, name, description) VALUES
 ('CUSTOM_TRAINING', 'Custom Dataset Training', 'CSV 업로드부터 5-fold 학습과 모델 등록까지')
ON CONFLICT (code) DO NOTHING;

INSERT INTO workflow_step_definitions (workflow_definition_id, step_no, code, name, executor)
SELECT d.id, s.step_no, s.code, s.name, s.executor
FROM workflow_definitions d
CROSS JOIN (VALUES
  (1,  'DATASET_UPLOAD',     'Dataset Upload',          'USER'),
  (2,  'COLUMN_MAPPING',     'Column Mapping',          'USER'),
  (3,  'VALIDATION',         'Validation',              'USER'),
  (4,  'TASK_CONFIGURATION', 'Task Configuration',      'USER'),
  (5,  'DESCRIPTOR_MODE',    'Descriptor Mode',         'USER'),
  (6,  'SPLIT_STRATEGY',     'Split Strategy',          'USER'),
  (7,  'REPRESENTATION',     'Representation 생성/조회', 'SYSTEM'),
  (8,  'TRAINING',           '5-Fold Training',         'SYSTEM'),
  (9,  'EVALUATION',         'Evaluation',              'SYSTEM'),
  (10, 'MODEL_REGISTRATION', 'Model Registration',      'SYSTEM')
) AS s(step_no, code, name, executor)
WHERE d.code = 'CUSTOM_TRAINING'
ON CONFLICT (workflow_definition_id, step_no) DO NOTHING;
