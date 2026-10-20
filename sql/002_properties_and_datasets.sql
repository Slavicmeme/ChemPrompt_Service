-- 002. Properties & Datasets (docs/00, docs/06, docs/09)

CREATE TABLE IF NOT EXISTS datasets (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    owner_id        BIGINT       REFERENCES users(id) ON DELETE CASCADE,  -- NULL = built-in(system)
    name            VARCHAR(200) NOT NULL,
    description     TEXT,
    is_builtin      BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CHECK (is_builtin OR owner_id IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS ix_datasets_owner ON datasets(owner_id);
CREATE UNIQUE INDEX IF NOT EXISTS ux_datasets_builtin_name ON datasets(name) WHERE is_builtin;

CREATE TABLE IF NOT EXISTS dataset_versions (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id          BIGINT       NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    version             INTEGER      NOT NULL,
    storage_key         TEXT         NOT NULL,           -- Storage Service가 생성한 canonical object key
    raw_checksum        CHAR(64)     NOT NULL,           -- 업로드 원본 sha256
    checksum            CHAR(64),                        -- validation 후 정규화 데이터 sha256 (split/representation identity)
    original_filename   TEXT,
    columns             JSONB        NOT NULL DEFAULT '[]'::jsonb,
    smiles_column       TEXT,
    target_columns      JSONB,                           -- ["label"] or ["FDA_APPROVED","CT_TOX"]
    task_type           VARCHAR(16)  CHECK (task_type IN ('regression', 'classification', 'multilabel')),
    row_count           INTEGER,
    valid_row_count     INTEGER,
    status              VARCHAR(16)  NOT NULL DEFAULT 'UPLOADED'
                        CHECK (status IN ('UPLOADED', 'MAPPED', 'VALIDATED', 'INVALID')),
    validation_report   JSONB,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT now(),
    UNIQUE (dataset_id, version)
);

CREATE TABLE IF NOT EXISTS properties (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    code                VARCHAR(64)  NOT NULL UNIQUE,     -- FreeSolv, ESOL, Lipo, ClinTox
    name                VARCHAR(200) NOT NULL,
    description         TEXT,
    task_type           VARCHAR(16)  NOT NULL CHECK (task_type IN ('regression', 'classification', 'multilabel')),
    unit                VARCHAR(64),
    target_columns      JSONB        NOT NULL DEFAULT '[]'::jsonb,
    is_builtin          BOOLEAN      NOT NULL DEFAULT TRUE,
    dataset_id          BIGINT       REFERENCES datasets(id) ON DELETE SET NULL,
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT now()
);

INSERT INTO properties (code, name, description, task_type, unit, target_columns) VALUES
 ('FreeSolv', 'Hydration Free Energy (FreeSolv)', 'Experimental hydration free energy of small molecules in water.', 'regression', 'kcal/mol', '["label"]'),
 ('ESOL',     'Aqueous Solubility (ESOL)',        'Water solubility, log solubility in mols per litre.',             'regression', 'log(mol/L)', '["label"]'),
 ('Lipo',     'Lipophilicity (Lipo)',             'Octanol/water distribution coefficient (logD at pH 7.4).',        'regression', 'logD', '["label"]'),
 ('ClinTox',  'Clinical Toxicity (ClinTox)',      'FDA approval status and clinical-trial toxicity (two binary tasks).', 'multilabel', NULL, '["FDA_APPROVED","CT_TOX"]')
ON CONFLICT (code) DO NOTHING;
