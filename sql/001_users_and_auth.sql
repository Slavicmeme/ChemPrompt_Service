-- 001. Users & Auth (docs/02_AUTH.md)
-- ownership은 내부 users.id를 사용한다. password는 Argon2id hash만 저장한다.

CREATE TABLE IF NOT EXISTS users (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    username        VARCHAR(64)  NOT NULL UNIQUE,
    password_hash   TEXT         NOT NULL,
    role            VARCHAR(16)  NOT NULL DEFAULT 'USER' CHECK (role IN ('USER', 'ADMIN')),
    is_active       BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- Opaque session token. 원문 token은 저장하지 않고 SHA-256 hash만 저장한다.
CREATE TABLE IF NOT EXISTS user_sessions (
    token_hash      CHAR(64)     PRIMARY KEY,
    user_id         BIGINT       NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    expires_at      TIMESTAMPTZ  NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_user_sessions_user ON user_sessions(user_id);
