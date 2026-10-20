-- 008. Agent conversations (docs/12)

CREATE TABLE IF NOT EXISTS agent_conversations (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id         BIGINT       NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title           VARCHAR(200),
    trace_id        VARCHAR(64),
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_agent_conversations_user ON agent_conversations(user_id);

CREATE TABLE IF NOT EXISTS agent_messages (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    conversation_id BIGINT       NOT NULL REFERENCES agent_conversations(id) ON DELETE CASCADE,
    role            VARCHAR(16)  NOT NULL CHECK (role IN ('user', 'assistant', 'tool')),
    content         TEXT,
    tool_calls      JSONB,       -- assistant의 tool call 요청
    tool_call_id    VARCHAR(128),
    tool_name       VARCHAR(64),
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_agent_messages_conv ON agent_messages(conversation_id, id);

ALTER TABLE predictions DROP CONSTRAINT IF EXISTS fk_predictions_conversation;
ALTER TABLE predictions ADD CONSTRAINT fk_predictions_conversation
    FOREIGN KEY (conversation_id) REFERENCES agent_conversations(id) ON DELETE SET NULL;
