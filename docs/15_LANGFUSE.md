# 15. Langfuse

관찰 대상:
- trace/session
- internal user reference
- workflow id
- OpenAI generation
- prompt/version
- tool calls
- input/output metadata
- token usage
- latency
- error

Langfuse 장애가 scientific prediction/training correctness를 깨뜨리면 안 된다.

PostgreSQL = 완료 상태
LangGraph = 현재 실행 orchestration
Langfuse = 실행 관찰
