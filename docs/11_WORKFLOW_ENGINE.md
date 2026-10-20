# 11. Workflow Engine

PostgreSQL workflow state가 Source of Truth.

`current_step=N`이면 1..N이 모두 완료된 상태다.

N+1 이전에 N+2를 실행할 수 없다.

Step result 저장과 current_step transition은 atomic transaction으로 처리한다.

LangGraph는 orchestration이지 persistent workflow source of truth가 아니다.

Workflow Service가 허용하지 않는 step을 Agent가 요청하면 거부한다.
