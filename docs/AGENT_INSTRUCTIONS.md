# AGENT INSTRUCTIONS — MUST FOLLOW

1. 먼저 `./Chemprompt`를 조사한다.
2. 기존 scientific implementation이 있으면 재사용/adapter화한다.
3. GA와 일반 sklearn model 모두 공통 5-fold Split Service를 사용한다.
4. 각 fold는 대략 80% train / 20% validation이다.
5. RANDOM과 SCAFFOLD를 지원한다.
6. Scaffold 관련 기존 3-seed 구조가 있으면 기존 seed 규칙을 확인한 후 5-seed로 확장한다.
7. seed 값/규칙을 추측하지 않는다.
8. split configuration과 seed를 DB/model metadata에 저장한다.
9. outer validation을 GA/feature selection/model fitting에 누출하지 않는다.
10. Genetic의 descriptor/flags는 fold-specific일 수 있다.
11. V1 GA 계산은 mock 가능하지만 실제 GA replacement가 가능해야 한다.
12. V1 serving 기본은 5-fold ensemble이다.
13. Aya는 representation 전용이며 Chat Agent가 아니다.
14. Aya는 GPU server에서 분리 serving 가능해야 한다.
15. sklearn training/inference는 CPU/사무용 PC에서도 가능해야 한다.
16. Chat Agent는 OpenAI API를 사용한다.
17. LLM이 prediction 값을 생성하면 안 된다.
18. Workflow persistent state는 PostgreSQL이다.
19. LangGraph는 orchestration이다.
20. Langfuse는 observability이며 core prediction dependency가 아니다.
21. arbitrary model/dataset filesystem path를 허용하지 않는다.
22. Admin은 model validation/approval/serving 권한을 가진다.
23. JEV는 V1에서 구현하지 않는다.
