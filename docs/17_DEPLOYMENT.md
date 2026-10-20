# 17. Deployment

Web stack:
- Nginx
- FastAPI
- PostgreSQL
- Docker Compose
- Artifact Storage

GPU stack:
- Aya Representation Service
- H100 host

환경 변수 예:
DATABASE_URL
SECRET_KEY
OPENAI_API_KEY
OPENAI_AGENT_MODEL
LANGFUSE_PUBLIC_KEY
LANGFUSE_SECRET_KEY
LANGFUSE_BASE_URL
AYA_MODEL_ID
REPRESENTATION_SERVICE_URL
MODEL_STORAGE_ROOT

.env는 commit하지 않는다.
