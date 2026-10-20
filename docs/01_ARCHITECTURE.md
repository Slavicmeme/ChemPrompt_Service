# 01. Architecture

```text
Browser
  |
Nginx
  |
FastAPI
  |-- Auth
  |-- Prediction API
  |-- Agent API
  |     `-- LangGraph -> OpenAI API
  |-- Workflow Service
  |-- Dataset/Training Service
  |-- Model Registry
  `-- Admin API
        |
        +---- PostgreSQL
        +---- Artifact Storage
        |
        +---- Representation Client
                    |
              GPU Server / H100
                    |
                   Aya
                    |
                Embedding
                    |
        +-----------+
        |
 sklearn Training / Inference
```

Aya GPU server와 sklearn execution environment를 분리한다.

관리자 사무용 PC/CPU server에서도 cached embedding 또는 Representation API 결과를 이용하여 sklearn 학습, validation, inference 및 joblib 생성을 할 수 있어야 한다.
