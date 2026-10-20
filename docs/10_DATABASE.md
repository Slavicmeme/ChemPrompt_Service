# 10. Database

PostgreSQL + SQLAlchemy 2.x + Alembic.

주요 테이블:
- users
- properties
- datasets
- dataset_versions
- split_configs
- split_fold_assignments 또는 split artifact reference
- representation_artifacts
- training_jobs
- training_fold_results
- model_versions
- model_artifacts
- predictions
- prediction_fold_results
- ga_experiments
- ga_fold_results
- workflow_definitions
- workflow_step_definitions
- workflows
- workflow_steps
- agent_conversations
- agent_messages

## split_configs
예시 필드:
- id
- dataset_version_id
- strategy (RANDOM/SCAFFOLD)
- n_folds
- seed
- config JSONB
- checksum
- created_at

동일 dataset/version + split config로 동일 split을 재현할 수 있어야 한다.
