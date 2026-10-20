"""SQLAlchemy 2.x ORM mapping. The schema itself is owned by ./sql/*.sql (no create_all here)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Integer, SmallInteger, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# Child collections use cascade="all, delete-orphan", passive_deletes=True: deletes are executed by the
# ON DELETE CASCADE foreign keys defined in sql/*.sql instead of the ORM nulling child FKs.
class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSONB, list[Any]: JSONB, Any: JSONB}


def _id() -> Mapped[int]:
    return mapped_column(BigInteger, primary_key=True, autoincrement=True)


def _created() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now())


# --- 001 -------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = _id()
    username: Mapped[str] = mapped_column(String(64), unique=True)
    password_hash: Mapped[str] = mapped_column(Text)
    role: Mapped[str] = mapped_column(String(16), default="USER")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = _created()

    @property
    def is_admin(self) -> bool:
        return self.role == "ADMIN"


class UserSession(Base):
    __tablename__ = "user_sessions"
    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = _created()
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    user: Mapped[User] = relationship()


# --- 002 -------------------------------------------------------------------
class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[int] = _id()
    owner_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"))
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[Optional[str]] = mapped_column(Text)
    is_builtin: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = _created()
    versions: Mapped[list["DatasetVersion"]] = relationship(back_populates="dataset", order_by="DatasetVersion.version", cascade="all, delete-orphan", passive_deletes=True)


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    id: Mapped[int] = _id()
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    version: Mapped[int] = mapped_column(Integer)
    storage_key: Mapped[str] = mapped_column(Text)
    raw_checksum: Mapped[str] = mapped_column(String(64))
    checksum: Mapped[Optional[str]] = mapped_column(String(64))
    original_filename: Mapped[Optional[str]] = mapped_column(Text)
    columns: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    smiles_column: Mapped[Optional[str]] = mapped_column(Text)
    target_columns: Mapped[Optional[list[Any]]] = mapped_column(JSONB)
    task_type: Mapped[Optional[str]] = mapped_column(String(16))
    row_count: Mapped[Optional[int]] = mapped_column(Integer)
    valid_row_count: Mapped[Optional[int]] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="UPLOADED")
    validation_report: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = _created()
    dataset: Mapped[Dataset] = relationship(back_populates="versions")


class Property(Base):
    __tablename__ = "properties"
    id: Mapped[int] = _id()
    code: Mapped[str] = mapped_column(String(64), unique=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[Optional[str]] = mapped_column(Text)
    task_type: Mapped[str] = mapped_column(String(16))
    unit: Mapped[Optional[str]] = mapped_column(String(64))
    target_columns: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    is_builtin: Mapped[bool] = mapped_column(Boolean, default=True)
    dataset_id: Mapped[Optional[int]] = mapped_column(ForeignKey("datasets.id"))
    name_ko: Mapped[Optional[str]] = mapped_column(String(200))
    target_info: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = _created()


# --- 003 -------------------------------------------------------------------
class SplitConfig(Base):
    __tablename__ = "split_configs"
    id: Mapped[int] = _id()
    dataset_version_id: Mapped[int] = mapped_column(ForeignKey("dataset_versions.id"))
    strategy: Mapped[str] = mapped_column(String(16))
    n_folds: Mapped[int] = mapped_column(Integer, default=5)
    seed: Mapped[int] = mapped_column(Integer)
    seeds: Mapped[list[Any]] = mapped_column(JSONB)
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    checksum: Mapped[str] = mapped_column(String(64))
    assignment_checksum: Mapped[Optional[str]] = mapped_column(String(64))
    created_at: Mapped[datetime] = _created()


class SplitFoldAssignment(Base):
    __tablename__ = "split_fold_assignments"
    split_config_id: Mapped[int] = mapped_column(ForeignKey("split_configs.id"), primary_key=True)
    fold_no: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    row_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    partition: Mapped[str] = mapped_column(String(5))


class RepresentationArtifact(Base):
    __tablename__ = "representation_artifacts"
    id: Mapped[int] = _id()
    owner_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"))
    dataset_version_id: Mapped[Optional[int]] = mapped_column(ForeignKey("dataset_versions.id"))
    cache_key: Mapped[str] = mapped_column(String(64), unique=True)
    representation_model: Mapped[str] = mapped_column(Text)
    representation_backend: Mapped[str] = mapped_column(String(16))
    prompt_version: Mapped[str] = mapped_column(String(32))
    descriptor_mode: Mapped[str] = mapped_column(String(32))
    descriptor_config_hash: Mapped[str] = mapped_column(String(64))
    preprocessing_version: Mapped[str] = mapped_column(String(32))
    storage_key: Mapped[str] = mapped_column(Text)
    checksum: Mapped[str] = mapped_column(String(64))
    n_rows: Mapped[int] = mapped_column(Integer)
    dim: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = _created()


# --- 004 -------------------------------------------------------------------
class TrainingJob(Base):
    __tablename__ = "training_jobs"
    id: Mapped[int] = _id()
    owner_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"))
    dataset_version_id: Mapped[int] = mapped_column(ForeignKey("dataset_versions.id"))
    split_config_id: Mapped[int] = mapped_column(ForeignKey("split_configs.id"))
    property_id: Mapped[Optional[int]] = mapped_column(ForeignKey("properties.id"))
    workflow_id: Mapped[Optional[int]] = mapped_column(ForeignKey("workflows.id"))
    task_type: Mapped[str] = mapped_column(String(16))
    descriptor_mode: Mapped[str] = mapped_column(String(32))
    sklearn_model: Mapped[str] = mapped_column(String(64))
    model_name: Mapped[Optional[str]] = mapped_column(String(200))
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="QUEUED")
    stage: Mapped[Optional[str]] = mapped_column(String(64))
    progress: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    metrics: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    error: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = _created()
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    fold_results: Mapped[list["TrainingFoldResult"]] = relationship(order_by="TrainingFoldResult.fold_no", cascade="all, delete-orphan", passive_deletes=True)


class TrainingFoldResult(Base):
    __tablename__ = "training_fold_results"
    id: Mapped[int] = _id()
    training_job_id: Mapped[int] = mapped_column(ForeignKey("training_jobs.id"))
    fold_no: Mapped[int] = mapped_column(SmallInteger)
    split_seed: Mapped[int] = mapped_column(Integer)
    n_train: Mapped[int] = mapped_column(Integer)
    n_valid: Mapped[int] = mapped_column(Integer)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB)
    selected_descriptors: Mapped[list[Any]] = mapped_column(JSONB)
    representation_id: Mapped[Optional[int]] = mapped_column(ForeignKey("representation_artifacts.id"))
    created_at: Mapped[datetime] = _created()


class GAExperiment(Base):
    __tablename__ = "ga_experiments"
    id: Mapped[int] = _id()
    training_job_id: Mapped[int] = mapped_column(ForeignKey("training_jobs.id"), unique=True)
    provider: Mapped[str] = mapped_column(String(16))
    candidate_k: Mapped[int] = mapped_column(Integer, default=50)
    candidate_method: Mapped[str] = mapped_column(String(32), default="f_regression")
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="RUNNING")
    created_at: Mapped[datetime] = _created()


class GAFoldResult(Base):
    __tablename__ = "ga_fold_results"
    id: Mapped[int] = _id()
    ga_experiment_id: Mapped[int] = mapped_column(ForeignKey("ga_experiments.id"))
    fold_no: Mapped[int] = mapped_column(SmallInteger)
    candidate_descriptors: Mapped[list[Any]] = mapped_column(JSONB)
    flags: Mapped[str] = mapped_column(String(256))
    full_flag: Mapped[str] = mapped_column(Text)
    selected_descriptors: Mapped[list[Any]] = mapped_column(JSONB)
    fitness: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = _created()


# --- 005 -------------------------------------------------------------------
class Model(Base):
    __tablename__ = "models"
    id: Mapped[int] = _id()
    owner_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"))
    property_id: Mapped[Optional[int]] = mapped_column(ForeignKey("properties.id"))
    scope: Mapped[str] = mapped_column(String(16))
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = _created()
    versions: Mapped[list["ModelVersion"]] = relationship(back_populates="model", order_by="ModelVersion.version", cascade="all, delete-orphan", passive_deletes=True)
    prop: Mapped[Optional[Property]] = relationship()


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[int] = _id()
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"))
    version: Mapped[int] = mapped_column(Integer)
    training_job_id: Mapped[Optional[int]] = mapped_column(ForeignKey("training_jobs.id"))
    split_config_id: Mapped[Optional[int]] = mapped_column(ForeignKey("split_configs.id"))
    status: Mapped[str] = mapped_column(String(16), default="UPLOADED")
    task_type: Mapped[str] = mapped_column(String(16))
    target_columns: Mapped[list[Any]] = mapped_column(JSONB, default=list)
    descriptor_mode: Mapped[str] = mapped_column(String(32))
    split_strategy: Mapped[str] = mapped_column(String(16))
    n_folds: Mapped[int] = mapped_column(Integer, default=5)
    split_seeds: Mapped[list[Any]] = mapped_column(JSONB)
    representation_model: Mapped[str] = mapped_column(Text)
    prompt_version: Mapped[str] = mapped_column(String(32))
    sklearn_model: Mapped[str] = mapped_column(String(64))
    sklearn_version: Mapped[Optional[str]] = mapped_column(String(32))
    storage_prefix: Mapped[str] = mapped_column(Text)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB)
    metrics: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    validation_report: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    reviewed_by: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"))
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    review_note: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = _created()
    model: Mapped[Model] = relationship(back_populates="versions")
    artifacts: Mapped[list["ModelArtifact"]] = relationship(order_by="ModelArtifact.id", cascade="all, delete-orphan", passive_deletes=True)

    @property
    def label(self) -> str:
        return f"{self.model.name}:v{self.version}"


class ModelArtifact(Base):
    __tablename__ = "model_artifacts"
    id: Mapped[int] = _id()
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"))
    artifact_type: Mapped[str] = mapped_column(String(32))
    fold_no: Mapped[Optional[int]] = mapped_column(SmallInteger)
    storage_key: Mapped[str] = mapped_column(Text)
    checksum: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = _created()


# --- 006 -------------------------------------------------------------------
class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = _id()
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"))
    smiles: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16))
    task_type: Mapped[str] = mapped_column(String(16))
    result: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    error: Mapped[Optional[str]] = mapped_column(Text)
    model_version_label: Mapped[str] = mapped_column(Text)
    split_strategy: Mapped[str] = mapped_column(String(16))
    n_folds: Mapped[int] = mapped_column(Integer)
    split_config_id: Mapped[Optional[int]] = mapped_column(ForeignKey("split_configs.id"))
    split_seeds: Mapped[list[Any]] = mapped_column(JSONB)
    descriptor_mode: Mapped[str] = mapped_column(String(32))
    representation_model: Mapped[str] = mapped_column(Text)
    prompt_version: Mapped[str] = mapped_column(String(32))
    aggregation: Mapped[str] = mapped_column(String(64))
    source: Mapped[str] = mapped_column(String(16), default="WEB")
    conversation_id: Mapped[Optional[int]] = mapped_column(ForeignKey("agent_conversations.id"))
    created_at: Mapped[datetime] = _created()
    fold_results: Mapped[list["PredictionFoldResult"]] = relationship(order_by="PredictionFoldResult.fold_no", cascade="all, delete-orphan", passive_deletes=True)
    model_version: Mapped["ModelVersion"] = relationship()


class PredictionFoldResult(Base):
    __tablename__ = "prediction_fold_results"
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), primary_key=True)
    fold_no: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    output: Mapped[dict[str, Any]] = mapped_column(JSONB)


# --- 007 -------------------------------------------------------------------
class WorkflowDefinition(Base):
    __tablename__ = "workflow_definitions"
    id: Mapped[int] = _id()
    code: Mapped[str] = mapped_column(String(64), unique=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[Optional[str]] = mapped_column(Text)
    steps: Mapped[list["WorkflowStepDefinition"]] = relationship(order_by="WorkflowStepDefinition.step_no", cascade="all, delete-orphan", passive_deletes=True)


class WorkflowStepDefinition(Base):
    __tablename__ = "workflow_step_definitions"
    id: Mapped[int] = _id()
    workflow_definition_id: Mapped[int] = mapped_column(ForeignKey("workflow_definitions.id"))
    step_no: Mapped[int] = mapped_column(SmallInteger)
    code: Mapped[str] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(200))
    executor: Mapped[str] = mapped_column(String(16))


class Workflow(Base):
    __tablename__ = "workflows"
    id: Mapped[int] = _id()
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    workflow_definition_id: Mapped[int] = mapped_column(ForeignKey("workflow_definitions.id"))
    current_step: Mapped[int] = mapped_column(SmallInteger, default=0)
    status: Mapped[str] = mapped_column(String(16), default="ACTIVE")
    context: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    error: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = _created()
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    definition: Mapped[WorkflowDefinition] = relationship()
    steps: Mapped[list["WorkflowStep"]] = relationship(order_by="WorkflowStep.step_no", cascade="all, delete-orphan", passive_deletes=True)


class WorkflowStep(Base):
    __tablename__ = "workflow_steps"
    id: Mapped[int] = _id()
    workflow_id: Mapped[int] = mapped_column(ForeignKey("workflows.id"))
    step_no: Mapped[int] = mapped_column(SmallInteger)
    code: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16))
    input: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    result: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB)
    error: Mapped[Optional[str]] = mapped_column(Text)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# --- 008 -------------------------------------------------------------------
class AgentConversation(Base):
    __tablename__ = "agent_conversations"
    id: Mapped[int] = _id()
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    title: Mapped[Optional[str]] = mapped_column(String(200))
    trace_id: Mapped[Optional[str]] = mapped_column(String(64))
    created_at: Mapped[datetime] = _created()
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    messages: Mapped[list["AgentMessage"]] = relationship(order_by="AgentMessage.id", cascade="all, delete-orphan", passive_deletes=True)


class AgentMessage(Base):
    __tablename__ = "agent_messages"
    id: Mapped[int] = _id()
    conversation_id: Mapped[int] = mapped_column(ForeignKey("agent_conversations.id"))
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[Optional[str]] = mapped_column(Text)
    tool_calls: Mapped[Optional[list[Any]]] = mapped_column(JSONB)
    tool_call_id: Mapped[Optional[str]] = mapped_column(String(128))
    tool_name: Mapped[Optional[str]] = mapped_column(String(64))
    created_at: Mapped[datetime] = _created()
