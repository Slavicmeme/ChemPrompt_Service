"""Runtime configuration (docs/17_DEPLOYMENT.md). Secrets come only from the environment."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote_plus

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _database_url() -> str:
    """DATABASE_URL if given, otherwise composed from POSTGRES_HOST/PORT/USER/PASSWORD/DB."""
    if _env("DATABASE_URL"):
        return _env("DATABASE_URL")
    user = quote_plus(_env("POSTGRES_USER", "chemprompt"))
    password = quote_plus(os.environ.get("POSTGRES_PASSWORD", ""))
    host, port = _env("POSTGRES_HOST", "127.0.0.1"), _env("POSTGRES_PORT", "5432")
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{_env('POSTGRES_DB', 'chemprompt')}"


@dataclass(frozen=True)
class SplitPolicy:
    """Split seed policy (docs/03).

    Existing ChemPrompt code (Chemprompt/utils/data_splitter.py::_scaffold_seeds) derives scaffold
    seeds as ``base_seed + i`` with ``base_seed = 42`` and used ``{'scaffold': 3}`` folds
    (seeds 42, 43, 44). The same rule extended to 5 folds gives 42..46.
    KFold/StratifiedKFold uses the single ``RANDOM_SEED = 42`` used throughout the notebooks.
    """

    n_folds: int = 5
    random_seed: int = 42
    scaffold_seeds: tuple[int, ...] = (42, 43, 44, 45, 46)


@dataclass(frozen=True)
class Settings:
    database_url: str = field(default_factory=_database_url)
    secret_key: str = field(default_factory=lambda: _env("SECRET_KEY"))
    session_ttl_hours: int = field(default_factory=lambda: int(_env("SESSION_TTL_HOURS", "24")))
    cookie_secure: bool = field(default_factory=lambda: _env("COOKIE_SECURE", "false").lower() == "true")

    storage_root: Path = field(default_factory=lambda: Path(_env("MODEL_STORAGE_ROOT", str(PROJECT_ROOT / "storage"))).resolve())

    # Representation (Aya) — docs/07
    representation_mode: str = field(default_factory=lambda: _env("REPRESENTATION_MODE", "http").lower())  # http (Aya) | mock (dev/tests)
    representation_service_url: str = field(default_factory=lambda: _env("REPRESENTATION_SERVICE_URL", "http://localhost:8100"))
    representation_timeout: float = field(default_factory=lambda: float(_env("REPRESENTATION_TIMEOUT", "600")))
    representation_batch_size: int = field(default_factory=lambda: int(_env("REPRESENTATION_BATCH_SIZE", "64")))
    aya_model_id: str = field(default_factory=lambda: _env("AYA_MODEL_ID", "CohereLabs/aya-expanse-8b"))
    mock_embedding_dim: int = field(default_factory=lambda: int(_env("MOCK_EMBEDDING_DIM", "256")))
    prompt_version: str = "chemprompt-v1"          # LLMModel._create_prompt format
    preprocessing_version: str = "descfilter-v1"    # LLMModel._filter_descriptors (std>1 or unique>10)

    # Descriptor modes / GA — docs/04
    feature_selection_k: int = 50
    feature_selection_method: str = "f_regression"
    ga_mode: str = field(default_factory=lambda: _env("GA_MODE", "genetic").lower())  # genetic (original GA) | mock (dev/tests)
    ga_params: dict = field(default_factory=lambda: {
        # Optimization_of_Chemprompt.ipynb defaults
        "num_generations": 50, "pop_size": 40, "top_k": 5, "mutation_rate": 0.05, "inner_folds": 3,
    })

    split: SplitPolicy = field(default_factory=SplitPolicy)
    classification_threshold: float = 0.5

    # Agent — docs/12, docs/15
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    openai_agent_model: str = field(default_factory=lambda: _env("OPENAI_AGENT_MODEL", "gpt-4o-mini"))
    # reasoning models (e.g. gpt-5.x): none | minimal | low | medium | high. Empty = parameter not sent.
    openai_reasoning_effort: str = field(default_factory=lambda: _env("OPENAI_REASONING_EFFORT").lower())
    langfuse_public_key: str = field(default_factory=lambda: _env("LANGFUSE_PUBLIC_KEY"))
    langfuse_secret_key: str = field(default_factory=lambda: _env("LANGFUSE_SECRET_KEY"))
    langfuse_base_url: str = field(default_factory=lambda: _env("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"))

    training_workers: int = field(default_factory=lambda: int(_env("TRAINING_WORKERS", "2")))
    max_upload_mb: int = field(default_factory=lambda: int(_env("MAX_UPLOAD_MB", "50")))
    max_prediction_batch: int = 100


settings = Settings()
