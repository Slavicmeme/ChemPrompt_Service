"""Langfuse observability (docs/15_LANGFUSE.md).

Langfuse is optional and strictly non-critical: if keys are missing or any Langfuse call fails, a
no-op observation is used and prediction/training/agent behaviour is unchanged.
"""
from __future__ import annotations

import logging
from typing import Any

from app.config import settings

log = logging.getLogger(__name__)

LOCAL_SYSTEM_PROMPT_VERSION = "agent-system-v1"
LANGFUSE_PROMPT_NAME = "chemprompt-agent-system"


class _NoopObs:
    trace_id = None

    def start(self, **_: Any) -> "_NoopObs":
        return self

    def update(self, **_: Any) -> "_NoopObs":
        return self

    def end(self) -> None:
        pass


class _Obs:
    def __init__(self, inner: Any):
        self._inner = inner
        self.trace_id = getattr(inner, "trace_id", None)

    def start(self, **kw: Any) -> "_Obs | _NoopObs":
        try:
            return _Obs(self._inner.start_observation(**kw))
        except Exception as exc:  # noqa: BLE001
            log.warning("langfuse start_observation failed: %s", exc)
            return _NoopObs()

    def update(self, **kw: Any) -> "_Obs":
        try:
            self._inner.update(**kw)
        except Exception as exc:  # noqa: BLE001
            log.warning("langfuse update failed: %s", exc)
        return self

    def end(self) -> None:
        try:
            self._inner.end()
        except Exception as exc:  # noqa: BLE001
            log.warning("langfuse end failed: %s", exc)


class Observer:
    def __init__(self) -> None:
        self._client = None
        if settings.langfuse_public_key and settings.langfuse_secret_key:
            try:
                from langfuse import Langfuse

                self._client = Langfuse(public_key=settings.langfuse_public_key, secret_key=settings.langfuse_secret_key,
                                        base_url=settings.langfuse_base_url)
            except Exception as exc:  # noqa: BLE001
                log.warning("Langfuse disabled: %s", exc)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def trace(self, *, name: str, user_ref: str, session_id: str, input: Any, metadata: dict) -> _Obs | _NoopObs:
        if not self._client:
            return _NoopObs()
        try:
            root = self._client.start_observation(
                name=name, as_type="agent", input=input,
                metadata={"user_ref": user_ref, "session_id": session_id, **metadata},
            )
            return _Obs(root)
        except Exception as exc:  # noqa: BLE001
            log.warning("langfuse trace failed: %s", exc)
            return _NoopObs()

    def system_prompt(self, fallback: str) -> tuple[str, str]:
        """Prompt management: use the Langfuse-managed prompt when available, else the local one."""
        if self._client:
            try:
                p = self._client.get_prompt(LANGFUSE_PROMPT_NAME, type="text", cache_ttl_seconds=300)
                return p.compile(), f"langfuse:{LANGFUSE_PROMPT_NAME}:v{p.version}"
            except Exception:  # noqa: BLE001 — prompt not created in Langfuse yet
                pass
        return fallback, LOCAL_SYSTEM_PROMPT_VERSION

    def flush(self) -> None:
        if self._client:
            try:
                self._client.flush()
            except Exception:  # noqa: BLE001
                pass


observer = Observer()
