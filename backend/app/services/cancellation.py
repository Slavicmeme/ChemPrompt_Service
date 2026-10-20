"""Cooperative cancellation for training jobs.

A training job runs in one worker thread. ``cancel_scope`` installs a thread-local check function;
long loops (stage boundaries, representation batches, prompt building) call ``check_cancelled()``,
which raises ``JobCancelled`` once a cancel was requested. Work in progress inside a single call
(e.g. one embedding batch) finishes first.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Callable, Iterator

_local = threading.local()


class JobCancelled(RuntimeError):
    pass


@contextmanager
def cancel_scope(should_cancel: Callable[[], bool] | None) -> Iterator[None]:
    prev = getattr(_local, "check", None)
    _local.check = should_cancel
    try:
        yield
    finally:
        _local.check = prev


def check_cancelled() -> None:
    check = getattr(_local, "check", None)
    if check is not None and check():
        raise JobCancelled("사용자 요청으로 학습이 중단됨")
