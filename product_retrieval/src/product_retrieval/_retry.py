"""Single place for throttling retries: Databricks endpoints signal overload with 429 / RESOURCE_EXHAUSTED style errors."""

from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")
THROTTLE_MARKERS: tuple[str, ...] = (
    "429",
    "RESOURCE_EXHAUSTED",
    "REQUEST_LIMIT_EXCEEDED",
    "Too Many Requests",
    "TEMPORARILY_UNAVAILABLE",
)


def is_throttled(error: BaseException) -> bool:
    return any(marker in str(error) for marker in THROTTLE_MARKERS)


def with_backoff(fn: Callable[[], T], attempts: int = 5, base_delay: float = 1.0) -> T:
    """Call `fn`, retrying throttling errors with exponential backoff; other errors propagate immediately."""
    for attempt in range(attempts - 1):
        try:
            return fn()
        except Exception as e:
            if not is_throttled(e):
                raise
            time.sleep(base_delay * 2**attempt)
    return fn()
