"""Deadline, rate limiting and retrying HTTP GET shared by the retrieval clients."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import requests

log = logging.getLogger("evidence")


class DeadlineExceeded(Exception):
    """The whole request ran out of its time budget."""


class Deadline:
    def __init__(self, seconds: float, clock=time.monotonic):
        self._clock = clock
        self.started = clock()
        self.ends = self.started + seconds

    def remaining(self) -> float:
        return max(0.0, self.ends - self._clock())

    def elapsed(self) -> float:
        return self._clock() - self.started

    def expired(self) -> bool:
        return self.remaining() <= 0

    def check(self, stage: str) -> None:
        if self.expired():
            raise DeadlineExceeded(stage)

    def timeout(self, cap: float) -> float:
        """A per-call timeout that never exceeds what is left."""
        return max(1.0, min(cap, self.remaining()))


class RateLimiter:
    """Allows at most `per_second` calls a second across threads."""

    def __init__(self, per_second: float, clock=time.monotonic, sleep=time.sleep):
        self.interval = 1.0 / per_second if per_second > 0 else 0.0
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        if self.interval <= 0:
            return
        with self._lock:
            now = self._clock()
            if now < self._next_allowed:
                self._sleep(self._next_allowed - now)
                now = self._next_allowed
            self._next_allowed = now + self.interval


RETRY_STATUSES = {429, 500, 502, 503, 504}


class UpstreamError(Exception):
    def __init__(self, service: str, detail: str):
        super().__init__(f"{service}: {detail}")
        self.service = service
        self.detail = detail


def get_with_retries(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any],
    service: str,
    deadline: Deadline,
    limiter: RateLimiter | None = None,
    attempts: int = 3,
    per_call_timeout: float = 20.0,
    sleep=time.sleep,
    headers: dict[str, str] | None = None,
) -> requests.Response:
    """GET with exponential backoff on 429 and 5xx. Never logs the query text."""
    last: str = "no attempt made"
    for attempt in range(1, attempts + 1):
        deadline.check(f"{service} request")
        if limiter is not None:
            limiter.wait()
        try:
            response = session.get(
                url,
                params=params,
                timeout=deadline.timeout(per_call_timeout),
                headers=headers,
            )
        except requests.RequestException as exc:  # connection errors, timeouts
            last = type(exc).__name__
            log.warning("%s attempt %d failed: %s", service, attempt, last)
        else:
            if response.status_code < 400:
                return response
            last = f"HTTP {response.status_code}"
            if response.status_code not in RETRY_STATUSES:
                raise UpstreamError(service, last)
            log.warning("%s attempt %d returned %s", service, attempt, last)
        if attempt < attempts:
            backoff = min(0.5 * (2 ** (attempt - 1)), deadline.remaining())
            if backoff <= 0:
                break
            sleep(backoff)
    raise UpstreamError(service, f"gave up after {attempts} attempts ({last})")
