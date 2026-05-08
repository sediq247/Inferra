from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterator, TypeVar

import structlog

from config import settings


T = TypeVar("T")


@dataclass(slots=True)
class TimerResult:
    name: str
    elapsed_ms: float


class HighResolutionTimer:
    def __init__(self) -> None:
        self._start_time: float | None = None
        self._end_time: float | None = None

    def start(self) -> None:
        self._start_time = time.perf_counter()

    def stop(self) -> None:
        self._end_time = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        if self._start_time is None:
            raise RuntimeError("Timer has not started")

        end_time = self._end_time or time.perf_counter()

        return (end_time - self._start_time) * 1000


@contextmanager
def timed_section(name: str) -> Iterator[TimerResult]:
    timer = HighResolutionTimer()
    timer.start()

    result = TimerResult(name=name, elapsed_ms=0.0)

    try:
        yield result
    finally:
        timer.stop()
        result.elapsed_ms = timer.elapsed_ms


@asynccontextmanager
async def async_timed_section(name: str) -> AsyncIterator[TimerResult]:
    timer = HighResolutionTimer()
    timer.start()

    result = TimerResult(name=name, elapsed_ms=0.0)

    try:
        yield result
    finally:
        timer.stop()
        result.elapsed_ms = timer.elapsed_ms


class GracefulShutdown:
    def __init__(self) -> None:
        self.shutdown_requested = asyncio.Event()

    def install_handlers(self) -> None:
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum: int, _: Any) -> None:
        logger.warning(
            "shutdown_signal_received",
            signal=signal.Signals(signum).name,
        )

        self.shutdown_requested.set()


class AsyncIntervalWorker:
    def __init__(
        self,
        interval_seconds: float,
        callback: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.interval_seconds = interval_seconds
        self.callback = callback
        self.args = args
        self.kwargs = kwargs

        self._task: asyncio.Task[Any] | None = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self._running = False

        if self._task:
            self._task.cancel()

            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run_loop(self) -> None:
        while self._running:
            try:
                result = self.callback(*self.args, **self.kwargs)

                if asyncio.iscoroutine(result):
                    await result

            except Exception:
                logger.exception("interval_worker_failed")

            await asyncio.sleep(self.interval_seconds)


class RequestIDGenerator:
    @staticmethod
    def generate() -> str:
        return uuid.uuid4().hex


class MovingAverage:
    def __init__(self, window_size: int) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be greater than zero")

        self.window_size = window_size
        self.values: list[float] = []

    def add(self, value: float) -> None:
        self.values.append(value)

        if len(self.values) > self.window_size:
            self.values.pop(0)

    @property
    def average(self) -> float:
        if not self.values:
            return 0.0

        return sum(self.values) / len(self.values)


class RateTracker:
    def __init__(self) -> None:
        self.start_time = time.perf_counter()
        self.total_events = 0

    def increment(self, value: int = 1) -> None:
        self.total_events += value

    @property
    def rate(self) -> float:
        elapsed = time.perf_counter() - self.start_time

        if elapsed <= 0:
            return 0.0

        return self.total_events / elapsed


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def current_timestamp_ms() -> int:
    return int(time.time() * 1000)


def serialize_json(data: dict[str, Any]) -> str:
    return json.dumps(data, separators=(",", ":"), default=str)


def setup_logging() -> structlog.stdlib.BoundLogger:
    ensure_directory(settings.logging.log_dir)

    timestamper = structlog.processors.TimeStamper(fmt="iso")

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.logging.json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    logging.basicConfig(
        level=getattr(logging, settings.logging.level.value),
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger("inferra")


logger = setup_logging()