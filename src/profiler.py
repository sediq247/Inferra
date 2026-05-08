from __future__ import annotations

import asyncio
import cProfile
import io
import pstats
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

from config import settings
from utils import logger


@dataclass(slots=True)
class TraceEvent:
    name: str
    duration_ms: float
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SectionStatistics:
    section: str
    calls: int
    total_duration_ms: float
    average_duration_ms: float
    max_duration_ms: float
    min_duration_ms: float


class TraceRecorder:
    def __init__(self, max_history: int = 100000) -> None:
        self._events: deque[TraceEvent] = deque(maxlen=max_history)
        self._lock = threading.Lock()

    def record(
        self,
        name: str,
        duration_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        event = TraceEvent(
            name=name,
            duration_ms=duration_ms,
            timestamp=time.time(),
            metadata=metadata or {},
        )

        with self._lock:
            self._events.append(event)

    def get_events(self) -> list[TraceEvent]:
        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


class RuntimeProfiler:
    def __init__(self) -> None:
        self.enabled = settings.profiler.enabled

        self.trace_recorder = TraceRecorder()

        self._active_profiles: dict[str, cProfile.Profile] = {}
        self._lock = threading.Lock()

    @contextmanager
    def profile_section(
        self,
        section_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        start_time = time.perf_counter()

        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000

            self.trace_recorder.record(
                name=section_name,
                duration_ms=duration_ms,
                metadata=metadata,
            )

    @asynccontextmanager
    async def async_profile_section(
        self,
        section_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        start_time = time.perf_counter()

        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000

            self.trace_recorder.record(
                name=section_name,
                duration_ms=duration_ms,
                metadata=metadata,
            )

    def start_cpu_profile(self, profile_name: str) -> None:
        if not self.enabled:
            return

        profiler = cProfile.Profile()
        profiler.enable()

        with self._lock:
            self._active_profiles[profile_name] = profiler

        logger.info(
            "cpu_profile_started",
            profile_name=profile_name,
        )

    def stop_cpu_profile(self, profile_name: str) -> str:
        with self._lock:
            profiler = self._active_profiles.pop(profile_name, None)

        if profiler is None:
            raise ValueError(f"Unknown profile session: {profile_name}")

        profiler.disable()

        output = io.StringIO()

        stats = pstats.Stats(profiler, stream=output)
        stats.sort_stats("cumulative")
        stats.print_stats()

        logger.info(
            "cpu_profile_completed",
            profile_name=profile_name,
        )

        return output.getvalue()

    def generate_statistics(self) -> list[SectionStatistics]:
        events = self.trace_recorder.get_events()

        grouped: dict[str, list[float]] = defaultdict(list)

        for event in events:
            grouped[event.name].append(event.duration_ms)

        statistics: list[SectionStatistics] = []

        for section, durations in grouped.items():
            statistics.append(
                SectionStatistics(
                    section=section,
                    calls=len(durations),
                    total_duration_ms=sum(durations),
                    average_duration_ms=sum(durations) / len(durations),
                    max_duration_ms=max(durations),
                    min_duration_ms=min(durations),
                )
            )

        statistics.sort(
            key=lambda item: item.total_duration_ms,
            reverse=True,
        )

        return statistics

    def generate_report(self) -> dict[str, Any]:
        statistics = self.generate_statistics()

        return {
            "enabled": self.enabled,
            "generated_at": time.time(),
            "sections": [asdict(stat) for stat in statistics],
        }

    async def monitor_event_loop_lag(
        self,
        interval_seconds: float = 1.0,
    ) -> None:
        if not self.enabled:
            return

        loop = asyncio.get_running_loop()

        while True:
            expected_time = loop.time() + interval_seconds

            await asyncio.sleep(interval_seconds)

            lag_ms = max(
                0.0,
                (loop.time() - expected_time) * 1000,
            )

            self.trace_recorder.record(
                name="event_loop_lag",
                duration_ms=lag_ms,
            )

    def export_traces(self) -> list[dict[str, Any]]:
        return [
            asdict(event)
            for event in self.trace_recorder.get_events()
        ]

    def reset(self) -> None:
        self.trace_recorder.clear()

        logger.info("profiler_state_reset")


profiler = RuntimeProfiler()