from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from statistics import mean, pstdev
from typing import Any

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

from config import settings
from utils import (
    MovingAverage,
    RateTracker,
    logger,
)


REQUEST_COUNTER = Counter(
    "inferra_requests_total",
    "Total inference requests",
)

TOKEN_COUNTER = Counter(
    "inferra_generated_tokens_total",
    "Total generated tokens",
)

QUEUE_SIZE_GAUGE = Gauge(
    "inferra_queue_size",
    "Current scheduler queue size",
)

ACTIVE_BATCHES_GAUGE = Gauge(
    "inferra_active_batches",
    "Current active inference batches",
)

GPU_UTILIZATION_GAUGE = Gauge(
    "inferra_gpu_utilization",
    "GPU utilization percentage",
)

CACHE_HIT_RATE_GAUGE = Gauge(
    "inferra_cache_hit_rate",
    "KV cache hit rate",
)

LATENCY_HISTOGRAM = Histogram(
    "inferra_request_latency_ms",
    "Inference latency in milliseconds",
    buckets=(
        1,
        5,
        10,
        25,
        50,
        100,
        250,
        500,
        1000,
        2000,
        5000,
    ),
)

BATCH_SIZE_HISTOGRAM = Histogram(
    "inferra_batch_size",
    "Batch size distribution",
    buckets=(1, 2, 4, 8, 16, 32, 64, 128),
)

TOKENS_PER_SECOND_GAUGE = Gauge(
    "inferra_tokens_per_second",
    "Current token throughput",
)

REQUESTS_PER_SECOND_GAUGE = Gauge(
    "inferra_requests_per_second",
    "Current request throughput",
)

PADDING_EFFICIENCY_GAUGE = Gauge(
    "inferra_padding_efficiency",
    "Batch padding efficiency",
)

PRESSURE_SCORE_GAUGE = Gauge(
    "inferra_runtime_pressure_score",
    "Runtime pressure score",
)

EFFICIENCY_SCORE_GAUGE = Gauge(
    "inferra_runtime_efficiency_score",
    "Overall runtime efficiency score",
)

TPS_STABILITY_GAUGE = Gauge(
    "inferra_throughput_stability",
    "Throughput stability score",
)


@dataclass(slots=True)
class LatencySnapshot:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    average_ms: float


@dataclass(slots=True)
class ThroughputSnapshot:
    requests_per_second: float
    tokens_per_second: float
    throughput_stability: float


@dataclass(slots=True)
class RuntimeMetricsSnapshot:
    queue_size: int
    active_batches: int

    total_requests: int
    total_tokens: int

    avg_batch_size: float
    padding_efficiency: float

    cache_hit_rate: float

    runtime_pressure_score: float
    runtime_efficiency_score: float

    latency: LatencySnapshot
    throughput: ThroughputSnapshot

    timestamp: float = field(
        default_factory=time.time,
    )


class LatencyTracker:
    def __init__(
        self,
        history_size: int = 10000,
    ) -> None:
        self._history = deque(
            maxlen=history_size,
        )

        self._lock = threading.Lock()

    def record(
        self,
        latency_ms: float,
    ) -> None:
        with self._lock:
            self._history.append(
                latency_ms,
            )

        LATENCY_HISTOGRAM.observe(
            latency_ms,
        )

    def snapshot(
        self,
    ) -> LatencySnapshot:
        with self._lock:
            values = sorted(
                self._history,
            )

        if not values:
            return LatencySnapshot(
                0.0,
                0.0,
                0.0,
                0.0,
            )

        return LatencySnapshot(
            p50_ms=self._percentile(values, 50),
            p95_ms=self._percentile(values, 95),
            p99_ms=self._percentile(values, 99),
            average_ms=mean(values),
        )

    @staticmethod
    def _percentile(
        values: list[float],
        percentile: int,
    ) -> float:
        if not values:
            return 0.0

        rank = (
            percentile / 100
        ) * (len(values) - 1)

        lower = math.floor(rank)
        upper = math.ceil(rank)

        if lower == upper:
            return values[lower]

        interpolation = rank - lower

        return (
            values[lower]
            + (
                values[upper]
                - values[lower]
            )
            * interpolation
        )


class ThroughputTracker:
    def __init__(self) -> None:
        self.request_tracker = (
            RateTracker()
        )

        self.token_tracker = (
            RateTracker()
        )

        self.tps_history = deque(
            maxlen=1000,
        )

        self._lock = threading.Lock()

    def record_request(self) -> None:
        self.request_tracker.increment()

        REQUEST_COUNTER.inc()

    def record_tokens(
        self,
        count: int,
    ) -> None:
        self.token_tracker.increment(
            count,
        )

        TOKEN_COUNTER.inc(
            count,
        )

        with self._lock:
            self.tps_history.append(
                self.token_tracker.rate,
            )

    def snapshot(
        self,
    ) -> ThroughputSnapshot:
        requests_per_second = (
            self.request_tracker.rate
        )

        tokens_per_second = (
            self.token_tracker.rate
        )

        throughput_stability = (
            self._throughput_stability()
        )

        REQUESTS_PER_SECOND_GAUGE.set(
            requests_per_second,
        )

        TOKENS_PER_SECOND_GAUGE.set(
            tokens_per_second,
        )

        TPS_STABILITY_GAUGE.set(
            throughput_stability,
        )

        return ThroughputSnapshot(
            requests_per_second=(
                requests_per_second
            ),
            tokens_per_second=(
                tokens_per_second
            ),
            throughput_stability=(
                throughput_stability
            ),
        )

    def _throughput_stability(
        self,
    ) -> float:
        with self._lock:
            values = list(
                self.tps_history,
            )

        if len(values) < 2:
            return 1.0

        deviation = pstdev(values)

        average_tps = mean(values)

        if average_tps <= 0:
            return 0.0

        stability = max(
            0.0,
            1.0 - (
                deviation
                / average_tps
            ),
        )

        return round(
            stability,
            4,
        )


class BatchMetricsTracker:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

        self.padding_efficiency = (
            MovingAverage(
                window_size=1000,
            )
        )

        self._lock = threading.Lock()

    def record_batch(
        self,
        batch_size: int,
        active_tokens: int,
        padded_tokens: int,
    ) -> None:
        with self._lock:
            self.batch_sizes.append(
                batch_size,
            )

            if len(self.batch_sizes) > 10000:
                self.batch_sizes.pop(0)

        BATCH_SIZE_HISTOGRAM.observe(
            batch_size,
        )

        efficiency = (
            active_tokens
            / padded_tokens
            if padded_tokens > 0
            else 1.0
        )

        self.padding_efficiency.add(
            efficiency,
        )

        PADDING_EFFICIENCY_GAUGE.set(
            self.padding_efficiency.average,
        )

    @property
    def average_batch_size(
        self,
    ) -> float:
        with self._lock:
            if not self.batch_sizes:
                return 0.0

            return mean(
                self.batch_sizes,
            )


class CacheMetricsTracker:
    def __init__(self) -> None:
        self.cache_hits = 0
        self.cache_misses = 0

        self._lock = threading.Lock()

    def record_hit(self) -> None:
        with self._lock:
            self.cache_hits += 1

        CACHE_HIT_RATE_GAUGE.set(
            self.hit_rate,
        )

    def record_miss(self) -> None:
        with self._lock:
            self.cache_misses += 1

        CACHE_HIT_RATE_GAUGE.set(
            self.hit_rate,
        )

    @property
    def hit_rate(self) -> float:
        total = (
            self.cache_hits
            + self.cache_misses
        )

        if total == 0:
            return 0.0

        return (
            self.cache_hits
            / total
        )


class RuntimeMetrics:
    def __init__(self) -> None:
        self.latency = (
            LatencyTracker()
        )

        self.throughput = (
            ThroughputTracker()
        )

        self.batch = (
            BatchMetricsTracker()
        )

        self.cache = (
            CacheMetricsTracker()
        )

        self.queue_size = 0
        self.active_batches = 0

        self._lock = threading.Lock()

        self._prometheus_started = False

        if settings.metrics.prometheus_enabled:
            self._start_prometheus_server()

    def _start_prometheus_server(
        self,
    ) -> None:
        if self._prometheus_started:
            return

        start_http_server(
            settings.metrics.prometheus_port,
        )

        self._prometheus_started = True

        logger.info(
            "prometheus_server_started",
            port=settings.metrics.prometheus_port,
        )

    def set_queue_size(
        self,
        size: int,
    ) -> None:
        with self._lock:
            self.queue_size = size

        QUEUE_SIZE_GAUGE.set(size)

    def set_active_batches(
        self,
        count: int,
    ) -> None:
        with self._lock:
            self.active_batches = count

        ACTIVE_BATCHES_GAUGE.set(
            count,
        )

    def set_gpu_utilization(
        self,
        utilization: float,
    ) -> None:
        GPU_UTILIZATION_GAUGE.set(
            utilization,
        )

    def record_request(
        self,
        latency_ms: float,
        generated_tokens: int,
    ) -> None:
        self.latency.record(
            latency_ms,
        )

        self.throughput.record_request()

        self.throughput.record_tokens(
            generated_tokens,
        )

    def record_batch(
        self,
        batch_size: int,
        active_tokens: int,
        padded_tokens: int,
    ) -> None:
        self.batch.record_batch(
            batch_size=batch_size,
            active_tokens=active_tokens,
            padded_tokens=padded_tokens,
        )

    def record_cache_hit(
        self,
    ) -> None:
        self.cache.record_hit()

    def record_cache_miss(
        self,
    ) -> None:
        self.cache.record_miss()

    def runtime_pressure_score(
        self,
    ) -> float:
        latency = (
            self.latency.snapshot().p95_ms
        )

        pressure = (
            self.queue_size * latency
        )

        PRESSURE_SCORE_GAUGE.set(
            pressure,
        )

        return round(
            pressure,
            2,
        )

    def runtime_efficiency_score(
        self,
    ) -> float:
        throughput = (
            self.throughput.snapshot()
            .tokens_per_second
        )

        latency = max(
            self.latency.snapshot().average_ms,
            1.0,
        )

        padding_efficiency = (
            self.batch.padding_efficiency.average
        )

        efficiency = (
            throughput
            * padding_efficiency
            / latency
        )

        EFFICIENCY_SCORE_GAUGE.set(
            efficiency,
        )

        return round(
            efficiency,
            4,
        )

    def snapshot(
        self,
    ) -> RuntimeMetricsSnapshot:
        latency_snapshot = (
            self.latency.snapshot()
        )

        throughput_snapshot = (
            self.throughput.snapshot()
        )

        return RuntimeMetricsSnapshot(
            queue_size=self.queue_size,
            active_batches=self.active_batches,
            total_requests=(
                self.throughput
                .request_tracker
                .total_events
            ),
            total_tokens=(
                self.throughput
                .token_tracker
                .total_events
            ),
            avg_batch_size=(
                self.batch.average_batch_size
            ),
            padding_efficiency=(
                self.batch
                .padding_efficiency
                .average
            ),
            cache_hit_rate=(
                self.cache.hit_rate
            ),
            runtime_pressure_score=(
                self.runtime_pressure_score()
            ),
            runtime_efficiency_score=(
                self.runtime_efficiency_score()
            ),
            latency=latency_snapshot,
            throughput=throughput_snapshot,
        )

    def benchmark_summary(
        self,
    ) -> dict[str, Any]:
        snapshot = self.snapshot()

        return {
            "requests": snapshot.total_requests,
            "tokens": snapshot.total_tokens,
            "avg_batch_size": round(
                snapshot.avg_batch_size,
                2,
            ),
            "padding_efficiency": round(
                snapshot.padding_efficiency,
                4,
            ),
            "cache_hit_rate": round(
                snapshot.cache_hit_rate,
                4,
            ),
            "p50_latency_ms": round(
                snapshot.latency.p50_ms,
                2,
            ),
            "p95_latency_ms": round(
                snapshot.latency.p95_ms,
                2,
            ),
            "p99_latency_ms": round(
                snapshot.latency.p99_ms,
                2,
            ),
            "tokens_per_second": round(
                snapshot.throughput
                .tokens_per_second,
                2,
            ),
            "requests_per_second": round(
                snapshot.throughput
                .requests_per_second,
                2,
            ),
            "throughput_stability": round(
                snapshot.throughput
                .throughput_stability,
                4,
            ),
            "runtime_pressure_score": round(
                snapshot.runtime_pressure_score,
                2,
            ),
            "runtime_efficiency_score": round(
                snapshot.runtime_efficiency_score,
                4,
            ),
        }

    def export(
        self,
    ) -> dict[str, Any]:
        snapshot = self.snapshot()

        return asdict(
            snapshot,
        )

    def p95_latency(
        self,
    ) -> float:
        return (
            self.latency.snapshot().p95_ms
        )

    def throughput_tokens_per_second(
        self,
    ) -> float:
        return (
            self.throughput.snapshot()
            .tokens_per_second
        )


metrics = RuntimeMetrics()