from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any

from prometheus_client import Counter, Gauge, Histogram, start_http_server

from config import settings
from utils import MovingAverage, RateTracker, logger


REQUEST_COUNTER = Counter(
    "inferra_requests_total",
    "Total number of inference requests",
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
    "Number of active inference batches",
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
    "Inference request latency in milliseconds",
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 5000),
)

BATCH_SIZE_HISTOGRAM = Histogram(
    "inferra_batch_size",
    "Inference batch sizes",
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
    "Batch padding efficiency ratio",
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


@dataclass(slots=True)
class RuntimeMetricsSnapshot:
    queue_size: int
    active_batches: int
    total_requests: int
    total_tokens: int
    avg_batch_size: float
    cache_hit_rate: float
    padding_efficiency: float
    latency: LatencySnapshot
    throughput: ThroughputSnapshot
    timestamp: float = field(default_factory=time.time)


class LatencyTracker:
    def __init__(self, history_size: int = 10000) -> None:
        self._history = deque(maxlen=history_size)
        self._lock = threading.Lock()

    def record(self, latency_ms: float) -> None:
        with self._lock:
            self._history.append(latency_ms)

        LATENCY_HISTOGRAM.observe(latency_ms)

    def snapshot(self) -> LatencySnapshot:
        with self._lock:
            values = sorted(self._history)

        if not values:
            return LatencySnapshot(0.0, 0.0, 0.0, 0.0)

        return LatencySnapshot(
            p50_ms=self._percentile(values, 50),
            p95_ms=self._percentile(values, 95),
            p99_ms=self._percentile(values, 99),
            average_ms=mean(values),
        )

    @staticmethod
    def _percentile(values: list[float], percentile: int) -> float:
        if not values:
            return 0.0

        index = int((percentile / 100) * (len(values) - 1))
        return values[index]


class ThroughputTracker:
    def __init__(self) -> None:
        self.request_tracker = RateTracker()
        self.token_tracker = RateTracker()

    def record_request(self) -> None:
        self.request_tracker.increment()
        REQUEST_COUNTER.inc()

    def record_tokens(self, count: int) -> None:
        self.token_tracker.increment(count)
        TOKEN_COUNTER.inc(count)

    def snapshot(self) -> ThroughputSnapshot:
        requests_per_second = self.request_tracker.rate
        tokens_per_second = self.token_tracker.rate

        REQUESTS_PER_SECOND_GAUGE.set(requests_per_second)
        TOKENS_PER_SECOND_GAUGE.set(tokens_per_second)

        return ThroughputSnapshot(
            requests_per_second=requests_per_second,
            tokens_per_second=tokens_per_second,
        )


class BatchMetricsTracker:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.padding_efficiency = MovingAverage(window_size=1000)
        self._lock = threading.Lock()

    def record_batch(
        self,
        batch_size: int,
        active_tokens: int,
        padded_tokens: int,
    ) -> None:
        with self._lock:
            self.batch_sizes.append(batch_size)

            if len(self.batch_sizes) > 10000:
                self.batch_sizes.pop(0)

        BATCH_SIZE_HISTOGRAM.observe(batch_size)

        efficiency = (
            active_tokens / padded_tokens
            if padded_tokens > 0
            else 1.0
        )

        self.padding_efficiency.add(efficiency)
        PADDING_EFFICIENCY_GAUGE.set(self.padding_efficiency.average)

    @property
    def average_batch_size(self) -> float:
        with self._lock:
            if not self.batch_sizes:
                return 0.0

            return mean(self.batch_sizes)


class CacheMetricsTracker:
    def __init__(self) -> None:
        self.cache_hits = 0
        self.cache_misses = 0
        self._lock = threading.Lock()

    def record_hit(self) -> None:
        with self._lock:
            self.cache_hits += 1

        CACHE_HIT_RATE_GAUGE.set(self.hit_rate)

    def record_miss(self) -> None:
        with self._lock:
            self.cache_misses += 1

        CACHE_HIT_RATE_GAUGE.set(self.hit_rate)

    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses

        if total == 0:
            return 0.0

        return self.cache_hits / total


class RuntimeMetrics:
    def __init__(self) -> None:
        self.latency = LatencyTracker()
        self.throughput = ThroughputTracker()
        self.batch = BatchMetricsTracker()
        self.cache = CacheMetricsTracker()

        self.queue_size = 0
        self.active_batches = 0

        self._lock = threading.Lock()

        self._prometheus_started = False

        if settings.metrics.prometheus_enabled:
            self._start_prometheus_server()

    def _start_prometheus_server(self) -> None:
        if self._prometheus_started:
            return

        start_http_server(settings.metrics.prometheus_port)

        self._prometheus_started = True

        logger.info(
            "prometheus_metrics_server_started",
            port=settings.metrics.prometheus_port,
        )

    def set_queue_size(self, size: int) -> None:
        with self._lock:
            self.queue_size = size

        QUEUE_SIZE_GAUGE.set(size)

    def set_active_batches(self, count: int) -> None:
        with self._lock:
            self.active_batches = count

        ACTIVE_BATCHES_GAUGE.set(count)

    def set_gpu_utilization(self, utilization: float) -> None:
        GPU_UTILIZATION_GAUGE.set(utilization)

    def record_request(
        self,
        latency_ms: float,
        generated_tokens: int,
    ) -> None:
        self.latency.record(latency_ms)

        self.throughput.record_request()
        self.throughput.record_tokens(generated_tokens)

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

    def record_cache_hit(self) -> None:
        self.cache.record_hit()

    def record_cache_miss(self) -> None:
        self.cache.record_miss()

    def snapshot(self) -> RuntimeMetricsSnapshot:
        latency_snapshot = self.latency.snapshot()
        throughput_snapshot = self.throughput.snapshot()

        return RuntimeMetricsSnapshot(
            queue_size=self.queue_size,
            active_batches=self.active_batches,
            total_requests=self.throughput.request_tracker.total_events,
            total_tokens=self.throughput.token_tracker.total_events,
            avg_batch_size=self.batch.average_batch_size,
            cache_hit_rate=self.cache.hit_rate,
            padding_efficiency=self.batch.padding_efficiency.average,
            latency=latency_snapshot,
            throughput=throughput_snapshot,
        )

    def export(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        return asdict(snapshot)


metrics = RuntimeMetrics()