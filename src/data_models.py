from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from utils import RequestIDGenerator


class RuntimeMode(str, Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    HYBRID = "hybrid"


class RequestPriority(int, Enum):
    HIGH = 0
    NORMAL = 1
    LOW = 2


class RequestStatus(str, Enum):
    QUEUED = "queued"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class InferenceRequest:
    prompt: str
    max_tokens: int
    request_id: str = field(
        default_factory=RequestIDGenerator.generate,
    )
    priority: RequestPriority = RequestPriority.NORMAL
    status: RequestStatus = RequestStatus.QUEUED
    arrival_time: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def estimated_prompt_tokens(self) -> int:
        return len(self.prompt.split())

    @property
    def estimated_total_tokens(self) -> int:
        return (
            self.estimated_prompt_tokens + self.max_tokens
        )


@dataclass(slots=True)
class ScheduledBatch:
    batch_id: str
    requests: list[InferenceRequest]
    total_estimated_tokens: int
    max_request_tokens: int
    padding_tokens: int
    created_at: float = field(default_factory=time.time)

    @property
    def batch_size(self) -> int:
        return len(self.requests)

    @property
    def padding_efficiency(self) -> float:
        total_tokens = (
            self.total_estimated_tokens + self.padding_tokens
        )

        if total_tokens <= 0:
            return 1.0

        return self.total_estimated_tokens / total_tokens


@dataclass(slots=True)
class ExecutionResult:
    request_id: str
    generated_text: str
    generated_tokens: int
    latency_ms: float
    completed_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BatchExecutionResult:
    batch_id: str
    results: list[ExecutionResult]
    total_latency_ms: float
    tokens_per_second: float
    completed_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class SchedulerDecision:
    selected_request_ids: list[str]
    total_estimated_tokens: int
    estimated_padding_tokens: int
    queue_size_before: int
    queue_size_after: int
    runtime_mode: RuntimeMode
    created_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class OptimizerDecision:
    runtime_mode: RuntimeMode
    recommended_batch_size: int
    recommended_batch_tokens: int
    queue_pressure: float
    expected_latency_ms: float
    reason: str
    created_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class CacheAllocation:
    request_id: str
    allocated_tokens: int
    cache_key: str
    created_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class RuntimeSnapshot:
    runtime_mode: RuntimeMode
    queue_size: int
    active_batches: int
    avg_latency_ms: float
    p95_latency_ms: float
    throughput_tokens_per_second: float
    gpu_utilization: float
    cache_hit_rate: float
    timestamp: float = field(default_factory=time.time)