from future import annotations

import time from dataclasses import dataclass from typing import Optional

from config import settings from datamodels import OptimizerDecision, RuntimeMode from metrics import metrics from mode_switcher import mode_switcher from utils import MovingAverage, logger

@dataclass(slots=True) class OptimizationState: avg_latency_ms: float = 0.0 queue_pressure: float = 0.0 tokens_per_second: float = 0.0

class Optimizer: """ Core adaptive optimization brain.

Responsibilities:
- Translate runtime metrics into execution parameters
- Recommend batch size + token budget
- Coordinate with mode_switcher
"""

def __init__(self) -> None:
    self.latency_ma = MovingAverage(window_size=50)
    self.tps_ma = MovingAverage(window_size=50)
    self.queue_ma = MovingAverage(window_size=50)

    self.last_decision_time = time.time()

    self.min_adjust_interval_s = (
        settings.execution.optimizer_tick_s
    )

    self.base_batch_size = settings.scheduler.max_batch_size
    self.base_batch_tokens = settings.scheduler.max_batch_tokens

def observe(
    self,
    queue_size: int,
    avg_latency_ms: float,
    tokens_per_second: float,
) -> None:
    self.latency_ma.add(avg_latency_ms)
    self.tps_ma.add(tokens_per_second)
    self.queue_ma.add(float(queue_size))

def compute(self) -> OptimizerDecision:
    now = time.time()

    avg_latency = self.latency_ma.average
    avg_tps = self.tps_ma.average
    avg_queue = self.queue_ma.average

    queue_pressure = (
        avg_queue / max(self.base_batch_size, 1)
    )

    runtime_mode = mode_switcher.get_mode()

    # batch sizing logic
    if runtime_mode == RuntimeMode.THROUGHPUT:
        batch_multiplier = 1.5 + min(queue_pressure, 2.0)
    else:
        batch_multiplier = 1.0 - min(avg_latency / 2000.0, 0.5)

    recommended_batch_size = int(
        max(1, self.base_batch_size * batch_multiplier)
    )

    # token budget scaling
    if runtime_mode == RuntimeMode.THROUGHPUT:
        token_multiplier = 1.5 + queue_pressure
    else:
        token_multiplier = 1.0

    recommended_batch_tokens = int(
        max(256, self.base_batch_tokens * token_multiplier)
    )

    expected_latency_ms = avg_latency * (
        1.0 if runtime_mode == RuntimeMode.LATENCY else 1.2
    )

    reason = self._explain(
        runtime_mode,
        queue_pressure,
        avg_latency,
        avg_tps,
    )

    decision = OptimizerDecision(
        runtime_mode=runtime_mode,
        recommended_batch_size=recommended_batch_size,
        recommended_batch_tokens=recommended_batch_tokens,
        queue_pressure=queue_pressure,
        expected_latency_ms=expected_latency_ms,
        reason=reason,
    )

    logger.info(
        "optimizer_decision",
        runtime_mode=runtime_mode.value,
        batch_size=recommended_batch_size,
        batch_tokens=recommended_batch_tokens,
        queue_pressure=queue_pressure,
        avg_latency_ms=avg_latency,
    )

    return decision

def _explain(
    self,
    mode: RuntimeMode,
    queue_pressure: float,
    avg_latency: float,
    avg_tps: float,
) -> str:
    if mode == RuntimeMode.THROUGHPUT:
        if queue_pressure > 1.5:
            return "high_queue_pressure_throughput_mode"
        return "throughput_mode_default"

    if avg_latency > 1000:
        return "high_latency_latency_mode"

    if avg_tps < 10:
        return "low_throughput_latency_mode"

    return "balanced_latency_mode"

optimizer = Optimizer()