from __future__ import annotations

import math
import time
from dataclasses import dataclass

from config import settings
from datamodels import (
    OptimizerDecision,
    RuntimeMode,
)
from mode_switcher import mode_switcher
from utils import (
    MovingAverage,
    logger,
)


@dataclass(slots=True)
class OptimizationState:
    avg_latency_ms: float = 0.0
    queue_pressure: float = 0.0
    throughput_tokens_per_second: float = 0.0
    pressure_score: float = 0.0


class Optimizer:
    """
    Adaptive optimization controller.

    Responsibilities:
    - dynamic batch scaling
    - adaptive token budgeting
    - runtime pressure analysis
    - throughput/latency balancing
    """

    def __init__(self) -> None:
        self.latency_ma = MovingAverage(
            window_size=50,
        )

        self.tps_ma = MovingAverage(
            window_size=50,
        )

        self.queue_ma = MovingAverage(
            window_size=50,
        )

        self.pressure_ma = MovingAverage(
            window_size=50,
        )

        self.base_batch_size = (
            settings.scheduler.max_batch_size
        )

        self.base_batch_tokens = (
            settings.scheduler.max_batch_tokens
        )

        self.min_batch_size = 1
        self.max_batch_size = (
            self.base_batch_size * 4
        )

        self.min_batch_tokens = 256
        self.max_batch_tokens = (
            self.base_batch_tokens * 4
        )

        self.last_decision_time = time.time()

        self.optimizer_tick_s = (
            settings.execution.optimizer_tick_s
        )

    # ---------------------------------------------------------
    # Runtime Observation
    # ---------------------------------------------------------

    def observe(
        self,
        queue_size: int,
        avg_latency_ms: float,
        tokens_per_second: float,
    ) -> None:
        pressure_score = (
            queue_size * avg_latency_ms
        )

        self.queue_ma.add(
            float(queue_size),
        )

        self.latency_ma.add(
            avg_latency_ms,
        )

        self.tps_ma.add(
            tokens_per_second,
        )

        self.pressure_ma.add(
            pressure_score,
        )

    # ---------------------------------------------------------
    # Optimization Logic
    # ---------------------------------------------------------

    def compute(self) -> OptimizerDecision:
        now = time.time()

        if (
            now - self.last_decision_time
            < self.optimizer_tick_s
        ):
            runtime_mode = (
                mode_switcher.get_mode()
            )

            return OptimizerDecision(
                runtime_mode=runtime_mode,
                recommended_batch_size=(
                    self.base_batch_size
                ),
                recommended_batch_tokens=(
                    self.base_batch_tokens
                ),
                queue_pressure=0.0,
                expected_latency_ms=(
                    self.latency_ma.average
                ),
                reason="optimizer_tick_interval",
            )

        self.last_decision_time = now

        avg_latency = max(
            self.latency_ma.average,
            1.0,
        )

        avg_tps = max(
            self.tps_ma.average,
            1.0,
        )

        avg_queue = max(
            self.queue_ma.average,
            0.0,
        )

        pressure_score = (
            self.pressure_ma.average
        )

        queue_pressure = (
            avg_queue
            / max(self.base_batch_size, 1)
        )

        runtime_mode = (
            mode_switcher.get_mode()
        )

        # ---------------------------------------------------------
        # Dynamic scaling model
        # ---------------------------------------------------------

        throughput_factor = math.log1p(
            avg_tps,
        )

        latency_penalty = min(
            avg_latency / 2000.0,
            2.0,
        )

        pressure_factor = min(
            queue_pressure,
            4.0,
        )

        if runtime_mode == RuntimeMode.THROUGHPUT:
            batch_scale = (
                1.0
                + (pressure_factor * 0.8)
                + (throughput_factor * 0.15)
            )

            token_scale = (
                1.0
                + (pressure_factor * 0.9)
            )

        else:
            batch_scale = max(
                0.5,
                1.0 - (latency_penalty * 0.35),
            )

            token_scale = max(
                0.75,
                1.0 - (latency_penalty * 0.25),
            )

        recommended_batch_size = int(
            min(
                self.max_batch_size,
                max(
                    self.min_batch_size,
                    round(
                        self.base_batch_size
                        * batch_scale,
                    ),
                ),
            )
        )

        recommended_batch_tokens = int(
            min(
                self.max_batch_tokens,
                max(
                    self.min_batch_tokens,
                    round(
                        self.base_batch_tokens
                        * token_scale,
                    ),
                ),
            )
        )

        expected_latency_ms = (
            avg_latency
            * (
                1.15
                if runtime_mode
                == RuntimeMode.THROUGHPUT
                else 0.92
            )
        )

        reason = self._explain(
            runtime_mode=runtime_mode,
            queue_pressure=queue_pressure,
            pressure_score=pressure_score,
            avg_latency=avg_latency,
            avg_tps=avg_tps,
        )

        decision = OptimizerDecision(
            runtime_mode=runtime_mode,
            recommended_batch_size=(
                recommended_batch_size
            ),
            recommended_batch_tokens=(
                recommended_batch_tokens
            ),
            queue_pressure=queue_pressure,
            expected_latency_ms=(
                expected_latency_ms
            ),
            reason=reason,
        )

        logger.info(
            "optimizer_decision",
            runtime_mode=runtime_mode.value,
            batch_size=recommended_batch_size,
            batch_tokens=(
                recommended_batch_tokens
            ),
            queue_pressure=round(
                queue_pressure,
                3,
            ),
            pressure_score=round(
                pressure_score,
                2,
            ),
            avg_latency_ms=round(
                avg_latency,
                2,
            ),
            avg_tokens_per_second=round(
                avg_tps,
                2,
            ),
        )

        return decision

    # ---------------------------------------------------------
    # Adaptive Candidate Pool
    # ---------------------------------------------------------

    def dynamic_candidate_pool(
        self,
        queue_size: int,
    ) -> int:
        pressure = (
            queue_size
            / max(self.base_batch_size, 1)
        )

        scale = min(
            max(1.0, pressure),
            4.0,
        )

        return int(
            self.base_batch_size * scale
        )

    # ---------------------------------------------------------
    # Reasoning Layer
    # ---------------------------------------------------------

    def _explain(
        self,
        runtime_mode: RuntimeMode,
        queue_pressure: float,
        pressure_score: float,
        avg_latency: float,
        avg_tps: float,
    ) -> str:
        if runtime_mode == RuntimeMode.THROUGHPUT:
            if pressure_score > 10000:
                return (
                    "extreme_runtime_pressure"
                )

            if queue_pressure > 2.0:
                return (
                    "queue_saturation_detected"
                )

            return (
                "throughput_optimization_active"
            )

        if avg_latency > 1200:
            return (
                "latency_reduction_priority"
            )

        if avg_tps < 20:
            return (
                "low_throughput_recovery_mode"
            )

        return (
            "balanced_runtime_mode"
        )


optimizer = Optimizer()