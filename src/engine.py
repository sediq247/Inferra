from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

from scheduler import scheduler
from executor import executor
from optimizer import optimizer
from predictor import predictor
from mode_switcher import mode_switcher

from datamodels import RuntimeSnapshot
from metrics import metrics
from profiler import profiler
from utils import logger


@dataclass(slots=True)
class EngineState:
    running: bool = False
    last_cycle_time: float = 0.0
    completed_cycles: int = 0


class Engine:
    """
    Core Inferra orchestration runtime.

    Responsibilities:
    - adaptive scheduling
    - runtime optimization
    - execution coordination
    - predictive workload analysis
    - metrics feedback integration
    """

    def __init__(self) -> None:
        self.state = EngineState()

        self._task: Optional[asyncio.Task] = None

        self._idle_sleep_ms = 20
        self._active_sleep_ms = 2

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    async def submit(
        self,
        prompt: str,
        max_tokens: int,
        **kwargs,
    ) -> str:
        request = await scheduler.submit(
            prompt=prompt,
            max_tokens=max_tokens,
            **kwargs,
        )

        return request.request_id

    async def start(self) -> None:
        if self.state.running:
            return

        self.state.running = True

        logger.info("engine_started")

        self._task = asyncio.create_task(
            self._run_loop(),
        )

    async def stop(self) -> None:
        self.state.running = False

        if self._task:
            self._task.cancel()

            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("engine_stopped")

    # ---------------------------------------------------------
    # Main Runtime Loop
    # ---------------------------------------------------------

    async def _run_loop(self) -> None:
        while self.state.running:
            try:
                with profiler.profile_section(
                    "engine.cycle",
                ):
                    queue_size = (
                        await scheduler.queue.size()
                    )

                    batch = (
                        await scheduler.build_next_batch()
                    )

                    if batch is None:
                        idle_sleep = (
                            self._active_sleep_ms
                            if queue_size > 0
                            else self._idle_sleep_ms
                        )

                        await asyncio.sleep(
                            idle_sleep / 1000,
                        )

                        continue

                    runtime_mode = (
                        mode_switcher.get_mode()
                    )

                    executor.set_mode(
                        runtime_mode,
                    )

                    scheduler.max_candidate_pool = (
                        optimizer.dynamic_candidate_pool(
                            queue_size,
                        )
                    )

                    execution_result = (
                        await executor.execute_batch(
                            batch,
                        )
                    )

                    avg_latency = (
                        execution_result.total_latency_ms
                    )

                    tokens_per_sec = (
                        execution_result.tokens_per_second
                    )

                    generated_tokens = int(
                        tokens_per_sec
                        * (avg_latency / 1000)
                    )

                    metrics.record_request(
                        latency_ms=avg_latency,
                        generated_tokens=generated_tokens,
                    )

                    p95_latency = (
                        metrics.p95_latency()
                    )

                    throughput = (
                        metrics.throughput_tokens_per_second()
                    )

                    pressure_score = (
                        queue_size * avg_latency
                    )

                    snapshot = RuntimeSnapshot(
                        runtime_mode=runtime_mode,
                        queue_size=queue_size,
                        active_batches=1,
                        avg_latency_ms=avg_latency,
                        p95_latency_ms=p95_latency,
                        throughput_tokens_per_second=throughput,
                        gpu_utilization=0.0,
                        cache_hit_rate=0.0,
                    )

                    predictor.update(
                        snapshot,
                    )

                    prediction = (
                        predictor.predict()
                    )

                    optimizer.observe(
                        queue_size=queue_size,
                        avg_latency_ms=avg_latency,
                        tokens_per_second=throughput,
                    )

                    decision = (
                        optimizer.compute()
                    )

                    mode_switcher.evaluate(
                        queue_size=queue_size,
                        avg_latency_ms=avg_latency,
                        tokens_per_second=throughput,
                    )

                    if (
                        decision.recommended_batch_size
                        > 0
                    ):
                        scheduler.max_candidate_pool = (
                            decision.recommended_batch_size
                            * 4
                        )

                    self.state.last_cycle_time = (
                        time.time()
                    )

                    self.state.completed_cycles += 1

                    logger.info(
                        "engine_cycle_completed",
                        mode=runtime_mode.value,
                        queue_size=queue_size,
                        batch_id=batch.batch_id,
                        batch_size=batch.batch_size,
                        latency_ms=avg_latency,
                        p95_latency_ms=p95_latency,
                        throughput_tokens_per_second=throughput,
                        pressure_score=pressure_score,
                        predicted_queue_size=(
                            prediction.predicted_queue_size
                        ),
                        recommended_batch_size=(
                            decision.recommended_batch_size
                        ),
                    )

            except asyncio.CancelledError:
                break

            except Exception as error:
                logger.exception(
                    "engine_cycle_error",
                    error=str(error),
                )

                await asyncio.sleep(0.1)

    # ---------------------------------------------------------
    # Runtime Status
    # ---------------------------------------------------------

    async def status(self) -> dict:
        queue_size = (
            await scheduler.queue.size()
        )

        return {
            "running": self.state.running,
            "queue_size": queue_size,
            "runtime_mode": (
                mode_switcher.get_mode().value
            ),
            "completed_cycles": (
                self.state.completed_cycles
            ),
            "last_cycle_time": (
                self.state.last_cycle_time
            ),
            "throughput_tokens_per_second": (
                metrics.throughput_tokens_per_second()
            ),
            "p95_latency_ms": (
                metrics.p95_latency()
            ),
        }


engine = Engine()