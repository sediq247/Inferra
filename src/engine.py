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


class Engine:
    """
    Core Inferra orchestration engine.

    Responsibilities:
    - Scheduling
    - Batch execution
    - Adaptive optimization coordination
    - Predictive feedback loop
    """

    def __init__(self) -> None:
        self.state = EngineState()
        self._task: Optional[asyncio.Task] = None

    # -------------------------
    # Public API
    # -------------------------

    async def submit(self, prompt: str, max_tokens: int, **kwargs) -> str:
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

        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self.state.running = False

        if self._task:
            self._task.cancel()

        logger.info("engine_stopped")

    # -------------------------
    # Main Runtime Loop
    # -------------------------

    async def _run_loop(self) -> None:
        while self.state.running:
            try:
                with profiler.profile_section("engine.cycle"):

                    batch = await scheduler.build_next_batch()

                    if batch is None:
                        await asyncio.sleep(0.01)
                        continue

                    # Execute batch
                    result = await executor.execute_batch(batch)

                    # Collect runtime stats
                    queue_size = await scheduler.queue.size()

                    avg_latency = result.total_latency_ms
                    tokens_per_sec = result.tokens_per_second

                    # Snapshot for predictor
                    snapshot = RuntimeSnapshot(
                        runtime_mode=mode_switcher.get_mode(),
                        queue_size=queue_size,
                        active_batches=1,
                        avg_latency_ms=avg_latency,
                        p95_latency_ms=avg_latency,  # approximation
                        throughput_tokens_per_second=tokens_per_sec,
                        gpu_utilization=0.0,  # placeholder (future hook)
                        cache_hit_rate=0.0,   # placeholder (kv_cache extension)
                    )

                    # Feed predictor
                    predictor.update(snapshot)

                    prediction = predictor.predict()

                    # Feed optimizer
                    optimizer.observe(
                        queue_size=queue_size,
                        avg_latency_ms=avg_latency,
                        tokens_per_second=tokens_per_sec,
                    )

                    decision = optimizer.compute()

                    # Mode switching (control plane)
                    mode_switcher.evaluate(
                        queue_size=queue_size,
                        avg_latency_ms=avg_latency,
                        tokens_per_second=tokens_per_sec,
                    )

                    # Metrics update
                    metrics.record_request(
                        latency_ms=avg_latency,
                        generated_tokens=int(tokens_per_sec * (avg_latency / 1000)),
                    )

                    self.state.last_cycle_time = time.time()

                    logger.info(
                        "engine_cycle_completed",
                        queue_size=queue_size,
                        latency_ms=avg_latency,
                        tokens_per_sec=tokens_per_sec,
                        predicted_queue=prediction.predicted_queue_size,
                        mode=mode_switcher.get_mode().value,
                        batch_size=decision.recommended_batch_size,
                    )

            except asyncio.CancelledError:
                break

            except Exception as e:
                logger.exception("engine_cycle_error", error=str(e))
                await asyncio.sleep(0.1)

    # -------------------------
    # Debug / Introspection
    # -------------------------

    async def status(self) -> dict:
        queue_size = await scheduler.queue.size()

        return {
            "running": self.state.running,
            "queue_size": queue_size,
            "mode": mode_switcher.get_mode().value,
            "last_cycle_time": self.state.last_cycle_time,
        }


engine = Engine()