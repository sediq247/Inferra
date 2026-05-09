from __future__ import annotations

import asyncio
import random
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any

from engine import engine
from metrics import metrics
from utils import logger


@dataclass(slots=True)
class BenchmarkConfig:
    num_requests: int = 500
    max_tokens: int = 256

    concurrency: int = 32

    prompt_length: int = 64

    warmup_requests: int = 25
    drain_timeout_s: float = 5.0

    benchmark_name: str = "inferra_runtime_benchmark"


@dataclass(slots=True)
class BenchmarkResult:
    benchmark_name: str

    total_requests: int
    completed_requests: int

    total_runtime_s: float

    requests_per_second: float
    tokens_per_second: float

    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    avg_batch_size: float
    padding_efficiency: float

    throughput_stability: float
    runtime_efficiency_score: float
    runtime_pressure_score: float

    cache_hit_rate: float


class BenchmarkRunner:
    """
    Competition-grade Inferra benchmark runner.

    Measures:
    - throughput
    - latency
    - runtime efficiency
    - batching quality
    - runtime stability
    """

    def __init__(
        self,
        config: BenchmarkConfig,
    ) -> None:
        self.config = config

        self.start_time = 0.0
        self.end_time = 0.0

        self.request_latencies: list[float] = []

    async def run(self) -> BenchmarkResult:
        logger.info(
            "benchmark_started",
            config=asdict(self.config),
        )

        await engine.start()

        try:
            await self._warmup()

            self.start_time = time.perf_counter()

            await self._generate_load()

            await self._wait_for_drain()

            self.end_time = time.perf_counter()

            result = self._collect_results()

            logger.info(
                "benchmark_completed",
                results=asdict(result),
            )

            return result

        finally:
            await engine.stop()

    # ---------------------------------------------------------
    # Warmup
    # ---------------------------------------------------------

    async def _warmup(self) -> None:
        logger.info(
            "benchmark_warmup_started",
            requests=self.config.warmup_requests,
        )

        tasks = [
            engine.submit(
                prompt=self._random_prompt(),
                max_tokens=self.config.max_tokens,
            )
            for _ in range(
                self.config.warmup_requests
            )
        ]

        await asyncio.gather(*tasks)

        await asyncio.sleep(1)

        logger.info(
            "benchmark_warmup_completed",
        )

    # ---------------------------------------------------------
    # Load Generation
    # ---------------------------------------------------------

    async def _generate_load(self) -> None:
        logger.info(
            "benchmark_load_generation_started",
            requests=self.config.num_requests,
            concurrency=self.config.concurrency,
        )

        semaphore = asyncio.Semaphore(
            self.config.concurrency,
        )

        async def worker(
            request_index: int,
        ) -> None:
            async with semaphore:
                started = time.perf_counter()

                await engine.submit(
                    prompt=self._random_prompt(),
                    max_tokens=self.config.max_tokens,
                )

                latency_ms = (
                    time.perf_counter()
                    - started
                ) * 1000

                self.request_latencies.append(
                    latency_ms,
                )

        tasks = [
            asyncio.create_task(
                worker(i),
            )
            for i in range(
                self.config.num_requests
            )
        ]

        await asyncio.gather(*tasks)

        logger.info(
            "benchmark_load_generation_completed",
        )

    # ---------------------------------------------------------
    # Drain Handling
    # ---------------------------------------------------------

    async def _wait_for_drain(
        self,
    ) -> None:
        logger.info(
            "benchmark_drain_wait_started",
        )

        deadline = (
            time.time()
            + self.config.drain_timeout_s
        )

        while time.time() < deadline:
            status = await engine.status()

            if status["queue_size"] == 0:
                break

            await asyncio.sleep(0.05)

        logger.info(
            "benchmark_drain_wait_completed",
        )

    # ---------------------------------------------------------
    # Result Collection
    # ---------------------------------------------------------

    def _collect_results(
        self,
    ) -> BenchmarkResult:
        runtime_s = max(
            self.end_time - self.start_time,
            0.001,
        )

        benchmark_summary = (
            metrics.benchmark_summary()
        )

        requests_per_second = (
            self.config.num_requests
            / runtime_s
        )

        return BenchmarkResult(
            benchmark_name=(
                self.config.benchmark_name
            ),
            total_requests=(
                self.config.num_requests
            ),
            completed_requests=(
                benchmark_summary["requests"]
            ),
            total_runtime_s=round(
                runtime_s,
                3,
            ),
            requests_per_second=round(
                requests_per_second,
                2,
            ),
            tokens_per_second=(
                benchmark_summary[
                    "tokens_per_second"
                ]
            ),
            avg_latency_ms=round(
                statistics.mean(
                    self.request_latencies,
                ),
                2,
            ),
            p95_latency_ms=(
                benchmark_summary[
                    "p95_latency_ms"
                ]
            ),
            p99_latency_ms=(
                benchmark_summary[
                    "p99_latency_ms"
                ]
            ),
            avg_batch_size=(
                benchmark_summary[
                    "avg_batch_size"
                ]
            ),
            padding_efficiency=(
                benchmark_summary[
                    "padding_efficiency"
                ]
            ),
            throughput_stability=(
                benchmark_summary[
                    "throughput_stability"
                ]
            ),
            runtime_efficiency_score=(
                benchmark_summary[
                    "runtime_efficiency_score"
                ]
            ),
            runtime_pressure_score=(
                benchmark_summary[
                    "runtime_pressure_score"
                ]
            ),
            cache_hit_rate=(
                benchmark_summary[
                    "cache_hit_rate"
                ]
            ),
        )

    # ---------------------------------------------------------
    # Prompt Generation
    # ---------------------------------------------------------

    def _random_prompt(self) -> str:
        vocabulary = [
            "inferra",
            "runtime",
            "adaptive",
            "optimization",
            "scheduler",
            "throughput",
            "latency",
            "qwen",
            "inference",
            "token",
            "batching",
            "predictive",
            "cache",
            "serving",
            "gpu",
            "executor",
        ]

        return " ".join(
            random.choices(
                vocabulary,
                k=self.config.prompt_length,
            )
        )


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

async def main() -> None:
    config = BenchmarkConfig(
        num_requests=500,
        concurrency=32,
        max_tokens=256,
    )

    runner = BenchmarkRunner(config)

    results = await runner.run()

    print("\n=== INFERA BENCHMARK RESULTS ===\n")

    for key, value in asdict(results).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())