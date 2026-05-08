from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import List

from engine import engine
from metrics import metrics
from utils import logger


@dataclass
class BenchmarkConfig:
    num_requests: int = 200
    max_tokens: int = 256
    concurrency: int = 10
    prompt_length: int = 50


class BenchmarkRunner:
    """
    Production-grade benchmark suite for Inferra.

    Measures:
    - throughput (tokens/sec)
    - latency (avg)
    - queue stability
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.start_time: float = 0.0

    async def run(self) -> dict:
        logger.info("benchmark_started", config=self.config.__dict__)

        await engine.start()

        self.start_time = time.perf_counter()

        # warmup phase
        await self._warmup()

        # load generation
        await self._generate_load()

        # allow system to drain
        await asyncio.sleep(2)

        await engine.stop()

        results = self._collect_results()

        logger.info("benchmark_completed", results=results)

        return results

    async def _warmup(self) -> None:
        logger.info("benchmark_warmup_started")

        tasks = [
            engine.submit(
                prompt=self._random_prompt(),
                max_tokens=self.config.max_tokens,
            )
            for _ in range(10)
        ]

        await asyncio.gather(*tasks)

        await asyncio.sleep(1)

    async def _generate_load(self) -> None:
        logger.info("benchmark_load_generation_started")

        sem = asyncio.Semaphore(self.config.concurrency)

        async def worker(_: int) -> None:
            async with sem:
                await engine.submit(
                    prompt=self._random_prompt(),
                    max_tokens=self.config.max_tokens,
                )

        tasks = [
            asyncio.create_task(worker(i))
            for i in range(self.config.num_requests)
        ]

        await asyncio.gather(*tasks)

    def _collect_results(self) -> dict:
        total_time = time.perf_counter() - self.start_time

        throughput = metrics.get_throughput()
        avg_latency = metrics.get_avg_latency()

        return {
            "total_time_s": total_time,
            "throughput_tokens_per_s": throughput,
            "avg_latency_ms": avg_latency,
            "num_requests": self.config.num_requests,
        }

    def _random_prompt(self) -> str:
        words = ["inferra", "benchmark", "optimization", "llm", "throughput"]
        return " ".join(random.choices(words, k=self.config.prompt_length))


async def main() -> None:
    config = BenchmarkConfig(
        num_requests=300,
        concurrency=20,
    )

    runner = BenchmarkRunner(config)
    results = await runner.run()

    print("\n=== BENCHMARK RESULTS ===")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    asyncio.run(main())