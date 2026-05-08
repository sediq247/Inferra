from future import annotations

import asyncio import random import time from dataclasses import dataclass

from engine import engine from metrics import metrics from utils import logger

@dataclass class StressConfig: duration_s: int = 30 start_concurrency: int = 5 max_concurrency: int = 50 ramp_step: int = 5 max_tokens: int = 256 prompt_len: int = 40

class StressTest: """ Stress test for Inferra runtime.

Goal:
- push scheduler + batching + executor to saturation
- observe stability under increasing load
"""

def __init__(self, config: StressConfig) -> None:
    self.config = config
    self._start_time = 0.0

async def run(self) -> dict:
    logger.info("stress_test_started", config=self.config.__dict__)

    await engine.start()

    self._start_time = time.perf_counter()

    tasks = []
    current_concurrency = self.config.start_concurrency

    end_time = self._start_time + self.config.duration_s

    while time.perf_counter() < end_time:
        # ramp concurrency slowly
        current_concurrency = min(
            self.config.max_concurrency,
            current_concurrency + self.config.ramp_step,
        )

        batch_tasks = []

        for _ in range(current_concurrency):
            batch_tasks.append(
                asyncio.create_task(self._send_request())
            )

        tasks.extend(batch_tasks)

        # small delay between waves
        await asyncio.sleep(0.2)

    await asyncio.gather(*tasks, return_exceptions=True)

    # cooldown drain
    await asyncio.sleep(3)

    await engine.stop()

    results = self._collect()

    logger.info("stress_test_completed", results=results)

    return results

async def _send_request(self) -> None:
    try:
        await engine.submit(
            prompt=self._random_prompt(),
            max_tokens=self.config.max_tokens,
        )
    except Exception as e:
        logger.warning("stress_request_failed", error=str(e))

def _collect(self) -> dict:
    total_time = time.perf_counter() - self._start_time

    return {
        "duration_s": total_time,
        "avg_latency_ms": metrics.get_avg_latency(),
        "throughput_tokens_per_s": metrics.get_throughput(),
        "note": "stress test completed under ramped concurrency",
    }

def _random_prompt(self) -> str:
    words = [
        "inferra",
        "stress",
        "load",
        "batch",
        "optimizer",
        "throughput",
        "latency",
    ]
    return " ".join(random.choice(words) for _ in range(self.config.prompt_len))

async def main() -> None: config = StressConfig() test = StressTest(config) results = await test.run()

print("\n=== STRESS TEST RESULTS ===")
for k, v in results.items():
    print(f"{k}: {v}")

if name == "main": asyncio.run(main())