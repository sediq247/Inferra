from __future__ import annotations

import asyncio
import time
import random
from dataclasses import dataclass
from typing import List, Dict, Any

from engine import engine
from metrics import metrics
from utils import logger


@dataclass
class EvalConfig:
    num_requests: int = 200
    max_tokens: int = 256
    prompt_length: int = 25
    seed: int = 42


class Evaluator:
    """
    Official evaluation runner for Inferra.

    Produces:
    - throughput score
    - latency score
    - final weighted score
    """

    def __init__(self, config: EvalConfig) -> None:
        self.config = config
        random.seed(config.seed)

        self.results: List[Dict[str, Any]] = []
        self.start_time: float = 0.0

    async def run(self) -> dict:
        logger.info("evaluation_started", config=self.config.__dict__)

        await engine.start()

        self.start_time = time.perf_counter()

        await self._run_requests()

        # drain pipeline
        await asyncio.sleep(2)

        await engine.stop()

        result = self._compute_score()

        logger.info("evaluation_completed", result=result)

        return result


    async def _run_requests(self) -> None:
        tasks = []

        for _ in range(self.config.num_requests):
            prompt = self._generate_prompt()

            tasks.append(
                asyncio.create_task(
                    engine.submit(
                        prompt=prompt,
                        max_tokens=self.config.max_tokens,
                    )
                )
            )

        await asyncio.gather(*tasks, return_exceptions=True)

    def _generate_prompt(self) -> str:
        words = [
            "inferra", "evaluation", "benchmark", "optimizer",
            "scheduler", "batching", "latency", "throughput"
        ]

        return " ".join(
            random.choice(words)
            for _ in range(self.config.prompt_length)
        )

    # -----------------------
    # scoring logic
    # -----------------------

    def _compute_score(self) -> dict:
        total_time = time.perf_counter() - self.start_time

        throughput = metrics.get_throughput()
        avg_latency = metrics.get_avg_latency()

        latency_score = 1000 / (avg_latency + 1e-6)
        throughput_score = throughput

        stability_score = max(0.0, 100 - abs(avg_latency - 500) / 10)

        final_score = (
            0.5 * throughput_score +
            0.3 * latency_score +
            0.2 * stability_score
        )

        return {
            "total_time_s": total_time,
            "throughput_tokens_per_s": throughput,
            "avg_latency_ms": avg_latency,
            "stability_score": stability_score,
            "final_score": final_score,
            "num_requests": self.config.num_requests,
        }


# -----------------------
# CLI entry
# -----------------------

async def main() -> None:
    config = EvalConfig(
        num_requests=300,
        max_tokens=256,
    )

    evaluator = Evaluator(config)
    result = await evaluator.run()

    print("\n=== FINAL EVALUATION RESULTS ===")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    asyncio.run(main())