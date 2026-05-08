from __future__ import annotations

import asyncio
import time
import json
import random
from dataclasses import dataclass
from typing import List, Dict, Any

from engine import engine
from utils import logger
from metrics import metrics


@dataclass
class ReplayConfig:
    trace_file: str | None = None
    replay_speed: float = 1.0   # 2.0 = twice as fast, 0.5 = slower
    max_tokens: int = 256
    deterministic: bool = True


class ReplayEngine:
    """
    Deterministic workload replay system.

    Purpose:
    - reproduce identical workloads
    - validate optimizer changes
    - compare benchmark runs fairly
    """

    def __init__(self, config: ReplayConfig) -> None:
        self.config = config
        self.trace: List[Dict[str, Any]] = []

    async def run(self) -> dict:
        logger.info("replay_started", config=self.config.__dict__)

        await engine.start()

        self._load_trace()

        start_time = time.perf_counter()
        await self._replay()
        total_time = time.perf_counter() - start_time

        await asyncio.sleep(2)
        await engine.stop()

        results = self._collect(total_time)

        logger.info("replay_completed", results=results)

        return results

    # -----------------------
    # Core replay logic
    # -----------------------

    async def _replay(self) -> None:
        if not self.trace:
            logger.warning("empty_trace")
            return

        prev_time = self.trace[0]["timestamp"]

        for item in self.trace:
            ts = item["timestamp"]
            prompt = item["prompt"]

            # simulate time gap
            delay = (ts - prev_time) / self.config.replay_speed
            prev_time = ts

            if delay > 0:
                await asyncio.sleep(delay)

            await self._send(prompt)

    async def _send(self, prompt: str) -> None:
        try:
            await engine.submit(
                prompt=prompt,
                max_tokens=self.config.max_tokens,
            )
        except Exception as e:
            logger.warning("replay_submit_failed", error=str(e))

    # -----------------------
    # Trace handling
    # -----------------------

    def _load_trace(self) -> None:
        if self.config.trace_file:
            with open(self.config.trace_file, "r") as f:
                self.trace = json.load(f)
        else:
            # fallback synthetic deterministic trace
            self.trace = self._generate_trace()

        logger.info(
            "trace_loaded",
            size=len(self.trace),
        )

    def _generate_trace(self) -> List[Dict[str, Any]]:
        random.seed(42)

        words = [
            "inferra", "replay", "benchmark", "optimizer",
            "batch", "latency", "throughput", "engine"
        ]

        trace = []
        t = 0.0

        for i in range(200):
            t += random.uniform(0.05, 0.2)

            trace.append({
                "timestamp": t,
                "prompt": " ".join(random.choice(words) for _ in range(20))
            })

        return trace

    # -----------------------
    # Results
    # -----------------------

    def _collect(self, duration: float) -> dict:
        return {
            "duration_s": duration,
            "requests": len(self.trace),
            "avg_latency_ms": metrics.get_avg_latency(),
            "throughput_tokens_per_s": metrics.get_throughput(),
            "replay_speed": self.config.replay_speed,
        }


# -----------------------
# CLI entry
# -----------------------

async def main() -> None:
    config = ReplayConfig(
        trace_file=None,   # or provide JSON trace path
        replay_speed=1.0,
    )

    runner = ReplayEngine(config)
    results = await runner.run()

    print("\n=== REPLAY RESULTS ===")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    asyncio.run(main())