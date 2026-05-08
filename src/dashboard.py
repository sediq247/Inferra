from future import annotations

import asyncio import time

from engine import engine from optimizer import optimizer from kv_cache import kv_cache from mode_switcher import mode_switcher from utils import logger

class Dashboard: """ Lightweight live observability dashboard.

Prints runtime system state at a fixed interval.
Designed for CLI / benchmarking environments.
"""

def __init__(self, refresh_interval_s: float = 1.0) -> None:
    self.refresh_interval_s = refresh_interval_s
    self._task: asyncio.Task | None = None
    self._running = False

async def start(self) -> None:
    if self._running:
        return

    self._running = True
    logger.info("dashboard_started")

    self._task = asyncio.create_task(self._loop())

async def stop(self) -> None:
    self._running = False

    if self._task:
        self._task.cancel()

    logger.info("dashboard_stopped")

async def _loop(self) -> None:
    while self._running:
        try:
            await self.render()
            await asyncio.sleep(self.refresh_interval_s)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.exception("dashboard_error", error=str(e))
            await asyncio.sleep(0.5)

async def render(self) -> None:
    engine_status = await engine.status()
    queue_size = engine_status.get("queue_size", 0)
    mode = engine_status.get("mode", "unknown")

    kv_usage = kv_cache.get_usage()

    latency = optimizer.latency_ma.average
    tps = optimizer.tps_ma.average
    queue_pressure = optimizer.queue_ma.average

    line = (
        f"[Inferra] "
        f"mode={mode} | "
        f"queue={queue_size} | "
        f"lat={latency:.2f}ms | "
        f"tps={tps:.2f} | "
        f"q_pressure={queue_pressure:.2f} | "
        f"kv_tokens={kv_usage.get('total_tokens', 0)} | "
        f"kv_util={kv_usage.get('utilization', 0):.2f}"
    )

    print(line)

global singleton

_dashboard = Dashboard()

def get_dashboard() -> Dashboard: return _dashboard