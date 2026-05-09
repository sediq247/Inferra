from __future__ import annotations

import asyncio
import shutil
import time
from collections import deque
from statistics import mean

from engine import engine
from kv_cache import kv_cache
from metrics import metrics
from mode_switcher import mode_switcher
from optimizer import optimizer
from utils import logger


class Dashboard:
    """
    Live runtime observability dashboard.

    Features:
    - live throughput monitoring
    - latency tracking
    - runtime pressure visibility
    - KV cache telemetry
    - adaptive runtime state visibility
    - benchmark-oriented terminal streaming
    """

    def __init__(
        self,
        refresh_interval_s: float = 1.0,
        history_size: int = 30,
    ) -> None:
        self.refresh_interval_s = refresh_interval_s

        self._task: asyncio.Task | None = None
        self._running = False

        self._tps_history = deque(
            maxlen=history_size,
        )

        self._latency_history = deque(
            maxlen=history_size,
        )

        self._pressure_history = deque(
            maxlen=history_size,
        )

    # ---------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return

        self._running = True

        logger.info(
            "dashboard_started",
        )

        self._task = asyncio.create_task(
            self._loop(),
        )

    async def stop(self) -> None:
        self._running = False

        if self._task:
            self._task.cancel()

            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info(
            "dashboard_stopped",
        )

    # ---------------------------------------------------------
    # Main Loop
    # ---------------------------------------------------------

    async def _loop(self) -> None:
        while self._running:
            try:
                await self.render()

                await asyncio.sleep(
                    self.refresh_interval_s,
                )

            except asyncio.CancelledError:
                break

            except Exception as error:
                logger.exception(
                    "dashboard_render_error",
                    error=str(error),
                )

                await asyncio.sleep(0.5)

    # ---------------------------------------------------------
    # Rendering
    # ---------------------------------------------------------

    async def render(self) -> None:
        snapshot = metrics.snapshot()

        engine_status = await engine.status()

        queue_size = snapshot.queue_size

        runtime_mode = (
            mode_switcher.get_mode().value
        )

        kv_usage = kv_cache.get_usage()

        tps = (
            snapshot.throughput
            .tokens_per_second
        )

        latency = (
            snapshot.latency.p95_ms
        )

        pressure = (
            snapshot.runtime_pressure_score
        )

        efficiency = (
            snapshot.runtime_efficiency_score
        )

        stability = (
            snapshot.throughput
            .throughput_stability
        )

        self._tps_history.append(tps)
        self._latency_history.append(latency)
        self._pressure_history.append(
            pressure,
        )

        self._clear_terminal()

        print(
            self._header(),
        )

        print(
            self._runtime_section(
                runtime_mode,
                queue_size,
                snapshot.active_batches,
            )
        )

        print(
            self._performance_section(
                tps=tps,
                latency=latency,
                efficiency=efficiency,
                stability=stability,
            )
        )

        print(
            self._batching_section(
                snapshot.avg_batch_size,
                snapshot.padding_efficiency,
            )
        )

        print(
            self._cache_section(
                kv_usage,
                snapshot.cache_hit_rate,
            )
        )

        print(
            self._pressure_section(
                pressure,
            )
        )

        print(
            self._graphs_section(),
        )

        print(
            self._footer(
                engine_status,
            )
        )

    # ---------------------------------------------------------
    # Sections
    # ---------------------------------------------------------

    def _header(self) -> str:
        return (
            "\n"
            "=====================================================\n"
            "                INFERA LIVE DASHBOARD                \n"
            "=====================================================\n"
        )

    def _runtime_section(
        self,
        runtime_mode: str,
        queue_size: int,
        active_batches: int,
    ) -> str:
        return (
            "[ Runtime ]\n"
            f"mode               : {runtime_mode}\n"
            f"queue_size         : {queue_size}\n"
            f"active_batches     : {active_batches}\n"
        )

    def _performance_section(
        self,
        tps: float,
        latency: float,
        efficiency: float,
        stability: float,
    ) -> str:
        return (
            "\n"
            "[ Performance ]\n"
            f"tokens/sec         : {tps:.2f}\n"
            f"p95_latency_ms     : {latency:.2f}\n"
            f"runtime_efficiency : {efficiency:.4f}\n"
            f"throughput_stab.   : {stability:.4f}\n"
        )

    def _batching_section(
        self,
        avg_batch_size: float,
        padding_efficiency: float,
    ) -> str:
        return (
            "\n"
            "[ Batching ]\n"
            f"avg_batch_size     : {avg_batch_size:.2f}\n"
            f"padding_efficiency : {padding_efficiency:.4f}\n"
        )

    def _cache_section(
        self,
        kv_usage: dict,
        cache_hit_rate: float,
    ) -> str:
        return (
            "\n"
            "[ KV Cache ]\n"
            f"cache_hit_rate     : {cache_hit_rate:.4f}\n"
            f"cached_tokens      : "
            f"{kv_usage.get('total_tokens', 0)}\n"
            f"cache_utilization  : "
            f"{kv_usage.get('utilization', 0):.4f}\n"
        )

    def _pressure_section(
        self,
        pressure: float,
    ) -> str:
        return (
            "\n"
            "[ Runtime Pressure ]\n"
            f"pressure_score     : {pressure:.2f}\n"
        )

    def _graphs_section(self) -> str:
        tps_graph = self._sparkline(
            self._tps_history,
        )

        latency_graph = self._sparkline(
            self._latency_history,
        )

        pressure_graph = self._sparkline(
            self._pressure_history,
        )

        return (
            "\n"
            "[ Live Graphs ]\n"
            f"TPS        : {tps_graph}\n"
            f"Latency    : {latency_graph}\n"
            f"Pressure   : {pressure_graph}\n"
        )

    def _footer(
        self,
        engine_status: dict,
    ) -> str:
        return (
            "\n"
            "-----------------------------------------------------\n"
            f"completed_cycles  : "
            f"{engine_status.get('completed_cycles', 0)}\n"
            f"last_cycle_time   : "
            f"{engine_status.get('last_cycle_time', 0):.2f}\n"
            "=====================================================\n"
        )

    # ---------------------------------------------------------
    # Visualization Helpers
    # ---------------------------------------------------------

    @staticmethod
    def _sparkline(
        values: deque[float],
    ) -> str:
        if not values:
            return ""

        ticks = "▁▂▃▄▅▆▇█"

        minimum = min(values)
        maximum = max(values)

        if maximum == minimum:
            return ticks[0] * len(values)

        normalized = [
            int(
                (
                    (value - minimum)
                    / (maximum - minimum)
                )
                * (len(ticks) - 1)
            )
            for value in values
        ]

        return "".join(
            ticks[index]
            for index in normalized
        )

    @staticmethod
    def _clear_terminal() -> None:
        print(
            "\033[2J\033[H",
            end="",
        )


# ---------------------------------------------------------
# Global Singleton
# ---------------------------------------------------------

_dashboard = Dashboard()


def get_dashboard() -> Dashboard:
    return _dashboard

if __name__ == "__main__":

    async def main() -> None:
        dashboard = get_dashboard()

        await dashboard.start()

        try:
            while True:
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            pass

        finally:
            await dashboard.stop()

    asyncio.run(main())