from __future__ import annotations

import time from dataclasses import dataclass from typing import Optional

from config import settings from datamodels import RuntimeMode from metrics import metrics from utils import logger

@dataclass(slots=True) class ModeState: current_mode: RuntimeMode last_switch_time: float reason: str = "init"

class ModeSwitcher: """ Adaptive runtime mode controller.

Switches between LATENCY and THROUGHPUT modes based on:
- queue pressure
- latency trends
- throughput signals
"""

def __init__(self) -> None:
    self.state = ModeState(
        current_mode=RuntimeMode.LATENCY,
        last_switch_time=time.time(),
        reason="initialization",
    )

    self.min_switch_interval = (
        settings.execution.mode_switch_cooldown_s
    )

    self.high_queue_threshold = (
        settings.scheduler.mode_switch_queue_high
    )

    self.low_queue_threshold = (
        settings.scheduler.mode_switch_queue_low
    )

    self.high_latency_ms = (
        settings.execution.high_latency_ms_threshold
    )

def evaluate(
    self,
    queue_size: int,
    avg_latency_ms: float,
    tokens_per_second: float,
) -> RuntimeMode:
    now = time.time()

    # cooldown protection
    if now - self.state.last_switch_time < self.min_switch_interval:
        return self.state.current_mode

    new_mode = self.state.current_mode
    reason = "stable"

    # HIGH LOAD → throughput mode
    if queue_size >= self.high_queue_threshold:
        new_mode = RuntimeMode.THROUGHPUT
        reason = "high_queue_pressure"

    # LOW LOAD → latency mode
    elif queue_size <= self.low_queue_threshold:
        new_mode = RuntimeMode.LATENCY
        reason = "low_queue_pressure"

    # latency spike → throughput mode
    elif avg_latency_ms > self.high_latency_ms:
        new_mode = RuntimeMode.THROUGHPUT
        reason = "high_latency_detected"

    # good latency + low queue → latency mode
    elif (
        avg_latency_ms < self.high_latency_ms * 0.5
        and queue_size < self.high_queue_threshold
    ):
        new_mode = RuntimeMode.LATENCY
        reason = "optimal_latency_conditions"

    if new_mode != self.state.current_mode:
        self.state = ModeState(
            current_mode=new_mode,
            last_switch_time=now,
            reason=reason,
        )

        logger.info(
            "runtime_mode_switched",
            new_mode=new_mode.value,
            reason=reason,
            queue_size=queue_size,
            avg_latency_ms=avg_latency_ms,
            tokens_per_second=tokens_per_second,
        )

    return self.state.current_mode

def get_mode(self) -> RuntimeMode:
    return self.state.current_mode

def force_mode(self, mode: RuntimeMode, reason: str = "manual") -> None:
    self.state = ModeState(
        current_mode=mode,
        last_switch_time=time.time(),
        reason=reason,
    )

    logger.warning(
        "runtime_mode_forced",
        mode=mode.value,
        reason=reason,
    )

mode_switcher = ModeSwitcher()