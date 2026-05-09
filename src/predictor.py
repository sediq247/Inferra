from __future__ import annotations

import time from dataclasses import dataclass from typing import Deque from collections import deque

from datamodels import RuntimeSnapshot from utils import MovingAverage, logger

@dataclass(slots=True) class PredictionResult: predicted_queue_size: float predicted_latency_ms: float predicted_tokens_per_second: float confidence: float horizon_s: float

class WorkloadPredictor: """ Lightweight runtime workload predictor.

Uses time-series smoothing to estimate near-future:
- queue size
- latency
- throughput

Designed for optimizer + mode_switcher guidance.
"""

def __init__(self, history_size: int = 100) -> None:
    self.history: Deque[RuntimeSnapshot] = deque(maxlen=history_size)

    self.queue_ma = MovingAverage(window_size=30)
    self.latency_ma = MovingAverage(window_size=30)
    self.tps_ma = MovingAverage(window_size=30)

    self.last_update_time = time.time()

def update(self, snapshot: RuntimeSnapshot) -> None:
    self.history.append(snapshot)

    self.queue_ma.add(snapshot.queue_size)
    self.latency_ma.add(snapshot.avg_latency_ms)
    self.tps_ma.add(snapshot.throughput_tokens_per_second)

    self.last_update_time = snapshot.timestamp

def predict(self, horizon_s: float = 5.0) -> PredictionResult:
    if len(self.history) < 2:
        return PredictionResult(
            predicted_queue_size=self.queue_ma.average,
            predicted_latency_ms=self.latency_ma.average,
            predicted_tokens_per_second=self.tps_ma.average,
            confidence=0.1,
            horizon_s=horizon_s,
        )

    # simple trend estimation (delta per second)
    recent = list(self.history)[-10:]

    queue_trend = self._trend(
        [s.queue_size for s in recent]
    )
    latency_trend = self._trend(
        [s.avg_latency_ms for s in recent]
    )
    tps_trend = self._trend(
        [s.throughput_tokens_per_second for s in recent]
    )

    predicted_queue = max(
        0.0,
        self.queue_ma.average + queue_trend * horizon_s,
    )

    predicted_latency = max(
        0.0,
        self.latency_ma.average + latency_trend * horizon_s,
    )

    predicted_tps = max(
        0.0,
        self.tps_ma.average + tps_trend * horizon_s,
    )

    confidence = self._compute_confidence()

    logger.debug(
        "workload_prediction",
        predicted_queue_size=predicted_queue,
        predicted_latency_ms=predicted_latency,
        predicted_tps=predicted_tps,
        confidence=confidence,
    )

    return PredictionResult(
        predicted_queue_size=predicted_queue,
        predicted_latency_ms=predicted_latency,
        predicted_tokens_per_second=predicted_tps,
        confidence=confidence,
        horizon_s=horizon_s,
    )

def _trend(self, values: list[float]) -> float:
    if len(values) < 2:
        return 0.0

    n = len(values)
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n

    numerator = 0.0
    denominator = 0.0

    for i, y in enumerate(values):
        dx = i - x_mean
        dy = y - y_mean
        numerator += dx * dy
        denominator += dx * dx

    if denominator == 0:
        return 0.0

    slope = numerator / denominator
    return slope

def _compute_confidence(self) -> float:
    if len(self.history) < 5:
        return 0.3

    variance = self._variance(
        [s.queue_size for s in self.history]
    )

    if variance == 0:
        return 0.95

    confidence = 1.0 / (1.0 + variance)
    return max(0.1, min(confidence, 0.95))

def _variance(self, values: list[float]) -> float:
    if not values:
        return 0.0

    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / len(values)

predictor = WorkloadPredictor()