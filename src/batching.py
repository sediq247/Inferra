from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from config import settings
from datamodels import (
    InferenceRequest,
    ScheduledBatch,
)
from profiler import profiler
from utils import (
    RequestIDGenerator,
    logger,
)


@dataclass(slots=True)
class BatchCostEstimate:
    total_tokens: int
    padded_tokens: int
    padding_efficiency: float
    estimated_memory_cost_mb: float


class TokenBudgetManager:
    def __init__(self) -> None:
        self.max_batch_tokens = (
            settings.scheduler.max_batch_tokens
        )

        self.max_batch_size = (
            settings.scheduler.max_batch_size
        )

    def can_fit(
        self,
        current_tokens: int,
        incoming_tokens: int,
        current_batch_size: int,
    ) -> bool:
        if current_batch_size >= self.max_batch_size:
            return False

        return (
            current_tokens + incoming_tokens
            <= self.max_batch_tokens
        )


class BatchBuilder:
    def __init__(self) -> None:
        self.token_budget = TokenBudgetManager()

    def build(
        self,
        requests: Iterable[InferenceRequest],
    ) -> ScheduledBatch | None:
        with profiler.profile_section(
            "batching.build",
        ):
            ordered_requests = self.optimize_ordering(
                requests,
            )

            if not ordered_requests:
                return None

            selected: list[InferenceRequest] = []

            current_tokens = 0

            for request in ordered_requests:
                estimated_tokens = (
                    request.estimated_total_tokens
                )

                if not self.token_budget.can_fit(
                    current_tokens=current_tokens,
                    incoming_tokens=estimated_tokens,
                    current_batch_size=len(selected),
                ):
                    continue

                selected.append(request)

                current_tokens += estimated_tokens

            if not selected:
                return None

            estimate = self.estimate_cost(
                selected,
            )

            batch = ScheduledBatch(
                batch_id=RequestIDGenerator.generate(),
                requests=selected,
                total_estimated_tokens=(
                    estimate.total_tokens
                ),
                max_request_tokens=max(
                    r.estimated_total_tokens
                    for r in selected
                ),
                padding_tokens=(
                    estimate.padded_tokens
                ),
            )

            logger.info(
                "batch_constructed",
                batch_id=batch.batch_id,
                batch_size=batch.batch_size,
                total_tokens=batch.total_estimated_tokens,
                padding_efficiency=(
                    estimate.padding_efficiency
                ),
            )

            return batch

    def estimate_cost(
        self,
        requests: list[InferenceRequest],
    ) -> BatchCostEstimate:
        if not requests:
            return BatchCostEstimate(
                total_tokens=0,
                padded_tokens=0,
                padding_efficiency=1.0,
                estimated_memory_cost_mb=0.0,
            )

        total_tokens = sum(
            r.estimated_total_tokens
            for r in requests
        )

        max_tokens = max(
            r.estimated_total_tokens
            for r in requests
        )

        padded_total = (
            max_tokens * len(requests)
        )

        padding_tokens = max(
            0,
            padded_total - total_tokens,
        )

        padding_efficiency = (
            total_tokens / padded_total
            if padded_total > 0
            else 1.0
        )

        estimated_memory_cost_mb = (
            self._estimate_memory_cost_mb(
                padded_total,
            )
        )

        return BatchCostEstimate(
            total_tokens=total_tokens,
            padded_tokens=padding_tokens,
            padding_efficiency=padding_efficiency,
            estimated_memory_cost_mb=(
                estimated_memory_cost_mb
            ),
        )

    def split_oversized_requests(
        self,
        requests: Iterable[InferenceRequest],
    ) -> tuple[
        list[InferenceRequest],
        list[InferenceRequest],
    ]:
        accepted: list[
            InferenceRequest
        ] = []

        rejected: list[
            InferenceRequest
        ] = []

        for request in requests:
            if (
                request.estimated_total_tokens
                > self.token_budget.max_batch_tokens
            ):
                rejected.append(request)
            else:
                accepted.append(request)

        return accepted, rejected

    def optimize_ordering(
        self,
        requests: Iterable[InferenceRequest],
    ) -> list[InferenceRequest]:
        return sorted(
            requests,
            key=lambda request: (
                request.estimated_total_tokens,
                -request.priority.value,
                request.arrival_time,
            ),
        )

    @staticmethod
    def _estimate_memory_cost_mb(
        padded_tokens: int,
    ) -> float:
        # lightweight approximation:
        # token count scaled into rough KV/cache usage
        return round(
            padded_tokens * 0.0025,
            2,
        )


batch_builder = BatchBuilder()