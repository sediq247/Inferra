from future import annotations

from dataclasses import dataclass from typing import Iterable

from config import settings from datamodels import InferenceRequest, ScheduledBatch from profiler import profiler from utils import RequestIDGenerator, logger

@dataclass(slots=True) class BatchCostEstimate: total_tokens: int padded_tokens: int padding_efficiency: float estimated_memory_cost: int

class TokenBudgetManager: def init(self) -> None: self.max_batch_tokens = ( settings.scheduler.max_batch_tokens ) self.max_batch_size = ( settings.scheduler.max_batch_size )

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

class BatchBuilder: def init(self) -> None: self.token_budget = TokenBudgetManager()

def build(
    self,
    requests: Iterable[InferenceRequest],
) -> ScheduledBatch | None:
    with profiler.profile_section("batching.build"):
        sorted_requests = sorted(
            requests,
            key=lambda request: (
                request.estimated_total_tokens,
                request.arrival_time,
            ),
        )

        if not sorted_requests:
            return None

        selected_requests: list[InferenceRequest] = []
        total_tokens = 0

        for request in sorted_requests:
            request_tokens = (
                request.estimated_total_tokens
            )

            if not self.token_budget.can_fit(
                current_tokens=total_tokens,
                incoming_tokens=request_tokens,
                current_batch_size=len(selected_requests),
            ):
                continue

            selected_requests.append(request)
            total_tokens += request_tokens

        if not selected_requests:
            return None

        cost_estimate = self.estimate_cost(
            selected_requests,
        )

        batch = ScheduledBatch(
            batch_id=RequestIDGenerator.generate(),
            requests=selected_requests,
            total_estimated_tokens=(
                cost_estimate.total_tokens
            ),
            max_request_tokens=max(
                request.estimated_total_tokens
                for request in selected_requests
            ),
            padding_tokens=(
                cost_estimate.padded_tokens
            ),
        )

        logger.info(
            "batch_constructed",
            batch_id=batch.batch_id,
            batch_size=batch.batch_size,
            total_tokens=batch.total_estimated_tokens,
            padding_efficiency=(
                cost_estimate.padding_efficiency
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
            estimated_memory_cost=0,
        )

    total_tokens = sum(
        request.estimated_total_tokens
        for request in requests
    )

    max_tokens = max(
        request.estimated_total_tokens
        for request in requests
    )

    padded_total = max_tokens * len(requests)

    padding_tokens = max(0, padded_total - total_tokens)

    padding_efficiency = (
        total_tokens / padded_total
        if padded_total > 0
        else 1.0
    )

    estimated_memory_cost = self._estimate_memory_cost(
        padded_total,
    )

    return BatchCostEstimate(
        total_tokens=total_tokens,
        padded_tokens=padding_tokens,
        padding_efficiency=padding_efficiency,
        estimated_memory_cost=estimated_memory_cost,
    )

def split_oversized_requests(
    self,
    requests: Iterable[InferenceRequest],
) -> tuple[
    list[InferenceRequest],
    list[InferenceRequest],
]:
    accepted: list[InferenceRequest] = []
    rejected: list[InferenceRequest] = []

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
            request.priority.value,
            request.arrival_time,
        ),
    )

@staticmethod
def _estimate_memory_cost(
    padded_tokens: int,
) -> int:
    # Approximation placeholder.
    # Real GPU profiling can refine this later
    # without changing public interfaces.
    return padded_tokens * 2

batch_builder = BatchBuilder()