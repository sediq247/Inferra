from __future__ import annotations

import asyncio
import heapq
from dataclasses import dataclass, field
from typing import Any

from batching import batch_builder
from config import settings
from datamodels import (
    InferenceRequest,
    RequestPriority,
    RequestStatus,
    ScheduledBatch,
)
from metrics import metrics
from profiler import profiler
from utils import logger


@dataclass(order=True, slots=True)
class QueueItem:
    sort_index: tuple[int, int, float]
    request: InferenceRequest = field(compare=False)


class RequestQueue:
    def __init__(self) -> None:
        self._queue: list[QueueItem] = []
        self._lock = asyncio.Lock()

    async def put(self, request: InferenceRequest) -> None:
        async with self._lock:
            # higher priority -> lower numeric value wins heap
            priority_score = -request.priority.value

            item = QueueItem(
                sort_index=(
                    priority_score,
                    request.estimated_total_tokens,
                    request.arrival_time,
                ),
                request=request,
            )

            heapq.heappush(self._queue, item)
            metrics.set_queue_size(len(self._queue))

    async def pop_candidates(self, limit: int) -> list[InferenceRequest]:
        async with self._lock:
            selected: list[InferenceRequest] = []

            for _ in range(min(limit, len(self._queue))):
                item = heapq.heappop(self._queue)
                selected.append(item.request)

            return selected

    async def remove_requests(self, request_ids: set[str]) -> None:
        async with self._lock:
            self._queue = [
                item
                for item in self._queue
                if item.request.request_id not in request_ids
            ]

            heapq.heapify(self._queue)
            metrics.set_queue_size(len(self._queue))

    async def snapshot(self) -> list[InferenceRequest]:
        async with self._lock:
            return [item.request for item in self._queue]

    async def size(self) -> int:
        async with self._lock:
            return len(self._queue)

    async def is_empty(self) -> bool:
        return await self.size() == 0


class Scheduler:
    def __init__(self) -> None:
        self.queue = RequestQueue()

        self.queue_capacity = settings.scheduler.queue_capacity
        self.max_candidate_pool = settings.scheduler.max_batch_size * 4

    async def submit(
        self,
        prompt: str,
        max_tokens: int,
        priority: RequestPriority = RequestPriority.NORMAL,
        metadata: dict[str, Any] | None = None,
    ) -> InferenceRequest:
        if await self.queue.size() >= self.queue_capacity:
            raise RuntimeError("Scheduler queue capacity exceeded")

        request = InferenceRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            priority=priority,
            metadata=metadata or {},
        )

        await self.queue.put(request)

        logger.info(
            "request_submitted",
            request_id=request.request_id,
            priority=request.priority.name,
            estimated_total_tokens=request.estimated_total_tokens,
        )

        return request

    async def build_next_batch(self) -> ScheduledBatch | None:
        if await self.queue.size() == 0:
            return None

        with profiler.profile_section("scheduler.build_next_batch"):
            candidates = await self.queue.pop_candidates(
                self.max_candidate_pool
            )

            accepted, rejected = batch_builder.split_oversized_requests(
                candidates
            )

            if rejected:
                logger.warning(
                    "oversized_requests_detected",
                    rejected_count=len(rejected),
                )

            optimized = batch_builder.optimize_ordering(accepted)

            batch = batch_builder.build(optimized)

            if batch is None:
                return None

            await self.queue.remove_requests(
                {r.request_id for r in batch.requests}
            )

            for r in batch.requests:
                r.status = RequestStatus.SCHEDULED

            metrics.record_batch(
                batch_size=batch.batch_size,
                active_tokens=batch.total_estimated_tokens,
                padded_tokens=(
                    batch.total_estimated_tokens + batch.padding_tokens
                ),
            )

            logger.info(
                "batch_scheduled",
                batch_id=batch.batch_id,
                batch_size=batch.batch_size,
                total_estimated_tokens=batch.total_estimated_tokens,
                padding_efficiency=batch.padding_efficiency,
            )

            return batch

    async def wait_for_batch(
        self,
        poll_interval_ms: int | None = None,
    ) -> ScheduledBatch | None:
        interval = (
            poll_interval_ms or settings.scheduler.scheduler_tick_ms
        )

        while True:
            batch = await self.build_next_batch()
            if batch:
                return batch
            await asyncio.sleep(interval / 1000)

    async def queue_snapshot(self) -> list[dict[str, Any]]:
        snapshot = await self.queue.snapshot()

        return [
            {
                "request_id": r.request_id,
                "priority": r.priority.name,
                "status": r.status.value,
                "estimated_total_tokens": r.estimated_total_tokens,
                "arrival_time": r.arrival_time,
            }
            for r in snapshot
        ]


scheduler = Scheduler()