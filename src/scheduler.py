scheduler.py
src/scheduler.py

from future import annotations

import asyncio import heapq from dataclasses import dataclass, field from typing import Any

from batching import batch_builder from config import settings from datamodels import ( InferenceRequest, RequestPriority, RequestStatus, ScheduledBatch, ) from metrics import metrics from profiler import profiler from utils import logger

@dataclass(order=True, slots=True) class QueueItem: sort_index: tuple[int,int float] request: InferenceRequest = field(compare=False)

class RequestQueue: def init(self) -> None: self._queue: list[QueueItem] = [] self._lock = asyncio.Lock()

async def put(self, request: InferenceRequest) -> None:
    async with self._lock:
        queue_item = QueueItem(
            sort_index=(
                request.priority.value,
                request.estimated_total_tokens,
                request.arrival_time,
            ),
            request=request,
        )

        heapq.heappush(self._queue, queue_item)

        metrics.set_queue_size(len(self._queue))

async def pop_candidates(
    self,
    limit: int,
) -> list[InferenceRequest]:
    async with self._lock:
        selected: list[InferenceRequest] = []
        temporary: list[QueueItem] = []

        while self._queue and len(selected) < limit:
            item = heapq.heappop(self._queue)
            selected.append(item.request)
            temporary.append(item)

        for item in temporary:
            heapq.heappush(self._queue, item)

        return selected

async def remove_requests(
    self,
    request_ids: set[str],
) -> None:
    async with self._lock:
        remaining_items = [
            item
            for item in self._queue
            if item.request.request_id not in request_ids
        ]

        heapq.heapify(remaining_items)

        self._queue = remaining_items

        metrics.set_queue_size(len(self._queue))

async def snapshot(self) -> list[InferenceRequest]:
    async with self._lock:
        return [
            item.request
            for item in sorted(self._queue)
        ]

async def size(self) -> int:
    async with self._lock:
        return len(self._queue)

async def is_empty(self) -> bool:
    return await self.size() == 0

class Scheduler: def init(self) -> None: self.queue = RequestQueue()

self.queue_capacity = (
        settings.scheduler.queue_capacity
    )

    self.max_candidate_pool = (
        settings.scheduler.max_batch_size * 4
    )

async def submit(
    self,
    prompt: str,
    max_tokens: int,
    priority: RequestPriority = RequestPriority.NORMAL,
    metadata: dict[str, Any] | None = None,
) -> InferenceRequest:
    queue_size = await self.queue.size()

    if queue_size >= self.queue_capacity:
        raise RuntimeError(
            "Scheduler queue capacity exceeded",
        )

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
        estimated_total_tokens=(
            request.estimated_total_tokens
        ),
    )

    return request

async def build_next_batch(
    self,
) -> ScheduledBatch | None:
    queue_size = await self.queue.size()

    if queue_size == 0:
        return None

    with profiler.profile_section(
        "scheduler.build_next_batch",
    ):
        candidate_requests = (
            await self.queue.pop_candidates(
                self.max_candidate_pool,
            )
        )

        accepted, rejected = (
            batch_builder.split_oversized_requests(
                candidate_requests,
            )
        )

        if rejected:
            logger.warning(
                "oversized_requests_detected",
                rejected_count=len(rejected),
            )

        optimized_requests = (
            batch_builder.optimize_ordering(
                accepted,
            )
        )

        batch = batch_builder.build(
            optimized_requests,
        )

        if batch is None:
            return None

        request_ids = {
            request.request_id
            for request in batch.requests
        }

        await self.queue.remove_requests(
            request_ids,
        )

        for request in batch.requests:
            request.status = (
                RequestStatus.SCHEDULED
            )

        metrics.record_batch(
            batch_size=batch.batch_size,
            active_tokens=(
                batch.total_estimated_tokens
            ),
            padded_tokens=(
                batch.total_estimated_tokens
                + batch.padding_tokens
            ),
        )

        logger.info(
            "batch_scheduled",
            batch_id=batch.batch_id,
            batch_size=batch.batch_size,
            total_estimated_tokens=(
                batch.total_estimated_tokens
            ),
            padding_efficiency=(
                batch.padding_efficiency
            ),
        )

        return batch

async def wait_for_batch(
    self,
    poll_interval_ms: int | None = None,
) -> ScheduledBatch | None:
    interval_ms = (
        poll_interval_ms
        or settings.scheduler.scheduler_tick_ms
    )

    while True:
        batch = await self.build_next_batch()

        if batch is not None:
            return batch

        await asyncio.sleep(interval_ms / 1000)

async def queue_snapshot(
    self,
) -> list[dict[str, Any]]:
    snapshot = await self.queue.snapshot()

    return [
        {
            "request_id": request.request_id,
            "priority": request.priority.name,
            "status": request.status.value,
            "estimated_total_tokens": (
                request.estimated_total_tokens
            ),
            "arrival_time": request.arrival_time,
        }
        for request in snapshot
    ]

scheduler = Scheduler()