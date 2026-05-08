from future import annotations

import asyncio import time from dataclasses import asdict from typing import Any

from batching import ScheduledBatch from datamodels import BatchExecutionResult, ExecutionResult from kv_cache import kv_cache from metrics import metrics from model_loader import model_loader from profiler import profiler from utils import logger

class Executor: """ Production-grade inference executor for Inferra.

Responsibilities:
- Execute scheduled batches on Qwen3-4B (vLLM)
- Track latency + throughput
- Manage KV cache lifecycle
- Emit structured metrics
"""

def __init__(self) -> None:
    self._llm = model_loader.get_model()
    self._tokenizer = model_loader.get_tokenizer()

async def execute_batch(
    self,
    batch: ScheduledBatch,
) -> BatchExecutionResult:
    if batch.batch_size == 0:
        raise ValueError("Empty batch cannot be executed")

    start_time = time.perf_counter()

    logger.info(
        "batch_execution_started",
        batch_id=batch.batch_id,
        batch_size=batch.batch_size,
    )

    with profiler.profile_section("executor.execute_batch"):
        results: list[ExecutionResult] = await self._run_inference(batch)

    total_latency_ms = (time.perf_counter() - start_time) * 1000

    total_tokens = sum(r.generated_tokens for r in results)

    tokens_per_second = (
        total_tokens / (total_latency_ms / 1000)
        if total_latency_ms > 0
        else 0.0
    )

    # Metrics
    metrics.record_request(
        latency_ms=total_latency_ms,
        generated_tokens=total_tokens,
    )

    logger.info(
        "batch_execution_completed",
        batch_id=batch.batch_id,
        latency_ms=total_latency_ms,
        tokens_per_second=tokens_per_second,
    )

    return BatchExecutionResult(
        batch_id=batch.batch_id,
        results=results,
        total_latency_ms=total_latency_ms,
        tokens_per_second=tokens_per_second,
    )

async def _run_inference(
    self,
    batch: ScheduledBatch,
) -> list[ExecutionResult]:
    prompts = [req.prompt for req in batch.requests]

    sampling_params = model_loader.create_sampling_params(
        temperature=0.7,
        top_p=0.9,
        max_tokens=max(
            req.max_request_tokens
            for req in batch.requests
        ),
    )

    llm = model_loader.get_model()

    # vLLM call is blocking → run in thread
    outputs = await asyncio.to_thread(
        llm.generate,
        prompts,
        sampling_params,
        use_tqdm=False,
    )

    results: list[ExecutionResult] = []

    for req, output in zip(batch.requests, outputs):
        generated_text = output.outputs[0].text
        generated_tokens = len(
            self._tokenizer.encode(generated_text)
        )

        # KV cache allocation per request (logical binding)
        kv_cache.allocate(
            request_id=req.request_id,
            tokens=generated_tokens,
        )

        results.append(
            ExecutionResult(
                request_id=req.request_id,
                generated_text=generated_text,
                generated_tokens=generated_tokens,
                latency_ms=(
                    batch.total_estimated_tokens
                ),
                metadata={
                    "batch_id": batch.batch_id,
                },
            )
        )

    return results

def warmup(self) -> None:
    logger.info("executor_warmup_started")

    llm = model_loader.get_model()

    prompts = ["warmup"]

    sampling_params = model_loader.create_sampling_params(
        temperature=0.0,
        top_p=1.0,
        max_tokens=8,
    )

    llm.generate(
        prompts,
        sampling_params,
        use_tqdm=False,
    )

    logger.info("executor_warmup_completed")

executor = Executor()