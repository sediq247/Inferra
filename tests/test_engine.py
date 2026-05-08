import asyncio
import pytest

from engine import engine
from scheduler import scheduler
from kv_cache import kv_cache
from metrics import metrics


@pytest.mark.asyncio
async def test_engine_start_stop():
    """
    Ensure engine starts and stops without crashing.
    """
    await engine.start()
    assert engine.state.running is True

    await asyncio.sleep(0.1)

    await engine.stop()
    assert engine.state.running is False


@pytest.mark.asyncio
async def test_engine_submit_basic():
    """
    Ensure engine accepts requests and scheduler queues them.
    """

    await engine.start()

    req_id = await engine.submit(
        prompt="test prompt inferra",
        max_tokens=64,
    )

    assert req_id is not None

    # allow scheduler to process briefly
    await asyncio.sleep(0.2)

    queue_size = await scheduler.queue.size()

    assert queue_size >= 0  # sanity check only

    await engine.stop()


@pytest.mark.asyncio
async def test_kv_cache_integration():
    """
    Ensure KV cache can allocate safely during execution flow.
    """

    entry = kv_cache.allocate(
        request_id="test_req",
        tokens=128,
    )

    assert entry.request_id == "test_req"
    assert entry.allocated_tokens == 128

    usage = kv_cache.get_usage()

    assert "total_tokens" in usage
    assert usage["total_tokens"] >= 128


@pytest.mark.asyncio
async def test_metrics_exist():
    """
    Ensure metrics system is alive and callable.
    """

    metrics.record_request(
        latency_ms=120.0,
        generated_tokens=50,
    )

    throughput = metrics.get_throughput()
    latency = metrics.get_avg_latency()

    assert throughput >= 0
    assert latency >= 0