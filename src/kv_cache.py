from __future__ import annotations

import time from collections import OrderedDict from dataclasses import dataclass from typing import Dict, Optional

from config import settings from datamodels import CacheAllocation from utils import logger

@dataclass(slots=True) class KVCacheEntry: request_id: str cache_key: str allocated_tokens: int last_accessed: float

class KVCacheManager: """ Lightweight LRU-based KV cache manager. Designed for compatibility with vLLM-style attention caching. """

def __init__(self) -> None:
    self.max_cache_tokens = settings.cache.max_cache_tokens
    self.max_entries = settings.cache.max_cache_entries

    self._cache: OrderedDict[str, KVCacheEntry] = OrderedDict()
    self._total_tokens = 0

def allocate(self, request_id: str, tokens: int) -> CacheAllocation:
    cache_key = f"kv_{request_id}"

    self._ensure_capacity(tokens)

    entry = KVCacheEntry(
        request_id=request_id,
        cache_key=cache_key,
        allocated_tokens=tokens,
        last_accessed=time.time(),
    )

    self._cache[cache_key] = entry
    self._cache.move_to_end(cache_key)

    self._total_tokens += tokens

    logger.debug(
        "kv_cache_allocated",
        request_id=request_id,
        tokens=tokens,
        total_cache_tokens=self._total_tokens,
    )

    return CacheAllocation(
        request_id=request_id,
        allocated_tokens=tokens,
        cache_key=cache_key,
    )

def access(self, cache_key: str) -> Optional[KVCacheEntry]:
    entry = self._cache.get(cache_key)

    if not entry:
        return None

    entry.last_accessed = time.time()
    self._cache.move_to_end(cache_key)

    return entry

def release(self, cache_key: str) -> None:
    entry = self._cache.pop(cache_key, None)

    if not entry:
        return

    self._total_tokens -= entry.allocated_tokens

    logger.debug(
        "kv_cache_released",
        cache_key=cache_key,
        freed_tokens=entry.allocated_tokens,
    )

def get_usage(self) -> Dict[str, float]:
    return {
        "total_tokens": self._total_tokens,
        "max_tokens": self.max_cache_tokens,
        "utilization": (
            self._total_tokens / self.max_cache_tokens
            if self.max_cache_tokens > 0
            else 0.0
        ),
        "entries": len(self._cache),
        "max_entries": self.max_entries,
    }

def _ensure_capacity(self, required_tokens: int) -> None:
    while (
        self._total_tokens + required_tokens
        > self.max_cache_tokens
        or len(self._cache) >= self.max_entries
    ):
        self._evict_lru()

def _evict_lru(self) -> None:
    if not self._cache:
        return

    _, entry = self._cache.popitem(last=False)
    self._total_tokens -= entry.allocated_tokens

    logger.debug(
        "kv_cache_evicted",
        request_id=entry.request_id,
        freed_tokens=entry.allocated_tokens,
    )

def clear(self) -> None:
    self._cache.clear()
    self._total_tokens = 0

    logger.info("kv_cache_cleared")

kv_cache = KVCacheManager()