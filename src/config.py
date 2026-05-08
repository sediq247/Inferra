from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RuntimeMode(str, Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    HYBRID = "hybrid"


class SchedulerPolicy(str, Enum):
    FIFO = "fifo"
    SHORTEST_FIRST = "shortest_first"
    TOKEN_AWARE = "token_aware"


class CachePolicy(str, Enum):
    LRU = "lru"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ModelConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFERRA_MODEL_")

    model_name: str = "Qwen/Qwen3-4B"
    model_path: str | None = None
    tokenizer_path: str | None = None

    device: str = "cuda"
    dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"

    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = Field(default=0.92, ge=0.1, le=1.0)

    max_model_len: int = Field(default=32768, gt=0)
    trust_remote_code: bool = True
    enforce_eager: bool = False

    quantization: str | None = None
    seed: int = 42


class SchedulerConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFERRA_SCHEDULER_")

    policy: SchedulerPolicy = SchedulerPolicy.TOKEN_AWARE

    max_batch_size: int = Field(default=32, gt=0)
    max_batch_tokens: int = Field(default=8192, gt=0)

    queue_capacity: int = Field(default=10000, gt=0)

    scheduler_tick_ms: int = Field(default=10, ge=1)

    enable_length_grouping: bool = True
    enable_dynamic_batching: bool = True

    max_queue_wait_ms: int = Field(default=100, ge=1)


class ExecutionConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFERRA_EXECUTION_")

    default_mode: RuntimeMode = RuntimeMode.HYBRID

    max_concurrent_batches: int = Field(default=2, gt=0)

    enable_streaming: bool = True

    request_timeout_seconds: int = Field(default=300, gt=0)

    warmup_steps: int = Field(default=2, ge=0)

    cuda_graphs: bool = False

    prefill_chunk_size: int = Field(default=2048, gt=0)


class KVCacheConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFERRA_KV_")

    policy: CachePolicy = CachePolicy.LRU

    max_cache_entries: int = Field(default=10000, gt=0)

    block_size_tokens: int = Field(default=16, gt=0)

    enable_prefix_caching: bool = True

    eviction_interval_seconds: int = Field(default=30, gt=0)

    cache_watermark_ratio: float = Field(default=0.90, ge=0.1, le=1.0)


class OptimizerConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFERRA_OPTIMIZER_")

    latency_threshold_ms: int = Field(default=2000, gt=0)

    throughput_queue_threshold: int = Field(default=64, gt=0)

    long_context_threshold_tokens: int = Field(default=4096, gt=0)

    mode_switch_cooldown_seconds: int = Field(default=10, gt=0)

    enable_predictive_scaling: bool = True

    moving_average_window: int = Field(default=20, gt=0)


class MetricsConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFERRA_METRICS_")

    enabled: bool = True

    export_interval_seconds: int = Field(default=5, gt=0)

    prometheus_enabled: bool = True

    prometheus_port: int = Field(default=9090, ge=1, le=65535)

    retain_history_minutes: int = Field(default=60, gt=0)


class ProfilingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFERRA_PROFILER_")

    enabled: bool = True

    trace_scheduler: bool = True
    trace_executor: bool = True
    trace_cache: bool = True

    record_gpu_metrics: bool = True

    profiling_window_seconds: int = Field(default=30, gt=0)


class DashboardConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFERRA_DASHBOARD_")

    enabled: bool = True

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)

    websocket_enabled: bool = True

    refresh_interval_ms: int = Field(default=1000, ge=100)


class LoggingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFERRA_LOGGING_")

    level: LogLevel = LogLevel.INFO

    json_logs: bool = True

    log_dir: Path = Path("logs")

    enable_console_logging: bool = True

    @field_validator("log_dir")
    @classmethod
    def validate_log_dir(cls, value: Path) -> Path:
        value.mkdir(parents=True, exist_ok=True)
        return value


class BenchmarkConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFERRA_BENCHMARK_")

    synthetic_request_count: int = Field(default=1000, gt=0)

    concurrent_clients: int = Field(default=64, gt=0)

    benchmark_duration_seconds: int = Field(default=300, gt=0)

    random_seed: int = 42


class InferraSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    environment: Literal["development", "staging", "production"] = (
        "development"
    )

    debug: bool = False

    model: ModelConfig = Field(default_factory=ModelConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    kv_cache: KVCacheConfig = Field(default_factory=KVCacheConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    profiler: ProfilingConfig = Field(default_factory=ProfilingConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)


@lru_cache(maxsize=1)
def get_settings() -> InferraSettings:
    return InferraSettings()


settings = get_settings()