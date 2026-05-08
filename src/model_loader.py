from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from config import settings
from profiler import profiler
from utils import logger


@dataclass(slots=True)
class ModelArtifacts:
    llm: LLM
    tokenizer: Any


class ModelLoader:
    def __init__(self) -> None:
        self._artifacts: ModelArtifacts | None = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._artifacts is not None

    def load(self) -> ModelArtifacts:
        if self._artifacts is not None:
            return self._artifacts

        with self._lock:
            if self._artifacts is not None:
                return self._artifacts

            logger.info(
                "model_loading_started",
                model=settings.model.model_name,
            )

            with profiler.profile_section("model_loader.load"):
                tokenizer = self._load_tokenizer()
                llm = self._load_llm()

                self._artifacts = ModelArtifacts(
                    llm=llm,
                    tokenizer=tokenizer,
                )

            logger.info(
                "model_loading_completed",
                model=settings.model.model_name,
            )

            return self._artifacts

    def unload(self) -> None:
        with self._lock:
            if self._artifacts is None:
                return

            del self._artifacts
            self._artifacts = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("model_unloaded")

    def reload(self) -> ModelArtifacts:
        self.unload()
        return self.load()

    def get_model(self) -> LLM:
        artifacts = self.load()
        return artifacts.llm

    def get_tokenizer(self) -> Any:
        artifacts = self.load()
        return artifacts.tokenizer

    def create_sampling_params(
        self,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        repetition_penalty: float = 1.0,
        stop: list[str] | None = None,
    ) -> SamplingParams:
        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
            stop=stop,
        )

    def warmup(self) -> None:
        logger.info("model_warmup_started")

        llm = self.get_model()

        warmup_prompt = "Inferra runtime warmup request."

        sampling_params = self.create_sampling_params(
            temperature=0.0,
            top_p=1.0,
            max_tokens=8,
        )

        with profiler.profile_section("model_loader.warmup"):
            for _ in range(settings.execution.warmup_steps):
                llm.generate(
                    prompts=[warmup_prompt],
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )

        logger.info("model_warmup_completed")

    def model_metadata(self) -> dict[str, Any]:
        return {
            "model_name": settings.model.model_name,
            "device": settings.model.device,
            "dtype": settings.model.dtype,
            "tensor_parallel_size": (
                settings.model.tensor_parallel_size
            ),
            "max_model_len": settings.model.max_model_len,
            "quantization": settings.model.quantization,
        }

    def _load_tokenizer(self) -> Any:
        tokenizer_path = (
            settings.model.tokenizer_path
            or settings.model.model_path
            or settings.model.model_name
        )

        logger.info(
            "tokenizer_loading_started",
            tokenizer_path=tokenizer_path,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=settings.model.trust_remote_code,
        )

        logger.info("tokenizer_loading_completed")

        return tokenizer

    def _load_llm(self) -> LLM:
        model_path = (
            settings.model.model_path
            or settings.model.model_name
        )

        logger.info(
            "llm_loading_started",
            model_path=model_path,
        )

        llm = LLM(
            model=model_path,
            tensor_parallel_size=(
                settings.model.tensor_parallel_size
            ),
            dtype=settings.model.dtype,
            gpu_memory_utilization=(
                settings.model.gpu_memory_utilization
            ),
            trust_remote_code=(
                settings.model.trust_remote_code
            ),
            max_model_len=settings.model.max_model_len,
            enforce_eager=settings.model.enforce_eager,
            quantization=settings.model.quantization,
            seed=settings.model.seed,
        )

        logger.info("llm_loading_completed")

        return llm


model_loader = ModelLoader()