from __future__ import annotations

import logging
import re
import threading
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .config import RerankerConfig, ResourcesConfig
from .embeddings import resolve_device
from .memory import (
    MemoryBudgetExceeded,
    configure_mps_memory_limit,
    enforce_process_memory_budget,
    is_memory_exhaustion_error,
    release_accelerator_memory,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "Qwen/Qwen3-Reranker-4B"
_TOKEN_RE = re.compile(r"[\w]+(?:[-+][\w]+)*", re.UNICODE)
_QUANTIZED_PRECISIONS = {
    "int8",
    "uint8",
    "binary",
    "ubinary",
    "4bit",
    "8bit",
    "int4",
}


class RerankerProvider(ABC):
    """Backend-independent scoring interface for the hybrid candidate set."""

    @abstractmethod
    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Return one relevance score per document, preserving input order."""

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        scores = self.score(query, documents)
        ranked = sorted(enumerate(scores), key=lambda item: (-item[1], item[0]))
        if top_k is None:
            return ranked
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        return ranked[:top_k]


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.casefold())


class LexicalReranker(RerankerProvider):
    """Deterministic token-overlap reranker used only as an explicit fallback."""

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        query_tokens = _tokens(query)
        query_counts = Counter(query_tokens)
        if not query_tokens:
            return [0.0] * len(documents)

        query_phrase = " ".join(query_tokens)
        scores: list[float] = []
        for document in documents:
            document_tokens = _tokens(document)
            if not document_tokens:
                scores.append(0.0)
                continue
            document_counts = Counter(document_tokens)
            overlap = sum((query_counts & document_counts).values())
            precision = overlap / len(document_tokens)
            recall = overlap / len(query_tokens)
            f1 = 0.0 if not overlap else (2.0 * precision * recall) / (precision + recall)
            phrase_bonus = (
                0.1 if query_phrase and query_phrase in " ".join(document_tokens) else 0.0
            )
            scores.append(min(1.0, f1 + phrase_bonus))
        return scores


def _validate_full_quality(config: RerankerConfig) -> None:
    if config.quantization:
        raise ValueError("Reranker quantization is disabled for the full-quality provider")
    if config.precision.strip().lower() in _QUANTIZED_PRECISIONS:
        raise ValueError(f"Quantized reranker precision is not supported: {config.precision}")
    if config.batch_size <= 0:
        raise ValueError("Reranker batch_size must be positive")
    if config.batch_size > 1:
        raise ValueError("Reranker batch_size may not exceed the safe ceiling of 1")
    if config.max_length <= 0:
        raise ValueError("Reranker max_length must be positive")
    if config.max_length > 2048:
        raise ValueError("Reranker max_length may not exceed the safe ceiling of 2048")


def _torch_dtype(precision: str) -> Any | None:
    normalized = precision.strip().lower().replace("-", "_")
    if normalized in {"", "auto", "full", "full_quality"}:
        return None
    names = {
        "fp32": "float32",
        "float32": "float32",
        "fp16": "float16",
        "float16": "float16",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
    }
    dtype_name = names.get(normalized)
    if dtype_name is None:
        raise ValueError(f"Unsupported reranker precision: {precision}")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to select an explicit model precision") from exc
    return getattr(torch, dtype_name)


def _default_model_factory(model_name: str, **kwargs: Any) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, **kwargs)


def _scores(values: Any, expected: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        array = array.reshape(1)
    elif array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    elif array.ndim == 2:
        # A two-or-more-label CrossEncoder returns one column per label. The final
        # column is conventionally the positive/relevant class.
        array = array[:, -1]
    if array.ndim != 1 or array.shape[0] != expected:
        raise ValueError(f"Reranker backend returned shape {array.shape}; expected ({expected},)")
    return [float(value) for value in array]


class CrossEncoderReranker(RerankerProvider):
    """Lazy, reusable Qwen3 CrossEncoder reranker with an explicit fallback."""

    def __init__(
        self,
        config: RerankerConfig,
        *,
        model_factory: Callable[..., Any] | None = None,
        device_resolver: Callable[[str], str] = resolve_device,
        resources: ResourcesConfig | None = None,
    ) -> None:
        _validate_full_quality(config)
        self.config = config
        self._model_factory = model_factory or _default_model_factory
        self._device_resolver = device_resolver
        self.resources = resources or ResourcesConfig()
        self._model: Any | None = None
        self._fallback: LexicalReranker | None = None
        self._load_lock = threading.Lock()
        self._device: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None or self._fallback is not None

    @property
    def active_backend(self) -> str:
        if self._fallback is not None:
            return "lexical"
        return "sentence_transformers"

    @property
    def device(self) -> str | None:
        return self._device

    def _activate_fallback(self, exc: BaseException) -> LexicalReranker:
        if not self.config.allow_explicit_lexical_fallback:
            raise RuntimeError(
                f"Unable to use reranker model {self.config.model}; lexical fallback is not allowed"
            ) from exc
        if self._fallback is None:
            LOGGER.warning(
                "Reranker model %s is unavailable (%s: %s); explicitly allowed fallback "
                "to deterministic lexical reranking is active",
                self.config.model,
                type(exc).__name__,
                exc,
            )
            self._fallback = LexicalReranker()
        return self._fallback

    def _load_model(self) -> Any | LexicalReranker:
        if self._fallback is not None:
            return self._fallback
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._fallback is not None:
                return self._fallback
            if self._model is not None:
                return self._model
            try:
                self._device = self._device_resolver(self.config.device)
                if self.resources is not None:
                    enforce_process_memory_budget(self.resources, "reranker model load")
                    try:
                        configure_mps_memory_limit(self._device, self.resources)
                    except MemoryBudgetExceeded:
                        raise
                    except Exception as exc:
                        raise MemoryBudgetExceeded(
                            "reranker model cannot start because the configured MPS "
                            f"memory cap could not be enforced ({type(exc).__name__}: {exc})"
                        ) from None
                model_kwargs: dict[str, Any] = {}
                dtype = _torch_dtype(self.config.precision)
                if dtype is not None:
                    model_kwargs["torch_dtype"] = dtype
                if self.resources is not None:
                    kwargs_config = {
                        "use_cache": False,
                        "attn_implementation": "sdpa",
                    }
                kwargs: dict[str, Any] = {
                    "device": self._device,
                    "max_length": self.config.max_length,
                }
                if self.config.revision:
                    kwargs["revision"] = self.config.revision
                if self.config.instruction.strip():
                    kwargs["prompts"] = {
                        "papers_mcp": self.config.instruction.strip(),
                    }
                    kwargs["default_prompt_name"] = "papers_mcp"
                if model_kwargs:
                    kwargs["model_kwargs"] = model_kwargs
                if self.resources is not None:
                    kwargs["config_kwargs"] = kwargs_config
                LOGGER.info(
                    "Loading reranker backend=sentence_transformers model=%s device=%s "
                    "precision=%s quantization=disabled",
                    self.config.model,
                    self._device,
                    self.config.precision,
                )
                self._model = self._model_factory(self.config.model, **kwargs)
                if self.resources is not None:
                    enforce_process_memory_budget(self.resources, "reranker model load")
            except Exception as exc:
                self._model = None
                memory_failure = is_memory_exhaustion_error(exc)
                error = f"{type(exc).__name__}: {exc}"
            else:
                return self._model
        release_accelerator_memory(self._device, self.resources)
        if memory_failure:
            raise MemoryBudgetExceeded(
                f"reranker model stopped within the configured memory cap ({error})"
            ) from None
        return self._activate_fallback(RuntimeError(error))

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        backend = self._load_model()
        if isinstance(backend, LexicalReranker):
            return backend.score(query, documents)
        try:
            scores: list[float] = []
            for start in range(0, len(documents), self.config.batch_size):
                pairs = [
                    (query, document)
                    for document in documents[start : start + self.config.batch_size]
                ]
                if self.resources is not None:
                    enforce_process_memory_budget(self.resources, "reranker batch")
                values = backend.predict(
                    pairs,
                    batch_size=len(pairs),
                    show_progress_bar=False,
                )
                scores.extend(_scores(values, len(pairs)))
                release_accelerator_memory(self._device, self.resources)
                if self.resources is not None:
                    enforce_process_memory_budget(self.resources, "reranker batch")
            return scores
        except Exception as exc:
            self._model = None
            memory_failure = is_memory_exhaustion_error(exc)
            error = f"{type(exc).__name__}: {exc}"
        del backend
        release_accelerator_memory(self._device, self.resources)
        if memory_failure:
            raise MemoryBudgetExceeded(
                f"reranker batch stopped within the configured memory cap ({error})"
            ) from None
        return self._activate_fallback(RuntimeError(error)).score(query, documents)


def create_reranker(
    config: RerankerConfig,
    resources: ResourcesConfig | None = None,
) -> RerankerProvider:
    """Create the configured reranker without loading the production model."""

    backend = config.backend.strip().lower().replace("-", "_")
    if backend in {"sentence_transformers", "sentence_transformer", "cross_encoder", "qwen3"}:
        return CrossEncoderReranker(config, resources=resources)
    if backend in {"lexical", "token_overlap"}:
        _validate_full_quality(config)
        if not config.allow_explicit_lexical_fallback:
            raise ValueError(
                "Lexical reranker backend requires allow_explicit_lexical_fallback=true"
            )
        LOGGER.warning("Using explicitly configured deterministic lexical reranker backend")
        return LexicalReranker()
    raise ValueError(f"Unsupported reranker backend: {config.backend}")


build_reranker = create_reranker


__all__ = [
    "DEFAULT_RERANKER_MODEL",
    "CrossEncoderReranker",
    "LexicalReranker",
    "RerankerProvider",
    "build_reranker",
    "create_reranker",
]
