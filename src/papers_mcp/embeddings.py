from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .config import EmbeddingConfig, ResourcesConfig
from .memory import (
    MemoryBudgetExceeded,
    configure_mps_memory_limit,
    enforce_process_memory_budget,
    is_memory_exhaustion_error,
    release_accelerator_memory,
)
from .vectors import model_fingerprint

LOGGER = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
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


class EmbeddingProvider(ABC):
    """Backend-independent interface used by ingestion and retrieval."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed one search query as a normalized vector."""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed documents as normalized vectors in input order."""


def resolve_device(requested: str = "auto") -> str:
    """Resolve ``auto`` to the best available torch device, preferring Apple MPS."""

    requested = requested.strip().lower()
    if requested and requested != "auto":
        return requested

    try:
        import torch
    except ImportError:
        return "cpu"

    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _validate_full_quality(config: EmbeddingConfig) -> None:
    if config.quantization:
        raise ValueError("Embedding quantization is disabled for the full-quality provider")
    if config.precision.strip().lower() in _QUANTIZED_PRECISIONS:
        raise ValueError(f"Quantized embedding precision is not supported: {config.precision}")
    if config.fallback_dimensions <= 0:
        raise ValueError("Embedding fallback_dimensions must be positive")
    if config.fallback_dimensions > 8192:
        raise ValueError("Embedding fallback_dimensions may not exceed 8192")
    if config.batch_size <= 0:
        raise ValueError("Embedding batch_size must be positive")
    if config.batch_size > 1:
        raise ValueError("Embedding batch_size may not exceed the safe ceiling of 1")
    if config.max_length <= 0:
        raise ValueError("Embedding max_length must be positive")
    if config.max_length > 2048:
        raise ValueError("Embedding max_length may not exceed the safe ceiling of 2048")


def _torch_dtype(precision: str) -> Any | None:
    normalized = precision.strip().lower().replace("-", "_")
    if normalized in {"", "auto", "full", "full_quality"}:
        # SentenceTransformer/transformers defaults to its highest compatible model dtype.
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
        raise ValueError(f"Unsupported embedding precision: {precision}")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to select an explicit model precision") from exc
    return getattr(torch, dtype_name)


def _default_model_factory(model_name: str, **kwargs: Any) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, **kwargs)


def configured_embedding_identity(
    config: EmbeddingConfig,
    dimensions: int,
    *,
    active_backend: str | None = None,
) -> tuple[str, str]:
    """Return the immutable storage identity for a configured embedding backend."""

    backend = (active_backend or config.backend).strip().lower().replace("-", "_")
    if backend in {"sentence_transformer", "qwen3"}:
        backend = "sentence_transformers"
    if backend in {"hash", "deterministic_hash"}:
        model_name = f"hash-v1:{dimensions}"
        provider = "hash"
        revision = ""
    else:
        model_name = config.model
        provider = backend
        revision = config.revision
    profile = ""
    if provider == "sentence_transformers":
        profile = hashlib.sha256(
            json.dumps(
                {
                    "context_format": "papers-document-context-v1",
                    "max_length": config.max_length,
                    "precision": config.precision.strip().lower().replace("-", "_"),
                    "query_instruction": config.query_instruction,
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    return model_name, model_fingerprint(
        model_name,
        provider=provider,
        dimensions=dimensions,
        revision=revision,
        profile=profile,
    )


def embedding_provider_identity(
    provider: EmbeddingProvider,
    dimensions: int,
) -> tuple[str, str]:
    """Resolve a loaded provider, including an activated fallback, to its vector identity."""

    active = str(getattr(provider, "active_backend", "")).strip().lower().replace("-", "_")
    if active == "hash" or provider.__class__.__name__ == "HashEmbeddingProvider":
        model_name = f"hash-v1:{dimensions}"
        return model_name, model_fingerprint(
            model_name,
            provider="hash",
            dimensions=dimensions,
        )
    config = getattr(provider, "config", None)
    if isinstance(config, EmbeddingConfig):
        return configured_embedding_identity(
            config,
            dimensions,
            active_backend=active or None,
        )
    model_name = provider.__class__.__name__
    backend = active or model_name
    return model_name, model_fingerprint(
        model_name,
        provider=backend,
        dimensions=dimensions,
    )


def _normalize_matrix(values: Any, expected_rows: int) -> list[list[float]]:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows:
        raise ValueError(
            "Embedding backend returned an unexpected shape: "
            f"{matrix.shape}; expected ({expected_rows}, dimensions)"
        )

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
    return matrix.tolist()


def _format_query(instruction: str, text: str) -> str:
    instruction = instruction.strip()
    if not instruction:
        return text
    return f"Instruct: {instruction}\nQuery: {text}"


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.casefold())


def _features(text: str) -> Counter[str]:
    tokens = _tokens(text)
    features: Counter[str] = Counter(f"word:{token}" for token in tokens)
    features.update(
        f"bigram:{left}\x1f{right}" for left, right in zip(tokens, tokens[1:], strict=False)
    )
    return features


class HashEmbeddingProvider(EmbeddingProvider):
    """Deterministic normalized feature hashing for explicit offline fallback.

    Word unigrams carry the largest weight and adjacent-token bigrams add modest
    phrase sensitivity. Identical lexical features always land in the same signed
    bucket, so cosine similarity remains useful for literal/lexical retrieval.
    """

    def __init__(self, dimensions: int = 768) -> None:
        if dimensions <= 0:
            raise ValueError("Hash embedding dimensions must be positive")
        self.dimensions = dimensions

    @staticmethod
    def _bucket(feature: str, dimensions: int) -> tuple[int, float]:
        digest = hashlib.blake2b(
            feature.encode("utf-8"),
            digest_size=16,
            person=b"papers-vector-v1",
        ).digest()
        index = int.from_bytes(digest[:8], "little") % dimensions
        sign = 1.0 if digest[8] & 1 else -1.0
        return index, sign

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for feature, count in _features(text).items():
            index, sign = self._bucket(feature, self.dimensions)
            kind_weight = 0.55 if feature.startswith("bigram:") else 1.0
            vector[index] += sign * kind_weight * (1.0 + math.log(count))

        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Lazy, reusable SentenceTransformer provider with an explicit hash fallback."""

    def __init__(
        self,
        config: EmbeddingConfig,
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
        self._fallback: HashEmbeddingProvider | None = None
        self._load_lock = threading.Lock()
        self._device: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None or self._fallback is not None

    @property
    def active_backend(self) -> str:
        if self._fallback is not None:
            return "hash"
        return "sentence_transformers"

    @property
    def device(self) -> str | None:
        return self._device

    def _activate_fallback(self, exc: BaseException) -> HashEmbeddingProvider:
        if not self.config.allow_explicit_hash_fallback:
            raise RuntimeError(
                f"Unable to use embedding model {self.config.model}; hash fallback is not allowed"
            ) from exc
        if self._fallback is None:
            LOGGER.warning(
                "Embedding model %s is unavailable (%s: %s); explicitly allowed fallback "
                "to deterministic %d-dimensional hash embeddings is active",
                self.config.model,
                type(exc).__name__,
                exc,
                self.config.fallback_dimensions,
            )
            self._fallback = HashEmbeddingProvider(self.config.fallback_dimensions)
        return self._fallback

    def _load_model(self) -> Any | HashEmbeddingProvider:
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
                    enforce_process_memory_budget(self.resources, "embedding model load")
                    try:
                        configure_mps_memory_limit(self._device, self.resources)
                    except MemoryBudgetExceeded:
                        raise
                    except Exception as exc:
                        raise MemoryBudgetExceeded(
                            "embedding model cannot start because the configured MPS "
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
                kwargs: dict[str, Any] = {"device": self._device}
                if self.config.revision:
                    kwargs["revision"] = self.config.revision
                if model_kwargs:
                    kwargs["model_kwargs"] = model_kwargs
                if self.resources is not None:
                    kwargs["config_kwargs"] = kwargs_config
                LOGGER.info(
                    "Loading embedding backend=sentence_transformers model=%s device=%s "
                    "precision=%s quantization=disabled",
                    self.config.model,
                    self._device,
                    self.config.precision,
                )
                model = self._model_factory(self.config.model, **kwargs)
                if hasattr(model, "max_seq_length"):
                    model.max_seq_length = self.config.max_length
                self._model = model
                if self.resources is not None:
                    enforce_process_memory_budget(self.resources, "embedding model load")
            except Exception as exc:
                self._model = None
                if "model" in locals():
                    del model
                memory_failure = is_memory_exhaustion_error(exc)
                error = f"{type(exc).__name__}: {exc}"
            else:
                return self._model
        release_accelerator_memory(self._device, self.resources)
        if memory_failure:
            raise MemoryBudgetExceeded(
                f"embedding model stopped within the configured memory cap ({error})"
            ) from None
        return self._activate_fallback(RuntimeError(error))

    def _encode(
        self,
        texts: Sequence[str],
        *,
        fallback_texts: Sequence[str] | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        backend = self._load_model()
        if isinstance(backend, HashEmbeddingProvider):
            return backend.embed_documents(fallback_texts or texts)
        try:
            vectors: list[list[float]] = []
            for start in range(0, len(texts), self.config.batch_size):
                batch = list(texts[start : start + self.config.batch_size])
                if self.resources is not None:
                    enforce_process_memory_budget(self.resources, "embedding batch")
                values = backend.encode(
                    batch,
                    batch_size=len(batch),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                vectors.extend(_normalize_matrix(values, len(batch)))
                release_accelerator_memory(self._device, self.resources)
                if self.resources is not None:
                    enforce_process_memory_budget(self.resources, "embedding batch")
            return vectors
        except Exception as exc:
            self._model = None
            memory_failure = is_memory_exhaustion_error(exc)
            error = f"{type(exc).__name__}: {exc}"
        del backend
        release_accelerator_memory(self._device, self.resources)
        if memory_failure:
            raise MemoryBudgetExceeded(
                f"embedding batch stopped within the configured memory cap ({error})"
            ) from None
        fallback = self._activate_fallback(RuntimeError(error))
        return fallback.embed_documents(fallback_texts or texts)

    def embed_query(self, text: str) -> list[float]:
        query = _format_query(self.config.query_instruction, text)
        # The long semantic instruction is useful to Qwen but would overwhelm
        # literal similarity if the explicit hash fallback is active.
        return self._encode([query], fallback_texts=[text])[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts)


def create_embedding_provider(
    config: EmbeddingConfig,
    resources: ResourcesConfig | None = None,
) -> EmbeddingProvider:
    """Create the configured provider without loading a production model."""

    backend = config.backend.strip().lower().replace("-", "_")
    if backend in {"sentence_transformers", "sentence_transformer", "qwen3"}:
        return SentenceTransformerEmbeddingProvider(config, resources=resources)
    if backend in {"hash", "deterministic_hash"}:
        _validate_full_quality(config)
        if not config.allow_explicit_hash_fallback:
            raise ValueError("Hash embedding backend requires allow_explicit_hash_fallback=true")
        LOGGER.warning(
            "Using explicitly configured deterministic hash embedding backend (%d dimensions)",
            config.fallback_dimensions,
        )
        return HashEmbeddingProvider(config.fallback_dimensions)
    raise ValueError(f"Unsupported embedding backend: {config.backend}")


build_embedding_provider = create_embedding_provider


__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "EmbeddingProvider",
    "HashEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "configured_embedding_identity",
    "embedding_provider_identity",
    "build_embedding_provider",
    "create_embedding_provider",
    "resolve_device",
]
