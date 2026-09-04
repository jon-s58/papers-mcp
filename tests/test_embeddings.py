from __future__ import annotations

import logging
import math
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import papers_mcp.embeddings as embeddings_module
from papers_mcp.config import EmbeddingConfig, ResourcesConfig
from papers_mcp.embeddings import (
    HashEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    configured_embedding_identity,
    create_embedding_provider,
    resolve_device,
)


def embedding_config(**overrides: Any) -> EmbeddingConfig:
    values: dict[str, Any] = {
        "backend": "sentence_transformers",
        "model": "Qwen/Qwen3-Embedding-4B",
        "device": "auto",
        "precision": "auto",
        "quantization": False,
        "batch_size": 1,
        "max_length": 2048,
        "allow_explicit_hash_fallback": True,
        "fallback_dimensions": 128,
        "query_instruction": "Retrieve exact CAD mathematics.",
    }
    values.update(overrides)
    return EmbeddingConfig(**values)


def test_hash_embeddings_are_deterministic_normalized_and_lexically_meaningful() -> None:
    provider = HashEmbeddingProvider(256)
    query = provider.embed_query("NURBS surface continuity constraints")
    same = HashEmbeddingProvider(256).embed_query("NURBS surface continuity constraints")
    related, unrelated = provider.embed_documents(
        [
            "G1 continuity constraints for adjacent NURBS surface patches",
            "stochastic language model training and image captions",
        ]
    )

    assert query == same
    assert math.isclose(np.linalg.norm(query), 1.0, rel_tol=1e-6)
    assert float(np.dot(query, related)) > float(np.dot(query, unrelated))


def test_sentence_transformer_is_lazy_reused_and_applies_query_instruction() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeModel:
        max_seq_length = 0

        def __init__(self) -> None:
            self.encoded: list[tuple[list[str], dict[str, Any]]] = []

        def encode(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            self.encoded.append((texts, kwargs))
            return [[3.0, 4.0] for _ in texts]

    model = FakeModel()

    def factory(name: str, **kwargs: Any) -> FakeModel:
        calls.append((name, kwargs))
        return model

    provider = SentenceTransformerEmbeddingProvider(
        embedding_config(), model_factory=factory, device_resolver=lambda _: "mps"
    )
    assert not provider.is_loaded
    assert calls == []

    query = provider.embed_query("vertex enclosure")
    documents = provider.embed_documents(["first", "second"])

    assert calls == [
        (
            "Qwen/Qwen3-Embedding-4B",
            {
                "device": "mps",
                "config_kwargs": {"use_cache": False, "attn_implementation": "sdpa"},
            },
        )
    ]
    assert provider.is_loaded
    assert provider.active_backend == "sentence_transformers"
    assert provider.device == "mps"
    assert model.max_seq_length == 2048
    assert model.encoded[0][0] == [
        "Instruct: Retrieve exact CAD mathematics.\nQuery: vertex enclosure"
    ]
    assert model.encoded[0][1]["normalize_embeddings"] is True
    assert np.allclose(query, [0.6, 0.8])
    assert documents == [[0.6000000238418579, 0.800000011920929]] * 2


def test_auto_device_prefers_available_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        cuda=SimpleNamespace(is_available=lambda: True),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert resolve_device("auto") == "mps"
    assert resolve_device("cpu") == "cpu"


def test_hash_fallback_is_logged_and_only_activated_when_allowed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def unavailable(*args: Any, **kwargs: Any) -> None:
        raise ImportError("sentence-transformers missing")

    provider = SentenceTransformerEmbeddingProvider(
        embedding_config(), model_factory=unavailable, device_resolver=lambda _: "cpu"
    )
    with caplog.at_level(logging.WARNING):
        vector = provider.embed_query("surface intersection")
    assert provider.active_backend == "hash"
    assert len(vector) == 128
    assert "explicitly allowed fallback" in caplog.text

    strict = SentenceTransformerEmbeddingProvider(
        embedding_config(allow_explicit_hash_fallback=False),
        model_factory=unavailable,
        device_resolver=lambda _: "cpu",
    )
    with pytest.raises(RuntimeError, match="fallback is not allowed"):
        strict.embed_query("surface intersection")


def test_explicit_hash_backend_is_configuration_gated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        provider = create_embedding_provider(embedding_config(backend="hash"))
    assert isinstance(provider, HashEmbeddingProvider)
    assert "explicitly configured" in caplog.text

    with pytest.raises(ValueError, match="requires allow_explicit_hash_fallback"):
        create_embedding_provider(
            embedding_config(backend="hash", allow_explicit_hash_fallback=False)
        )


def test_quantization_is_rejected() -> None:
    with pytest.raises(ValueError, match="quantization is disabled"):
        SentenceTransformerEmbeddingProvider(embedding_config(quantization=True))


def test_revision_is_passed_to_loader_and_enters_vector_identity() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeModel:
        max_seq_length = 0

        def encode(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    def factory(name: str, **kwargs: Any) -> FakeModel:
        calls.append((name, kwargs))
        return FakeModel()

    config = embedding_config(revision="commit-a")
    provider = SentenceTransformerEmbeddingProvider(
        config,
        model_factory=factory,
        device_resolver=lambda _: "cpu",
    )
    provider.embed_query("query")

    assert calls[0][1]["revision"] == "commit-a"
    first = configured_embedding_identity(config, 2)[1]
    changed_revision = configured_embedding_identity(embedding_config(revision="commit-b"), 2)[1]
    changed_profile = configured_embedding_identity(
        embedding_config(revision="commit-a", max_length=4096), 2
    )[1]
    assert len({first, changed_revision, changed_profile}) == 3


def test_embedding_requests_are_split_into_explicit_bounded_batches() -> None:
    class FakeModel:
        max_seq_length = 0

        def __init__(self) -> None:
            self.batches: list[list[str]] = []

        def encode(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            self.batches.append(texts)
            return [[1.0, 0.0] for _ in texts]

    model = FakeModel()
    factory_kwargs: dict[str, Any] = {}

    def factory(_name: str, **kwargs: Any) -> FakeModel:
        factory_kwargs.update(kwargs)
        return model

    provider = SentenceTransformerEmbeddingProvider(
        embedding_config(batch_size=1),
        model_factory=factory,
        device_resolver=lambda _: "cpu",
        resources=ResourcesConfig(release_memory_after_batch=False),
    )

    vectors = provider.embed_documents(["one", "two", "three"])

    assert len(vectors) == 3
    assert model.batches == [["one"], ["two"], ["three"]]
    assert factory_kwargs["config_kwargs"] == {
        "use_cache": False,
        "attn_implementation": "sdpa",
    }


def test_mps_oom_discards_loaded_model_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    releases: list[str | None] = []

    class FailingModel:
        max_seq_length = 0

        def encode(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            raise RuntimeError("MPS backend out of memory")

    monkeypatch.setattr(embeddings_module, "configure_mps_memory_limit", lambda *_args: 1)
    monkeypatch.setattr(
        embeddings_module,
        "release_accelerator_memory",
        lambda device, _resources: releases.append(device),
    )
    provider = SentenceTransformerEmbeddingProvider(
        embedding_config(batch_size=1),
        model_factory=lambda *_args, **_kwargs: FailingModel(),
        device_resolver=lambda _: "mps",
        resources=ResourcesConfig(),
    )

    with pytest.raises(MemoryError, match="stopped within the configured memory cap"):
        provider.embed_query("bounded failure")

    assert provider.active_backend == "sentence_transformers"
    assert provider._model is None
    assert releases == ["mps"]


def test_failure_to_install_mps_cap_never_activates_hash_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory_calls = 0

    def factory(*_args, **_kwargs):
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("model must not load without the cap")

    monkeypatch.setattr(
        embeddings_module,
        "configure_mps_memory_limit",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("allocator limit unavailable")),
    )
    provider = SentenceTransformerEmbeddingProvider(
        embedding_config(allow_explicit_hash_fallback=True),
        model_factory=factory,
        device_resolver=lambda _: "mps",
        resources=ResourcesConfig(),
    )

    with pytest.raises(MemoryError, match="memory cap could not be enforced"):
        provider.embed_query("must fail closed")

    assert factory_calls == 0
    assert provider.active_backend == "sentence_transformers"
