from __future__ import annotations

import logging
from typing import Any

import pytest

from papers_mcp.config import RerankerConfig, ResourcesConfig
from papers_mcp.reranker import (
    CrossEncoderReranker,
    LexicalReranker,
    create_reranker,
)


def reranker_config(**overrides: Any) -> RerankerConfig:
    values: dict[str, Any] = {
        "backend": "sentence_transformers",
        "model": "Qwen/Qwen3-Reranker-4B",
        "device": "auto",
        "precision": "auto",
        "quantization": False,
        "batch_size": 1,
        "max_length": 2048,
        "allow_explicit_lexical_fallback": True,
        "instruction": "Prefer exact mathematical procedures.",
    }
    values.update(overrides)
    return RerankerConfig(**values)


def test_lexical_reranker_prefers_overlap_and_is_deterministic() -> None:
    reranker = LexicalReranker()
    documents = [
        "G1 continuity and vertex enclosure for spline patches",
        "image caption generation with a transformer",
    ]
    first = reranker.score("G1 spline continuity", documents)
    second = reranker.score("G1 spline continuity", documents)
    assert first == second
    assert first[0] > first[1]
    assert reranker.rerank("G1 spline continuity", documents, top_k=1) == [(0, first[0])]


def test_cross_encoder_is_lazy_reused_and_applies_instruction() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeCrossEncoder:
        def __init__(self) -> None:
            self.predictions: list[tuple[list[tuple[str, str]], dict[str, Any]]] = []

        def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> list[float]:
            value = (0.2, 0.9)[len(self.predictions) % 2]
            self.predictions.append((pairs, kwargs))
            return [value]

    model = FakeCrossEncoder()

    def factory(name: str, **kwargs: Any) -> FakeCrossEncoder:
        calls.append((name, kwargs))
        return model

    provider = CrossEncoderReranker(
        reranker_config(), model_factory=factory, device_resolver=lambda _: "mps"
    )
    assert not provider.is_loaded
    scores = provider.score("surface fitting", ["document one", "document two"])

    assert scores == [0.2, 0.9]
    assert provider.rerank("surface fitting", ["document one", "document two"]) == [
        (1, 0.9),
        (0, 0.2),
    ]
    assert len(calls) == 1
    assert calls[0] == (
        "Qwen/Qwen3-Reranker-4B",
        {
            "device": "mps",
            "max_length": 2048,
            "prompts": {
                "papers_mcp": "Prefer exact mathematical procedures.",
            },
            "default_prompt_name": "papers_mcp",
            "config_kwargs": {"use_cache": False, "attn_implementation": "sdpa"},
        },
    )
    assert model.predictions[0][0][0] == ("surface fitting", "document one")
    assert model.predictions[0][1] == {"batch_size": 1, "show_progress_bar": False}


def test_lexical_fallback_is_logged_and_only_activated_when_allowed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def unavailable(*args: Any, **kwargs: Any) -> None:
        raise OSError("model unavailable")

    provider = CrossEncoderReranker(
        reranker_config(), model_factory=unavailable, device_resolver=lambda _: "cpu"
    )
    with caplog.at_level(logging.WARNING):
        scores = provider.score("NURBS fitting", ["NURBS surface fitting", "unrelated"])
    assert provider.active_backend == "lexical"
    assert scores[0] > scores[1]
    assert "explicitly allowed fallback" in caplog.text

    strict = CrossEncoderReranker(
        reranker_config(allow_explicit_lexical_fallback=False),
        model_factory=unavailable,
        device_resolver=lambda _: "cpu",
    )
    with pytest.raises(RuntimeError, match="fallback is not allowed"):
        strict.score("NURBS fitting", ["document"])


def test_explicit_lexical_backend_is_configuration_gated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        provider = create_reranker(reranker_config(backend="lexical"))
    assert isinstance(provider, LexicalReranker)
    assert "explicitly configured" in caplog.text

    with pytest.raises(ValueError, match="requires allow_explicit_lexical_fallback"):
        create_reranker(reranker_config(backend="lexical", allow_explicit_lexical_fallback=False))


def test_quantization_is_rejected() -> None:
    with pytest.raises(ValueError, match="quantization is disabled"):
        CrossEncoderReranker(reranker_config(quantization=True))


def test_reranker_revision_is_passed_to_loader() -> None:
    calls: list[dict[str, Any]] = []

    class FakeCrossEncoder:
        def predict(self, pairs, **kwargs):
            return [1.0 for _ in pairs]

    def factory(name: str, **kwargs: Any) -> FakeCrossEncoder:
        calls.append(kwargs)
        return FakeCrossEncoder()

    provider = CrossEncoderReranker(
        reranker_config(revision="commit-reranker"),
        model_factory=factory,
        device_resolver=lambda _: "cpu",
    )
    provider.score("query", ["document"])

    assert calls[0]["revision"] == "commit-reranker"


def test_reranker_requests_are_split_into_explicit_bounded_batches() -> None:
    class FakeCrossEncoder:
        def __init__(self) -> None:
            self.batches: list[list[tuple[str, str]]] = []

        def predict(self, pairs, **kwargs):
            self.batches.append(list(pairs))
            return [float(index) for index, _ in enumerate(pairs, start=1)]

    model = FakeCrossEncoder()
    provider = CrossEncoderReranker(
        reranker_config(batch_size=1),
        model_factory=lambda *_args, **_kwargs: model,
        device_resolver=lambda _: "cpu",
        resources=ResourcesConfig(release_memory_after_batch=False),
    )

    scores = provider.score("query", ["one", "two", "three"])

    assert scores == [1.0, 1.0, 1.0]
    assert model.batches == [
        [("query", "one")],
        [("query", "two")],
        [("query", "three")],
    ]


def test_failure_to_install_mps_cap_never_activates_lexical_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory_calls = 0

    def factory(*_args, **_kwargs):
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("model must not load without the cap")

    monkeypatch.setattr(
        "papers_mcp.reranker.configure_mps_memory_limit",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("allocator limit unavailable")),
    )
    provider = CrossEncoderReranker(
        reranker_config(allow_explicit_lexical_fallback=True),
        model_factory=factory,
        device_resolver=lambda _: "mps",
        resources=ResourcesConfig(),
    )

    with pytest.raises(MemoryError, match="memory cap could not be enforced"):
        provider.score("query", ["must fail closed"])

    assert factory_calls == 0
    assert provider.active_backend == "sentence_transformers"
