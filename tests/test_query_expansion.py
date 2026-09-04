from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from papers_mcp.config import QueryExpansionConfig
from papers_mcp.query_expansion import (
    NoOpQueryExpansionProvider,
    OpenAICompatibleQueryExpansionProvider,
    create_query_expansion_provider,
    parse_expansions,
)


def expansion_config(**overrides: Any) -> QueryExpansionConfig:
    values: dict[str, Any] = {
        "enabled": True,
        "provider": "openai_compatible",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "local-research-model",
        "api_key_env": "TEST_EXPANSION_API_KEY",
        "timeout_seconds": 7,
    }
    values.update(overrides)
    return QueryExpansionConfig(**values)


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.data = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def read(self) -> bytes:
        return self.data


def test_openai_compatible_expansion_posts_chat_request_and_deduplicates() -> None:
    captured: dict[str, Any] = {}
    response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        [
                            "G1 continuity extraordinary vertices",
                            "vertex enclosure spline patch network",
                            "our weird topology regions",
                            "G1 continuity extraordinary vertices",
                        ]
                    )
                }
            }
        ]
    }

    def opener(request: Any, *, timeout: float) -> FakeResponse:
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse(response)

    provider = OpenAICompatibleQueryExpansionProvider(
        expansion_config(),
        opener=opener,
        environ={"TEST_EXPANSION_API_KEY": "secret"},
    )
    result = provider.expand("our weird topology regions")

    assert result == [
        "G1 continuity extraordinary vertices",
        "vertex enclosure spline patch network",
    ]
    assert captured["url"] == "http://127.0.0.1:11434/v1/chat/completions"
    assert captured["timeout"] == 7.0
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["payload"]["model"] == "local-research-model"
    assert captured["payload"]["temperature"] == 0.1


def test_parser_accepts_json_code_fences_and_bullets() -> None:
    assert parse_expansions(
        '```json\n["surface intersection topology", "trim curve congruence"]\n```',
        "original",
    ) == ["surface intersection topology", "trim curve congruence"]
    assert parse_expansions(
        "- G1 continuity\n2. vertex enclosure\nnot a list item",
        "original",
    ) == ["G1 continuity", "vertex enclosure"]


def test_query_expansion_failure_is_a_logged_noop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def failing_opener(*args: Any, **kwargs: Any) -> None:
        raise TimeoutError("local endpoint did not respond")

    provider = OpenAICompatibleQueryExpansionProvider(expansion_config(), opener=failing_opener)
    with caplog.at_level(logging.WARNING):
        assert provider.expand("surface fitting") == []
    assert "continuing with the original query" in caplog.text


def test_disabled_missing_or_unknown_provider_is_noop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert isinstance(
        create_query_expansion_provider(expansion_config(enabled=False)),
        NoOpQueryExpansionProvider,
    )
    with caplog.at_level(logging.WARNING):
        missing = create_query_expansion_provider(expansion_config(base_url=""))
        unknown = create_query_expansion_provider(expansion_config(provider="mystery"))
    assert isinstance(missing, NoOpQueryExpansionProvider)
    assert isinstance(unknown, NoOpQueryExpansionProvider)
    assert missing.expand("query") == []
    assert unknown.expand("query") == []
    assert "base_url is empty" in caplog.text
    assert "Unsupported query expansion provider" in caplog.text


def test_empty_query_does_not_call_endpoint() -> None:
    def should_not_run(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("network opener should not be called")

    provider = OpenAICompatibleQueryExpansionProvider(expansion_config(), opener=should_not_run)
    assert provider.expand("   ") == []
