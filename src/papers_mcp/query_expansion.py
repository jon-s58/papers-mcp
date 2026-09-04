from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from .config import QueryExpansionConfig

LOGGER = logging.getLogger(__name__)

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s*")


class QueryExpansionProvider(ABC):
    """Optional query-expansion interface. Empty output means use the original query."""

    @abstractmethod
    def expand(self, query: str) -> list[str]:
        """Return additional queries; never require callers to replace the original."""


class NoOpQueryExpansionProvider(QueryExpansionProvider):
    def expand(self, query: str) -> list[str]:
        return []


def _chat_completions_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


def _deduplicate(values: list[Any], original: str, limit: int) -> list[str]:
    original_key = " ".join(original.casefold().split())
    seen = {original_key}
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = " ".join(value.strip().strip("\"'").split())
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
        if len(result) >= limit:
            break
    return result


def parse_expansions(content: str, original: str, *, limit: int = 4) -> list[str]:
    """Parse JSON-first model output, with a conservative bullet-line fallback."""

    if limit <= 0:
        return []
    stripped = _CODE_FENCE_RE.sub("", content.strip()).strip()
    values: list[Any]
    try:
        decoded = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        values = [
            _LIST_PREFIX_RE.sub("", line).strip()
            for line in stripped.splitlines()
            if _LIST_PREFIX_RE.match(line)
        ]
    else:
        if isinstance(decoded, list):
            values = decoded
        elif isinstance(decoded, dict):
            candidate = decoded.get("queries", decoded.get("expansions", []))
            values = candidate if isinstance(candidate, list) else []
        else:
            values = []
    return _deduplicate(values, original, limit)


def _default_opener(request: urllib.request.Request, *, timeout: float) -> Any:
    return urllib.request.urlopen(request, timeout=timeout)


class OpenAICompatibleQueryExpansionProvider(QueryExpansionProvider):
    """Lazy HTTP client for a configured local OpenAI-compatible endpoint."""

    def __init__(
        self,
        config: QueryExpansionConfig,
        *,
        opener: Callable[..., Any] | None = None,
        environ: dict[str, str] | None = None,
        max_expansions: int = 4,
    ) -> None:
        if config.timeout_seconds <= 0:
            raise ValueError("Query expansion timeout_seconds must be positive")
        if max_expansions <= 0:
            raise ValueError("max_expansions must be positive")
        self.config = config
        self._opener = opener or _default_opener
        self._environ = os.environ if environ is None else environ
        self.max_expansions = max_expansions

    def _payload(self, query: str) -> dict[str, Any]:
        prompt = (
            "Expand the academic research query into up to four concise alternatives. "
            "Use terminology from computational geometry, CAD, optimization, and mathematical "
            "algorithm design when relevant. Include genuinely useful synonyms or formal names, "
            "not paraphrases with identical wording. Return only a JSON array of strings.\n\n"
            f"Query: {query}"
        )
        payload: dict[str, Any] = {
            "messages": [
                {
                    "role": "system",
                    "content": "You produce precise academic search-query expansions.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
        }
        if self.config.model:
            payload["model"] = self.config.model
        return payload

    def expand(self, query: str) -> list[str]:
        if not query.strip() or not self.config.base_url.strip():
            return []

        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        api_key = self._environ.get(self.config.api_key_env, "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            _chat_completions_url(self.config.base_url),
            data=json.dumps(self._payload(query)).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with self._opener(request, timeout=float(self.config.timeout_seconds)) as response:
                decoded = json.loads(response.read().decode("utf-8"))
            choices = decoded.get("choices", [])
            if not choices:
                raise ValueError("response has no choices")
            first = choices[0]
            message = first.get("message", {}) if isinstance(first, dict) else {}
            content = message.get("content") if isinstance(message, dict) else None
            if content is None and isinstance(first, dict):
                content = first.get("text")
            if not isinstance(content, str):
                raise ValueError("response choice has no text content")
            return parse_expansions(content, query, limit=self.max_expansions)
        except Exception as exc:
            LOGGER.warning(
                "Query expansion failed against %s (%s: %s); continuing with the original query",
                self.config.base_url,
                type(exc).__name__,
                exc,
            )
            return []


def create_query_expansion_provider(
    config: QueryExpansionConfig,
) -> QueryExpansionProvider:
    if not config.enabled:
        return NoOpQueryExpansionProvider()
    provider = config.provider.strip().lower().replace("-", "_")
    if provider in {"openai_compatible", "openai"}:
        if not config.base_url.strip():
            LOGGER.warning(
                "Query expansion is enabled but base_url is empty; expansion will be a no-op"
            )
            return NoOpQueryExpansionProvider()
        return OpenAICompatibleQueryExpansionProvider(config)
    LOGGER.warning(
        "Unsupported query expansion provider %s; expansion will be a no-op",
        config.provider,
    )
    return NoOpQueryExpansionProvider()


build_query_expansion_provider = create_query_expansion_provider


__all__ = [
    "NoOpQueryExpansionProvider",
    "OpenAICompatibleQueryExpansionProvider",
    "QueryExpansionProvider",
    "build_query_expansion_provider",
    "create_query_expansion_provider",
    "parse_expansions",
]
