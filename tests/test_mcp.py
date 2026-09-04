from __future__ import annotations

import asyncio
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from papers_mcp.mcp_server import (
    SERVER_INSTRUCTIONS,
    TOOL_NAMES,
    MCPTools,
    _log_startup_summary,
    build_mcp_server,
)
from papers_mcp.models import SearchResult


class FakeCorpus:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def _record(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self.calls.append((name, args, kwargs))

    def search(self, query: str, *, mode: str, top_k: int) -> list[SearchResult]:
        self._record("search", (query,), {"mode": mode, "top_k": top_k})
        return [
            SearchResult(
                result_id="result_1",
                kind="section",
                paper_id="paper_1",
                section_id=11,
                chunk_id=17,
                title="Spline Networks",
                authors=["A. Author"],
                year=2025,
                section_path="4.2 Vertex Enclosure",
                page_start=7,
                page_end=8,
                source_kind="pdf",
                rank=1,
                score=0.94,
                snippet="Compatibility conditions.",
                next_actions={"expand_context": "result_1"},
            )
        ]

    def research_search(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        self._record("research_search", (query,), {"top_k": top_k})
        return [{"result_id": "result_2", "paper_id": "paper_2"}]

    def expand_context(self, result_id: str, *, level: str) -> dict[str, Any]:
        self._record("expand_context", (result_id,), {"level": level})
        return {"paper_id": "paper_1", "section_id": 11, "text": "Full context"}

    def outline(self, paper_id: str) -> dict[str, Any]:
        self._record("outline", (paper_id,), {})
        return {"title": "Spline Networks", "sections": [{"section_id": 11}]}

    def read_section(
        self,
        paper_id: str,
        section_id: int,
        *,
        offset: int,
        max_tokens: int,
    ) -> dict[str, Any]:
        self._record(
            "read_section",
            (paper_id, section_id),
            {"offset": offset, "max_tokens": max_tokens},
        )
        return {"text": "Authoritative section", "next_offset": None}

    def find_in_paper(
        self,
        paper_id: str,
        query: str,
        *,
        top_k: int,
    ) -> list[dict[str, Any]]:
        self._record("find_in_paper", (paper_id, query), {"top_k": top_k})
        return [{"paper_id": paper_id, "section_id": 11, "snippet": query}]

    def related_papers(self, paper_id: str, *, top_k: int) -> list[dict[str, Any]]:
        self._record("related_papers", (paper_id,), {"top_k": top_k})
        return [{"paper_id": "paper_related", "score": 0.8}]


def test_wrapper_calls_all_seven_corpus_operations_and_structures_results() -> None:
    corpus = FakeCorpus()
    tools = MCPTools(corpus)

    search = tools.search_papers(" vertex enclosure ", mode="PRECISION", top_k=4)
    research = tools.research_search("smooth patch network", top_k=5)
    expanded = tools.expand_context("result_1", level="subsection")
    outline = tools.paper_outline("paper_1")
    section = tools.read_section("paper_1", 11, offset=20, max_tokens=1000)
    found = tools.find_in_paper("paper_1", "constraint", top_k=3)
    related = tools.related_papers("paper_1", top_k=2)

    assert search["ok"] is True
    assert search["results"][0]["pages"] == {"start": 7, "end": 8, "label": "7-8"}
    assert search["results"][0]["result_id"] == "result_1"
    assert research["results"][0]["paper_id"] == "paper_2"
    assert expanded["context"]["section_id"] == 11
    assert outline["paper"]["sections"] == [{"section_id": 11}]
    assert section["section"]["text"] == "Authoritative section"
    assert found["results"][0]["snippet"] == "constraint"
    assert related["papers"][0]["paper_id"] == "paper_related"
    assert [call[0] for call in corpus.calls] == [
        "search",
        "research_search",
        "expand_context",
        "outline",
        "read_section",
        "find_in_paper",
        "related_papers",
    ]
    assert corpus.calls[0] == (
        "search",
        ("vertex enclosure",),
        {"mode": "precision", "top_k": 4},
    )


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (lambda tools: tools.search_papers("", top_k=1), "query must be a non-empty"),
        (lambda tools: tools.search_papers("query", mode="wide"), "mode must be one of"),
        (lambda tools: tools.research_search("query", top_k=21), "top_k must be at most"),
        (lambda tools: tools.expand_context("result", level="paper"), "level must be one of"),
        (lambda tools: tools.paper_outline("   "), "paper_id must be a non-empty"),
        (lambda tools: tools.read_section("paper", 0), "section_id must be at least"),
        (
            lambda tools: tools.read_section("paper", 1, max_tokens=12001),
            "max_tokens must be at most",
        ),
        (lambda tools: tools.find_in_paper("paper", "query", top_k=0), "top_k must be at least"),
        (lambda tools: tools.related_papers("paper", top_k=True), "top_k must be an integer"),
    ],
)
def test_wrapper_returns_structured_validation_errors(response: Any, message: str) -> None:
    corpus = FakeCorpus()
    result = response(MCPTools(corpus))
    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_argument"
    assert message in result["error"]["message"]
    assert corpus.calls == []


def test_wrapper_returns_structured_not_found_and_service_errors() -> None:
    class FailingCorpus(FakeCorpus):
        def outline(self, paper_id: str) -> dict[str, Any]:
            raise KeyError("unknown paper")

        def related_papers(self, paper_id: str, *, top_k: int) -> list[dict[str, Any]]:
            raise RuntimeError("database unavailable")

    tools = MCPTools(FailingCorpus())
    missing = tools.paper_outline("missing")
    failed = tools.related_papers("paper")

    assert missing == {
        "ok": False,
        "error": {
            "tool": "paper_outline",
            "code": "not_found",
            "message": "unknown paper",
        },
    }
    assert failed["ok"] is False
    assert failed["error"]["code"] == "service_error"
    assert failed["error"]["details"] == {"exception": "RuntimeError"}
    assert "database unavailable" not in failed["error"]["message"]


def test_server_instructions_lead_with_search_expand_read_workflow() -> None:
    opening = SERVER_INSTRUCTIONS[:512].casefold()
    assert "search" in opening
    assert "expand" in opening
    assert "read" in opening
    assert opening.index("search") < opening.index("expand") < opening.index("read")


def test_startup_summary_reports_models_device_and_corpus_without_loading_models(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = SimpleNamespace(
        embedding=SimpleNamespace(
            backend="sentence_transformers",
            model="Qwen/Qwen3-Embedding-8B",
            device="auto",
        ),
        reranker=SimpleNamespace(
            backend="sentence_transformers",
            model="Qwen/Qwen3-Reranker-8B",
        ),
    )
    corpus = SimpleNamespace(
        embedding=SimpleNamespace(active_backend="sentence_transformers", device=None),
        reranker=SimpleNamespace(active_backend="sentence_transformers"),
        status=lambda: {"counts": {"papers": 8, "chunks": 196}},
    )
    monkeypatch.setattr("papers_mcp.mcp_server.resolve_device", lambda _value: "mps")

    with caplog.at_level("INFO", logger="papers_mcp.mcp_server"):
        _log_startup_summary(config, corpus)

    assert "Embedding backend: sentence_transformers" in caplog.text
    assert "Embedding model: Qwen/Qwen3-Embedding-8B" in caplog.text
    assert "Reranker: Qwen/Qwen3-Reranker-8B (sentence_transformers)" in caplog.text
    assert "Device: mps" in caplog.text
    assert "Corpus: 8 papers / 196 chunks" in caplog.text


def test_build_server_registers_exactly_seven_tools_when_mcp_is_installed() -> None:
    pytest.importorskip("mcp")
    server = build_mcp_server(corpus=FakeCorpus())
    registered = asyncio.run(server.list_tools())
    assert {tool.name for tool in registered} == set(TOOL_NAMES)
    assert len(registered) == 7
    assert server.instructions[:512] == SERVER_INSTRUCTIONS[:512]
    descriptions = {tool.name: tool.description for tool in registered}
    assert "Start here" in descriptions["search_papers"]
    assert "Final reading step" in descriptions["read_section"]

    def _input_schema(t):
        return getattr(t, "input_schema", getattr(t, "inputSchema", None))

    def _output_schema(t):
        return getattr(t, "output_schema", getattr(t, "outputSchema", None))

    def _is_error(res):
        return getattr(res, "is_error", getattr(res, "isError", None))

    def _structured_content(res):
        return getattr(res, "structured_content", getattr(res, "structuredContent", None))

    schemas = {tool.name: tool for tool in registered}
    search_schema = _input_schema(schemas["search_papers"])["properties"]
    assert search_schema["mode"]["enum"] == ["precision", "discovery", "exact"]
    assert search_schema["top_k"]["minimum"] == 1
    assert search_schema["top_k"]["maximum"] == 20
    context_schema = _input_schema(schemas["expand_context"])["properties"]
    assert context_schema["level"]["enum"] == ["subsection", "section"]
    section_schema = _input_schema(schemas["read_section"])["properties"]
    assert section_schema["section_id"]["minimum"] == 1
    assert section_schema["offset"]["minimum"] == 0
    assert section_schema["max_tokens"]["maximum"] == 12000

    assert "results" in _output_schema(schemas["search_papers"])["properties"]
    assert "context" in _output_schema(schemas["expand_context"])["properties"]
    assert "paper" in _output_schema(schemas["paper_outline"])["properties"]
    assert "section" in _output_schema(schemas["read_section"])["properties"]
    assert "papers" in _output_schema(schemas["related_papers"])["properties"]


@pytest.mark.asyncio
async def test_stdio_transport_calls_all_seven_tools_and_marks_execution_errors() -> None:
    mcp = pytest.importorskip("mcp")
    from mcp.client.stdio import stdio_client

    child = textwrap.dedent(
        """
        from papers_mcp.mcp_server import build_mcp_server

        class FixtureCorpus:
            def search(self, query, *, mode, top_k):
                return [{
                    "result_id": "r1:section:7",
                    "kind": "section",
                    "paper_id": "paper-1",
                    "section_id": 7,
                    "title": "Fixture Paper",
                    "section_path": "Methods > Constraint",
                    "snippet": query,
                }]

            def research_search(self, query, *, top_k):
                return [{"result_id": "r1:paper:paper-1", "paper_id": "paper-1"}]

            def expand_context(self, result_id, *, level):
                return {"paper_id": "paper-1", "section_id": 7, "text": "context"}

            def outline(self, paper_id):
                if paper_id == "missing":
                    raise KeyError("unknown paper")
                return {
                    "paper_id": paper_id,
                    "title": "Fixture Paper",
                    "section_tree": [{"section_id": 7, "heading": "Methods"}],
                }

            def read_section(self, paper_id, section_id, *, offset, max_tokens):
                return {
                    "paper_id": paper_id,
                    "section_id": section_id,
                    "offset": offset,
                    "text": "authoritative section",
                    "next_offset": None,
                }

            def find_in_paper(self, paper_id, query, *, top_k):
                return [{"paper_id": paper_id, "section_id": 7, "snippet": query}]

            def related_papers(self, paper_id, *, top_k):
                return [{"paper_id": "paper-2", "title": "Related Fixture"}]

        build_mcp_server(corpus=FixtureCorpus()).run()
        """
    )
    parameters = mcp.StdioServerParameters(
        command=sys.executable,
        args=["-c", child],
        cwd=Path(__file__).resolve().parents[1],
    )
    calls = (
        ("search_papers", {"query": "G1", "mode": "exact", "top_k": 1}),
        ("research_search", {"query": "smooth patches", "top_k": 1}),
        (
            "expand_context",
            {"result_id": "r1:section:7", "level": "section"},
        ),
        ("paper_outline", {"paper_id": "paper-1"}),
        (
            "read_section",
            {"paper_id": "paper-1", "section_id": 7, "offset": 0, "max_tokens": 100},
        ),
        (
            "find_in_paper",
            {"paper_id": "paper-1", "query": "constraint", "top_k": 1},
        ),
        ("related_papers", {"paper_id": "paper-1", "top_k": 1}),
    )

    async with asyncio.timeout(15):
        async with stdio_client(parameters) as (read, write):
            async with mcp.ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                assert [tool.name for tool in listed.tools] == list(TOOL_NAMES)

                def _is_error(res):
                    return getattr(res, "is_error", getattr(res, "isError", None))

                def _structured_content(res):
                    return getattr(res, "structured_content", getattr(res, "structuredContent", None))

                for name, arguments in calls:
                    result = await session.call_tool(name, arguments)
                    assert _is_error(result) is False
                    assert _structured_content(result) is not None
                    assert _structured_content(result)["ok"] is True

                schema_error = await session.call_tool(
                    "search_papers",
                    {"query": "G1", "mode": "unsupported", "top_k": 1},
                )
                assert _is_error(schema_error) is True

                strict_integer_error = await session.call_tool(
                    "related_papers",
                    {"paper_id": "paper-1", "top_k": True},
                )
                assert _is_error(strict_integer_error) is True

                not_found = await session.call_tool(
                    "paper_outline",
                    {"paper_id": "missing"},
                )
                assert _is_error(not_found) is True
                assert _structured_content(not_found) == {
                    "ok": False,
                    "error": {
                        "tool": "paper_outline",
                        "code": "not_found",
                        "message": "unknown paper",
                    },
                }
