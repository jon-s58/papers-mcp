from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from .config import AppConfig, MCPConfig, load_config
from .embeddings import resolve_device

LOGGER = logging.getLogger(__name__)

TOOL_NAMES = (
    "search_papers",
    "research_search",
    "expand_context",
    "paper_outline",
    "read_section",
    "find_in_paper",
    "related_papers",
)

SERVER_INSTRUCTIONS = (
    "SEARCH → EXPAND → READ workflow: start with search_papers for a precise question or "
    "research_search for broad discovery. Follow each result's next_actions. For a section or "
    "chunk result, call expand_context with result_id, then read_section with paper_id and the "
    "numeric section_id. A paper-level result may be routed by an expert INDEX.md note; use "
    "paper_outline or find_in_paper first instead of assigning that note to an arbitrary source "
    "page. An unlinked curated result is an explicitly unresolved INDEX.md lead, not paper "
    "evidence. Use related_papers to compare approaches. Search snippets locate evidence; read "
    "the source section before implementing non-trivial mathematics."
)

_SEARCH_MODES = frozenset({"precision", "discovery", "exact"})
_CONTEXT_LEVELS = frozenset({"subsection", "section"})

SearchMode = Literal["precision", "discovery", "exact"]
ContextLevel = Literal["subsection", "section"]
NonEmptyText = Annotated[
    str,
    Field(min_length=1, description="Non-empty text; surrounding whitespace is ignored."),
]
TopK = Annotated[
    int,
    Field(strict=True, ge=1, le=20, description="Number of results to return (1-20)."),
]
SectionId = Annotated[
    int,
    Field(strict=True, ge=1, description="Numeric section ID from search or outline."),
]
SectionOffset = Annotated[
    int,
    Field(strict=True, ge=0, description="Token offset for section pagination."),
]
SectionTokenLimit = Annotated[
    int,
    Field(
        strict=True,
        ge=1,
        le=12000,
        description="Maximum section tokens to return (1-12000).",
    ),
]


class PageRangeOutput(BaseModel):
    """Known source-page range, or null bounds with label ``unknown``."""

    start: int | None = None
    end: int | None = None
    label: str = "unknown"


class SearchResultOutput(BaseModel):
    """One sourced retrieval result and its recommended follow-up calls."""

    model_config = ConfigDict(extra="allow")

    result_id: str | None = None
    kind: Literal["paper", "section", "chunk", "curated"] | None = None
    paper_id: str | None = None
    section_id: int | None = None
    chunk_id: int | None = None
    title: str | None = None
    authors: list[str] | None = None
    year: int | None = None
    section_path: str | None = None
    pages: PageRangeOutput | None = None
    source_kind: str | None = None
    rank: int | None = None
    score: float | None = None
    snippet: str | None = None
    routing_note: str | None = None
    next_actions: dict[str, Any] | None = None


class ToolErrorOutput(BaseModel):
    """Stable error envelope returned in structured MCP error content."""

    tool: str
    code: Literal["invalid_argument", "not_found", "service_error"]
    message: str
    details: dict[str, Any] | None = None


class SearchPapersOutput(BaseModel):
    ok: bool
    query: str | None = None
    mode: SearchMode | None = None
    results: list[SearchResultOutput] | None = None
    error: ToolErrorOutput | None = None


class ResearchSearchOutput(BaseModel):
    ok: bool
    query: str | None = None
    results: list[SearchResultOutput] | None = None
    error: ToolErrorOutput | None = None


class ExpandContextOutput(BaseModel):
    ok: bool
    result_id: str | None = None
    level: ContextLevel | None = None
    context: dict[str, Any] | None = None
    error: ToolErrorOutput | None = None


class OutlineSectionOutput(BaseModel):
    """One node in a paper's section tree."""

    model_config = ConfigDict(extra="allow")

    section_id: int | None = None
    parent_section_id: int | None = None
    heading: str | None = None
    heading_path: str | None = None
    level: int | None = None
    section_order: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    children: list[OutlineSectionOutput] | None = None


class PaperOutlinePayload(BaseModel):
    """Paper metadata plus flat and hierarchical section navigation."""

    model_config = ConfigDict(extra="allow")

    paper_id: str | None = None
    title: str | None = None
    authors: list[str] | None = None
    year: int | None = None
    doi: str | None = None
    abstract: str | None = None
    section_tree: list[OutlineSectionOutput] | None = None
    sections: list[OutlineSectionOutput] | None = None


class PaperOutlineOutput(BaseModel):
    ok: bool
    paper_id: str | None = None
    paper: PaperOutlinePayload | None = None
    error: ToolErrorOutput | None = None


class SectionPayload(BaseModel):
    """A paginated authoritative section response."""

    model_config = ConfigDict(extra="allow")

    paper_id: str | None = None
    title: str | None = None
    section_id: int | None = None
    section: str | None = None
    heading: str | None = None
    included_section_ids: list[int] | None = None
    page_start: int | None = None
    page_end: int | None = None
    offset: int | None = None
    next_offset: int | None = None
    total_tokens: int | None = None
    returned_tokens: int | None = None
    truncated: bool | None = None
    text: str | None = None


class ReadSectionOutput(BaseModel):
    ok: bool
    paper_id: str | None = None
    section_id: int | None = None
    offset: int | None = None
    section: SectionPayload | None = None
    error: ToolErrorOutput | None = None


class FindInPaperOutput(BaseModel):
    ok: bool
    paper_id: str | None = None
    query: str | None = None
    results: list[SearchResultOutput] | None = None
    error: ToolErrorOutput | None = None


class RelatedPaperOutput(BaseModel):
    """One paper-level related-literature candidate."""

    model_config = ConfigDict(extra="allow")

    paper_id: str | None = None
    title: str | None = None
    authors: list[str] | None = None
    year: int | None = None
    score: float | None = None
    source_kind: str | None = None
    snippet: str | None = None
    next_actions: dict[str, Any] | None = None


class RelatedPapersOutput(BaseModel):
    ok: bool
    paper_id: str | None = None
    papers: list[RelatedPaperOutput] | None = None
    error: ToolErrorOutput | None = None


READ_ONLY_TOOL_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


class _ToolValidationError(ValueError):
    pass


def _structured(value: Any) -> Any:
    """Convert service dataclasses and model objects to MCP-safe Python values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _structured(value.to_dict())
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _structured(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return _structured(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _structured(item) for key, item in value.items()}
    if isinstance(value, Iterable):
        return [_structured(item) for item in value]
    return str(value)


def _error(tool: str, code: str, message: str, **details: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "ok": False,
        "error": {"tool": tool, "code": code, "message": message},
    }
    if details:
        value["error"]["details"] = _structured(details)
    return value


def _mcp_result(value: Mapping[str, Any]) -> CallToolResult:
    """Preserve structured envelopes while exposing failures through MCP semantics."""

    payload = _structured(value)
    if not isinstance(payload, dict):  # pragma: no cover - adapter methods always return mappings
        raise TypeError("MCP tool response must be an object")
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            )
        ],
        structuredContent=payload,
        isError=payload.get("ok") is False,
    )


def _require_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _ToolValidationError(f"{name} must be a non-empty string")
    return value.strip()


def _require_integer(
    name: str,
    value: Any,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _ToolValidationError(f"{name} must be an integer")
    if value < minimum:
        raise _ToolValidationError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise _ToolValidationError(f"{name} must be at most {maximum}")
    return value


class MCPTools:
    """Thin, defensive MCP adapter around a ``ResearchCorpus``-compatible object."""

    def __init__(
        self,
        corpus: Any,
        *,
        max_top_k: int = 20,
        max_section_tokens: int = 12000,
    ) -> None:
        if max_top_k <= 0:
            raise ValueError("max_top_k must be positive")
        if max_section_tokens <= 0:
            raise ValueError("max_section_tokens must be positive")
        self.corpus = corpus
        self.max_top_k = max_top_k
        self.max_section_tokens = max_section_tokens

    def _corpus_method(self, *names: str) -> Any:
        for name in names:
            method = getattr(self.corpus, name, None)
            if callable(method):
                return method
        raise AttributeError(f"corpus does not implement {' or '.join(names)}")

    def _run(self, tool: str, operation: Any) -> dict[str, Any]:
        try:
            return operation()
        except _ToolValidationError as exc:
            return _error(tool, "invalid_argument", str(exc))
        except (KeyError, LookupError) as exc:
            message = str(exc.args[0]) if exc.args else "Requested item was not found"
            return _error(tool, "not_found", message)
        except Exception as exc:
            LOGGER.exception("MCP tool %s failed", tool)
            return _error(
                tool,
                "service_error",
                "The research corpus could not complete this request.",
                exception=type(exc).__name__,
            )

    def search_papers(
        self,
        query: str,
        mode: str = "precision",
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Start here for focused retrieval. Returns short sourced results with stable IDs.

        Use mode="precision" for a mathematical or implementation question, "exact" for a
        known term or quotation, and "discovery" for multiple approaches. After selecting a
        result, follow its next_actions. Expand section/chunk hits; for a paper-level expert-note
        route, use paper_outline or find_in_paper before read_section.
        """

        def operation() -> dict[str, Any]:
            clean_query = _require_text("query", query)
            clean_mode = _require_text("mode", mode).lower()
            if clean_mode not in _SEARCH_MODES:
                allowed = ", ".join(sorted(_SEARCH_MODES))
                raise _ToolValidationError(f"mode must be one of: {allowed}")
            bounded_top_k = _require_integer("top_k", top_k, minimum=1, maximum=self.max_top_k)
            method = self._corpus_method("search", "search_papers")
            results = _structured(method(clean_query, mode=clean_mode, top_k=bounded_top_k))
            return {
                "ok": True,
                "query": clean_query,
                "mode": clean_mode,
                "results": results,
            }

        return self._run("search_papers", operation)

    def research_search(self, query: str, top_k: int = 12) -> dict[str, Any]:
        """Explore a broad research problem and retrieve diverse papers and sections.

        Use this when you do not yet know the formal term or want competing approaches. Results
        are source candidates, not a generated literature answer. Follow each result's
        next_actions: expand section/chunk hits, or outline/search a paper-level routed hit, then
        read a complete source section before relying on equations or algorithms.
        """

        def operation() -> dict[str, Any]:
            clean_query = _require_text("query", query)
            bounded_top_k = _require_integer("top_k", top_k, minimum=1, maximum=self.max_top_k)
            results = _structured(
                self._corpus_method("research_search")(
                    clean_query,
                    top_k=bounded_top_k,
                )
            )
            return {"ok": True, "query": clean_query, "results": results}

        return self._run("research_search", operation)

    def expand_context(
        self,
        result_id: str,
        level: str = "section",
    ) -> dict[str, Any]:
        """Second step after search: expand one result into its surrounding source context.

        Copy result_id exactly from search_papers or research_search. Choose "subsection" for the
        smallest useful context or "section" for the full parent discussion. The response keeps
        paper_id and section_id so read_section can retrieve or paginate the authoritative text.
        """

        def operation() -> dict[str, Any]:
            clean_result_id = _require_text("result_id", result_id)
            clean_level = _require_text("level", level).lower()
            if clean_level not in _CONTEXT_LEVELS:
                allowed = ", ".join(sorted(_CONTEXT_LEVELS))
                raise _ToolValidationError(f"level must be one of: {allowed}")
            context = _structured(
                self._corpus_method("expand_context")(
                    clean_result_id,
                    level=clean_level,
                )
            )
            return {
                "ok": True,
                "result_id": clean_result_id,
                "level": clean_level,
                "context": context,
            }

        return self._run("expand_context", operation)

    def paper_outline(self, paper_id: str) -> dict[str, Any]:
        """List a paper's metadata and section tree, including numeric section IDs.

        Use this when a search result identifies a promising paper but you need another section.
        Choose a section_id from the outline and pass it with the same paper_id to read_section;
        do not copy a long heading when a stable numeric ID is available.
        """

        def operation() -> dict[str, Any]:
            clean_paper_id = _require_text("paper_id", paper_id)
            paper = _structured(self._corpus_method("outline", "paper_outline")(clean_paper_id))
            return {"ok": True, "paper_id": clean_paper_id, "paper": paper}

        return self._run("paper_outline", operation)

    def read_section(
        self,
        paper_id: str,
        section_id: int,
        offset: int = 0,
        max_tokens: int = 8000,
    ) -> dict[str, Any]:
        """Final reading step: retrieve authoritative section text by stable paper and section IDs.

        Get section_id from a search result, expand_context, or paper_outline. Start with offset=0.
        If the response indicates more text, call again with the next offset. max_tokens is bounded
        by the server, so use pagination instead of asking a search tool for giant snippets.
        """

        def operation() -> dict[str, Any]:
            clean_paper_id = _require_text("paper_id", paper_id)
            clean_section_id = _require_integer("section_id", section_id, minimum=1)
            clean_offset = _require_integer("offset", offset, minimum=0)
            clean_max_tokens = _require_integer(
                "max_tokens",
                max_tokens,
                minimum=1,
                maximum=self.max_section_tokens,
            )
            section = _structured(
                self._corpus_method("read_section")(
                    clean_paper_id,
                    clean_section_id,
                    offset=clean_offset,
                    max_tokens=clean_max_tokens,
                )
            )
            return {
                "ok": True,
                "paper_id": clean_paper_id,
                "section_id": clean_section_id,
                "offset": clean_offset,
                "section": section,
            }

        return self._run("read_section", operation)

    def find_in_paper(
        self,
        paper_id: str,
        query: str,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Search inside one already-selected paper for a term, equation, or concept.

        Supply paper_id from search results. This is narrower than search_papers and is useful for
        locating assumptions, derivations, constraints, numerical details, or failure cases. Use
        a returned section_id with read_section to inspect the complete source context.
        """

        def operation() -> dict[str, Any]:
            clean_paper_id = _require_text("paper_id", paper_id)
            clean_query = _require_text("query", query)
            bounded_top_k = _require_integer("top_k", top_k, minimum=1, maximum=self.max_top_k)
            results = _structured(
                self._corpus_method("find_in_paper")(
                    clean_paper_id,
                    clean_query,
                    top_k=bounded_top_k,
                )
            )
            return {
                "ok": True,
                "paper_id": clean_paper_id,
                "query": clean_query,
                "results": results,
            }

        return self._run("find_in_paper", operation)

    def related_papers(self, paper_id: str, top_k: int = 10) -> dict[str, Any]:
        """Find paper-level alternatives or companions to one known paper.

        Supply a paper_id from search results. This returns papers rather than repeated chunks, so
        use it to compare methods or broaden coverage. Search within or outline a returned paper
        before reading its relevant section.
        """

        def operation() -> dict[str, Any]:
            clean_paper_id = _require_text("paper_id", paper_id)
            bounded_top_k = _require_integer("top_k", top_k, minimum=1, maximum=self.max_top_k)
            papers = _structured(
                self._corpus_method("related_papers")(
                    clean_paper_id,
                    top_k=bounded_top_k,
                )
            )
            return {"ok": True, "paper_id": clean_paper_id, "papers": papers}

        return self._run("related_papers", operation)


def _mcp_server_class() -> type[Any]:
    """Return the v2 MCPServer class or the v1 FastMCP compatibility class."""

    try:
        from mcp.server import MCPServer
    except (ImportError, AttributeError):
        try:
            from mcp.server.fastmcp import FastMCP
        except ImportError as exc:  # pragma: no cover - package is a required runtime dependency
            raise RuntimeError(
                "The 'mcp' package is required to build the MCP server. Install project dependencies."
            ) from exc
        return FastMCP
    return MCPServer


def _config_for_injected_corpus(corpus: Any) -> AppConfig | None:
    config = getattr(corpus, "config", None)
    return config if isinstance(config, AppConfig) else None


def _log_startup_summary(config: AppConfig, corpus: Any) -> None:
    """Log useful process state to stderr without forcing either model to load."""

    embedding = getattr(corpus, "embedding", None)
    reranker = getattr(corpus, "reranker", None)
    embedding_backend = getattr(embedding, "active_backend", config.embedding.backend)
    reranker_backend = getattr(reranker, "active_backend", config.reranker.backend)
    device = getattr(embedding, "device", None)
    if not device:
        try:
            device = resolve_device(config.embedding.device)
        except Exception as exc:  # pragma: no cover - hardware/dependency dependent
            device = f"unavailable ({type(exc).__name__})"

    LOGGER.info("Embedding backend: %s", embedding_backend)
    LOGGER.info("Embedding model: %s", config.embedding.model)
    LOGGER.info("Reranker: %s (%s)", config.reranker.model, reranker_backend)
    LOGGER.info("Device: %s", device)
    resources = getattr(config, "resources", None)
    if resources is not None:
        LOGGER.info(
            "Memory caps: process %.2f GiB / MPS allocator %.2f GiB",
            resources.max_process_memory_gb,
            resources.mps_memory_limit_gb,
        )

    counts: Mapping[str, Any] = {}
    status_method = getattr(corpus, "status", None)
    if callable(status_method):
        try:
            status = status_method()
            if isinstance(status, Mapping) and isinstance(status.get("counts"), Mapping):
                counts = status["counts"]
        except Exception as exc:  # pragma: no cover - startup should survive diagnostics
            LOGGER.warning("Corpus startup status unavailable: %s", exc)
    LOGGER.info(
        "Corpus: %s papers / %s chunks",
        counts.get("papers", "unknown"),
        counts.get("chunks", "unknown"),
    )


def _register_transport_tools(server: Any, tools: MCPTools) -> None:
    """Register typed MCP wrappers around the ergonomic direct-call adapter."""

    def search_papers(
        query: NonEmptyText,
        mode: SearchMode = "precision",
        top_k: TopK = 10,
    ) -> Annotated[CallToolResult, SearchPapersOutput]:
        return _mcp_result(tools.search_papers(query, mode=mode, top_k=top_k))

    def research_search(
        query: NonEmptyText,
        top_k: TopK = 12,
    ) -> Annotated[CallToolResult, ResearchSearchOutput]:
        return _mcp_result(tools.research_search(query, top_k=top_k))

    def expand_context(
        result_id: NonEmptyText,
        level: ContextLevel = "section",
    ) -> Annotated[CallToolResult, ExpandContextOutput]:
        return _mcp_result(tools.expand_context(result_id, level=level))

    def paper_outline(
        paper_id: NonEmptyText,
    ) -> Annotated[CallToolResult, PaperOutlineOutput]:
        return _mcp_result(tools.paper_outline(paper_id))

    def read_section(
        paper_id: NonEmptyText,
        section_id: SectionId,
        offset: SectionOffset = 0,
        max_tokens: SectionTokenLimit = 8000,
    ) -> Annotated[CallToolResult, ReadSectionOutput]:
        return _mcp_result(
            tools.read_section(
                paper_id,
                section_id,
                offset=offset,
                max_tokens=max_tokens,
            )
        )

    def find_in_paper(
        paper_id: NonEmptyText,
        query: NonEmptyText,
        top_k: TopK = 10,
    ) -> Annotated[CallToolResult, FindInPaperOutput]:
        return _mcp_result(tools.find_in_paper(paper_id, query, top_k=top_k))

    def related_papers(
        paper_id: NonEmptyText,
        top_k: TopK = 10,
    ) -> Annotated[CallToolResult, RelatedPapersOutput]:
        return _mcp_result(tools.related_papers(paper_id, top_k=top_k))

    configured_top_k = Annotated[
        int,
        Field(
            strict=True,
            ge=1,
            le=tools.max_top_k,
            description=f"Number of results to return (1-{tools.max_top_k}).",
        ),
    ]
    for wrapper in (search_papers, research_search, find_in_paper, related_papers):
        wrapper.__annotations__["top_k"] = configured_top_k
    configured_section_limit = Annotated[
        int,
        Field(
            strict=True,
            ge=1,
            le=tools.max_section_tokens,
            description=(f"Maximum section tokens to return (1-{tools.max_section_tokens})."),
        ),
    ]
    read_section.__annotations__["max_tokens"] = configured_section_limit

    search_papers.__defaults__ = ("precision", min(10, tools.max_top_k))
    research_search.__defaults__ = (min(12, tools.max_top_k),)
    read_section.__defaults__ = (0, min(8000, tools.max_section_tokens))
    find_in_paper.__defaults__ = (min(10, tools.max_top_k),)
    related_papers.__defaults__ = (min(10, tools.max_top_k),)

    wrappers = {
        "search_papers": search_papers,
        "research_search": research_search,
        "expand_context": expand_context,
        "paper_outline": paper_outline,
        "read_section": read_section,
        "find_in_paper": find_in_paper,
        "related_papers": related_papers,
    }
    for tool_name in TOOL_NAMES:
        server.tool(
            name=tool_name,
            description=getattr(MCPTools, tool_name).__doc__,
            annotations=READ_ONLY_TOOL_ANNOTATIONS,
            structured_output=True,
        )(wrappers[tool_name])


def build_mcp_server(
    config_path: str | os.PathLike[str] | None = None,
    corpus: Any | None = None,
) -> Any:
    """Build an MCP server with exactly the seven public research tools.

    Passing ``corpus`` avoids opening a database or loading models and is useful for tests and
    embedding the server. Without one, the configured ``ResearchCorpus`` is constructed lazily here.
    """

    config: AppConfig | None
    if config_path is not None:
        config = load_config(config_path)
    elif corpus is not None:
        config = _config_for_injected_corpus(corpus)
    else:
        config = load_config()

    if corpus is None:
        assert config is not None
        from .service import ResearchCorpus

        corpus = ResearchCorpus(config)

    if config is not None:
        _log_startup_summary(config, corpus)

    mcp_config = config.mcp if config is not None else MCPConfig()
    tools = MCPTools(
        corpus,
        max_top_k=mcp_config.max_top_k,
        max_section_tokens=mcp_config.max_section_tokens,
    )
    server_class = _mcp_server_class()
    server = server_class(name="Papers Research", instructions=SERVER_INSTRUCTIONS)
    _register_transport_tools(server, tools)
    return server


__all__ = [
    "MCPTools",
    "SERVER_INSTRUCTIONS",
    "TOOL_NAMES",
    "build_mcp_server",
]
