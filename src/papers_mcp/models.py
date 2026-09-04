from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

SourceKind = Literal["pdf", "markdown_reference"]
ResultKind = Literal["paper", "section", "chunk", "curated"]


@dataclass(slots=True)
class Paper:
    id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    abstract: str = ""
    source_path: str = ""
    source_kind: SourceKind = "pdf"
    markdown_path: str | None = None
    content_hash: str = ""
    extraction_backend: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(slots=True)
class Section:
    paper_id: str
    heading: str
    heading_path: str
    text: str
    level: int = 1
    section_order: int = 0
    page_start: int | None = None
    page_end: int | None = None
    parent_index: int | None = None
    parent_section_id: int | None = None
    id: int | None = None


@dataclass(slots=True)
class Chunk:
    paper_id: str
    heading_path: str
    text: str
    token_count: int
    chunk_index: int = 0
    page_start: int | None = None
    page_end: int | None = None
    section_index: int | None = None
    section_id: int | None = None
    id: int | None = None


@dataclass(slots=True)
class CuratedEntry:
    heading_path: str
    text: str
    linked_paper_ids: list[str] = field(default_factory=list)
    source_line: int | None = None
    entry_type: str = "note"
    artifacts: list[str] = field(default_factory=list)
    id: int | None = None


@dataclass(slots=True)
class ExtractedDocument:
    markdown: str
    backend: str
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    abstract: str = ""
    page_count: int | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Candidate:
    key: str
    kind: ResultKind
    entity_id: str
    paper_id: str | None
    section_id: int | None
    chunk_id: int | None
    title: str
    authors: list[str]
    year: int | None
    section_path: str
    page_start: int | None
    page_end: int | None
    source_kind: str
    text: str
    score: float = 0.0
    lexical_score: float | None = None
    dense_score: float | None = None
    rerank_score: float | None = None
    routing_note: str | None = None


@dataclass(slots=True)
class SearchResult:
    result_id: str
    kind: ResultKind
    paper_id: str | None
    section_id: int | None
    chunk_id: int | None
    title: str
    authors: list[str]
    year: int | None
    section_path: str
    page_start: int | None
    page_end: int | None
    source_kind: str
    rank: int
    score: float
    snippet: str
    routing_note: str | None = None
    next_actions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.page_start is None:
            label = "unknown"
        elif self.page_end is None or self.page_end == self.page_start:
            label = str(self.page_start)
        else:
            label = f"{self.page_start}-{self.page_end}"
        value["pages"] = {
            "start": self.page_start,
            "end": self.page_end,
            "label": label,
        }
        value.pop("page_start")
        value.pop("page_end")
        return value


@dataclass(slots=True)
class IngestFailure:
    source_path: str
    stage: str
    error: str


@dataclass(slots=True)
class IngestReport:
    found: int = 0
    indexed: int = 0
    skipped: int = 0
    duplicates: int = 0
    failed: int = 0
    failures: list[IngestFailure] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SourceDocument:
    path: Path
    source_kind: SourceKind
    content_hash: str
    paper_id_hint: str | None = None
