from __future__ import annotations

import re
from typing import Any

from .config import AppConfig, load_config
from .database import CorpusDatabase
from .embeddings import EmbeddingProvider, create_embedding_provider
from .models import SearchResult, Section
from .query_expansion import QueryExpansionProvider, create_query_expansion_provider
from .reranker import RerankerProvider, create_reranker
from .retrieval import RetrievalEngine, reciprocal_rank_fusion, snippet_for

RESULT_ID_RE = re.compile(r"^r(\d+):(paper|section|chunk|curated):(.+)$")
MAJOR_SECTION_RE = re.compile(
    r"^(?:\*{0,2})\s*(?:\d+\.?\s+|abstract\b|introduction\b|background\b|"
    r"related\s+work\b|methods?(?:ology)?\b|results?\b|discussion\b|"
    r"conclusions?\b|references\b)",
    re.IGNORECASE,
)
NUMBERED_SUBSECTION_RE = re.compile(r"^(?:\*{0,2})\s*\d+(?:\.\d+)+\.?\s+", re.I)


class StaleResultError(LookupError):
    pass


class ResearchCorpus:
    """Ordinary service layer shared by the CLI and MCP adapter."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        database: CorpusDatabase | None = None,
        embedding: EmbeddingProvider | None = None,
        reranker: RerankerProvider | None = None,
        expansion: QueryExpansionProvider | None = None,
    ) -> None:
        self.config = config or load_config()
        self.config.ensure_output_dirs()
        self.database = database or CorpusDatabase(self.config.paths.database)
        self._owns_database = database is None
        self.embedding = embedding or create_embedding_provider(
            self.config.embedding,
            self.config.resources,
        )
        self.reranker = reranker or create_reranker(
            self.config.reranker,
            self.config.resources,
        )
        self.expansion = expansion or create_query_expansion_provider(self.config.query_expansion)
        self.retrieval = RetrievalEngine(
            self.database,
            self.embedding,
            self.reranker,
            self.expansion,
            self.config.retrieval,
        )

    @classmethod
    def from_config(cls, path: str | None = None) -> ResearchCorpus:
        return cls(load_config(path))

    def close(self) -> None:
        if self._owns_database:
            self.database.close()

    def __enter__(self) -> ResearchCorpus:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def search(
        self,
        query: str,
        *,
        mode: str = "precision",
        top_k: int = 10,
        pipeline: str = "hybrid_rerank",
        max_results_per_paper: int | None = None,
    ) -> list[SearchResult]:
        return self.retrieval.search(
            query,
            mode=mode,
            top_k=min(top_k, self.config.mcp.max_top_k),
            pipeline=pipeline,
            max_results_per_paper=max_results_per_paper,
        )

    search_papers = search

    def research_search(
        self,
        query: str,
        *,
        top_k: int = 12,
        pipeline: str = "hybrid_rerank",
    ) -> list[SearchResult]:
        return self.retrieval.research_search(
            query,
            top_k=min(top_k, self.config.mcp.max_top_k),
            pipeline=pipeline,
        )

    def outline(self, paper_id: str) -> dict[str, Any]:
        return self.database.paper_outline(paper_id)

    paper_outline = outline

    def read_section(
        self,
        paper_id: str,
        section_id: int,
        *,
        offset: int = 0,
        max_tokens: int = 8000,
    ) -> dict[str, Any]:
        bounded = min(max_tokens, self.config.mcp.max_section_tokens)
        return self.database.read_section(
            paper_id,
            section_id,
            offset=offset,
            max_tokens=bounded,
        )

    def find_in_paper(
        self,
        paper_id: str,
        query: str,
        *,
        top_k: int = 10,
    ) -> list[SearchResult]:
        if self.database.get_paper(paper_id) is None:
            raise KeyError(f"unknown paper: {paper_id}")
        return self.retrieval.search(
            query,
            mode="precision",
            top_k=min(top_k, self.config.mcp.max_top_k),
            pipeline="hybrid_rerank",
            paper_id=paper_id,
        )

    def _resolve_result(self, result_id: str) -> tuple[str, str]:
        match = RESULT_ID_RE.fullmatch(result_id.strip())
        if not match:
            raise KeyError(
                "invalid result_id; copy the complete ID returned by search_papers or research_search"
            )
        revision, kind, entity_id = match.groups()
        current = self.database.get_revision()
        if int(revision) != current:
            raise StaleResultError(
                f"result_id is from corpus revision {revision}; current revision is {current}. Search again."
            )
        return kind, entity_id

    def _logical_parent(self, section: Section) -> Section:
        heading = section.heading.strip()
        if MAJOR_SECTION_RE.match(heading):
            return section
        if section.level <= 1:
            return section
        if section.level == 2 and not NUMBERED_SUBSECTION_RE.match(heading):
            parent = (
                self.database.get_section_by_id(section.parent_section_id)
                if section.parent_section_id is not None
                else None
            )
            if parent is None or not MAJOR_SECTION_RE.match(parent.heading.strip()):
                return section
        current = section
        visited: set[int] = set()
        while current.parent_section_id is not None:
            if current.id is not None:
                if current.id in visited:
                    break
                visited.add(current.id)
            parent = self.database.get_section_by_id(current.parent_section_id)
            if parent is None:
                break
            if parent.level <= 2 or MAJOR_SECTION_RE.match(parent.heading.strip()):
                return parent
            current = parent
        return current

    def expand_context(
        self,
        result_id: str,
        *,
        level: str = "section",
    ) -> dict[str, Any]:
        if level not in {"subsection", "section"}:
            raise ValueError("level must be subsection or section")
        kind, entity_id = self._resolve_result(result_id)
        if kind == "chunk":
            try:
                chunk = self.database.get_chunk_by_id(int(entity_id))
            except ValueError as exc:
                raise KeyError(f"invalid chunk result ID: {result_id}") from exc
            if chunk is None or chunk.section_id is None:
                raise KeyError(f"chunk no longer exists: {entity_id}")
            section = self.database.get_section_by_id(chunk.section_id)
        elif kind == "section":
            try:
                section = self.database.get_section_by_id(int(entity_id))
            except ValueError as exc:
                raise KeyError(f"invalid section result ID: {result_id}") from exc
        elif kind == "paper":
            return self.paper_outline(entity_id)
        else:
            raise KeyError(
                "curated-index results are catalog evidence, not source sections; "
                "use a linked paper action when present or locate and ingest the unresolved source"
            )
        if section is None or section.id is None:
            raise KeyError(f"source section no longer exists for {result_id}")
        target = section if level == "subsection" else self._logical_parent(section)
        page = self.database.read_section(
            target.paper_id,
            target.id,
            offset=0,
            max_tokens=self.config.mcp.max_section_tokens,
        )
        next_action = None
        if page["next_offset"] is not None:
            next_action = {
                "tool": "read_section",
                "paper_id": target.paper_id,
                "section_id": target.id,
                "offset": page["next_offset"],
            }
        return {
            **page,
            "level": level,
            "next_action": next_action,
        }

    def related_papers(self, paper_id: str, *, top_k: int = 10) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        paper = self.database.get_paper(paper_id)
        if paper is None:
            raise KeyError(f"unknown paper: {paper_id}")
        notes = [
            entry.text
            for entry in self.database.get_curated_entries()
            if paper_id in entry.linked_paper_ids
        ]
        limit = top_k + 12
        rankings = []

        abstract_query = "\n\n".join(part for part in (paper.title, paper.abstract) if part)
        abstract_vector = self.embedding.embed_query(abstract_query) if abstract_query else None
        abstract_model = (
            self.retrieval._model_fingerprint(len(abstract_vector))
            if abstract_vector is not None
            else None
        )
        notes_query = "\n".join(notes)
        note_vector = self.embedding.embed_query(notes_query) if notes_query else None
        note_model = (
            self.retrieval._model_fingerprint(len(note_vector)) if note_vector is not None else None
        )
        # A lazy production provider can activate its explicitly configured
        # fallback during the first encode, so resolve the selector afterwards.
        if abstract_model:
            source_vector = self.database.get_embedding_for_model(
                "paper",
                paper_id,
                abstract_model,
            )
            if source_vector is not None:
                rankings.append(
                    self.database.dense_search(
                        "paper",
                        source_vector,
                        limit,
                        model=abstract_model,
                    )
                )

        if abstract_query and abstract_vector is not None:
            rankings.append(
                self.database.lexical_search(
                    "paper",
                    abstract_query,
                    limit,
                    exact=False,
                )
            )
            rankings.append(
                self.database.dense_search(
                    "paper",
                    abstract_vector,
                    limit,
                    model=abstract_model,
                )
            )

        if notes and note_vector is not None:
            rankings.append(
                self.database.dense_search(
                    "paper",
                    note_vector,
                    limit,
                    model=note_model,
                )
            )
            note_hits = self.database.lexical_search(
                "curated",
                notes_query,
                limit,
                exact=False,
            )
            _, routed_notes = self.database.route_curated_candidates(note_hits)
            rankings.append(routed_notes)

        rankings.append(self.database.shared_curated_paper_candidates(paper_id, limit))
        candidates = reciprocal_rank_fusion(
            [ranking for ranking in rankings if ranking],
            k=self.config.retrieval.rrf_k,
        )
        results: list[dict[str, Any]] = []
        for item in candidates:
            if item.paper_id == paper_id or item.paper_id is None:
                continue
            results.append(
                {
                    "paper_id": item.paper_id,
                    "title": item.title,
                    "authors": item.authors,
                    "year": item.year,
                    "score": round(float(item.score), 6),
                    "source_kind": item.source_kind,
                    "snippet": snippet_for(item.text, abstract_query, max_words=180),
                    "next_actions": {
                        "paper_outline": {"paper_id": item.paper_id},
                        "find_in_paper": {
                            "paper_id": item.paper_id,
                            "query": abstract_query,
                        },
                    },
                }
            )
            if len(results) >= top_k:
                break
        return results

    def status(self) -> dict[str, Any]:
        return {
            "revision": self.database.get_revision(),
            "counts": self.database.counts(),
            "embedding_backend": getattr(
                self.embedding, "active_backend", self.config.embedding.backend
            ),
            "embedding_model": self.config.embedding.model,
            "reranker_backend": getattr(
                self.reranker, "active_backend", self.config.reranker.backend
            ),
            "reranker_model": self.config.reranker.model,
        }
