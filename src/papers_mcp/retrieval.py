from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import replace

from .config import RetrievalConfig
from .embeddings import EmbeddingProvider, embedding_provider_identity
from .models import Candidate, SearchResult
from .query_expansion import QueryExpansionProvider
from .reranker import RerankerProvider

VALID_MODES = {"precision", "discovery", "exact"}
VALID_PIPELINES = {"bm25", "dense", "hybrid", "hybrid_rerank"}
SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def normalize_math_search(text: str) -> str:
    value = text.translate(SUPERSCRIPTS)
    value = re.sub(r"\b([CG])\s*\^\s*\{?([0-9])\}?\b", r"\1\2", value, flags=re.I)
    value = re.sub(r"\b([CG])\s+([0-9])\b", r"\1\2", value, flags=re.I)
    return " ".join(value.split())


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Candidate]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[Candidate]:
    if k <= 0:
        raise ValueError("RRF k must be positive")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("RRF weights must match ranking lists")
    scores: dict[str, float] = {}
    representatives: dict[str, Candidate] = {}
    first_seen: dict[str, int] = {}
    sequence = 0
    for ranking, weight in zip(rankings, weights, strict=True):
        seen_in_ranking: set[str] = set()
        for rank, candidate in enumerate(ranking, start=1):
            if candidate.key in seen_in_ranking:
                continue
            seen_in_ranking.add(candidate.key)
            if candidate.key not in representatives:
                representatives[candidate.key] = candidate
                first_seen[candidate.key] = sequence
                sequence += 1
            else:
                representative = representatives[candidate.key]
                incoming = candidate.text.strip()
                existing = representative.text.strip()
                if incoming and incoming not in existing:
                    if existing and incoming.startswith(existing):
                        incoming = incoming[len(existing) :].strip()
                    if incoming:
                        representatives[candidate.key] = replace(
                            representative,
                            text="\n\n".join(part for part in (existing, incoming) if part),
                            section_path=(representative.section_path or candidate.section_path),
                        )
                if candidate.routing_note and not representative.routing_note:
                    representatives[candidate.key] = replace(
                        representatives[candidate.key],
                        routing_note=candidate.routing_note,
                    )
            scores[candidate.key] = scores.get(candidate.key, 0.0) + float(weight) / (k + rank)
    ordered = sorted(scores, key=lambda key: (-scores[key], first_seen[key]))
    return [replace(representatives[key], score=scores[key]) for key in ordered]


def _normalize_scores(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if math.isclose(low, high):
        return [1.0] * len(values)
    return [(value - low) / (high - low) for value in values]


def diversify(
    candidates: Iterable[Candidate],
    *,
    top_k: int,
    max_per_paper: int,
) -> list[Candidate]:
    selected: list[Candidate] = []
    counts: dict[str, int] = {}
    for candidate in candidates:
        paper_key = candidate.paper_id or f"{candidate.kind}:{candidate.entity_id}"
        if counts.get(paper_key, 0) >= max_per_paper:
            continue
        counts[paper_key] = counts.get(paper_key, 0) + 1
        selected.append(candidate)
        if len(selected) >= top_k:
            break
    return selected


def snippet_for(text: str, query: str, *, max_words: int = 300) -> str:
    words = list(re.finditer(r"\S+", text))
    if len(words) <= max_words:
        return text.strip()
    normalized_text = normalize_math_search(text).casefold()
    normalized_query = normalize_math_search(query).casefold().strip()
    match_at = normalized_text.find(normalized_query) if normalized_query else -1
    if match_at >= 0:
        center = next((index for index, word in enumerate(words) if word.start() >= match_at), 0)
        start_word = max(0, center - max_words // 3)
    else:
        start_word = 0
    end_word = min(len(words), start_word + max_words)
    start_char = words[start_word].start()
    end_char = words[end_word - 1].end()
    snippet = text[start_char:end_char].strip()
    return ("… " if start_word else "") + snippet + (" …" if end_word < len(words) else "")


class RetrievalEngine:
    def __init__(
        self,
        database,
        embedding: EmbeddingProvider,
        reranker: RerankerProvider,
        expansion: QueryExpansionProvider,
        config: RetrievalConfig,
    ) -> None:
        self.database = database
        self.embedding = embedding
        self.reranker = reranker
        self.expansion = expansion
        self.config = config
        self._query_vectors: dict[str, tuple[str, list[float]]] = {}

    def _model_fingerprint(self, dimensions: int | None = None) -> str | None:
        active = getattr(self.embedding, "active_backend", None)
        if dimensions is None and (
            active == "hash" or self.embedding.__class__.__name__ == "HashEmbeddingProvider"
        ):
            dimensions = getattr(
                getattr(self.embedding, "_fallback", self.embedding), "dimensions", None
            )
        if dimensions is None:
            return None
        return embedding_provider_identity(self.embedding, dimensions)[1]

    def _lexical(
        self,
        kind: str,
        query: str,
        limit: int,
        *,
        paper_id: str | None = None,
        exact: bool = False,
    ) -> list[Candidate]:
        return self.database.lexical_search(
            kind,
            query,
            limit,
            paper_id=paper_id,
            exact=exact,
        )

    def _dense(
        self,
        kind: str,
        query: str,
        limit: int,
        *,
        paper_id: str | None = None,
    ) -> list[Candidate]:
        cached = self._query_vectors.get(query)
        if cached is not None:
            cached_fingerprint, vector = cached
            if self._model_fingerprint(len(vector)) != cached_fingerprint:
                self._query_vectors.pop(query, None)
                cached = None
        if cached is None:
            vector = self.embedding.embed_query(query)
            fingerprint = self._model_fingerprint(len(vector))
            if fingerprint is None:
                return []
            if len(self._query_vectors) >= 32:
                self._query_vectors.pop(next(iter(self._query_vectors)))
            self._query_vectors[query] = (fingerprint, vector)
        else:
            fingerprint = cached[0]
        return self.database.dense_search(
            kind,
            vector,
            limit,
            model=fingerprint,
            paper_id=paper_id,
        )

    def _rankings(
        self,
        query: str,
        *,
        mode: str,
        pipeline: str,
        paper_id: str | None = None,
    ) -> tuple[list[list[Candidate]], list[float]]:
        lexical_enabled = pipeline in {"bm25", "hybrid", "hybrid_rerank"}
        dense_enabled = pipeline in {"dense", "hybrid", "hybrid_rerank"}
        rankings: list[list[Candidate]] = []
        weights: list[float] = []
        normalized = normalize_math_search(query)

        if mode == "exact":
            if lexical_enabled:
                if paper_id is None:
                    rankings.append(
                        self._lexical("paper", query, self.config.bm25_candidates, exact=True)
                    )
                    weights.append(2.75)
                rankings.append(
                    self._lexical(
                        "chunk", query, self.config.bm25_candidates, paper_id=paper_id, exact=True
                    )
                )
                weights.append(2.5)
                if normalized.casefold() != query.casefold():
                    rankings.append(
                        self._lexical(
                            "chunk",
                            normalized,
                            self.config.bm25_candidates,
                            paper_id=paper_id,
                            exact=True,
                        )
                    )
                    weights.append(2.0)
            if dense_enabled:
                rankings.append(
                    self._dense("chunk", query, self.config.dense_candidates, paper_id=paper_id)
                )
                weights.append(0.35)
            return rankings, weights

        kinds = ["chunk"] if mode == "precision" else ["paper", "section", "chunk", "curated"]
        for kind in kinds:
            if paper_id is not None and kind in {"paper", "curated"}:
                continue
            kind_weight = {"paper": 1.15, "section": 1.1, "chunk": 1.0, "curated": 0.8}[kind]
            if lexical_enabled:
                lexical = self._lexical(
                    kind,
                    normalized,
                    self.config.bm25_candidates,
                    paper_id=paper_id,
                )
                if kind == "curated":
                    unresolved, paper_boosts = self.database.route_curated_candidates(lexical)
                    if unresolved:
                        rankings.append(unresolved)
                        weights.append(kind_weight)
                    if paper_boosts:
                        rankings.append(paper_boosts)
                        weights.append(1.2)
                else:
                    rankings.append(lexical)
                    weights.append(kind_weight)
            if dense_enabled:
                dense = self._dense(
                    kind,
                    query,
                    self.config.dense_candidates,
                    paper_id=paper_id,
                )
                if kind == "curated":
                    unresolved, paper_boosts = self.database.route_curated_candidates(dense)
                    if unresolved:
                        rankings.append(unresolved)
                        weights.append(kind_weight)
                    if paper_boosts:
                        rankings.append(paper_boosts)
                        weights.append(1.2)
                else:
                    rankings.append(dense)
                    weights.append(kind_weight)
        return rankings, weights

    def _rerank(
        self,
        query: str,
        candidates: list[Candidate],
        *,
        enabled: bool,
        exact: bool,
    ) -> list[Candidate]:
        pool = candidates[: self.config.rerank_candidates]
        if not pool:
            return []
        if enabled:
            documents = [
                f"DOCUMENT: {item.title}\nSECTION: {item.section_path}\n"
                f"CONTENT:\n{item.text[:12000]}"
                + (
                    f"\n\nCURATED ROUTING NOTE:\n{item.routing_note[:4000]}"
                    if item.routing_note
                    else ""
                )
                for item in pool
            ]
            raw = self.reranker.score(query, documents)
            normalized = _normalize_scores(raw)
            rrf = _normalize_scores([item.score for item in pool])
            rescored = [
                replace(
                    item,
                    rerank_score=raw[index],
                    score=0.85 * normalized[index] + 0.15 * rrf[index],
                )
                for index, item in enumerate(pool)
            ]
        else:
            rescored = pool

        if exact:
            needle = normalize_math_search(query).casefold()
            rescored = [
                replace(
                    item,
                    score=item.score + 2.0,
                )
                for item in rescored
                if needle
                and needle
                in normalize_math_search(
                    f"{item.title}\n{item.text}"
                    if item.kind == "paper"
                    else f"{item.section_path}\n{item.text}"
                ).casefold()
            ]
        return sorted(rescored, key=lambda item: (-item.score, item.key))

    def _result_id(self, candidate: Candidate) -> str:
        revision = self.database.get_revision()
        return f"r{revision}:{candidate.kind}:{candidate.entity_id}"

    def _as_results(
        self,
        candidates: list[Candidate],
        *,
        query: str,
    ) -> list[SearchResult]:
        results: list[SearchResult] = []
        seen_result_ids: set[str] = set()
        for item in candidates:
            if item.section_id is not None and not item.text.strip():
                preview = self.database.read_section(
                    item.paper_id,
                    item.section_id,
                    offset=0,
                    max_tokens=max(64, self.config.snippet_words * 2),
                )
                if not preview["total_tokens"]:
                    continue
                item = replace(
                    item,
                    text=preview["text"],
                    page_start=preview["page_start"],
                    page_end=preview["page_end"],
                )
            result_id = self._result_id(item)
            if result_id in seen_result_ids:
                continue
            seen_result_ids.add(result_id)
            actions: dict[str, object] = {}
            if item.section_id is not None:
                actions["expand_context"] = {"result_id": result_id, "level": "section"}
                actions["read_section"] = {
                    "paper_id": item.paper_id,
                    "section_id": item.section_id,
                }
            if item.paper_id:
                actions["paper_outline"] = {"paper_id": item.paper_id}
                actions["find_in_paper"] = {
                    "paper_id": item.paper_id,
                    "query": query,
                }
            rank = len(results) + 1
            results.append(
                SearchResult(
                    result_id=result_id,
                    kind=item.kind,
                    paper_id=item.paper_id,
                    section_id=item.section_id,
                    chunk_id=item.chunk_id,
                    title=item.title,
                    authors=item.authors,
                    year=item.year,
                    section_path=item.section_path,
                    page_start=item.page_start,
                    page_end=item.page_end,
                    source_kind=item.source_kind,
                    rank=rank,
                    score=round(float(item.score), 6),
                    snippet=snippet_for(item.text, query, max_words=self.config.snippet_words),
                    routing_note=item.routing_note,
                    next_actions=actions,
                )
            )
        return results

    def search(
        self,
        query: str,
        *,
        mode: str = "precision",
        top_k: int = 10,
        pipeline: str = "hybrid_rerank",
        paper_id: str | None = None,
        max_results_per_paper: int | None = None,
    ) -> list[SearchResult]:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(VALID_MODES)}")
        if pipeline not in VALID_PIPELINES:
            raise ValueError(f"pipeline must be one of {sorted(VALID_PIPELINES)}")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if max_results_per_paper is not None and max_results_per_paper <= 0:
            raise ValueError("max_results_per_paper must be positive")
        rankings, weights = self._rankings(query, mode=mode, pipeline=pipeline, paper_id=paper_id)
        fused = reciprocal_rank_fusion(rankings, k=self.config.rrf_k, weights=weights)
        ranked = self._rerank(
            query,
            fused,
            enabled=pipeline == "hybrid_rerank" and mode != "exact",
            exact=mode == "exact",
        )
        max_per_paper = (
            max_results_per_paper
            or {
                "precision": self.config.precision_max_results_per_paper,
                "discovery": self.config.discovery_max_results_per_paper,
                "exact": self.config.exact_max_results_per_paper,
            }[mode]
        )
        final = diversify(ranked, top_k=top_k, max_per_paper=max_per_paper)
        return self._as_results(final, query=query)

    def research_search(
        self,
        query: str,
        *,
        top_k: int = 12,
        pipeline: str = "hybrid_rerank",
    ) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        expanded = [query, *self.expansion.expand(query)]
        rankings: list[list[Candidate]] = []
        weights: list[float] = []
        for query_index, variant in enumerate(expanded):
            variant_rankings, variant_weights = self._rankings(
                variant,
                mode="discovery",
                pipeline=pipeline,
            )
            expansion_weight = 1.0 if query_index == 0 else 0.75
            rankings.extend(variant_rankings)
            weights.extend(weight * expansion_weight for weight in variant_weights)
        fused = reciprocal_rank_fusion(rankings, k=self.config.rrf_k, weights=weights)
        ranked = self._rerank(
            query,
            fused,
            enabled=pipeline == "hybrid_rerank",
            exact=False,
        )
        final = diversify(
            ranked,
            top_k=top_k,
            max_per_paper=self.config.discovery_max_results_per_paper,
        )
        return self._as_results(final, query=query)
