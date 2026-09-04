from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class EvaluationQuery:
    query: str
    relevant_papers: tuple[str, ...]
    mode: str = "precision"
    label_source: str = "weak_index_label"
    notes: str = ""


def load_evaluation_queries(path: Path) -> list[EvaluationQuery]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency is part of the package
            raise RuntimeError("PyYAML is required for non-JSON evaluation files") from exc
        payload = yaml.safe_load(text)
    rows = payload.get("queries", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("evaluation file must contain a list or a {queries: [...]} object")
    queries: list[EvaluationQuery] = []
    for row in rows:
        if not isinstance(row, dict) or not str(row.get("query", "")).strip():
            raise ValueError("every evaluation row needs a non-empty query")
        labels = row.get("relevant_papers", [])
        queries.append(
            EvaluationQuery(
                query=str(row["query"]),
                relevant_papers=tuple(str(item) for item in labels),
                mode=str(row.get("mode", "precision")),
                label_source=str(row.get("label_source", "weak_index_label")),
                notes=str(row.get("notes", "")),
            )
        )
    return queries


def _unique_ranked_papers(results: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    ranked: list[str] = []
    for result in results:
        if isinstance(result, dict):
            paper_id = result.get("paper_id")
        else:
            paper_id = getattr(result, "paper_id", None)
        if paper_id and paper_id not in seen:
            seen.add(str(paper_id))
            ranked.append(str(paper_id))
    return ranked


def query_metrics(ranked_papers: list[str], relevant: set[str]) -> dict[str, float]:
    if not relevant:
        return {"recall_at_5": 0.0, "recall_at_10": 0.0, "mrr": 0.0, "ndcg_at_10": 0.0}

    def recall_at(k: int) -> float:
        return len(relevant.intersection(ranked_papers[:k])) / len(relevant)

    reciprocal_rank = 0.0
    for rank, paper_id in enumerate(ranked_papers, start=1):
        if paper_id in relevant:
            reciprocal_rank = 1.0 / rank
            break

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, paper_id in enumerate(ranked_papers[:10], start=1)
        if paper_id in relevant
    )
    ideal_count = min(10, len(relevant))
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return {
        "recall_at_5": recall_at(5),
        "recall_at_10": recall_at(10),
        "mrr": reciprocal_rank,
        "ndcg_at_10": dcg / ideal_dcg if ideal_dcg else 0.0,
    }


def evaluate(
    corpus: Any,
    queries: list[EvaluationQuery],
    *,
    system: str = "hybrid_rerank",
    top_k: int = 10,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    evaluation_depth = max(10, top_k)
    per_query: list[dict[str, Any]] = []
    for item in queries:
        results = corpus.search(
            item.query,
            mode=item.mode,
            top_k=evaluation_depth,
            pipeline=system,
            max_results_per_paper=1,
        )
        ranked = _unique_ranked_papers(results)
        metrics = query_metrics(ranked, set(item.relevant_papers))
        per_query.append(
            {
                "query": item.query,
                "mode": item.mode,
                "label_source": item.label_source,
                "relevant_papers": list(item.relevant_papers),
                "ranked_papers": ranked,
                **metrics,
            }
        )
    metric_names = ("recall_at_5", "recall_at_10", "mrr", "ndcg_at_10")
    aggregate = {
        name: sum(row[name] for row in per_query) / len(per_query) if per_query else 0.0
        for name in metric_names
    }
    return {
        "system": system,
        "evaluation_depth": evaluation_depth,
        "ranking_unit": "paper",
        "query_count": len(per_query),
        "labels": "Weak relevance labels derived from the human INDEX; not rigorous ground truth.",
        "aggregate": aggregate,
        "queries": per_query,
    }
