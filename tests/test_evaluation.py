from types import SimpleNamespace

import pytest

from papers_mcp.evaluation import EvaluationQuery, evaluate, query_metrics


def test_metrics() -> None:
    metrics = query_metrics(["other", "relevant", "second"], {"relevant", "second"})
    assert metrics["recall_at_5"] == 1.0
    assert metrics["mrr"] == 0.5
    assert metrics["ndcg_at_10"] == pytest.approx(
        (1 / __import__("math").log2(3) + 1 / __import__("math").log2(4))
        / (1 + 1 / __import__("math").log2(3))
    )


def test_evaluation_requests_a_ten_deep_unique_paper_ranking() -> None:
    calls = []

    class Corpus:
        def search(self, query: str, **kwargs):
            calls.append((query, kwargs))
            return [SimpleNamespace(paper_id=f"p{index}") for index in range(10)]

    result = evaluate(
        Corpus(),
        [EvaluationQuery("query", ("p9",))],
        top_k=3,
    )

    assert calls[0][1]["top_k"] == 10
    assert calls[0][1]["max_results_per_paper"] == 1
    assert result["evaluation_depth"] == 10
    assert result["queries"][0]["recall_at_10"] == 1.0
