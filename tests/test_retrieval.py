from papers_mcp.models import Candidate
from papers_mcp.retrieval import (
    diversify,
    normalize_math_search,
    reciprocal_rank_fusion,
    snippet_for,
)


def candidate(key: str, paper: str) -> Candidate:
    return Candidate(
        key=key,
        kind="chunk",
        entity_id=key,
        paper_id=paper,
        section_id=1,
        chunk_id=1,
        title=paper,
        authors=[],
        year=None,
        section_path="Method",
        page_start=None,
        page_end=None,
        source_kind="pdf",
        text=key,
    )


def test_rrf_combines_and_is_understandable() -> None:
    a, b, c = candidate("a", "p1"), candidate("b", "p2"), candidate("c", "p3")
    ranked = reciprocal_rank_fusion([[a, b], [b, c]], k=60)
    assert [item.key for item in ranked] == ["b", "a", "c"]
    assert ranked[0].score == 1 / 62 + 1 / 61


def test_rrf_keeps_additional_evidence_for_the_same_routed_paper() -> None:
    paper = candidate("paper:p1", "p1")
    paper.kind = "paper"
    paper.text = "Paper abstract."
    curated = candidate("paper:p1", "p1")
    curated.kind = "paper"
    curated.text = "Paper abstract."
    curated.routing_note = "vertex parity route"

    fused = reciprocal_rank_fusion([[paper], [curated]])

    assert fused[0].text == "Paper abstract."
    assert fused[0].routing_note == "vertex parity route"


def test_diversity_caps_results_per_paper() -> None:
    items = [candidate("a", "p1"), candidate("b", "p1"), candidate("c", "p2")]
    assert [item.key for item in diversify(items, top_k=3, max_per_paper=1)] == ["a", "c"]


def test_math_normalization_and_bounded_snippet() -> None:
    assert normalize_math_search("G¹ and G^1") == "G1 and G1"
    text = " ".join(f"word{i}" for i in range(500))
    snippet = snippet_for(text, "word300", max_words=100)
    assert "word300" in snippet
    assert len(snippet.split()) <= 102
