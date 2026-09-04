from papers_mcp.cards import build_research_card
from papers_mcp.models import Paper, Section


def test_card_is_deterministic_and_labels_authority() -> None:
    paper = Paper(id="p", title="G1 Spline Fitting", abstract="Tangent continuity constraints")
    sections = [Section(paper_id="p", heading="Method", heading_path="3 Method", text="")]
    card = build_research_card(paper, sections, ["vertex enclosure", "vertex enclosure"])
    assert card["curated_notes"] == "vertex enclosure"
    assert "continuity" in card["topics"]
    assert "authoritative" in str(card["authority"])
