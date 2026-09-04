from papers_mcp.curated_index import parse_curated_markdown


def test_parses_multiple_table_shapes_and_links_artifacts() -> None:
    text = """
# Geometry
## Continuity
| name | field | what it gives | pdf-or-md |
|---|---|---|---|
| peters_vertex | splines | Vertex enclosure equations | peters_vertex.pdf |

| citation | tags | contribution | limitation | source |
|---|---|---|---|---|
| A Study (2025) | topology | Robust cells | slow | study_2025.pdf |

- See peters_vertex.pdf when deriving G1 constraints.
"""
    parsed = parse_curated_markdown(text, known_artifacts=["peters_vertex.pdf", "study_2025.pdf"])
    assert len(parsed.records) == 2
    assert parsed.records[0].heading_path == "Geometry > Continuity"
    assert parsed.artifact_to_paper_id["peters_vertex.pdf"] == "peters-vertex"
    assert "peters-vertex" in parsed.entries[-1].linked_paper_ids


def test_unresolved_rows_remain_searchable() -> None:
    text = """
# Gaps
| citation | reason |
|---|---|
| Missing Surface Paper | paywalled |
"""
    parsed = parse_curated_markdown(text)
    assert parsed.records[0].artifacts == []
    assert "paywalled" in parsed.entries[0].text
    assert parsed.entries[0].linked_paper_ids == []


def test_relative_artifact_paths_link_to_the_local_basename() -> None:
    text = """
# Geometry
| name | account |
|---|---|
| tensor_voting | Full account: docs/research/papers/tensor_voting.md |

| file | citation |
|---|---|
| `sweep-2026-06-18/02_DeFillet_SIGGRAPH2025.pdf` | Jiang et al. (2025) |
"""
    parsed = parse_curated_markdown(
        text,
        known_artifacts=[
            "tensor_voting.md",
            "sweep-2026-06-18/02_DeFillet_SIGGRAPH2025.pdf",
        ],
    )

    assert parsed.artifact_to_paper_id["tensor_voting.md"] == "tensor-voting"
    assert parsed.artifact_to_paper_id["02_DeFillet_SIGGRAPH2025.pdf"] == "02-defillet-siggraph2025"


def test_duplicate_alias_inventory_cannot_overwrite_canonical_paper_id() -> None:
    text = """
# Papers
| name | file |
|---|---|
| point2cad_liu_2024 | point2cad_liu_2024.pdf |

## Duplicate aliases
| sweep alias | canonical file |
|---|---|
| sweep/W1_Point2CAD.pdf | point2cad_liu_2024.pdf |
"""

    parsed = parse_curated_markdown(text)

    assert parsed.artifact_to_paper_id["point2cad_liu_2024.pdf"] == "point2cad-liu-2024"
    assert "point2cad-liu-2024" in parsed.entries[-1].linked_paper_ids


def test_markdown_link_destinations_in_bullets_remain_artifact_links() -> None:
    parsed = parse_curated_markdown("- Important source: [Paper](docs/note.md)\n")

    assert parsed.entries[0].text == "Important source: Paper\nartifacts: docs/note.md"
    assert parsed.entries[0].artifacts == ["docs/note.md"]
    assert parsed.entries[0].linked_paper_ids == ["note"]


def test_unique_basename_aliases_are_case_insensitive_and_keep_the_catalog_id() -> None:
    parsed = parse_curated_markdown(
        """
| name | file |
|---|---|
| canonical-paper | Archive/PAPER.PDF |
""",
        known_artifacts=["sources/paper.pdf"],
    )

    assert parsed.artifact_to_paper_id["Archive/PAPER.PDF"] == "canonical-paper"
    assert parsed.artifact_to_paper_id["sources/paper.pdf"] == "canonical-paper"
    assert parsed.artifact_to_paper_id["PAPER.PDF"] == "canonical-paper"
    assert parsed.artifact_to_paper_id["paper.pdf"] == "canonical-paper"


def test_casefold_colliding_basenames_do_not_publish_ambiguous_aliases() -> None:
    parsed = parse_curated_markdown(
        """
| name | file |
|---|---|
| first-paper | one/Shared.PDF |
| second-paper | two/shared.pdf |
""",
        known_artifacts=["one/shared.pdf", "two/SHARED.PDF"],
    )

    assert parsed.artifact_to_paper_id["one/shared.pdf"] == "first-paper"
    assert parsed.artifact_to_paper_id["two/SHARED.PDF"] == "second-paper"
    assert "Shared.PDF" not in parsed.artifact_to_paper_id
    assert "shared.pdf" not in parsed.artifact_to_paper_id


def test_explicit_bare_artifact_stays_unlinked_when_known_sources_share_its_basename() -> None:
    parsed = parse_curated_markdown(
        """
| name | file |
|---|---|
| shared | shared.pdf |
""",
        known_artifacts=["shared.pdf", "nested/SHARED.PDF"],
    )

    assert "shared.pdf" not in parsed.artifact_to_paper_id
    assert parsed.entries[0].linked_paper_ids == []
