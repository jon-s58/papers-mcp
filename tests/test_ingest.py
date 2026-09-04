from pathlib import Path

from papers_mcp.curated_index import parse_curated_markdown
from papers_mcp.ingest import YEAR_RE, _artifact_key_for_source, _usable_title
from papers_mcp.models import SourceDocument


def test_curated_filename_years_are_detected_between_underscores() -> None:
    match = YEAR_RE.search("rabbani_point_cloud_smoothness_segmentation_2006")

    assert match is not None
    assert match.group(1) == "2006"


def test_equation_fragment_is_not_accepted_as_a_pdf_title() -> None:
    assert not _usable_title("(E,,) becomes", Path("peters_vertex_enclosure_1989.pdf"))


def test_exact_relative_artifact_wins_over_an_ambiguous_bare_basename(
    tmp_path: Path,
) -> None:
    curated = parse_curated_markdown(
        """
| name | file |
|---|---|
| unsafe-bare-route | shared.pdf |
| exact-route | FOLDER/SHARED.pdf |
"""
    )
    root_source = SourceDocument(tmp_path / "shared.PDF", "pdf", "root-hash")
    nested_source = SourceDocument(
        tmp_path / "folder" / "Shared.PDF",
        "pdf",
        "nested-hash",
    )
    sources = [root_source, nested_source]

    assert (
        _artifact_key_for_source(curated, nested_source, tmp_path, sources) == "FOLDER/SHARED.pdf"
    )
    assert _artifact_key_for_source(curated, root_source, tmp_path, sources) == "./shared.PDF"


def test_unique_artifact_basename_matching_is_case_insensitive(tmp_path: Path) -> None:
    curated = parse_curated_markdown(
        """
| name | file |
|---|---|
| routed-paper | Archive/PAPER.PDF |
"""
    )
    source = SourceDocument(tmp_path / "sources" / "paper.pdf", "pdf", "paper-hash")

    assert _artifact_key_for_source(curated, source, tmp_path, [source]) == "Archive/PAPER.PDF"
