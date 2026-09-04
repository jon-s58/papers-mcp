from __future__ import annotations

from papers_mcp.chunker import chunk_section, chunk_sections, estimate_token_count
from papers_mcp.config import ChunksConfig
from papers_mcp.models import Section


def _section(text: str, *, paper_id: str = "paper", order: int = 0) -> Section:
    return Section(
        paper_id=paper_id,
        heading="Method",
        heading_path="Method > Derivation",
        text=text,
        section_order=order,
        page_start=1,
        page_end=2,
    )


def test_equation_stays_with_immediate_explanation() -> None:
    section = _section(
        "We minimize the following fitting objective.\n\n"
        "$$\nE(S)=\\sum_i ||S(u_i,v_i)-p_i||^2 + \\lambda E_{fair}(S).\n$$\n\n"
        "where lambda controls fairness and each p_i is a sample point.\n\n"
        + "A later implementation paragraph gives numerical details. "
        * 8
    )
    chunks = chunk_section(section, ChunksConfig(target_tokens=45, min_tokens=15, max_tokens=70))

    equation_chunk = next(chunk for chunk in chunks if "E(S)=" in chunk.text)
    assert "We minimize" in equation_chunk.text
    assert "where lambda controls fairness" in equation_chunk.text
    assert equation_chunk.text.count("$$") == 2
    assert equation_chunk.heading_path == section.heading_path


def test_fenced_code_and_table_blocks_are_never_split() -> None:
    code_lines = "\n".join(f"step_{index} = value_{index}" for index in range(25))
    section = _section(
        f"Algorithm.\n\n```python\n{code_lines}\n```\n\n"
        "| variable | meaning |\n|---|---|\n| x | position |\n| n | normal |\n"
    )
    chunks = chunk_section(section, ChunksConfig(target_tokens=20, min_tokens=8, max_tokens=30))

    code_chunks = [chunk for chunk in chunks if "```python" in chunk.text]
    assert len(code_chunks) == 1
    assert code_chunks[0].text.count("```") == 2
    assert all(
        "| variable | meaning |" in chunk.text for chunk in chunks if "|---|---|" in chunk.text
    )


def test_long_plain_prose_respects_the_hard_maximum() -> None:
    section = _section(" ".join(f"word{index}" for index in range(130)))
    chunks = chunk_section(section, ChunksConfig(target_tokens=30, min_tokens=10, max_tokens=40))

    assert len(chunks) >= 3
    assert all(chunk.token_count <= 40 for chunk in chunks)
    assert "word0" in chunks[0].text
    assert "word129" in chunks[-1].text


def test_chunk_page_ranges_follow_preserved_markers() -> None:
    section = _section(
        "<!-- page: 7 -->\n" + "first page sentence. " * 8 + "\n\n"
        "<!-- page: 8 -->\n" + "second page sentence. " * 8
    )
    chunks = chunk_section(section, ChunksConfig(target_tokens=16, min_tokens=5, max_tokens=24))

    assert chunks[0].page_start == 7
    assert chunks[-1].page_end == 8
    assert any("<!-- page: 7 -->" in chunk.text for chunk in chunks)
    assert any("<!-- page: 8 -->" in chunk.text for chunk in chunks)


def test_chunk_indices_are_contiguous_per_paper_and_reset_between_papers() -> None:
    sections = [
        _section("one two three four five six", paper_id="a", order=0),
        _section("seven eight nine ten eleven", paper_id="a", order=1),
        _section("alpha beta gamma delta", paper_id="b", order=0),
    ]
    chunks = chunk_sections(
        sections,
        ChunksConfig(target_tokens=4, min_tokens=2, max_tokens=6),
    )

    assert [chunk.chunk_index for chunk in chunks if chunk.paper_id == "a"] == list(
        range(len([chunk for chunk in chunks if chunk.paper_id == "a"]))
    )
    assert [chunk.chunk_index for chunk in chunks if chunk.paper_id == "b"] == [0]
    assert estimate_token_count("<!-- page: 9 --> words and symbols $x_i$") == 6


def test_page_marker_only_sections_do_not_create_empty_retrieval_chunks() -> None:
    sections = [
        _section("<!-- page: 1 -->", order=0),
        _section("<!-- page: 2 -->\nsubstantive geometric evidence", order=1),
    ]

    chunks = chunk_sections(sections)

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].token_count > 0
    assert chunks[0].page_start == 2


def test_pathological_atomic_table_is_hard_bounded_for_model_input() -> None:
    rows = "\n".join(f"| row {index} | value {index} |" for index in range(500))
    section = _section(f"| name | value |\n|---|---|\n{rows}")

    chunks = chunk_section(
        section,
        ChunksConfig(target_tokens=400, min_tokens=100, max_tokens=600),
    )

    assert len(chunks) > 1
    assert all(0 < chunk.token_count <= 600 for chunk in chunks)
    assert "row 0" in chunks[0].text
    assert "row 499" in chunks[-1].text


def test_single_unbroken_prose_run_is_hard_bounded() -> None:
    section = _section("/".join(f"token{index}" for index in range(300)))

    chunks = chunk_section(
        section,
        ChunksConfig(target_tokens=30, min_tokens=10, max_tokens=40),
    )

    assert len(chunks) > 1
    assert all(0 < chunk.token_count <= 40 for chunk in chunks)


def test_multpage_bare_bracket_math_remains_atomic_and_page_aware() -> None:
    section = _section("<!-- page: 1 -->\n[\nx = 1\n<!-- page: 2 -->\n# not a heading\ny = 2\n]\n")
    chunks = chunk_section(
        section,
        ChunksConfig(target_tokens=3, min_tokens=1, max_tokens=5),
    )

    equation = next(chunk for chunk in chunks if "x = 1" in chunk.text)
    assert "y = 2" in equation.text
    assert (equation.page_start, equation.page_end) == (1, 2)
