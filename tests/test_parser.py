from __future__ import annotations

from papers_mcp.parser import parse_sections


def test_heading_tree_pages_and_math_are_preserved() -> None:
    markdown = r"""
<!-- page: 1 -->
# Construction
The construction begins here.

<!-- page: 2 -->
## Vertex Enclosure
We require

$$
a_1 + a_2 = 0.
$$

where the variables are cross-boundary derivatives.

```python
# this is code, not a heading
solve(constraints)
```

<!-- page: 3 -->
## Numerical Method
Use a sparse solve.
"""
    sections = parse_sections(markdown, "peters-1989", "A Paper")

    assert [section.heading for section in sections] == [
        "Construction",
        "Vertex Enclosure",
        "Numerical Method",
    ]
    assert sections[1].heading_path == "Construction > Vertex Enclosure"
    assert sections[1].parent_index == 0
    assert sections[2].parent_index == 0
    assert sections[0].page_start == 1
    assert sections[0].page_end == 3
    assert sections[1].page_start == 2
    assert sections[1].page_end == 2
    assert "a_1 + a_2 = 0" in sections[1].text
    assert "# this is code, not a heading" in sections[1].text
    assert "<!-- page: 2 -->" in sections[1].text


def test_preamble_becomes_a_source_section_and_parent_indices_remain_valid() -> None:
    sections = parse_sections(
        "Front matter.\n\n# Main\nBody.\n\n## Child\nDetails.",
        "paper",
        "Paper title",
    )

    assert [section.heading for section in sections] == ["Paper title", "Main", "Child"]
    assert sections[2].parent_index == 1


def test_no_heading_fallback_is_page_aware() -> None:
    sections = parse_sections(
        "<!-- page: 4 -->\nFirst page text.\n\n<!-- page: 5 -->\nSecond page text.",
        "scan",
    )

    assert [section.heading for section in sections] == ["Page 4", "Page 5"]
    assert [(section.page_start, section.page_end) for section in sections] == [(4, 4), (5, 5)]
    assert "<!-- page: 4 -->" in sections[0].text


def test_hashes_inside_an_unclosed_code_fence_are_not_headings() -> None:
    sections = parse_sections("# Real\n\n```text\n# not-a-section\nmalformed", "paper")

    assert len(sections) == 1
    assert sections[0].heading == "Real"
    assert "# not-a-section" in sections[0].text


def test_empty_input_has_no_sections() -> None:
    assert parse_sections(" \n\x00\n", "empty") == []


def test_outer_heading_emphasis_is_not_part_of_section_identity() -> None:
    sections = parse_sections("# **Paper Title**\n\n## __Method__\nBody.", "paper")

    assert [section.heading for section in sections] == ["Paper Title", "Method"]
    assert sections[1].heading_path == "Paper Title > Method"


def test_numbered_hierarchy_overrides_flat_ocr_markdown_levels() -> None:
    markdown = """# Report

## 1 INTRODUCTION

## 1.1 Motivation

Evidence in the first child.

## 1.2 Prior work

Evidence in the second child.

## 2 METHOD

Method evidence.
"""

    sections = parse_sections(markdown, "paper")

    introduction = next(item for item in sections if item.heading == "1 INTRODUCTION")
    motivation = next(item for item in sections if item.heading == "1.1 Motivation")
    prior = next(item for item in sections if item.heading == "1.2 Prior work")
    method = next(item for item in sections if item.heading == "2 METHOD")
    assert motivation.parent_index == sections.index(introduction)
    assert prior.parent_index == sections.index(introduction)
    assert method.parent_index == 0
    assert motivation.heading_path == "Report > 1 INTRODUCTION > 1.1 Motivation"


def test_page_marker_survives_blank_lines_before_the_next_heading() -> None:
    sections = parse_sections(
        "<!-- page: 1 -->\n\n# First\nFirst evidence.\n"
        "<!-- page: 2 -->\n\n# Second\nSecond evidence.",
        "paper",
    )

    assert [(item.heading, item.page_start, item.page_end) for item in sections] == [
        ("First", 1, 1),
        ("Second", 2, 2),
    ]
    assert "<!-- page: 2 -->" not in sections[0].text
    assert "<!-- page: 2 -->" in sections[1].text


def test_page_markers_inside_math_update_provenance_without_splitting_math() -> None:
    sections = parse_sections(
        "<!-- page: 1 -->\n# Derivation\n$$\nx = 1\n<!-- page: 2 -->\ny = 2\n$$\n",
        "paper",
    )

    assert len(sections) == 1
    assert (sections[0].page_start, sections[0].page_end) == (1, 2)
    assert "<!-- page: 2 -->" in sections[0].text


def test_bare_bracket_display_math_does_not_create_formula_headings() -> None:
    sections = parse_sections(
        "# Derivation\n[\n\\operatorname{tr}(T)\n# \\sum_f \\mu_f\n] \n## Result\nDone.",
        "paper",
    )

    assert [item.heading for item in sections] == ["Derivation", "Result"]
    assert "# \\sum_f" in sections[0].text
