from __future__ import annotations

import hashlib

import pytest

from papers_mcp.ids import (
    metadata_paper_id,
    normalize_paper_id,
    paper_id_from_stem,
    sha256_file,
    stable_paper_id,
    unique_paper_id,
)


def test_sha256_file_streams_the_source(tmp_path) -> None:
    source = tmp_path / "paper.pdf"
    payload = (b"mathematical paper\n" * 1000) + b"end"
    source.write_bytes(payload)

    assert sha256_file(source, block_size=17) == hashlib.sha256(payload).hexdigest()


def test_normalize_and_stem_ids_preserve_catalog_meaning() -> None:
    assert normalize_paper_id("Péters — G¹ Spline Surfaces") == "peters-g1-spline-surfaces"
    assert paper_id_from_stem("VSA_cohen_steiner_2004.pdf") == "vsa-cohen-steiner-2004"


def test_explicit_catalog_id_wins_over_filename_and_metadata() -> None:
    assert (
        stable_paper_id(
            "download.pdf",
            catalog_id="Peters-Vertex-Enclosure-1989",
            title="A different title",
            authors=["Someone Else"],
            year=2020,
            content_hash="a" * 64,
        )
        == "peters-vertex-enclosure-1989"
    )


def test_descriptive_existing_stem_is_stable() -> None:
    assert (
        stable_paper_id(
            "/papers/barsky_derose_G1G2_1989.md",
            title="Geometric Continuity",
            authors=["Brian Barsky"],
            year=1989,
            content_hash="b" * 64,
        )
        == "barsky-derose-g1g2-1989"
    )


def test_generic_stem_uses_metadata_then_hash_fallback() -> None:
    assert stable_paper_id(
        "paper.pdf",
        title="Spline Surfaces Around Extraordinary Vertices",
        authors=["Jörg Peters"],
        year=2002,
        content_hash="c" * 64,
    ).startswith("peters-2002-spline-surfaces")
    assert stable_paper_id("download.pdf", content_hash="d" * 64) == "paper-dddddddddddd"


def test_family_given_author_form_uses_the_family_name() -> None:
    assert metadata_paper_id(title="Spline construction", authors=["Peters, Jörg"], year=2002) == (
        "peters-2002-spline-construction"
    )


def test_metadata_id_rejects_invalid_hash() -> None:
    with pytest.raises(ValueError, match="hexadecimal"):
        metadata_paper_id(title="", content_hash="not-a-hash")


def test_collision_suffix_is_deterministic() -> None:
    assert unique_paper_id("same-paper", "abcdef12" + "0" * 56, {"same-paper"}) == (
        "same-paper-abcdef12"
    )
    assert unique_paper_id("free", "abcdef12" + "0" * 56, set()) == "free"
