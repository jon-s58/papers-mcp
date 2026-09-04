from __future__ import annotations

import types

import pytest

from papers_mcp.chunker import chunk_sections
from papers_mcp.config import ChunksConfig, ExtractionConfig, ResourcesConfig
from papers_mcp.extract import (
    ExtractionError,
    ExtractorChain,
    ExtractorUnavailable,
    MarkerExtractor,
    PyMuPDF4LLMExtractor,
    extract_markdown_reference,
    markdown_from_pages,
)
from papers_mcp.memory import MemoryBudgetExceeded
from papers_mcp.models import ExtractedDocument
from papers_mcp.parser import parse_sections


class _FailingExtractor:
    name = "broken"

    def extract(self, path):
        raise ExtractionError("cannot parse this file")


class _WorkingExtractor:
    name = "working"

    def extract(self, path):
        return ExtractedDocument(markdown="# Result\n\n$E=mc^2$\n", backend=self.name)


class _MemoryFailingExtractor:
    name = "memory"

    def extract(self, path):
        raise MemoryBudgetExceeded("bounded worker stopped")


class _RawMemoryFailingExtractor:
    name = "raw-memory"

    def extract(self, path):
        raise MemoryError("native allocation failed")


def test_markdown_from_pages_preserves_math_and_page_markers() -> None:
    markdown = markdown_from_pages(
        ["We minimize\n\n$$E(x)=x^2$$", "an e\x1ecient fit over pages 1\x152"]
    )

    assert "<!-- page: 1 -->" in markdown
    assert "<!-- page: 2 -->" in markdown
    assert "$$E(x)=x^2$$" in markdown
    assert "an efficient fit over pages 1–2" in markdown
    assert "\x1e" not in markdown


def test_chain_reports_every_fallback_explicitly() -> None:
    result = ExtractorChain([_FailingExtractor(), _WorkingExtractor()]).extract("paper.pdf")

    assert result.backend == "working"
    assert result.warnings == [
        "Extraction fallback: broken: cannot parse this file",
    ]


def test_chain_raises_a_summary_when_all_providers_fail() -> None:
    with pytest.raises(ExtractionError, match="All PDF extractors failed"):
        ExtractorChain([_FailingExtractor()]).extract("paper.pdf")


def test_chain_never_turns_a_memory_stop_into_an_extractor_fallback() -> None:
    with pytest.raises(MemoryBudgetExceeded, match="bounded worker stopped"):
        ExtractorChain([_MemoryFailingExtractor(), _WorkingExtractor()]).extract("paper.pdf")

    with pytest.raises(MemoryError, match="native allocation failed"):
        ExtractorChain([_RawMemoryFailingExtractor(), _WorkingExtractor()]).extract("paper.pdf")


def test_marker_is_optional_and_does_not_invoke_a_missing_command() -> None:
    extractor = MarkerExtractor(which=lambda command: None)
    with pytest.raises(ExtractorUnavailable, match="unavailable"):
        extractor.extract("paper.pdf")


def test_marker_requests_and_normalizes_paginated_output(tmp_path, monkeypatch) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"unused by fake runner")
    captured = []

    def fake_runner(invocation, **kwargs):
        captured.extend(invocation)
        output_dir = invocation[invocation.index("--output_dir") + 1]
        output = tmp_path.__class__(output_dir) / "paper" / "paper.md"
        output.parent.mkdir()
        output.write_text(
            "{0}"
            + "-" * 48
            + "\n\nThe first page derives a constrained surface objective $x$.\n\n{1}"
            + "-" * 48
            + "\n\nThe second page proves tangent compatibility for $y$."
        )
        return types.SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr("papers_mcp.extract._pdf_metadata", lambda path: {})
    result = MarkerExtractor(runner=fake_runner, which=lambda command: command).extract(source)

    assert "--paginate_output" in captured
    assert "<!-- page: 1 -->" in result.markdown
    assert "<!-- page: 2 -->" in result.markdown
    assert not result.warnings


def test_marker_can_explicitly_disable_ocr(tmp_path, monkeypatch) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"unused")
    captured = []

    def fake_runner(invocation, **kwargs):
        captured.extend(invocation)
        output_dir = invocation[invocation.index("--output_dir") + 1]
        output = tmp_path.__class__(output_dir) / "paper.md"
        output.write_text(
            "# Paper\n\nA sufficiently detailed geometric derivation for the OCR option test.\n"
        )
        return types.SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr("papers_mcp.extract._pdf_metadata", lambda path: {})
    MarkerExtractor(
        ocr=False,
        page_markers=False,
        runner=fake_runner,
        which=lambda command: command,
    ).extract(source)

    assert "--disable_ocr" in captured


def test_marker_rejects_nonempty_output_without_usable_text(tmp_path, monkeypatch) -> None:
    source = tmp_path / "scan.pdf"
    source.write_bytes(b"unused")

    def fake_runner(invocation, **kwargs):
        output_dir = invocation[invocation.index("--output_dir") + 1]
        output = tmp_path.__class__(output_dir) / "scan.md"
        output.write_text("<!-- page: 1 -->\n\n$x$\n")
        return types.SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(
        "papers_mcp.extract._pdf_metadata",
        lambda path: {"page_count": 12},
    )
    with pytest.raises(ExtractionError, match="too little usable text"):
        MarkerExtractor(runner=fake_runner, which=lambda command: command).extract(source)


def test_pymupdf4llm_page_chunks_are_rendered_without_a_real_pdf(monkeypatch) -> None:
    captured = {}

    def fake_ocr(*args, **kwargs):
        return None

    def to_markdown(path, **kwargs):
        captured.update(kwargs)
        return [
            {
                "text": "First page derives the objective $x^2$ and explains every "
                "optimization variable in the geometric construction."
            },
            {
                "text": "Second page constrains the variable $y$ and gives the boundary "
                "conditions needed for a reproducible solution."
            },
        ]

    fake_module = types.SimpleNamespace(
        to_markdown=to_markdown,
    )

    def fake_import(name):
        if name == "pymupdf4llm":
            return fake_module
        if name == "pymupdf4llm.ocr.tesseract_api":
            return types.SimpleNamespace(exec_ocr=fake_ocr)
        if name == "pymupdf":
            raise ImportError
        raise AssertionError(name)

    monkeypatch.setattr("papers_mcp.extract.importlib.import_module", fake_import)
    result = PyMuPDF4LLMExtractor(_unsafe_in_process_for_tests=True).extract("unused.pdf")

    assert result.backend == "pymupdf4llm"
    assert result.page_count == 2
    assert "<!-- page: 1 -->" in result.markdown
    assert "Second page constrains the variable $y$" in result.markdown
    assert captured == {
        "page_chunks": True,
        "use_ocr": True,
        "ocr_function": fake_ocr,
    }


def test_pymupdf4llm_rejects_page_markers_without_usable_ocr_text(monkeypatch) -> None:
    fake_module = types.SimpleNamespace(
        to_markdown=lambda path, **kwargs: [{"text": "x"} for _ in range(28)]
    )

    def fake_import(name):
        if name == "pymupdf4llm":
            return fake_module
        if name == "pymupdf4llm.ocr.tesseract_api":
            raise ImportError
        if name == "pymupdf":
            raise ImportError
        raise AssertionError(name)

    monkeypatch.setattr("papers_mcp.extract.importlib.import_module", fake_import)
    with pytest.raises(ExtractionError, match="too little usable text"):
        PyMuPDF4LLMExtractor(_unsafe_in_process_for_tests=True).extract("scan.pdf")


def test_pymupdf4llm_retries_replacement_character_corruption_with_ocr(
    monkeypatch,
) -> None:
    extractor = PyMuPDF4LLMExtractor(resources=ResourcesConfig())
    calls: list[tuple[bool, bool]] = []

    def extract_isolated(source, *, ocr, forced_raster_ocr=False):
        calls.append((ocr, forced_raster_ocr))
        if not ocr:
            return ["\ufffd" * 200 + " readable words " * 8]
        return [
            "[\n" + "recovered geometric evidence " * 120,
            "$$\n" + "reliable numerical procedure " * 120,
        ]

    monkeypatch.setattr(extractor, "_extract_isolated", extract_isolated)
    monkeypatch.setattr("papers_mcp.extract._pdf_metadata", lambda path: {})

    result = extractor.extract("corrupt-font-map.pdf")

    assert calls == [(False, False), (True, True)]
    assert "recovered geometric evidence" in result.markdown
    assert result.backend == "pymupdf-tesseract-ocr"
    assert result.markdown.count("# OCR page ") == 2
    sections = parse_sections(result.markdown, "ocr-paper", "Recovered paper")
    chunks = chunk_sections(
        sections,
        ChunksConfig(target_tokens=30, min_tokens=10, max_tokens=40),
    )
    assert len(sections) == 2
    assert chunks
    assert all(0 < chunk.token_count <= 40 for chunk in chunks)


def test_pymupdf4llm_uses_isolated_worker_when_resources_are_configured(
    monkeypatch,
) -> None:
    extractor = PyMuPDF4LLMExtractor(resources=ResourcesConfig())
    monkeypatch.setattr(
        extractor,
        "_extract_isolated",
        lambda source, *, ocr: [
            "First isolated page derives a geometric objective and defines every variable "
            "used by the constrained optimization procedure.",
            "Second isolated page verifies boundary compatibility and documents the "
            "numerical conditions needed for a stable solution.",
        ],
    )
    monkeypatch.setattr("papers_mcp.extract._pdf_metadata", lambda path: {})

    result = extractor.extract("unused.pdf")

    assert result.page_count == 2
    assert "<!-- page: 2 -->" in result.markdown


def test_extractor_chain_propagates_resources_to_pymupdf4llm() -> None:
    resources = ResourcesConfig()
    chain = ExtractorChain.from_config(
        ExtractionConfig(providers=("pymupdf4llm",)),
        resources,
    )

    assert isinstance(chain.extractors[0], PyMuPDF4LLMExtractor)
    assert chain.extractors[0].resources is resources


def test_markdown_reference_keeps_source_and_extracts_basic_metadata(tmp_path) -> None:
    source = tmp_path / "summary.md"
    source.write_text(
        "# Vertex Enclosure\n\n**Authors:** Jörg Peters and Jane Doe\n\n"
        "Published in 1989. DOI: 10.1000/example.\n\n$$a+b=0$$\n",
        encoding="utf-8",
    )

    result = extract_markdown_reference(source)

    assert result.backend == "markdown"
    assert result.title == "Vertex Enclosure"
    assert result.authors == ["Jörg Peters", "Jane Doe"]
    assert result.year == 1989
    assert result.doi == "10.1000/example"
    assert "$$a+b=0$$" in result.markdown


def test_markdown_reference_splits_a_comma_separated_author_list(tmp_path) -> None:
    source = tmp_path / "summary.md"
    source.write_text(
        "# Continuity\n\n**Authors:** Brian A. Barsky, Tony D. DeRose\n",
        encoding="utf-8",
    )

    assert extract_markdown_reference(source).authors == ["Brian A. Barsky", "Tony D. DeRose"]


def test_markdown_reference_strips_outer_emphasis_from_title(tmp_path) -> None:
    source = tmp_path / "summary.md"
    source.write_text("# **Energy-based Geometric Fitting**\n", encoding="utf-8")

    assert extract_markdown_reference(source).title == "Energy-based Geometric Fitting"
