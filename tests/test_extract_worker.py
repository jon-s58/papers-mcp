from __future__ import annotations

import json
from types import SimpleNamespace

import papers_mcp.extract_worker as worker


class _Document(list):
    def __init__(self, pages):
        super().__init__(pages)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_dense_vector_probe_closes_document_and_applies_limit(monkeypatch) -> None:
    sparse = SimpleNamespace(get_bboxlog=lambda: [("fill-text",), ("stroke-path",)])
    dense = SimpleNamespace(get_bboxlog=lambda: [("stroke-path",)] * 4)
    document = _Document([sparse, dense])
    monkeypatch.setattr(
        worker.importlib,
        "import_module",
        lambda name: SimpleNamespace(open=lambda source: document),
    )

    assert worker._dense_vector_pdf(worker.Path("paper.pdf"), limit=3)
    assert document.closed


def test_worker_replaces_invalid_lone_surrogates(monkeypatch, tmp_path) -> None:
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        worker,
        "_extract",
        lambda source, *, ocr, legacy: {"kind": "text", "text": "bad-\ud835-text"},
    )

    assert worker.main(["--source", "paper.pdf", "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["text"] == "bad-?-text"


def test_worker_forces_ocr_after_parent_quality_rejection(monkeypatch) -> None:
    captured = {}
    module = SimpleNamespace(
        to_markdown=lambda source, **options: captured.update(options) or ["recovered text"]
    )

    def import_module(name):
        if name == "pymupdf4llm":
            return module
        if name == "pymupdf4llm.ocr.tesseract_api":
            raise ImportError
        raise AssertionError(name)

    monkeypatch.setattr(worker.importlib, "import_module", import_module)

    worker._extract(worker.Path("paper.pdf"), ocr=True, legacy=False)

    assert captured["use_ocr"] is True
    assert captured["force_ocr"] is True


def test_worker_forced_raster_ocr_discards_corrupt_embedded_text(monkeypatch) -> None:
    calls = {}

    class Page:
        def get_textpage_ocr(self, **kwargs):
            calls["ocr"] = kwargs
            return "ocr-textpage"

        def get_text(self, kind, **kwargs):
            calls["text"] = (kind, kwargs)
            return "clean raster OCR text"

    class Document(list):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(
        worker.importlib,
        "import_module",
        lambda name: SimpleNamespace(open=lambda source: Document([Page()])),
    )

    result = worker._extract(
        worker.Path("paper.pdf"),
        ocr=True,
        legacy=False,
        forced_raster_ocr=True,
    )

    assert result == {"kind": "pages", "pages": ["clean raster OCR text"]}
    assert calls["ocr"] == {"language": "eng", "dpi": 150, "full": True}
    assert calls["text"] == ("text", {"textpage": "ocr-textpage"})
