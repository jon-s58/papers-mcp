from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DENSE_VECTOR_PATH_LIMIT = 10_000


def _dense_vector_pdf(source: Path, *, limit: int = DENSE_VECTOR_PATH_LIMIT) -> bool:
    """Cheaply identify pages whose vector paths make table/layout analysis pathological."""

    pymupdf = importlib.import_module("pymupdf")
    document = pymupdf.open(source)
    try:
        for page in document:
            path_count = sum("path" in str(item[0]) for item in page.get_bboxlog())
            if path_count > limit:
                return True
    finally:
        document.close()
    return False


def _extract(
    source: Path,
    *,
    ocr: bool,
    legacy: bool,
    forced_raster_ocr: bool = False,
) -> dict[str, Any]:
    """Run PyMuPDF4LLM in an expendable process and return only required text."""

    if forced_raster_ocr:
        pymupdf = importlib.import_module("pymupdf")
        pages: list[str] = []
        with pymupdf.open(source) as document:
            for page in document:
                textpage = page.get_textpage_ocr(
                    language="eng",
                    dpi=150,
                    full=True,
                )
                pages.append(page.get_text("text", textpage=textpage))
        return {"kind": "pages", "pages": pages}

    if not legacy and not ocr:
        legacy = _dense_vector_pdf(source)
    pymupdf4llm = importlib.import_module("pymupdf4llm")
    if legacy:
        pymupdf4llm.use_layout(False)
    options: dict[str, Any] = {"page_chunks": True}
    if legacy:
        # This path is reached only after the neural layout worker times out.
        # Skip pathological vector/table analysis while retaining legacy heading
        # detection, reading order, page chunks, and all extractable text.
        options.update(ignore_graphics=True, table_strategy=None)
    if not legacy:
        options["use_ocr"] = ocr
    if ocr and not legacy:
        # OCR is only requested after the text-first extraction fails a quality
        # gate.  Force it even when a corrupt embedded font map made the page
        # appear to contain plenty of (unusable) text.
        options["force_ocr"] = True
        try:
            tesseract_api = importlib.import_module("pymupdf4llm.ocr.tesseract_api")
        except ImportError:
            pass
        else:
            ocr_function = getattr(tesseract_api, "exec_ocr", None)
            if callable(ocr_function):
                options["ocr_function"] = ocr_function

    extracted = pymupdf4llm.to_markdown(str(source), **options)
    if isinstance(extracted, str):
        return {"kind": "text", "text": extracted}
    if isinstance(extracted, Sequence):
        pages: list[str] = []
        for chunk in extracted:
            if isinstance(chunk, str):
                pages.append(chunk)
            elif isinstance(chunk, Mapping):
                pages.append(str(chunk.get("text") or chunk.get("markdown") or ""))
            else:
                pages.append(str(chunk))
        return {"kind": "pages", "pages": pages}
    raise TypeError(f"unsupported PyMuPDF4LLM output type: {type(extracted).__name__}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--forced-raster-ocr", action="store_true")
    args = parser.parse_args(argv)
    kwargs = {"ocr": args.ocr, "legacy": args.legacy}
    if args.forced_raster_ocr:
        kwargs["forced_raster_ocr"] = True
    payload = _extract(args.source, **kwargs)
    # Some old PDF font maps yield lone UTF-16 surrogate code points. They are
    # not valid UTF-8 and must not make an otherwise successful extraction fail.
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
        errors="replace",
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the parent process
    raise SystemExit(main())
