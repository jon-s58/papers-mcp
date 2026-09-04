from __future__ import annotations

import importlib
import json
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from .config import ExtractionConfig, ResourcesConfig
from .memory import GB, MemoryBudgetExceeded, is_memory_exhaustion_error
from .models import ExtractedDocument

LOGGER = logging.getLogger(__name__)
PAGE_MARKER_TEMPLATE = "<!-- page: {page} -->"
PAGE_MARKER_RE = re.compile(r"<!--\s*page\s*:\s*(\d+)\s*-->", re.IGNORECASE)
MARKER_PAGE_RE = re.compile(r"(?m)^\s*\{(-?\d+)\}\s*-{20,}\s*$")
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_TEX_T1_CONTROLS = str.maketrans(
    {
        "\x15": "–",
        "\x16": "—",
        "\x1b": "ff",
        "\x1c": "fi",
        "\x1d": "fl",
        "\x1e": "ffi",
        "\x1f": "ffl",
    }
)
MIN_ALNUM_PER_PAGE = 64
MAX_REPLACEMENT_CHARACTER_RATIO = 0.05
MIN_REPLACEMENT_CHARACTERS_FOR_REJECTION = 32


class ExtractionError(RuntimeError):
    """Raised when a provider cannot extract a usable document."""


class ExtractorUnavailable(ExtractionError):
    """Raised when an optional extraction provider is not installed."""


class ExtractionWorkerTimeout(ExtractionError):
    """Raised when an isolated extractor exceeds its bounded wall-clock budget."""


class CorruptExtractionText(ExtractionError):
    """Raised when broken PDF character maps dominate otherwise dense output."""


class DocumentExtractor(Protocol):
    name: str

    def extract(self, path: str | Path) -> ExtractedDocument: ...


def _clean_markdown(text: str) -> str:
    text = text.translate(_TEX_T1_CONTROLS)
    text = "".join(character for character in text if character in "\n\t\r" or ord(character) >= 32)
    return text.replace("\r\n", "\n").replace("\r", "\n").strip() + "\n"


def _require_text_density(markdown: str, page_count: int | None, provider: str) -> None:
    without_markers = PAGE_MARKER_RE.sub("", markdown)
    replacements = without_markers.count("\ufffd")
    replacement_ratio = replacements / max(1, len(without_markers))
    if (
        replacements >= MIN_REPLACEMENT_CHARACTERS_FOR_REJECTION
        and replacement_ratio > MAX_REPLACEMENT_CHARACTER_RATIO
    ):
        raise CorruptExtractionText(
            f"{provider} produced corrupt text ({replacements} Unicode replacement "
            f"characters; {replacement_ratio:.1%} of output); OCR may be required"
        )
    visible = sum(character.isalnum() for character in without_markers)
    minimum = max(32, MIN_ALNUM_PER_PAGE * max(1, page_count or 1))
    if visible < minimum:
        raise ExtractionError(
            f"{provider} produced too little usable text ({visible} characters across "
            f"{page_count or 'an unknown number of'} pages); OCR may be required"
        )


def markdown_from_pages(pages: Sequence[str], *, include_markers: bool = True) -> str:
    """Join ordered page text while retaining explicit one-based page provenance."""

    rendered: list[str] = []
    for page_number, page_text in enumerate(pages, start=1):
        if include_markers:
            rendered.append(PAGE_MARKER_TEMPLATE.format(page=page_number))
        rendered.append(str(page_text).strip())
    return _clean_markdown("\n\n".join(rendered))


def _forced_ocr_page_markdown(text: str, page_number: int) -> str:
    """Render plain OCR as one explicit page section without accidental Markdown state."""

    safe_lines: list[str] = []
    for line in str(text).splitlines():
        stripped = line.strip()
        if not stripped:
            safe_lines.append("")
            continue
        # Tesseract output is plain text, but source glyphs can resemble Markdown
        # controls.  Neutralize only constructs that would make the parser or
        # chunker protect many subsequent pages as one giant math/code block.
        stripped = stripped.replace("$$", "$ $")
        stripped = re.sub(r"^\\\[", "[", stripped)
        stripped = re.sub(r"^\\begin\{", "begin{", stripped)
        if stripped in {"[", "]"}:
            stripped = f"OCR symbol: {stripped}"
        if re.match(r"^(?:#{1,6}\s|`{3,}|~{3,}|<!--\s*page\s*:)", stripped, re.I):
            stripped = f"OCR text: {stripped}"
        if re.match(
            r"^(?:theorem|lemma|proposition|corollary|definition|proof)\b",
            stripped,
            re.I,
        ):
            stripped = f"OCR text: {stripped}"
        safe_lines.append(stripped)
    body = "\n".join(safe_lines).strip()
    return f"# OCR page {page_number}\n\n{body}".strip()


def _normalize_marker_pagination(markdown: str) -> str:
    matches = list(MARKER_PAGE_RE.finditer(markdown))
    if not matches:
        return markdown
    page_values = [int(match.group(1)) for match in matches]
    offset = 1 if min(page_values) == 0 else 0
    return MARKER_PAGE_RE.sub(
        lambda match: PAGE_MARKER_TEMPLATE.format(page=int(match.group(1)) + offset),
        markdown,
    )


def _split_authors(value: str | None) -> list[str]:
    if not value:
        return []
    parts = re.split(r"\s*(?:;|\band\b)\s*", value, flags=re.IGNORECASE)
    if len(parts) == 1 and "," in value:
        comma_parts = [part.strip() for part in value.split(",") if part.strip()]
        # Preserve conventional "Family, Given" names, but split a comma-separated
        # author list when both sides look like complete names.
        if len(comma_parts) > 1 and all(len(part.split()) >= 2 for part in comma_parts):
            parts = comma_parts
    return [part.strip() for part in parts if part.strip()]


def _strip_outer_markdown(value: str) -> str:
    value = value.strip()
    changed = True
    while changed:
        changed = False
        for marker in ("**", "__", "`", "*", "_"):
            if value.startswith(marker) and value.endswith(marker) and len(value) > 2 * len(marker):
                value = value[len(marker) : -len(marker)].strip()
                changed = True
                break
    return value


def _metadata_from_text(text: str) -> dict[str, Any]:
    title = ""
    heading = re.search(r"(?m)^#\s+(.+?)\s*$", text)
    if heading:
        title = _strip_outer_markdown(heading.group(1))
    author_match = re.search(r"(?im)^\*\*Authors?:?\*\*\s*:?\s*(.+?)\s*$", text)
    doi_match = DOI_RE.search(text[:12000])
    year_match = YEAR_RE.search(text[:8000])
    return {
        "title": title,
        "authors": _split_authors(author_match.group(1) if author_match else None),
        "year": int(year_match.group(0)) if year_match else None,
        "doi": doi_match.group(0).rstrip(".,;)") if doi_match else None,
    }


def _pdf_metadata(path: Path) -> dict[str, Any]:
    """Read cheap PDF metadata when PyMuPDF is available; never block extraction."""

    try:
        pymupdf = importlib.import_module("pymupdf")
    except ImportError:
        return {}
    try:
        document = pymupdf.open(path)
        try:
            raw = document.metadata or {}
            page_count = len(document)
        finally:
            document.close()
    except MemoryError:
        raise
    except Exception:  # metadata is best effort and must not reject a readable extractor result
        return {}

    date_value = str(raw.get("creationDate") or raw.get("modDate") or "")
    year_match = YEAR_RE.search(date_value)
    return {
        "title": str(raw.get("title") or "").strip(),
        "authors": _split_authors(str(raw.get("author") or "")),
        "year": int(year_match.group(0)) if year_match else None,
        "page_count": page_count,
    }


def extract_markdown_reference(path: str | Path) -> ExtractedDocument:
    """Load a curated Markdown-only paper without treating it as generated extraction."""

    source = Path(path)
    markdown = _clean_markdown(source.read_text(encoding="utf-8", errors="replace"))
    metadata = _metadata_from_text(markdown)
    return ExtractedDocument(markdown=markdown, backend="markdown", **metadata)


class MarkerExtractor:
    name = "marker"

    def __init__(
        self,
        command: str = "marker_single",
        *,
        page_markers: bool = True,
        ocr: bool = True,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        which: Callable[[str], str | None] = shutil.which,
    ) -> None:
        self.command = command
        self.page_markers = page_markers
        self.ocr = ocr
        self._runner = runner
        self._which = which

    def extract(self, path: str | Path) -> ExtractedDocument:
        source = Path(path)
        command = shlex.split(self.command)
        if not command or self._which(command[0]) is None:
            raise ExtractorUnavailable(f"Marker command is unavailable: {self.command}")

        with tempfile.TemporaryDirectory(prefix="papers-marker-") as output_dir:
            invocation = [
                *command,
                str(source),
                "--output_dir",
                output_dir,
                "--output_format",
                "markdown",
            ]
            if self.page_markers:
                invocation.append("--paginate_output")
            if not self.ocr:
                invocation.append("--disable_ocr")
            completed = self._runner(
                invocation,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "unknown Marker error").strip()
                raise ExtractionError(
                    f"Marker exited with {completed.returncode}: {detail[-1000:]}"
                )

            outputs = list(Path(output_dir).rglob("*.md"))
            if not outputs:
                raise ExtractionError("Marker completed without producing Markdown")
            exact = [candidate for candidate in outputs if candidate.stem == source.stem]
            selected = max(exact or outputs, key=lambda candidate: candidate.stat().st_size)
            markdown = selected.read_text(encoding="utf-8", errors="replace")
            if self.page_markers:
                markdown = _normalize_marker_pagination(markdown)
            markdown = _clean_markdown(markdown)

        if not markdown.strip():
            raise ExtractionError("Marker produced empty Markdown")
        metadata = _pdf_metadata(source)
        _require_text_density(markdown, metadata.get("page_count"), "Marker")
        text_metadata = _metadata_from_text(markdown)
        warnings: list[str] = []
        if self.page_markers and PAGE_MARKER_RE.search(markdown) is None:
            warnings.append(
                "Marker output did not expose trustworthy page boundaries; page ranges are unknown."
            )
        return ExtractedDocument(
            markdown=markdown,
            backend=self.name,
            title=metadata.get("title") or text_metadata["title"],
            authors=metadata.get("authors") or text_metadata["authors"],
            year=metadata.get("year") or text_metadata["year"],
            doi=text_metadata["doi"],
            page_count=metadata.get("page_count"),
            warnings=warnings,
        )


class PyMuPDF4LLMExtractor:
    name = "pymupdf4llm"

    def __init__(
        self,
        *,
        page_markers: bool = True,
        ocr: bool = True,
        resources: ResourcesConfig | None = None,
        _unsafe_in_process_for_tests: bool = False,
    ) -> None:
        self.page_markers = page_markers
        self.ocr = ocr
        self.resources = resources or ResourcesConfig()
        self._unsafe_in_process_for_tests = _unsafe_in_process_for_tests

    def _extract_isolated(
        self,
        source: Path,
        *,
        ocr: bool,
        legacy: bool = False,
        forced_raster_ocr: bool = False,
    ) -> str | list[str]:
        try:
            psutil = importlib.import_module("psutil")
        except ImportError as error:
            raise ExtractorUnavailable(
                "psutil is required to supervise isolated PyMuPDF4LLM extraction"
            ) from error

        limit = int(self.resources.max_process_memory_gb * GB)
        timeout = self.resources.extraction_worker_timeout_seconds
        with tempfile.TemporaryDirectory(prefix="papers-pymupdf4llm-") as temp_dir:
            temp_path = Path(temp_dir)
            output_path = temp_path / "result.json"
            log_path = temp_path / "worker.log"
            command = [
                sys.executable,
                "-m",
                "papers_mcp.extract_worker",
                "--source",
                str(source),
                "--output",
                str(output_path),
            ]
            if ocr:
                command.append("--ocr")
            if legacy:
                command.append("--legacy")
            if forced_raster_ocr:
                command.append("--forced-raster-ocr")
            with log_path.open("w+", encoding="utf-8") as worker_log:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=worker_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                parent = psutil.Process(os.getpid())
                worker = psutil.Process(process.pid)
                peak = 0
                started = time.monotonic()
                try:
                    while process.poll() is None:
                        processes = [parent, worker]
                        try:
                            processes.extend(worker.children(recursive=True))
                        except psutil.Error:
                            pass
                        rss = 0
                        seen: set[int] = set()
                        for candidate in processes:
                            if candidate.pid in seen:
                                continue
                            seen.add(candidate.pid)
                            try:
                                rss += int(candidate.memory_info().rss)
                            except psutil.Error:
                                continue
                        peak = max(peak, rss)
                        if rss >= limit:
                            self._stop_worker(process, worker, psutil)
                            raise MemoryBudgetExceeded(
                                "extraction worker stopped within the configured memory cap: "
                                f"tree_rss={rss / GB:.2f} GB "
                                f"limit={self.resources.max_process_memory_gb:.2f} GB"
                            )
                        if time.monotonic() - started >= timeout:
                            self._stop_worker(process, worker, psutil)
                            raise ExtractionWorkerTimeout(
                                "isolated PyMuPDF4LLM worker timed out after "
                                f"{timeout} seconds (ocr={ocr}, legacy={legacy}, "
                                f"forced_raster_ocr={forced_raster_ocr})"
                            )
                        time.sleep(0.05)
                except BaseException:
                    if process.poll() is None:
                        self._stop_worker(process, worker, psutil)
                    raise
                returncode = process.wait()
                worker_log.flush()
                worker_log.seek(0)
                detail = worker_log.read()[-4000:].strip()

            if returncode != 0:
                if (
                    returncode < 0
                    or "memoryerror" in detail.casefold()
                    or is_memory_exhaustion_error(RuntimeError(detail))
                ):
                    raise MemoryBudgetExceeded(
                        "isolated PyMuPDF4LLM worker failed closed after a possible memory "
                        f"exhaustion (exit={returncode}): {detail or 'no diagnostic output'}"
                    )
                raise ExtractionError(
                    f"isolated PyMuPDF4LLM worker exited with {returncode}: "
                    f"{detail or 'no diagnostic output'}"
                )
            if not output_path.is_file():
                raise ExtractionError("isolated PyMuPDF4LLM worker produced no result")
            try:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ExtractionError(
                    f"invalid isolated PyMuPDF4LLM worker result: {error}"
                ) from error
            LOGGER.debug(
                "Isolated PyMuPDF4LLM extraction peak tree RSS %.2f GB for %s",
                peak / GB,
                source,
            )
            if payload.get("kind") == "text" and isinstance(payload.get("text"), str):
                return payload["text"]
            if payload.get("kind") == "pages" and isinstance(payload.get("pages"), list):
                return [str(page) for page in payload["pages"]]
            raise ExtractionError("isolated PyMuPDF4LLM worker returned an unsupported payload")

    @staticmethod
    def _stop_worker(process: subprocess.Popen[str], worker: Any, psutil: Any) -> None:
        try:
            descendants = worker.children(recursive=True)
        except psutil.Error:
            descendants = []
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        # Reap or kill any descendants that escaped the process group. This is
        # defensive for platform-specific OCR helpers that create their own session.
        _, alive = psutil.wait_procs(descendants, timeout=1)
        for child in alive:
            try:
                child.kill()
            except psutil.Error:
                pass
        psutil.wait_procs(alive, timeout=1)

    def extract(self, path: str | Path) -> ExtractedDocument:
        source = Path(path)
        used_forced_raster_ocr = False
        try:
            if self._unsafe_in_process_for_tests:
                try:
                    pymupdf4llm = importlib.import_module("pymupdf4llm")
                except ImportError as error:
                    raise ExtractorUnavailable("PyMuPDF4LLM is not installed") from error
                options: dict[str, Any] = {
                    "page_chunks": True,
                    "use_ocr": self.ocr,
                }
                if self.ocr:
                    try:
                        tesseract_api = importlib.import_module("pymupdf4llm.ocr.tesseract_api")
                    except ImportError:
                        pass
                    else:
                        ocr_function = getattr(tesseract_api, "exec_ocr", None)
                        if callable(ocr_function):
                            # Prefer the stable built-in Tesseract bridge explicitly;
                            # auto-detection may import an unrelated broken RapidOCR.
                            options["ocr_function"] = ocr_function
                extracted = pymupdf4llm.to_markdown(str(source), **options)
            else:
                try:
                    extracted = self._extract_isolated(source, ocr=False)
                except ExtractionWorkerTimeout:
                    LOGGER.warning(
                        "PyMuPDF4LLM layout timed out; retrying the isolated legacy parser: %s",
                        source,
                    )
                    extracted = self._extract_isolated(source, ocr=False, legacy=True)
                page_text = (
                    extracted
                    if isinstance(extracted, str)
                    else "\n".join(str(page) for page in extracted)
                )
                page_count = 1 if isinstance(extracted, str) else len(extracted)
                try:
                    _require_text_density(page_text, page_count, "PyMuPDF4LLM without OCR")
                except CorruptExtractionText:
                    if not self.ocr:
                        raise
                    LOGGER.info(
                        "Retrying corrupt PDF character maps with isolated forced raster OCR: %s",
                        source,
                    )
                    extracted = self._extract_isolated(
                        source,
                        ocr=True,
                        forced_raster_ocr=True,
                    )
                    used_forced_raster_ocr = True
                except ExtractionError:
                    if not self.ocr:
                        raise
                    LOGGER.info("Retrying low-density PDF with isolated OCR: %s", source)
                    extracted = self._extract_isolated(source, ocr=True)
        except MemoryError:
            raise
        except Exception as error:
            raise ExtractionError(f"PyMuPDF4LLM failed: {error}") from error

        warnings: list[str] = []
        if used_forced_raster_ocr:
            warnings.append(
                "Embedded PDF character maps were corrupt; used full-page PyMuPDF/Tesseract "
                "OCR at 150 dpi. Markdown headings and equations may be degraded."
            )
        if isinstance(extracted, str):
            markdown = _clean_markdown(extracted)
            if self.page_markers and PAGE_MARKER_RE.search(markdown) is None:
                warnings.append("PyMuPDF4LLM returned unpaged Markdown; page ranges are unknown.")
        elif isinstance(extracted, Sequence):
            page_texts: list[str] = []
            for page_number, chunk in enumerate(extracted, start=1):
                if isinstance(chunk, str):
                    page_text = chunk
                elif isinstance(chunk, Mapping):
                    page_text = str(chunk.get("text") or chunk.get("markdown") or "")
                else:
                    page_text = str(chunk)
                if used_forced_raster_ocr:
                    page_text = _forced_ocr_page_markdown(page_text, page_number)
                page_texts.append(page_text)
            markdown = markdown_from_pages(page_texts, include_markers=self.page_markers)
        else:
            raise ExtractionError(
                f"PyMuPDF4LLM returned unsupported output type: {type(extracted).__name__}"
            )
        metadata = _pdf_metadata(source)
        page_count = metadata.get("page_count") or (
            len(extracted)
            if isinstance(extracted, Sequence) and not isinstance(extracted, str)
            else None
        )
        _require_text_density(markdown, page_count, "PyMuPDF4LLM")
        text_metadata = _metadata_from_text(markdown)
        return ExtractedDocument(
            markdown=markdown,
            backend=("pymupdf-tesseract-ocr" if used_forced_raster_ocr else self.name),
            title=metadata.get("title")
            or ("" if used_forced_raster_ocr else text_metadata["title"]),
            authors=metadata.get("authors") or text_metadata["authors"],
            year=metadata.get("year") or text_metadata["year"],
            doi=text_metadata["doi"],
            page_count=page_count,
            warnings=warnings,
        )


class PyMuPDFExtractor:
    name = "pymupdf"

    def __init__(self, *, page_markers: bool = True) -> None:
        self.page_markers = page_markers

    def extract(self, path: str | Path) -> ExtractedDocument:
        source = Path(path)
        try:
            pymupdf = importlib.import_module("pymupdf")
        except ImportError as error:
            raise ExtractorUnavailable("PyMuPDF is not installed") from error
        try:
            document = pymupdf.open(source)
            try:
                raw_metadata = document.metadata or {}
                page_texts: list[str] = []
                for page in document:
                    try:
                        page_texts.append(page.get_text("text", sort=True))
                    except TypeError:  # older compatible PyMuPDF releases
                        page_texts.append(page.get_text("text"))
            finally:
                document.close()
        except MemoryError:
            raise
        except Exception as error:
            raise ExtractionError(f"PyMuPDF failed: {error}") from error

        markdown = markdown_from_pages(page_texts, include_markers=self.page_markers)
        _require_text_density(markdown, len(page_texts), "PyMuPDF")
        text_metadata = _metadata_from_text(markdown)
        date_value = str(raw_metadata.get("creationDate") or raw_metadata.get("modDate") or "")
        year_match = YEAR_RE.search(date_value)
        warnings = [
            "Used plain PyMuPDF fallback; equations, reading order, and tables may be degraded."
        ]
        if any(not text.strip() for text in page_texts):
            warnings.append(
                "One or more PDF pages contained no extractable text; OCR may be required."
            )
        return ExtractedDocument(
            markdown=markdown,
            backend=self.name,
            title=str(raw_metadata.get("title") or text_metadata["title"] or "").strip(),
            authors=_split_authors(str(raw_metadata.get("author") or ""))
            or text_metadata["authors"],
            year=int(year_match.group(0)) if year_match else text_metadata["year"],
            doi=text_metadata["doi"],
            page_count=len(page_texts),
            warnings=warnings,
        )


class ExtractorChain:
    """Try configured providers in order and make every fallback visible."""

    def __init__(self, extractors: Sequence[DocumentExtractor]) -> None:
        if not extractors:
            raise ValueError("at least one extractor is required")
        self.extractors = tuple(extractors)

    @classmethod
    def from_config(
        cls,
        config: ExtractionConfig,
        resources: ResourcesConfig | None = None,
    ) -> ExtractorChain:
        providers: list[DocumentExtractor] = []
        for provider in config.providers:
            normalized = provider.strip().lower()
            if normalized == "marker":
                providers.append(
                    MarkerExtractor(
                        config.marker_command,
                        page_markers=config.page_markers,
                        ocr=config.ocr,
                    )
                )
            elif normalized == "pymupdf4llm":
                providers.append(
                    PyMuPDF4LLMExtractor(
                        page_markers=config.page_markers,
                        ocr=config.ocr,
                        resources=resources,
                    )
                )
            elif normalized == "pymupdf":
                providers.append(PyMuPDFExtractor(page_markers=config.page_markers))
            else:
                providers.append(_UnavailableExtractor(provider))
        return cls(providers)

    def extract(self, path: str | Path) -> ExtractedDocument:
        failures: list[str] = []
        for extractor in self.extractors:
            try:
                document = extractor.extract(path)
            except MemoryError:
                raise
            except Exception as error:
                message = f"{extractor.name}: {error}"
                failures.append(message)
                LOGGER.warning("PDF extractor fallback for %s: %s", path, message)
                continue
            if failures:
                document.warnings[:0] = [f"Extraction fallback: {failure}" for failure in failures]
            return document
        detail = "; ".join(failures) or "no providers configured"
        raise ExtractionError(f"All PDF extractors failed for {Path(path)}: {detail}")


class _UnavailableExtractor:
    def __init__(self, name: str) -> None:
        self.name = name

    def extract(self, path: str | Path) -> ExtractedDocument:
        raise ExtractorUnavailable(f"unknown extraction provider: {self.name}")


def extract_pdf(
    path: str | Path,
    config: ExtractionConfig,
    resources: ResourcesConfig | None = None,
) -> ExtractedDocument:
    return ExtractorChain.from_config(config, resources).extract(path)
