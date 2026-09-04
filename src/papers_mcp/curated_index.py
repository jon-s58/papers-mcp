from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .models import CuratedEntry

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")
ARTIFACT_RE = re.compile(
    r"(?<![\w.-])((?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9][A-Za-z0-9_.-]*\.(?:pdf|md))"
    r"(?![\w.-])",
    re.I,
)
LINK_RE = re.compile(r"\[([^]]+)]\([^)]+\)")
MARKUP_RE = re.compile(r"[`*~]")


@dataclass(slots=True)
class CatalogRecord:
    record_id: str
    heading_path: str
    fields: dict[str, str]
    artifacts: list[str] = field(default_factory=list)
    source_line: int | None = None

    @property
    def searchable_text(self) -> str:
        parts = [self.record_id]
        for key, value in self.fields.items():
            if value:
                parts.append(f"{key}: {value}")
        if self.artifacts:
            parts.append(f"artifacts: {'; '.join(self.artifacts)}")
        return "\n".join(parts)


@dataclass(slots=True)
class CuratedIndex:
    entries: list[CuratedEntry]
    records: list[CatalogRecord]
    artifact_to_paper_id: dict[str, str]

    def notes_for(self, paper_id: str) -> list[str]:
        return [entry.text for entry in self.entries if paper_id in entry.linked_paper_ids]


def _split_row(line: str) -> list[str]:
    value = line.strip()
    if value.startswith("|"):
        value = value[1:]
    if value.endswith("|"):
        value = value[:-1]
    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for char in value:
        if char == "|" and not escaped:
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        escaped = char == "\\" and not escaped
        if char != "\\":
            escaped = False
    cells.append("".join(current).strip())
    return cells


def _clean_cell(value: str) -> str:
    value = LINK_RE.sub(lambda match: match.group(1), value)
    value = value.replace("<br>", "; ").replace("<br/>", "; ").replace("<br />", "; ")
    value = MARKUP_RE.sub("", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_record_id(value: str) -> str:
    value = _clean_cell(value).casefold()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value[:120]


def normalize_artifact_path(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return Path(normalized).as_posix()


def _record_id(first_cell: str, artifacts: list[str], first_header: str = "") -> str:
    candidate = normalize_record_id(first_cell)
    explicit_id_headers = {"name", "id", "paper", "paper id", "record", "work"}
    if first_header in explicit_id_headers and candidate and len(candidate) <= 120:
        return candidate
    if artifacts:
        return normalize_record_id(Path(artifacts[0]).stem)
    if candidate and len(candidate) <= 100 and not candidate.startswith(("http", "doi-org")):
        return candidate
    return candidate


def _heading_path(stack: list[str]) -> str:
    return " > ".join(item for item in stack if item)


def parse_curated_markdown(text: str, *, known_artifacts: Iterable[str] = ()) -> CuratedIndex:
    lines = text.splitlines()
    known_paths = list(dict.fromkeys(normalize_artifact_path(item) for item in known_artifacts))
    heading_stack: list[str] = []
    records: list[CatalogRecord] = []
    entries: list[CuratedEntry] = []
    artifact_to_paper_id: dict[str, str] = {}
    index = 0

    while index < len(lines):
        line = lines[index]
        heading_match = HEADING_RE.match(line)
        if heading_match:
            level = len(heading_match.group(1))
            title = _clean_cell(heading_match.group(2))
            heading_stack = heading_stack[: level - 1]
            heading_stack.extend([""] * max(0, level - 1 - len(heading_stack)))
            heading_stack.append(title)
            index += 1
            continue

        if line.lstrip().startswith("|") and index + 1 < len(lines):
            headers = [
                _clean_cell(cell).casefold() or f"column_{position + 1}"
                for position, cell in enumerate(_split_row(line))
            ]
            separators = [_clean_cell(cell) for cell in _split_row(lines[index + 1])]
            if separators and all(SEPARATOR_RE.match(cell.replace(" ", "")) for cell in separators):
                index += 2
                while index < len(lines) and lines[index].lstrip().startswith("|"):
                    cells = [_clean_cell(cell) for cell in _split_row(lines[index])]
                    if not any(cells):
                        index += 1
                        continue
                    if len(cells) < len(headers):
                        cells.extend([""] * (len(headers) - len(cells)))
                    fields = {
                        headers[pos]: cells[pos] for pos in range(min(len(headers), len(cells)))
                    }
                    artifacts = []
                    for match in ARTIFACT_RE.finditer(lines[index]):
                        artifact = normalize_artifact_path(match.group(1))
                        if artifact not in artifacts:
                            artifacts.append(artifact)
                    first = cells[0] if cells else ""
                    record_id = _record_id(first, artifacts, headers[0] if headers else "")
                    if not record_id:
                        record_id = f"curated-row-{index + 1}"
                    for artifact in artifacts:
                        # The canonical catalog row appears before the duplicate-
                        # alias inventory. Never let a later provenance row rename
                        # the paper and sever its rich notes/evaluation labels.
                        artifact_to_paper_id.setdefault(artifact, record_id)
                    linked = list(
                        dict.fromkeys(artifact_to_paper_id[artifact] for artifact in artifacts)
                    )
                    record = CatalogRecord(
                        record_id=record_id,
                        heading_path=_heading_path(heading_stack),
                        fields=fields,
                        artifacts=artifacts,
                        source_line=index + 1,
                    )
                    records.append(record)
                    entries.append(
                        CuratedEntry(
                            heading_path=record.heading_path,
                            text=record.searchable_text,
                            linked_paper_ids=linked,
                            source_line=index + 1,
                            entry_type="table_row",
                            artifacts=list(artifacts),
                        )
                    )
                    index += 1
                continue

        stripped = line.strip()
        if stripped.startswith(("- ", "* ")) and len(stripped) > 3:
            note_lines = [stripped[2:].strip()]
            start_line = index + 1
            index += 1
            while index < len(lines) and (
                lines[index].startswith("  ") or lines[index].startswith("\t")
            ):
                if lines[index].strip():
                    note_lines.append(lines[index].strip())
                index += 1
            raw_note = " ".join(note_lines)
            note = _clean_cell(raw_note)
            artifacts = [
                normalize_artifact_path(match.group(1)) for match in ARTIFACT_RE.finditer(raw_note)
            ]
            artifacts = list(dict.fromkeys(artifacts))
            if artifacts:
                note = f"{note}\nartifacts: {'; '.join(artifacts)}"
            linked = [
                artifact_to_paper_id.get(item, normalize_record_id(Path(item).stem))
                for item in artifacts
            ]
            entries.append(
                CuratedEntry(
                    heading_path=_heading_path(heading_stack),
                    text=note,
                    linked_paper_ids=list(dict.fromkeys(item for item in linked if item)),
                    source_line=start_line,
                    entry_type="bullet",
                    artifacts=artifacts,
                )
            )
            continue

        index += 1

    # Resolve known source paths against explicit catalog artifacts. Full relative paths
    # have priority; basename matching is permitted only when both the known-source
    # inventory and catalog references are unique under case folding.
    referenced_artifacts = sorted(
        {artifact for entry in entries for artifact in entry.artifacts}
        | {artifact for record in records for artifact in record.artifacts},
        key=lambda artifact: (artifact.casefold(), artifact),
    )
    for artifact in referenced_artifacts:
        artifact_to_paper_id.setdefault(
            artifact,
            normalize_record_id(Path(artifact).stem),
        )

    references_by_path: dict[str, list[str]] = {}
    references_by_basename: dict[str, list[str]] = {}
    for artifact in referenced_artifacts:
        references_by_path.setdefault(artifact.casefold(), []).append(artifact)
        references_by_basename.setdefault(Path(artifact).name.casefold(), []).append(artifact)

    known_by_path: dict[str, list[str]] = {}
    known_by_basename: dict[str, list[str]] = {}
    for artifact in known_paths:
        known_by_path.setdefault(artifact.casefold(), []).append(artifact)
        known_by_basename.setdefault(Path(artifact).name.casefold(), []).append(artifact)

    referenced_set = set(referenced_artifacts)
    for artifact in known_paths:
        basename_key = Path(artifact).name.casefold()
        basename_is_unique = len(known_by_basename[basename_key]) == 1
        is_bare_basename = len(Path(artifact).parts) == 1
        matched_reference: str | None = None
        if (not is_bare_basename or basename_is_unique) and artifact in referenced_set:
            matched_reference = artifact
        elif not is_bare_basename or basename_is_unique:
            path_matches = references_by_path.get(artifact.casefold(), [])
            if len(path_matches) == 1 and len(known_by_path[artifact.casefold()]) == 1:
                matched_reference = path_matches[0]
        if matched_reference is None and basename_is_unique:
            basename_matches = references_by_basename.get(basename_key, [])
            if len(basename_matches) == 1:
                matched_reference = basename_matches[0]
        if matched_reference is not None:
            artifact_to_paper_id[artifact] = artifact_to_paper_id[matched_reference]
        else:
            artifact_to_paper_id.setdefault(
                artifact,
                normalize_record_id(Path(artifact).stem),
            )

    # Publish basename aliases using every observed spelling only when the basename
    # identifies one known source and at most one catalog artifact. Without a known
    # inventory, one catalog path is sufficient. Case-only collisions stay ambiguous.
    basename_keys = set(references_by_basename) | set(known_by_basename)
    for basename_key in sorted(basename_keys):
        known_matches = known_by_basename.get(basename_key, [])
        reference_matches = references_by_basename.get(basename_key, [])
        if known_matches:
            if len(known_matches) != 1 or len(reference_matches) > 1:
                continue
            anchor = known_matches[0]
        else:
            if len(reference_matches) != 1:
                continue
            anchor = reference_matches[0]
        paper_id = artifact_to_paper_id[anchor]
        for artifact in [*known_matches, *reference_matches]:
            artifact_to_paper_id.setdefault(Path(artifact).name, paper_id)

    ambiguous_known_basenames = {
        basename_key for basename_key, artifacts in known_by_basename.items() if len(artifacts) > 1
    }
    for artifact in list(artifact_to_paper_id):
        if (
            len(Path(artifact).parts) == 1
            and Path(artifact).name.casefold() in ambiguous_known_basenames
        ):
            artifact_to_paper_id.pop(artifact)

    # Artifact-bearing entries may have been created before a later row established
    # the canonical display ID or a known source supplied a case-insensitive alias.
    for entry in entries:
        if entry.artifacts:
            entry.linked_paper_ids = list(
                dict.fromkeys(
                    artifact_to_paper_id[artifact]
                    for artifact in entry.artifacts
                    if artifact in artifact_to_paper_id
                )
            )
    # A rich catalog row may define the paper's explicit ID while a later,
    # inventory-style row supplies its artifact. Backfill those rich entries
    # after the complete table has been seen so their expert notes route too.
    resolved_ids = set(artifact_to_paper_id.values())
    records_by_line = {record.source_line: record for record in records}
    for entry in entries:
        record = records_by_line.get(entry.source_line)
        if (
            record is not None
            and not entry.artifacts
            and not entry.linked_paper_ids
            and record.record_id in resolved_ids
        ):
            entry.linked_paper_ids = [record.record_id]
    return CuratedIndex(entries=entries, records=records, artifact_to_paper_id=artifact_to_paper_id)


def parse_curated_index(path: Path, *, known_artifacts: Iterable[str] = ()) -> CuratedIndex:
    return parse_curated_markdown(path.read_text(encoding="utf-8"), known_artifacts=known_artifacts)
