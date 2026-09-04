from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import Section

HEADING_RE = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
PAGE_MARKER_RE = re.compile(r"^\s*<!--\s*page\s*:\s*(\d+)\s*-->\s*$", re.IGNORECASE)
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
SECTION_NUMBER_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)(?:\.|\s)")


@dataclass(slots=True)
class _SectionBuilder:
    heading: str
    heading_path: str
    level: int
    parent_index: int | None
    page_start: int | None
    lines: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)

    def append(self, line: str, page: int | None) -> None:
        self.lines.append(line)
        if page is not None:
            self.pages.append(page)


def _clean_heading(value: str) -> str:
    value = re.sub(r"\s+#+\s*$", "", value).strip()
    value = re.sub(r"\s+\{#[^}]+}\s*$", "", value).strip()
    changed = True
    while changed:
        changed = False
        for marker in ("**", "__", "`", "*", "_"):
            if value.startswith(marker) and value.endswith(marker) and len(value) > 2 * len(marker):
                value = value[len(marker) : -len(marker)].strip()
                changed = True
                break
    return value


def _section_number(value: str) -> str | None:
    match = SECTION_NUMBER_RE.match(value)
    return match.group(1) if match else None


def _toggle_math_fence(line: str, active: bool) -> bool:
    stripped = line.strip()
    if stripped == "[" and not active:
        return True
    if active and stripped == "]":
        return False
    if stripped.startswith("\\[") and not stripped.endswith("\\]"):
        return True
    if active and stripped.endswith("\\]"):
        return False
    if stripped.count("$$") % 2 == 1:
        return not active
    return active


def _fallback_sections(markdown: str, paper_id: str, document_title: str) -> list[Section]:
    """Create page-aware synthetic sections when extraction found no headings."""

    segments: list[tuple[int | None, int | None, list[str]]] = []
    current_page: int | None = None
    current_pages: list[int] = []
    current_lines: list[str] = []
    code_fence: str | None = None
    math_fence = False

    def flush() -> None:
        nonlocal current_lines, current_pages
        if any(line.strip() for line in current_lines):
            segments.append(
                (
                    min(current_pages) if current_pages else current_page,
                    max(current_pages) if current_pages else current_page,
                    current_lines,
                )
            )
        current_lines = []
        current_pages = []

    for line in markdown.splitlines():
        protected = code_fence is not None or math_fence
        page_match = PAGE_MARKER_RE.match(line)
        if page_match:
            if not protected:
                flush()
            current_page = int(page_match.group(1))
            current_pages.append(current_page)
            current_lines.append(line)
            continue
        current_lines.append(line)

        fence_match = FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)[0]
            if code_fence is None:
                code_fence = marker
            elif code_fence == marker:
                code_fence = None
        elif code_fence is None:
            math_fence = _toggle_math_fence(line, math_fence)
    flush()

    if not segments:
        return []
    multiple_pages = len(segments) > 1 or any(start is not None for start, _, _ in segments)
    sections: list[Section] = []
    for order, (page_start, page_end, lines) in enumerate(segments):
        if multiple_pages and page_start is not None:
            heading = (
                f"Page {page_start}"
                if page_end in {None, page_start}
                else f"Pages {page_start}–{page_end}"
            )
        elif order == 0:
            heading = document_title or "Document"
        else:
            heading = f"Document part {order + 1}"
        sections.append(
            Section(
                paper_id=paper_id,
                heading=heading,
                heading_path=heading,
                text="\n".join(lines).strip(),
                level=1,
                section_order=order,
                page_start=page_start,
                page_end=page_end,
            )
        )
    return sections


class MarkdownSectionParser:
    """Parse Markdown headings into a flat section list with parent indices."""

    def parse(
        self,
        markdown: str,
        paper_id: str,
        document_title: str = "",
    ) -> list[Section]:
        normalized = markdown.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
        if not normalized.strip():
            return []

        builders: list[_SectionBuilder] = []
        stack: list[int] = []
        preamble: list[tuple[str, int | None]] = []
        current_index: int | None = None
        current_page: int | None = None
        pending_page_marker: str | None = None
        code_fence: str | None = None
        math_fence = False
        found_heading = False

        for line in normalized.splitlines():
            protected = code_fence is not None or math_fence
            page_match = PAGE_MARKER_RE.match(line)
            if page_match:
                current_page = int(page_match.group(1))
                if protected:
                    target = builders[current_index] if current_index is not None else None
                    if target is None:
                        preamble.append((line, current_page))
                    else:
                        target.append(line, current_page)
                    continue
                pending_page_marker = line
                continue

            # Extractors commonly leave a blank line between a page marker and
            # the first heading on that page. Keep the marker pending so it is
            # attached to the new section rather than the previous one.
            if pending_page_marker is not None and not line.strip():
                continue

            heading_match = HEADING_RE.match(line) if not protected else None
            if heading_match:
                found_heading = True
                level = len(heading_match.group(1))
                heading = _clean_heading(heading_match.group(2)) or "Untitled section"
                while stack and builders[stack[-1]].level >= level:
                    stack.pop()
                parent_index = stack[-1] if stack else None
                section_number = _section_number(heading)
                if section_number and "." in section_number:
                    parent_number = section_number.rsplit(".", 1)[0]
                    numbered_parent = next(
                        (
                            index
                            for index in range(len(builders) - 1, -1, -1)
                            if _section_number(builders[index].heading) == parent_number
                        ),
                        None,
                    )
                    if numbered_parent is not None:
                        parent_index = numbered_parent
                parent_path = (
                    builders[parent_index].heading_path if parent_index is not None else ""
                )
                heading_path = f"{parent_path} > {heading}" if parent_path else heading
                builder = _SectionBuilder(
                    heading=heading,
                    heading_path=heading_path,
                    level=level,
                    parent_index=parent_index,
                    page_start=current_page,
                )
                if pending_page_marker is not None:
                    builder.append(pending_page_marker, current_page)
                    pending_page_marker = None
                builders.append(builder)
                current_index = len(builders) - 1
                stack.append(current_index)
                continue

            target = builders[current_index] if current_index is not None else None
            if pending_page_marker is not None:
                if target is None:
                    preamble.append((pending_page_marker, current_page))
                else:
                    target.append(pending_page_marker, current_page)
                pending_page_marker = None
            if target is None:
                preamble.append((line, current_page))
            else:
                target.append(line, current_page)

            fence_match = FENCE_RE.match(line)
            if fence_match:
                marker = fence_match.group(1)[0]
                if code_fence is None:
                    code_fence = marker
                elif code_fence == marker:
                    code_fence = None
            elif code_fence is None:
                math_fence = _toggle_math_fence(line, math_fence)

        if not found_heading:
            return _fallback_sections(normalized, paper_id, document_title)

        if pending_page_marker is not None:
            target = builders[current_index] if current_index is not None else None
            if target is None:
                preamble.append((pending_page_marker, current_page))
            else:
                target.append(pending_page_marker, current_page)

        if any(line.strip() and PAGE_MARKER_RE.match(line) is None for line, _ in preamble):
            preamble_pages = [page for _, page in preamble if page is not None]
            preamble_heading = document_title or "Preamble"
            preamble_builder = _SectionBuilder(
                heading=preamble_heading,
                heading_path=preamble_heading,
                level=1,
                parent_index=None,
                page_start=min(preamble_pages) if preamble_pages else None,
                lines=[line for line, _ in preamble],
                pages=preamble_pages,
            )
            builders.insert(0, preamble_builder)
            for builder in builders[1:]:
                if builder.parent_index is not None:
                    builder.parent_index += 1

        sections: list[Section] = []
        for order, builder in enumerate(builders):
            pages = builder.pages or (
                [builder.page_start] if builder.page_start is not None else []
            )
            sections.append(
                Section(
                    paper_id=paper_id,
                    heading=builder.heading,
                    heading_path=builder.heading_path,
                    text="\n".join(builder.lines).strip(),
                    level=builder.level,
                    section_order=order,
                    page_start=min(pages) if pages else None,
                    page_end=max(pages) if pages else None,
                    parent_index=builder.parent_index,
                )
            )

        # Top-level logical sections cover their descendants even though text remains direct.
        for index in range(len(sections) - 1, -1, -1):
            parent_index = sections[index].parent_index
            if parent_index is None:
                continue
            child = sections[index]
            parent = sections[parent_index]
            page_values = [
                page
                for page in (parent.page_start, parent.page_end, child.page_start, child.page_end)
                if page is not None
            ]
            if page_values:
                parent.page_start = min(page_values)
                parent.page_end = max(page_values)
        return sections


def parse_sections(markdown: str, paper_id: str, document_title: str = "") -> list[Section]:
    return MarkdownSectionParser().parse(markdown, paper_id, document_title)
