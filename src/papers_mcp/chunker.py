from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from .config import ChunksConfig
from .models import Chunk, Section

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
PAGE_MARKER_RE = re.compile(r"^\s*<!--\s*page\s*:\s*(\d+)\s*-->\s*$", re.IGNORECASE)
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
MATH_ENV_START_RE = re.compile(
    r"\\begin\{(equation\*?|align\*?|aligned|gather\*?|multline\*?|cases|split|array|matrix|pmatrix|bmatrix)\}"
)
TABLE_LINE_RE = re.compile(r"^\s*\|?.*\|.*\|?\s*$")
THEOREM_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:theorem|lemma|proposition|corollary|definition)\b",
    re.IGNORECASE,
)
PROOF_RE = re.compile(r"^\s*(?:\*\*)?proof\b", re.IGNORECASE)
MIN_ATOMIC_HARD_LIMIT = 512


@dataclass(slots=True)
class _Block:
    text: str
    kind: str
    page_start: int | None = None
    page_end: int | None = None
    atomic: bool = False


def estimate_token_count(text: str) -> int:
    """Cheap deterministic token estimate suitable before a model tokenizer is loaded."""

    without_page_markers = re.sub(
        r"<!--\s*page\s*:\s*\d+\s*-->",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return len(TOKEN_RE.findall(without_page_markers))


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped and "|" in stripped and TABLE_LINE_RE.match(line))


def _parse_blocks(text: str) -> list[_Block]:
    blocks: list[_Block] = []
    buffer: list[str] = []
    kind = "prose"
    current_page: int | None = None
    block_pages: list[int] = []
    fence_character: str | None = None
    math_mode: str | None = None

    def flush(*, atomic: bool | None = None) -> None:
        nonlocal buffer, block_pages, kind
        if not any(line.strip() for line in buffer):
            buffer = []
            block_pages = []
            kind = "prose"
            return
        pages = block_pages or ([current_page] if current_page is not None else [])
        blocks.append(
            _Block(
                text="\n".join(buffer).strip(),
                kind=kind,
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
                atomic=(kind in {"math", "code", "table"}) if atomic is None else atomic,
            )
        )
        buffer = []
        block_pages = []
        kind = "prose"

    def append(line: str) -> None:
        buffer.append(line)
        if current_page is not None:
            block_pages.append(current_page)

    for line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        page_match = PAGE_MARKER_RE.match(line)
        if page_match:
            current_page = int(page_match.group(1))
            if fence_character is not None or math_mode is not None:
                append(line)
                continue
            flush()
            blocks.append(_Block(line.strip(), "page", current_page, current_page, atomic=True))
            continue

        if fence_character is not None:
            append(line)
            fence_match = FENCE_RE.match(line)
            if fence_match and fence_match.group(1)[0] == fence_character:
                flush(atomic=True)
                fence_character = None
            continue

        if math_mode is not None:
            append(line)
            stripped = line.strip()
            closes = (
                (math_mode == "$$" and stripped.count("$$") % 2 == 1)
                or (math_mode == "\\[" and "\\]" in stripped)
                or (math_mode == "[" and stripped == "]")
                or (
                    math_mode.startswith("env:")
                    and f"\\end{{{math_mode.removeprefix('env:')}}}" in stripped
                )
            )
            if closes:
                flush(atomic=True)
                math_mode = None
            continue

        fence_match = FENCE_RE.match(line)
        if fence_match:
            flush()
            kind = "code"
            fence_character = fence_match.group(1)[0]
            append(line)
            # A same-line pair such as ```code``` is uncommon but valid.
            if line.count(fence_character * 3) >= 2:
                flush(atomic=True)
                fence_character = None
            continue

        stripped = line.strip()
        env_match = MATH_ENV_START_RE.search(line)
        starts_dollars = stripped.startswith("$$")
        starts_bracket = stripped.startswith("\\[")
        starts_bare_bracket = stripped == "["
        if env_match or starts_dollars or starts_bracket or starts_bare_bracket:
            flush()
            kind = "math"
            append(line)
            if env_match:
                environment = env_match.group(1)
                if f"\\end{{{environment}}}" in line:
                    flush(atomic=True)
                else:
                    math_mode = f"env:{environment}"
            elif starts_bracket:
                if "\\]" in stripped[2:]:
                    flush(atomic=True)
                else:
                    math_mode = "\\["
            elif starts_bare_bracket:
                math_mode = "["
            elif stripped.count("$$") >= 2:
                flush(atomic=True)
            else:
                math_mode = "$$"
            continue

        if not stripped:
            flush()
            continue

        line_kind = "table" if _is_table_line(line) else "prose"
        if buffer and kind != line_kind:
            flush()
        kind = line_kind
        append(line)

    # Malformed/unclosed blocks are still kept whole rather than losing source text.
    flush(atomic=fence_character is not None or math_mode is not None or kind == "table")
    return blocks


def _merge_blocks(blocks: Iterable[_Block], kind: str | None = None) -> _Block:
    values = list(blocks)
    pages = [
        page for block in values for page in (block.page_start, block.page_end) if page is not None
    ]
    return _Block(
        text="\n\n".join(block.text for block in values if block.text).strip(),
        kind=kind or (values[0].kind if values else "prose"),
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        atomic=any(block.atomic for block in values),
    )


def _semantic_units(blocks: Sequence[_Block]) -> list[_Block]:
    """Attach equations/tables to their immediate mathematical explanation."""

    units: list[_Block] = []
    pending_pages: list[_Block] = []
    index = 0
    while index < len(blocks):
        block = blocks[index]
        if block.kind == "page":
            pending_pages.append(block)
            index += 1
            continue

        if pending_pages:
            block = _merge_blocks([*pending_pages, block], kind=block.kind)
            # Page markers carry provenance, not semantic atomicity.  Treating
            # their marker blocks as atomic accidentally disabled long-prose
            # splitting for the first content block on every page.
            block.atomic = blocks[index].atomic
            pending_pages = []

        if block.kind in {"math", "table", "code"}:
            group: list[_Block] = []
            if units and units[-1].kind == "prose":
                group.append(units.pop())
            group.append(block)
            if index + 1 < len(blocks) and blocks[index + 1].kind == "prose":
                group.append(blocks[index + 1])
                index += 1
            units.append(_merge_blocks(group, kind=block.kind))
        elif block.kind == "prose" and THEOREM_RE.match(block.text):
            group = [block]
            if index + 1 < len(blocks) and PROOF_RE.match(blocks[index + 1].text):
                group.append(blocks[index + 1])
                index += 1
            theorem = _merge_blocks(group, kind="theorem")
            theorem.atomic = True
            units.append(theorem)
        else:
            units.append(block)
        index += 1

    if pending_pages:
        if units:
            units[-1] = _merge_blocks([units[-1], *pending_pages], kind=units[-1].kind)
        else:
            units.extend(pending_pages)
    return units


def _split_long_prose(
    block: _Block,
    max_tokens: int,
    token_counter: Callable[[str], int],
) -> list[_Block]:
    if block.atomic or token_counter(block.text) <= max_tokens:
        return [block]

    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[])|\n+(?=\S)", block.text)
    pieces: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            pieces.append(" ".join(current).strip())
            current.clear()

    for sentence in sentences:
        if not sentence.strip():
            continue
        if token_counter(sentence) > max_tokens:
            flush()
            words = sentence.split()
            word_buffer: list[str] = []
            for word in words:
                proposed = " ".join([*word_buffer, word])
                if word_buffer and token_counter(proposed) > max_tokens:
                    pieces.append(" ".join(word_buffer))
                    word_buffer = [word]
                else:
                    word_buffer.append(word)
            if word_buffer:
                pieces.append(" ".join(word_buffer))
            continue
        proposed = " ".join([*current, sentence])
        if current and token_counter(proposed) > max_tokens:
            flush()
        current.append(sentence)
    flush()
    return [
        _Block(piece, "prose", block.page_start, block.page_end, atomic=False)
        for piece in pieces
        if piece
    ]


def _hard_split_block(
    block: _Block,
    max_tokens: int,
    token_counter: Callable[[str], int],
) -> list[_Block]:
    """Bound pathological blocks after semantic grouping, preserving all source text.

    Normal equations, tables, and code remain atomic. Only a block large enough
    to be truncated by the embedding context is divided, preferably at a line or
    word boundary. Authoritative section text remains untouched for reading.
    """

    limit = max(max_tokens, MIN_ATOMIC_HARD_LIMIT) if block.atomic else max_tokens
    if token_counter(block.text) <= limit:
        return [block]

    remaining = block.text
    pieces: list[_Block] = []
    while remaining and token_counter(remaining) > limit:
        low, high = 1, len(remaining)
        best = 0
        while low <= high:
            middle = (low + high) // 2
            if token_counter(remaining[:middle]) <= limit:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if best <= 0:
            best = 1
        boundary = max(
            remaining.rfind("\n", 0, best + 1),
            remaining.rfind(" ", 0, best + 1),
            remaining.rfind("\t", 0, best + 1),
        )
        if boundary >= max(1, best // 2):
            best = boundary + 1
        piece = remaining[:best].rstrip()
        if not piece:
            piece = remaining[:best]
        pieces.append(
            _Block(
                piece,
                block.kind,
                block.page_start,
                block.page_end,
                atomic=block.atomic,
            )
        )
        remaining = remaining[best:].lstrip()
    if remaining:
        pieces.append(
            _Block(
                remaining,
                block.kind,
                block.page_start,
                block.page_end,
                atomic=block.atomic,
            )
        )
    return pieces


class MathAwareChunker:
    def __init__(
        self,
        config: ChunksConfig | None = None,
        *,
        token_counter: Callable[[str], int] = estimate_token_count,
    ) -> None:
        self.config = config or ChunksConfig()
        if not 0 < self.config.min_tokens <= self.config.target_tokens <= self.config.max_tokens:
            raise ValueError(
                "chunk sizes must satisfy 0 < min_tokens <= target_tokens <= max_tokens"
            )
        self.token_counter = token_counter

    def chunk_section(self, section: Section) -> list[Chunk]:
        blocks = _semantic_units(_parse_blocks(section.text))
        units = [
            bounded
            for block in blocks
            for piece in _split_long_prose(block, self.config.max_tokens, self.token_counter)
            for bounded in _hard_split_block(
                piece,
                self.config.max_tokens,
                self.token_counter,
            )
        ]
        packed: list[_Block] = []
        current: list[_Block] = []

        def current_text(extra: _Block | None = None) -> str:
            values = [*current, *([extra] if extra is not None else [])]
            return "\n\n".join(value.text for value in values if value.text)

        def flush() -> None:
            if current:
                packed.append(_merge_blocks(current, kind="chunk"))
                current.clear()

        for unit in units:
            proposed_tokens = self.token_counter(current_text(unit))
            if current and proposed_tokens > self.config.max_tokens:
                flush()
            current.append(unit)
            current_tokens = self.token_counter(current_text())
            if current_tokens >= self.config.target_tokens:
                flush()
        flush()

        if len(packed) >= 2:
            last_tokens = self.token_counter(packed[-1].text)
            merged_tokens = self.token_counter(f"{packed[-2].text}\n\n{packed[-1].text}")
            if last_tokens < self.config.min_tokens and merged_tokens <= self.config.max_tokens:
                tail = packed.pop()
                packed[-1] = _merge_blocks([packed[-1], tail], kind="chunk")

        chunks: list[Chunk] = []
        for block in packed:
            token_count = self.token_counter(block.text)
            if token_count <= 0:
                # A page marker preserves provenance but is not a retrieval
                # document on its own.  Empty PDF pages must not become dense or
                # FTS candidates.
                continue
            page_start = block.page_start if block.page_start is not None else section.page_start
            page_end = block.page_end if block.page_end is not None else section.page_end
            chunks.append(
                Chunk(
                    paper_id=section.paper_id,
                    heading_path=section.heading_path,
                    text=block.text,
                    token_count=token_count,
                    chunk_index=len(chunks),
                    page_start=page_start,
                    page_end=page_end,
                    section_index=section.section_order,
                    section_id=section.id,
                )
            )
        return chunks

    def chunk_sections(self, sections: Sequence[Section]) -> list[Chunk]:
        chunks: list[Chunk] = []
        next_index_by_paper: dict[str, int] = {}
        for section in sections:
            section_chunks = self.chunk_section(section)
            next_index = next_index_by_paper.get(section.paper_id, 0)
            for chunk in section_chunks:
                chunk.chunk_index = next_index
                next_index += 1
            next_index_by_paper[section.paper_id] = next_index
            chunks.extend(section_chunks)
        return chunks


def chunk_section(
    section: Section,
    config: ChunksConfig | None = None,
    *,
    token_counter: Callable[[str], int] = estimate_token_count,
) -> list[Chunk]:
    return MathAwareChunker(config, token_counter=token_counter).chunk_section(section)


def chunk_sections(
    sections: Sequence[Section],
    config: ChunksConfig | None = None,
    *,
    token_counter: Callable[[str], int] = estimate_token_count,
) -> list[Chunk]:
    return MathAwareChunker(config, token_counter=token_counter).chunk_sections(sections)
