from __future__ import annotations

import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

from .models import Paper, Section

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+^-]{2,}")
STOPWORDS = {
    "about",
    "after",
    "also",
    "among",
    "and",
    "are",
    "based",
    "been",
    "being",
    "between",
    "can",
    "could",
    "does",
    "each",
    "for",
    "from",
    "have",
    "into",
    "its",
    "more",
    "most",
    "not",
    "only",
    "other",
    "paper",
    "results",
    "section",
    "should",
    "such",
    "than",
    "that",
    "the",
    "their",
    "then",
    "these",
    "this",
    "through",
    "using",
    "was",
    "were",
    "which",
    "with",
}


def infer_topics(text: str, *, limit: int = 16) -> list[str]:
    tokens = [match.group(0).casefold() for match in WORD_RE.finditer(text)]
    counts = Counter(token for token in tokens if token not in STOPWORDS and not token.isdigit())
    return [token for token, _ in counts.most_common(limit)]


def build_research_card(
    paper: Paper,
    sections: list[Section],
    curated_notes: list[str],
) -> dict[str, object]:
    headings = [section.heading_path for section in sections if section.heading][:80]
    notes = "\n".join(dict.fromkeys(note.strip() for note in curated_notes if note.strip()))
    topic_text = "\n".join([paper.title, paper.abstract, *headings, notes])
    return {
        "generated_by": "papers-mcp",
        "paper_id": paper.id,
        "title": paper.title,
        "authors": paper.authors,
        "year": paper.year,
        "doi": paper.doi,
        "abstract": paper.abstract,
        "source_kind": paper.source_kind,
        "section_headings": headings,
        "curated_notes": notes,
        "topics": infer_topics(topic_text),
        "authority": "Original paper text remains authoritative; this card is deterministic routing metadata.",
    }


def write_research_card(path: Path, card: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    staged = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(card, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, path)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
