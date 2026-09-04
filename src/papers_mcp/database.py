from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .chunker import TOKEN_RE
from .models import (
    Candidate,
    Chunk,
    CuratedEntry,
    IngestFailure,
    Paper,
    Section,
    SourceDocument,
)
from .vectors import Vector, cosine_top_k, decode_float32, encode_float32

SCHEMA_VERSION = 1
CORPUS_SNAPSHOT_COMPLETE_KEY = "corpus_snapshot_complete"
CORPUS_SOURCE_MANIFEST_KEY = "corpus_source_manifest"
ENTITY_KINDS = frozenset({"paper", "section", "chunk", "curated"})
FTS_TABLES = {
    "paper": ("papers_fts", "paper_id"),
    "section": ("sections_fts", "section_id"),
    "chunk": ("chunks_fts", "chunk_id"),
    "curated": ("curated_entries_fts", "entry_id"),
}


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS papers (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    authors_json TEXT NOT NULL DEFAULT '[]',
    year INTEGER,
    doi TEXT,
    abstract TEXT NOT NULL DEFAULT '',
    source_path TEXT NOT NULL,
    source_kind TEXT NOT NULL DEFAULT 'pdf',
    markdown_path TEXT,
    content_hash TEXT NOT NULL,
    extraction_backend TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS papers_content_hash_idx ON papers(content_hash);
CREATE INDEX IF NOT EXISTS papers_normalized_title_idx ON papers(title COLLATE NOCASE, year);

CREATE TABLE IF NOT EXISTS source_aliases (
    source_path TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    source_kind TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS source_aliases_paper_idx ON source_aliases(paper_id);
CREATE INDEX IF NOT EXISTS source_aliases_hash_idx ON source_aliases(content_hash);

CREATE TABLE IF NOT EXISTS sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    parent_section_id INTEGER REFERENCES sections(id) ON DELETE CASCADE,
    heading TEXT NOT NULL DEFAULT '',
    heading_path TEXT NOT NULL DEFAULT '',
    level INTEGER NOT NULL DEFAULT 1 CHECK(level >= 0),
    section_order INTEGER NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    text TEXT NOT NULL,
    CHECK(page_start IS NULL OR page_start >= 1),
    CHECK(page_end IS NULL OR page_end >= 1),
    CHECK(page_start IS NULL OR page_end IS NULL OR page_end >= page_start)
);
CREATE INDEX IF NOT EXISTS sections_paper_order_idx ON sections(paper_id, section_order, id);
CREATE INDEX IF NOT EXISTS sections_parent_idx ON sections(parent_section_id);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    section_id INTEGER REFERENCES sections(id) ON DELETE SET NULL,
    chunk_index INTEGER NOT NULL,
    heading_path TEXT NOT NULL DEFAULT '',
    page_start INTEGER,
    page_end INTEGER,
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL CHECK(token_count >= 0),
    CHECK(page_start IS NULL OR page_start >= 1),
    CHECK(page_end IS NULL OR page_end >= 1),
    CHECK(page_start IS NULL OR page_end IS NULL OR page_end >= page_start)
);
CREATE INDEX IF NOT EXISTS chunks_paper_order_idx ON chunks(paper_id, chunk_index, id);
CREATE INDEX IF NOT EXISTS chunks_section_idx ON chunks(section_id);

CREATE TABLE IF NOT EXISTS curated_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    heading_path TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL,
    source_line INTEGER,
    entry_type TEXT NOT NULL DEFAULT 'note'
);

CREATE TABLE IF NOT EXISTS curated_entry_papers (
    entry_id INTEGER NOT NULL REFERENCES curated_entries(id) ON DELETE CASCADE,
    paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    PRIMARY KEY(entry_id, paper_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS curated_entry_papers_paper_idx ON curated_entry_papers(paper_id);

CREATE TABLE IF NOT EXISTS embeddings (
    kind TEXT NOT NULL CHECK(kind IN ('paper', 'section', 'chunk', 'curated')),
    entity_id TEXT NOT NULL,
    paper_id TEXT,
    model_name TEXT NOT NULL,
    model_fingerprint TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK(dimensions > 0),
    vector BLOB NOT NULL,
    normalized INTEGER NOT NULL DEFAULT 1 CHECK(normalized IN (0, 1)),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(kind, entity_id, model_fingerprint),
    CHECK(length(vector) = dimensions * 4)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS embeddings_lookup_idx
    ON embeddings(model_fingerprint, kind, paper_id);

CREATE TABLE IF NOT EXISTS failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_path TEXT NOT NULL,
    content_hash TEXT,
    stage TEXT NOT NULL,
    error TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(source_path, stage)
);
CREATE INDEX IF NOT EXISTS failures_source_idx ON failures(source_path);

CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    paper_id UNINDEXED,
    title,
    authors,
    abstract,
    tokenize='unicode61 remove_diacritics 2'
);
CREATE VIRTUAL TABLE IF NOT EXISTS sections_fts USING fts5(
    section_id UNINDEXED,
    paper_id UNINDEXED,
    title,
    heading_path,
    text,
    tokenize='unicode61 remove_diacritics 2'
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    paper_id UNINDEXED,
    title,
    heading_path,
    text,
    tokenize='unicode61 remove_diacritics 2'
);
CREATE VIRTUAL TABLE IF NOT EXISTS curated_entries_fts USING fts5(
    entry_id UNINDEXED,
    heading_path,
    entry_type,
    text,
    tokenize='unicode61 remove_diacritics 2'
);

DROP TRIGGER IF EXISTS papers_fts_insert;
DROP TRIGGER IF EXISTS papers_fts_delete;
DROP TRIGGER IF EXISTS papers_fts_update;
DROP TRIGGER IF EXISTS sections_fts_insert;
DROP TRIGGER IF EXISTS sections_fts_delete;
DROP TRIGGER IF EXISTS sections_fts_update;
DROP TRIGGER IF EXISTS chunks_fts_insert;
DROP TRIGGER IF EXISTS chunks_fts_delete;
DROP TRIGGER IF EXISTS chunks_fts_update;
DROP TRIGGER IF EXISTS curated_fts_insert;
DROP TRIGGER IF EXISTS curated_fts_delete;
DROP TRIGGER IF EXISTS curated_fts_update;

CREATE TRIGGER IF NOT EXISTS papers_fts_insert AFTER INSERT ON papers BEGIN
    INSERT INTO papers_fts(paper_id, title, authors, abstract)
    VALUES (new.id, new.title, new.authors_json, new.abstract);
END;
CREATE TRIGGER IF NOT EXISTS papers_fts_delete AFTER DELETE ON papers BEGIN
    DELETE FROM papers_fts WHERE paper_id = old.id;
END;
CREATE TRIGGER IF NOT EXISTS papers_fts_update AFTER UPDATE ON papers BEGIN
    DELETE FROM papers_fts WHERE paper_id = old.id;
    INSERT INTO papers_fts(paper_id, title, authors, abstract)
    VALUES (new.id, new.title, new.authors_json, new.abstract);
END;

CREATE TRIGGER IF NOT EXISTS sections_fts_insert AFTER INSERT ON sections BEGIN
    INSERT INTO sections_fts(rowid, section_id, paper_id, title, heading_path, text)
    VALUES (
        new.id,
        new.id,
        new.paper_id,
        COALESCE((SELECT title FROM papers WHERE id = new.paper_id), ''),
        new.heading_path,
        new.text
    );
END;
CREATE TRIGGER IF NOT EXISTS sections_fts_delete AFTER DELETE ON sections BEGIN
    DELETE FROM sections_fts WHERE rowid = old.id;
END;
CREATE TRIGGER IF NOT EXISTS sections_fts_update AFTER UPDATE ON sections BEGIN
    DELETE FROM sections_fts WHERE rowid = old.id;
    INSERT INTO sections_fts(rowid, section_id, paper_id, title, heading_path, text)
    VALUES (
        new.id,
        new.id,
        new.paper_id,
        COALESCE((SELECT title FROM papers WHERE id = new.paper_id), ''),
        new.heading_path,
        new.text
    );
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, chunk_id, paper_id, title, heading_path, text)
    VALUES (
        new.id,
        new.id,
        new.paper_id,
        COALESCE((SELECT title FROM papers WHERE id = new.paper_id), ''),
        new.heading_path,
        new.text
    );
END;
CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE rowid = old.id;
END;
CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE rowid = old.id;
    INSERT INTO chunks_fts(rowid, chunk_id, paper_id, title, heading_path, text)
    VALUES (
        new.id,
        new.id,
        new.paper_id,
        COALESCE((SELECT title FROM papers WHERE id = new.paper_id), ''),
        new.heading_path,
        new.text
    );
END;

CREATE TRIGGER IF NOT EXISTS curated_fts_insert AFTER INSERT ON curated_entries BEGIN
    INSERT INTO curated_entries_fts(rowid, entry_id, heading_path, entry_type, text)
    VALUES (new.id, new.id, new.heading_path, new.entry_type, new.text);
END;
CREATE TRIGGER IF NOT EXISTS curated_fts_delete AFTER DELETE ON curated_entries BEGIN
    DELETE FROM curated_entries_fts WHERE rowid = old.id;
END;
CREATE TRIGGER IF NOT EXISTS curated_fts_update AFTER UPDATE ON curated_entries BEGIN
    DELETE FROM curated_entries_fts WHERE rowid = old.id;
    INSERT INTO curated_entries_fts(rowid, entry_id, heading_path, entry_type, text)
    VALUES (new.id, new.id, new.heading_path, new.entry_type, new.text);
END;
"""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _json_authors(value: Sequence[str]) -> str:
    return json.dumps(list(value), ensure_ascii=False, separators=(",", ":"))


def _load_authors(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    return [str(item) for item in decoded] if isinstance(decoded, list) else []


def _fts_query(query: str, *, exact: bool) -> str | None:
    cleaned = query.strip()
    if not cleaned:
        return None
    if exact:
        return f'"{cleaned.replace(chr(34), chr(34) * 2)}"'
    tokens = re.findall(r"[^\W_]+(?:[_'’-][^\W_]+)*", cleaned, flags=re.UNICODE)
    unique_tokens = list(dict.fromkeys(token.casefold() for token in tokens if token))
    if not unique_tokens:
        return None
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in unique_tokens)


class CorpusDatabase:
    """SQLite persistence for the paper/section/chunk retrieval hierarchy."""

    def __init__(
        self,
        path: str | Path,
        *,
        busy_timeout_ms: int = 5_000,
        initialize: bool = True,
    ) -> None:
        self.path = str(Path(path).expanduser()) if str(path) != ":memory:" else ":memory:"
        if self.path != ":memory:":
            Path(self.path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._transaction_depth = 0
        self.connection = sqlite3.connect(
            self.path,
            timeout=max(0.001, busy_timeout_ms / 1_000),
            isolation_level=None,
            check_same_thread=False,
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
        self.connection.execute("PRAGMA synchronous = NORMAL")
        self.connection.execute("PRAGMA journal_mode = WAL")
        if initialize:
            self.initialize()

    def __enter__(self) -> CorpusDatabase:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def initialize(self) -> None:
        with self._lock:
            meta_exists = self.connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta'"
            ).fetchone()
            if meta_exists is not None:
                row = self.connection.execute(
                    "SELECT value FROM meta WHERE key = 'schema_version'"
                ).fetchone()
                if row is not None:
                    try:
                        current = int(row[0])
                    except (TypeError, ValueError) as exc:
                        raise RuntimeError("database schema version is invalid") from exc
                    if current != SCHEMA_VERSION:
                        raise RuntimeError(
                            f"database schema {current} is incompatible with expected "
                            f"{SCHEMA_VERSION}"
                        )
            try:
                self.connection.executescript(SCHEMA_SQL)
            except sqlite3.OperationalError as exc:
                if "fts5" in str(exc).casefold():
                    raise RuntimeError("this SQLite build does not provide FTS5") from exc
                raise
            self.connection.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self.connection.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('revision', '0')"
            )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            depth = self._transaction_depth
            savepoint = f"papers_savepoint_{depth}"
            if depth == 0:
                self.connection.execute("BEGIN IMMEDIATE")
            else:
                self.connection.execute(f"SAVEPOINT {savepoint}")
            self._transaction_depth += 1
            try:
                yield self.connection
            except BaseException:
                self._transaction_depth -= 1
                if depth == 0:
                    self.connection.execute("ROLLBACK")
                else:
                    self.connection.execute(f"ROLLBACK TO {savepoint}")
                    self.connection.execute(f"RELEASE {savepoint}")
                raise
            else:
                self._transaction_depth -= 1
                if depth == 0:
                    self.connection.execute("COMMIT")
                else:
                    self.connection.execute(f"RELEASE {savepoint}")

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        with self._lock:
            row = self.connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return str(row[0]) if row is not None else default

    def set_meta(self, key: str, value: str) -> None:
        with self.transaction():
            self.connection.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(value)),
            )

    def delete_meta(self, key: str) -> bool:
        with self.transaction():
            cursor = self.connection.execute("DELETE FROM meta WHERE key = ?", (key,))
        return bool(cursor.rowcount)

    @property
    def revision(self) -> int:
        return int(self.get_meta("revision", "0") or 0)

    def get_revision(self) -> int:
        return self.revision

    def _bump_revision(self) -> int:
        self.connection.execute(
            "UPDATE meta SET value = CAST(CAST(value AS INTEGER) + 1 AS TEXT) "
            "WHERE key = 'revision'"
        )
        row = self.connection.execute("SELECT value FROM meta WHERE key = 'revision'").fetchone()
        return int(row[0])

    def _paper_from_row(self, row: sqlite3.Row) -> Paper:
        return Paper(
            id=str(row["id"]),
            title=str(row["title"]),
            authors=_load_authors(row["authors_json"]),
            year=row["year"],
            doi=row["doi"],
            abstract=str(row["abstract"] or ""),
            source_path=str(row["source_path"]),
            source_kind=str(row["source_kind"]),  # type: ignore[arg-type]
            markdown_path=row["markdown_path"],
            content_hash=str(row["content_hash"]),
            extraction_backend=row["extraction_backend"],
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def get_paper(self, paper_id: str) -> Paper | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM papers WHERE id = ?", (paper_id,)
            ).fetchone()
        return self._paper_from_row(row) if row is not None else None

    def list_papers(self) -> list[Paper]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM papers ORDER BY COALESCE(year, 9999), title COLLATE NOCASE, id"
            ).fetchall()
        return [self._paper_from_row(row) for row in rows]

    def find_paper_by_hash(self, content_hash: str) -> Paper | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM papers WHERE content_hash = ? ORDER BY id LIMIT 1",
                (content_hash,),
            ).fetchone()
        return self._paper_from_row(row) if row is not None else None

    paper_by_hash = find_paper_by_hash

    def find_paper_by_source(self, source_path: str | Path) -> Paper | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT p.* FROM source_aliases a JOIN papers p ON p.id = a.paper_id "
                "WHERE a.source_path = ?",
                (str(source_path),),
            ).fetchone()
            if row is None:
                row = self.connection.execute(
                    "SELECT * FROM papers WHERE source_path = ?", (str(source_path),)
                ).fetchone()
        return self._paper_from_row(row) if row is not None else None

    paper_by_source = find_paper_by_source

    def is_unchanged(self, source_path: str | Path, content_hash: str) -> bool:
        with self._lock:
            row = self.connection.execute(
                "SELECT 1 FROM source_aliases WHERE source_path = ? AND content_hash = ?",
                (str(source_path), content_hash),
            ).fetchone()
            if row is None:
                row = self.connection.execute(
                    "SELECT 1 FROM papers WHERE source_path = ? AND content_hash = ?",
                    (str(source_path), content_hash),
                ).fetchone()
        return row is not None

    def add_source_alias(
        self,
        paper_id: str,
        source_path: str | Path,
        *,
        content_hash: str,
        source_kind: str = "pdf",
    ) -> None:
        now = _utc_now()
        with self.transaction():
            self._upsert_alias(paper_id, str(source_path), source_kind, content_hash, now)
            self._bump_revision()

    def remove_source_alias(self, source_path: str | Path) -> bool:
        with self.transaction():
            cursor = self.connection.execute(
                "DELETE FROM source_aliases WHERE source_path = ?", (str(source_path),)
            )
            if cursor.rowcount:
                self._bump_revision()
        return bool(cursor.rowcount)

    def _upsert_alias(
        self,
        paper_id: str,
        source_path: str,
        source_kind: str,
        content_hash: str,
        now: str,
    ) -> None:
        self.connection.execute(
            "INSERT INTO source_aliases("
            "source_path, paper_id, source_kind, content_hash, created_at, updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(source_path) DO UPDATE SET "
            "paper_id = excluded.paper_id, source_kind = excluded.source_kind, "
            "content_hash = excluded.content_hash, updated_at = excluded.updated_at",
            (source_path, paper_id, source_kind, content_hash, now, now),
        )

    def source_aliases(self, paper_id: str) -> list[dict[str, str]]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT source_path, source_kind, content_hash FROM source_aliases "
                "WHERE paper_id = ? ORDER BY source_path",
                (paper_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def repoint_paper_source(self, paper_id: str, source_path: str | Path) -> None:
        """Make an existing alias the paper's canonical source without changing its content."""

        path = str(source_path)
        now = _utc_now()
        with self.transaction():
            alias = self.connection.execute(
                "SELECT source_kind, content_hash FROM source_aliases "
                "WHERE paper_id = ? AND source_path = ?",
                (paper_id, path),
            ).fetchone()
            if alias is None:
                raise KeyError(f"unknown source alias {path!r} for paper {paper_id}")
            markdown_path = path if str(alias["source_kind"]) == "markdown_reference" else None
            cursor = self.connection.execute(
                "UPDATE papers SET source_path = ?, source_kind = ?, "
                "markdown_path = COALESCE(?, markdown_path), updated_at = ? WHERE id = ?",
                (path, str(alias["source_kind"]), markdown_path, now, paper_id),
            )
            if not cursor.rowcount:
                raise KeyError(f"unknown paper: {paper_id}")
            self._bump_revision()

    def replace_paper(
        self,
        paper: Paper,
        sections: Sequence[Section],
        chunks: Sequence[Chunk],
        *,
        aliases: Iterable[str | Path | SourceDocument | Mapping[str, Any]] = (),
        reconcile_aliases: bool = False,
        before_commit: Callable[[], None] | None = None,
    ) -> str:
        """Atomically replace a paper and all of its hierarchical retrieval units."""

        for section in sections:
            if section.paper_id != paper.id:
                raise ValueError("all sections must belong to the replaced paper")
            if section.parent_index is not None and not 0 <= section.parent_index < len(sections):
                raise ValueError("section parent_index is out of range")
        for chunk in chunks:
            if chunk.paper_id != paper.id:
                raise ValueError("all chunks must belong to the replaced paper")
            if chunk.section_index is not None and not 0 <= chunk.section_index < len(sections):
                raise ValueError("chunk section_index is out of range")

        alias_items = list(aliases)
        now = _utc_now()
        original_section_ids = [section.id for section in sections]
        original_parent_ids = [section.parent_section_id for section in sections]
        original_chunk_section_ids = [chunk.section_id for chunk in chunks]
        with self.transaction():
            existing = self.connection.execute(
                "SELECT created_at FROM papers WHERE id = ?", (paper.id,)
            ).fetchone()
            created_at = str(existing[0]) if existing is not None else paper.created_at or now
            self.connection.execute(
                "INSERT INTO papers("
                "id, title, authors_json, year, doi, abstract, source_path, source_kind, "
                "markdown_path, content_hash, extraction_backend, created_at, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "title = excluded.title, authors_json = excluded.authors_json, "
                "year = excluded.year, doi = excluded.doi, abstract = excluded.abstract, "
                "source_path = excluded.source_path, source_kind = excluded.source_kind, "
                "markdown_path = excluded.markdown_path, content_hash = excluded.content_hash, "
                "extraction_backend = excluded.extraction_backend, updated_at = excluded.updated_at",
                (
                    paper.id,
                    paper.title,
                    _json_authors(paper.authors),
                    paper.year,
                    paper.doi,
                    paper.abstract,
                    paper.source_path,
                    paper.source_kind,
                    paper.markdown_path,
                    paper.content_hash,
                    paper.extraction_backend,
                    created_at,
                    now,
                ),
            )
            self.connection.execute(
                "DELETE FROM embeddings WHERE paper_id = ? "
                "AND kind IN ('paper', 'section', 'chunk')",
                (paper.id,),
            )
            self.connection.execute("DELETE FROM chunks WHERE paper_id = ?", (paper.id,))
            self.connection.execute("DELETE FROM sections WHERE paper_id = ?", (paper.id,))

            new_section_ids: list[int] = []
            for section in sections:
                cursor = self.connection.execute(
                    "INSERT INTO sections("
                    "paper_id, parent_section_id, heading, heading_path, level, section_order, "
                    "page_start, page_end, text"
                    ") VALUES (?, NULL, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        paper.id,
                        section.heading,
                        section.heading_path,
                        section.level,
                        section.section_order,
                        section.page_start,
                        section.page_end,
                        section.text,
                    ),
                )
                new_section_ids.append(int(cursor.lastrowid))

            old_to_new = {
                old_id: new_section_ids[index]
                for index, old_id in enumerate(original_section_ids)
                if old_id is not None
            }
            for index, section in enumerate(sections):
                parent_id: int | None = None
                if section.parent_index is not None:
                    parent_id = new_section_ids[section.parent_index]
                elif original_parent_ids[index] is not None:
                    parent_id = old_to_new.get(original_parent_ids[index])
                if parent_id == new_section_ids[index]:
                    raise ValueError("a section cannot be its own parent")
                if parent_id is not None:
                    self.connection.execute(
                        "UPDATE sections SET parent_section_id = ? WHERE id = ?",
                        (parent_id, new_section_ids[index]),
                    )
                section.id = new_section_ids[index]
                section.parent_section_id = parent_id

            for index, chunk in enumerate(chunks):
                section_id: int | None = None
                if chunk.section_index is not None:
                    section_id = new_section_ids[chunk.section_index]
                elif original_chunk_section_ids[index] is not None:
                    section_id = old_to_new.get(original_chunk_section_ids[index])
                cursor = self.connection.execute(
                    "INSERT INTO chunks("
                    "paper_id, section_id, chunk_index, heading_path, page_start, page_end, "
                    "text, token_count"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        paper.id,
                        section_id,
                        chunk.chunk_index,
                        chunk.heading_path,
                        chunk.page_start,
                        chunk.page_end,
                        chunk.text,
                        chunk.token_count,
                    ),
                )
                chunk.id = int(cursor.lastrowid)
                chunk.section_id = section_id

            if reconcile_aliases:
                retained_paths = {str(paper.source_path)} if paper.source_path else set()
                for alias in alias_items:
                    if isinstance(alias, SourceDocument):
                        retained_paths.add(str(alias.path))
                    elif isinstance(alias, Mapping):
                        retained_paths.add(str(alias["source_path"]))
                    else:
                        retained_paths.add(str(alias))
                if retained_paths:
                    placeholders = ",".join("?" for _ in retained_paths)
                    self.connection.execute(
                        f"DELETE FROM source_aliases WHERE paper_id = ? "
                        f"AND source_path NOT IN ({placeholders})",
                        (paper.id, *sorted(retained_paths)),
                    )
                else:
                    self.connection.execute(
                        "DELETE FROM source_aliases WHERE paper_id = ?", (paper.id,)
                    )
            if paper.source_path:
                self._upsert_alias(
                    paper.id, paper.source_path, paper.source_kind, paper.content_hash, now
                )
            for alias in alias_items:
                if isinstance(alias, SourceDocument):
                    alias_path = str(alias.path)
                    alias_kind = alias.source_kind
                    alias_hash = alias.content_hash
                elif isinstance(alias, Mapping):
                    alias_path = str(alias["source_path"])
                    alias_kind = str(alias.get("source_kind", paper.source_kind))
                    alias_hash = str(alias.get("content_hash", paper.content_hash))
                else:
                    alias_path = str(alias)
                    alias_kind = paper.source_kind
                    alias_hash = paper.content_hash
                self._upsert_alias(paper.id, alias_path, alias_kind, alias_hash, now)
            if paper.source_path:
                self.connection.execute(
                    "DELETE FROM failures WHERE source_path = ?", (paper.source_path,)
                )
            if before_commit is not None:
                before_commit()
            self._bump_revision()

        paper.created_at = created_at
        paper.updated_at = now
        return paper.id

    def delete_paper(self, paper_id: str) -> bool:
        with self.transaction():
            self.connection.execute("DELETE FROM embeddings WHERE paper_id = ?", (paper_id,))
            cursor = self.connection.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
            if cursor.rowcount:
                self.connection.execute(
                    "DELETE FROM meta WHERE key = ?", (f"paper_embedding_state:{paper_id}",)
                )
                self._bump_revision()
        return bool(cursor.rowcount)

    def get_sections(self, paper_id: str) -> list[Section]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM sections WHERE paper_id = ? ORDER BY section_order, id",
                (paper_id,),
            ).fetchall()
        return [
            Section(
                id=int(row["id"]),
                paper_id=str(row["paper_id"]),
                parent_section_id=row["parent_section_id"],
                heading=str(row["heading"]),
                heading_path=str(row["heading_path"]),
                level=int(row["level"]),
                section_order=int(row["section_order"]),
                page_start=row["page_start"],
                page_end=row["page_end"],
                text=str(row["text"]),
            )
            for row in rows
        ]

    def get_section(self, paper_id: str, section_id: int) -> Section | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM sections WHERE paper_id = ? AND id = ?", (paper_id, section_id)
            ).fetchone()
        if row is None:
            return None
        return Section(
            id=int(row["id"]),
            paper_id=str(row["paper_id"]),
            parent_section_id=row["parent_section_id"],
            heading=str(row["heading"]),
            heading_path=str(row["heading_path"]),
            level=int(row["level"]),
            section_order=int(row["section_order"]),
            page_start=row["page_start"],
            page_end=row["page_end"],
            text=str(row["text"]),
        )

    def get_section_by_id(self, section_id: int) -> Section | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM sections WHERE id = ?", (section_id,)
            ).fetchone()
        if row is None:
            return None
        return Section(
            id=int(row["id"]),
            paper_id=str(row["paper_id"]),
            parent_section_id=row["parent_section_id"],
            heading=str(row["heading"]),
            heading_path=str(row["heading_path"]),
            level=int(row["level"]),
            section_order=int(row["section_order"]),
            page_start=row["page_start"],
            page_end=row["page_end"],
            text=str(row["text"]),
        )

    def get_chunks(self, paper_id: str, section_id: int | None = None) -> list[Chunk]:
        sql = "SELECT * FROM chunks WHERE paper_id = ?"
        parameters: list[Any] = [paper_id]
        if section_id is not None:
            sql += " AND section_id = ?"
            parameters.append(section_id)
        sql += " ORDER BY chunk_index, id"
        with self._lock:
            rows = self.connection.execute(sql, parameters).fetchall()
        return [
            Chunk(
                id=int(row["id"]),
                paper_id=str(row["paper_id"]),
                section_id=row["section_id"],
                chunk_index=int(row["chunk_index"]),
                heading_path=str(row["heading_path"]),
                page_start=row["page_start"],
                page_end=row["page_end"],
                text=str(row["text"]),
                token_count=int(row["token_count"]),
            )
            for row in rows
        ]

    def get_chunk_by_id(self, chunk_id: int) -> Chunk | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM chunks WHERE id = ?", (chunk_id,)
            ).fetchone()
        if row is None:
            return None
        return Chunk(
            id=int(row["id"]),
            paper_id=str(row["paper_id"]),
            section_id=row["section_id"],
            chunk_index=int(row["chunk_index"]),
            heading_path=str(row["heading_path"]),
            page_start=row["page_start"],
            page_end=row["page_end"],
            text=str(row["text"]),
            token_count=int(row["token_count"]),
        )

    def paper_outline(self, paper_id: str) -> dict[str, Any]:
        paper = self.get_paper(paper_id)
        if paper is None:
            raise KeyError(f"unknown paper: {paper_id}")
        sections = self.get_sections(paper_id)
        nodes = {
            section.id: {
                "section_id": section.id,
                "parent_section_id": section.parent_section_id,
                "heading": section.heading,
                "heading_path": section.heading_path,
                "level": section.level,
                "section_order": section.section_order,
                "page_start": section.page_start,
                "page_end": section.page_end,
                "children": [],
            }
            for section in sections
        }
        roots: list[dict[str, Any]] = []
        for section in sections:
            node = nodes[section.id]
            parent = nodes.get(section.parent_section_id)
            if parent is None:
                roots.append(node)
            else:
                parent["children"].append(node)
        return {
            "paper_id": paper.id,
            "title": paper.title,
            "authors": paper.authors,
            "year": paper.year,
            "doi": paper.doi,
            "abstract": paper.abstract,
            "section_tree": roots,
            "sections": [
                {key: value for key, value in nodes[section.id].items() if key != "children"}
                for section in sections
            ],
        }

    outline = paper_outline

    def read_section(
        self,
        paper_id: str,
        section_id: int,
        *,
        offset: int = 0,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        paper = self.get_paper(paper_id)
        section = self.get_section(paper_id, section_id)
        if paper is None or section is None:
            raise KeyError(f"unknown section {section_id} for paper {paper_id}")
        all_sections = self.get_sections(paper_id)
        included_ids: set[int] = {section_id}
        changed = True
        while changed:
            changed = False
            for candidate in all_sections:
                if (
                    candidate.id is not None
                    and candidate.parent_section_id in included_ids
                    and candidate.id not in included_ids
                ):
                    included_ids.add(candidate.id)
                    changed = True
        selected = [candidate for candidate in all_sections if candidate.id in included_ids]
        selected.sort(key=lambda candidate: candidate.section_order)
        rendered: list[str] = []
        for candidate in selected:
            if candidate.id == section_id:
                part = candidate.text.strip()
            else:
                prefix = "#" * max(1, min(6, candidate.level))
                part = f"{prefix} {candidate.heading}\n\n{candidate.text}".strip()
            if part:
                rendered.append(part)
        complete_text = "\n\n".join(rendered)
        matches = list(TOKEN_RE.finditer(complete_text))
        total_tokens = len(matches)
        if offset >= total_tokens:
            text = ""
            end_offset = total_tokens
        else:
            end_offset = (
                total_tokens if max_tokens is None else min(total_tokens, offset + max_tokens)
            )
            start_char = matches[offset].start()
            end_char = (
                len(complete_text) if end_offset == total_tokens else matches[end_offset].start()
            )
            text = complete_text[start_char:end_char].rstrip()
        pages = [
            page
            for candidate in selected
            for page in (candidate.page_start, candidate.page_end)
            if page is not None
        ]
        return {
            "paper_id": paper.id,
            "title": paper.title,
            "section_id": section.id,
            "section": section.heading_path,
            "heading": section.heading,
            "included_section_ids": [
                candidate.id for candidate in selected if candidate.id is not None
            ],
            "page_start": min(pages) if pages else section.page_start,
            "page_end": max(pages) if pages else section.page_end,
            "offset": offset,
            "next_offset": end_offset if end_offset < total_tokens else None,
            "total_tokens": total_tokens,
            "returned_tokens": max(0, end_offset - min(offset, total_tokens)),
            "truncated": end_offset < total_tokens,
            "text": text,
        }

    def replace_curated_entries(
        self,
        entries: Sequence[CuratedEntry],
        *,
        before_commit: Callable[[], None] | None = None,
    ) -> None:
        with self.transaction():
            self.connection.execute("DELETE FROM embeddings WHERE kind = 'curated'")
            self.connection.execute("DELETE FROM curated_entries")
            for entry in entries:
                cursor = self.connection.execute(
                    "INSERT INTO curated_entries(heading_path, text, source_line, entry_type) "
                    "VALUES (?, ?, ?, ?)",
                    (entry.heading_path, entry.text, entry.source_line, entry.entry_type),
                )
                entry.id = int(cursor.lastrowid)
                for paper_id in dict.fromkeys(entry.linked_paper_ids):
                    exists = self.connection.execute(
                        "SELECT 1 FROM papers WHERE id = ?", (paper_id,)
                    ).fetchone()
                    if exists is not None:
                        self.connection.execute(
                            "INSERT INTO curated_entry_papers(entry_id, paper_id) VALUES (?, ?)",
                            (entry.id, paper_id),
                        )
            if before_commit is not None:
                before_commit()
            self._bump_revision()

    def get_curated_entries(self) -> list[CuratedEntry]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM curated_entries ORDER BY COALESCE(source_line, 2147483647), id"
            ).fetchall()
            links = self.connection.execute(
                "SELECT entry_id, paper_id FROM curated_entry_papers ORDER BY entry_id, paper_id"
            ).fetchall()
        by_entry: dict[int, list[str]] = {}
        for link in links:
            by_entry.setdefault(int(link["entry_id"]), []).append(str(link["paper_id"]))
        return [
            CuratedEntry(
                id=int(row["id"]),
                heading_path=str(row["heading_path"]),
                text=str(row["text"]),
                source_line=row["source_line"],
                entry_type=str(row["entry_type"]),
                linked_paper_ids=by_entry.get(int(row["id"]), []),
            )
            for row in rows
        ]

    def curated_links(self, entry_id: int) -> list[str]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT paper_id FROM curated_entry_papers WHERE entry_id = ? ORDER BY paper_id",
                (entry_id,),
            ).fetchall()
        return [str(row[0]) for row in rows]

    def record_failure(
        self,
        source_path: str | Path,
        stage: str,
        error: str,
        *,
        content_hash: str | None = None,
    ) -> None:
        now = _utc_now()
        with self.transaction():
            self.connection.execute(
                "INSERT INTO failures("
                "source_path, content_hash, stage, error, created_at, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(source_path, stage) DO UPDATE SET "
                "content_hash = excluded.content_hash, error = excluded.error, "
                "updated_at = excluded.updated_at",
                (str(source_path), content_hash, stage, error, now, now),
            )
            self._bump_revision()

    def clear_failures(self, source_path: str | Path | None = None) -> int:
        with self.transaction():
            if source_path is None:
                cursor = self.connection.execute("DELETE FROM failures")
            else:
                cursor = self.connection.execute(
                    "DELETE FROM failures WHERE source_path = ?", (str(source_path),)
                )
            if cursor.rowcount:
                self._bump_revision()
        return int(cursor.rowcount)

    def list_failures(self) -> list[IngestFailure]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT source_path, stage, error FROM failures ORDER BY updated_at DESC, id DESC"
            ).fetchall()
        return [IngestFailure(**dict(row)) for row in rows]

    def _resolve_embedding_paper_id(self, kind: str, entity_id: str) -> str | None:
        if kind == "paper":
            return entity_id
        if kind in {"section", "chunk"}:
            table = "sections" if kind == "section" else "chunks"
            row = self.connection.execute(
                f"SELECT paper_id FROM {table} WHERE id = ?", (int(entity_id),)
            ).fetchone()
            return str(row[0]) if row is not None else None
        if kind == "curated":
            row = self.connection.execute(
                "SELECT paper_id FROM curated_entry_papers WHERE entry_id = ? ORDER BY paper_id LIMIT 1",
                (int(entity_id),),
            ).fetchone()
            return str(row[0]) if row is not None else None
        return None

    def store_embedding(
        self,
        kind: str,
        entity_id: str | int,
        vector: Vector,
        *,
        model_name: str,
        model_fingerprint: str,
        paper_id: str | None = None,
    ) -> None:
        self.store_embeddings(
            [(kind, entity_id, paper_id, vector)],
            model_name=model_name,
            model_fingerprint=model_fingerprint,
        )

    put_embedding = store_embedding

    def store_embeddings(
        self,
        embeddings: Iterable[tuple[str, str | int, str | None, Vector]],
        *,
        model_name: str,
        model_fingerprint: str,
    ) -> int:
        prepared = list(embeddings)
        if not prepared:
            return 0
        encoded = []
        dimensions_seen: set[int] = set()
        for kind, entity_id_value, paper_id, vector in prepared:
            blob, dimensions = encode_float32(vector, normalized=True)
            encoded.append((kind, entity_id_value, paper_id, blob, dimensions))
            dimensions_seen.add(dimensions)
        if len(dimensions_seen) != 1:
            raise ValueError("one embedding fingerprint cannot contain mixed dimensions")
        dimensions = next(iter(dimensions_seen))
        now = _utc_now()
        with self.transaction():
            existing = self.connection.execute(
                "SELECT DISTINCT model_name, dimensions FROM embeddings "
                "WHERE model_fingerprint = ?",
                (model_fingerprint,),
            ).fetchall()
            if any(
                row["model_name"] != model_name or int(row["dimensions"]) != dimensions
                for row in existing
            ):
                raise ValueError(
                    "embedding fingerprint is already associated with another model or dimension"
                )
            for kind, entity_id_value, paper_id, blob, row_dimensions in encoded:
                if kind not in ENTITY_KINDS:
                    raise ValueError(f"unsupported embedding kind: {kind}")
                entity_id = str(entity_id_value)
                resolved_paper_id = paper_id or self._resolve_embedding_paper_id(kind, entity_id)
                self.connection.execute(
                    "INSERT INTO embeddings("
                    "kind, entity_id, paper_id, model_name, model_fingerprint, dimensions, "
                    "vector, normalized, created_at, updated_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?) "
                    "ON CONFLICT(kind, entity_id, model_fingerprint) DO UPDATE SET "
                    "paper_id = excluded.paper_id, model_name = excluded.model_name, "
                    "dimensions = excluded.dimensions, vector = excluded.vector, "
                    "normalized = 1, updated_at = excluded.updated_at",
                    (
                        kind,
                        entity_id,
                        resolved_paper_id,
                        model_name,
                        model_fingerprint,
                        row_dimensions,
                        blob,
                        now,
                        now,
                    ),
                )
            self._bump_revision()
        return len(prepared)

    def replace_curated_embeddings(
        self,
        embeddings: Iterable[tuple[str, str | int, str | None, Vector]],
        *,
        model_name: str,
        model_fingerprint: str,
    ) -> int:
        prepared = list(embeddings)
        with self.transaction():
            self.connection.execute("DELETE FROM embeddings WHERE kind = 'curated'")
            if not prepared:
                self._bump_revision()
                return 0
            return self.store_embeddings(
                prepared,
                model_name=model_name,
                model_fingerprint=model_fingerprint,
            )

    def repair_embedding_provenance(self) -> int:
        """Repair denormalized paper IDs without recomputing valid vectors.

        Entity ownership is authoritative in the hierarchy and curated-link
        tables.  This also migrates databases written by versions that copied an
        unresolved catalog shorthand into a curated embedding row.
        """

        statements = (
            (
                "UPDATE embeddings SET paper_id = entity_id "
                "WHERE kind = 'paper' AND paper_id IS NOT entity_id",
                (),
            ),
            (
                "UPDATE embeddings AS e SET paper_id = ("
                "SELECT s.paper_id FROM sections s "
                "WHERE CAST(s.id AS TEXT) = e.entity_id) "
                "WHERE e.kind = 'section' AND EXISTS ("
                "SELECT 1 FROM sections s WHERE CAST(s.id AS TEXT) = e.entity_id) "
                "AND e.paper_id IS NOT (SELECT s.paper_id FROM sections s "
                "WHERE CAST(s.id AS TEXT) = e.entity_id)",
                (),
            ),
            (
                "UPDATE embeddings AS e SET paper_id = ("
                "SELECT c.paper_id FROM chunks c "
                "WHERE CAST(c.id AS TEXT) = e.entity_id) "
                "WHERE e.kind = 'chunk' AND EXISTS ("
                "SELECT 1 FROM chunks c WHERE CAST(c.id AS TEXT) = e.entity_id) "
                "AND e.paper_id IS NOT (SELECT c.paper_id FROM chunks c "
                "WHERE CAST(c.id AS TEXT) = e.entity_id)",
                (),
            ),
            (
                "UPDATE embeddings AS e SET paper_id = ("
                "SELECT MIN(cep.paper_id) FROM curated_entry_papers cep "
                "WHERE cep.entry_id = CAST(e.entity_id AS INTEGER)) "
                "WHERE e.kind = 'curated' AND EXISTS ("
                "SELECT 1 FROM curated_entries ce "
                "WHERE CAST(ce.id AS TEXT) = e.entity_id) "
                "AND e.paper_id IS NOT (SELECT MIN(cep.paper_id) "
                "FROM curated_entry_papers cep "
                "WHERE cep.entry_id = CAST(e.entity_id AS INTEGER))",
                (),
            ),
        )
        repaired = 0
        with self.transaction():
            for sql, parameters in statements:
                repaired += max(0, int(self.connection.execute(sql, parameters).rowcount))
            if repaired:
                self._bump_revision()
        return repaired

    def remove_empty_chunks(self) -> int:
        """Remove marker-only retrieval rows and compact per-paper chunk indices."""

        with self.transaction():
            empty_rows = self.connection.execute(
                "SELECT id, paper_id FROM chunks WHERE token_count = 0 ORDER BY id"
            ).fetchall()
            if not empty_rows:
                return 0
            empty_ids = [str(row["id"]) for row in empty_rows]
            placeholders = ",".join("?" for _ in empty_ids)
            self.connection.execute(
                f"DELETE FROM embeddings WHERE kind = 'chunk' "
                f"AND entity_id IN ({placeholders})",
                empty_ids,
            )
            self.connection.execute(
                f"DELETE FROM chunks WHERE id IN ({placeholders})",
                [int(value) for value in empty_ids],
            )
            for paper_id in sorted({str(row["paper_id"]) for row in empty_rows}):
                remaining = self.connection.execute(
                    "SELECT id, chunk_index FROM chunks WHERE paper_id = ? "
                    "ORDER BY chunk_index, id",
                    (paper_id,),
                ).fetchall()
                for chunk_index, row in enumerate(remaining):
                    if int(row["chunk_index"]) != chunk_index:
                        self.connection.execute(
                            "UPDATE chunks SET chunk_index = ? WHERE id = ?",
                            (chunk_index, int(row["id"])),
                        )
            self._bump_revision()
        return len(empty_rows)

    def replace_chunks_for_paper(
        self,
        paper_id: str,
        chunks: Sequence[Chunk],
        *,
        before_commit: Callable[[], None] | None = None,
    ) -> int:
        """Atomically replace derived chunks while preserving paper and section IDs."""

        if not chunks:
            raise ValueError("replacement chunks must not be empty")
        if any(chunk.paper_id != paper_id for chunk in chunks):
            raise ValueError("all replacement chunks must belong to the selected paper")
        original_identity = [
            (chunk.id, chunk.section_id, chunk.chunk_index) for chunk in chunks
        ]
        try:
            with self.transaction():
                paper_exists = self.connection.execute(
                    "SELECT 1 FROM papers WHERE id = ?",
                    (paper_id,),
                ).fetchone()
                if paper_exists is None:
                    raise KeyError(f"unknown paper: {paper_id}")
                section_rows = self.connection.execute(
                    "SELECT id, section_order FROM sections WHERE paper_id = ? "
                    "ORDER BY section_order, id",
                    (paper_id,),
                ).fetchall()
                section_id_by_order = {
                    int(row["section_order"]): int(row["id"]) for row in section_rows
                }
                self.connection.execute(
                    "DELETE FROM embeddings WHERE paper_id = ? "
                    "AND kind IN ('paper', 'section', 'chunk')",
                    (paper_id,),
                )
                self.connection.execute(
                    "DELETE FROM meta WHERE key = ?",
                    (f"paper_embedding_state:{paper_id}",),
                )
                self.connection.execute("DELETE FROM chunks WHERE paper_id = ?", (paper_id,))
                for chunk_index, chunk in enumerate(chunks):
                    section_order = chunk.section_index
                    if section_order is None:
                        raise ValueError("replacement chunks must identify their section order")
                    section_id = section_id_by_order.get(section_order)
                    if section_id is None:
                        raise ValueError("replacement chunk refers to an unknown section order")
                    cursor = self.connection.execute(
                        "INSERT INTO chunks("
                        "paper_id, section_id, chunk_index, heading_path, page_start, page_end, "
                        "text, token_count"
                        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            paper_id,
                            section_id,
                            chunk_index,
                            chunk.heading_path,
                            chunk.page_start,
                            chunk.page_end,
                            chunk.text,
                            chunk.token_count,
                        ),
                    )
                    chunk.id = int(cursor.lastrowid)
                    chunk.section_id = section_id
                    chunk.chunk_index = chunk_index
                if before_commit is not None:
                    before_commit()
                self._bump_revision()
        except BaseException:
            for chunk, (chunk_id, section_id, chunk_index) in zip(
                chunks, original_identity, strict=True
            ):
                chunk.id = chunk_id
                chunk.section_id = section_id
                chunk.chunk_index = chunk_index
            raise
        return len(chunks)

    def replace_embeddings_for_paper(
        self,
        paper_id: str,
        embeddings: Iterable[tuple[str, str | int, Vector]],
        *,
        model_name: str,
        model_fingerprint: str,
    ) -> int:
        prepared = list(embeddings)
        encoded = []
        dimensions_seen: set[int] = set()
        for kind, entity_id_value, vector in prepared:
            blob, dimensions = encode_float32(vector, normalized=True)
            encoded.append((kind, entity_id_value, blob, dimensions))
            dimensions_seen.add(dimensions)
        if len(dimensions_seen) > 1:
            raise ValueError("one embedding fingerprint cannot contain mixed dimensions")
        now = _utc_now()
        with self.transaction():
            self.connection.execute(
                "DELETE FROM embeddings WHERE paper_id = ? "
                "AND kind IN ('paper', 'section', 'chunk')",
                (paper_id,),
            )
            if dimensions_seen:
                dimensions = next(iter(dimensions_seen))
                existing = self.connection.execute(
                    "SELECT DISTINCT model_name, dimensions FROM embeddings "
                    "WHERE model_fingerprint = ?",
                    (model_fingerprint,),
                ).fetchall()
                if any(
                    row["model_name"] != model_name or int(row["dimensions"]) != dimensions
                    for row in existing
                ):
                    raise ValueError(
                        "embedding fingerprint is already associated with another model or dimension"
                    )
            for kind, entity_id_value, blob, dimensions in encoded:
                if kind not in {"paper", "section", "chunk"}:
                    raise ValueError("paper embedding batches support paper, section, and chunk")
                self.connection.execute(
                    "INSERT INTO embeddings("
                    "kind, entity_id, paper_id, model_name, model_fingerprint, dimensions, "
                    "vector, normalized, created_at, updated_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
                    (
                        kind,
                        str(entity_id_value),
                        paper_id,
                        model_name,
                        model_fingerprint,
                        dimensions,
                        blob,
                        now,
                        now,
                    ),
                )
            self._bump_revision()
        return len(prepared)

    def get_embedding(
        self, kind: str, entity_id: str | int, model_fingerprint: str
    ) -> tuple[float, ...] | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT vector, dimensions FROM embeddings "
                "WHERE kind = ? AND entity_id = ? AND model_fingerprint = ?",
                (kind, str(entity_id), model_fingerprint),
            ).fetchone()
        return decode_float32(row["vector"], int(row["dimensions"])) if row is not None else None

    def get_embedding_for_model(
        self,
        kind: str,
        entity_id: str | int,
        model: str,
    ) -> tuple[float, ...] | None:
        """Load an entity vector by configured model name or exact fingerprint."""

        with self._lock:
            row = self.connection.execute(
                "SELECT vector, dimensions FROM embeddings "
                "WHERE kind = ? AND entity_id = ? "
                "AND (model_fingerprint = ? OR model_name = ?) "
                "ORDER BY updated_at DESC LIMIT 1",
                (kind, str(entity_id), model, model),
            ).fetchone()
        return decode_float32(row["vector"], int(row["dimensions"])) if row is not None else None

    def embedding_fingerprints(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT model_fingerprint, model_name, dimensions, COUNT(*) AS count "
                "FROM embeddings GROUP BY model_fingerprint, model_name, dimensions "
                "ORDER BY model_name, dimensions"
            ).fetchall()
        return [dict(row) for row in rows]

    def paper_embedding_fingerprints(self) -> list[dict[str, Any]]:
        """Return identities that have at least one paper-level vector.

        Paper rows distinguish an existing core corpus from stale or independently
        generated curated vectors.  Ingest uses this during crash recovery, when
        per-paper embedding-state metadata may not have been committed yet.
        """

        with self._lock:
            rows = self.connection.execute(
                "SELECT model_fingerprint, model_name, dimensions, COUNT(*) AS count "
                "FROM embeddings WHERE kind = 'paper' "
                "GROUP BY model_fingerprint, model_name, dimensions "
                "ORDER BY count DESC, model_name, dimensions"
            ).fetchall()
        return [dict(row) for row in rows]

    def embedding_fingerprint_consistent(self, model_fingerprint: str) -> bool:
        with self._lock:
            rows = self.connection.execute(
                "SELECT model_name, dimensions, normalized, COUNT(*) AS count "
                "FROM embeddings WHERE model_fingerprint = ? "
                "GROUP BY model_name, dimensions, normalized",
                (model_fingerprint,),
            ).fetchall()
        return bool(
            len(rows) == 1
            and int(rows[0]["dimensions"]) > 0
            and int(rows[0]["normalized"]) == 1
            and int(rows[0]["count"]) > 0
        )

    def paper_embeddings_complete(self, paper_id: str, model_fingerprint: str) -> bool:
        """Return whether one fingerprint covers the paper and every current child row."""

        with self._lock:
            expected = self.connection.execute(
                "SELECT "
                "(SELECT COUNT(*) FROM papers WHERE id = ?) AS papers, "
                "(SELECT COUNT(*) FROM sections WHERE paper_id = ?) AS sections, "
                "(SELECT COUNT(*) FROM chunks WHERE paper_id = ?) AS chunks",
                (paper_id, paper_id, paper_id),
            ).fetchone()
            actual = self.connection.execute(
                "SELECT "
                "SUM(CASE WHEN e.kind = 'paper' AND e.entity_id = ? THEN 1 ELSE 0 END) "
                "AS papers, "
                "SUM(CASE WHEN e.kind = 'section' AND s.id IS NOT NULL THEN 1 ELSE 0 END) "
                "AS sections, "
                "SUM(CASE WHEN e.kind = 'chunk' AND c.id IS NOT NULL THEN 1 ELSE 0 END) "
                "AS chunks "
                "FROM embeddings e "
                "LEFT JOIN sections s ON e.kind = 'section' "
                "AND e.entity_id = CAST(s.id AS TEXT) AND s.paper_id = e.paper_id "
                "LEFT JOIN chunks c ON e.kind = 'chunk' "
                "AND e.entity_id = CAST(c.id AS TEXT) AND c.paper_id = e.paper_id "
                "WHERE e.paper_id = ? AND e.model_fingerprint = ?",
                (paper_id, paper_id, model_fingerprint),
            ).fetchone()
        if (
            expected is None
            or int(expected["papers"]) != 1
            or int(expected["sections"]) < 1
            or int(expected["chunks"]) < 1
            or not self.paper_hierarchy_complete(paper_id)
            or not self.embedding_fingerprint_consistent(model_fingerprint)
        ):
            return False
        return all(
            int(actual[name] or 0) == int(expected[name])
            for name in ("papers", "sections", "chunks")
        )

    def paper_hierarchy_complete(self, paper_id: str) -> bool:
        with self._lock:
            row = self.connection.execute(
                "SELECT "
                "(SELECT COUNT(*) FROM sections WHERE paper_id = ?) AS sections, "
                "(SELECT COUNT(*) FROM chunks WHERE paper_id = ?) AS chunks, "
                "(SELECT COUNT(*) FROM chunks c LEFT JOIN sections s ON s.id = c.section_id "
                "WHERE c.paper_id = ? AND (s.id IS NULL OR s.paper_id != c.paper_id)) AS invalid",
                (paper_id, paper_id, paper_id),
            ).fetchone()
        structurally_valid = bool(
            row is not None
            and int(row["sections"]) > 0
            and int(row["chunks"]) > 0
            and int(row["invalid"]) == 0
        )
        if not structurally_valid:
            return False
        sections = self.get_sections(paper_id)
        parent_by_id = {
            section.id: section.parent_section_id for section in sections if section.id is not None
        }
        for section_id in parent_by_id:
            seen: set[int] = set()
            current: int | None = section_id
            while current is not None:
                if current in seen:
                    return False
                seen.add(current)
                parent = parent_by_id.get(current)
                if parent is not None and parent not in parent_by_id:
                    return False
                current = parent
        return True

    def corpus_hierarchy_complete(self) -> bool:
        paper_ids = [paper.id for paper in self.list_papers()]
        return bool(paper_ids) and all(
            self.paper_hierarchy_complete(paper_id) for paper_id in paper_ids
        )

    def fts_complete(self) -> bool:
        mirrors = (
            (
                "papers",
                "papers_fts",
                "SELECT CAST(id AS TEXT), title, authors_json, abstract FROM papers",
                "SELECT CAST(paper_id AS TEXT), title, authors, abstract FROM papers_fts",
            ),
            (
                "sections",
                "sections_fts",
                "SELECT CAST(s.id AS TEXT), s.paper_id, p.title, s.heading_path, s.text "
                "FROM sections s JOIN papers p ON p.id = s.paper_id",
                "SELECT CAST(section_id AS TEXT), paper_id, title, heading_path, text "
                "FROM sections_fts",
            ),
            (
                "chunks",
                "chunks_fts",
                "SELECT CAST(c.id AS TEXT), c.paper_id, p.title, c.heading_path, c.text "
                "FROM chunks c JOIN papers p ON p.id = c.paper_id",
                "SELECT CAST(chunk_id AS TEXT), paper_id, title, heading_path, text "
                "FROM chunks_fts",
            ),
            (
                "curated_entries",
                "curated_entries_fts",
                "SELECT CAST(id AS TEXT), heading_path, entry_type, text FROM curated_entries",
                "SELECT CAST(entry_id AS TEXT), heading_path, entry_type, text "
                "FROM curated_entries_fts",
            ),
        )
        with self._lock:
            for table, fts_table, canonical_query, mirror_query in mirrors:
                base_count = int(
                    self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                )
                fts_count = int(
                    self.connection.execute(f"SELECT COUNT(*) FROM {fts_table}").fetchone()[0]
                )
                if base_count != fts_count:
                    return False
                for left, right in (
                    (canonical_query, mirror_query),
                    (mirror_query, canonical_query),
                ):
                    difference = self.connection.execute(
                        f"SELECT 1 FROM ({left} EXCEPT {right}) LIMIT 1"
                    ).fetchone()
                    if difference is not None:
                        return False
        return True

    def corpus_embeddings_complete(self, model_fingerprint: str) -> bool:
        """Return whether one exact fingerprint covers every indexed paper hierarchy."""

        paper_ids = [paper.id for paper in self.list_papers()]
        return bool(paper_ids) and all(
            self.paper_embeddings_complete(paper_id, model_fingerprint) for paper_id in paper_ids
        )

    def curated_embeddings_complete(self, model_fingerprint: str) -> bool:
        """Return whether one fingerprint covers every current curated entry."""

        with self._lock:
            expected = self.connection.execute(
                "SELECT COUNT(*) AS count FROM curated_entries"
            ).fetchone()
            actual = self.connection.execute(
                "SELECT COUNT(*) AS count FROM embeddings e "
                "JOIN curated_entries ce ON e.kind = 'curated' "
                "AND e.entity_id = CAST(ce.id AS TEXT) "
                "WHERE e.model_fingerprint = ? "
                "AND e.paper_id IS ("
                "SELECT MIN(cep.paper_id) FROM curated_entry_papers cep "
                "WHERE cep.entry_id = ce.id)",
                (model_fingerprint,),
            ).fetchone()
        expected_count = int(expected["count"] or 0)
        actual_count = int(actual["count"] or 0)
        return actual_count == expected_count and (
            expected_count == 0 or self.embedding_fingerprint_consistent(model_fingerprint)
        )

    def lexical_search(
        self,
        kind: str,
        query: str,
        limit: int,
        *,
        paper_id: str | None = None,
        exact: bool = False,
    ) -> list[Candidate]:
        """Search one hierarchy level with FTS5/BM25."""

        return self.search_lexical(
            query,
            kinds=(kind,),
            limit=limit,
            paper_id=paper_id,
            exact=exact,
        )

    def search_lexical(
        self,
        query: str,
        *,
        kinds: Iterable[str] | str | None = None,
        limit: int = 60,
        paper_id: str | None = None,
        exact: bool = False,
    ) -> list[Candidate]:
        if limit <= 0:
            return []
        match_query = _fts_query(query, exact=exact)
        if match_query is None:
            return []
        selected = self._normalize_kinds(kinds)
        scored: list[Candidate] = []
        per_kind_limit = max(limit, min(1_000, limit * 2))
        with self._lock:
            for kind in selected:
                table, id_column = FTS_TABLES[kind]
                # Titles remain mirrored in child FTS tables for integrity checks,
                # but a title-only hit is not evidence for an arbitrary section or
                # chunk.  Search child-owned fields and let the paper index surface
                # title matches honestly as paper results.
                kind_match_query = (
                    f"{{heading_path text}}: ({match_query})"
                    if kind in {"section", "chunk"}
                    else match_query
                )
                parameters: list[Any] = [kind_match_query]
                if kind == "paper":
                    sql = (
                        f"SELECT {id_column} AS entity_id, bm25({table}) AS rank "
                        f"FROM {table} WHERE {table} MATCH ?"
                    )
                    if paper_id is not None:
                        sql += " AND paper_id = ?"
                        parameters.append(paper_id)
                elif kind in {"section", "chunk"}:
                    sql = (
                        f"SELECT {id_column} AS entity_id, bm25({table}) AS rank "
                        f"FROM {table} WHERE {table} MATCH ?"
                    )
                    if paper_id is not None:
                        sql += " AND paper_id = ?"
                        parameters.append(paper_id)
                else:
                    sql = (
                        "SELECT curated_entries_fts.entry_id AS entity_id, "
                        "bm25(curated_entries_fts) AS rank FROM curated_entries_fts "
                    )
                    if paper_id is not None:
                        sql += (
                            "JOIN curated_entry_papers cep ON "
                            "cep.entry_id = CAST(curated_entries_fts.entry_id AS INTEGER) "
                        )
                    sql += "WHERE curated_entries_fts MATCH ?"
                    if paper_id is not None:
                        sql += " AND cep.paper_id = ?"
                        parameters.append(paper_id)
                sql += " ORDER BY rank LIMIT ?"
                parameters.append(per_kind_limit)
                rows = self.connection.execute(sql, parameters).fetchall()
                for row in rows:
                    lexical_score = -float(row["rank"])
                    candidate = self._candidate_from_entity(
                        kind,
                        str(row["entity_id"]),
                        score=lexical_score,
                        paper_id_hint=paper_id,
                    )
                    if candidate is None:
                        continue
                    exact_text = (
                        f"{candidate.title}\n{candidate.text}"
                        if candidate.kind == "paper"
                        else f"{candidate.section_path}\n{candidate.text}"
                    )
                    if exact and query.casefold() in exact_text.casefold():
                        lexical_score += 1.0
                    candidate.score = lexical_score
                    candidate.lexical_score = lexical_score
                    scored.append(candidate)
        scored.sort(key=lambda item: (-item.score, item.key))
        return scored[:limit]

    def dense_search(
        self,
        kind: str,
        query_vector: Vector,
        limit: int,
        *,
        model: str | None = None,
        paper_id: str | None = None,
        use_numpy: bool | None = None,
    ) -> list[Candidate]:
        """Search one hierarchy level by exact cosine similarity.

        ``model`` may be either a stored model fingerprint or its configured
        model name. If omitted, the most recently written compatible set is used.
        """

        selected = self._normalize_kinds((kind,))
        with self._lock:
            sql = "SELECT model_fingerprint FROM embeddings WHERE kind = ?"
            parameters: list[Any] = [kind]
            if model is not None:
                sql += " AND (model_fingerprint = ? OR model_name = ?)"
                parameters.extend((model, model))
            if paper_id is not None:
                sql += " AND paper_id = ?"
                parameters.append(paper_id)
            sql += " ORDER BY updated_at DESC LIMIT 1"
            row = self.connection.execute(sql, parameters).fetchone()
        if row is None:
            return []
        return self.search_dense(
            query_vector,
            model_fingerprint=str(row["model_fingerprint"]),
            kinds=selected,
            limit=limit,
            paper_id=paper_id,
            use_numpy=use_numpy,
        )

    def search_dense(
        self,
        query_vector: Vector,
        *,
        model_fingerprint: str,
        kinds: Iterable[str] | str | None = None,
        limit: int = 60,
        paper_id: str | None = None,
        use_numpy: bool | None = None,
    ) -> list[Candidate]:
        if limit <= 0:
            return []
        selected = self._normalize_kinds(kinds)
        placeholders = ",".join("?" for _ in selected)
        parameters: list[Any] = [model_fingerprint, *selected]
        sql = (
            "SELECT kind, entity_id, vector, dimensions FROM embeddings "
            f"WHERE model_fingerprint = ? AND kind IN ({placeholders})"
        )
        if paper_id is not None:
            sql += " AND paper_id = ?"
            parameters.append(paper_id)
        with self._lock:
            rows = self.connection.execute(sql, parameters).fetchall()
        by_token: dict[str, tuple[str, str]] = {}
        packed: list[tuple[str, bytes, int]] = []
        for index, row in enumerate(rows):
            token = str(index)
            by_token[token] = (str(row["kind"]), str(row["entity_id"]))
            packed.append((token, bytes(row["vector"]), int(row["dimensions"])))
        ranked = cosine_top_k(
            query_vector,
            packed,
            top_k=min(len(packed), max(limit, limit * 2)),
            use_numpy=use_numpy,
        )
        candidates: list[Candidate] = []
        with self._lock:
            for token, dense_score in ranked:
                kind, entity_id = by_token[token]
                candidate = self._candidate_from_entity(
                    kind, entity_id, score=dense_score, paper_id_hint=paper_id
                )
                if candidate is None:
                    continue
                candidate.score = dense_score
                candidate.dense_score = dense_score
                candidates.append(candidate)
                if len(candidates) >= limit:
                    break
        return candidates

    def route_curated_candidates(
        self,
        candidates: Sequence[Candidate],
    ) -> tuple[list[Candidate], list[Candidate]]:
        """Turn linked expert-note hits into paper boosts and retain unresolved notes."""

        unresolved: list[Candidate] = []
        paper_boosts: list[Candidate] = []
        seen_papers: set[str] = set()
        with self._lock:
            for note in candidates:
                if note.kind != "curated":
                    continue
                rows = self.connection.execute(
                    "SELECT paper_id FROM curated_entry_papers "
                    "WHERE entry_id = ? ORDER BY paper_id",
                    (int(note.entity_id),),
                ).fetchall()
                if not rows:
                    unresolved.append(note)
                    continue
                for row in rows:
                    paper_id = str(row["paper_id"])
                    if paper_id in seen_papers:
                        continue
                    routed = self._candidate_from_entity(
                        "paper",
                        paper_id,
                        score=note.score,
                    )
                    if routed is None:
                        continue
                    routed.lexical_score = note.lexical_score
                    routed.dense_score = note.dense_score
                    routed.section_path = note.section_path
                    routed.routing_note = note.text.strip()
                    paper_boosts.append(routed)
                    seen_papers.add(paper_id)
        return unresolved, paper_boosts

    def shared_curated_paper_candidates(
        self,
        paper_id: str,
        limit: int,
    ) -> list[Candidate]:
        """Rank other papers directly linked from the same expert index entries."""

        if limit <= 0:
            return []
        with self._lock:
            rows = self.connection.execute(
                "SELECT other.paper_id, COUNT(*) AS shared_count "
                "FROM curated_entry_papers source "
                "JOIN curated_entry_papers other ON other.entry_id = source.entry_id "
                "WHERE source.paper_id = ? AND other.paper_id <> ? "
                "GROUP BY other.paper_id "
                "ORDER BY shared_count DESC, other.paper_id LIMIT ?",
                (paper_id, paper_id, limit),
            ).fetchall()
            candidates: list[Candidate] = []
            for row in rows:
                candidate = self._candidate_from_entity(
                    "paper",
                    str(row["paper_id"]),
                    score=float(row["shared_count"]),
                )
                if candidate is not None:
                    candidates.append(candidate)
        return candidates

    def _normalize_kinds(self, kinds: Iterable[str] | str | None) -> tuple[str, ...]:
        if kinds is None:
            return ("paper", "section", "chunk", "curated")
        values = (kinds,) if isinstance(kinds, str) else tuple(dict.fromkeys(kinds))
        invalid = set(values) - ENTITY_KINDS
        if invalid:
            raise ValueError(f"unsupported result kinds: {', '.join(sorted(invalid))}")
        return tuple(kind for kind in ("paper", "section", "chunk", "curated") if kind in values)

    def _candidate_from_entity(
        self,
        kind: str,
        entity_id: str,
        *,
        score: float,
        paper_id_hint: str | None = None,
    ) -> Candidate | None:
        if kind == "paper":
            row = self.connection.execute(
                "SELECT * FROM papers WHERE id = ?", (entity_id,)
            ).fetchone()
            if row is None:
                return None
            return Candidate(
                key=f"paper:{entity_id}",
                kind="paper",
                entity_id=entity_id,
                paper_id=entity_id,
                section_id=None,
                chunk_id=None,
                title=str(row["title"]),
                authors=_load_authors(row["authors_json"]),
                year=row["year"],
                section_path="",
                page_start=None,
                page_end=None,
                source_kind=str(row["source_kind"]),
                text=str(row["abstract"] or row["title"]),
                score=score,
            )
        if kind == "section":
            row = self.connection.execute(
                "SELECT s.*, p.title, p.authors_json, p.year, p.source_kind "
                "FROM sections s JOIN papers p ON p.id = s.paper_id WHERE s.id = ?",
                (int(entity_id),),
            ).fetchone()
            if row is None:
                return None
            return Candidate(
                key=f"section:{entity_id}",
                kind="section",
                entity_id=entity_id,
                paper_id=str(row["paper_id"]),
                section_id=int(row["id"]),
                chunk_id=None,
                title=str(row["title"]),
                authors=_load_authors(row["authors_json"]),
                year=row["year"],
                section_path=str(row["heading_path"]),
                page_start=row["page_start"],
                page_end=row["page_end"],
                source_kind=str(row["source_kind"]),
                text=str(row["text"]),
                score=score,
            )
        if kind == "chunk":
            row = self.connection.execute(
                "SELECT c.*, p.title, p.authors_json, p.year, p.source_kind "
                "FROM chunks c JOIN papers p ON p.id = c.paper_id WHERE c.id = ?",
                (int(entity_id),),
            ).fetchone()
            if row is None:
                return None
            return Candidate(
                key=f"chunk:{entity_id}",
                kind="chunk",
                entity_id=entity_id,
                paper_id=str(row["paper_id"]),
                section_id=row["section_id"],
                chunk_id=int(row["id"]),
                title=str(row["title"]),
                authors=_load_authors(row["authors_json"]),
                year=row["year"],
                section_path=str(row["heading_path"]),
                page_start=row["page_start"],
                page_end=row["page_end"],
                source_kind=str(row["source_kind"]),
                text=str(row["text"]),
                score=score,
            )
        if kind == "curated":
            if paper_id_hint is not None:
                row = self.connection.execute(
                    "SELECT ce.*, p.id AS linked_paper_id, p.title, p.authors_json, p.year, "
                    "p.source_kind FROM curated_entries ce "
                    "JOIN curated_entry_papers cp ON cp.entry_id = ce.id "
                    "JOIN papers p ON p.id = cp.paper_id "
                    "WHERE ce.id = ? AND cp.paper_id = ? LIMIT 1",
                    (int(entity_id), paper_id_hint),
                ).fetchone()
            else:
                row = self.connection.execute(
                    "SELECT ce.*, p.id AS linked_paper_id, p.title, p.authors_json, p.year, "
                    "p.source_kind FROM curated_entries ce "
                    "LEFT JOIN curated_entry_papers cp ON cp.entry_id = ce.id "
                    "LEFT JOIN papers p ON p.id = cp.paper_id "
                    "WHERE ce.id = ? ORDER BY p.id LIMIT 1",
                    (int(entity_id),),
                ).fetchone()
            if row is None:
                return None
            linked_paper_id = row["linked_paper_id"]
            return Candidate(
                key=f"curated:{entity_id}",
                kind="curated",
                entity_id=entity_id,
                paper_id=str(linked_paper_id) if linked_paper_id is not None else None,
                section_id=None,
                chunk_id=None,
                title=str(row["title"] or row["heading_path"] or "Curated index"),
                authors=_load_authors(row["authors_json"]),
                year=row["year"],
                section_path=str(row["heading_path"]),
                page_start=None,
                page_end=None,
                source_kind=str(row["source_kind"] or "curated_index"),
                text=str(row["text"]),
                score=score,
            )
        return None

    def rebuild_fts(self) -> None:
        with self.transaction():
            for table, _ in FTS_TABLES.values():
                self.connection.execute(f"DELETE FROM {table}")
            self.connection.execute(
                "INSERT INTO papers_fts(paper_id, title, authors, abstract) "
                "SELECT id, title, authors_json, abstract FROM papers"
            )
            self.connection.execute(
                "INSERT INTO sections_fts(rowid, section_id, paper_id, title, heading_path, text) "
                "SELECT s.id, s.id, s.paper_id, p.title, s.heading_path, s.text "
                "FROM sections s JOIN papers p ON p.id = s.paper_id"
            )
            self.connection.execute(
                "INSERT INTO chunks_fts(rowid, chunk_id, paper_id, title, heading_path, text) "
                "SELECT c.id, c.id, c.paper_id, p.title, c.heading_path, c.text "
                "FROM chunks c JOIN papers p ON p.id = c.paper_id"
            )
            self.connection.execute(
                "INSERT INTO curated_entries_fts(rowid, entry_id, heading_path, entry_type, text) "
                "SELECT id, id, heading_path, entry_type, text FROM curated_entries"
            )
            self._bump_revision()

    def counts(self) -> dict[str, int]:
        names = (
            "papers",
            "source_aliases",
            "sections",
            "chunks",
            "curated_entries",
            "embeddings",
            "failures",
        )
        with self._lock:
            return {
                name: int(self.connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
                for name in names
            }


Database = CorpusDatabase


__all__ = [
    "CORPUS_SNAPSHOT_COMPLETE_KEY",
    "CORPUS_SOURCE_MANIFEST_KEY",
    "CorpusDatabase",
    "Database",
    "ENTITY_KINDS",
    "FTS_TABLES",
    "SCHEMA_VERSION",
]
