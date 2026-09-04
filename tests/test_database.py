from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from papers_mcp.database import CorpusDatabase
from papers_mcp.models import Chunk, CuratedEntry, Paper, Section, SourceDocument


def sample_hierarchy(
    *,
    content_hash: str = "hash-v1",
    title: str = "Smooth Patch Networks",
    phrase: str = "vertex enclosure condition",
) -> tuple[Paper, list[Section], list[Chunk]]:
    paper = Paper(
        id="peters-2002-smooth-patches",
        title=title,
        authors=["Jane Peters", "A. Geometer"],
        year=2002,
        doi="10.example/patches",
        abstract=f"A construction based on the {phrase}.",
        source_path="papers/original/peters.pdf",
        markdown_path="papers/markdown/peters.md",
        content_hash=content_hash,
        extraction_backend="fixture",
    )
    sections = [
        Section(
            paper_id=paper.id,
            heading="Construction",
            heading_path="4 Construction",
            text=f"We introduce the {phrase} for a patch network.",
            level=1,
            section_order=0,
            page_start=6,
            page_end=8,
        ),
        Section(
            paper_id=paper.id,
            heading="Compatibility",
            heading_path="4 Construction > 4.2 Compatibility",
            text="one two three four five",
            level=2,
            section_order=1,
            page_start=7,
            page_end=8,
            parent_index=0,
        ),
    ]
    chunks = [
        Chunk(
            paper_id=paper.id,
            section_index=0,
            chunk_index=0,
            heading_path=sections[0].heading_path,
            page_start=6,
            page_end=7,
            text=f"The exact {phrase} couples cross-boundary derivatives.",
            token_count=8,
        ),
        Chunk(
            paper_id=paper.id,
            section_index=1,
            chunk_index=1,
            heading_path=sections[1].heading_path,
            page_start=7,
            page_end=8,
            text="Tangent compatibility is solved at extraordinary vertices.",
            token_count=7,
        ),
    ]
    return paper, sections, chunks


@pytest.fixture
def database(tmp_path: Path) -> CorpusDatabase:
    with CorpusDatabase(tmp_path / "papers.db") as value:
        yield value


def test_schema_pragmas_and_four_fts_tables(database: CorpusDatabase) -> None:
    assert database.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert database.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert database.connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5_000
    fts_tables = {
        row[0]
        for row in database.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN "
            "('papers_fts', 'sections_fts', 'chunks_fts', 'curated_entries_fts')"
        )
    }
    assert fts_tables == {
        "papers_fts",
        "sections_fts",
        "chunks_fts",
        "curated_entries_fts",
    }
    assert database.get_meta("schema_version") == "1"
    assert database.get_revision() == 0


def test_incompatible_schema_is_rejected_before_schema_mutation(tmp_path: Path) -> None:
    path = tmp_path / "future.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute("INSERT INTO meta(key, value) VALUES ('schema_version', '2')")
    connection.commit()
    connection.close()

    with pytest.raises(RuntimeError, match="incompatible"):
        CorpusDatabase(path)

    connection = sqlite3.connect(path)
    try:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    finally:
        connection.close()
    assert tables == {"meta"}


def test_transactional_replace_alias_hash_and_hierarchy_helpers(
    database: CorpusDatabase,
) -> None:
    paper, sections, chunks = sample_hierarchy()
    alias = SourceDocument(
        path=Path("legacy/peters-copy.pdf"),
        source_kind="pdf",
        content_hash="hash-v1",
    )
    database.replace_paper(paper, sections, chunks, aliases=[alias])

    assert database.counts() == {
        "papers": 1,
        "source_aliases": 2,
        "sections": 2,
        "chunks": 2,
        "curated_entries": 0,
        "embeddings": 0,
        "failures": 0,
    }
    assert database.is_unchanged(paper.source_path, "hash-v1")
    assert database.is_unchanged(alias.path, "hash-v1")
    assert not database.is_unchanged(alias.path, "changed")
    assert database.find_paper_by_hash("hash-v1").id == paper.id  # type: ignore[union-attr]
    assert database.find_paper_by_source(alias.path).id == paper.id  # type: ignore[union-attr]

    assert sections[0].id is not None
    assert sections[1].parent_section_id == sections[0].id
    assert chunks[0].section_id == sections[0].id
    assert database.get_section_by_id(sections[1].id).heading == "Compatibility"  # type: ignore[arg-type,union-attr]
    assert database.get_chunk_by_id(chunks[0].id).paper_id == paper.id  # type: ignore[arg-type,union-attr]

    outline = database.paper_outline(paper.id)
    assert outline["title"] == paper.title
    assert outline["section_tree"][0]["children"][0]["heading"] == "Compatibility"
    page = database.read_section(paper.id, sections[1].id, offset=1, max_tokens=2)  # type: ignore[arg-type]
    assert page["text"] == "two three"
    assert page["next_offset"] == 3
    assert page["truncated"] is True
    parent = database.read_section(paper.id, sections[0].id)  # type: ignore[arg-type]
    assert "one two three four five" in parent["text"]
    assert parent["included_section_ids"] == [sections[0].id, sections[1].id]


def test_lexical_candidates_have_traceability_and_paper_filter(
    database: CorpusDatabase,
) -> None:
    paper, sections, chunks = sample_hierarchy()
    database.replace_paper(paper, sections, chunks)

    results = database.lexical_search("chunk", "vertex enclosure", 10, exact=True)
    assert len(results) == 1
    result = results[0]
    assert result.kind == "chunk"
    assert result.paper_id == paper.id
    assert result.section_id == sections[0].id
    assert result.chunk_id == chunks[0].id
    assert result.title == paper.title
    assert result.authors == paper.authors
    assert result.page_start == 6
    assert result.lexical_score == result.score
    assert result.score > 1.0

    assert database.lexical_search("chunk", "vertex enclosure", 10, paper_id="not-this-paper") == []
    assert database.lexical_search("chunk", "symbols ???", 0) == []


def test_title_only_lexical_hit_is_not_attributed_to_arbitrary_children(
    database: CorpusDatabase,
) -> None:
    paper, sections, chunks = sample_hierarchy(
        title="PEARL: A Global Approach",
        phrase="arrangement cell labeling",
    )
    database.replace_paper(paper, sections, chunks)

    assert database.lexical_search("paper", "PEARL", 10, exact=True)
    assert database.lexical_search("section", "PEARL", 10, exact=True) == []
    assert database.lexical_search("chunk", "PEARL", 10, exact=True) == []
    assert database.paper_embedding_fingerprints() == []


def test_replacement_removes_old_rows_embeddings_and_fts_ghosts(
    database: CorpusDatabase,
) -> None:
    paper, sections, chunks = sample_hierarchy()
    database.replace_paper(paper, sections, chunks)
    database.store_embedding(
        "chunk",
        chunks[0].id,
        [1.0, 0.0],
        model_name="toy-model",
        model_fingerprint="toy-v1",
    )
    old_chunk_id = chunks[0].id
    assert database.lexical_search("chunk", "vertex enclosure", 10)

    replacement, new_sections, new_chunks = sample_hierarchy(
        content_hash="hash-v2",
        title="Watertight Cell Selection",
        phrase="arrangement cell labeling",
    )
    database.replace_paper(replacement, new_sections, new_chunks)

    assert database.counts()["papers"] == 1
    assert database.counts()["sections"] == 2
    assert database.counts()["chunks"] == 2
    assert database.counts()["embeddings"] == 0
    assert database.get_chunk_by_id(old_chunk_id) is None  # type: ignore[arg-type]
    assert database.lexical_search("paper", "vertex enclosure", 10) == []
    assert database.lexical_search("section", "vertex enclosure", 10) == []
    assert database.lexical_search("chunk", "vertex enclosure", 10) == []
    assert database.lexical_search("chunk", "arrangement cell labeling", 10, exact=True)

    assert database.connection.execute("SELECT COUNT(*) FROM papers_fts").fetchone()[0] == 1
    assert database.connection.execute("SELECT COUNT(*) FROM sections_fts").fetchone()[0] == 2
    assert database.connection.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == 2


def test_failed_replace_rolls_back_without_damaging_existing_paper(
    database: CorpusDatabase,
) -> None:
    paper, sections, chunks = sample_hierarchy()
    database.replace_paper(paper, sections, chunks)
    revision = database.get_revision()
    bad_paper, bad_sections, bad_chunks = sample_hierarchy(content_hash="bad")
    bad_sections[1].parent_index = 99

    with pytest.raises(ValueError, match="out of range"):
        database.replace_paper(bad_paper, bad_sections, bad_chunks)

    assert database.find_paper_by_hash("hash-v1") is not None
    assert database.find_paper_by_hash("bad") is None
    assert database.get_revision() == revision


def test_curated_entries_links_failures_and_fts(database: CorpusDatabase) -> None:
    paper, sections, chunks = sample_hierarchy()
    database.replace_paper(paper, sections, chunks)
    entries = [
        CuratedEntry(
            heading_path="Patch Networks > Continuity",
            text="Downstream continuity feasibility is especially important.",
            linked_paper_ids=[paper.id, "unresolved-paper"],
            source_line=42,
        )
    ]
    database.replace_curated_entries(entries)
    assert entries[0].id is not None
    assert database.curated_links(entries[0].id) == [paper.id]
    assert database.get_curated_entries()[0].linked_paper_ids == [paper.id]

    candidate = database.lexical_search("curated", "continuity feasibility", 5, paper_id=paper.id)[
        0
    ]
    assert candidate.paper_id == paper.id
    assert candidate.kind == "curated"
    assert (
        database.lexical_search("curated", "continuity feasibility", 5, paper_id="unresolved-paper")
        == []
    )
    unresolved, paper_boosts = database.route_curated_candidates([candidate])
    assert unresolved == []
    assert [item.paper_id for item in paper_boosts] == [paper.id]
    assert paper_boosts[0].key == f"paper:{paper.id}"
    assert paper_boosts[0].routing_note == entries[0].text
    assert entries[0].text not in paper_boosts[0].text

    database.record_failure("broken.pdf", "extract", "first error", content_hash="bad")
    database.record_failure("broken.pdf", "extract", "updated error", content_hash="bad")
    failures = database.list_failures()
    assert len(failures) == 1
    assert failures[0].error == "updated error"
    assert database.clear_failures("broken.pdf") == 1


def test_shared_curated_entries_rank_linked_papers(database: CorpusDatabase) -> None:
    first, sections, chunks = sample_hierarchy()
    database.replace_paper(first, sections, chunks)
    second, second_sections, second_chunks = sample_hierarchy(
        content_hash="second-hash",
        title="Alternative Patch Construction",
    )
    second.id = "alternative-patch-construction"
    second.source_path = "papers/original/alternative.pdf"
    for section in second_sections:
        section.paper_id = second.id
    for chunk in second_chunks:
        chunk.paper_id = second.id
    database.replace_paper(second, second_sections, second_chunks)
    database.replace_curated_entries(
        [
            CuratedEntry(
                "Patch Networks",
                "Both papers address the same compatibility route.",
                linked_paper_ids=[first.id, second.id],
            )
        ]
    )

    related = database.shared_curated_paper_candidates(first.id, 5)

    assert [candidate.paper_id for candidate in related] == [second.id]


@pytest.mark.parametrize("use_numpy", [True, False])
def test_dense_search_returns_candidates_for_model_name_or_fingerprint(
    database: CorpusDatabase, use_numpy: bool
) -> None:
    paper, sections, chunks = sample_hierarchy()
    database.replace_paper(paper, sections, chunks)
    database.store_embeddings(
        [
            ("chunk", chunks[0].id, paper.id, [1.0, 0.0, 0.0]),
            ("chunk", chunks[1].id, paper.id, [0.0, 1.0, 0.0]),
        ],
        model_name="toy-model",
        model_fingerprint="toy-v1",
    )

    by_name = database.dense_search(
        "chunk", [0.9, 0.1, 0.0], 2, model="toy-model", use_numpy=use_numpy
    )
    by_fingerprint = database.dense_search(
        "chunk", [0.9, 0.1, 0.0], 2, model="toy-v1", use_numpy=use_numpy
    )
    automatic = database.dense_search("chunk", [0.9, 0.1, 0.0], 2, use_numpy=use_numpy)

    for results in (by_name, by_fingerprint, automatic):
        assert [item.chunk_id for item in results] == [chunks[0].id, chunks[1].id]
        assert all(item.dense_score == item.score for item in results)
        assert all(item.paper_id == paper.id for item in results)
    assert database.dense_search("chunk", [1.0, 0.0, 0.0], 2, model="missing-model") == []


def test_complete_embedding_check_requires_every_current_hierarchy_row(
    database: CorpusDatabase,
) -> None:
    paper, sections, chunks = sample_hierarchy()
    database.replace_paper(paper, sections, chunks)
    database.store_embeddings(
        [
            ("paper", paper.id, paper.id, [1.0, 0.0]),
            *[("section", section.id, paper.id, [1.0, 0.0]) for section in sections],
            *[("chunk", chunk.id, paper.id, [1.0, 0.0]) for chunk in chunks],
        ],
        model_name="toy",
        model_fingerprint="toy-v1",
    )

    assert database.paper_embeddings_complete(paper.id, "toy-v1")
    database.connection.execute(
        "DELETE FROM embeddings WHERE kind = 'chunk' AND entity_id = ?",
        (str(chunks[0].id),),
    )
    assert not database.paper_embeddings_complete(paper.id, "toy-v1")


def test_embedding_batches_reject_mixed_dimensions_without_deleting_prior_vectors(
    database: CorpusDatabase,
) -> None:
    paper, sections, chunks = sample_hierarchy()
    database.replace_paper(paper, sections, chunks)
    valid = [
        ("paper", paper.id, [1.0, 0.0]),
        *[("section", section.id, [1.0, 0.0]) for section in sections],
        *[("chunk", chunk.id, [1.0, 0.0]) for chunk in chunks],
    ]
    database.replace_embeddings_for_paper(
        paper.id,
        valid,
        model_name="toy",
        model_fingerprint="toy-v1",
    )

    with pytest.raises(ValueError, match="mixed dimensions"):
        database.replace_embeddings_for_paper(
            paper.id,
            [
                ("paper", paper.id, [1.0, 0.0]),
                ("section", sections[0].id, [1.0, 0.0, 0.0]),
            ],
            model_name="toy",
            model_fingerprint="toy-v1",
        )

    assert database.paper_embeddings_complete(paper.id, "toy-v1")


def test_embedding_completeness_rejects_one_fingerprint_with_mixed_model_identity(
    database: CorpusDatabase,
) -> None:
    paper, sections, chunks = sample_hierarchy()
    database.replace_paper(paper, sections, chunks)
    database.replace_embeddings_for_paper(
        paper.id,
        [
            ("paper", paper.id, [1.0, 0.0]),
            *[("section", section.id, [1.0, 0.0]) for section in sections],
            *[("chunk", chunk.id, [1.0, 0.0]) for chunk in chunks],
        ],
        model_name="toy",
        model_fingerprint="toy-v1",
    )
    database.connection.execute(
        "UPDATE embeddings SET model_name = 'different-model' "
        "WHERE kind = 'chunk' AND entity_id = ?",
        (str(chunks[0].id),),
    )

    assert not database.embedding_fingerprint_consistent("toy-v1")
    assert not database.paper_embeddings_complete(paper.id, "toy-v1")


def test_hierarchy_completeness_rejects_parent_cycles(database: CorpusDatabase) -> None:
    paper, sections, chunks = sample_hierarchy()
    database.replace_paper(paper, sections, chunks)
    database.connection.execute(
        "UPDATE sections SET parent_section_id = id WHERE id = ?",
        (sections[0].id,),
    )

    assert not database.paper_hierarchy_complete(paper.id)
    assert not database.corpus_hierarchy_complete()


def test_fts_completeness_compares_ids_and_mirrored_content(
    database: CorpusDatabase,
) -> None:
    paper, sections, chunks = sample_hierarchy()
    database.replace_paper(paper, sections, chunks)
    database.replace_curated_entries([CuratedEntry("Topic", "Expert note")])
    assert database.fts_complete()

    database.connection.execute(
        "UPDATE chunks_fts SET text = 'corrupt mirror' WHERE rowid = ?",
        (chunks[0].id,),
    )
    assert not database.fts_complete()


def test_complete_curated_embedding_check_requires_every_current_entry(
    database: CorpusDatabase,
) -> None:
    entries = [
        CuratedEntry("Topic", "First expert note"),
        CuratedEntry("Topic", "Second expert note"),
    ]
    database.replace_curated_entries(entries)
    database.store_embeddings(
        [("curated", entry.id, None, [1.0, 0.0]) for entry in entries if entry.id is not None],
        model_name="toy",
        model_fingerprint="toy-v1",
    )

    assert database.curated_embeddings_complete("toy-v1")
    database.connection.execute(
        "UPDATE embeddings SET paper_id = 'unresolved-catalog-id' "
        "WHERE kind = 'curated' AND entity_id = ?",
        (str(entries[0].id),),
    )
    assert not database.curated_embeddings_complete("toy-v1")
    database.connection.execute(
        "UPDATE embeddings SET paper_id = NULL "
        "WHERE kind = 'curated' AND entity_id = ?",
        (str(entries[0].id),),
    )
    assert database.curated_embeddings_complete("toy-v1")
    database.connection.execute(
        "DELETE FROM embeddings WHERE kind = 'curated' AND entity_id = ?",
        (str(entries[0].id),),
    )
    assert not database.curated_embeddings_complete("toy-v1")


def test_embedding_provenance_repair_uses_authoritative_entity_owners(
    database: CorpusDatabase,
) -> None:
    paper, sections, chunks = sample_hierarchy()
    database.replace_paper(paper, sections, chunks)
    linked = CuratedEntry("Topic", "Linked note", linked_paper_ids=[paper.id])
    unresolved = CuratedEntry(
        "Topic", "Unresolved note", linked_paper_ids=["catalog-shorthand"]
    )
    database.replace_curated_entries([linked, unresolved])
    database.store_embeddings(
        [
            ("paper", paper.id, "wrong-paper", [1.0, 0.0]),
            ("section", sections[0].id, "wrong-paper", [1.0, 0.0]),
            ("chunk", chunks[0].id, "wrong-paper", [1.0, 0.0]),
            ("curated", linked.id, "catalog-shorthand", [1.0, 0.0]),
            ("curated", unresolved.id, "catalog-shorthand", [1.0, 0.0]),
        ],
        model_name="toy",
        model_fingerprint="toy-v1",
    )

    assert database.repair_embedding_provenance() == 5
    rows = database.connection.execute(
        "SELECT kind, entity_id, paper_id FROM embeddings ORDER BY kind, entity_id"
    ).fetchall()
    by_entity = {(row["kind"], row["entity_id"]): row["paper_id"] for row in rows}
    assert by_entity[("paper", paper.id)] == paper.id
    assert by_entity[("section", str(sections[0].id))] == paper.id
    assert by_entity[("chunk", str(chunks[0].id))] == paper.id
    assert by_entity[("curated", str(linked.id))] == paper.id
    assert by_entity[("curated", str(unresolved.id))] is None
    assert database.repair_embedding_provenance() == 0


def test_remove_empty_chunks_deletes_vectors_and_compacts_indices(
    database: CorpusDatabase,
) -> None:
    paper, sections, chunks = sample_hierarchy()
    chunks[0].text = "<!-- page: 1 -->"
    chunks[0].token_count = 0
    database.replace_paper(paper, sections, chunks)
    database.store_embeddings(
        [
            ("chunk", chunks[0].id, paper.id, [1.0, 0.0]),
            ("chunk", chunks[1].id, paper.id, [0.0, 1.0]),
        ],
        model_name="toy",
        model_fingerprint="toy-v1",
    )

    assert database.remove_empty_chunks() == 1
    remaining = database.get_chunks(paper.id)
    assert len(remaining) == 1
    assert remaining[0].id == chunks[1].id
    assert remaining[0].chunk_index == 0
    assert database.connection.execute(
        "SELECT COUNT(*) FROM embeddings WHERE kind = 'chunk'"
    ).fetchone()[0] == 1
    assert database.fts_complete()
    assert database.remove_empty_chunks() == 0


def test_replace_chunks_preserves_sections_and_rebuilds_fts_atomically(
    database: CorpusDatabase,
) -> None:
    paper, sections, chunks = sample_hierarchy()
    database.replace_paper(paper, sections, chunks)
    section_ids = [section.id for section in database.get_sections(paper.id)]
    replacement = [
        Chunk(
            paper_id=paper.id,
            section_index=0,
            chunk_index=99,
            heading_path=sections[0].heading_path,
            page_start=6,
            page_end=8,
            text="Freshly bounded rechunked evidence.",
            token_count=5,
        )
    ]

    assert database.replace_chunks_for_paper(paper.id, replacement) == 1
    stored = database.get_chunks(paper.id)
    assert [section.id for section in database.get_sections(paper.id)] == section_ids
    assert len(stored) == 1
    assert stored[0].chunk_index == 0
    assert stored[0].section_id == section_ids[0]
    assert database.lexical_search("chunk", "freshly bounded", 5)[0].chunk_id == stored[0].id
    assert database.fts_complete()


def test_replace_chunks_rejects_empty_output_without_mutating_state(
    database: CorpusDatabase,
) -> None:
    paper, sections, chunks = sample_hierarchy()
    database.replace_paper(paper, sections, chunks)
    before = [(chunk.id, chunk.text) for chunk in database.get_chunks(paper.id)]
    revision = database.get_revision()

    with pytest.raises(ValueError, match="must not be empty"):
        database.replace_chunks_for_paper(paper.id, [])

    assert [(chunk.id, chunk.text) for chunk in database.get_chunks(paper.id)] == before
    assert database.get_revision() == revision
    assert database.paper_hierarchy_complete(paper.id)


def test_replace_chunks_without_vectors_clears_embedding_state(
    database: CorpusDatabase,
) -> None:
    paper, sections, chunks = sample_hierarchy()
    database.replace_paper(paper, sections, chunks)
    database.store_embedding(
        "paper", paper.id, [1.0, 0.0], model_name="toy", model_fingerprint="toy"
    )
    state_key = f"paper_embedding_state:{paper.id}"
    database.set_meta(state_key, '{"model_fingerprint":"toy"}')
    replacement = [
        Chunk(
            paper_id=paper.id,
            section_index=0,
            heading_path="4 Construction",
            text="Replacement text.",
            token_count=2,
        )
    ]

    database.replace_chunks_for_paper(paper.id, replacement)

    assert database.get_meta(state_key) is None
    assert database.counts()["embeddings"] == 0


def test_replace_chunks_callback_failure_restores_database_and_caller_objects(
    database: CorpusDatabase,
) -> None:
    paper, sections, chunks = sample_hierarchy()
    database.replace_paper(paper, sections, chunks)
    database.store_embedding(
        "paper", paper.id, [1.0, 0.0], model_name="toy", model_fingerprint="toy"
    )
    state_key = f"paper_embedding_state:{paper.id}"
    database.set_meta(state_key, '{"model_fingerprint":"toy"}')
    before_chunks = [(chunk.id, chunk.text) for chunk in database.get_chunks(paper.id)]
    revision = database.get_revision()
    replacement = [
        Chunk(
            paper_id=paper.id,
            section_index=0,
            chunk_index=99,
            heading_path="4 Construction",
            text="Rolled-back text.",
            token_count=2,
        )
    ]

    def fail_commit() -> None:
        raise RuntimeError("injected commit failure")

    with pytest.raises(RuntimeError, match="injected commit failure"):
        database.replace_chunks_for_paper(
            paper.id, replacement, before_commit=fail_commit
        )

    assert [(chunk.id, chunk.text) for chunk in database.get_chunks(paper.id)] == before_chunks
    assert database.get_meta(state_key) == '{"model_fingerprint":"toy"}'
    assert database.get_embedding("paper", paper.id, "toy") == pytest.approx((1.0, 0.0))
    assert database.get_revision() == revision
    assert database.fts_complete()
    assert (replacement[0].id, replacement[0].section_id, replacement[0].chunk_index) == (
        None,
        None,
        99,
    )


def test_delete_paper_cascades_hierarchy_aliases_curated_links_and_fts(
    database: CorpusDatabase,
) -> None:
    paper, sections, chunks = sample_hierarchy()
    database.replace_paper(paper, sections, chunks)
    entry = CuratedEntry("Topic", "A linked note", linked_paper_ids=[paper.id])
    database.replace_curated_entries([entry])
    database.store_embedding(
        "paper", paper.id, [1.0, 0.0], model_name="toy", model_fingerprint="toy"
    )

    assert database.delete_paper(paper.id) is True
    assert database.delete_paper(paper.id) is False
    counts = database.counts()
    assert counts["papers"] == counts["sections"] == counts["chunks"] == 0
    assert counts["source_aliases"] == counts["embeddings"] == 0
    assert database.curated_links(entry.id) == []  # type: ignore[arg-type]
    assert database.search_lexical("linked", kinds=None)  # unlinked note remains searchable
    assert database.search_lexical("vertex enclosure", kinds=None) == []
