from pathlib import Path

from papers_mcp.config import load_config
from papers_mcp.database import CORPUS_SNAPSHOT_COMPLETE_KEY, CorpusDatabase
from papers_mcp.doctor import run_doctor
from papers_mcp.embeddings import configured_embedding_identity
from papers_mcp.ingest import CorpusIngestor


def test_doctor_distinguishes_environment_health_from_search_readiness(
    tmp_path: Path,
) -> None:
    (tmp_path / "INDEX.md").write_text(
        "| name | contribution | file |\n|---|---|---|\n| paper | catalog only | paper.md |\n",
        encoding="utf-8",
    )
    (tmp_path / "paper.md").write_text(
        "# Paper\n\n## Method\n\nMethod evidence.",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[paths]
pdf_roots = ["."]
database = "data/papers.db"
human_index = "INDEX.md"
[extraction]
providers = ["pymupdf"]
[models.embedding]
backend = "hash"
allow_explicit_hash_fallback = true
[models.reranker]
backend = "lexical"
allow_explicit_lexical_fallback = true
""",
        encoding="utf-8",
    )

    report = run_doctor(load_config(config_path))

    assert report["ok"] is True
    assert report["ready"] is False
    assert any("not indexed" in warning for warning in report["warnings"])


def test_doctor_requires_one_compatible_complete_corpus_embedding_set(
    tmp_path: Path,
) -> None:
    (tmp_path / "INDEX.md").write_text(
        "| name | contribution | file |\n|---|---|---|\n| paper | catalog only | paper.md |\n",
        encoding="utf-8",
    )
    (tmp_path / "paper.md").write_text(
        "# Paper\n\n## Method\n\nMethod evidence.",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[paths]
pdf_roots = ["."]
database = "data/papers.db"
human_index = "INDEX.md"
[extraction]
providers = ["pymupdf"]
[models.embedding]
backend = "hash"
fallback_dimensions = 8
allow_explicit_hash_fallback = true
[models.reranker]
backend = "lexical"
allow_explicit_lexical_fallback = true
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    with CorpusDatabase(config.paths.database) as database:
        ingest_report = CorpusIngestor(config, database).ingest()
        assert ingest_report.failed == 0
        paper = database.list_papers()[0]
        sections = database.get_sections(paper.id)
        chunks = database.get_chunks(paper.id)
        curated = database.get_curated_entries()[0]
        with database.transaction():
            database.connection.execute("DELETE FROM embeddings")
        model_name, fingerprint = configured_embedding_identity(config.embedding, 8)
        database.store_embedding(
            "curated",
            curated.id,
            [1.0] + [0.0] * 7,
            model_name=model_name,
            model_fingerprint=fingerprint,
        )

    incomplete = run_doctor(config)
    assert incomplete["ready"] is False
    assert incomplete["vectors"]["dense_ready"] is False

    with CorpusDatabase(config.paths.database) as database:
        database.replace_embeddings_for_paper(
            paper.id,
            [
                ("paper", paper.id, [1.0] + [0.0] * 7),
                *(("section", section.id, [1.0] + [0.0] * 7) for section in sections),
                *(("chunk", chunk.id, [1.0] + [0.0] * 7) for chunk in chunks),
            ],
            model_name=model_name,
            model_fingerprint=fingerprint,
        )

    complete = run_doctor(config)
    assert complete["ready"] is True
    assert complete["vectors"]["complete_model_fingerprint"] == fingerprint

    with CorpusDatabase(config.paths.database) as database:
        database.connection.execute("DELETE FROM embeddings WHERE kind = 'curated'")
    missing_curated = run_doctor(config)
    assert missing_curated["ready"] is False
    assert missing_curated["vectors"]["dense_ready"] is False

    with CorpusDatabase(config.paths.database) as database:
        database.store_embedding(
            "curated",
            curated.id,
            [1.0] + [0.0] * 7,
            model_name=model_name,
            model_fingerprint=fingerprint,
        )
        with database.transaction():
            database.connection.execute(
                "UPDATE chunks_fts SET text = 'corrupt mirror' WHERE rowid = ?",
                (chunks[0].id,),
            )
    corrupt_fts = run_doctor(config)
    assert corrupt_fts["ready"] is False
    assert corrupt_fts["corpus"]["fts_complete"] is False

    with CorpusDatabase(config.paths.database) as database:
        database.rebuild_fts()
        database.record_failure("unreadable/subdir", "source_discovery", "permission denied")
    partial = run_doctor(config)
    assert partial["ready"] is False
    assert any("unresolved ingestion" in warning for warning in partial["warnings"])


def test_doctor_rejects_partial_ingest_until_an_unfiltered_scan_completes(
    tmp_path: Path,
) -> None:
    (tmp_path / "INDEX.md").write_text(
        "| name | file |\n|---|---|\n| a | a.md |\n| b | b.md |\n",
        encoding="utf-8",
    )
    (tmp_path / "a.md").write_text("# A\n\n## Method\n\nAlpha evidence.", encoding="utf-8")
    (tmp_path / "b.md").write_text("# B\n\n## Method\n\nBeta evidence.", encoding="utf-8")
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[paths]
pdf_roots = ["."]
database = "data/papers.db"
human_index = "INDEX.md"
exclude_dirs = ["data", "papers"]
[models.embedding]
backend = "hash"
fallback_dimensions = 8
allow_explicit_hash_fallback = true
[models.reranker]
backend = "lexical"
allow_explicit_lexical_fallback = true
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    database = CorpusDatabase(config.paths.database)

    partial = CorpusIngestor(config, database).ingest(limit=1)

    assert partial.found == 2 and partial.indexed == 1 and partial.failed == 0
    partial_doctor = run_doctor(config)
    assert partial_doctor["ready"] is False
    assert partial_doctor["corpus"]["snapshot_complete"] is False
    assert any("filter/limit" in warning for warning in partial_doctor["warnings"])

    complete = CorpusIngestor(config, database).ingest()

    assert complete.failed == 0
    complete_doctor = run_doctor(config)
    assert complete_doctor["ready"] is True
    assert complete_doctor["corpus"]["snapshot_complete"] is True
    assert complete_doctor["corpus"]["snapshot_current"] is True

    database.delete_meta(CORPUS_SNAPSHOT_COMPLETE_KEY)
    missing_marker = run_doctor(config)
    assert missing_marker["ready"] is False
    assert missing_marker["corpus"]["snapshot_complete"] is False
    assert missing_marker["corpus"]["snapshot_current"] is False
    database.set_meta(CORPUS_SNAPSHOT_COMPLETE_KEY, "1")
    assert run_doctor(config)["ready"] is True

    (tmp_path / "a.md").write_text(
        "# A\n\n## Method\n\nChanged alpha evidence.",
        encoding="utf-8",
    )
    changed_source = run_doctor(config)
    assert changed_source["ready"] is False
    assert changed_source["corpus"]["snapshot_complete"] is True
    assert changed_source["corpus"]["snapshot_current"] is False
    assert any("changed after" in warning for warning in changed_source["warnings"])

    CorpusIngestor(config, database).ingest()
    assert run_doctor(config)["ready"] is True

    with config.paths.human_index.open("a", encoding="utf-8") as handle:
        handle.write("\n<!-- catalog revision -->\n")
    changed_index = run_doctor(config)
    assert changed_index["ready"] is False
    assert changed_index["corpus"]["snapshot_current"] is False

    CorpusIngestor(config, database).ingest()
    assert run_doctor(config)["ready"] is True

    (tmp_path / "new.pdf").write_bytes(b"%PDF-new")
    added_source = run_doctor(config)
    assert added_source["ready"] is False
    assert added_source["corpus"]["snapshot_current"] is False
