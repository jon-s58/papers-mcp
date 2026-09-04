import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import papers_mcp.ingest as ingest_module
from papers_mcp.config import load_config
from papers_mcp.curated_index import parse_curated_index
from papers_mcp.database import CorpusDatabase
from papers_mcp.ingest import (
    CorpusIngestor,
    _extract_abstract,
    _usable_authors,
    discover_sources,
)
from papers_mcp.memory import MemoryBudgetExceeded
from papers_mcp.models import ExtractedDocument
from papers_mcp.service import ResearchCorpus


def _config(tmp_path: Path):
    (tmp_path / "INDEX.md").write_text(
        """
# Geometry
| name | field | contribution | pdf-or-md |
|---|---|---|---|
| alpha-paper | splines | Vertex enclosure constraints; curated-only parity oracle | alpha.md |
| beta-paper | fitting | Neighboring tangent planes | beta.md |
| broken-paper | malformed | Failure fixture | broken.md |
""",
        encoding="utf-8",
    )
    (tmp_path / "alpha.md").write_text(
        """# Vertex Enclosure

## Method

The exact vertex enclosure condition enforces G1 compatibility.

$$
E(S) = E_{fit}(S) + \\lambda E_{fair}(S).
$$

where the variables share cross-boundary derivatives.
""",
        encoding="utf-8",
    )
    (tmp_path / "beta.md").write_text(
        """# Joint Surface Fitting

## Tangent Constraints

Neighboring analytic surfaces are fitted together while sharing tangent planes.
""",
        encoding="utf-8",
    )
    (tmp_path / "broken.md").write_text("", encoding="utf-8")
    path = tmp_path / "config.toml"
    path.write_text(
        """
[paths]
pdf_roots = ["."]
generated_markdown = "papers/markdown"
cards = "papers/cards"
database = "data/papers.db"
human_index = "INDEX.md"
exclude_dirs = ["data", "papers/markdown", "papers/cards"]
markdown_sources_from_index = true
[models.embedding]
backend = "hash"
fallback_dimensions = 128
allow_explicit_hash_fallback = true
[models.reranker]
backend = "lexical"
allow_explicit_lexical_fallback = true
[chunks]
target_tokens = 30
min_tokens = 5
max_tokens = 80
[retrieval]
bm25_candidates = 20
dense_candidates = 20
rerank_candidates = 20
rrf_k = 60
default_top_k = 10
precision_max_results_per_paper = 3
discovery_max_results_per_paper = 2
exact_max_results_per_paper = 10
snippet_words = 100
""",
        encoding="utf-8",
    )
    return load_config(path)


def test_incremental_ingest_and_complete_research_flow(tmp_path: Path) -> None:
    config = _config(tmp_path)
    database = CorpusDatabase(config.paths.database)
    report = CorpusIngestor(config, database).ingest()
    assert (report.indexed, report.failed) == (2, 1)
    assert database.counts()["papers"] == 2
    assert (config.paths.cards / "alpha-paper.json").is_file()
    assert (
        database.connection.execute(
            "SELECT COUNT(*) FROM embeddings e "
            "LEFT JOIN curated_entry_papers cep "
            "ON cep.entry_id = CAST(e.entity_id AS INTEGER) "
            "AND cep.paper_id = e.paper_id "
            "WHERE e.kind = 'curated' "
            "AND ((e.paper_id IS NULL AND EXISTS ("
            "SELECT 1 FROM curated_entry_papers linked "
            "WHERE linked.entry_id = CAST(e.entity_id AS INTEGER))) "
            "OR (e.paper_id IS NOT NULL AND cep.paper_id IS NULL))"
        ).fetchone()[0]
        == 0
    )

    second = CorpusIngestor(config, database).ingest()
    assert second.indexed == 0
    assert second.skipped == 2
    assert second.failed == 1

    corpus = ResearchCorpus(config, database=database)
    exact = corpus.search("vertex enclosure", mode="exact", top_k=5)
    assert exact and exact[0].paper_id == "alpha-paper"
    context = corpus.expand_context(exact[0].result_id)
    assert "E(S)" in context["text"]
    outline = corpus.paper_outline("alpha-paper")
    assert outline["sections"]
    section_id = next(
        item["section_id"] for item in outline["sections"] if item["heading"] == "Method"
    )
    assert corpus.read_section("alpha-paper", section_id)["text"]
    assert corpus.find_in_paper("alpha-paper", "cross-boundary derivatives")
    discovery = corpus.research_search("surfaces agree across tangent boundaries", top_k=5)
    assert {result.paper_id for result in discovery} >= {"alpha-paper", "beta-paper"}
    curated_route = corpus.search(
        "curated-only parity oracle",
        mode="discovery",
        top_k=3,
        pipeline="bm25",
    )
    assert curated_route[0].paper_id == "alpha-paper"
    assert curated_route[0].kind == "paper"
    assert "curated-only parity oracle" in (curated_route[0].routing_note or "")
    assert "curated-only parity oracle" not in curated_route[0].snippet
    assert "expand_context" not in curated_route[0].next_actions
    assert "paper_outline" in curated_route[0].next_actions
    assert "find_in_paper" in curated_route[0].next_actions

    unresolved = corpus.search(
        "Failure fixture",
        mode="discovery",
        top_k=3,
        pipeline="bm25",
    )
    assert unresolved[0].kind == "curated"
    assert unresolved[0].paper_id is None
    assert unresolved[0].source_kind == "curated_index"
    assert unresolved[0].next_actions == {}
    related = corpus.related_papers("alpha-paper", top_k=3)
    assert related[0]["paper_id"] == "beta-paper"


def test_markdown_discovery_resolves_curated_paths_and_skips_admin_files(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with config.paths.human_index.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n| name | account |\n|---|---|\n"
            "| tensor-note | docs/research/papers/tensor_note.md |\n"
            "| backlog | JON_DOWNLOAD_LIST.md |\n"
        )
    (tmp_path / "tensor_note.md").write_text("# Tensor voting\n", encoding="utf-8")
    (tmp_path / "JON_DOWNLOAD_LIST.md").write_text("# Backlog\n", encoding="utf-8")

    curated = parse_curated_index(config.paths.human_index)
    names = {source.path.name for source in discover_sources(config, curated)}

    assert "tensor_note.md" in names
    assert "JON_DOWNLOAD_LIST.md" not in names


def test_markdown_discovery_prefers_exact_casefolded_paths_over_ambiguous_basenames(
    tmp_path: Path,
) -> None:
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    exact_source = tmp_path / "one" / "Shared.MD"
    exact_source.write_text("# Exact source\n", encoding="utf-8")
    (tmp_path / "two" / "shared.md").write_text("# Other source\n", encoding="utf-8")
    (tmp_path / "INDEX.md").write_text(
        "| name | file |\n|---|---|\n| exact | ONE/shared.md |\n| ambiguous | shared.md |\n",
        encoding="utf-8",
    )
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
[models.reranker]
backend = "lexical"
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    curated = parse_curated_index(config.paths.human_index)

    sources = discover_sources(config, curated)

    assert [source.path for source in sources] == [exact_source.resolve()]


def test_ambiguous_bare_pdf_reference_cannot_link_by_generated_id_coincidence(
    tmp_path: Path,
) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "shared.pdf").write_bytes(b"%PDF-root")
    (tmp_path / "nested" / "SHARED.PDF").write_bytes(b"%PDF-nested")
    (tmp_path / "INDEX.md").write_text(
        "| name | contribution | file |\n|---|---|---|\n"
        "| shared | AMBIGUOUS BARE NOTE | shared.pdf |\n"
        "| exact-route | EXACT PATH NOTE | NESTED/shared.pdf |\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[paths]
pdf_roots = ["."]
generated_markdown = "papers/markdown"
cards = "papers/cards"
database = "data/papers.db"
human_index = "INDEX.md"
exclude_dirs = ["data", "papers"]
[models.embedding]
backend = "hash"
[models.reranker]
backend = "lexical"
""",
        encoding="utf-8",
    )
    config = load_config(config_path)

    class Extractor:
        def extract(self, source: Path) -> ExtractedDocument:
            return ExtractedDocument(
                markdown=f"# {source.stem}\n\n## Method\n\nSource-specific evidence.",
                backend="fake",
                title=source.stem,
            )

    database = CorpusDatabase(config.paths.database)
    report = CorpusIngestor(config, database, extractor=Extractor()).ingest(embeddings=False)

    assert report.failed == 0
    root_paper = database.find_paper_by_source("shared.pdf")
    nested_paper = database.find_paper_by_source("nested/SHARED.PDF")
    assert root_paper is not None and root_paper.id == "shared"
    assert nested_paper is not None and nested_paper.id == "exact-route"
    entries = database.get_curated_entries()
    bare_entry = next(entry for entry in entries if "AMBIGUOUS BARE NOTE" in entry.text)
    exact_entry = next(entry for entry in entries if "EXACT PATH NOTE" in entry.text)
    assert bare_entry.linked_paper_ids == []
    assert exact_entry.linked_paper_ids == [nested_paper.id]


def test_multi_artifact_basename_row_links_every_nested_pdf(tmp_path: Path) -> None:
    source_root = tmp_path / "pdfs"
    source_root.mkdir()
    (source_root / "a.pdf").write_bytes(b"%PDF-a")
    (source_root / "b.pdf").write_bytes(b"%PDF-b")
    (tmp_path / "INDEX.md").write_text(
        "| name | contribution | file |\n|---|---|---|\n"
        "| combo | SHARED EXPERT NOTE | a.pdf; b.pdf |\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[paths]
pdf_roots = ["pdfs"]
generated_markdown = "papers/markdown"
cards = "papers/cards"
database = "data/papers.db"
human_index = "INDEX.md"
exclude_dirs = ["data", "papers"]
[models.embedding]
backend = "hash"
[models.reranker]
backend = "lexical"
""",
        encoding="utf-8",
    )
    config = load_config(config_path)

    class Extractor:
        def extract(self, source: Path) -> ExtractedDocument:
            return ExtractedDocument(
                markdown=f"# {source.stem}\n\n## Method\n\n{source.stem} evidence.",
                backend="fake",
                title=source.stem,
            )

    database = CorpusDatabase(config.paths.database)
    report = CorpusIngestor(config, database, extractor=Extractor()).ingest(embeddings=False)

    assert report.failed == 0
    owners = [
        database.find_paper_by_source("pdfs/a.pdf"),
        database.find_paper_by_source("pdfs/b.pdf"),
    ]
    assert None not in owners
    owner_ids = {paper.id for paper in owners if paper is not None}
    assert len(owner_ids) == 2
    entry = next(
        item for item in database.get_curated_entries() if "SHARED EXPERT NOTE" in item.text
    )
    assert set(entry.linked_paper_ids) == owner_ids
    for paper_id in owner_ids:
        card = json.loads((config.paths.cards / f"{paper_id}.json").read_text(encoding="utf-8"))
        assert "SHARED EXPERT NOTE" in card["curated_notes"]


def test_exact_artifacts_relative_to_pdf_root_link_despite_ambiguous_basename(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "pdfs"
    (source_root / "one").mkdir(parents=True)
    (source_root / "two").mkdir()
    (source_root / "one" / "shared.pdf").write_bytes(b"%PDF-one")
    (source_root / "two" / "SHARED.PDF").write_bytes(b"%PDF-two")
    (tmp_path / "INDEX.md").write_text(
        "| name | contribution | file |\n|---|---|---|\n"
        "| one-route | ONE EXACT NOTE | one/shared.pdf |\n"
        "| two-route | TWO EXACT NOTE | TWO/shared.pdf |\n"
        "| ambiguous | BARE NOTE | shared.pdf |\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[paths]
pdf_roots = ["pdfs"]
generated_markdown = "papers/markdown"
cards = "papers/cards"
database = "data/papers.db"
human_index = "INDEX.md"
exclude_dirs = ["data", "papers"]
[models.embedding]
backend = "hash"
[models.reranker]
backend = "lexical"
""",
        encoding="utf-8",
    )
    config = load_config(config_path)

    class Extractor:
        def extract(self, source: Path) -> ExtractedDocument:
            return ExtractedDocument(
                markdown=f"# {source.parent.name}\n\n## Method\n\nExact evidence.",
                backend="fake",
                title=source.parent.name,
            )

    database = CorpusDatabase(config.paths.database)
    report = CorpusIngestor(config, database, extractor=Extractor()).ingest(embeddings=False)

    assert report.failed == 0
    one = database.find_paper_by_source("pdfs/one/shared.pdf")
    two = database.find_paper_by_source("pdfs/two/SHARED.PDF")
    assert one is not None and one.id == "one-route"
    assert two is not None and two.id == "two-route"
    entries = database.get_curated_entries()
    assert next(item for item in entries if "ONE EXACT NOTE" in item.text).linked_paper_ids == [
        one.id
    ]
    assert next(item for item in entries if "TWO EXACT NOTE" in item.text).linked_paper_ids == [
        two.id
    ]
    assert next(item for item in entries if "BARE NOTE" in item.text).linked_paper_ids == []


def test_normal_ingest_repairs_embeddings_after_no_embeddings_run(tmp_path: Path) -> None:
    config = _config(tmp_path)
    database = CorpusDatabase(config.paths.database)

    first = CorpusIngestor(config, database).ingest(embeddings=False)
    assert first.indexed == 2
    assert database.counts()["embeddings"] == 0

    repaired = CorpusIngestor(config, database).ingest()
    assert repaired.indexed == 2
    assert repaired.skipped == 0
    assert database.counts()["embeddings"] > 0

    unchanged = CorpusIngestor(config, database).ingest()
    assert unchanged.indexed == 0
    assert unchanged.skipped == 2


def test_rechunk_rebuilds_only_changed_derived_chunks_without_extraction(tmp_path: Path) -> None:
    config = _config(tmp_path)
    database = CorpusDatabase(config.paths.database)
    CorpusIngestor(config, database).ingest()
    alpha = database.get_paper("alpha-paper")
    assert alpha is not None
    database.connection.execute(
        "UPDATE chunks SET chunk_index = 99 WHERE paper_id = ?",
        (alpha.id,),
    )

    class NoExtraction:
        def extract(self, source: Path) -> ExtractedDocument:
            raise AssertionError("unchanged rechunking must use stored sections")

    report = CorpusIngestor(config, database, extractor=NoExtraction()).ingest(rechunk=True)

    assert report.indexed == 1
    assert report.skipped == 1
    chunks = database.get_chunks(alpha.id)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    fingerprint = database.embedding_fingerprints()[0]["model_fingerprint"]
    assert database.paper_embeddings_complete(alpha.id, fingerprint)


def test_rechunk_empty_output_preserves_last_good_hierarchy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    database = CorpusDatabase(config.paths.database)
    CorpusIngestor(config, database).ingest()
    alpha = database.get_paper("alpha-paper")
    assert alpha is not None
    before = [(chunk.id, chunk.text) for chunk in database.get_chunks(alpha.id)]
    monkeypatch.setattr(ingest_module, "chunk_sections", lambda *_args, **_kwargs: [])

    report = CorpusIngestor(config, database).ingest(
        rechunk=True, patterns=("alpha.md",)
    )

    assert report.failed == 1
    assert report.failures[0].stage == "rechunk"
    assert [(chunk.id, chunk.text) for chunk in database.get_chunks(alpha.id)] == before
    assert database.paper_hierarchy_complete(alpha.id)


def test_embedding_config_and_linked_note_changes_refresh_unchanged_papers(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    database = CorpusDatabase(config.paths.database)
    CorpusIngestor(config, database).ingest()

    resized = replace(
        config,
        embedding=replace(config.embedding, fallback_dimensions=256),
    )
    model_refresh = CorpusIngestor(resized, database).ingest()
    assert model_refresh.indexed == 2
    assert {row["dimensions"] for row in database.embedding_fingerprints()} == {256}

    index_text = config.paths.human_index.read_text(encoding="utf-8")
    config.paths.human_index.write_text(
        index_text.replace(
            "Vertex enclosure constraints",
            "Vertex enclosure parity constraints with a new feasibility note",
        ),
        encoding="utf-8",
    )
    note_refresh = CorpusIngestor(resized, database).ingest()
    assert note_refresh.indexed == 1
    assert note_refresh.skipped == 1
    card = json.loads((config.paths.cards / "alpha-paper.json").read_text(encoding="utf-8"))
    assert "new feasibility note" in card["curated_notes"]


def test_normal_ingest_repairs_curated_vectors_after_index_only_no_embedding_change(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    database = CorpusDatabase(config.paths.database)
    CorpusIngestor(config, database).ingest()
    assert database.connection.execute(
        "SELECT COUNT(*) FROM embeddings WHERE kind = 'curated'"
    ).fetchone()[0]

    with config.paths.human_index.open("a", encoding="utf-8") as handle:
        handle.write("\n## Unlinked synthesis\n\nA brand-new independent research direction.\n")
    changed_without_vectors = CorpusIngestor(config, database).ingest(embeddings=False)
    assert changed_without_vectors.indexed == 0
    assert (
        database.connection.execute(
            "SELECT COUNT(*) FROM embeddings WHERE kind = 'curated'"
        ).fetchone()[0]
        == 0
    )

    repaired = CorpusIngestor(config, database).ingest()
    curated_count = database.connection.execute("SELECT COUNT(*) FROM curated_entries").fetchone()[
        0
    ]
    vector_count = database.connection.execute(
        "SELECT COUNT(*) FROM embeddings WHERE kind = 'curated'"
    ).fetchone()[0]
    assert repaired.indexed == 0
    assert vector_count == curated_count


def test_incremental_duplicate_convergence_retires_orphan_paper_and_artifacts(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    database = CorpusDatabase(config.paths.database)
    CorpusIngestor(config, database).ingest(embeddings=False)
    alpha_bytes = (tmp_path / "alpha.md").read_bytes()
    (tmp_path / "beta.md").write_bytes(alpha_bytes)

    report = CorpusIngestor(config, database).ingest(embeddings=False)

    assert report.duplicates == 1
    assert database.counts()["papers"] == 1
    assert database.get_paper("beta-paper") is None
    assert not (config.paths.cards / "beta-paper.json").exists()
    assert {row["source_path"] for row in database.source_aliases("alpha-paper")} == {
        "alpha.md",
        "beta.md",
    }
    assert not database.lexical_search("paper", "neighboring tangent planes", 5)


def test_incremental_duplicate_alias_can_diverge_into_a_new_paper(tmp_path: Path) -> None:
    config = _config(tmp_path)
    beta_text = (tmp_path / "beta.md").read_text(encoding="utf-8")
    (tmp_path / "beta.md").write_bytes((tmp_path / "alpha.md").read_bytes())
    database = CorpusDatabase(config.paths.database)
    CorpusIngestor(config, database).ingest(embeddings=False)
    assert database.counts()["papers"] == 1

    (tmp_path / "beta.md").write_text(beta_text, encoding="utf-8")
    report = CorpusIngestor(config, database).ingest(embeddings=False)

    assert report.indexed == 1
    assert database.counts()["papers"] == 2
    assert database.find_paper_by_source("alpha.md").id == "alpha-paper"  # type: ignore[union-attr]
    assert database.find_paper_by_source("beta.md").id == "beta-paper"  # type: ignore[union-attr]
    assert (
        database.lexical_search("chunk", "neighboring tangent planes", 5)[0].paper_id
        == "beta-paper"
    )


def test_successful_unchanged_or_duplicate_source_clears_stale_failures(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    database = CorpusDatabase(config.paths.database)
    CorpusIngestor(config, database).ingest(embeddings=False)
    original = (tmp_path / "alpha.md").read_bytes()

    (tmp_path / "alpha.md").write_text("", encoding="utf-8")
    failed = CorpusIngestor(config, database).ingest(embeddings=False)
    assert failed.failed >= 1
    (tmp_path / "alpha.md").write_bytes(original)
    CorpusIngestor(config, database).ingest(embeddings=False)
    assert not [item for item in database.list_failures() if item.source_path == "alpha.md"]

    (tmp_path / "broken.md").write_bytes(original)
    CorpusIngestor(config, database).ingest(embeddings=False)
    assert not [item for item in database.list_failures() if item.source_path == "broken.md"]


def _single_pdf_config(tmp_path: Path):
    (tmp_path / "INDEX.md").write_text(
        "| name | file |\n|---|---|\n| paper | paper.pdf |\n",
        encoding="utf-8",
    )
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-v1")
    path = tmp_path / "config.toml"
    path.write_text(
        """
[paths]
pdf_roots = ["."]
generated_markdown = "papers/markdown"
cards = "papers/cards"
database = "data/papers.db"
human_index = "INDEX.md"
exclude_dirs = ["data", "papers/markdown", "papers/cards"]
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
    return load_config(path)


def test_failed_generated_markdown_publish_rolls_back_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _single_pdf_config(tmp_path)
    database = CorpusDatabase(config.paths.database)

    class Extractor:
        markdown = "# Version One\n\nOriginal evidence."

        def extract(self, path: Path) -> ExtractedDocument:
            return ExtractedDocument(markdown=self.markdown, backend="fake", title="Paper")

    extractor = Extractor()
    CorpusIngestor(config, database, extractor=extractor).ingest(embeddings=False)
    original = database.get_paper("paper")
    generated = config.paths.generated_markdown / "paper.md"
    assert original is not None and "Version One" in generated.read_text(encoding="utf-8")

    (tmp_path / "paper.pdf").write_bytes(b"%PDF-v2")
    extractor.markdown = "# Version Two\n\nReplacement evidence."
    monkeypatch.setattr(
        "papers_mcp.ingest._publish_staged_file",
        lambda source, target: (_ for _ in ()).throw(OSError("publish failed")),
    )
    failed = CorpusIngestor(config, database, extractor=extractor).ingest(embeddings=False)

    assert failed.failed == 1
    assert database.get_paper("paper").content_hash == original.content_hash  # type: ignore[union-attr]
    assert "Version One" in generated.read_text(encoding="utf-8")


def test_memory_budget_failure_preserves_last_good_paper_and_aborts_ingest(
    tmp_path: Path,
) -> None:
    config = _single_pdf_config(tmp_path)
    database = CorpusDatabase(config.paths.database)

    class Extractor:
        markdown = "# Version One\n\n## Method\n\nOriginal evidence."

        def extract(self, path: Path) -> ExtractedDocument:
            return ExtractedDocument(markdown=self.markdown, backend="fake", title="Paper")

    extractor = Extractor()
    CorpusIngestor(config, database, extractor=extractor).ingest(embeddings=False)
    original = database.get_paper("paper")
    original_sections = [
        (section.heading_path, section.text) for section in database.get_sections("paper")
    ]
    generated = config.paths.generated_markdown / "paper.md"
    assert original is not None
    original_generated = generated.read_text(encoding="utf-8")

    (tmp_path / "paper.pdf").write_bytes(b"%PDF-v2")
    extractor.markdown = "# Version Two\n\n## Method\n\nReplacement evidence."

    class BudgetFailureEmbedding:
        active_backend = "sentence_transformers"
        is_loaded = True

        def __init__(self) -> None:
            self.config = config.embedding

        def embed_documents(self, texts):
            raise MemoryBudgetExceeded("bounded test failure")

        def embed_query(self, text):
            raise MemoryBudgetExceeded("bounded test failure")

    with pytest.raises(MemoryBudgetExceeded, match="bounded test failure"):
        CorpusIngestor(
            config,
            database,
            extractor=extractor,
            embedding=BudgetFailureEmbedding(),
        ).ingest()

    preserved = database.get_paper("paper")
    assert preserved is not None and preserved.content_hash == original.content_hash
    assert [
        (section.heading_path, section.text) for section in database.get_sections("paper")
    ] == original_sections
    assert generated.read_text(encoding="utf-8") == original_generated
    failure = next(item for item in database.list_failures() if item.source_path == "paper.pdf")
    assert failure.stage == "ingest"


def test_index_snapshot_hash_matches_the_entries_parsed_during_ingest(tmp_path: Path) -> None:
    config = _single_pdf_config(tmp_path)
    old_index = (
        "| name | contribution | file |\n|---|---|---|\n| paper | OLD expert note | paper.pdf |\n"
    )
    new_index = old_index.replace("OLD expert note", "NEW expert note")
    config.paths.human_index.write_text(old_index, encoding="utf-8")
    database = CorpusDatabase(config.paths.database)

    class MutatingExtractor:
        def extract(self, path: Path) -> ExtractedDocument:
            config.paths.human_index.write_text(new_index, encoding="utf-8")
            return ExtractedDocument(
                markdown="# Paper\n\nStable source evidence.",
                backend="fake",
                title="Paper",
            )

    CorpusIngestor(config, database, extractor=MutatingExtractor()).ingest(embeddings=False)
    assert any("OLD expert note" in item.text for item in database.get_curated_entries())
    assert (
        database.get_meta("curated_index_hash")
        == hashlib.sha256(old_index.encode("utf-8")).hexdigest()
    )

    CorpusIngestor(config, database, extractor=MutatingExtractor()).ingest(embeddings=False)
    assert any("NEW expert note" in item.text for item in database.get_curated_entries())


def test_curated_links_refresh_when_a_referenced_alias_appears(tmp_path: Path) -> None:
    (tmp_path / "INDEX.md").write_text(
        "| name | contribution | file |\n|---|---|---|\n"
        "| alpha | alpha note | alpha.md |\n"
        "| beta | beta note | beta.md |\n",
        encoding="utf-8",
    )
    (tmp_path / "alpha.md").write_text("# Shared\n\nEvidence.", encoding="utf-8")
    path = tmp_path / "config.toml"
    path.write_text(
        """
[paths]
pdf_roots = ["."]
database = "data/papers.db"
human_index = "INDEX.md"
exclude_dirs = ["data"]
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
    config = load_config(path)
    database = CorpusDatabase(config.paths.database)
    CorpusIngestor(config, database).ingest(embeddings=False)
    beta_before = next(item for item in database.get_curated_entries() if "beta note" in item.text)
    assert beta_before.linked_paper_ids == []

    (tmp_path / "beta.md").write_bytes((tmp_path / "alpha.md").read_bytes())
    CorpusIngestor(config, database).ingest(embeddings=False)
    beta_after = next(item for item in database.get_curated_entries() if "beta note" in item.text)
    assert beta_after.linked_paper_ids == ["alpha"]


def test_full_ingest_retires_removed_sources_but_filtered_ingest_does_not(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    database = CorpusDatabase(config.paths.database)
    CorpusIngestor(config, database).ingest(embeddings=False)
    (tmp_path / "beta.md").unlink()

    CorpusIngestor(config, database).ingest(patterns=("alpha.md",), embeddings=False)
    assert database.get_paper("beta-paper") is not None

    CorpusIngestor(config, database).ingest(embeddings=False)
    assert database.get_paper("beta-paper") is None
    assert not (config.paths.cards / "beta-paper.json").exists()


def test_removing_canonical_duplicate_repoints_paper_to_live_alias(tmp_path: Path) -> None:
    config = _config(tmp_path)
    (tmp_path / "beta.md").write_bytes((tmp_path / "alpha.md").read_bytes())
    database = CorpusDatabase(config.paths.database)
    CorpusIngestor(config, database).ingest(embeddings=False)
    (tmp_path / "alpha.md").unlink()

    CorpusIngestor(config, database).ingest(embeddings=False)

    paper = database.list_papers()[0]
    assert paper.source_path == "beta.md"
    assert [item["source_path"] for item in database.source_aliases(paper.id)] == ["beta.md"]


def test_mid_ingest_backend_fallback_reconciles_every_paper_to_one_identity(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    production_config = replace(
        config,
        embedding=replace(
            config.embedding,
            backend="sentence_transformers",
            model="fake-production-model",
        ),
    )
    database = CorpusDatabase(config.paths.database)

    class TransitionEmbedding:
        def __init__(self) -> None:
            self.config = production_config.embedding
            self.active_backend = "sentence_transformers"
            self.is_loaded = True
            self.document_calls = 0

        def embed_documents(self, texts):
            self.document_calls += 1
            if self.document_calls >= 2:
                self.active_backend = "hash"
            dimensions = 4 if self.active_backend == "sentence_transformers" else 8
            return [[1.0] + [0.0] * (dimensions - 1) for _ in texts]

        def embed_query(self, text):
            dimensions = 4 if self.active_backend == "sentence_transformers" else 8
            return [1.0] + [0.0] * (dimensions - 1)

    provider = TransitionEmbedding()
    report = CorpusIngestor(
        production_config,
        database,
        embedding=provider,
    ).ingest()

    assert report.failed == 1  # only the intentionally empty broken.md fixture
    paper_sets = database.connection.execute(
        "SELECT DISTINCT model_name, model_fingerprint, dimensions "
        "FROM embeddings WHERE kind IN ('paper', 'section', 'chunk')"
    ).fetchall()
    assert len(paper_sets) == 1
    assert paper_sets[0]["model_name"] == "hash-v1:8"
    assert database.corpus_embeddings_complete(paper_sets[0]["model_fingerprint"])


def test_generated_pdf_markdown_never_overwrites_a_canonical_markdown_source(
    tmp_path: Path,
) -> None:
    (tmp_path / "refs").mkdir()
    canonical = tmp_path / "refs" / "note.md"
    canonical.write_text("# Handwritten Note\n\nCanonical evidence.", encoding="utf-8")
    (tmp_path / "note.pdf").write_bytes(b"%PDF-note")
    (tmp_path / "INDEX.md").write_text(
        "| name | files |\n|---|---|\n| note | note.pdf; refs/note.md |\n",
        encoding="utf-8",
    )
    path = tmp_path / "config.toml"
    path.write_text(
        """
[paths]
pdf_roots = ["."]
generated_markdown = "refs"
cards = "cards"
database = "data/papers.db"
human_index = "INDEX.md"
exclude_dirs = ["data", "cards"]
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
    config = load_config(path)

    class Extractor:
        def extract(self, source: Path) -> ExtractedDocument:
            return ExtractedDocument(
                markdown="# PDF Extraction\n\nGenerated evidence.",
                backend="fake",
                title="PDF",
            )

    report = CorpusIngestor(
        config,
        CorpusDatabase(config.paths.database),
        extractor=Extractor(),
    ).ingest(embeddings=False)

    assert report.failed >= 1
    assert canonical.read_text(encoding="utf-8").startswith("# Handwritten Note")


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        (
            "## **Abstract**\n\nA bold-heading abstract.\n\n## Introduction\nNot abstract.",
            "A bold-heading abstract.",
        ),
        (
            "**Abstract** An inline abstract.\n\n# Method\nNot abstract.",
            "An inline abstract.",
        ),
        (
            "Summary: A plain summary.\n\n## Results\nNot abstract.",
            "A plain summary.",
        ),
    ],
)
def test_common_abstract_markup_is_extracted_and_bounded(
    markdown: str,
    expected: str,
) -> None:
    assert _extract_abstract(markdown) == expected


def test_extracted_full_authors_are_preferred_but_junk_is_rejected() -> None:
    assert _usable_authors(["Qi Li", "Yixin Zhuang", "Xiaohu Guo"])
    assert not _usable_authors(["maas"])
    assert not _usable_authors(["Microsoft Word"])


def test_final_paper_ids_drive_curated_links_cards_and_failed_source_retention(
    tmp_path: Path,
) -> None:
    (tmp_path / "INDEX.md").write_text(
        "| name | contribution | file |\n"
        "|---|---|---|\n"
        "| same | X NOTE | x.md |\n"
        "| same | Y NOTE | y.md |\n"
        "| same | SHARED RICH NOTE | |\n",
        encoding="utf-8",
    )
    (tmp_path / "x.md").write_text("# X Paper\n\n## Method\n\nX evidence.", encoding="utf-8")
    (tmp_path / "y.md").write_text("# Y Paper\n\n## Method\n\nY evidence.", encoding="utf-8")
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
[models.reranker]
backend = "lexical"
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    database = CorpusDatabase(config.paths.database)

    first = CorpusIngestor(config, database).ingest(embeddings=False)
    assert first.failed == 0
    x_owner = database.find_paper_by_source("x.md")
    y_owner = database.find_paper_by_source("y.md")
    assert x_owner is not None and y_owner is not None and x_owner.id != y_owner.id
    entries = database.get_curated_entries()
    x_note = next(entry for entry in entries if "X NOTE" in entry.text)
    y_note = next(entry for entry in entries if "Y NOTE" in entry.text)
    shared = next(entry for entry in entries if "SHARED RICH NOTE" in entry.text)
    assert x_note.linked_paper_ids == [x_owner.id]
    assert y_note.linked_paper_ids == [y_owner.id]
    assert set(shared.linked_paper_ids) == {x_owner.id, y_owner.id}

    x_card = json.loads((config.paths.cards / f"{x_owner.id}.json").read_text())
    y_card = json.loads((config.paths.cards / f"{y_owner.id}.json").read_text())
    assert "X NOTE" in x_card["curated_notes"] and "Y NOTE" not in x_card["curated_notes"]
    assert "Y NOTE" in y_card["curated_notes"] and "X NOTE" not in y_card["curated_notes"]

    (tmp_path / "y.md").write_text("", encoding="utf-8")
    failed = CorpusIngestor(config, database).ingest(embeddings=False)
    assert failed.failed == 1
    retained_y_note = next(
        entry for entry in database.get_curated_entries() if "Y NOTE" in entry.text
    )
    assert retained_y_note.linked_paper_ids == [y_owner.id]


def test_catalog_association_not_stable_database_id_drives_artifactless_relink(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.md").write_text("# A\n\n## Method\n\nA evidence.", encoding="utf-8")
    (tmp_path / "INDEX.md").write_text(
        "| name | contribution | file |\n|---|---|---|\n| same | OTHER | a.md |\n",
        encoding="utf-8",
    )
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
[models.reranker]
backend = "lexical"
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    database = CorpusDatabase(config.paths.database)
    CorpusIngestor(config, database).ingest(embeddings=False)

    (tmp_path / "x.md").write_text("# X\n\n## Method\n\nX evidence.", encoding="utf-8")
    (tmp_path / "INDEX.md").write_text(
        "| name | contribution | file |\n"
        "|---|---|---|\n"
        "| other | OTHER | a.md |\n"
        "| same | RICH | |\n"
        "| same | X INVENTORY | x.md |\n",
        encoding="utf-8",
    )
    CorpusIngestor(config, database).ingest(embeddings=False)

    x_owner = database.find_paper_by_source("x.md")
    a_owner = database.find_paper_by_source("a.md")
    rich = next(entry for entry in database.get_curated_entries() if "RICH" in entry.text)
    assert x_owner is not None and a_owner is not None
    assert rich.linked_paper_ids == [x_owner.id]
    assert a_owner.id not in rich.linked_paper_ids


def test_missing_source_root_aborts_without_retiring_last_good_data(tmp_path: Path) -> None:
    source_root = tmp_path / "pdfs"
    source_root.mkdir()
    (source_root / "UPPER.PDF").write_bytes(b"%PDF-upper")
    (tmp_path / "INDEX.md").write_text(
        "| name | file |\n|---|---|\n| upper | UPPER.PDF |\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[paths]
pdf_roots = ["pdfs"]
database = "data/papers.db"
human_index = "INDEX.md"
exclude_dirs = ["data", "papers"]
[models.embedding]
backend = "hash"
fallback_dimensions = 8
[models.reranker]
backend = "lexical"
""",
        encoding="utf-8",
    )
    config = load_config(config_path)

    class Extractor:
        def extract(self, source: Path) -> ExtractedDocument:
            return ExtractedDocument(
                markdown="# Upper\n\n## Method\n\nUppercase-extension evidence.",
                backend="fake",
                title="Upper Paper",
            )

    database = CorpusDatabase(config.paths.database)
    first = CorpusIngestor(config, database, extractor=Extractor()).ingest(embeddings=False)
    assert first.indexed == 1
    paper = database.list_papers()[0]
    card = config.paths.cards / f"{paper.id}.json"
    generated = config.paths.generated_markdown / f"{paper.id}.md"
    source_root.rename(tmp_path / "pdfs-offline")

    with pytest.raises(RuntimeError, match="missing or unreadable"):
        CorpusIngestor(config, database, extractor=Extractor()).ingest(embeddings=False)

    assert database.get_paper(paper.id) is not None
    assert card.is_file() and generated.is_file()


def test_generated_markdown_is_restored_if_commit_fails_after_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _single_pdf_config(tmp_path)
    database = CorpusDatabase(config.paths.database)

    class Extractor:
        markdown = "# Version One\n\nOriginal evidence."

        def extract(self, path: Path) -> ExtractedDocument:
            return ExtractedDocument(markdown=self.markdown, backend="fake", title="Paper")

    extractor = Extractor()
    CorpusIngestor(config, database, extractor=extractor).ingest(embeddings=False)
    original = database.get_paper("paper")
    generated = config.paths.generated_markdown / "paper.md"
    (tmp_path / "paper.pdf").write_bytes(b"%PDF-v2")
    extractor.markdown = "# Version Two\n\nReplacement evidence."
    original_bump = database._bump_revision
    calls = 0

    def fail_once() -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("commit bookkeeping failed")
        return original_bump()

    monkeypatch.setattr(database, "_bump_revision", fail_once)
    failed = CorpusIngestor(config, database, extractor=extractor).ingest(embeddings=False)

    assert failed.failed == 1
    assert original is not None
    assert database.get_paper("paper").content_hash == original.content_hash  # type: ignore[union-attr]
    assert "Version One" in generated.read_text(encoding="utf-8")


def test_curated_embedding_failure_is_reported_without_aborting_ingest(tmp_path: Path) -> None:
    config = _config(tmp_path)
    database = CorpusDatabase(config.paths.database)

    class FailingCuratedEmbedding:
        calls = 0

        def embed_documents(self, texts):
            self.calls += 1
            if self.calls > 2:  # two good papers, then the curated batch
                raise RuntimeError("curated embedding failed")
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

        def embed_query(self, text):
            return [1.0, 0.0, 0.0, 0.0]

    report = CorpusIngestor(
        config,
        database,
        embedding=FailingCuratedEmbedding(),
    ).ingest()

    assert any(failure.stage == "embedding_consistency" for failure in report.failures)
    assert database.get_meta("last_ingest_report") is not None
    assert any(
        failure.source_path == "__embedding_consistency__" for failure in database.list_failures()
    )


def test_hash_failure_does_not_abort_good_sources_or_retire_last_good_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "INDEX.md").write_text(
        "| name | file |\n|---|---|\n| x | x.md |\n| y | y.md |\n",
        encoding="utf-8",
    )
    (tmp_path / "x.md").write_text("# X\n\n## Method\n\nX evidence.", encoding="utf-8")
    (tmp_path / "y.md").write_text("# Y\n\n## Method\n\nY evidence.", encoding="utf-8")
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
[models.reranker]
backend = "lexical"
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    database = CorpusDatabase(config.paths.database)
    first = CorpusIngestor(config, database).ingest(embeddings=False)
    assert first.indexed == 2
    prior_y = database.find_paper_by_source("y.md")
    real_hash = ingest_module.sha256_file

    def fail_y(path: Path) -> str:
        if Path(path).name == "y.md":
            raise OSError("simulated hash failure")
        return real_hash(path)

    monkeypatch.setattr(ingest_module, "sha256_file", fail_y)
    second = CorpusIngestor(config, database).ingest(embeddings=False)

    assert second.found == 2
    assert second.failed == 1
    assert second.skipped == 1
    assert any(failure.stage == "source_hash" for failure in second.failures)
    assert prior_y is not None
    assert database.find_paper_by_source("y.md").id == prior_y.id  # type: ignore[union-attr]


@pytest.mark.parametrize("failure_mode", ["discovery", "hash"])
def test_incomplete_snapshot_preserves_unobserved_duplicate_alias_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    source_root = tmp_path / "pdfs"
    blocked = source_root / "blocked"
    blocked.mkdir(parents=True)
    source = source_root / "a.pdf"
    hidden_alias = blocked / "x.pdf"
    source.write_bytes(b"%PDF-shared-old")
    hidden_alias.write_bytes(source.read_bytes())
    (tmp_path / "INDEX.md").write_text("# Corpus\n", encoding="utf-8")
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[paths]
pdf_roots = ["pdfs"]
database = "data/papers.db"
human_index = "INDEX.md"
exclude_dirs = ["data", "papers"]
[models.embedding]
backend = "hash"
[models.reranker]
backend = "lexical"
""",
        encoding="utf-8",
    )
    config = load_config(config_path)

    class Extractor:
        def extract(self, pdf_path: Path) -> ExtractedDocument:
            marker = pdf_path.read_bytes().decode("latin-1")
            return ExtractedDocument(
                markdown=f"# Shared Paper\n\n## Method\n\nEvidence from {marker}.",
                backend="fake",
                title="Shared Paper",
            )

    database = CorpusDatabase(config.paths.database)
    first = CorpusIngestor(config, database, extractor=Extractor()).ingest(embeddings=False)
    assert first.indexed == 1
    old_owner = database.find_paper_by_source("pdfs/blocked/x.pdf")
    assert old_owner is not None
    source.write_bytes(b"%PDF-new-a")

    if failure_mode == "discovery":
        monkeypatch.setattr(
            ingest_module,
            "_discover_pdfs_with_errors",
            lambda config: ([source], [(blocked, "permission denied")]),
        )
    else:
        real_hash = ingest_module.sha256_file

        def fail_hidden(path: Path) -> str:
            if Path(path) == hidden_alias:
                raise OSError("simulated hash failure")
            return real_hash(path)

        monkeypatch.setattr(ingest_module, "sha256_file", fail_hidden)

    report = CorpusIngestor(config, database, extractor=Extractor()).ingest(embeddings=False)

    preserved = database.find_paper_by_source("pdfs/blocked/x.pdf")
    changed = database.find_paper_by_source("pdfs/a.pdf")
    assert report.failed == 1
    assert any("retirement was disabled" in warning for warning in report.warnings)
    assert preserved is not None and preserved.id == old_owner.id
    assert preserved.content_hash == old_owner.content_hash
    assert changed is not None and changed.id != old_owner.id
    assert database.counts()["papers"] == 2


def test_pattern_matching_a_duplicate_alias_indexes_its_deduplicated_group(
    tmp_path: Path,
) -> None:
    (tmp_path / "INDEX.md").write_text(
        "| name | file |\n|---|---|\n| x | x.md |\n| y | y.md |\n",
        encoding="utf-8",
    )
    content = "# Shared\n\n## Method\n\nShared evidence."
    (tmp_path / "x.md").write_text(content, encoding="utf-8")
    (tmp_path / "y.md").write_text(content, encoding="utf-8")
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
[models.reranker]
backend = "lexical"
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    database = CorpusDatabase(config.paths.database)

    report = CorpusIngestor(config, database).ingest(
        patterns=("y.md",),
        embeddings=False,
    )

    assert report.indexed == 1
    assert database.counts()["papers"] == 1
    assert database.find_paper_by_source("y.md") is not None


def test_markdown_paper_linked_only_from_a_bullet_is_discovered(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.md").write_text(
        "# Handwritten Paper\n\n## Method\n\nCanonical evidence.",
        encoding="utf-8",
    )
    (tmp_path / "INDEX.md").write_text(
        "- Important source: [Paper](docs/note.md)\n",
        encoding="utf-8",
    )
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
[models.reranker]
backend = "lexical"
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    database = CorpusDatabase(config.paths.database)

    report = CorpusIngestor(config, database).ingest(embeddings=False)

    assert report.indexed == 1
    paper = database.find_paper_by_source("docs/note.md")
    assert paper is not None
    entry = database.get_curated_entries()[0]
    assert entry.linked_paper_ids == [paper.id]


def test_incomplete_subdirectory_scan_disables_retirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "pdfs"
    source_root.mkdir()
    paper_path = source_root / "paper.pdf"
    paper_path.write_bytes(b"%PDF-paper")
    (tmp_path / "INDEX.md").write_text(
        "| name | file |\n|---|---|\n| paper | paper.pdf |\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[paths]
pdf_roots = ["pdfs"]
database = "data/papers.db"
human_index = "INDEX.md"
exclude_dirs = ["data", "papers"]
[models.embedding]
backend = "hash"
[models.reranker]
backend = "lexical"
""",
        encoding="utf-8",
    )
    config = load_config(config_path)

    class Extractor:
        def extract(self, source: Path) -> ExtractedDocument:
            return ExtractedDocument(
                markdown="# Paper\n\n## Method\n\nEvidence.",
                backend="fake",
                title="Paper",
            )

    database = CorpusDatabase(config.paths.database)
    CorpusIngestor(config, database, extractor=Extractor()).ingest(embeddings=False)
    prior = database.list_papers()[0]
    monkeypatch.setattr(
        ingest_module,
        "_discover_pdfs_with_errors",
        lambda config: ([], [(source_root / "unreadable", "permission denied")]),
    )

    report = CorpusIngestor(config, database, extractor=Extractor()).ingest(embeddings=False)

    assert report.failed == 1
    assert any("retirement was disabled" in warning for warning in report.warnings)
    assert database.get_paper(prior.id) is not None


def test_truncated_generated_card_is_repaired_on_unchanged_ingest(tmp_path: Path) -> None:
    config = _config(tmp_path)
    database = CorpusDatabase(config.paths.database)
    CorpusIngestor(config, database).ingest(embeddings=False)
    card_path = config.paths.cards / "alpha-paper.json"
    card_path.write_text("{", encoding="utf-8")

    CorpusIngestor(config, database).ingest(embeddings=False)

    repaired = json.loads(card_path.read_text(encoding="utf-8"))
    assert repaired["generated_by"] == "papers-mcp"
    assert repaired["paper_id"] == "alpha-paper"


def test_normal_ingest_repairs_incomplete_hierarchy_and_fts_mirrors(tmp_path: Path) -> None:
    config = _config(tmp_path)
    database = CorpusDatabase(config.paths.database)
    CorpusIngestor(config, database).ingest(embeddings=False)
    alpha = database.get_paper("alpha-paper")
    assert alpha is not None
    chunk = database.get_chunks(alpha.id)[0]
    database.connection.execute("DELETE FROM chunks WHERE id = ?", (chunk.id,))
    assert not database.paper_hierarchy_complete(alpha.id)

    hierarchy_repair = CorpusIngestor(config, database).ingest(embeddings=False)

    assert hierarchy_repair.indexed >= 1
    assert database.paper_hierarchy_complete(alpha.id)
    repaired_chunk = database.get_chunks(alpha.id)[0]
    database.connection.execute(
        "DELETE FROM chunks_fts WHERE rowid = ?",
        (repaired_chunk.id,),
    )
    assert not database.fts_complete()

    fts_repair = CorpusIngestor(config, database).ingest(embeddings=False)

    assert database.fts_complete()
    assert any("Rebuilt incomplete SQLite FTS" in warning for warning in fts_repair.warnings)


def test_transient_hash_fallback_does_not_downgrade_existing_production_corpus(
    tmp_path: Path,
) -> None:
    base_config = _config(tmp_path)
    config = replace(
        base_config,
        embedding=replace(
            base_config.embedding,
            backend="sentence_transformers",
            model="fake-production-model",
            revision="a" * 40,
        ),
    )
    database = CorpusDatabase(config.paths.database)

    class StableProductionEmbedding:
        def __init__(self) -> None:
            self.config = config.embedding
            self.active_backend = "sentence_transformers"
            self.is_loaded = True

        def embed_documents(self, texts):
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

        def embed_query(self, text):
            return [1.0, 0.0, 0.0, 0.0]

    CorpusIngestor(
        config,
        database,
        embedding=StableProductionEmbedding(),
    ).ingest()
    assert database.counts()["papers"] == 2

    (tmp_path / "broken.md").write_text(
        "# Recovered Paper\n\n## Method\n\nRecovered evidence.",
        encoding="utf-8",
    )

    class DegradedEmbedding:
        def __init__(self) -> None:
            self.config = config.embedding
            self.active_backend = "sentence_transformers"
            self.is_loaded = True

        def embed_documents(self, texts):
            self.active_backend = "hash"
            return [[1.0] + [0.0] * 7 for _ in texts]

        def embed_query(self, text):
            return [1.0] + [0.0] * 7

    report = CorpusIngestor(
        config,
        database,
        embedding=DegradedEmbedding(),
    ).ingest()

    assert any(failure.stage == "embedding_consistency" for failure in report.failures)
    paper_sets = database.connection.execute(
        "SELECT paper_id, model_name FROM embeddings WHERE kind = 'paper' ORDER BY paper_id"
    ).fetchall()
    assert sum(row["model_name"] == "fake-production-model" for row in paper_sets) == 2
    assert sum(str(row["model_name"]).startswith("hash-v1:") for row in paper_sets) == 0
    assert database.counts()["papers"] == 2

    retry = CorpusIngestor(
        config,
        database,
        embedding=DegradedEmbedding(),
    ).ingest()
    assert any(failure.stage == "embedding_consistency" for failure in retry.failures)
    retry_sets = database.connection.execute(
        "SELECT paper_id, model_name FROM embeddings WHERE kind = 'paper' ORDER BY paper_id"
    ).fetchall()
    assert sum(row["model_name"] == "fake-production-model" for row in retry_sets) == 2
    assert sum(str(row["model_name"]).startswith("hash-v1:") for row in retry_sets) == 0
    assert database.counts()["papers"] == 2


def test_changed_source_and_missing_state_survive_transient_hash_fallback(
    tmp_path: Path,
) -> None:
    base_config = _config(tmp_path)
    config = replace(
        base_config,
        embedding=replace(
            base_config.embedding,
            backend="sentence_transformers",
            model="fake-production-model",
            revision="a" * 40,
        ),
    )
    database = CorpusDatabase(config.paths.database)

    class StableProductionEmbedding:
        def __init__(self) -> None:
            self.config = config.embedding
            self.active_backend = "sentence_transformers"
            self.is_loaded = True

        def embed_documents(self, texts):
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

        def embed_query(self, text):
            return [1.0, 0.0, 0.0, 0.0]

    CorpusIngestor(
        config,
        database,
        embedding=StableProductionEmbedding(),
    ).ingest()
    before = database.get_paper("alpha-paper")
    assert before is not None
    before_sections = [
        (section.heading_path, section.text) for section in database.get_sections(before.id)
    ]
    database.delete_meta(f"paper_embedding_state:{before.id}")
    (tmp_path / "alpha.md").write_text(
        "# Changed Title\n\n## Changed Method\n\nContent that must not replace the last good snapshot.",
        encoding="utf-8",
    )

    class DegradedEmbedding:
        def __init__(self) -> None:
            self.config = config.embedding
            self.active_backend = "sentence_transformers"
            self.is_loaded = True

        def embed_documents(self, texts):
            self.active_backend = "hash"
            return [[1.0] + [0.0] * 7 for _ in texts]

        def embed_query(self, text):
            return [1.0] + [0.0] * 7

    report = CorpusIngestor(
        config,
        database,
        embedding=DegradedEmbedding(),
    ).ingest()

    after = database.get_paper(before.id)
    assert after is not None
    assert after.content_hash == before.content_hash
    assert after.title == before.title
    assert [
        (section.heading_path, section.text) for section in database.get_sections(after.id)
    ] == (before_sections)
    assert any(
        failure.stage == "ingest" and failure.source_path == "alpha.md"
        for failure in report.failures
    )
    paper_sets = database.connection.execute(
        "SELECT model_name FROM embeddings WHERE kind = 'paper' AND paper_id = ?",
        (before.id,),
    ).fetchall()
    assert [row["model_name"] for row in paper_sets] == ["fake-production-model"]


def test_changed_curated_index_survives_transient_hash_fallback_atomically(
    tmp_path: Path,
) -> None:
    base_config = _config(tmp_path)
    config = replace(
        base_config,
        embedding=replace(
            base_config.embedding,
            backend="sentence_transformers",
            model="fake-production-model",
            revision="a" * 40,
        ),
    )
    database = CorpusDatabase(config.paths.database)

    class StableProductionEmbedding:
        def __init__(self) -> None:
            self.config = config.embedding
            self.active_backend = "sentence_transformers"
            self.is_loaded = True

        def embed_documents(self, texts):
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

        def embed_query(self, text):
            return [1.0, 0.0, 0.0, 0.0]

    first = CorpusIngestor(
        config,
        database,
        embedding=StableProductionEmbedding(),
    ).ingest()
    assert first.indexed == 2
    before_entries = [
        (entry.heading_path, entry.text, entry.linked_paper_ids)
        for entry in database.get_curated_entries()
    ]
    before_index_hash = database.get_meta("curated_index_hash")
    with config.paths.human_index.open("a", encoding="utf-8") as index:
        index.write("\n- NEW CURATED TERM that must wait for production embeddings.\n")

    class DegradedEmbedding:
        def __init__(self) -> None:
            self.config = config.embedding
            self.active_backend = "sentence_transformers"
            self.is_loaded = True

        def embed_documents(self, texts):
            self.active_backend = "hash"
            return [[1.0] + [0.0] * 7 for _ in texts]

        def embed_query(self, text):
            return [1.0] + [0.0] * 7

    report = CorpusIngestor(
        config,
        database,
        embedding=DegradedEmbedding(),
    ).ingest()

    assert any(failure.stage == "curated_index" for failure in report.failures)
    assert [
        (entry.heading_path, entry.text, entry.linked_paper_ids)
        for entry in database.get_curated_entries()
    ] == before_entries
    assert database.get_meta("curated_index_hash") == before_index_hash
    curated_models = database.connection.execute(
        "SELECT DISTINCT model_name FROM embeddings WHERE kind = 'curated'"
    ).fetchall()
    assert [row["model_name"] for row in curated_models] == ["fake-production-model"]
