from pathlib import Path
from types import SimpleNamespace

from papers_mcp.config import load_config
from papers_mcp.database import CorpusDatabase
from papers_mcp.embeddings import HashEmbeddingProvider
from papers_mcp.models import Chunk, Paper, Section
from papers_mcp.query_expansion import NoOpQueryExpansionProvider
from papers_mcp.reranker import LexicalReranker
from papers_mcp.service import ResearchCorpus
from papers_mcp.vectors import model_fingerprint


def fixture_config(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
        """
[paths]
pdf_roots = ["."]
database = "data/test.db"
human_index = "INDEX.md"
[models.embedding]
backend = "hash"
fallback_dimensions = 64
allow_explicit_hash_fallback = true
[models.reranker]
backend = "lexical"
allow_explicit_lexical_fallback = true
[chunks]
target_tokens = 10
min_tokens = 2
max_tokens = 30
""",
        encoding="utf-8",
    )
    (tmp_path / "INDEX.md").write_text("# Index\n", encoding="utf-8")
    return load_config(path)


def populated_service(tmp_path: Path) -> ResearchCorpus:
    config = fixture_config(tmp_path)
    database = CorpusDatabase(config.paths.database)
    embedding = HashEmbeddingProvider(64)
    for paper_id, title, body in [
        ("peters-1989", "Vertex Enclosure", "G1 vertex enclosure tangent compatibility."),
        ("blidia-2020", "G1 Surface Fitting", "Tangent compatible spline fitting."),
    ]:
        paper = Paper(
            id=paper_id,
            title=title,
            abstract=body,
            source_path=f"{paper_id}.pdf",
            content_hash=paper_id,
        )
        sections = [
            Section(
                paper_id=paper_id,
                heading="Method",
                heading_path="Method",
                text=body,
                level=1,
                section_order=0,
            ),
            Section(
                paper_id=paper_id,
                heading="Constraint",
                heading_path="Method > Constraint",
                text=body,
                level=2,
                section_order=1,
                parent_index=0,
            ),
        ]
        chunks = [
            Chunk(
                paper_id=paper_id,
                heading_path="Method > Constraint",
                text=body,
                token_count=5,
                section_index=1,
            )
        ]
        database.replace_paper(paper, sections, chunks)
        texts = [f"{title} {body}", *[item.text for item in sections], body]
        vectors = embedding.embed_documents(texts)
        fingerprint = model_fingerprint("hash-v1:64", provider="hash", dimensions=64)
        database.replace_embeddings_for_paper(
            paper_id,
            [
                ("paper", paper_id, vectors[0]),
                ("section", sections[0].id, vectors[1]),
                ("section", sections[1].id, vectors[2]),
                ("chunk", chunks[0].id, vectors[3]),
            ],
            model_name="hash-v1:64",
            model_fingerprint=fingerprint,
        )
    return ResearchCorpus(
        config,
        database=database,
        embedding=embedding,
        reranker=LexicalReranker(),
        expansion=NoOpQueryExpansionProvider(),
    )


def test_search_expand_outline_read_and_related(tmp_path: Path) -> None:
    corpus = populated_service(tmp_path)
    results = corpus.search("vertex enclosure", top_k=3)
    assert results and results[0].paper_id == "peters-1989"
    context = corpus.expand_context(results[0].result_id)
    assert "vertex enclosure" in context["text"].casefold()
    outline = corpus.paper_outline("peters-1989")
    section_id = outline["sections"][0]["section_id"]
    assert corpus.read_section("peters-1989", section_id)["text"]
    assert corpus.find_in_paper("peters-1989", "tangent")
    related = corpus.related_papers("peters-1989", top_k=2)
    assert related and related[0]["paper_id"] == "blidia-2020"
    assert related[0]["next_actions"]["find_in_paper"]["query"]


def test_exact_title_match_returns_paper_not_unrelated_child_context(tmp_path: Path) -> None:
    corpus = populated_service(tmp_path)
    paper = Paper(
        id="pearl-2012",
        title="PEARL: A Global Approach",
        abstract="A graph-cut energy for assigning geometric labels.",
        source_path="pearl.pdf",
        content_hash="pearl",
    )
    sections = [
        Section(
            paper_id=paper.id,
            heading="Optimization",
            heading_path="Optimization",
            text="We optimize the labeling energy globally.",
            level=1,
            section_order=0,
        )
    ]
    chunks = [
        Chunk(
            paper_id=paper.id,
            heading_path="Optimization",
            text="We optimize the labeling energy globally.",
            token_count=6,
            section_index=0,
        )
    ]
    corpus.database.replace_paper(paper, sections, chunks)

    results = corpus.search("PEARL", mode="exact", top_k=10, pipeline="hybrid")

    assert [(result.kind, result.paper_id) for result in results] == [("paper", paper.id)]
    assert "expand_context" not in results[0].next_actions
    assert "paper_outline" in results[0].next_actions


def test_discovery_results_advertise_source_faithful_follow_up_workflow(
    tmp_path: Path,
) -> None:
    corpus = populated_service(tmp_path)

    results = corpus.research_search("vertex enclosure", top_k=10, pipeline="bm25")

    assert results
    assert any("expand_context" in result.next_actions for result in results)
    for result in results:
        if result.section_id is not None:
            expanded = corpus.expand_context(result.result_id)
            assert expanded["paper_id"] == result.paper_id
            assert expanded["text"]
        else:
            assert result.kind == "paper"
            assert "paper_outline" in result.next_actions
            assert "find_in_paper" in result.next_actions


def test_related_papers_refreshes_model_selector_after_lazy_fallback(
    tmp_path: Path,
) -> None:
    corpus = populated_service(tmp_path)

    class LazyFallbackEmbedding:
        def __init__(self) -> None:
            self.active_backend = "sentence_transformers"
            self.config = SimpleNamespace(model="Qwen/Qwen3-Embedding-8B")
            self.dimensions = 64
            self.delegate = HashEmbeddingProvider(self.dimensions)

        def embed_query(self, text: str) -> list[float]:
            self.active_backend = "hash"
            return self.delegate.embed_query(text)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            self.active_backend = "hash"
            return self.delegate.embed_documents(texts)

    lazy = LazyFallbackEmbedding()
    corpus.embedding = lazy
    corpus.retrieval.embedding = lazy

    related = corpus.related_papers("peters-1989", top_k=2)

    assert lazy.active_backend == "hash"
    assert related and related[0]["paper_id"] == "blidia-2020"


def test_expand_context_keeps_irregular_numbered_heading_as_logical_section(
    tmp_path: Path,
) -> None:
    config = fixture_config(tmp_path)
    database = CorpusDatabase(config.paths.database)
    paper = Paper(
        id="scan",
        title="Scanned Report",
        source_path="scan.pdf",
        content_hash="scan",
    )
    sections = [
        Section(
            paper_id=paper.id,
            heading="Document Title",
            heading_path="Document Title",
            text="",
            level=2,
            section_order=0,
        ),
        Section(
            paper_id=paper.id,
            heading="1. Introduction",
            heading_path="Document Title > 1. Introduction",
            text="introduction evidence",
            level=5,
            section_order=1,
            parent_index=0,
        ),
        Section(
            paper_id=paper.id,
            heading="2. Later Method",
            heading_path="Document Title > 2. Later Method",
            text="unrelated later section",
            level=3,
            section_order=2,
            parent_index=0,
        ),
    ]
    chunks = [
        Chunk(
            paper_id=paper.id,
            heading_path=sections[1].heading_path,
            text=sections[1].text,
            token_count=2,
            section_index=1,
        )
    ]
    database.replace_paper(paper, sections, chunks)
    corpus = ResearchCorpus(config, database=database)

    context = corpus.expand_context(f"r{database.get_revision()}:section:{sections[1].id}")

    assert "introduction evidence" in context["text"]
    assert "unrelated later section" not in context["text"]
    assert context["next_action"] is None


def test_section_expansion_climbs_flat_h2_numbered_subsection(tmp_path: Path) -> None:
    config = fixture_config(tmp_path)
    database = CorpusDatabase(config.paths.database)
    paper = Paper(
        id="flat-numbering",
        title="Flat OCR Headings",
        source_path="flat.pdf",
        content_hash="flat",
    )
    sections = [
        Section(
            paper_id=paper.id,
            heading="1 INTRODUCTION",
            heading_path="1 INTRODUCTION",
            text="",
            level=2,
            section_order=0,
        ),
        Section(
            paper_id=paper.id,
            heading="1.1 Motivation",
            heading_path="1 INTRODUCTION > 1.1 Motivation",
            text="motivating evidence",
            level=2,
            section_order=1,
            parent_index=0,
        ),
        Section(
            paper_id=paper.id,
            heading="1.2 Prior work",
            heading_path="1 INTRODUCTION > 1.2 Prior work",
            text="prior evidence",
            level=2,
            section_order=2,
            parent_index=0,
        ),
    ]
    chunks = [
        Chunk(
            paper_id=paper.id,
            heading_path=sections[1].heading_path,
            text=sections[1].text,
            token_count=2,
            section_index=1,
        )
    ]
    database.replace_paper(paper, sections, chunks)
    corpus = ResearchCorpus(config, database=database)
    result_id = f"r{database.get_revision()}:chunk:{chunks[0].id}"

    subsection = corpus.expand_context(result_id, level="subsection")
    section = corpus.expand_context(result_id, level="section")

    assert subsection["section_id"] == sections[1].id
    assert section["section_id"] == sections[0].id
    assert "motivating evidence" in section["text"]
    assert "prior evidence" in section["text"]


def test_section_expansion_climbs_unnumbered_h2_under_methods(tmp_path: Path) -> None:
    config = fixture_config(tmp_path)
    database = CorpusDatabase(config.paths.database)
    paper = Paper(id="methods", title="Methods", source_path="m.pdf", content_hash="m")
    sections = [
        Section(
            paper_id=paper.id,
            heading="Methods",
            heading_path="Methods",
            text="",
            level=1,
            section_order=0,
        ),
        Section(
            paper_id=paper.id,
            heading="Objective",
            heading_path="Methods > Objective",
            text="objective evidence",
            level=2,
            section_order=1,
            parent_index=0,
        ),
        Section(
            paper_id=paper.id,
            heading="Solver",
            heading_path="Methods > Solver",
            text="solver evidence",
            level=2,
            section_order=2,
            parent_index=0,
        ),
    ]
    chunks = [
        Chunk(
            paper_id=paper.id,
            heading_path=sections[1].heading_path,
            text=sections[1].text,
            token_count=2,
            section_index=1,
        )
    ]
    database.replace_paper(paper, sections, chunks)
    corpus = ResearchCorpus(config, database=database)
    result_id = f"r{database.get_revision()}:chunk:{chunks[0].id}"

    subsection = corpus.expand_context(result_id, level="subsection")
    section = corpus.expand_context(result_id, level="section")

    assert subsection["section_id"] == sections[1].id
    assert section["section_id"] == sections[0].id
    assert "objective evidence" in section["text"]
    assert "solver evidence" in section["text"]


def test_query_vector_cache_invalidates_when_lazy_backend_changes(tmp_path: Path) -> None:
    corpus = populated_service(tmp_path)

    class TransitionEmbedding:
        def __init__(self) -> None:
            self.active_backend = "sentence_transformers"
            self.config = SimpleNamespace(model="fake-production")
            self.delegate = HashEmbeddingProvider(64)
            self.calls: list[str] = []

        def embed_query(self, text: str) -> list[float]:
            self.calls.append(text)
            if len(self.calls) >= 2:
                self.active_backend = "hash"
            return self.delegate.embed_query(text)

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self.delegate.embed_documents(texts)

    embedding = TransitionEmbedding()
    corpus.embedding = embedding
    corpus.retrieval.embedding = embedding

    assert corpus.search("vertex enclosure", pipeline="dense") == []
    assert corpus.search("tangent compatible", pipeline="dense")
    assert corpus.search("vertex enclosure", pipeline="dense")
    assert embedding.calls == [
        "vertex enclosure",
        "tangent compatible",
        "vertex enclosure",
    ]
