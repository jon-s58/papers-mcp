from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from papers_mcp import cli
from papers_mcp.models import IngestFailure, IngestReport, SearchResult


def fake_config(tmp_path: Path) -> Any:
    return SimpleNamespace(
        config_path=tmp_path / "chosen.toml",
        log_level="WARNING",
        paths=SimpleNamespace(root=tmp_path, database=tmp_path / "papers.db"),
    )


class ContextValue:
    def __init__(self, value: Any) -> None:
        self.value = value

    def __enter__(self) -> Any:
        return self.value

    def __exit__(self, *_: object) -> None:
        return None


class FakeCorpus:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def __enter__(self) -> FakeCorpus:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def search(self, query: str, **kwargs: Any) -> list[SearchResult]:
        self.calls.append(("search", query, kwargs))
        return [fixture_result()]

    def research_search(self, query: str, **kwargs: Any) -> list[SearchResult]:
        self.calls.append(("research", query, kwargs))
        return [fixture_result()]

    def outline(self, paper_id: str) -> dict[str, Any]:
        self.calls.append(("outline", paper_id))
        return {
            "paper_id": paper_id,
            "title": "Patch Networks",
            "authors": ["Jane Geometer"],
            "year": 2002,
            "doi": None,
            "abstract": "An abstract.",
            "section_tree": [
                {
                    "section_id": 7,
                    "heading": "Vertex Enclosure",
                    "page_start": 6,
                    "page_end": 8,
                    "children": [],
                }
            ],
        }

    def read_section(self, paper_id: str, section_id: int, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("read", paper_id, section_id, kwargs))
        return {
            "paper_id": paper_id,
            "title": "Patch Networks",
            "section_id": section_id,
            "section": "4.2 Vertex Enclosure",
            "page_start": 6,
            "page_end": 8,
            "text": "The complete mathematical section.",
            "next_offset": 900,
        }


def fixture_result() -> SearchResult:
    return SearchResult(
        result_id="r3:chunk:19",
        kind="chunk",
        paper_id="peters-2002-patches",
        section_id=7,
        chunk_id=19,
        title="Patch Networks",
        authors=["Jane Geometer"],
        year=2002,
        section_path="4.2 Vertex Enclosure",
        page_start=6,
        page_end=8,
        source_kind="pdf",
        rank=1,
        score=0.93,
        snippet="The vertex enclosure condition constrains tangent data.",
        next_actions={"paper_outline": {"paper_id": "peters-2002-patches"}},
    )


def install_config(monkeypatch: pytest.MonkeyPatch, config: Any) -> list[Any]:
    received: list[Any] = []

    def load(path: Any = None) -> Any:
        received.append(path)
        return config

    monkeypatch.setattr(cli, "load_config", load)
    return received


def test_doctor_accepts_global_config_and_prints_clean_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = fake_config(tmp_path)
    received = install_config(monkeypatch, config)
    deep_values: list[bool] = []

    def doctor(_config: Any, *, deep: bool) -> dict[str, Any]:
        deep_values.append(deep)
        return {"ok": True, "sqlite_fts5": {"ok": True}, "corpus": {}}

    monkeypatch.setattr(cli, "run_doctor", doctor)
    assert cli.main(["--config", "custom.toml", "doctor", "--deep", "--json"]) == 0

    captured = capsys.readouterr()
    assert json.loads(captured.out)["ok"] is True
    assert received == ["custom.toml"]
    assert deep_values == [True]


def test_doctor_not_ready_has_nonzero_exit_and_human_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = fake_config(tmp_path)
    install_config(monkeypatch, config)
    monkeypatch.setattr(
        cli,
        "run_doctor",
        lambda *_args, **_kwargs: {
            "ok": False,
            "sqlite_fts5": {"ok": True, "sqlite_version": "3.42"},
            "device": {"device": "mps", "machine": "arm64"},
            "corpus": {"initialized": False},
            "models": {},
        },
    )
    assert cli.main(["doctor"]) == 1
    output = capsys.readouterr().out
    assert "Doctor: NOT READY" in output
    assert "SQLite FTS5: ok (3.42)" in output
    assert "Device: mps on arm64" in output


def test_doctor_environment_can_be_healthy_while_search_is_not_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = fake_config(tmp_path)
    install_config(monkeypatch, config)
    monkeypatch.setattr(
        cli,
        "run_doctor",
        lambda *_args, **_kwargs: {
            "ok": True,
            "ready": False,
            "sqlite_fts5": {"ok": True},
            "corpus": {"initialized": False},
            "models": {},
        },
    )

    assert cli.main(["doctor"]) == 1
    output = capsys.readouterr().out
    assert "Doctor: ENVIRONMENT OK" in output
    assert "Search: NOT READY" in output


def test_ingest_forwards_all_flags_and_serializes_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = fake_config(tmp_path)
    install_config(monkeypatch, config)
    database = object()
    database_paths: list[Any] = []
    received: list[tuple[Any, Any, dict[str, Any]]] = []

    def database_factory(path: Any) -> ContextValue:
        database_paths.append(path)
        return ContextValue(database)

    def ingest(config_value: Any, database_value: Any, **kwargs: Any) -> IngestReport:
        received.append((config_value, database_value, kwargs))
        return IngestReport(found=3, indexed=2, skipped=1)

    monkeypatch.setattr(cli, "CorpusDatabase", database_factory)
    monkeypatch.setattr(cli, "ingest_corpus", ingest)
    code = cli.main(
        [
            "ingest",
            "--force",
            "--limit",
            "3",
            "--pattern",
            "*g1*",
            "--pattern",
            "*.md",
            "--no-embeddings",
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["indexed"] == 2
    assert database_paths == [config.paths.database]
    assert received == [
        (
            config,
            database,
            {
                "force": True,
                "rechunk": False,
                "limit": 3,
                "patterns": ("*g1*", "*.md"),
                "embeddings": False,
            },
        )
    ]

    received.clear()
    assert cli.main(["ingest", "--rechunk", "--json"]) == 0
    capsys.readouterr()
    assert received == [
        (
            config,
            database,
            {
                "force": False,
                "rechunk": True,
                "limit": None,
                "patterns": (),
                "embeddings": True,
            },
        )
    ]


def test_ingest_partial_failure_is_reported_and_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = fake_config(tmp_path)
    install_config(monkeypatch, config)
    monkeypatch.setattr(cli, "CorpusDatabase", lambda _path: ContextValue(object()))
    monkeypatch.setattr(
        cli,
        "ingest_corpus",
        lambda *_args, **_kwargs: IngestReport(
            found=1,
            failed=1,
            failures=[IngestFailure("bad.pdf", "extract", "malformed")],
        ),
    )
    assert cli.main(["ingest"]) == 1
    output = capsys.readouterr().out
    assert "failed 1" in output
    assert "bad.pdf [extract]: malformed" in output


def test_search_human_output_is_concise_and_sourced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = fake_config(tmp_path)
    install_config(monkeypatch, config)
    corpus = FakeCorpus()
    monkeypatch.setattr(cli, "ResearchCorpus", lambda _config: corpus)
    code = cli.main(
        [
            "search",
            "G1 compatibility",
            "--mode",
            "exact",
            "--top-k",
            "4",
            "--pipeline",
            "hybrid",
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "Patch Networks (2002) [peters-2002-patches]" in output
    assert "4.2 Vertex Enclosure | pages 6-8" in output
    assert "section_id 7" in output
    assert "r3:chunk:19" in output
    assert corpus.calls == [
        (
            "search",
            "G1 compatibility",
            {"mode": "exact", "top_k": 4, "pipeline": "hybrid"},
        )
    ]


def test_human_output_labels_curated_paper_routing_without_source_pages(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = SearchResult(
        result_id="r4:paper:alpha",
        kind="paper",
        paper_id="alpha",
        section_id=None,
        chunk_id=None,
        title="Alpha Paper",
        authors=[],
        year=2020,
        section_path="Geometry > Continuity",
        page_start=None,
        page_end=None,
        source_kind="pdf",
        rank=1,
        score=0.8,
        snippet="The paper abstract.",
        routing_note="Expert vocabulary that is not assigned to a source page.",
    )

    cli._print_results([result])

    output = capsys.readouterr().out
    assert "Curated routing note (INDEX.md)" in output
    assert "Next: outline alpha" in output
    assert "pages unknown" in output


def test_research_json_contains_only_result_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = fake_config(tmp_path)
    install_config(monkeypatch, config)
    corpus = FakeCorpus()
    monkeypatch.setattr(cli, "ResearchCorpus", lambda _config: corpus)
    assert cli.main(["research", "alternative patch methods", "--top-k", "6", "--json"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload[0]["paper_id"] == "peters-2002-patches"
    assert "INFO" not in captured.out
    assert corpus.calls[0] == (
        "research",
        "alternative patch methods",
        {"top_k": 6, "pipeline": "hybrid_rerank"},
    )


def test_outline_and_read_commands_forward_stable_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = fake_config(tmp_path)
    install_config(monkeypatch, config)
    corpus = FakeCorpus()
    monkeypatch.setattr(cli, "ResearchCorpus", lambda _config: corpus)

    assert cli.main(["outline", "paper-1"]) == 0
    outline_output = capsys.readouterr().out
    assert "Patch Networks [paper-1]" in outline_output
    assert "[7] Vertex Enclosure (pages 6-8)" in outline_output

    assert cli.main(["read", "paper-1", "7", "--offset", "100", "--max-tokens", "800"]) == 0
    read_output = capsys.readouterr().out
    assert "Section 7: 4.2 Vertex Enclosure (pages 6-8)" in read_output
    assert "complete mathematical section" in read_output
    assert "--offset 900" in read_output
    assert corpus.calls[-1] == (
        "read",
        "paper-1",
        7,
        {"offset": 100, "max_tokens": 800},
    )


def test_evaluate_resolves_default_path_and_forwards_system(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = fake_config(tmp_path)
    install_config(monkeypatch, config)
    corpus = FakeCorpus()
    monkeypatch.setattr(cli, "ResearchCorpus", lambda _config: corpus)
    query_sentinel = [object()]
    paths: list[Path] = []

    def load_queries(path: Path) -> list[object]:
        paths.append(path)
        return query_sentinel

    def run_evaluation(
        corpus_value: Any, queries: Any, *, system: str, top_k: int
    ) -> dict[str, Any]:
        assert corpus_value is corpus
        assert queries is query_sentinel
        return {
            "system": system,
            "query_count": 1,
            "aggregate": {"mrr": 1.0},
            "top_k": top_k,
        }

    monkeypatch.setattr(cli, "load_evaluation_queries", load_queries)
    monkeypatch.setattr(cli, "evaluate", run_evaluation)
    assert cli.main(["evaluate", "--system", "bm25", "--top-k", "5", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "aggregate": {"mrr": 1.0},
        "query_count": 1,
        "system": "bm25",
        "top_k": 5,
    }
    assert paths == [tmp_path / "eval/queries.yaml"]


def test_serve_mcp_runs_server_without_polluting_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = fake_config(tmp_path)
    install_config(monkeypatch, config)
    calls: list[Any] = []

    class Server:
        def run(self) -> None:
            calls.append("run")

    def build(**kwargs: Any) -> Server:
        calls.append(kwargs)
        return Server()

    monkeypatch.setattr(cli, "build_mcp_server", build)
    monkeypatch.setattr(
        cli,
        "ResearchCorpus",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must not construct corpus")),
    )
    assert cli.main(["serve-mcp"]) == 0
    assert capsys.readouterr().out == ""
    assert calls == [{"config_path": config.config_path}, "run"]


def test_runtime_errors_return_one_and_are_logged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = fake_config(tmp_path)
    install_config(monkeypatch, config)

    class BrokenCorpus(FakeCorpus):
        def search(self, query: str, **kwargs: Any) -> list[SearchResult]:
            raise RuntimeError("database unavailable")

    monkeypatch.setattr(cli, "ResearchCorpus", lambda _config: BrokenCorpus())
    with caplog.at_level(logging.ERROR, logger="papers_mcp"):
        assert cli.main(["search", "query", "--json"]) == 1
    assert capsys.readouterr().out == ""
    assert "database unavailable" in caplog.text


def test_parser_rejects_invalid_positive_values() -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["search", "query", "--top-k", "0"])
    assert error.value.code == 2
