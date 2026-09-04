from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .config import AppConfig, load_config
from .database import CorpusDatabase
from .doctor import run_doctor
from .evaluation import evaluate, load_evaluation_queries
from .ingest import ingest_corpus
from .mcp_server import build_mcp_server
from .service import ResearchCorpus

LOGGER = logging.getLogger("papers_mcp")
PIPELINES = ("bm25", "dense", "hybrid", "hybrid_rerank")
SEARCH_MODES = ("precision", "discovery", "exact")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Iterable):
        return [_jsonable(item) for item in value]
    return str(value)


def _print_json(value: Any) -> None:
    print(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True))


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger().setLevel(getattr(logging, level.upper(), logging.INFO))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="papers-mcp",
        description="Local academic retrieval and MCP service for the academic paper corpus.",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="configuration file (default: ./config.toml or PAPERS_MCP_CONFIG)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="check paths, dependencies, models, and database")
    doctor.add_argument("--deep", action="store_true", help="load providers for a deeper check")
    doctor.add_argument("--json", dest="json_output", action="store_true")

    ingest = subparsers.add_parser("ingest", help="incrementally extract and index the corpus")
    maintenance = ingest.add_mutually_exclusive_group()
    maintenance.add_argument("--force", action="store_true", help="reprocess unchanged sources")
    maintenance.add_argument(
        "--rechunk",
        action="store_true",
        help="rebuild changed chunk layouts from stored extraction without rerunning PDF extraction",
    )
    ingest.add_argument("--limit", type=_positive_int, help="process at most this many sources")
    ingest.add_argument(
        "--pattern",
        action="append",
        default=[],
        metavar="GLOB",
        help="include a source glob; repeat for multiple patterns",
    )
    ingest.add_argument(
        "--no-embeddings",
        action="store_true",
        help="extract and index lexical content without generating embeddings",
    )
    ingest.add_argument("--json", dest="json_output", action="store_true")

    search = subparsers.add_parser("search", help="retrieve sourced paper passages")
    search.add_argument("query", metavar="QUERY")
    search.add_argument("--mode", choices=SEARCH_MODES, default="precision")
    search.add_argument("--top-k", type=_positive_int, default=10)
    search.add_argument("--pipeline", choices=PIPELINES, default="hybrid_rerank")
    search.add_argument("--json", dest="json_output", action="store_true")

    research = subparsers.add_parser(
        "research", help="broader hierarchical discovery across papers and curated notes"
    )
    research.add_argument("query", metavar="QUERY")
    research.add_argument("--top-k", type=_positive_int, default=12)
    research.add_argument("--pipeline", choices=PIPELINES, default="hybrid_rerank")
    research.add_argument("--json", dest="json_output", action="store_true")

    outline = subparsers.add_parser("outline", help="show a paper's metadata and section tree")
    outline.add_argument("paper_id", metavar="PAPER_ID")
    outline.add_argument("--json", dest="json_output", action="store_true")

    read = subparsers.add_parser("read", help="read a section by stable paper and section IDs")
    read.add_argument("paper_id", metavar="PAPER_ID")
    read.add_argument("section_id", metavar="SECTION_ID", type=_positive_int)
    read.add_argument("--offset", type=_nonnegative_int, default=0)
    read.add_argument("--max-tokens", type=_positive_int, default=8000)
    read.add_argument("--json", dest="json_output", action="store_true")

    evaluation = subparsers.add_parser("evaluate", help="run the weak-label retrieval evaluation")
    evaluation.add_argument("--queries", default="eval/queries.yaml", metavar="PATH")
    evaluation.add_argument("--system", choices=PIPELINES, default="hybrid_rerank")
    evaluation.add_argument("--top-k", type=_positive_int, default=10)
    evaluation.add_argument("--json", dest="json_output", action="store_true")

    subparsers.add_parser("serve-mcp", help="run the seven-tool MCP server over stdio")
    return parser


def _page_label(row: Mapping[str, Any]) -> str:
    pages = row.get("pages")
    if isinstance(pages, Mapping):
        return str(pages.get("label", "unknown"))
    start = row.get("page_start")
    end = row.get("page_end")
    if start is None:
        return "unknown"
    if end is None or end == start:
        return str(start)
    return f"{start}-{end}"


def _print_results(results: Sequence[Any]) -> None:
    rows = [_jsonable(result) for result in results]
    if not rows:
        print("No results.")
        return
    for index, raw in enumerate(rows, start=1):
        row = raw if isinstance(raw, Mapping) else {"snippet": str(raw)}
        title = str(row.get("title") or "Untitled")
        year = f" ({row['year']})" if row.get("year") is not None else ""
        paper_id = str(row.get("paper_id") or "curated-index")
        print(f"{index}. {title}{year} [{paper_id}]")
        section = str(row.get("section_path") or row.get("section") or "paper level")
        section_id = row.get("section_id")
        section_id_text = f" | section_id {section_id}" if section_id is not None else ""
        score = row.get("score")
        score_text = f" | score {float(score):.4f}" if isinstance(score, (int, float)) else ""
        print(f"   Section: {section} | pages {_page_label(row)}{section_id_text}{score_text}")
        snippet = str(row.get("snippet") or row.get("text") or "").strip()
        if snippet:
            print(textwrap.indent(snippet, "   "))
        routing_note = str(row.get("routing_note") or "").strip()
        if routing_note:
            print("   Curated routing note (INDEX.md):")
            print(textwrap.indent(routing_note, "     "))
        if section_id is None and row.get("paper_id"):
            print(f"   Next: outline {row['paper_id']}, then read a sourced section.")
        if row.get("result_id"):
            print(f"   result_id: {row['result_id']}")
        if index != len(rows):
            print()


def _print_doctor(report: Mapping[str, Any]) -> None:
    print(f"Doctor: {'ENVIRONMENT OK' if report.get('ok') else 'NOT READY'}")
    if report.get("ready") is not None:
        print(f"Search: {'READY' if report.get('ready') else 'NOT READY'}")
    sqlite = report.get("sqlite_fts5", {})
    if isinstance(sqlite, Mapping):
        print(
            "SQLite FTS5: "
            f"{'ok' if sqlite.get('ok') else 'unavailable'}"
            + (f" ({sqlite.get('sqlite_version')})" if sqlite.get("sqlite_version") else "")
        )
    device = report.get("device", {})
    if isinstance(device, Mapping):
        print(
            f"Device: {device.get('device', 'unknown')}"
            + (f" on {device.get('machine')}" if device.get("machine") else "")
        )
    resources = report.get("resources", {})
    if isinstance(resources, Mapping):
        print(
            f"Memory caps: process {resources.get('max_process_memory_gb', 'unknown')} GB / "
            f"MPS allocator {resources.get('mps_memory_limit_gb', 'unknown')} GB"
        )
    corpus = report.get("corpus", {})
    if isinstance(corpus, Mapping):
        if corpus.get("initialized"):
            print(
                "Corpus: "
                f"{corpus.get('papers', 0)} papers / {corpus.get('sections', 0)} sections / "
                f"{corpus.get('chunks', 0)} chunks (revision {corpus.get('revision', 0)})"
            )
        else:
            print("Corpus: not initialized")
    models = report.get("models", {})
    if isinstance(models, Mapping):
        for role in ("embedding", "reranker"):
            model = models.get(role, {})
            if isinstance(model, Mapping):
                cached = "cached" if model.get("cached") else "not cached"
                print(f"{role.capitalize()}: {model.get('model', 'unknown')} ({cached})")
    deep = report.get("deep")
    if isinstance(deep, Mapping):
        for role in ("embedding", "reranker"):
            result = deep.get(role)
            if isinstance(result, Mapping):
                state = "ok" if result.get("ok") else "failed/degraded"
                print(f"Deep {role}: {state} ({result.get('backend', 'unknown')})")
    warnings = report.get("warnings", [])
    if isinstance(warnings, Sequence) and not isinstance(warnings, str):
        for warning in warnings:
            print(f"Warning: {warning}")


def _print_ingest(report: Any) -> None:
    value = _jsonable(report)
    row = value if isinstance(value, Mapping) else {}
    print(
        f"Found {row.get('found', 0)} | indexed {row.get('indexed', 0)} | "
        f"skipped {row.get('skipped', 0)} | duplicates {row.get('duplicates', 0)} | "
        f"failed {row.get('failed', 0)}"
    )
    failures = row.get("failures", [])
    if isinstance(failures, Sequence) and failures:
        print("Failures:")
        for failure in failures:
            if isinstance(failure, Mapping):
                print(
                    f"- {failure.get('source_path', 'unknown')} "
                    f"[{failure.get('stage', 'unknown')}]: {failure.get('error', '')}"
                )


def _print_outline(value: Mapping[str, Any]) -> None:
    print(f"{value.get('title', 'Untitled')} [{value.get('paper_id', 'unknown')}]")
    authors = value.get("authors", [])
    if isinstance(authors, Sequence) and not isinstance(authors, str) and authors:
        print("Authors: " + ", ".join(str(author) for author in authors))
    if value.get("year") is not None:
        print(f"Year: {value['year']}")
    if value.get("doi"):
        print(f"DOI: {value['doi']}")
    abstract = str(value.get("abstract") or "").strip()
    if abstract:
        print("\nAbstract\n" + abstract)
    print("\nSections")

    def render(nodes: Any, depth: int = 0) -> None:
        if not isinstance(nodes, Sequence) or isinstance(nodes, str):
            return
        for node in nodes:
            if not isinstance(node, Mapping):
                continue
            pages = _page_label(node)
            print(
                f"{'  ' * depth}- [{node.get('section_id')}] "
                f"{node.get('heading', 'Untitled')} (pages {pages})"
            )
            render(node.get("children", []), depth + 1)

    render(value.get("section_tree", []))


def _print_section(value: Mapping[str, Any]) -> None:
    print(f"{value.get('title', 'Untitled')} [{value.get('paper_id', 'unknown')}]")
    print(
        f"Section {value.get('section_id')}: {value.get('section') or value.get('heading', '')} "
        f"(pages {_page_label(value)})"
    )
    print()
    print(str(value.get("text") or ""))
    if value.get("next_offset") is not None:
        print(f"\nMore text: repeat with --offset {value['next_offset']}")


def _print_evaluation(value: Mapping[str, Any]) -> None:
    print(f"System: {value.get('system', 'unknown')} | queries: {value.get('query_count', 0)}")
    aggregate = value.get("aggregate", {})
    if not isinstance(aggregate, Mapping):
        return
    labels = (
        ("Paper Recall@5", "recall_at_5"),
        ("Paper Recall@10", "recall_at_10"),
        ("MRR", "mrr"),
        ("nDCG@10", "ndcg_at_10"),
    )
    for label, key in labels:
        score = aggregate.get(key, 0.0)
        print(f"{label}: {float(score):.4f}")
    if value.get("labels"):
        print(str(value["labels"]))


def _resolve_queries_path(config: AppConfig, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else config.paths.root / path


def _run_command(args: argparse.Namespace, config: AppConfig) -> int:
    if args.command == "doctor":
        report = run_doctor(config, deep=args.deep)
        _print_json(report) if args.json_output else _print_doctor(report)
        environment_ok = bool(report.get("ok"))
        search_ready = bool(report.get("ready", environment_ok))
        return 0 if environment_ok and search_ready else 1

    if args.command == "ingest":
        with CorpusDatabase(config.paths.database) as database:
            report = ingest_corpus(
                config,
                database,
                force=args.force,
                rechunk=args.rechunk,
                limit=args.limit,
                patterns=tuple(args.pattern),
                embeddings=not args.no_embeddings,
            )
        _print_json(report) if args.json_output else _print_ingest(report)
        return 1 if getattr(report, "failed", 0) else 0

    if args.command == "serve-mcp":
        # Stdout is the MCP stdio transport. Never print status text in this branch.
        build_mcp_server(config_path=config.config_path).run()
        return 0

    with ResearchCorpus(config) as corpus:
        if args.command == "search":
            result = corpus.search(
                args.query,
                mode=args.mode,
                top_k=args.top_k,
                pipeline=args.pipeline,
            )
            _print_json(result) if args.json_output else _print_results(result)
            return 0
        if args.command == "research":
            result = corpus.research_search(
                args.query,
                top_k=args.top_k,
                pipeline=args.pipeline,
            )
            _print_json(result) if args.json_output else _print_results(result)
            return 0
        if args.command == "outline":
            result = corpus.outline(args.paper_id)
            _print_json(result) if args.json_output else _print_outline(result)
            return 0
        if args.command == "read":
            result = corpus.read_section(
                args.paper_id,
                args.section_id,
                offset=args.offset,
                max_tokens=args.max_tokens,
            )
            _print_json(result) if args.json_output else _print_section(result)
            return 0
        if args.command == "evaluate":
            queries = load_evaluation_queries(_resolve_queries_path(config, args.queries))
            result = evaluate(corpus, queries, system=args.system, top_k=args.top_k)
            _print_json(result) if args.json_output else _print_evaluation(result)
            return 0
    raise AssertionError(f"unhandled command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging("INFO")
    try:
        config = load_config(args.config)
        _configure_logging(config.log_level)
        return _run_command(args, config)
    except KeyboardInterrupt:
        LOGGER.error("Interrupted")
        return 130
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 1


__all__ = ["build_parser", "main"]
