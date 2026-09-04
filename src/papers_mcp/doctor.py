from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import shutil
import sqlite3
from typing import Any

from .config import AppConfig
from .embeddings import configured_embedding_identity, embedding_provider_identity


def _current_source_manifest(config: AppConfig) -> tuple[dict[str, object] | None, list[str]]:
    """Hash the currently discoverable corpus without mutating ingestion state."""

    from .curated_index import parse_curated_markdown
    from .ids import sha256_file
    from .ingest import (
        _discover_markdown_with_errors,
        _discover_pdfs_with_errors,
        _discovered_source_locations,
        _relative,
        build_source_manifest,
    )
    from .models import SourceDocument

    errors: list[str] = []
    discovered_pdfs, pdf_errors = _discover_pdfs_with_errors(config)
    errors.extend(f"{_relative(path, config.paths.root)}: {error}" for path, error in pdf_errors)
    try:
        index_bytes = config.paths.human_index.read_bytes()
        index_hash = hashlib.sha256(index_bytes).hexdigest()
        curated = parse_curated_markdown(
            index_bytes.decode("utf-8"),
            known_artifacts=[_relative(path, config.paths.root) for path in discovered_pdfs],
        )
    except Exception as exc:
        errors.append(f"{_relative(config.paths.human_index, config.paths.root)}: {exc}")
        return None, errors

    discovered_markdown, markdown_errors = _discover_markdown_with_errors(config, curated)
    errors.extend(
        f"{_relative(path, config.paths.root)}: {error}" for path, error in markdown_errors
    )
    sources: list[SourceDocument] = []
    for path, source_kind in _discovered_source_locations(
        config,
        curated,
        pdfs=discovered_pdfs,
        markdown=discovered_markdown,
    ):
        try:
            sources.append(SourceDocument(path, source_kind, sha256_file(path)))
        except Exception as exc:
            errors.append(f"{_relative(path, config.paths.root)}: {exc}")
    if errors:
        return None, errors
    return build_source_manifest(config, index_hash, sources), []


def _package(name: str, module: str | None = None) -> dict[str, Any]:
    module_name = module or name.replace("-", "_")
    installed = importlib.util.find_spec(module_name) is not None
    version = None
    if installed:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"ok": installed, "version": version}


def _fts_check() -> dict[str, Any]:
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE VIRTUAL TABLE probe USING fts5(text)")
        connection.execute("INSERT INTO probe(text) VALUES ('vertex enclosure')")
        matched = connection.execute(
            "SELECT COUNT(*) FROM probe WHERE probe MATCH 'vertex'"
        ).fetchone()[0]
        return {"ok": matched == 1, "sqlite_version": sqlite3.sqlite_version}
    except sqlite3.Error as exc:
        return {"ok": False, "sqlite_version": sqlite3.sqlite_version, "error": str(exc)}
    finally:
        connection.close()


def _device_check() -> dict[str, Any]:
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": "cpu",
        "mps_available": False,
    }
    try:
        import torch
    except ImportError:
        result["warning"] = "Torch is not installed; production Qwen providers are unavailable."
        return result
    result["torch_version"] = torch.__version__
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        result["device"] = "mps"
        result["mps_available"] = True
        try:
            tensor = torch.ones(2, device="mps", dtype=torch.bfloat16)
            result["bfloat16"] = bool(float(tensor.sum().cpu()) == 2.0)
        except Exception as exc:  # pragma: no cover - hardware dependent
            result["bfloat16"] = False
            result["bfloat16_error"] = str(exc)
    elif torch.cuda.is_available():
        result["device"] = "cuda"
    return result


def _model_cached(model_id: str, revision: str = "") -> bool:
    try:
        from huggingface_hub import scan_cache_dir

        repository = next(
            (repo for repo in scan_cache_dir().repos if repo.repo_id == model_id),
            None,
        )
        if repository is None:
            return False
        for cached_revision in repository.revisions:
            commit_hash = str(getattr(cached_revision, "commit_hash", ""))
            if revision and commit_hash != revision:
                continue
            files = {item.file_name: item for item in cached_revision.files}
            has_weight_index = False
            for index_name in (
                "model.safetensors.index.json",
                "pytorch_model.bin.index.json",
            ):
                index = files.get(index_name)
                if index is None or not index.blob_path.is_file():
                    continue
                has_weight_index = True
                metadata = json.loads(index.blob_path.read_text(encoding="utf-8"))
                required = set(metadata.get("weight_map", {}).values())
                if required and all(
                    name in files and files[name].blob_path.is_file() for name in required
                ):
                    return True
            if has_weight_index:
                continue
            if any(
                item.blob_path.is_file()
                and (
                    item.file_name.endswith(".safetensors")
                    or item.file_name.startswith("pytorch_model")
                    and item.file_name.endswith(".bin")
                )
                for item in cached_revision.files
            ):
                return True
        return False
    except Exception:
        return False


def run_doctor(config: AppConfig, *, deep: bool = False) -> dict[str, Any]:
    paths = {
        "config": {"path": str(config.config_path), "ok": config.config_path.is_file()},
        "human_index": {
            "path": str(config.paths.human_index),
            "ok": config.paths.human_index.is_file(),
        },
        "pdf_roots": [{"path": str(path), "ok": path.is_dir()} for path in config.paths.pdf_roots],
        "database": {
            "path": str(config.paths.database),
            "exists": config.paths.database.is_file(),
        },
    }
    dependencies = {
        "numpy": _package("numpy"),
        "pymupdf": _package("PyMuPDF", "pymupdf"),
        "pymupdf4llm": _package("pymupdf4llm"),
        "sentence_transformers": _package("sentence-transformers", "sentence_transformers"),
        "torch": _package("torch"),
        "transformers": _package("transformers"),
        "mcp": _package("mcp"),
    }
    extractors = {
        "marker": {
            "configured": "marker" in config.extraction.providers,
            "available": shutil.which(config.extraction.marker_command.split()[0]) is not None,
            "optional": True,
        },
        "pymupdf4llm": {
            "configured": "pymupdf4llm" in config.extraction.providers,
            "available": dependencies["pymupdf4llm"]["ok"],
        },
        "pymupdf": {
            "configured": "pymupdf" in config.extraction.providers,
            "available": dependencies["pymupdf"]["ok"],
            "degraded_fallback": True,
        },
    }
    models: dict[str, Any] = {
        "embedding": {
            "model": config.embedding.model,
            "revision": config.embedding.revision or None,
            "backend": config.embedding.backend,
            "cached": _model_cached(config.embedding.model, config.embedding.revision),
            "quantization": config.embedding.quantization,
        },
        "reranker": {
            "model": config.reranker.model,
            "revision": config.reranker.revision or None,
            "backend": config.reranker.backend,
            "cached": _model_cached(config.reranker.model, config.reranker.revision),
            "quantization": config.reranker.quantization,
        },
    }
    corpus: dict[str, Any] = {"initialized": False}
    if config.paths.database.is_file():
        try:
            from .database import (
                CORPUS_SNAPSHOT_COMPLETE_KEY,
                CORPUS_SOURCE_MANIFEST_KEY,
                CorpusDatabase,
            )

            with CorpusDatabase(config.paths.database) as database:
                snapshot_marker = database.get_meta(CORPUS_SNAPSHOT_COMPLETE_KEY)
                stored_manifest_json = database.get_meta(CORPUS_SOURCE_MANIFEST_KEY)
                snapshot_complete = snapshot_marker == "1"
                snapshot_current = False
                snapshot_errors: list[str] = []
                manifest_recorded = stored_manifest_json is not None
                if snapshot_marker == "1":
                    if stored_manifest_json is None:
                        snapshot_errors.append(
                            "The complete-scan marker has no source manifest; rerun ingestion."
                        )
                    else:
                        try:
                            stored_manifest = json.loads(stored_manifest_json)
                        except (TypeError, json.JSONDecodeError) as exc:
                            snapshot_errors.append(f"Stored source manifest is invalid: {exc}")
                        else:
                            current_manifest, snapshot_errors = _current_source_manifest(config)
                            snapshot_current = (
                                current_manifest is not None
                                and not snapshot_errors
                                and current_manifest == stored_manifest
                            )
                corpus = {
                    "initialized": True,
                    "revision": database.revision,
                    **database.counts(),
                    "embedding_sets": database.embedding_fingerprints(),
                    "hierarchy_complete": database.corpus_hierarchy_complete(),
                    "fts_complete": database.fts_complete(),
                    "failures": len(database.list_failures()),
                    # Databases created before this marker remain compatible. Any
                    # ingest performed by this version writes an explicit value.
                    "snapshot_complete": snapshot_complete,
                    "snapshot_current": snapshot_current,
                    "snapshot_manifest_recorded": manifest_recorded,
                    "snapshot_marker_recorded": snapshot_marker is not None,
                    "snapshot_errors": snapshot_errors,
                }
        except Exception as exc:
            corpus = {"initialized": False, "error": str(exc)}

    deep_results: dict[str, Any] | None = None
    if deep:
        from .embeddings import create_embedding_provider
        from .memory import release_accelerator_memory
        from .reranker import create_reranker

        deep_results = {}
        embedding = None
        try:
            embedding = create_embedding_provider(config.embedding, config.resources)
            vector = embedding.embed_query("vertex enclosure")
            active_backend = getattr(
                embedding,
                "active_backend",
                config.embedding.backend,
            )
            expected_backend = config.embedding.backend.strip().lower().replace("-", "_")
            degraded = (
                expected_backend
                in {
                    "sentence_transformers",
                    "sentence_transformer",
                    "qwen3",
                }
                and active_backend != "sentence_transformers"
            )
            deep_results["embedding"] = {
                "ok": bool(vector) and not degraded,
                "dimensions": len(vector),
                "backend": active_backend,
                "degraded": degraded,
                "model_name": embedding_provider_identity(embedding, len(vector))[0],
                "model_fingerprint": embedding_provider_identity(embedding, len(vector))[1],
            }
        except Exception as exc:
            deep_results["embedding"] = {"ok": False, "error": str(exc)}
        finally:
            embedding_device = getattr(embedding, "device", None)
            embedding = None
            release_accelerator_memory(embedding_device, config.resources)
        reranker = None
        try:
            reranker = create_reranker(config.reranker, config.resources)
            scores = reranker.score("vertex enclosure", ["vertex enclosure condition"])
            active_backend = getattr(
                reranker,
                "active_backend",
                config.reranker.backend,
            )
            expected_backend = config.reranker.backend.strip().lower().replace("-", "_")
            degraded = (
                expected_backend
                in {
                    "sentence_transformers",
                    "sentence_transformer",
                    "cross_encoder",
                    "qwen3",
                }
                and active_backend != "sentence_transformers"
            )
            deep_results["reranker"] = {
                "ok": len(scores) == 1 and not degraded,
                "backend": active_backend,
                "degraded": degraded,
            }
        except Exception as exc:
            deep_results["reranker"] = {"ok": False, "error": str(exc)}
        finally:
            reranker_device = getattr(reranker, "device", None)
            reranker = None
            release_accelerator_memory(reranker_device, config.resources)

    configured_extractors = [value for value in extractors.values() if value.get("configured")]
    role_backends = {
        "embedding": config.embedding.backend.strip().lower().replace("-", "_"),
        "reranker": config.reranker.backend.strip().lower().replace("-", "_"),
    }
    production_names = {
        "sentence_transformers",
        "sentence_transformer",
        "cross_encoder",
        "qwen3",
    }
    production_models = any(backend in production_names for backend in role_backends.values())
    model_dependencies_ok = not production_models or all(
        dependencies[name]["ok"] for name in ("sentence_transformers", "transformers", "torch")
    )
    deep_ok = not deep or bool(
        deep_results
        and all(isinstance(value, dict) and value.get("ok") for value in deep_results.values())
    )
    environment_ok = (
        paths["config"]["ok"]
        and paths["human_index"]["ok"]
        and all(item["ok"] for item in paths["pdf_roots"])
        and _fts_check()["ok"]
        and dependencies["mcp"]["ok"]
        and dependencies["numpy"]["ok"]
        and bool(configured_extractors)
        and any(value.get("available") for value in configured_extractors)
        and model_dependencies_ok
        and deep_ok
    )
    corpus_ready = bool(
        corpus.get("initialized")
        and corpus.get("papers", 0)
        and corpus.get("sections", 0)
        and corpus.get("chunks", 0)
        and corpus.get("hierarchy_complete")
        and corpus.get("fts_complete")
        and corpus.get("snapshot_complete", True)
        and corpus.get("snapshot_current", True)
        and not corpus.get("failures", 0)
    )
    compatible_fingerprints: list[str] = []
    for row in corpus.get("embedding_sets", []):
        try:
            model_name, fingerprint = configured_embedding_identity(
                config.embedding,
                int(row["dimensions"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        if row.get("model_name") == model_name and row.get("model_fingerprint") == fingerprint:
            compatible_fingerprints.append(fingerprint)
    if deep_results and isinstance(deep_results.get("embedding"), dict):
        deep_fingerprint = deep_results["embedding"].get("model_fingerprint")
        if isinstance(deep_fingerprint, str):
            compatible_fingerprints.append(deep_fingerprint)
    dense_ready = False
    complete_fingerprint: str | None = None
    if corpus_ready and config.paths.database.is_file():
        try:
            from .database import CorpusDatabase

            with CorpusDatabase(config.paths.database) as database:
                for fingerprint in dict.fromkeys(compatible_fingerprints):
                    if database.corpus_embeddings_complete(
                        fingerprint
                    ) and database.curated_embeddings_complete(fingerprint):
                        dense_ready = True
                        complete_fingerprint = fingerprint
                        break
        except Exception:
            dense_ready = False
    ready = environment_ok and corpus_ready and dense_ready
    warnings: list[str] = []
    if not corpus.get("snapshot_complete", True):
        if not corpus.get("snapshot_marker_recorded", True):
            warnings.append(
                "Corpus is not indexed from a verified complete source snapshot; run an "
                "unfiltered papers-mcp ingest before treating the corpus as complete."
            )
        else:
            warnings.append(
                "The last ingest used a filter/limit or observed an incomplete source snapshot; "
                "run an unfiltered papers-mcp ingest before treating the corpus as complete."
            )
    elif not corpus.get("snapshot_current", True):
        if corpus.get("snapshot_errors"):
            warnings.append(
                "The current corpus snapshot could not be verified against the last complete "
                "ingest; run an unfiltered papers-mcp ingest. "
                + " ".join(str(error) for error in corpus["snapshot_errors"])
            )
        else:
            warnings.append(
                "Corpus sources or INDEX.md changed after the last complete ingest; "
                "run an unfiltered papers-mcp ingest."
            )
    elif corpus.get("failures", 0):
        warnings.append(
            "Corpus has unresolved ingestion or consistency failures; inspect the failure "
            "report and rerun ingestion after correcting them."
        )
    elif not corpus_ready:
        warnings.append(
            "Corpus is not indexed with a complete paper/section/chunk hierarchy and FTS mirror; "
            "run papers-mcp ingest."
        )
    elif not dense_ready:
        warnings.append(
            "Corpus has no complete embeddings for the configured model revision; "
            "normal hybrid search is not ready."
        )
    for role in ("embedding", "reranker"):
        if role_backends[role] in production_names and not models[role]["cached"]:
            warnings.append(f"Configured {role} model is not cached and may download on first use.")
    return {
        "ok": environment_ok,
        "ready": ready,
        "warnings": warnings,
        "paths": paths,
        "sqlite_fts5": _fts_check(),
        "vectors": {
            "backend": "SQLite float32 BLOBs + exact NumPy cosine",
            "numpy_available": dependencies["numpy"]["ok"],
            "sqlite_vec_required": False,
            "dense_ready": dense_ready,
            "complete_model_fingerprint": complete_fingerprint,
        },
        "dependencies": dependencies,
        "extractors": extractors,
        "device": _device_check(),
        "resources": {
            "max_process_memory_gb": config.resources.max_process_memory_gb,
            "mps_memory_limit_gb": config.resources.mps_memory_limit_gb,
            "extraction_worker_timeout_seconds": (
                config.resources.extraction_worker_timeout_seconds
            ),
            "release_memory_after_batch": config.resources.release_memory_after_batch,
        },
        "models": models,
        "corpus": corpus,
        "deep": deep_results,
    }
