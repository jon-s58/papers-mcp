from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_NAME = "config.toml"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
DEFAULT_EMBEDDING_REVISION = "5cf2132abc99cad020ac570b19d031efec650f2b"
DEFAULT_RERANKER_MODEL = "Qwen/Qwen3-Reranker-4B"
DEFAULT_RERANKER_REVISION = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
_QUANTIZED_PRECISIONS = {
    "int8",
    "int4",
    "uint8",
    "binary",
    "ubinary",
    "qint8",
    "4bit",
    "8bit",
}


@dataclass(frozen=True, slots=True)
class PathsConfig:
    root: Path
    pdf_roots: tuple[Path, ...]
    generated_markdown: Path
    cards: Path
    database: Path
    human_index: Path
    exclude_dirs: tuple[str, ...]
    markdown_sources_from_index: bool = True


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    providers: tuple[str, ...] = ("pymupdf4llm", "pymupdf")
    marker_command: str = "marker_single"
    page_markers: bool = True
    ocr: bool = True


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    backend: str
    model: str
    device: str
    precision: str
    quantization: bool
    batch_size: int
    max_length: int
    allow_explicit_hash_fallback: bool
    fallback_dimensions: int
    query_instruction: str
    revision: str = ""


@dataclass(frozen=True, slots=True)
class RerankerConfig:
    backend: str
    model: str
    device: str
    precision: str
    quantization: bool
    batch_size: int
    max_length: int
    allow_explicit_lexical_fallback: bool
    instruction: str
    revision: str = ""


@dataclass(frozen=True, slots=True)
class ChunksConfig:
    target_tokens: int = 850
    min_tokens: int = 250
    max_tokens: int = 1600


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    bm25_candidates: int = 60
    dense_candidates: int = 60
    rerank_candidates: int = 40
    rrf_k: int = 60
    default_top_k: int = 10
    precision_max_results_per_paper: int = 3
    discovery_max_results_per_paper: int = 2
    exact_max_results_per_paper: int = 10
    snippet_words: int = 300


@dataclass(frozen=True, slots=True)
class QueryExpansionConfig:
    enabled: bool = False
    provider: str = "openai_compatible"
    base_url: str = ""
    model: str = ""
    api_key_env: str = "PAPERS_MCP_EXPANSION_API_KEY"
    timeout_seconds: int = 20


@dataclass(frozen=True, slots=True)
class MCPConfig:
    max_top_k: int = 20
    max_section_tokens: int = 12000


@dataclass(frozen=True, slots=True)
class ResourcesConfig:
    """Process-tree safety limits for extraction and local model inference."""

    max_process_memory_gb: float = 45.0
    mps_memory_limit_gb: float = 24.0
    extraction_worker_timeout_seconds: int = 120
    release_memory_after_batch: bool = True


@dataclass(frozen=True, slots=True)
class AppConfig:
    config_path: Path
    paths: PathsConfig
    extraction: ExtractionConfig
    embedding: EmbeddingConfig
    reranker: RerankerConfig
    resources: ResourcesConfig = field(default_factory=ResourcesConfig)
    chunks: ChunksConfig = field(default_factory=ChunksConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    query_expansion: QueryExpansionConfig = field(default_factory=QueryExpansionConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    log_level: str = "INFO"

    def ensure_output_dirs(self) -> None:
        self.paths.generated_markdown.mkdir(parents=True, exist_ok=True)
        self.paths.cards.mkdir(parents=True, exist_ok=True)
        self.paths.database.parent.mkdir(parents=True, exist_ok=True)


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"[{name}] must be a TOML table")
    return value


def find_config(path: str | os.PathLike[str] | None = None) -> Path:
    if path is not None:
        candidate = Path(path).expanduser()
    elif os.environ.get("PAPERS_MCP_CONFIG"):
        candidate = Path(os.environ["PAPERS_MCP_CONFIG"]).expanduser()
    else:
        candidate = Path.cwd() / DEFAULT_CONFIG_NAME
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Configuration not found: {candidate}. Pass --config or set PAPERS_MCP_CONFIG."
        )
    return candidate


def load_config(path: str | os.PathLike[str] | None = None) -> AppConfig:
    config_path = find_config(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    base = config_path.parent

    paths = _section(raw, "paths")
    pdf_roots_raw = paths.get("pdf_roots", ["."])
    if not isinstance(pdf_roots_raw, list) or not pdf_roots_raw:
        raise ValueError("paths.pdf_roots must be a non-empty array")
    paths_config = PathsConfig(
        root=base,
        pdf_roots=tuple(_resolve(base, str(item)) for item in pdf_roots_raw),
        generated_markdown=_resolve(base, paths.get("generated_markdown", "papers/markdown")),
        cards=_resolve(base, paths.get("cards", "papers/cards")),
        database=_resolve(base, paths.get("database", "data/papers.db")),
        human_index=_resolve(base, paths.get("human_index", "INDEX.md")),
        exclude_dirs=tuple(str(item) for item in paths.get("exclude_dirs", [".git", "data"])),
        markdown_sources_from_index=bool(paths.get("markdown_sources_from_index", True)),
    )
    if paths_config.generated_markdown == paths_config.root:
        raise ValueError("paths.generated_markdown must be a dedicated subdirectory, not root")
    if paths_config.cards == paths_config.root:
        raise ValueError("paths.cards must be a dedicated subdirectory, not root")

    extraction = _section(raw, "extraction")
    providers_raw = extraction.get("providers", ["pymupdf4llm", "pymupdf"])
    if (
        not isinstance(providers_raw, list)
        or not providers_raw
        or any(not isinstance(provider, str) or not provider.strip() for provider in providers_raw)
    ):
        raise ValueError("extraction.providers must be a non-empty array of provider names")
    extraction_config = ExtractionConfig(
        providers=tuple(provider.strip() for provider in providers_raw),
        marker_command=str(extraction.get("marker_command", "marker_single")),
        page_markers=bool(extraction.get("page_markers", True)),
        ocr=bool(extraction.get("ocr", True)),
    )

    models = _section(raw, "models")
    embedding = models.get("embedding", {})
    reranker = models.get("reranker", {})
    embedding_model = str(embedding.get("model", DEFAULT_EMBEDDING_MODEL))
    embedding_revision = str(
        embedding.get(
            "revision",
            DEFAULT_EMBEDDING_REVISION if embedding_model == DEFAULT_EMBEDDING_MODEL else "",
        )
    )
    reranker_model = str(reranker.get("model", DEFAULT_RERANKER_MODEL))
    reranker_revision = str(
        reranker.get(
            "revision",
            DEFAULT_RERANKER_REVISION if reranker_model == DEFAULT_RERANKER_MODEL else "",
        )
    )
    embedding_config = EmbeddingConfig(
        backend=str(embedding.get("backend", "sentence_transformers")),
        model=embedding_model,
        revision=embedding_revision,
        device=str(embedding.get("device", "auto")),
        precision=str(embedding.get("precision", "bfloat16")),
        quantization=bool(embedding.get("quantization", False)),
        batch_size=int(embedding.get("batch_size", 1)),
        max_length=int(embedding.get("max_length", 2048)),
        allow_explicit_hash_fallback=bool(embedding.get("allow_explicit_hash_fallback", True)),
        fallback_dimensions=int(embedding.get("fallback_dimensions", 768)),
        query_instruction=str(embedding.get("query_instruction", "")),
    )
    reranker_config = RerankerConfig(
        backend=str(reranker.get("backend", "sentence_transformers")),
        model=reranker_model,
        revision=reranker_revision,
        device=str(reranker.get("device", "auto")),
        precision=str(reranker.get("precision", "bfloat16")),
        quantization=bool(reranker.get("quantization", False)),
        batch_size=int(reranker.get("batch_size", 1)),
        max_length=int(reranker.get("max_length", 2048)),
        allow_explicit_lexical_fallback=bool(reranker.get("allow_explicit_lexical_fallback", True)),
        instruction=str(reranker.get("instruction", "")),
    )
    if embedding_config.quantization or reranker_config.quantization:
        raise ValueError("model quantization is disabled for full-quality retrieval")
    production_backends = {
        "sentence_transformers",
        "sentence_transformer",
        "cross_encoder",
        "qwen3",
    }
    for role, model_config in (
        ("embedding", embedding_config),
        ("reranker", reranker_config),
    ):
        if not model_config.backend.strip() or not model_config.model.strip():
            raise ValueError(f"models.{role}.backend and model must be non-empty")
        if model_config.batch_size <= 0 or model_config.max_length <= 0:
            raise ValueError(f"models.{role}.batch_size and max_length must be positive")
        precision = model_config.precision.strip().lower().replace("-", "_")
        if precision in _QUANTIZED_PRECISIONS:
            raise ValueError(f"models.{role}.precision may not request quantization")
        backend = model_config.backend.strip().lower().replace("-", "_")
        if backend in production_backends and model_config.batch_size > 1:
            raise ValueError(f"models.{role}.batch_size may not exceed the safe ceiling of 1")
        if backend in production_backends and model_config.max_length > 2048:
            raise ValueError(f"models.{role}.max_length may not exceed the safe ceiling of 2048")
    if embedding_config.fallback_dimensions <= 0:
        raise ValueError("models.embedding.fallback_dimensions must be positive")
    if embedding_config.fallback_dimensions > 8192:
        raise ValueError("models.embedding.fallback_dimensions may not exceed 8192")
    for role, model_config in (
        ("embedding", embedding_config),
        ("reranker", reranker_config),
    ):
        backend = model_config.backend.strip().lower().replace("-", "_")
        model_path = Path(model_config.model).expanduser()
        local_model = (
            model_path.exists() if model_path.is_absolute() else (base / model_path).exists()
        )
        if (
            backend in production_backends
            and not local_model
            and re.fullmatch(r"[0-9a-fA-F]{40}", model_config.revision.strip()) is None
        ):
            raise ValueError(
                f"models.{role}.revision must be an immutable 40-character commit hash"
            )

    chunks = _section(raw, "chunks")
    chunks_config = ChunksConfig(
        target_tokens=int(chunks.get("target_tokens", 850)),
        min_tokens=int(chunks.get("min_tokens", 250)),
        max_tokens=int(chunks.get("max_tokens", 1600)),
    )
    if not 0 < chunks_config.min_tokens <= chunks_config.target_tokens <= chunks_config.max_tokens:
        raise ValueError("chunk sizes must satisfy 0 < min_tokens <= target_tokens <= max_tokens")
    if chunks_config.max_tokens > embedding_config.max_length:
        raise ValueError(
            "chunks.max_tokens may not exceed models.embedding.max_length; "
            "oversized retrieval chunks would be truncated during embedding"
        )

    retrieval = _section(raw, "retrieval")
    retrieval_config = RetrievalConfig(
        **{
            field_name: int(retrieval.get(field_name, getattr(RetrievalConfig(), field_name)))
            for field_name in RetrievalConfig.__dataclass_fields__
        }
    )
    if any(
        getattr(retrieval_config, field_name) <= 0
        for field_name in RetrievalConfig.__dataclass_fields__
    ):
        raise ValueError("all retrieval limits and candidate counts must be positive")
    expansion = _section(raw, "query_expansion")
    expansion_config = QueryExpansionConfig(
        enabled=bool(expansion.get("enabled", False)),
        provider=str(expansion.get("provider", "openai_compatible")),
        base_url=str(expansion.get("base_url", "")),
        model=str(expansion.get("model", "")),
        api_key_env=str(expansion.get("api_key_env", "PAPERS_MCP_EXPANSION_API_KEY")),
        timeout_seconds=int(expansion.get("timeout_seconds", 20)),
    )
    if expansion_config.timeout_seconds <= 0:
        raise ValueError("query_expansion.timeout_seconds must be positive")
    resources_raw = _section(raw, "resources")
    resources_config = ResourcesConfig(
        max_process_memory_gb=float(resources_raw.get("max_process_memory_gb", 45.0)),
        mps_memory_limit_gb=float(resources_raw.get("mps_memory_limit_gb", 24.0)),
        extraction_worker_timeout_seconds=int(
            resources_raw.get("extraction_worker_timeout_seconds", 120)
        ),
        release_memory_after_batch=bool(resources_raw.get("release_memory_after_batch", True)),
    )
    if not 0 < resources_config.max_process_memory_gb <= 45:
        raise ValueError("resources.max_process_memory_gb must be greater than 0 and at most 45")
    if not 0 < resources_config.mps_memory_limit_gb <= 24:
        raise ValueError("resources.mps_memory_limit_gb must be greater than 0 and at most 24")
    if resources_config.mps_memory_limit_gb >= resources_config.max_process_memory_gb:
        raise ValueError("resources.mps_memory_limit_gb must be below max_process_memory_gb")
    if not 1 <= resources_config.extraction_worker_timeout_seconds <= 3600:
        raise ValueError(
            "resources.extraction_worker_timeout_seconds must be between 1 and 3600"
        )
    mcp_raw = _section(raw, "mcp")
    mcp_config = MCPConfig(
        max_top_k=int(mcp_raw.get("max_top_k", 20)),
        max_section_tokens=int(mcp_raw.get("max_section_tokens", 12000)),
    )
    if mcp_config.max_top_k <= 0 or mcp_config.max_section_tokens <= 0:
        raise ValueError("mcp.max_top_k and max_section_tokens must be positive")
    logging_raw = _section(raw, "logging")

    return AppConfig(
        config_path=config_path,
        paths=paths_config,
        extraction=extraction_config,
        embedding=embedding_config,
        reranker=reranker_config,
        resources=resources_config,
        chunks=chunks_config,
        retrieval=retrieval_config,
        query_expansion=expansion_config,
        mcp=mcp_config,
        log_level=str(logging_raw.get("level", "INFO")).upper(),
    )
