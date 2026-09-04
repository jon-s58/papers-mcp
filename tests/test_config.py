from pathlib import Path

import pytest

from papers_mcp.config import load_config


def test_checked_in_production_profile_is_inside_memory_envelope() -> None:
    loaded = load_config(Path(__file__).resolve().parents[1] / "config.toml")

    assert loaded.embedding.model == "Qwen/Qwen3-Embedding-4B"
    assert loaded.embedding.revision == "5cf2132abc99cad020ac570b19d031efec650f2b"
    assert loaded.embedding.precision == "bfloat16"
    assert loaded.embedding.batch_size == 1
    assert loaded.embedding.max_length == 2048
    assert loaded.reranker.model == "Qwen/Qwen3-Reranker-4B"
    assert loaded.reranker.revision == "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
    assert loaded.reranker.precision == "bfloat16"
    assert loaded.reranker.batch_size == 1
    assert loaded.reranker.max_length == 2048
    assert loaded.resources.max_process_memory_gb == 45
    assert loaded.resources.mps_memory_limit_gb == 24
    assert loaded.resources.extraction_worker_timeout_seconds == 120


def test_config_paths_resolve_relative_to_config(tmp_path: Path) -> None:
    config = tmp_path / "nested" / "config.toml"
    config.parent.mkdir()
    config.write_text(
        """
[paths]
pdf_roots = ["../papers"]
database = "../data/test.db"

[models.embedding]
backend = "hash"

[models.reranker]
backend = "lexical"
""",
        encoding="utf-8",
    )
    loaded = load_config(config)
    assert loaded.paths.pdf_roots == ((tmp_path / "papers").resolve(),)
    assert loaded.paths.database == (tmp_path / "data" / "test.db").resolve()
    assert loaded.embedding.model == "Qwen/Qwen3-Embedding-4B"
    assert loaded.embedding.batch_size == 1
    assert loaded.embedding.max_length == 2048
    assert loaded.embedding.precision == "bfloat16"
    assert loaded.resources.max_process_memory_gb == 45
    assert loaded.resources.mps_memory_limit_gb == 24
    assert loaded.resources.extraction_worker_timeout_seconds == 120


def test_invalid_chunk_order_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
[paths]
pdf_roots = ["."]
[models.embedding]
[models.reranker]
[chunks]
min_tokens = 900
target_tokens = 500
max_tokens = 1000
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="chunk sizes"):
        load_config(config)


def test_chunks_cannot_exceed_embedding_context(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
[paths]
pdf_roots = ["."]
[models.embedding]
backend = "hash"
max_length = 1000
[models.reranker]
backend = "lexical"
[chunks]
min_tokens = 100
target_tokens = 800
max_tokens = 1200
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="may not exceed models.embedding.max_length"):
        load_config(config)


@pytest.mark.parametrize(
    ("fragment", "message"),
    [
        ("[extraction]\nproviders = []", "extraction.providers"),
        ("[retrieval]\nbm25_candidates = 0", "retrieval"),
        ("[query_expansion]\ntimeout_seconds = 0", "timeout_seconds"),
        ("[mcp]\nmax_top_k = 0", "mcp.max_top_k"),
        ("[resources]\nmax_process_memory_gb = 46", "max_process_memory_gb"),
        ("[resources]\nmps_memory_limit_gb = 25", "mps_memory_limit_gb"),
        (
            "[resources]\nextraction_worker_timeout_seconds = 0",
            "extraction_worker_timeout_seconds",
        ),
    ],
)
def test_nonpositive_runtime_configuration_is_rejected(
    tmp_path: Path,
    fragment: str,
    message: str,
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        '[paths]\npdf_roots = ["."]\n'
        '[models.embedding]\nbackend = "hash"\n'
        '[models.reranker]\nbackend = "lexical"\n'
        f"{fragment}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=message):
        load_config(config)


@pytest.mark.parametrize(
    "model_tables",
    [
        '[models.embedding]\nbackend = "hash"\nbatch_size = 0\n'
        '[models.reranker]\nbackend = "lexical"',
        '[models.embedding]\nbackend = "hash"\nquantization = true\n'
        '[models.reranker]\nbackend = "lexical"',
        '[models.embedding]\nbackend = "hash"\n'
        '[models.reranker]\nbackend = "lexical"\nmax_length = 0',
    ],
)
def test_invalid_model_quality_or_batch_configuration_is_rejected(
    tmp_path: Path,
    model_tables: str,
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        f'[paths]\npdf_roots = ["."]\n{model_tables}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_config(config)


@pytest.mark.parametrize(
    "unsafe_setting",
    [
        "batch_size = 2",
        "max_length = 2049",
    ],
)
def test_production_model_memory_envelope_cannot_be_overridden(
    tmp_path: Path,
    unsafe_setting: str,
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        '[paths]\npdf_roots = ["."]\n'
        '[models.embedding]\nbackend = "sentence_transformers"\n'
        f"{unsafe_setting}\n"
        '[models.reranker]\nbackend = "lexical"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="safe ceiling"):
        load_config(config)


@pytest.mark.parametrize("revision", ["", "main", "v1.0"])
def test_remote_production_models_require_immutable_commit_revisions(
    tmp_path: Path,
    revision: str,
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        '[paths]\npdf_roots = ["."]\n'
        "[models.embedding]\n"
        'backend = "sentence_transformers"\n'
        'model = "example/custom-embedding"\n'
        f'revision = "{revision}"\n'
        '[models.reranker]\nbackend = "lexical"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="immutable 40-character commit hash"):
        load_config(config)
