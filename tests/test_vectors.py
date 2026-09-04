from __future__ import annotations

import math

import pytest

from papers_mcp.vectors import (
    cosine_similarity,
    cosine_top_k,
    decode_float32,
    encode_float32,
    model_fingerprint,
)


def test_float32_round_trip_and_normalization() -> None:
    blob, dimensions = encode_float32([3.0, 4.0], normalized=True)
    assert dimensions == 2
    assert len(blob) == 8
    assert decode_float32(blob, dimensions) == pytest.approx((0.6, 0.8), abs=1e-6)

    with pytest.raises(ValueError, match="zero-length"):
        encode_float32([0.0, 0.0], normalized=True)
    with pytest.raises(ValueError, match="NaN"):
        encode_float32([math.nan])
    with pytest.raises(ValueError, match="expected 8 bytes"):
        decode_float32(blob[:4], 2)


def test_cosine_similarity_validates_dimensions_and_handles_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [2.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([0.0, 0.0], [2.0, 0.0]) == 0.0
    with pytest.raises(ValueError, match="equal dimensions"):
        cosine_similarity([1.0], [1.0, 2.0])


def test_cosine_top_k_numpy_and_stdlib_are_equivalent() -> None:
    rows = []
    for key, vector in (
        ("best", [1.0, 0.0, 0.0]),
        ("second", [0.8, 0.2, 0.0]),
        ("opposite", [-1.0, 0.0, 0.0]),
    ):
        blob, dimensions = encode_float32(vector)
        rows.append((key, blob, dimensions))
    wrong_dimension, wrong_dimensions = encode_float32([1.0, 0.0])
    rows.append(("ignored", wrong_dimension, wrong_dimensions))

    numpy_results = cosine_top_k([1.0, 0.0, 0.0], rows, top_k=3, use_numpy=True)
    stdlib_results = cosine_top_k([1.0, 0.0, 0.0], rows, top_k=3, use_numpy=False)

    assert [key for key, _ in numpy_results] == ["best", "second", "opposite"]
    assert [key for key, _ in stdlib_results] == ["best", "second", "opposite"]
    assert [score for _, score in numpy_results] == pytest.approx(
        [score for _, score in stdlib_results], abs=1e-6
    )


def test_cosine_top_k_ties_are_deterministic_and_corrupt_blobs_fail() -> None:
    blob, dimensions = encode_float32([1.0, 0.0])
    assert cosine_top_k(
        [1.0, 0.0], [("z", blob, dimensions), ("a", blob, dimensions)], top_k=2
    ) == pytest.approx([("a", 1.0), ("z", 1.0)])

    with pytest.raises(ValueError, match="invalid float32 BLOB"):
        cosine_top_k([1.0, 0.0], [("broken", blob[:2], 2)], top_k=1, use_numpy=False)


def test_model_fingerprint_is_stable_and_configuration_sensitive() -> None:
    first = model_fingerprint("Qwen/Qwen3-Embedding-8B", provider="sentence_transformers")
    second = model_fingerprint("Qwen/Qwen3-Embedding-8B", provider="sentence_transformers")
    changed = model_fingerprint(
        "Qwen/Qwen3-Embedding-8B", provider="sentence_transformers", dimensions=4096
    )
    assert first == second
    assert len(first) == 64
    assert changed != first
