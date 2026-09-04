from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Iterable, Sequence
from typing import TypeAlias

try:  # NumPy is a normal dependency, but the storage layer remains portable without it.
    import numpy as _np
except ImportError:  # pragma: no cover - exercised by forcing use_numpy=False in tests
    _np = None


Vector: TypeAlias = Sequence[float] | Iterable[float]
PackedVectorRow: TypeAlias = tuple[str, bytes, int]


def model_fingerprint(
    model: str,
    *,
    provider: str = "",
    dimensions: int | None = None,
    revision: str = "",
    profile: str = "",
    normalized: bool = True,
) -> str:
    """Return a stable fingerprint for vectors that may safely share an index."""

    payload = json.dumps(
        {
            "dimensions": dimensions,
            "model": model,
            "normalized": normalized,
            "provider": provider,
            "profile": profile,
            "revision": revision,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _coerce_finite(vector: Vector) -> list[float]:
    values = [float(value) for value in vector]
    if not values:
        raise ValueError("vectors must contain at least one value")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("vectors may not contain NaN or infinity")
    return values


def normalize(vector: Vector) -> list[float]:
    values = _coerce_finite(vector)
    norm = math.sqrt(math.fsum(value * value for value in values))
    if norm == 0.0:
        raise ValueError("zero-length vectors cannot be normalized")
    return [value / norm for value in values]


def encode_float32(vector: Vector, *, normalized: bool = False) -> tuple[bytes, int]:
    """Encode a vector as a portable little-endian float32 BLOB."""

    values = normalize(vector) if normalized else _coerce_finite(vector)
    return struct.pack(f"<{len(values)}f", *values), len(values)


def decode_float32(blob: bytes | bytearray | memoryview, dimensions: int) -> tuple[float, ...]:
    if dimensions <= 0:
        raise ValueError("dimensions must be positive")
    raw = bytes(blob)
    expected = dimensions * 4
    if len(raw) != expected:
        raise ValueError(f"invalid float32 BLOB: expected {expected} bytes, got {len(raw)}")
    return struct.unpack(f"<{dimensions}f", raw)


def cosine_similarity(left: Vector, right: Vector) -> float:
    left_values = _coerce_finite(left)
    right_values = _coerce_finite(right)
    if len(left_values) != len(right_values):
        raise ValueError("vectors must have equal dimensions")
    left_norm = math.sqrt(math.fsum(value * value for value in left_values))
    right_norm = math.sqrt(math.fsum(value * value for value in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot = math.fsum(a * b for a, b in zip(left_values, right_values, strict=True))
    return dot / (left_norm * right_norm)


def cosine_top_k(
    query: Vector,
    rows: Iterable[PackedVectorRow],
    *,
    top_k: int,
    use_numpy: bool | None = None,
) -> list[tuple[str, float]]:
    """Rank packed float32 vectors by exact cosine similarity.

    Rows with a different declared dimension are ignored. A malformed BLOB for a
    matching dimension is treated as corrupt data and raises ``ValueError``.
    Results are deterministic for tied scores.
    """

    if top_k <= 0:
        return []
    query_values = _coerce_finite(query)
    query_dimensions = len(query_values)
    compatible = [row for row in rows if row[2] == query_dimensions]
    if not compatible:
        return []

    enabled = _np is not None if use_numpy is None else use_numpy and _np is not None
    if enabled:
        assert _np is not None
        query_array = _np.asarray(query_values, dtype=_np.float32)
        query_norm = float(_np.linalg.norm(query_array))
        if query_norm == 0.0:
            scores = _np.zeros(len(compatible), dtype=_np.float32)
        else:
            matrix = _np.empty((len(compatible), query_dimensions), dtype=_np.float32)
            for index, (_, blob, dimensions) in enumerate(compatible):
                if len(blob) != dimensions * 4:
                    raise ValueError(
                        f"invalid float32 BLOB: expected {dimensions * 4} bytes, got {len(blob)}"
                    )
                matrix[index] = _np.frombuffer(blob, dtype="<f4", count=dimensions)
            norms = _np.linalg.norm(matrix, axis=1)
            dots = matrix @ query_array
            scores = _np.divide(
                dots,
                norms * query_norm,
                out=_np.zeros_like(dots),
                where=norms != 0,
            )
        ranked = [(compatible[index][0], float(score)) for index, score in enumerate(scores)]
    else:
        ranked = [
            (key, cosine_similarity(query_values, decode_float32(blob, dimensions)))
            for key, blob, dimensions in compatible
        ]

    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked[:top_k]


__all__ = [
    "PackedVectorRow",
    "Vector",
    "cosine_similarity",
    "cosine_top_k",
    "decode_float32",
    "encode_float32",
    "model_fingerprint",
    "normalize",
]
