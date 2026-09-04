from __future__ import annotations

import gc
import logging
import resource
import sys
import threading
from typing import Any

from .config import ResourcesConfig

LOGGER = logging.getLogger(__name__)
GB = 1_000_000_000
GIB = 1024**3

_MPS_LIMIT_LOCK = threading.Lock()
_MPS_LIMIT_BYTES: int | None = None


class MemoryBudgetExceeded(MemoryError):
    """Raised when continuing model work would violate the configured budget."""


def is_memory_exhaustion_error(exc: BaseException) -> bool:
    if isinstance(exc, (MemoryError, MemoryBudgetExceeded)):
        return True
    message = str(exc).casefold()
    return any(
        marker in message
        for marker in (
            "out of memory",
            "cannot allocate memory",
            "can't allocate memory",
            "invalid buffer size",
            "memory budget",
        )
    )


def peak_resident_memory_bytes() -> int:
    """Return this process's peak resident set using platform-native units."""

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Darwin reports bytes; Linux and most BSD Python builds report KiB.
    return value if sys.platform == "darwin" else value * 1024


def enforce_process_memory_budget(resources: ResourcesConfig, stage: str) -> None:
    """Fail closed once the process-level 50 GiB safety budget is reached."""

    used = peak_resident_memory_bytes()
    limit = int(resources.max_process_memory_gb * GB)
    if used >= limit:
        raise MemoryBudgetExceeded(
            f"process memory budget reached during {stage}: "
            f"peak={used / GB:.2f} GB limit={resources.max_process_memory_gb:.2f} GB"
        )


def configure_mps_memory_limit(
    device: str,
    resources: ResourcesConfig,
    *,
    torch_module: Any | None = None,
) -> int | None:
    """Apply a hard PyTorch MPS allocator cap and never loosen it in-process."""

    if device.strip().lower().split(":", 1)[0] != "mps":
        return None
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError:
            return

    mps = getattr(torch_module, "mps", None)
    required = (
        "recommended_max_memory",
        "set_per_process_memory_fraction",
    )
    if mps is None or any(not callable(getattr(mps, name, None)) for name in required):
        raise RuntimeError(
            "The installed PyTorch MPS backend cannot enforce the configured memory limit"
        )

    requested = int(resources.mps_memory_limit_gb * GB)
    global _MPS_LIMIT_BYTES
    with _MPS_LIMIT_LOCK:
        effective = requested if _MPS_LIMIT_BYTES is None else min(_MPS_LIMIT_BYTES, requested)
        recommended = int(mps.recommended_max_memory())
        if recommended <= 0:
            raise RuntimeError("PyTorch MPS returned an invalid recommended memory size")
        mps.set_per_process_memory_fraction(min(2.0, effective / recommended))
        if _MPS_LIMIT_BYTES != effective:
            LOGGER.info(
                "Applied hard MPS allocator limit %.2f GB (process budget %.2f GB)",
                effective / GB,
                resources.max_process_memory_gb,
            )
        _MPS_LIMIT_BYTES = effective
        return effective


def release_accelerator_memory(
    device: str | None,
    resources: ResourcesConfig | None,
    *,
    torch_module: Any | None = None,
) -> None:
    """Release unused MPS allocations after a bounded inference batch."""

    gc.collect()
    if (
        resources is None
        or not resources.release_memory_after_batch
        or (device or "").strip().lower().split(":", 1)[0] != "mps"
    ):
        return
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError:
            return

    mps = getattr(torch_module, "mps", None)
    if mps is None:
        return
    synchronize = getattr(mps, "synchronize", None)
    empty_cache = getattr(mps, "empty_cache", None)
    for operation, name in ((synchronize, "synchronize"), (empty_cache, "empty_cache")):
        if not callable(operation):
            continue
        try:
            operation()
        except Exception as exc:
            if is_memory_exhaustion_error(exc):
                raise MemoryBudgetExceeded(
                    f"MPS memory cleanup failed during {name}: {type(exc).__name__}: {exc}"
                ) from None
            # Cleanup is best effort. In particular, never let a secondary cleanup
            # error replace the original model OOM that prompted this call.
            LOGGER.warning("MPS memory cleanup %s failed: %s: %s", name, type(exc).__name__, exc)


__all__ = [
    "GB",
    "GIB",
    "MemoryBudgetExceeded",
    "configure_mps_memory_limit",
    "enforce_process_memory_budget",
    "is_memory_exhaustion_error",
    "peak_resident_memory_bytes",
    "release_accelerator_memory",
]
