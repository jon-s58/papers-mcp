from types import SimpleNamespace

import pytest

import papers_mcp.memory as memory
from papers_mcp.config import ResourcesConfig


def test_mps_allocator_limit_is_derived_from_gib_budget_and_never_loosened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fractions: list[float] = []
    fake_mps = SimpleNamespace(
        recommended_max_memory=lambda: 100 * memory.GB,
        set_per_process_memory_fraction=fractions.append,
    )
    fake_torch = SimpleNamespace(mps=fake_mps)
    monkeypatch.setattr(memory, "_MPS_LIMIT_BYTES", None)

    first = memory.configure_mps_memory_limit(
        "mps",
        ResourcesConfig(mps_memory_limit_gb=24),
        torch_module=fake_torch,
    )
    second = memory.configure_mps_memory_limit(
        "mps",
        ResourcesConfig(mps_memory_limit_gb=20),
        torch_module=fake_torch,
    )
    third = memory.configure_mps_memory_limit(
        "mps",
        ResourcesConfig(mps_memory_limit_gb=24),
        torch_module=fake_torch,
    )

    assert first == 24 * memory.GB
    assert second == third == 20 * memory.GB
    assert fractions == pytest.approx([0.24, 0.20, 0.20])


def test_indexed_mps_device_cannot_bypass_allocator_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fractions: list[float] = []
    fake_torch = SimpleNamespace(
        mps=SimpleNamespace(
            recommended_max_memory=lambda: 100 * memory.GB,
            set_per_process_memory_fraction=fractions.append,
        )
    )
    monkeypatch.setattr(memory, "_MPS_LIMIT_BYTES", None)

    memory.configure_mps_memory_limit(
        "mps:0",
        ResourcesConfig(mps_memory_limit_gb=24),
        torch_module=fake_torch,
    )

    assert fractions == pytest.approx([0.24])


def test_process_budget_fails_closed_at_forty_five_decimal_gb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(memory, "peak_resident_memory_bytes", lambda: 45 * memory.GB)

    with pytest.raises(MemoryError, match="process memory budget reached"):
        memory.enforce_process_memory_budget(ResourcesConfig(), "test batch")


def test_release_accelerator_memory_synchronizes_and_empties_mps() -> None:
    calls: list[str] = []
    fake_torch = SimpleNamespace(
        mps=SimpleNamespace(
            synchronize=lambda: calls.append("synchronize"),
            empty_cache=lambda: calls.append("empty_cache"),
        )
    )

    memory.release_accelerator_memory(
        "mps",
        ResourcesConfig(),
        torch_module=fake_torch,
    )

    assert calls == ["synchronize", "empty_cache"]


def test_release_cleanup_error_does_not_mask_an_existing_oom(caplog) -> None:
    calls: list[str] = []

    def fail_synchronize() -> None:
        calls.append("synchronize")
        raise RuntimeError("synchronize failed")

    fake_torch = SimpleNamespace(
        mps=SimpleNamespace(
            synchronize=fail_synchronize,
            empty_cache=lambda: calls.append("empty_cache"),
        )
    )

    memory.release_accelerator_memory(
        "mps:0",
        ResourcesConfig(),
        torch_module=fake_torch,
    )

    assert calls == ["synchronize", "empty_cache"]
    assert "cleanup synchronize failed" in caplog.text


def test_release_memory_oom_is_converted_to_budget_error() -> None:
    fake_torch = SimpleNamespace(
        mps=SimpleNamespace(
            synchronize=lambda: (_ for _ in ()).throw(RuntimeError("MPS out of memory")),
            empty_cache=lambda: None,
        )
    )

    with pytest.raises(memory.MemoryBudgetExceeded, match="cleanup failed"):
        memory.release_accelerator_memory(
            "mps",
            ResourcesConfig(),
            torch_module=fake_torch,
        )
