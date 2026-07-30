from __future__ import annotations

import json
from pathlib import Path

import pytest

from theory.sage12.artifact_budget import (
    BudgetLimits,
    StorageBudgetError,
    StorageGuard,
)


def _limits(*, file_bytes: int = 1024) -> BudgetLimits:
    return BudgetLimits(
        maximum_scratch_bytes=4096,
        maximum_local_cache_bytes=4096,
        maximum_derived_file_bytes=file_bytes,
        maximum_repository_bytes=1024 * 1024,
        minimum_free_bytes=0,
    )


def test_storage_guard_records_inventories_and_cleans_unique_scratch(
    tmp_path: Path,
) -> None:
    output = tmp_path / "training" / "sage12" / "v4_18"
    with StorageGuard(
        repo_root=tmp_path,
        output_dir=output,
        command="unit",
        limits=_limits(),
    ) as guard:
        (guard.scratch_dir / "temporary.bin").write_bytes(b"scratch")
        output.mkdir(parents=True, exist_ok=True)
        (output / "result.json").write_text("{}", encoding="utf-8")
        scratch = guard.scratch_dir

    assert not scratch.exists()
    events = [
        json.loads(line)
        for line in (output / "storage_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(events) == 1
    assert events[0]["scratch_removed"] is True
    assert events[0]["pre"]["inventory_checksum"]
    assert events[0]["post_after_cleanup"]["scratch_bytes"] == 0
    assert any(row["path"].endswith("result.json") for row in events[0]["outputs"])


def test_storage_guard_rejects_new_oversized_file_and_still_cleans(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out"
    scratch: Path | None = None
    with (
        pytest.raises(StorageBudgetError, match="derived file exceeds"),
        StorageGuard(
            repo_root=tmp_path,
            output_dir=output,
            command="oversized",
            limits=_limits(file_bytes=8),
        ) as guard,
    ):
        scratch = guard.scratch_dir
        output.mkdir(parents=True, exist_ok=True)
        (output / "too-large.bin").write_bytes(b"x" * 9)
    assert scratch is not None
    assert not scratch.exists()
