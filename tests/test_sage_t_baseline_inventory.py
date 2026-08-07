from __future__ import annotations

import hashlib
import json
from pathlib import Path

from theory.sage_t.baseline_inventory import build_inventory, write_inventory


def _checksum(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_baseline_inventory_is_complete_deterministic_and_self_excluding(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    artifacts = root / "training" / "sage_t"
    first_report = artifacts / "progress_witness_v10_0b" / "report.json"
    second_report = artifacts / "progress_witness_v10_1" / "report.json"
    omitted = artifacts / "raw" / "trace.jsonl"
    output = artifacts / "t10_1_omitted_artifacts_inventory.json"
    for path, content in (
        (first_report, "first"),
        (second_report, "second"),
        (omitted, "raw"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    inventory = build_inventory(
        repository_root=root,
        artifact_root=Path("training") / "sage_t",
        output_path=Path("training")
        / "sage_t"
        / "t10_1_omitted_artifacts_inventory.json",
    )
    write_inventory(inventory, output_path=output)
    rebuilt = build_inventory(
        repository_root=root,
        artifact_root=Path("training") / "sage_t",
        output_path=Path("training")
        / "sage_t"
        / "t10_1_omitted_artifacts_inventory.json",
    )

    assert inventory == rebuilt
    assert inventory["summary"] == {
        "included_files": 2,
        "included_bytes": 11,
        "omitted_files": 1,
        "omitted_bytes": 3,
    }
    assert [item["path"] for item in inventory["omitted_artifacts"]] == [
        "training/sage_t/raw/trace.jsonl"
    ]
    unsigned = dict(inventory)
    checksum = unsigned.pop("inventory_checksum")
    assert checksum == _checksum(unsigned)
