"""Build the compact, checksummed SAGE.T10.1 artifact inventory.

The baseline publishes only the two compact T10 reports.  Every other local
artifact under ``training/sage_t`` remains untouched and is represented here
by path, byte size, and SHA-256 so the exclusion is explicit and auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

FORMAT_VERSION = "sage-t10.1-omitted-artifact-inventory-v1"
BASELINE_HEAD = "64fec35a4211e0bbac896987d9efe1038b548f94"
DEFAULT_ROOT = Path("training") / "sage_t"
DEFAULT_OUTPUT = DEFAULT_ROOT / "t10_1_omitted_artifacts_inventory.json"
INCLUDED_COMPACT_REPORTS = (
    DEFAULT_ROOT / "progress_witness_v10_0b" / "report.json",
    DEFAULT_ROOT / "progress_witness_v10_1" / "report.json",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _relative(path: Path, *, repository_root: Path) -> str:
    return path.resolve().relative_to(repository_root.resolve()).as_posix()


def _record(path: Path, *, repository_root: Path) -> dict[str, Any]:
    return {
        "path": _relative(path, repository_root=repository_root),
        "bytes": path.stat().st_size,
        "sha256": _sha256_bytes(path.read_bytes()),
    }


def build_inventory(
    *,
    repository_root: str | Path = ".",
    artifact_root: str | Path = DEFAULT_ROOT,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    repository = Path(repository_root).resolve()
    artifacts = (repository / Path(artifact_root)).resolve()
    output = (repository / Path(output_path)).resolve()
    if not artifacts.is_relative_to(repository):
        raise ValueError("artifact root must stay inside the repository")
    if not output.is_relative_to(artifacts):
        raise ValueError("inventory output must stay inside the artifact root")

    included_paths = {
        (repository / path).resolve() for path in INCLUDED_COMPACT_REPORTS
    }
    included = [
        _record(path, repository_root=repository)
        for path in sorted(included_paths)
        if path.is_file()
    ]
    omitted_paths = [
        path
        for path in artifacts.rglob("*")
        if path.is_file()
        and path.resolve() != output
        and path.resolve() not in included_paths
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    ]
    omitted = [
        _record(path, repository_root=repository)
        for path in sorted(omitted_paths, key=lambda item: item.as_posix())
    ]
    unsigned: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "baseline_parent_head": BASELINE_HEAD,
        "policy": {
            "included": "compact T10.0b and T10.1 reports only",
            "omitted": "raw, regenerable, cache, trace, and historical run artifacts",
            "files_deleted": 0,
        },
        "included_compact_reports": included,
        "omitted_artifacts": omitted,
        "summary": {
            "included_files": len(included),
            "included_bytes": sum(item["bytes"] for item in included),
            "omitted_files": len(omitted),
            "omitted_bytes": sum(item["bytes"] for item in omitted),
        },
    }
    return {
        **unsigned,
        "inventory_checksum": _sha256_bytes(
            _canonical_json(unsigned).encode("utf-8")
        ),
    }


def write_inventory(
    inventory: dict[str, Any],
    *,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inventory local artifacts excluded from the T10.1 baseline."
    )
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--artifact-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    inventory = build_inventory(
        repository_root=args.repository_root,
        artifact_root=args.artifact_root,
        output_path=args.output,
    )
    output = write_inventory(inventory, output_path=args.output)
    print(
        json.dumps(
            {
                "output": output.as_posix(),
                "inventory_checksum": inventory["inventory_checksum"],
                **inventory["summary"],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "BASELINE_HEAD",
    "DEFAULT_OUTPUT",
    "DEFAULT_ROOT",
    "FORMAT_VERSION",
    "INCLUDED_COMPACT_REPORTS",
    "build_inventory",
    "write_inventory",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
