"""Build the versioned, non-destructive SAGE.T10.2 artifact inventory.

Only the registered compact lifecycle reports are publishable. Every other file in
the registered T10.2 artifact directory is retained locally and recorded as
omitted evidence with its byte size and SHA-256 digest.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

FORMAT_VERSION = "sage-t10.2-omitted-artifact-inventory-v1"
PROTOCOL_FORMAT_VERSION = "sage-t10.2-preregistered-protocol-v1"
CROSS_FIT_AUDIT_FORMAT_VERSION = "sage-t10.2-cross-fit-audit-v1"
VALIDATION_TIMING_PROOF_FORMAT_VERSION = "sage-t10.2-validation-timing-proof-v1"
REPORT_INVENTORY_BINDING_FORMAT_VERSION = "sage-t10.2-report-inventory-binding-v1"
MINIMUM_PUBLISHABLE_BYTES = 2
MAXIMUM_PUBLISHABLE_BYTES = 512 * 1024 * 1024
DEFAULT_ARTIFACT_ROOT = Path("training") / "sage_t" / "t10_2_gauge_posterior"
DEFAULT_OUTPUT = Path("training") / "sage_t" / "t10_2_omitted_artifacts_inventory.json"
REPORT_INVENTORY_BINDING_NAME = "t10_2_report_inventory_binding.json"
PUBLISHABLE_REPORT_NAMES = (
    "collection_report.json",
    "cross_fit_audit.json",
    "compile_report.json",
    "replay_report.json",
    "source_report.json",
    "validation_report.json",
    "validation_timing_proof.json",
    "report.json",
)
PUBLISHABLE_SIDECAR_NAMES = (
    DEFAULT_OUTPUT.name,
    REPORT_INVENTORY_BINDING_NAME,
)
REGISTERED_PUBLISHABLE_JSON_NAMES = (
    *PUBLISHABLE_REPORT_NAMES,
    *PUBLISHABLE_SIDECAR_NAMES,
)

_COMMON_REPORT_TYPES = {
    "format_version": str,
    "phase": str,
    "status": str,
    "manifest_checksum": str,
    "report_checksum": str,
}
_PUBLISHABLE_JSON_SCHEMAS: dict[str, dict[str, Any]] = {
    "collection_report.json": {
        "checksum_key": "report_checksum",
        "format_version": PROTOCOL_FORMAT_VERSION,
        "phase": "collect",
        "statuses": frozenset({"T10_2_SOURCE_COLLECTION_COMPLETE"}),
        "required_types": {
            **_COMMON_REPORT_TYPES,
            "games": list,
            "splits": Mapping,
            "event_count": int,
            "timing": Mapping,
            "events": Mapping,
            "cross_fit_audit": Mapping,
            "cross_fit_checks": Mapping,
            "firewall": Mapping,
        },
    },
    "cross_fit_audit.json": {
        "checksum_key": "audit_checksum",
        "format_version": CROSS_FIT_AUDIT_FORMAT_VERSION,
        "phase": None,
        "statuses": None,
        "required_types": {
            "format_version": str,
            "manifest_checksum": str,
            "source_events": Mapping,
            "source_event_ids_sha256": str,
            "factory": Mapping,
            "registered_unit_count": int,
            "units": list,
            "checks": Mapping,
            "passed": bool,
            "audit_checksum": str,
        },
    },
    "compile_report.json": {
        "checksum_key": "report_checksum",
        "format_version": PROTOCOL_FORMAT_VERSION,
        "phase": "compile",
        "statuses": frozenset(
            {
                "DATA_OR_PROVENANCE_INVALID",
                "PASS_T10_2_QA",
                "T10_2_FRESH_INTEGRITY_COMPLETE",
            }
        ),
        "required_types": {
            **_COMMON_REPORT_TYPES,
            "checks": Mapping,
            "passed": bool,
            "firewall": Mapping,
        },
    },
    "replay_report.json": {
        "checksum_key": "report_checksum",
        "format_version": PROTOCOL_FORMAT_VERSION,
        "phase": "replay",
        "statuses": frozenset({"T10_2_SOURCE_REPLAY_COMPLETE"}),
        "required_types": {
            **_COMMON_REPORT_TYPES,
            "event_count": int,
            "input": Mapping,
            "events": Mapping,
            "checks": Mapping,
        },
    },
    "source_report.json": {
        "checksum_key": "report_checksum",
        "format_version": PROTOCOL_FORMAT_VERSION,
        "phase": "source-train",
        "statuses": frozenset(
            {
                "DATA_OR_PROVENANCE_INVALID",
                "FAIL_T10_2_SOURCE_GATE",
                "PASS_T10_2_SOURCE_GATE",
            }
        ),
        "required_types": {
            **_COMMON_REPORT_TYPES,
            "verdict": str,
            "checks": Mapping,
            "passed": bool,
            "firewall": Mapping,
        },
    },
    "validation_report.json": {
        "checksum_key": "report_checksum",
        "format_version": PROTOCOL_FORMAT_VERSION,
        "phase": "validate",
        "statuses": frozenset({"FAIL_T10_2_VALIDATION", "PASS_T10_2_VALIDATION"}),
        "required_types": {
            **_COMMON_REPORT_TYPES,
            "verdict": str,
            "source_report_checksum": str,
            "metrics": Mapping,
            "checks": Mapping,
            "passed": bool,
            "firewall": Mapping,
        },
    },
    "validation_timing_proof.json": {
        "checksum_key": "timing_proof_checksum",
        "format_version": VALIDATION_TIMING_PROOF_FORMAT_VERSION,
        "phase": None,
        "statuses": None,
        "required_types": {
            "format_version": str,
            "producer": str,
            "manifest_checksum": str,
            "source_report": Mapping,
            "source_report_checksum": str,
            "validation_runs": Mapping,
            "validation_run_count": int,
            "validation_pairs_sha256": str,
            "code_sha256": Mapping,
            "validation_report_linked": bool,
            "cycle_free": bool,
            "timing_proof_checksum": str,
        },
        "exact_values": {
            "producer": "validate_phase:external_monotonic_clock_v1",
            "validation_report_linked": False,
            "cycle_free": True,
        },
    },
    "report.json": {
        "checksum_key": "report_checksum",
        "format_version": PROTOCOL_FORMAT_VERSION,
        "phase": "report",
        "statuses": frozenset({"T10_2_COMPLETE"}),
        "required_types": {
            **_COMMON_REPORT_TYPES,
            "verdict": str,
            "supported": bool,
            "inputs": Mapping,
            "artifact_inventory": Mapping,
            "firewall": Mapping,
        },
    },
    DEFAULT_OUTPUT.name: {
        "checksum_key": "inventory_checksum",
        "format_version": FORMAT_VERSION,
        "phase": None,
        "statuses": None,
        "required_types": {
            "format_version": str,
            "artifact_root": str,
            "inventory_path": str,
            "policy": Mapping,
            "included_compact_reports": list,
            "omitted_artifacts": list,
            "summary": Mapping,
            "inventory_checksum": str,
        },
    },
    REPORT_INVENTORY_BINDING_NAME: {
        "checksum_key": "binding_checksum",
        "format_version": REPORT_INVENTORY_BINDING_FORMAT_VERSION,
        "phase": None,
        "statuses": None,
        "required_types": {
            "format_version": str,
            "manifest_checksum": str,
            "report": Mapping,
            "inventory": Mapping,
            "inventory_checksum": str,
            "cycle_free": bool,
            "binding_checksum": str,
        },
        "exact_values": {"cycle_free": True},
    },
}

_KNOWN_CHECKSUM_KEYS = frozenset(
    {
        "audit_checksum",
        "binding_checksum",
        "inventory_checksum",
        "report_checksum",
        "timing_proof_checksum",
    }
)


def canonical_json(value: Any) -> str:
    """Return the canonical encoding used by the inventory checksum."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _resolve_inside(
    repository_root: Path, candidate: str | Path, *, label: str
) -> Path:
    path = Path(candidate)
    resolved = (path if path.is_absolute() else repository_root / path).resolve()
    if not resolved.is_relative_to(repository_root):
        raise ValueError(f"{label} must stay inside the repository")
    return resolved


def _relative(path: Path, *, repository_root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(repository_root):
        raise ValueError("inventoried artifact escaped the repository")
    return resolved.relative_to(repository_root).as_posix()


def _record(path: Path, *, repository_root: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": _relative(path, repository_root=repository_root),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _forbidden_publishable_payload_hits(value: Any) -> tuple[str, ...]:
    forbidden = {
        "adjacency",
        "adjacency_list",
        "adjacency_matrix",
        "absolute_coordinate",
        "absolute_coordinates",
        "bitmap",
        "canvas",
        "cells",
        "color",
        "colors",
        "colour",
        "colours",
        "coordinate",
        "coordinates",
        "coords",
        "edge_list",
        "edges",
        "entities",
        "false_facts",
        "frame",
        "frames",
        "frame_pixels",
        "full_graph",
        "graph",
        "graphs",
        "grid",
        "image",
        "images",
        "matrix",
        "nodes",
        "palette",
        "pixel",
        "pixels",
        "raster",
        "raw_frame",
        "raw_frames",
        "raw_graph",
        "raw_grid",
        "rgb",
        "rgba",
        "true_facts",
        "vertex",
        "vertices",
    }
    safe_negative_retention_flags = {
        "graphs_retained",
        "raw_frames_retained",
        "raw_runtime_state_retained",
    }
    hits: list[str] = []

    def visit(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for raw_key, child in item.items():
                key = str(raw_key).casefold().replace("-", "_")
                child_path = f"{path}.{raw_key}" if path else str(raw_key)
                tokens = tuple(token for token in key.split("_") if token)
                unsafe_token = bool(
                    set(tokens)
                    & {
                        "adjacency",
                        "bitmap",
                        "canvas",
                        "coordinate",
                        "coordinates",
                        "coords",
                        "edge",
                        "edges",
                        "graph",
                        "graphs",
                        "grid",
                        "image",
                        "images",
                        "matrix",
                        "node",
                        "nodes",
                        "pixel",
                        "pixels",
                        "raster",
                        "rgb",
                        "rgba",
                        "vertex",
                        "vertices",
                    }
                )
                safe_negative_flag = (
                    key in safe_negative_retention_flags and child is False
                )
                if not safe_negative_flag and (
                    key in forbidden or key.startswith("raw_") or unsafe_token
                ):
                    hits.append(child_path)
                visit(child, child_path)
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")

    visit(value, "")
    return tuple(sorted(set(hits)))


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _has_required_type(value: Any, expected: type[Any]) -> bool:
    if expected is bool:
        return type(value) is bool
    if expected is int:
        return type(value) is int
    return isinstance(value, expected)


def _matches_registered_schema(name: str, payload: Mapping[str, Any]) -> bool:
    schema = _PUBLISHABLE_JSON_SCHEMAS.get(name)
    if schema is None:
        return False
    required_types = schema["required_types"]
    if not isinstance(required_types, Mapping) or not all(
        key in payload and _has_required_type(payload[key], expected)
        for key, expected in required_types.items()
    ):
        return False
    if payload.get("format_version") != schema["format_version"]:
        return False
    expected_phase = schema["phase"]
    if expected_phase is None:
        if "phase" in payload:
            return False
    elif payload.get("phase") != expected_phase:
        return False
    statuses = schema["statuses"]
    if statuses is None:
        if "status" in payload:
            return False
    elif payload.get("status") not in statuses:
        return False
    exact_values = schema.get("exact_values", {})
    if not isinstance(exact_values, Mapping) or any(
        payload.get(key) != value for key, value in exact_values.items()
    ):
        return False
    if "manifest_checksum" in required_types and not _is_sha256(
        payload.get("manifest_checksum")
    ):
        return False
    for key, expected_type in required_types.items():
        if (
            expected_type is str
            and key.endswith(("_sha256", "_checksum"))
            and not _is_sha256(payload.get(key))
        ):
            return False
    checksum_key = str(schema["checksum_key"])
    allowed_checksum_keys = {
        key for key in required_types if key in _KNOWN_CHECKSUM_KEYS
    }
    if checksum_key not in allowed_checksum_keys or (
        set(payload) & _KNOWN_CHECKSUM_KEYS != allowed_checksum_keys
    ):
        return False
    unsigned = dict(payload)
    observed_checksum = unsigned.pop(checksum_key, None)
    return bool(
        _is_sha256(observed_checksum)
        and observed_checksum == canonical_sha256(unsigned)
    )


def _is_publishable_compact_json(path: Path) -> bool:
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if not MINIMUM_PUBLISHABLE_BYTES <= size <= MAXIMUM_PUBLISHABLE_BYTES:
        return False
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
        encoded = canonical_json(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return False
    if not isinstance(payload, Mapping) or raw not in {encoded, f"{encoded}\n"}:
        return False
    if not _matches_registered_schema(path.name, payload):
        return False
    return not _forbidden_publishable_payload_hits(payload)


def is_registered_publishable_compact_json(path: str | Path) -> bool:
    """Return whether ``path`` is a registered, safe, canonical T10.2 artifact."""

    return _is_publishable_compact_json(Path(path))


def build_inventory(
    *,
    repository_root: str | Path = ".",
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Describe publishable and omitted artifacts without deleting any file."""

    repository = Path(repository_root).resolve()
    artifacts = _resolve_inside(repository, artifact_root, label="artifact root")
    output = _resolve_inside(repository, output_path, label="inventory output")
    if output.name != DEFAULT_OUTPUT.name:
        raise ValueError(
            f"inventory output must use registered name: {DEFAULT_OUTPUT.name}"
        )

    publishable_paths = tuple(
        (artifacts / name).resolve() for name in PUBLISHABLE_REPORT_NAMES
    )
    included_paths = tuple(
        path
        for path in publishable_paths
        if path.is_file() and path != output and _is_publishable_compact_json(path)
    )
    included_set = set(included_paths)

    all_files = (
        tuple(path for path in artifacts.rglob("*") if path.is_file())
        if artifacts.is_dir()
        else ()
    )
    omitted_paths = tuple(
        sorted(
            (
                path
                for path in all_files
                if path.resolve() != output and path.resolve() not in included_set
            ),
            key=lambda path: _relative(path, repository_root=repository),
        )
    )

    included = [_record(path, repository_root=repository) for path in included_paths]
    omitted = [_record(path, repository_root=repository) for path in omitted_paths]
    unsigned: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "artifact_root": _relative(artifacts, repository_root=repository),
        "inventory_path": _relative(output, repository_root=repository),
        "policy": {
            "included": (
                "canonical compact JSON lifecycle reports at the artifact root only"
            ),
            "publishable_report_names": list(PUBLISHABLE_REPORT_NAMES),
            "omitted": (
                "ledgers, raw data, caches, checkpoints, and every other local file"
            ),
            "omitted_artifacts_retained_locally": True,
            "files_deleted": 0,
        },
        "included_compact_reports": included,
        "omitted_artifacts": omitted,
        "summary": {
            "included_files": len(included),
            "included_bytes": sum(int(item["bytes"]) for item in included),
            "omitted_files": len(omitted),
            "omitted_bytes": sum(int(item["bytes"]) for item in omitted),
            "files_deleted": 0,
        },
    }
    return {**unsigned, "inventory_checksum": canonical_sha256(unsigned)}


def write_inventory(
    inventory: Mapping[str, Any],
    *,
    repository_root: str | Path = ".",
    output_path: str | Path = DEFAULT_OUTPUT,
) -> Path:
    """Atomically write one canonical inventory after verifying its checksum."""

    repository = Path(repository_root).resolve()
    output = _resolve_inside(repository, output_path, label="inventory output")
    if output.name != DEFAULT_OUTPUT.name:
        raise ValueError(
            f"inventory output must use registered name: {DEFAULT_OUTPUT.name}"
        )
    payload = dict(inventory)
    if not _matches_registered_schema(output.name, payload):
        raise ValueError("inventory schema or checksum is missing or invalid")
    if _forbidden_publishable_payload_hits(payload):
        raise ValueError("inventory contains a forbidden raw or graph payload")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(canonical_json(dict(inventory)))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return output


__all__ = [
    "DEFAULT_ARTIFACT_ROOT",
    "DEFAULT_OUTPUT",
    "FORMAT_VERSION",
    "PUBLISHABLE_REPORT_NAMES",
    "PUBLISHABLE_SIDECAR_NAMES",
    "REGISTERED_PUBLISHABLE_JSON_NAMES",
    "build_inventory",
    "canonical_json",
    "canonical_sha256",
    "is_registered_publishable_compact_json",
    "write_inventory",
]
