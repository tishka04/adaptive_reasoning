from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from theory.sage_t import t10_2_artifact_inventory as inventory_module
from theory.sage_t.t10_2_artifact_inventory import (
    CROSS_FIT_AUDIT_FORMAT_VERSION,
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_OUTPUT,
    MAXIMUM_PUBLISHABLE_BYTES,
    PROTOCOL_FORMAT_VERSION,
    PUBLISHABLE_REPORT_NAMES,
    PUBLISHABLE_SIDECAR_NAMES,
    REGISTERED_PUBLISHABLE_JSON_NAMES,
    REPORT_INVENTORY_BINDING_FORMAT_VERSION,
    REPORT_INVENTORY_BINDING_NAME,
    VALIDATION_TIMING_PROOF_FORMAT_VERSION,
    build_inventory,
    canonical_json,
    canonical_sha256,
    is_registered_publishable_compact_json,
    write_inventory,
)
from theory.sage_t.t10_2_protocol import DEFAULT_CODE_FILES


def _write_compact(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{canonical_json(payload)}\n", encoding="utf-8")


def _signed_payload(path: Path, payload: dict[str, object]) -> dict[str, object]:
    checksum_key = {
        "cross_fit_audit.json": "audit_checksum",
        "validation_timing_proof.json": "timing_proof_checksum",
        DEFAULT_OUTPUT.name: "inventory_checksum",
        REPORT_INVENTORY_BINDING_NAME: "binding_checksum",
    }.get(path.name, "report_checksum")
    return {**payload, checksum_key: canonical_sha256(payload)}


def _write_signed_compact(path: Path, payload: dict[str, object]) -> None:
    _write_compact(path, _signed_payload(path, payload))


def _valid_payload(name: str, *, index: int = 0) -> dict[str, object]:
    digest = format(index + 1, "x")[-1] * 64
    descriptor = {"bytes": index + 1, "sha256": digest}
    common = {
        "format_version": PROTOCOL_FORMAT_VERSION,
        "manifest_checksum": digest,
    }
    payloads: dict[str, dict[str, object]] = {
        "collection_report.json": {
            **common,
            "phase": "collect",
            "status": "T10_2_SOURCE_COLLECTION_COMPLETE",
            "games": ["source-a"],
            "splits": {"discovery": [0]},
            "event_count": 1,
            "timing": {"monotonic_elapsed_seconds": 1.0},
            "events": descriptor,
            "cross_fit_audit": descriptor,
            "cross_fit_checks": {"exact_nine_units": True},
            "firewall": {"holdout_opened": False},
        },
        "cross_fit_audit.json": {
            "format_version": CROSS_FIT_AUDIT_FORMAT_VERSION,
            "manifest_checksum": digest,
            "source_events": descriptor,
            "source_event_ids_sha256": digest,
            "factory": {"code_bound": True},
            "registered_unit_count": 9,
            "units": [],
            "checks": {"exact_nine_units": True},
            "passed": True,
        },
        "compile_report.json": {
            **common,
            "phase": "compile",
            "status": "T10_2_FRESH_INTEGRITY_COMPLETE",
            "checks": {"source_event_schema": True},
            "passed": True,
            "firewall": {"holdout_opened": False},
        },
        "replay_report.json": {
            **common,
            "phase": "replay",
            "status": "T10_2_SOURCE_REPLAY_COMPLETE",
            "event_count": 1,
            "input": descriptor,
            "events": descriptor,
            "checks": {"source_only": True},
        },
        "source_report.json": {
            **common,
            "phase": "source-train",
            "status": "PASS_T10_2_SOURCE_GATE",
            "verdict": "PASS_T10_2_SOURCE_GATE",
            "checks": {"holdout_closed": True},
            "passed": True,
            "firewall": {"holdout_opened": False},
        },
        "validation_report.json": {
            **common,
            "phase": "validate",
            "status": "PASS_T10_2_VALIDATION",
            "verdict": "SAGE_T10_2_GAUGE_POSTERIOR_SUPPORTED",
            "source_report_checksum": digest,
            "metrics": {"all_pairs_executed": True},
            "checks": {"all_pairs_executed": True},
            "passed": True,
            "firewall": {"holdout_opened": False},
        },
        "validation_timing_proof.json": {
            "format_version": VALIDATION_TIMING_PROOF_FORMAT_VERSION,
            "producer": "validate_phase:external_monotonic_clock_v1",
            "manifest_checksum": digest,
            "source_report": descriptor,
            "source_report_checksum": digest,
            "validation_runs": descriptor,
            "validation_run_count": 15,
            "validation_pairs_sha256": digest,
            "reported_pair_wall_seconds": 15.0,
            "monotonic_started": 1.0,
            "monotonic_finished": 2.0,
            "monotonic_elapsed_seconds": 1.0,
            "code_sha256": {"producer.py": digest},
            "validation_report_linked": False,
            "cycle_free": True,
        },
        "report.json": {
            **common,
            "phase": "report",
            "status": "T10_2_COMPLETE",
            "verdict": "SAGE_T10_2_GAUGE_POSTERIOR_SUPPORTED",
            "supported": True,
            "inputs": {"source_report": descriptor},
            "artifact_inventory": {"cyclic_hash_dependency": False},
            "firewall": {"holdout_opened": False},
        },
        REPORT_INVENTORY_BINDING_NAME: {
            "format_version": REPORT_INVENTORY_BINDING_FORMAT_VERSION,
            "manifest_checksum": digest,
            "report": descriptor,
            "inventory": descriptor,
            "inventory_checksum": digest,
            "cycle_free": True,
        },
    }
    return payloads[name]


def test_t10_2_inventory_is_complete_non_destructive_and_idempotent(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    assert "theory/sage_t/t10_2_artifact_inventory.py" in DEFAULT_CODE_FILES
    assert "tests/test_sage_t_t10_2_artifact_inventory.py" in DEFAULT_CODE_FILES
    assert "validation_timing_proof.json" in PUBLISHABLE_REPORT_NAMES
    assert "cross_fit_audit.json" in PUBLISHABLE_REPORT_NAMES
    assert set(REGISTERED_PUBLISHABLE_JSON_NAMES) == {
        *PUBLISHABLE_REPORT_NAMES,
        *PUBLISHABLE_SIDECAR_NAMES,
    }
    artifacts = repository / DEFAULT_ARTIFACT_ROOT
    included_names = tuple(
        name for name in PUBLISHABLE_REPORT_NAMES if name != "source_report.json"
    )
    for index, name in enumerate(included_names):
        _write_signed_compact(
            artifacts / name,
            _valid_payload(name, index=index),
        )

    # An allowlisted filename is still omitted when it is not compact canonical JSON.
    invalid_allowed = artifacts / "source_report.json"
    invalid_allowed.parent.mkdir(parents=True, exist_ok=True)
    invalid_allowed.write_text(
        json.dumps(
            _signed_payload(invalid_allowed, _valid_payload(invalid_allowed.name)),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    omitted_paths = (
        invalid_allowed,
        artifacts / "source_events.jsonl",
        artifacts / "raw" / "frame.bin",
        artifacts / "cache" / "posterior.bin",
        artifacts / "checkpoints" / "model.ckpt",
        artifacts / "nested" / "report.json",
    )
    for index, path in enumerate(omitted_paths[1:], start=1):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"omitted-{index}".encode("ascii"))

    original_files = {
        path: path.read_bytes() for path in artifacts.rglob("*") if path.is_file()
    }
    inventory = build_inventory(repository_root=repository)
    output = write_inventory(inventory, repository_root=repository)
    first_bytes = output.read_bytes()
    rebuilt = build_inventory(repository_root=repository)
    write_inventory(rebuilt, repository_root=repository)

    assert output == (repository / DEFAULT_OUTPUT).resolve()
    assert inventory == rebuilt
    assert output.read_bytes() == first_bytes
    assert first_bytes == f"{canonical_json(inventory)}\n".encode()
    assert [
        Path(item["path"]).name for item in inventory["included_compact_reports"]
    ] == list(included_names)
    assert {item["path"] for item in inventory["omitted_artifacts"]} == {
        path.relative_to(repository).as_posix() for path in omitted_paths
    }
    assert inventory["policy"]["files_deleted"] == 0
    assert inventory["policy"]["omitted_artifacts_retained_locally"] is True
    assert inventory["summary"] == {
        "included_files": len(included_names),
        "included_bytes": sum(
            (artifacts / name).stat().st_size for name in included_names
        ),
        "omitted_files": len(omitted_paths),
        "omitted_bytes": sum(path.stat().st_size for path in omitted_paths),
        "files_deleted": 0,
    }

    unsigned = dict(inventory)
    checksum = unsigned.pop("inventory_checksum")
    assert checksum == canonical_sha256(unsigned)
    for record in (
        *inventory["included_compact_reports"],
        *inventory["omitted_artifacts"],
    ):
        path = repository / record["path"]
        payload = path.read_bytes()
        assert record["bytes"] == len(payload)
        assert record["sha256"] == hashlib.sha256(payload).hexdigest()
    assert all(path.read_bytes() == payload for path, payload in original_files.items())


def test_t10_2_inventory_self_excludes_and_rejects_unsafe_or_unsigned_output(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    artifacts = repository / DEFAULT_ARTIFACT_ROOT
    _write_signed_compact(
        artifacts / "collection_report.json",
        _valid_payload("collection_report.json"),
    )
    relative_output = DEFAULT_OUTPUT

    inventory = build_inventory(
        repository_root=repository,
        output_path=relative_output,
    )
    output = write_inventory(
        inventory,
        repository_root=repository,
        output_path=relative_output,
    )
    rebuilt = build_inventory(
        repository_root=repository,
        output_path=relative_output,
    )
    assert inventory == rebuilt
    assert output.relative_to(repository).as_posix() not in {
        item["path"]
        for item in (
            *rebuilt["included_compact_reports"],
            *rebuilt["omitted_artifacts"],
        )
    }

    corrupted = {**inventory, "inventory_checksum": "0" * 64}
    with pytest.raises(ValueError, match="checksum"):
        write_inventory(
            corrupted,
            repository_root=repository,
            output_path=relative_output,
        )
    with pytest.raises(ValueError, match="inside the repository"):
        build_inventory(
            repository_root=repository,
            artifact_root=repository.parent,
        )
    with pytest.raises(ValueError, match="registered name"):
        build_inventory(
            repository_root=repository,
            output_path=DEFAULT_ARTIFACT_ROOT / "inventory.json",
        )


def test_t10_2_inventory_omits_even_signed_canonical_raw_payloads(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    artifacts = repository / DEFAULT_ARTIFACT_ROOT
    unsafe = artifacts / "source_report.json"
    unsafe_payload = _valid_payload(unsafe.name)
    unsafe_payload["nested"] = {
        "raw_grid": [[0, 1], [1, 0]],
        "graph_payload": {"edge_list": [[0, 1]]},
        "rgb": [255, 0, 0],
        "coordinates": [[1, 2]],
    }
    _write_signed_compact(
        unsafe,
        unsafe_payload,
    )

    inventory = build_inventory(repository_root=repository)
    assert inventory["included_compact_reports"] == []
    assert [item["path"] for item in inventory["omitted_artifacts"]] == [
        unsafe.relative_to(repository).as_posix()
    ]


def test_registered_schema_rejects_wrong_name_phase_status_format_and_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    artifacts = repository / DEFAULT_ARTIFACT_ROOT
    registered = artifacts / "collection_report.json"
    valid = _valid_payload(registered.name)
    _write_signed_compact(registered, valid)
    assert is_registered_publishable_compact_json(registered) is True

    generic = artifacts / "generic.json"
    _write_compact(generic, _signed_payload(registered, valid))
    assert is_registered_publishable_compact_json(generic) is False

    for label, mutation in (
        ("phase", {"phase": "replay"}),
        ("status", {"status": "T10_2_COMPLETE"}),
        ("format", {"format_version": "generic-v1"}),
    ):
        candidate = artifacts / f"{label}" / registered.name
        _write_signed_compact(candidate, {**valid, **mutation})
        assert is_registered_publishable_compact_json(candidate) is False

    missing_schema_field = dict(valid)
    missing_schema_field.pop("cross_fit_audit")
    missing = artifacts / "missing" / registered.name
    _write_signed_compact(missing, missing_schema_field)
    assert is_registered_publishable_compact_json(missing) is False

    corrupted = artifacts / "corrupted" / registered.name
    _write_compact(
        corrupted,
        {**valid, "report_checksum": "0" * 64},
    )
    assert is_registered_publishable_compact_json(corrupted) is False

    binding = repository / DEFAULT_OUTPUT.parent / REPORT_INVENTORY_BINDING_NAME
    _write_signed_compact(binding, _valid_payload(binding.name))
    assert is_registered_publishable_compact_json(binding) is True

    inventory = build_inventory(repository_root=repository)
    inventory_path = write_inventory(inventory, repository_root=repository)
    assert is_registered_publishable_compact_json(inventory_path) is True

    empty = artifacts / "empty" / "report.json"
    empty.parent.mkdir(parents=True, exist_ok=True)
    empty.write_bytes(b"")
    assert is_registered_publishable_compact_json(empty) is False

    monkeypatch.setattr(
        inventory_module,
        "MAXIMUM_PUBLISHABLE_BYTES",
        registered.stat().st_size - 1,
    )
    assert registered.stat().st_size <= MAXIMUM_PUBLISHABLE_BYTES
    assert is_registered_publishable_compact_json(registered) is False
