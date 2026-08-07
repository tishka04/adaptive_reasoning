from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from theory.sage_t import t10_2_1_artifact_inventory as inventory
from theory.sage_t import t10_2_1_protocol as protocol


def _signed_report(payload: dict[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    unsigned.pop("report_checksum", None)
    return {**unsigned, "report_checksum": inventory.canonical_sha256(unsigned)}


def _write_canonical(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        inventory.canonical_json(payload) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _source_report(
    *,
    status: str = "DATA_OR_PROVENANCE_INVALID",
    verdict: str = "DATA_OR_PROVENANCE_INVALID",
) -> dict[str, Any]:
    return _signed_report(
        {
            "format_version": protocol.FORMAT_VERSION,
            "phase": "source-train",
            "status": status,
            "verdict": verdict,
            "manifest_checksum": "a" * 64,
            "checks": {"source_gate_passed": False},
            "passed": False,
            "firewall": {
                "source_validation_opened": False,
                "ar25_opened": False,
                "holdout_opened": False,
                "production_authority": False,
            },
        }
    )


def _minimal_data_collection_report() -> dict[str, Any]:
    return _signed_report(
        {
            "format_version": protocol.FORMAT_VERSION,
            "phase": "collect",
            "status": "DATA_OR_PROVENANCE_INVALID",
            "manifest_checksum": "a" * 64,
            "games": list(protocol.SOURCE_GAMES),
            "splits": {
                "discovery": list(protocol.DISCOVERY_SEEDS),
                "leave_one_game_out_confirmation": list(
                    protocol.CONFIRMATION_SEEDS
                ),
            },
            "event_count": 0,
            "action_accounting": {
                "authorized_intent_count": 0,
                "sealed_event_count": 0,
                "explicitly_unresolved_intent_count": 0,
                "unknown_intent_count": 1,
                "maximum_authorized_intents": protocol.SOURCE_MAXIMUM_ACTIONS,
                "equation_holds": False,
            },
            "timing": {
                "cumulative_active_seconds": 0.0,
                "stop_new_actions_seconds": (
                    protocol.SOURCE_STOP_NEW_ACTIONS_SECONDS
                ),
                "absolute_seconds": protocol.SOURCE_MAXIMUM_WALL_SECONDS,
            },
            "events": {"available": False, "reason": "unreconstructible"},
            "cross_fit_audit": {
                "available": False,
                "reason": "unreconstructible",
            },
            "cross_fit_checks": {},
            "durability": {
                "journal_reconstructed": False,
                "checkpoint_reconstructed": False,
            },
            "checks": {
                "action_equation_holds": False,
                "no_unknown_intents": False,
            },
            "firewall": {
                "source_validation_opened": False,
                "ar25_opened": False,
                "holdout_opened": False,
                "production_authority": False,
            },
        }
    )


def _final_report(manifest_checksum: str) -> dict[str, Any]:
    return _signed_report(
        {
            "format_version": protocol.FORMAT_VERSION,
            "phase": "report",
            "status": "T10_2_1_COMPLETE",
            "manifest_checksum": manifest_checksum,
            "verdict": "DATA_OR_PROVENANCE_INVALID",
            "supported": False,
            "inputs": {},
            "artifact_inventory": {
                "inventory_sidecar": f"../{protocol.ARTIFACT_INVENTORY_FILENAME}",
                "binding_sidecar": (
                    f"../{protocol.REPORT_INVENTORY_BINDING_FILENAME}"
                ),
                "cyclic_hash_dependency": False,
            },
            "firewall": {
                "source_validation_opened": False,
                "ar25_opened": False,
                "holdout_opened": False,
                "production_authority": False,
            },
        }
    )


def _filesystem_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_inventory_contract_matches_protocol_namespace() -> None:
    assert inventory.PROTOCOL_FORMAT_VERSION == protocol.FORMAT_VERSION
    assert inventory.DEFAULT_ARTIFACT_ROOT == protocol.DEFAULT_OUTPUT_DIR
    assert inventory.DEFAULT_OUTPUT.name == protocol.ARTIFACT_INVENTORY_FILENAME
    assert (
        inventory.REPORT_INVENTORY_BINDING_NAME
        == protocol.REPORT_INVENTORY_BINDING_FILENAME
    )
    assert "training/sage_t/t10_2_gauge_posterior" in {
        path.as_posix() for path in inventory.FORBIDDEN_T10_2_DATA_PATHS
    }


def test_only_registered_canonical_reports_are_publishable(tmp_path: Path) -> None:
    artifact_root = tmp_path / protocol.DEFAULT_OUTPUT_DIR
    valid_path = artifact_root / "source_report.json"
    _write_canonical(valid_path, _source_report())
    assert inventory.is_registered_publishable_compact_json(valid_path)

    unsafe = _source_report()
    unsafe["checks"] = {"raw_grid": [[0, 1]]}
    unsafe = _signed_report(unsafe)
    unsafe_path = tmp_path / "unsafe" / "source_report.json"
    _write_canonical(unsafe_path, unsafe)
    assert not inventory.is_registered_publishable_compact_json(unsafe_path)

    referenced_parent_runtime = _source_report()
    referenced_parent_runtime["checks"] = {
        "input": "training/sage_t/t10_2_gauge_posterior/source_events.jsonl"
    }
    referenced_parent_runtime = _signed_report(referenced_parent_runtime)
    parent_data_path = tmp_path / "parent-data" / "source_report.json"
    _write_canonical(parent_data_path, referenced_parent_runtime)
    assert not inventory.is_registered_publishable_compact_json(parent_data_path)

    noncanonical_path = tmp_path / "noncanonical" / "source_report.json"
    noncanonical_path.parent.mkdir(parents=True)
    noncanonical_path.write_text(
        json.dumps(_source_report(), indent=2), encoding="utf-8"
    )
    assert not inventory.is_registered_publishable_compact_json(noncanonical_path)


def test_terminal_acquisition_miss_is_a_publishable_compact_source_report(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "source_report.json"
    _write_canonical(
        report_path,
        _source_report(
            status="SOURCE_ACQUISITION_OR_RESOURCE_MISS",
            verdict="SOURCE_ACQUISITION_OR_RESOURCE_MISS",
        ),
    )

    assert inventory.is_registered_publishable_compact_json(report_path)


def test_minimal_terminal_data_collection_report_remains_publishable(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "collection_report.json"
    _write_canonical(report_path, _minimal_data_collection_report())

    assert inventory.is_registered_publishable_compact_json(report_path)


def test_inventory_is_complete_idempotent_and_non_destructive(tmp_path: Path) -> None:
    artifact_root = tmp_path / protocol.DEFAULT_OUTPUT_DIR
    output_path = tmp_path / inventory.DEFAULT_OUTPUT
    _write_canonical(artifact_root / "source_report.json", _source_report())
    (artifact_root / "source_events.jsonl").write_text(
        '{"event_id":"one"}\n', encoding="utf-8"
    )
    (artifact_root / "journals").mkdir(parents=True)
    (artifact_root / "journals" / "action_journal.jsonl").write_text(
        '{"intent_id":"intent-one"}\n', encoding="utf-8"
    )
    (artifact_root / "checkpoints").mkdir(parents=True)
    (artifact_root / "checkpoints" / "lane.ckpt").write_bytes(b"checkpoint")
    (artifact_root / "cache").mkdir(parents=True)
    (artifact_root / "cache" / "replay.cache").write_bytes(b"regenerable")
    before = _filesystem_bytes(artifact_root)

    first = inventory.build_inventory(
        repository_root=tmp_path,
        artifact_root=protocol.DEFAULT_OUTPUT_DIR,
        output_path=inventory.DEFAULT_OUTPUT,
    )
    included = {item["path"] for item in first["included_compact_reports"]}
    omitted = {item["path"] for item in first["omitted_artifacts"]}

    assert included == {
        (protocol.DEFAULT_OUTPUT_DIR / "source_report.json").as_posix()
    }
    assert omitted == {
        (protocol.DEFAULT_OUTPUT_DIR / "source_events.jsonl").as_posix(),
        (
            protocol.DEFAULT_OUTPUT_DIR / "journals" / "action_journal.jsonl"
        ).as_posix(),
        (protocol.DEFAULT_OUTPUT_DIR / "checkpoints" / "lane.ckpt").as_posix(),
        (protocol.DEFAULT_OUTPUT_DIR / "cache" / "replay.cache").as_posix(),
    }
    assert first["budget_audit"]["passed"] is True
    assert first["policy"]["files_deleted"] == 0
    assert first["summary"]["files_deleted"] == 0
    unsigned = dict(first)
    observed_checksum = unsigned.pop("inventory_checksum")
    assert observed_checksum == inventory.canonical_sha256(unsigned)

    written = inventory.write_inventory(
        first,
        repository_root=tmp_path,
        output_path=inventory.DEFAULT_OUTPUT,
    )
    first_bytes = written.read_bytes()
    assert first_bytes == (inventory.canonical_json(first) + "\n").encode("utf-8")
    assert len(first_bytes) <= inventory.MAXIMUM_DERIVED_FILE_BYTES
    assert _filesystem_bytes(artifact_root) == before

    second = inventory.build_inventory(
        repository_root=tmp_path,
        artifact_root=protocol.DEFAULT_OUTPUT_DIR,
        output_path=inventory.DEFAULT_OUTPUT,
    )
    assert second == first
    inventory.write_inventory(
        second,
        repository_root=tmp_path,
        output_path=inventory.DEFAULT_OUTPUT,
    )
    assert written.read_bytes() == first_bytes
    assert _filesystem_bytes(artifact_root) == before


def test_inventory_detects_budget_overrun_without_deleting_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / protocol.DEFAULT_OUTPUT_DIR
    ledger = artifact_root / "source_events.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_bytes(b"12345")
    monkeypatch.setitem(inventory.ARTIFACT_BYTE_LIMITS, "ledger", 4)

    payload = inventory.build_inventory(
        repository_root=tmp_path,
        artifact_root=protocol.DEFAULT_OUTPUT_DIR,
        output_path=inventory.DEFAULT_OUTPUT,
    )

    assert payload["budget_audit"]["passed"] is False
    assert payload["budget_audit"]["violations"] == [
        {
            "path": (protocol.DEFAULT_OUTPUT_DIR / "source_events.jsonl").as_posix(),
            "kind": "ledger",
            "bytes": 5,
            "maximum_bytes": 4,
        }
    ]
    assert ledger.read_bytes() == b"12345"


def test_inventory_refuses_prior_t10_2_runtime_root(tmp_path: Path) -> None:
    forbidden_root = inventory.FORBIDDEN_T10_2_DATA_ROOTS[0]
    (tmp_path / forbidden_root).mkdir(parents=True)

    with pytest.raises(ValueError, match="T10.2 runtime artifacts are forbidden"):
        inventory.build_inventory(
            repository_root=tmp_path,
            artifact_root=forbidden_root,
            output_path=inventory.DEFAULT_OUTPUT,
        )


def test_inventory_rejects_stale_reconstruction_without_mutation(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / protocol.DEFAULT_OUTPUT_DIR
    ledger = artifact_root / "source_events.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_bytes(b"first")
    payload = inventory.build_inventory(
        repository_root=tmp_path,
        artifact_root=protocol.DEFAULT_OUTPUT_DIR,
        output_path=inventory.DEFAULT_OUTPUT,
    )
    ledger.write_bytes(b"second")
    before_sha256 = hashlib.sha256(ledger.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="does not reconstruct"):
        inventory.write_inventory(
            payload,
            repository_root=tmp_path,
            output_path=inventory.DEFAULT_OUTPUT,
        )

    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == before_sha256
    assert not (tmp_path / inventory.DEFAULT_OUTPUT).exists()


def test_publication_binding_is_required_idempotent_and_tamper_evident(
    tmp_path: Path,
) -> None:
    manifest = {"manifest_checksum": "f" * 64}
    artifact_root = tmp_path / protocol.DEFAULT_OUTPUT_DIR
    report_path = artifact_root / "report.json"
    inventory_path = tmp_path / inventory.DEFAULT_OUTPUT
    binding_path = artifact_root.parent / inventory.REPORT_INVENTORY_BINDING_NAME
    _write_canonical(report_path, _final_report(manifest["manifest_checksum"]))

    payload = inventory.build_inventory(
        repository_root=tmp_path,
        artifact_root=protocol.DEFAULT_OUTPUT_DIR,
        output_path=inventory.DEFAULT_OUTPUT,
    )
    inventory.write_inventory(
        payload,
        repository_root=tmp_path,
        output_path=inventory.DEFAULT_OUTPUT,
    )
    with pytest.raises(protocol.ManifestDriftError, match="signed JSON"):
        protocol.verify_publication_binding(
            manifest=manifest,
            destination=artifact_root,
            repo_root=tmp_path,
        )

    binding = protocol.signed_payload(
        {
            "format_version": inventory.REPORT_INVENTORY_BINDING_FORMAT_VERSION,
            "manifest_checksum": manifest["manifest_checksum"],
            "report": protocol._t10_2.artifact_descriptor(report_path),
            "inventory": protocol._t10_2.artifact_descriptor(inventory_path),
            "inventory_checksum": payload["inventory_checksum"],
            "cycle_free": True,
        },
        checksum_key="binding_checksum",
    )
    _write_canonical(binding_path, binding)

    first = protocol.verify_publication_binding(
        manifest=manifest,
        destination=artifact_root,
        repo_root=tmp_path,
    )
    second = protocol.verify_publication_binding(
        manifest=manifest,
        destination=artifact_root,
        repo_root=tmp_path,
    )
    assert first == second == binding
    assert inventory.is_registered_publishable_compact_json(binding_path)

    omitted = artifact_root / "cache" / "late.cache"
    omitted.parent.mkdir(parents=True)
    omitted.write_bytes(b"filesystem drift")
    with pytest.raises(protocol.ManifestDriftError, match="did not reconstruct"):
        protocol.verify_publication_binding(
            manifest=manifest,
            destination=artifact_root,
            repo_root=tmp_path,
        )
