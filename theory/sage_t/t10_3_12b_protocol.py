"""Preregistered protocol for SAGE.T10.3.12b factor identification."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import t10_3_12_protocol as parent
from .factorial_invariants_v10_3_12b import ARMS, CONTEXTS, FACTORS, sha256_payload

FORMAT_VERSION = "sage-t10.3.12b-factor-identification-protocol-v1"
MANIFEST_STATUS = "FROZEN_BEFORE_T10_3_12B_COUNTERFACTUAL_EVALUATION"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name("sage_t10_3_12b_protocol_manifest.json")
DEFAULT_FREEZE_RECEIPT_PATH = Path(__file__).with_name("sage_t10_3_12b_freeze_receipt.json")
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "t10_3_12b_factorial_invariant_identification"

IntegrityError = parent.IntegrityError
ScientificGateMiss = parent.ScientificGateMiss
file_sha256 = parent.file_sha256
verify_signed = parent.verify_signed
write_json_once = parent.write_json_once

PARENT_OUTPUT_DIR = Path("training") / "sage_t" / "t10_3_12_relational_mechanism_invariance"
PARENT_ARTIFACT_PATHS = {
    "manifest": Path("theory/sage_t/sage_t10_3_12_protocol_manifest.json"),
    "freeze_receipt": Path("theory/sage_t/sage_t10_3_12_freeze_receipt.json"),
    "candidate_registry": PARENT_OUTPUT_DIR / "candidate_registry.json",
    "offline_report": PARENT_OUTPUT_DIR / "offline_equivariance_report.json",
    "active_report": PARENT_OUTPUT_DIR / "active_core_report.json",
    "adjudication": PARENT_OUTPUT_DIR / "adjudication_report.json",
    "terminal": PARENT_OUTPUT_DIR / "terminal_report.json",
}

EXPECTED_PARENT = {
    "verdict": "GENERIC_REDISCOVERY_ONLY",
    "authorized_actions": 336,
    "sealed_events": 336,
    "inflight_intents": 0,
    "unresolved_intents": 0,
    "sequence_games_opened": False,
    "production_authority": False,
    "physical_actions_replayed": 0,
}

ARTIFACT_CONTRACT = {
    "audit-parent": {
        "path": "parent_generic_rediscovery_audit.json",
        "checksum_field": "audit_checksum",
        "gate_field": "passed",
    },
    "preflight": {
        "path": "factor_preflight.json",
        "checksum_field": "preflight_checksum",
        "gate_field": "passed",
    },
    "materialize-variants": {
        "path": "counterfactual_variant_inventory.json",
        "checksum_field": "inventory_checksum",
        "gate_field": "passed",
    },
    "compile-factors": {
        "path": "factor_registry.json",
        "checksum_field": "registry_checksum",
        "gate_field": None,
    },
    "evaluate-interventions": {
        "path": "factorial_intervention_report.json",
        "checksum_field": "report_checksum",
        "gate_field": "passed",
    },
    "adjudicate": {
        "path": "factor_adjudication_report.json",
        "checksum_field": "report_checksum",
        "gate_field": "passed",
    },
    "report": {
        "path": "terminal_report.json",
        "checksum_field": "report_checksum",
        "gate_field": None,
    },
}


def _signed(payload: Mapping[str, Any], checksum_field: str) -> dict[str, Any]:
    result = dict(payload)
    result[checksum_field] = sha256_payload(result)
    return result


def parent_artifact_bindings(root: Path) -> dict[str, dict[str, str]]:
    bindings: dict[str, dict[str, str]] = {}
    for name, relative in PARENT_ARTIFACT_PATHS.items():
        path = root / relative
        if not path.is_file():
            raise IntegrityError(f"required T10.3.12 parent artifact is absent: {name}")
        bindings[name] = {"path": relative.as_posix(), "sha256": file_sha256(path)}
    return bindings


def parent_journal_digest(root: Path) -> str:
    journal = root / PARENT_OUTPUT_DIR / "journal"
    if not journal.is_dir():
        raise IntegrityError("T10.3.12 parent journal is absent")
    rows = [
        {
            "relative_path": path.relative_to(journal).as_posix(),
            "sha256": file_sha256(path),
        }
        for path in sorted(item for item in journal.rglob("*.json") if item.is_file())
    ]
    if not rows:
        raise IntegrityError("T10.3.12 parent journal is empty")
    return sha256_payload(rows)


def _verify_parent_terminal(root: Path) -> dict[str, Any]:
    parent.load_manifest(root)
    path = root / PARENT_ARTIFACT_PATHS["terminal"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_signed(payload, "report_checksum")
    accounting = payload.get("accounting", {})
    observed = {
        "verdict": payload.get("verdict"),
        "authorized_actions": accounting.get("authorized_actions"),
        "sealed_events": accounting.get("sealed_events"),
        "inflight_intents": accounting.get("inflight_intents"),
        "unresolved_intents": accounting.get("unresolved_intents"),
        "sequence_games_opened": payload.get("sequence_games_opened"),
        "production_authority": payload.get("production_authority"),
        "physical_actions_replayed": payload.get("physical_actions_replayed"),
    }
    if observed != EXPECTED_PARENT:
        raise IntegrityError("T10.3.12 terminal state does not match the frozen negative result")
    if not bool(accounting.get("equation_holds")) or not bool(accounting.get("inflight_valid")):
        raise IntegrityError("T10.3.12 parent accounting is not clean")
    return payload


def _code_hashes(root: Path) -> dict[str, str]:
    relative = (
        "theory/sage_t/factorial_invariants_v10_3_12b.py",
        "theory/sage_t/t10_3_12b_protocol.py",
        "theory/sage_t/t10_3_12b_runtime.py",
        "tests/test_sage_t_factorial_invariants_v10_3_12b.py",
        "tests/test_sage_t_t10_3_12b_protocol.py",
        "tests/test_sage_t_t10_3_12b_runtime.py",
        "reports/SAGE_T10_3_12B_FACTOR_IDENTIFICATION_PROTOCOL.md",
        "reports/SAGE_T10_3_12B_FACTOR_IDENTIFICATION_RUNBOOK.md",
    )
    output: dict[str, str] = {}
    for item in relative:
        path = root / item
        if not path.is_file():
            raise IntegrityError(f"T10.3.12b protocol dependency is absent: {item}")
        output[item] = file_sha256(path)
    return output


def build_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    parent_terminal = _verify_parent_terminal(root)
    bindings = parent_artifact_bindings(root)
    core = {
        "format_version": FORMAT_VERSION,
        "status": MANIFEST_STATUS,
        "objective": "identify_transportable_factor_candidates_from_repeat_and_path_successes",
        "hypothesis": (
            "operator role transition and termination factors remain necessary under "
            "counterfactual variation; in particular the source role proxy must survive "
            "a preregistered structural-causal decoupling"
        ),
        "claim_boundary": {
            "factor_candidates_identifiable": True,
            "cross_game_generalization_proven": False,
            "sequence_composition_authorized": False,
            "production_authority": False,
        },
        "code_hashes": _code_hashes(root),
        "parent_artifacts": bindings,
        "parent_manifest_checksum": parent_terminal["manifest_checksum"],
        "parent_terminal_checksum": parent_terminal["report_checksum"],
        "parent_journal_digest": parent_journal_digest(root),
        "parent_expected": EXPECTED_PARENT,
        "cli_phases": [
            "freeze",
            "status",
            "audit-parent",
            "preflight",
            "materialize-variants",
            "compile-factors",
            "evaluate-interventions",
            "adjudicate",
            "report",
        ],
        "exit_codes": {"success": 0, "integrity": 2, "scientific_gate": 3},
        "artifact_contract": ARTIFACT_CONTRACT,
        "matrix": {
            "contexts": list(CONTEXTS),
            "factors": list(FACTORS),
            "arms": list(ARMS),
            "variants": 128,
            "identification_variants": 64,
            "challenge_variants": 64,
            "ambiguous_variants": 32,
            "maximum_virtual_actions": 32_768,
            "maximum_wall_seconds": 600,
            "maximum_artifact_bytes": 5 * 1024 * 1024,
            "physical_actions": 0,
        },
        "gates": {
            "preflight_cases": 12,
            "full_source_correct": 128,
            "generic_correct": 128,
            "ambiguity_correct": 32,
            "source_role_decoupled_correct": 32,
            "minimum_distinct_state_hashes_per_context": 48,
            "minimum_factor_gap_per_context": 8,
            "minimum_factor_gap_per_challenge_context": 4,
            "maximum_source_to_generic_action_ratio": 0.80,
            "minimum_first_decision_divergence": 96,
            "program_hashes_per_arm_context": 1,
        },
        "firewall": {
            "sequence_games_opened": False,
            "source_validation_opened": False,
            "ar25_opened": False,
            "holdout_opened": False,
            "production_authority": False,
            "automatic_retuning": False,
            "new_arc_physical_actions_authorized": False,
            "t10_3_12_actions_used_for_training": False,
            "t10_3_12_registry_support_imported": False,
            "t10_3_12_grounded_arguments_compiled": False,
            "parent_actions_read_for_diagnostics_only": True,
        },
        "negative_result_policy": {
            "no_post_freeze_repair": True,
            "generic_tie_verdict": "GENERIC_PRIOR_ONLY",
            "factor_miss_is_negative": True,
            "no_program_promotion": True,
            "pass_authorizes_only_cross_game_preregistration": True,
        },
        "output_directory": DEFAULT_OUTPUT_DIR.as_posix(),
    }
    return _signed(core, "manifest_checksum")


def freeze_manifest(
    root: Path,
    *,
    manifest_path: Path | None = None,
    receipt_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    manifest = build_manifest(root)
    write_json_once(manifest_path or root / DEFAULT_MANIFEST_PATH, manifest)
    receipt = _signed(
        {
            "format_version": "sage-t10.3.12b-freeze-receipt-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "parent_terminal_checksum": manifest["parent_terminal_checksum"],
            "parent_journal_digest": manifest["parent_journal_digest"],
            "physical_actions": 0,
            "sequence_games_opened": False,
            "production_authority": False,
        },
        "receipt_checksum",
    )
    write_json_once(receipt_path or root / DEFAULT_FREEZE_RECEIPT_PATH, receipt)
    return manifest, receipt


def load_manifest(root: Path, *, verify_code: bool = True) -> dict[str, Any]:
    root = root.resolve()
    path = root / DEFAULT_MANIFEST_PATH
    if not path.is_file():
        raise IntegrityError("T10.3.12b manifest has not been frozen")
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_signed(payload, "manifest_checksum")
    if payload.get("format_version") != FORMAT_VERSION:
        raise IntegrityError("T10.3.12b manifest format drifted")
    _verify_parent_terminal(root)
    for name, binding in payload.get("parent_artifacts", {}).items():
        artifact = root / str(binding["path"])
        if not artifact.is_file() or file_sha256(artifact) != binding["sha256"]:
            raise IntegrityError(f"T10.3.12 parent artifact drifted: {name}")
    if payload.get("parent_journal_digest") != parent_journal_digest(root):
        raise IntegrityError("T10.3.12 parent journal changed after T10.3.12b freeze")
    if verify_code and payload.get("code_hashes") != _code_hashes(root):
        raise IntegrityError("T10.3.12b code changed after freeze")
    receipt_path = root / DEFAULT_FREEZE_RECEIPT_PATH
    if not receipt_path.is_file():
        raise IntegrityError("T10.3.12b freeze receipt is absent")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    verify_signed(receipt, "receipt_checksum")
    if receipt.get("manifest_checksum") != payload.get("manifest_checksum"):
        raise IntegrityError("T10.3.12b freeze receipt is detached")
    return payload


__all__ = [
    "ARTIFACT_CONTRACT",
    "DEFAULT_FREEZE_RECEIPT_PATH",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "EXPECTED_PARENT",
    "FORMAT_VERSION",
    "IntegrityError",
    "MANIFEST_STATUS",
    "PARENT_ARTIFACT_PATHS",
    "PARENT_OUTPUT_DIR",
    "ScientificGateMiss",
    "build_manifest",
    "file_sha256",
    "freeze_manifest",
    "load_manifest",
    "parent_artifact_bindings",
    "parent_journal_digest",
    "verify_signed",
    "write_json_once",
]
