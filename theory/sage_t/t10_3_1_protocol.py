"""Append-only correction protocol for the SAGE.T10.3 ROOTING_MISS."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import t10_2_1_protocol as _kernel
from . import t10_3_protocol as _parent

FORMAT_VERSION = "sage-t10.3.1-goal-progress-correction-protocol-v1"
HANDOFF_FORMAT_VERSION = "sage-t10.3.1-migration-receipt-v1"
MANIFEST_STATUS = "FROZEN_BEFORE_T10_3_1_SOURCE_ACTION"

PARENT_MANIFEST_CHECKSUM = "11b9f72f512802f45cb5d637aae3e5e6a3e644b7b9ad281a30a373c2b8023551"
PARENT_AUDIT_CHECKSUM = "c19905e98a04452290401293a141bd0fabf287c12461c1e34abb934606905b4a"
PARENT_CHECKPOINT_CHECKSUM = "ceb27ce3319de1e71d20d6ccfb590e9165c042ea49b7a44dfdc23ee43a785710"
PARENT_QA_CHECKSUM = "cd69d478391b407838ee69df3deb819800af1e295f2c67ef2e8c20d4f36d23d2"
PARENT_TERMINAL_CHECKSUM = "3b71e4b8e76ee9d016c070b2c8fbcb1a2a488412110b53a6d3a0f73e419f2d56"

SOURCE_GAMES = _parent.SOURCE_GAMES
POSITIVE_WITNESS_GAMES = _parent.POSITIVE_WITNESS_GAMES
PANEL_SEEDS = (3121, 3122, 3123, 3124)
CONFIRMATION_SEEDS = (3131, 3132)
PANEL_ARMS = _parent.PANEL_ARMS
CONFIRMATION_CONTROLLERS = _parent.CONFIRMATION_CONTROLLERS
MAXIMUM_ACTIONS_PER_RESET = 16
PANEL_RESETS = 48
CONFIRMATION_RESETS = 12
PANEL_MAXIMUM_ACTIONS = 768
CONFIRMATION_MAXIMUM_ACTIONS = 192
TOTAL_RESETS = 60
TOTAL_MAXIMUM_ACTIONS = 960
PHASES = _parent.PHASES
EXCLUSIVE_VERDICTS = _parent.EXCLUSIVE_VERDICTS

DEFAULT_MANIFEST_RELATIVE_PATH = Path("theory/sage_t/sage_t10_3_1_protocol_manifest.json")
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(DEFAULT_MANIFEST_RELATIVE_PATH.name)
DEFAULT_HANDOFF_RELATIVE_PATH = Path("theory/sage_t/sage_t10_3_1_migration_receipt.json")
DEFAULT_HANDOFF_PATH = Path(__file__).with_name(DEFAULT_HANDOFF_RELATIVE_PATH.name)
DEFAULT_OUTPUT_ROOT = Path("training") / "sage_t" / "t10_3_1_goal_progress_correction"
PARENT_OUTPUT_ROOT = _parent.DEFAULT_OUTPUT_ROOT

DEFAULT_CODE_FILES = (
    "theory/sage_t/frame_adapters_v10_3_1.py",
    "theory/sage_t/t10_3_1_protocol.py",
    "theory/sage_t/t10_3_1_runtime.py",
    "tests/test_sage_t_frame_adapters_v10_3_1.py",
    "tests/test_sage_t_t10_3_1_protocol.py",
    "tests/test_sage_t_t10_3_1_runtime.py",
)
DEFAULT_DOCUMENT_FILES = (
    "reports/SAGE_T10_3_1_GOAL_PROGRESS_CORRECTION_PROTOCOL.md",
    "reports/SAGE_T10_3_1_GOAL_PROGRESS_CORRECTION_RUNBOOK.md",
)

canonical_json = _kernel.canonical_json
canonical_sha256 = _kernel.canonical_sha256
signed_payload = _kernel.signed_payload
write_compact_json = _kernel.write_compact_json
_read_signed_json = _kernel._read_signed_json
ManifestDriftError = _kernel.ManifestDriftError
ProtocolError = _kernel.ProtocolError


def _root(repo_root: str | Path | None) -> Path:
    return Path(repo_root or _kernel._repo_root()).resolve()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_descriptor(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ProtocolError(f"required artifact is absent: {path}")
    return {"bytes": path.stat().st_size, "sha256": _file_sha256(path)}


def correction_policy() -> dict[str, Any]:
    return {
        "classification": "IMPLEMENTATION_AND_ACQUISITION_CORRECTION",
        "parent_t10_3_immutable": True,
        "parent_physical_events_fit_authorized": False,
        "parent_physical_events_relabel_authorized": False,
        "fresh_panel_required": True,
        "fresh_confirmation_required": True,
        "dynamic_action_regrounding_each_step": True,
        "pre_action_rooting_separate_from_after_continuation": True,
        "post_action_effect_root_inference_authorized": False,
        "common_quotient_transport_pair": [
            "allocentric_object_relative",
            "action_aligned_relational",
        ],
        "root_only_to_richer_frames_comparable": False,
        "intent_accounting_separate_from_branch_label_completeness": True,
        "automatic_retuning_authorized": False,
        "physical_replay_authorized": False,
    }


def scientific_gates() -> dict[str, Any]:
    gates = _parent.scientific_gates()
    gates = json.loads(canonical_json(gates))
    gates["qa"].update(
        {
            "root_correspondence_definition": "unique_pre_action_binding",
            "branch_label_completeness_required": True,
            "intent_accounting_equation_required": True,
            "comparable_transport_definition": "exact_orientation_erased_common_quotient",
            "terminal_after_root_absence_penalizes_rooting": False,
        }
    )
    return gates


def frozen_matrix() -> dict[str, Any]:
    return {
        "games": list(SOURCE_GAMES),
        "panel": {
            "seeds": list(PANEL_SEEDS),
            "arms": list(PANEL_ARMS),
            "resets": PANEL_RESETS,
            "maximum_actions": PANEL_MAXIMUM_ACTIONS,
        },
        "confirmation": {
            "seeds": list(CONFIRMATION_SEEDS),
            "controllers": list(CONFIRMATION_CONTROLLERS),
            "counterbalanced": True,
            "resets": CONFIRMATION_RESETS,
            "maximum_actions": CONFIRMATION_MAXIMUM_ACTIONS,
        },
        "maximum_actions_per_reset": MAXIMUM_ACTIONS_PER_RESET,
        "total_resets": TOTAL_RESETS,
        "total_maximum_actions": TOTAL_MAXIMUM_ACTIONS,
        "padding_authorized": False,
        "seed_replacement_authorized": False,
        "adaptive_substitution_authorized": False,
    }


def firewall_policy() -> dict[str, Any]:
    return {
        "source_games_only": list(SOURCE_GAMES),
        "source_validation_authorized": False,
        "ar25_authorized": False,
        "holdout_authorized": False,
        "production_authority": False,
        "automatic_retuning_authorized": False,
        "automatic_next_protocol_opening_authorized": False,
        "parent_t10_3_events_fit_authorized": False,
        "physical_replay_authorized": False,
    }


def artifact_contract() -> dict[str, Any]:
    return {
        "output_root": DEFAULT_OUTPUT_ROOT.as_posix(),
        "offline_audit": "offline_audit.json",
        "journal": "journal",
        "checkpoint": "checkpoint.json",
        "compiled_ledger": "compiled_source_events.jsonl",
        "compact_ledger": "compact_ledger.json",
        "qa_report": "qa_report.json",
        "model_recipe": "model_recipe.json",
        "confirmation_report": "confirmation_report.json",
        "terminal_report": "t10_3_1_report.json",
        "write_once": True,
        "intent_before_action": True,
        "event_sealed_immediately": True,
        "branch_label_receipt_after_reset": True,
    }


def _load_parent_evidence(root: Path) -> dict[str, Any]:
    manifest = _parent.load_manifest(repo_root=root)
    if manifest.get("manifest_checksum") != PARENT_MANIFEST_CHECKSUM:
        raise ManifestDriftError("T10.3 parent manifest drifted")
    destination = root / PARENT_OUTPUT_ROOT
    audit = _read_signed_json(destination / "offline_audit.json", checksum_key="audit_checksum")
    checkpoint = _read_signed_json(destination / "checkpoint.json", checksum_key="checkpoint_checksum")
    qa = _read_signed_json(destination / "qa_report.json", checksum_key="report_checksum")
    terminal = _read_signed_json(destination / "t10_3_report.json", checksum_key="terminal_checksum")
    if audit.get("audit_checksum") != PARENT_AUDIT_CHECKSUM or audit.get("passed") is not True:
        raise ManifestDriftError("T10.3 audit drifted")
    if checkpoint.get("checkpoint_checksum") != PARENT_CHECKPOINT_CHECKSUM:
        raise ManifestDriftError("T10.3 checkpoint drifted")
    if (
        checkpoint.get("authorized_intent_count") != 540
        or checkpoint.get("sealed_event_count") != 540
        or checkpoint.get("explicitly_unresolved_intent_count") != 0
        or checkpoint.get("equation_holds") is not True
        or checkpoint.get("physical_actions_replayed") != 0
    ):
        raise ManifestDriftError("T10.3 physical accounting drifted")
    if qa.get("report_checksum") != PARENT_QA_CHECKSUM or qa.get("passed") is not False:
        raise ManifestDriftError("T10.3 QA drifted")
    metrics = qa.get("metrics", {})
    if (
        metrics.get("event_count") != 540
        or metrics.get("goal_reachable_positive_count") != 60
        or metrics.get("unknown_target_count") != 140
        or metrics.get("exact_nonidentity_event_count") != 0
    ):
        raise ManifestDriftError("T10.3 diagnosis metrics drifted")
    if (
        terminal.get("terminal_checksum") != PARENT_TERMINAL_CHECKSUM
        or terminal.get("verdict") != "ROOTING_MISS"
        or terminal.get("passed") is not False
        or terminal.get("model_recipe_checksum") is not None
        or terminal.get("confirmation_report_checksum") is not None
    ):
        raise ManifestDriftError("T10.3 fail-closed terminal drifted")
    receipts = []
    branch_root = destination / "journal" / "branches"
    for path in sorted(branch_root.rglob("receipt.json")):
        receipts.append(_read_signed_json(path, checksum_key="receipt_checksum"))
    if len(receipts) != 48:
        raise ManifestDriftError("T10.3 branch receipt count drifted")
    canonical = [
        row
        for row in receipts
        if row.get("controller") == "canonical_option"
        and row.get("game_id") in POSITIVE_WITNESS_GAMES
    ]
    controls = [
        row
        for row in receipts
        if row.get("controller") != "canonical_option"
        and row.get("goal_reachable_within_option") is not None
    ]
    if len(canonical) != 8 or not all(row.get("goal_reachable_within_option") is True for row in canonical):
        raise ManifestDriftError("T10.3 positive witness reproduction drifted")
    if any(row.get("goal_reachable_within_option") is True for row in controls):
        raise ManifestDriftError("T10.3 negative controls drifted")
    return {
        "manifest": manifest,
        "audit": audit,
        "checkpoint": checkpoint,
        "qa": qa,
        "terminal": terminal,
        "receipts": receipts,
    }


def build_migration_receipt(*, repo_root: str | Path | None = None) -> dict[str, Any]:
    root = _root(repo_root)
    evidence = _load_parent_evidence(root)
    destination = root / PARENT_OUTPUT_ROOT
    parent_manifest = evidence["manifest"]
    return signed_payload(
        {
            "format_version": HANDOFF_FORMAT_VERSION,
            "parent_manifest_checksum": parent_manifest["manifest_checksum"],
            "parent_audit": artifact_descriptor(destination / "offline_audit.json"),
            "parent_audit_checksum": evidence["audit"]["audit_checksum"],
            "parent_checkpoint": artifact_descriptor(destination / "checkpoint.json"),
            "parent_checkpoint_checksum": evidence["checkpoint"]["checkpoint_checksum"],
            "parent_qa": artifact_descriptor(destination / "qa_report.json"),
            "parent_qa_checksum": evidence["qa"]["report_checksum"],
            "parent_terminal": artifact_descriptor(destination / "t10_3_report.json"),
            "parent_terminal_checksum": evidence["terminal"]["terminal_checksum"],
            "parent_compiled_ledger": artifact_descriptor(destination / "compiled_source_events.jsonl"),
            "parent_diagnosis": {
                "verdict": "ROOTING_MISS",
                "event_count": 540,
                "positive_event_count": 60,
                "unknown_target_count": 140,
                "exact_nonidentity_event_count": 0,
                "canonical_positive_branches": 8,
                "positive_control_branches": 0,
                "physical_actions_replayed": 0,
            },
            "canonical_witnesses": list(parent_manifest["handoff_receipt"]["canonical_witnesses"]),
            "parent_events_used_for_fit": 0,
            "parent_events_relabelled": 0,
            "fresh_panel_seeds": list(PANEL_SEEDS),
            "fresh_confirmation_seeds": list(CONFIRMATION_SEEDS),
            "correction_policy": correction_policy(),
        },
        checksum_key="receipt_checksum",
    )


def verify_migration_receipt_live(
    receipt: Mapping[str, Any], *, repo_root: str | Path | None = None
) -> dict[str, Any]:
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_checksum"}
    if canonical_sha256(unsigned) != receipt.get("receipt_checksum"):
        raise ManifestDriftError("T10.3.1 migration receipt checksum drifted")
    if dict(receipt) != build_migration_receipt(repo_root=repo_root):
        raise ManifestDriftError("T10.3.1 live migration evidence changed")
    return {
        "migration_verified": True,
        "parent_events_used_for_fit": 0,
        "fresh_panel_required": True,
    }


def build_manifest(
    *, repo_root: str | Path | None = None, migration_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    root = _root(repo_root)
    verify_migration_receipt_live(migration_receipt, repo_root=root)
    return signed_payload(
        {
            "format_version": FORMAT_VERSION,
            "status": MANIFEST_STATUS,
            "hash_algorithm": _kernel.HASH_ALGORITHM,
            "registered_phases": list(PHASES),
            "portable_code_sha256": _kernel._hash_paths(root, DEFAULT_CODE_FILES, portable=True),
            "document_sha256": _kernel._hash_paths(root, DEFAULT_DOCUMENT_FILES, portable=True),
            "frozen_matrix": frozen_matrix(),
            "scientific_gates": scientific_gates(),
            "firewall": firewall_policy(),
            "correction_policy": correction_policy(),
            "artifact_contract": artifact_contract(),
            "exclusive_verdicts": list(EXCLUSIVE_VERDICTS),
            "migration_receipt": dict(migration_receipt),
        },
        checksum_key="manifest_checksum",
    )


def freeze_manifest(
    *,
    output_path: str | Path = DEFAULT_MANIFEST_PATH,
    handoff_path: str | Path = DEFAULT_HANDOFF_PATH,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    root = _root(repo_root)
    receipt = build_migration_receipt(repo_root=root)
    manifest = build_manifest(repo_root=root, migration_receipt=receipt)
    receipt_path = Path(handoff_path)
    manifest_path = Path(output_path)
    if not receipt_path.is_absolute():
        receipt_path = root / receipt_path
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    if receipt_path.exists() or manifest_path.exists():
        existing = load_manifest(manifest_path, repo_root=root)
        if existing != manifest:
            raise ManifestDriftError("frozen T10.3.1 manifest is immutable")
        return existing
    write_compact_json(receipt_path, receipt)
    write_compact_json(manifest_path, manifest)
    return manifest


def load_manifest(
    path: str | Path = DEFAULT_MANIFEST_PATH,
    *,
    repo_root: str | Path | None = None,
    verify_repository: bool = True,
    verify_live_handoff: bool = True,
) -> dict[str, Any]:
    root = _root(repo_root)
    source = Path(path)
    if not source.is_absolute():
        source = root / source
    manifest = _read_signed_json(source, checksum_key="manifest_checksum")
    if manifest.get("format_version") != FORMAT_VERSION or manifest.get("status") != MANIFEST_STATUS:
        raise ManifestDriftError("T10.3.1 manifest identity drifted")
    for key, expected in (
        ("frozen_matrix", frozen_matrix()),
        ("scientific_gates", scientific_gates()),
        ("firewall", firewall_policy()),
        ("correction_policy", correction_policy()),
        ("artifact_contract", artifact_contract()),
    ):
        if manifest.get(key) != expected:
            raise ManifestDriftError(f"T10.3.1 {key} drifted")
    materialized = _read_signed_json(
        root / DEFAULT_HANDOFF_RELATIVE_PATH, checksum_key="receipt_checksum"
    )
    if manifest.get("migration_receipt") != materialized:
        raise ManifestDriftError("materialized T10.3.1 migration receipt drifted")
    if verify_repository:
        if manifest.get("portable_code_sha256") != _kernel._hash_paths(root, DEFAULT_CODE_FILES, portable=True):
            raise ManifestDriftError("T10.3.1 code bytes drifted")
        if manifest.get("document_sha256") != _kernel._hash_paths(root, DEFAULT_DOCUMENT_FILES, portable=True):
            raise ManifestDriftError("T10.3.1 documentation bytes drifted")
    if verify_live_handoff:
        verify_migration_receipt_live(materialized, repo_root=root)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "status"))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--repo-root", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.phase == "freeze":
            payload = freeze_manifest(output_path=args.manifest, repo_root=args.repo_root)
        else:
            manifest = load_manifest(args.manifest, repo_root=args.repo_root)
            payload = {
                "status": "READY_T10_3_1_OFFLINE_AUDIT",
                "manifest_checksum": manifest["manifest_checksum"],
                "maximum_actions": TOTAL_MAXIMUM_ACTIONS,
                "firewall": manifest["firewall"],
            }
    except (ProtocolError, OSError, ValueError, KeyError) as exc:
        print(canonical_json({"error": f"{type(exc).__name__}:{exc}"}))
        return 2
    print(canonical_json(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
