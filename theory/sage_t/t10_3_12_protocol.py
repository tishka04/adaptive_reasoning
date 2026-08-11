"""Frozen relational-mechanism protocol for SAGE.T10.3.12."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import t10_3_11_protocol as parent
from .relational_program_v10_3_12 import ARMS, sha256_payload

FORMAT_VERSION = "sage-t10.3.12-relational-mechanism-protocol-v1"
MANIFEST_STATUS = "FROZEN_BEFORE_T10_3_12_OFFLINE_EVALUATION"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name("sage_t10_3_12_protocol_manifest.json")
DEFAULT_FREEZE_RECEIPT_PATH = Path(__file__).with_name(
    "sage_t10_3_12_freeze_receipt.json"
)
DEFAULT_OUTPUT_DIR = (
    Path("training") / "sage_t" / "t10_3_12_relational_mechanism_invariance"
)

CORE_GAMES = parent.CORE_GAMES
if len(CORE_GAMES) != 2:
    raise RuntimeError("T10.3.12 requires exactly two frozen core games")
ACTIVE_LABELS = (3521, 3522, 3523, 3524)
ACTIVE_ACTION_BUDGET = 16
ACTIVE_RESET_WALL_SECONDS = 300.0
TOTAL_RESETS = len(CORE_GAMES) * len(ACTIVE_LABELS) * len(ARMS)
TOTAL_MAXIMUM_ACTIONS = TOTAL_RESETS * ACTIVE_ACTION_BUDGET

IntegrityError = parent.IntegrityError
ScientificGateMiss = parent.ScientificGateMiss
file_sha256 = parent.file_sha256
verify_signed = parent.verify_signed
write_json_once = parent.write_json_once

PARENT_ARTIFACTS = {
    "t10_3_11_manifest": {
        "path": "theory/sage_t/sage_t10_3_11_protocol_manifest.json",
        "sha256": "4a9c930a65e07a3175fd2fdee746d25a493f4921356d50adaed275532aa6a684",
    },
    "t10_3_11_migration": {
        "path": "theory/sage_t/sage_t10_3_11_migration_receipt.json",
        "sha256": "b771c9b46d9431801c0663d0c471466afee53e93bd439c2093517b7030ce701d",
    },
    "t10_3_11_terminal": {
        "path": "training/sage_t/t10_3_11_bounded_goal/terminal_report.json",
        "sha256": "75846274783cb52a6bc1a09e1bc17f3ebf3c516d7de4b2d8554a4250cb71b72d",
    },
    "t10_3_11_discovery": {
        "path": "training/sage_t/t10_3_11_bounded_goal/discovery_sequence_report.json",
        "sha256": "15baeb3f116bb493813dff8ad3c3c7041ffa2edcafe2a934bb1518a41bb504cd",
    },
}

SOURCE_ALLOWLIST = {
    "canonical_witness_projection": {
        "path": "training/sage_t/t10_3_8_witness_gate_adjudication/canonical_witness_report.json",
        "sha256": "1905703c454b113796a082553fb45d68eebdb09d43d366c9fa1339601c9c4c2c",
    },
    "progress_witness_projection": {
        "path": "training/sage_t/progress_witness_v10_0b/report.json",
        "sha256": "998b8af56b9933383bfe9b37aa4e245cffad3fb635d0abc8a19c5817724c044f",
    },
    "repeat_source_shard": {
        "path": "training/sage12/bound_mechanic_pilot_v4_3/source_train_shards/lp85.jsonl",
        "sha256": "7dee5fa89bace32af5f744c02489a22d4703b96e4dea3fef6d6971c9e1f7461c",
    },
    "path_source_shard": {
        "path": "training/sage12/bound_mechanic_pilot_v4_3/source_train_shards/su15.jsonl",
        "sha256": "d089568302893ba8e409694919d9c744400d53565d3e6f90ddbd5f1012753224",
    },
}

QUARANTINED_PARENT = {
    "status": "SUPERSEDED_INCOMPLETE_NEGATIVE",
    "authorized_actions": 1758,
    "sealed_events": 1756,
    "unresolved_intents": 0,
    "inflight_intents": 2,
    "branch_count": 24,
    "level_count": 0,
    "runtime_permission_errors": 3,
    "live_collector_lock": False,
    "inflight_valid": False,
    "used_for_training": False,
    "registry_loaded": False,
    "distance_metrics_loaded": False,
    "physical_actions_replayed": 0,
    "mutated_by_t10_3_12": False,
}

ARTIFACT_CONTRACT = {
    "audit-parent": {
        "path": "parent_quarantine_receipt.json",
        "checksum_field": "receipt_checksum",
        "gate_field": "passed",
    },
    "preflight": {
        "path": "synthetic_preflight.json",
        "checksum_field": "preflight_checksum",
        "gate_field": "passed",
    },
    "materialize-offline": {
        "path": "offline_fixture_inventory.json",
        "checksum_field": "inventory_checksum",
        "gate_field": "passed",
    },
    "compile-candidates": {
        "path": "candidate_registry.json",
        "checksum_field": "registry_checksum",
        "gate_field": None,
    },
    "evaluate-offline": {
        "path": "offline_equivariance_report.json",
        "checksum_field": "report_checksum",
        "gate_field": "passed",
    },
    "active-core": {
        "path": "active_core_report.json",
        "checksum_field": "report_checksum",
        "gate_field": "collection_complete",
    },
    "adjudicate": {
        "path": "adjudication_report.json",
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
    output = dict(payload)
    output[checksum_field] = sha256_payload(output)
    return output


def _parent_root(root: Path) -> Path:
    return root / "training" / "sage_t" / "t10_3_11_bounded_goal"


def _journal_files(root: Path, category: str) -> tuple[Path, ...]:
    directory = _parent_root(root) / "journal" / category
    return tuple(sorted(directory.rglob("*.json"))) if directory.exists() else ()


def _read_signed(path: Path, checksum_field: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_signed(payload, checksum_field)
    return payload


def parent_journal_digest(root: Path) -> str:
    base = _parent_root(root) / "journal"
    rows = [
        {
            "relative_path": path.relative_to(base).as_posix(),
            "sha256": file_sha256(path),
        }
        for path in sorted(item for item in base.rglob("*.json") if item.is_file())
    ]
    return sha256_payload(rows)


def parent_diagnosis(root: Path) -> dict[str, Any]:
    intents = _journal_files(root, "intents")
    events = _journal_files(root, "events")
    unresolved = _journal_files(root, "unresolved")
    receipts = [
        _read_signed(path, "receipt_checksum")
        for path in _journal_files(root, "branches")
        if path.name == "receipt.json"
    ]
    intent_ids = {
        path.relative_to(_parent_root(root) / "journal" / "intents").as_posix()
        for path in intents
    }
    event_ids = {
        path.relative_to(_parent_root(root) / "journal" / "events").as_posix()
        for path in events
    }
    unresolved_ids = {
        path.relative_to(_parent_root(root) / "journal" / "unresolved").as_posix()
        for path in unresolved
    }
    inflight = sorted(intent_ids - event_ids - unresolved_ids)
    lock = _parent_root(root) / "collector.lock.json"
    return {
        "authorized_actions": len(intents),
        "sealed_events": len(events),
        "unresolved_intents": len(unresolved),
        "inflight_intents": len(inflight),
        "inflight_paths": inflight,
        "branch_count": len(receipts),
        "level_count": sum(int(row.get("level_delta", 0)) for row in receipts),
        "runtime_permission_errors": sum(
            str(error) == "RUNTIME:PermissionError"
            for row in receipts
            for error in row.get("errors", ())
        ),
        "live_collector_lock": lock.exists(),
        "inflight_valid": len(inflight) <= 1 and lock.exists(),
    }


def _verify_parent(root: Path) -> dict[str, Any]:
    parent.load_manifest(root)
    for name, binding in PARENT_ARTIFACTS.items():
        path = root / binding["path"]
        if not path.is_file() or file_sha256(path) != binding["sha256"]:
            raise IntegrityError(f"required T10.3.11 artifact absent or drifted: {name}")
    diagnosis = parent_diagnosis(root)
    expected = {
        key: QUARANTINED_PARENT[key]
        for key in diagnosis
        if key != "inflight_paths"
    }
    observed = {key: value for key, value in diagnosis.items() if key != "inflight_paths"}
    if observed != expected:
        raise IntegrityError("T10.3.11 quarantine diagnosis diverged")
    return diagnosis


def verify_source_allowlist(root: Path) -> None:
    for name, binding in SOURCE_ALLOWLIST.items():
        path = root / binding["path"]
        if not path.is_file() or file_sha256(path) != binding["sha256"]:
            raise IntegrityError(f"allowlisted T10.3.12 source absent or drifted: {name}")


def _code_hashes(root: Path) -> dict[str, str]:
    relative = (
        "theory/sage_t/relational_program_v10_3_12.py",
        "theory/sage_t/goal_directed_v10_3_12.py",
        "theory/sage_t/t10_3_12_protocol.py",
        "theory/sage_t/t10_3_12_runtime.py",
        "tests/test_sage_t_relational_program_v10_3_12.py",
        "tests/test_sage_t_goal_directed_v10_3_12.py",
        "tests/test_sage_t_t10_3_12_protocol.py",
        "tests/test_sage_t_t10_3_12_runtime.py",
        "reports/SAGE_T10_3_12_RELATIONAL_MECHANISM_PROTOCOL.md",
        "reports/SAGE_T10_3_12_RELATIONAL_MECHANISM_RUNBOOK.md",
    )
    output = {}
    for item in relative:
        path = root / item
        if not path.is_file():
            raise IntegrityError(f"protocol dependency is absent: {item}")
        output[item] = file_sha256(path)
    return output


@dataclass(frozen=True)
class WorkSpec:
    phase: str
    game_id: str
    seed: int
    arm: str
    reset_index: int
    action_budget: int

    @property
    def work_id(self) -> str:
        return sha256_payload(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "game_id": self.game_id,
            "seed": self.seed,
            "arm": self.arm,
            "reset_index": self.reset_index,
            "action_budget": self.action_budget,
        }


def work_specs(phase: str) -> tuple[WorkSpec, ...]:
    if phase != "active-core":
        raise ValueError(f"unsupported T10.3.12 physical phase: {phase}")
    rows: list[WorkSpec] = []
    for game_index, game in enumerate(CORE_GAMES):
        for label_index, label in enumerate(ACTIVE_LABELS):
            rotation = (game_index + label_index) % len(ARMS)
            ordered_arms = ARMS[rotation:] + ARMS[:rotation]
            for arm in ordered_arms:
                rows.append(
                    WorkSpec(
                        phase=phase,
                        game_id=game,
                        seed=label,
                        arm=arm,
                        reset_index=label_index,
                        action_budget=ACTIVE_ACTION_BUDGET,
                    )
                )
    return tuple(rows)


def maximum_actions_for_phase(phase: str) -> int:
    return sum(row.action_budget for row in work_specs(phase))


def maximum_actions_for_specs(specs: Sequence[WorkSpec]) -> int:
    return sum(int(row.action_budget) for row in specs)


def reset_wall_seconds(work: WorkSpec) -> float:
    if work.phase != "active-core":
        raise ValueError("T10.3.12 reset wall budget is active-core only")
    return ACTIVE_RESET_WALL_SECONDS


def build_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    diagnosis = _verify_parent(root)
    verify_source_allowlist(root)
    core = {
        "format_version": FORMAT_VERSION,
        "status": MANIFEST_STATUS,
        "objective": "identify_source_specific_equivariant_relational_mechanisms",
        "hypothesis": (
            "a factorized operator-role-relation-stop program transfers more "
            "specifically or efficiently than source-free rediscovery"
        ),
        "code_hashes": _code_hashes(root),
        "parent_artifacts": PARENT_ARTIFACTS,
        "source_allowlist": SOURCE_ALLOWLIST,
        "quarantined_parent": {
            **QUARANTINED_PARENT,
            "diagnosis": diagnosis,
            "journal_digest": parent_journal_digest(root),
        },
        "cli_phases": [
            "freeze",
            "status",
            "audit-parent",
            "preflight",
            "materialize-offline",
            "compile-candidates",
            "evaluate-offline",
            "active-core",
            "adjudicate",
            "report",
        ],
        "exit_codes": {"success": 0, "integrity": 2, "scientific_gate": 3},
        "artifact_contract": ARTIFACT_CONTRACT,
        "offline_matrix": {
            "fixtures": 96,
            "positive_fixtures": 64,
            "control_fixtures": 32,
            "arms": list(ARMS),
            "maximum_candidate_inspections": 12_288,
            "maximum_wall_seconds": 600,
            "maximum_memory_mib": 512,
            "maximum_artifact_bytes": 10 * 1024 * 1024,
        },
        "active_matrix": {
            "games": list(CORE_GAMES),
            "replicate_labels": list(ACTIVE_LABELS),
            "labels_seed_environment": False,
            "arms": list(ARMS),
            "latin_square": True,
            "resets": TOTAL_RESETS,
            "maximum_actions_per_reset": ACTIVE_ACTION_BUDGET,
            "maximum_actions": TOTAL_MAXIMUM_ACTIONS,
            "maximum_legal_inspections_per_decision": 32,
            "posterior_history_limit": 16,
            "global_wall_seconds": 10_800,
        },
        "gates": {
            "preflight_cases": 12,
            "factorized_equivariance": "64/64",
            "factorized_controls": "32/32",
            "prefix_invariance": "6/6",
            "source_value_correctness_advantage": 8,
            "source_value_inspection_ratio_maximum": 0.5,
            "specificity_correctness_advantage": 8,
            "causality_correctness_advantage": 8,
            "factorized_active_success": "8/8",
            "repeat_action_median_maximum": 6,
            "repeat_action_maximum": 8,
            "path_action_median_maximum": 10,
            "path_action_maximum": 16,
            "active_success_advantage": 2,
            "equal_success_action_ratio_maximum": 0.75,
        },
        "firewall": {
            "sequence_games_opened": False,
            "source_validation_opened": False,
            "ar25_opened": False,
            "holdout_opened": False,
            "production_authority": False,
            "automatic_retuning": False,
            "t10_3_11_events_training_authorized": False,
            "t10_3_11_registry_authorized": False,
            "t10_3_8_grounded_paths_authorized": False,
            "source_shards_offline_projection_only": True,
        },
        "durability": {
            "intent_before_action": True,
            "event_immediate_seal": True,
            "physical_replay": False,
            "fresh_registry_per_arm_reset": True,
            "write_once": True,
        },
        "negative_result_policy": {
            "offline_miss_forbids_active": True,
            "generic_tie_verdict": "GENERIC_REDISCOVERY_ONLY",
            "partial_support_is_negative": True,
            "promotion_on_pass_only": True,
            "pass_opens_only_t10_3_13_preregistration": True,
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
            "format_version": "sage-t10.3.12-freeze-receipt-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "parent_status": QUARANTINED_PARENT["status"],
            "parent_journal_digest": manifest["quarantined_parent"]["journal_digest"],
            "physical_actions": 0,
            "parent_artifacts_mutated": False,
            "holdout_opened": False,
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
        raise IntegrityError("T10.3.12 manifest has not been frozen")
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_signed(payload, "manifest_checksum")
    if payload.get("format_version") != FORMAT_VERSION:
        raise IntegrityError("T10.3.12 manifest format drifted")
    _verify_parent(root)
    verify_source_allowlist(root)
    if payload.get("quarantined_parent", {}).get("journal_digest") != parent_journal_digest(root):
        raise IntegrityError("T10.3.11 journal changed after T10.3.12 freeze")
    if verify_code and payload.get("code_hashes") != _code_hashes(root):
        raise IntegrityError("T10.3.12 code hash drifted after freeze")
    receipt_path = root / DEFAULT_FREEZE_RECEIPT_PATH
    if not receipt_path.is_file():
        raise IntegrityError("T10.3.12 freeze receipt is absent")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    verify_signed(receipt, "receipt_checksum")
    if receipt.get("manifest_checksum") != payload.get("manifest_checksum"):
        raise IntegrityError("T10.3.12 freeze receipt is detached")
    return payload


__all__ = [
    "ACTIVE_ACTION_BUDGET",
    "ACTIVE_LABELS",
    "ARTIFACT_CONTRACT",
    "CORE_GAMES",
    "DEFAULT_FREEZE_RECEIPT_PATH",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_DIR",
    "FORMAT_VERSION",
    "IntegrityError",
    "MANIFEST_STATUS",
    "PARENT_ARTIFACTS",
    "QUARANTINED_PARENT",
    "SOURCE_ALLOWLIST",
    "ScientificGateMiss",
    "TOTAL_MAXIMUM_ACTIONS",
    "TOTAL_RESETS",
    "WorkSpec",
    "build_manifest",
    "file_sha256",
    "freeze_manifest",
    "load_manifest",
    "maximum_actions_for_phase",
    "maximum_actions_for_specs",
    "parent_diagnosis",
    "parent_journal_digest",
    "reset_wall_seconds",
    "verify_signed",
    "verify_source_allowlist",
    "work_specs",
    "write_json_once",
]
