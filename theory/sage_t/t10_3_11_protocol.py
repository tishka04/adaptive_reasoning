"""Frozen bounded-goal protocol for SAGE.T10.3.11."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import t10_3_10_protocol as parent

FORMAT_VERSION = "sage-t10.3.11-bounded-goal-protocol-v1"
MANIFEST_STATUS = "FROZEN_BEFORE_T10_3_11_SOURCE_ACTION"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "sage_t10_3_11_protocol_manifest.json"
)
DEFAULT_MIGRATION_PATH = Path(__file__).with_name(
    "sage_t10_3_11_migration_receipt.json"
)
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "t10_3_11_bounded_goal"

CORE_GAMES = parent.CORE_GAMES
SEQUENCE_GAMES = parent.SEQUENCE_GAMES
ALL_SOURCE_GAMES = (*CORE_GAMES, *SEQUENCE_GAMES)
DISCOVERY_SEEDS = (3421, 3422, 3423, 3424)
REPRODUCTION_SEEDS = (3431, 3432)
CONFIRMATION_SEEDS = (3441, 3442)
DISCOVERY_ARMS = ("goal_conditioned_sage_t", "goal_ablation_sage_t")
CONFIRMATION_ARMS = ("goal_conditioned_sage_t", "unified_sage_t_off")
CORE_ACTION_BUDGET = parent.CORE_ACTION_BUDGET
SEQUENCE_ACTION_BUDGET = parent.SEQUENCE_ACTION_BUDGET
CORE_RESET_WALL_SECONDS = parent.CORE_RESET_WALL_SECONDS
SEQUENCE_RESET_WALL_SECONDS = parent.SEQUENCE_RESET_WALL_SECONDS

IntegrityError = parent.IntegrityError
ScientificGateMiss = parent.ScientificGateMiss
sha256_payload = parent.sha256_payload
file_sha256 = parent.file_sha256
verify_signed = parent.verify_signed
write_json_once = parent.write_json_once

PARENT_ARTIFACTS = {
    "t10_3_10_manifest": {
        "path": "theory/sage_t/sage_t10_3_10_protocol_manifest.json",
        "sha256": "a9929ad9a9986aeca1a19c6a51689918aeb1cc6a9f5b503fa636b143189ca66b",
    },
    "t10_3_10_migration": {
        "path": "theory/sage_t/sage_t10_3_10_migration_receipt.json",
        "sha256": "6d3e9832a4ed5a6d670c419badd7efe8f0342bf42704ab71b56ad3d93055e872",
    },
    "t10_3_10_audit": {
        "path": "training/sage_t/t10_3_10_directional_progress/offline_audit.json",
        "sha256": "07eef71f0bafdd5fe4262a25e63a41e69852b917add46efd23480ced705b9d7d",
    },
    "t10_3_10_preflight": {
        "path": "training/sage_t/t10_3_10_directional_progress/synthetic_preflight.json",
        "sha256": "2f44b9f6c592b503ef6c3f3d847df9bed00d6e909cadcc84a21a8b45e655296c",
    },
    "t10_3_10_discovery": {
        "path": "training/sage_t/t10_3_10_directional_progress/discovery_sequence_report.json",
        "sha256": "4e98df09b5a12cfabb6f9f53fbafaddbdca570c9456b1f4adb8c7e9e4023c5a3",
    },
}

SUPERSEDED_T10_3_10 = {
    "status": "SUPERSEDED_COMPLETE_NEGATIVE",
    "manifest_checksum": "628ebc4a7b86e219202a0e951018c348c43fe067c5833bfd7f32d5601baa8695",
    "discovery_report_checksum": "99473ca240aaeca53476f907362f624c5cd6709254fb0ea1a4020325b73587df",
    "intent_count": 857,
    "event_count": 857,
    "branch_count": 12,
    "unresolved_count": 0,
    "incomplete_work_count": 0,
    "sequence_level_count": 0,
    "controller_observe_error_count": 5,
    "posterior_update_count": 852,
    "game_over_count": 3,
    "maximum_sage_identical_action_run": 2,
    "controller_cycle_p95_ms": 28714.00940004969,
    "used_for_training": False,
    "registry_used_as_prior": False,
    "mutated_by_t10_3_11": False,
    "physical_actions_replayed": 0,
}


def _signed(payload: Mapping[str, Any], checksum_field: str) -> dict[str, Any]:
    output = dict(payload)
    output[checksum_field] = sha256_payload(output)
    return output


def _parent_root(root: Path) -> Path:
    return root / "training" / "sage_t" / "t10_3_10_directional_progress"


def _read_signed(path: Path, checksum_field: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_signed(payload, checksum_field)
    return payload


def _journal_files(root: Path, category: str) -> tuple[Path, ...]:
    directory = _parent_root(root) / "journal" / category
    return tuple(sorted(directory.rglob("*.json"))) if directory.exists() else ()


def _parent_journal_digest(root: Path) -> str:
    base = _parent_root(root) / "journal"
    rows = [
        {
            "relative_path": path.relative_to(base).as_posix(),
            "sha256": file_sha256(path),
        }
        for path in sorted(item for item in base.rglob("*.json") if item.is_file())
    ]
    return sha256_payload(rows)


def _parent_diagnosis(root: Path) -> dict[str, Any]:
    parent_manifest = parent.load_manifest(root)
    intents = _journal_files(root, "intents")
    events = _journal_files(root, "events")
    unresolved = _journal_files(root, "unresolved")
    receipts = [
        _read_signed(path, "receipt_checksum")
        for path in _journal_files(root, "branches")
        if path.name == "receipt.json"
    ]
    report = _read_signed(
        _parent_root(root) / "discovery_sequence_report.json",
        "report_checksum",
    )
    started = {path.parent.name for path in intents}
    complete = {str(row["work_id"]) for row in receipts}
    return {
        "manifest_checksum": str(parent_manifest["manifest_checksum"]),
        "discovery_report_checksum": str(report["report_checksum"]),
        "intent_count": len(intents),
        "event_count": len(events),
        "branch_count": len(receipts),
        "unresolved_count": len(unresolved),
        "incomplete_work_count": len(started - complete),
        "sequence_level_count": sum(int(row.get("level_delta", 0)) for row in receipts),
        "controller_observe_error_count": sum(
            str(error).startswith("CONTROLLER_OBSERVE:")
            for row in receipts
            for error in row.get("errors", ())
        ),
        "posterior_update_count": sum(
            int(row.get("lightweight_observations", 0)) for row in receipts
        ),
        "game_over_count": sum(int(row.get("game_over_actions", 0)) for row in receipts),
        "maximum_sage_identical_action_run": int(
            report["metrics"]["maximum_sage_identical_action_run"]
        ),
        "controller_cycle_p95_ms": float(report["metrics"]["controller_cycle_p95_ms"]),
    }


def _verify_parent(root: Path) -> dict[str, Any]:
    parent.load_manifest(root)
    for name, binding in PARENT_ARTIFACTS.items():
        path = root / binding["path"]
        if not path.is_file() or file_sha256(path) != binding["sha256"]:
            raise IntegrityError(f"required T10.3.10 artifact absent or drifted: {name}")
    if (_parent_root(root) / "collector.lock.json").exists():
        raise IntegrityError("T10.3.10 collector lock must be absent")
    diagnosis = _parent_diagnosis(root)
    expected = {key: SUPERSEDED_T10_3_10[key] for key in diagnosis}
    if diagnosis != expected:
        raise IntegrityError("T10.3.10 negative diagnosis diverged")
    return diagnosis


def _code_hashes(root: Path) -> dict[str, str]:
    relative = (
        "theory/sage_t/goal_directed_v10_3_11.py",
        "theory/sage_t/t10_3_11_protocol.py",
        "theory/sage_t/t10_3_11_runtime.py",
        "theory/sage_t/goal_directed_v10_3_10.py",
        "theory/sage_t/t10_3_5_runtime.py",
        "tests/test_sage_t_goal_directed_v10_3_11.py",
        "tests/test_sage_t_t10_3_11_protocol.py",
        "tests/test_sage_t_t10_3_11_runtime.py",
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
    rows: list[WorkSpec] = []
    if phase == "discover-sequence":
        for game_index, game in enumerate(SEQUENCE_GAMES):
            for seed_index, seed in enumerate(DISCOVERY_SEEDS):
                arms = list(DISCOVERY_ARMS)
                if (game_index + seed_index) % 2:
                    arms.reverse()
                for arm in arms:
                    rows.append(
                        WorkSpec(
                            phase,
                            game,
                            seed,
                            arm,
                            seed_index,
                            SEQUENCE_ACTION_BUDGET,
                        )
                    )
        return tuple(rows)
    if phase == "reproduce-sequence":
        for game in SEQUENCE_GAMES:
            for index, seed in enumerate(REPRODUCTION_SEEDS):
                rows.append(
                    WorkSpec(
                        phase,
                        game,
                        seed,
                        "goal_conditioned_sage_t",
                        index,
                        SEQUENCE_ACTION_BUDGET,
                    )
                )
        return tuple(rows)
    if phase == "confirm":
        for game_index, game in enumerate(ALL_SOURCE_GAMES):
            budget = CORE_ACTION_BUDGET if game in CORE_GAMES else SEQUENCE_ACTION_BUDGET
            for seed_index, seed in enumerate(CONFIRMATION_SEEDS):
                arms = list(CONFIRMATION_ARMS)
                if (game_index + seed_index) % 2:
                    arms.reverse()
                for arm in arms:
                    rows.append(WorkSpec(phase, game, seed, arm, seed_index, budget))
        return tuple(rows)
    raise ValueError(f"unsupported physical phase: {phase}")


def maximum_actions_for_phase(phase: str) -> int:
    return sum(work.action_budget for work in work_specs(phase))


def maximum_actions_for_specs(specs: Sequence[WorkSpec]) -> int:
    return sum(int(item.action_budget) for item in specs)


def reset_wall_seconds(work: WorkSpec) -> float:
    return CORE_RESET_WALL_SECONDS if work.game_id in CORE_GAMES else SEQUENCE_RESET_WALL_SECONDS


def _matrix_payload() -> dict[str, Any]:
    phases = {
        phase: {
            "resets": len(work_specs(phase)),
            "maximum_actions": maximum_actions_for_phase(phase),
        }
        for phase in ("discover-sequence", "reproduce-sequence", "confirm")
    }
    return {
        "offline_diagnostic_actions": 0,
        "phases": phases,
        "total_resets": sum(row["resets"] for row in phases.values()),
        "total_maximum_actions": sum(row["maximum_actions"] for row in phases.values()),
    }


TOTAL_RESETS = _matrix_payload()["total_resets"]
TOTAL_MAXIMUM_ACTIONS = _matrix_payload()["total_maximum_actions"]


def build_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    diagnosis = _verify_parent(root)
    core = {
        "format_version": FORMAT_VERSION,
        "status": MANIFEST_STATUS,
        "objective": "test_bounded_goal_conditioned_end_to_end_sequence_control",
        "parent_artifacts": PARENT_ARTIFACTS,
        "superseded_t10_3_10": {
            **SUPERSEDED_T10_3_10,
            "journal_digest": _parent_journal_digest(root),
            "diagnosis": diagnosis,
        },
        "code_hashes": _code_hashes(root),
        "cli_phases": [
            "freeze",
            "status",
            "audit",
            "preflight",
            "discover-sequence",
            "reproduce-sequence",
            "compile",
            "confirm",
            "report",
        ],
        "exit_codes": {"success": 0, "integrity": 2, "scientific_gate": 3},
        "matrix": _matrix_payload(),
        "functional_contract": {
            "level_increment_is_only_success_credit": True,
            "goal_distance_is_planning_evidence_only": True,
            "program_posterior_history_limit": 16,
            "controller_transition_limit": 16,
            "live_posterior_repair": False,
            "reassemble_only_on_action_space_change": True,
            "goal_hypotheses_forwarded_to_sage_t": True,
            "paired_goal_ablation": True,
            "parent_events_diagnostic_only": True,
            "parent_registry_loaded": False,
            "t10_3_8_core_registry_prior_support_zero": True,
            "latency_is_telemetry_only": True,
        },
        "gates": {
            "discovery_sequence_level_total_minimum": 1,
            "discovery_mixed_winner_required": True,
            "goal_conditioned_not_below_ablation": True,
            "goal_conditioned_total_advantage_minimum": 1,
            "reproduction_same_game_required": True,
            "confirmation_core_level_each_game": True,
            "confirmation_sequence_level_total_minimum": 1,
            "confirmation_total_level_advantage_minimum": 1,
            "zero_controller_errors": True,
            "zero_illegal_actions": True,
            "posterior_each_event": True,
        },
        "firewall": {
            "source_validation_opened": False,
            "ar25_opened": False,
            "holdout_opened": False,
            "production_authority": False,
            "automatic_retuning": False,
            "t10_3_10_events_training_authorized": False,
            "t10_3_10_registry_prior_authorized": False,
            "t10_3_8_core_registry_structural_prior_authorized": True,
        },
        "durability": {
            "intent_before_action": True,
            "event_immediate_seal": True,
            "physical_replay": False,
            "write_once": True,
        },
        "output_directory": DEFAULT_OUTPUT_DIR.as_posix(),
    }
    return _signed(core, "manifest_checksum")


def freeze_manifest(
    root: Path,
    *,
    manifest_path: Path | None = None,
    migration_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    manifest = build_manifest(root)
    write_json_once(manifest_path or root / DEFAULT_MANIFEST_PATH, manifest)
    migration = _signed(
        {
            "format_version": "sage-t10.3.11-bounded-goal-migration-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "superseded_protocol": "SAGE.T10.3.10",
            "superseded_status": SUPERSEDED_T10_3_10["status"],
            "parent_journal_digest": manifest["superseded_t10_3_10"]["journal_digest"],
            "diagnosis": manifest["superseded_t10_3_10"]["diagnosis"],
            "parent_events_used_for_training": 0,
            "parent_registry_loaded": False,
            "parent_core_registry_loaded_as_structural_prior": True,
            "parent_core_registry_local_support": 0,
            "parent_artifacts_mutated": False,
            "parent_physical_actions_replayed": 0,
            "repairs": [
                "strictly_bounded_program_posterior_history",
                "incremental_no_repair_physical_observation",
                "action_space_change_only_program_reassembly",
                "bounded_live_goal_generation",
                "goal_conditioned_option_frontier",
                "paired_goal_ablation",
                "safe_observation_error_fingerprint",
            ],
        },
        "receipt_checksum",
    )
    write_json_once(migration_path or root / DEFAULT_MIGRATION_PATH, migration)
    return manifest, migration


def load_manifest(root: Path, *, verify_code: bool = True) -> dict[str, Any]:
    root = root.resolve()
    path = root / DEFAULT_MANIFEST_PATH
    if not path.is_file():
        raise IntegrityError("T10.3.11 manifest has not been frozen")
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_signed(payload, "manifest_checksum")
    if payload.get("format_version") != FORMAT_VERSION:
        raise IntegrityError("T10.3.11 manifest format drifted")
    _verify_parent(root)
    if payload.get("superseded_t10_3_10", {}).get("journal_digest") != _parent_journal_digest(root):
        raise IntegrityError("T10.3.10 journal changed after T10.3.11 freeze")
    if verify_code and payload.get("code_hashes") != _code_hashes(root):
        raise IntegrityError("T10.3.11 code hash drifted after freeze")
    migration_path = root / DEFAULT_MIGRATION_PATH
    if not migration_path.is_file():
        raise IntegrityError("T10.3.11 migration receipt is absent")
    migration = json.loads(migration_path.read_text(encoding="utf-8"))
    verify_signed(migration, "receipt_checksum")
    if migration.get("manifest_checksum") != payload.get("manifest_checksum"):
        raise IntegrityError("T10.3.11 migration receipt is detached")
    return payload


__all__ = [
    "ALL_SOURCE_GAMES",
    "CONFIRMATION_ARMS",
    "CONFIRMATION_SEEDS",
    "CORE_ACTION_BUDGET",
    "CORE_GAMES",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_MIGRATION_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DISCOVERY_ARMS",
    "DISCOVERY_SEEDS",
    "IntegrityError",
    "PARENT_ARTIFACTS",
    "REPRODUCTION_SEEDS",
    "SEQUENCE_ACTION_BUDGET",
    "SEQUENCE_GAMES",
    "SUPERSEDED_T10_3_10",
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
    "reset_wall_seconds",
    "sha256_payload",
    "verify_signed",
    "work_specs",
    "write_json_once",
]
