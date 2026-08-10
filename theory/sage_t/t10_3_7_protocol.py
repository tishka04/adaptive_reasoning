"""Frozen stable-successor continuation protocol for SAGE.T10.3.7."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import t10_3_6_protocol as parent

FORMAT_VERSION = "sage-t10.3.7-stable-successor-recovery-protocol-v1"
MANIFEST_STATUS = "FROZEN_BEFORE_T10_3_7_SOURCE_ACTION"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name("sage_t10_3_7_protocol_manifest.json")
DEFAULT_MIGRATION_PATH = Path(__file__).with_name("sage_t10_3_7_migration_receipt.json")
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "t10_3_7_stable_successor_recovery"

CORE_GAMES = parent.CORE_GAMES
SEQUENCE_GAMES = parent.SEQUENCE_GAMES
ALL_SOURCE_GAMES = parent.ALL_SOURCE_GAMES
WITNESS_PROGRAMS = parent.WITNESS_PROGRAMS
WITNESS_SEEDS = (3271, 3272)
DISCOVERY_SEEDS = (3281, 3282, 3283, 3284)
REPRODUCTION_SEEDS = (3291, 3292)
SEQUENCE_SEEDS = (3301, 3302)
CONFIRMATION_SEEDS = (3311, 3312)
CONFIRMATION_ARMS = parent.CONFIRMATION_ARMS
WITNESS_ACTION_BUDGET = parent.WITNESS_ACTION_BUDGET
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
    "t10_3_6_manifest": {
        "path": "theory/sage_t/sage_t10_3_6_protocol_manifest.json",
        "sha256": "b7159dcd78c4b8ededb47fc7ce21f0db79f0f8ea3cd3f9e3ac73598360ef95de",
    },
    "t10_3_6_migration": {
        "path": "theory/sage_t/sage_t10_3_6_migration_receipt.json",
        "sha256": "cb0ea2f3fe1a59c43cefcba7d47b3a6a165739be327de4dd72db1f4f716d696b",
    },
    "t10_3_6_audit": {
        "path": "training/sage_t/t10_3_6_functional_end_to_end/offline_audit.json",
        "sha256": "ca3b9500b7f338b6b2beb1463306ef69f4247d8183347b35782339404ea75189",
    },
    "t10_3_6_preflight": {
        "path": "training/sage_t/t10_3_6_functional_end_to_end/synthetic_preflight.json",
        "sha256": "1a1fa1d34d87bb9e044d7230e5c7852861f5263e71bf74bd39b283f9908d1ae7",
    },
    "t10_3_6_checkpoint": {
        "path": "training/sage_t/t10_3_6_functional_end_to_end/checkpoint.json",
        "sha256": "adfc4a23f0d530788a7a19f24d4e2cbd7df15102a51b8792175998a18f7b9673",
    },
    "t10_3_6_witness_report": {
        "path": "training/sage_t/t10_3_6_functional_end_to_end/canonical_witness_report.json",
        "sha256": "a05fd929bc7523b1f5760708c0df94ae189daecc8a1c624cbbe0dafb6138c6dd",
    },
}

SUPERSEDED_T10_3_6 = {
    "status": "SUPERSEDED_COMPLETE_NEGATIVE_TENTH_SUCCESSOR_MISS",
    "intent_count": 50,
    "event_count": 50,
    "unresolved_count": 0,
    "branch_count": 4,
    "lp85_level_delta": 1,
    "su15_level_delta": 0,
    "game_over_count": 1,
    "verdict": "CANONICAL_WITNESS_MISS",
    "first_nine_su15_waypoints_exact": True,
    "tenth_waypoint_repeated_ninth": True,
    "used_for_training": False,
    "mutated_by_t10_3_7": False,
    "physical_actions_replayed": 0,
}


def _signed(payload: Mapping[str, Any], checksum_field: str) -> dict[str, Any]:
    result = dict(payload)
    result[checksum_field] = sha256_payload(result)
    return result


def _parent_root(root: Path) -> Path:
    return root / "training" / "sage_t" / "t10_3_6_functional_end_to_end"


def _journal_files(root: Path, category: str) -> tuple[Path, ...]:
    directory = _parent_root(root) / "journal" / category
    return tuple(sorted(directory.rglob("*.json"))) if directory.exists() else ()


def _read_signed(path: Path, checksum_field: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_signed(payload, checksum_field)
    return payload


def _parent_journal_digest(root: Path) -> str:
    base = _parent_root(root) / "journal"
    rows = [
        {"relative_path": path.relative_to(base).as_posix(), "sha256": file_sha256(path)}
        for path in sorted(item for item in base.rglob("*.json") if item.is_file())
    ]
    return sha256_payload(rows)


def _parent_diagnosis(root: Path) -> dict[str, Any]:
    events = tuple(_read_signed(path, "event_checksum") for path in _journal_files(root, "events"))
    receipts = tuple(_read_signed(path, "receipt_checksum") for path in _journal_files(root, "branches"))
    report = _read_signed(_parent_root(root) / "canonical_witness_report.json", "report_checksum")
    levels = report.get("metrics", {}).get("levels", {})
    return {
        "intent_count": len(_journal_files(root, "intents")),
        "event_count": len(events),
        "unresolved_count": len(_journal_files(root, "unresolved")),
        "branch_count": len(receipts),
        "lp85_level_delta": int(levels.get("lp85-305b61c3", 0)),
        "su15_level_delta": int(levels.get("su15-4c352900", 0)),
        "game_over_count": sum(
            str(row.get("game_state_after", "")).upper() == "GAME_OVER" for row in events
        ),
        "verdict": str(report.get("verdict", "")),
    }


def _verify_parent(root: Path) -> dict[str, Any]:
    parent.load_manifest(root)
    for name, binding in PARENT_ARTIFACTS.items():
        path = root / binding["path"]
        if not path.is_file() or file_sha256(path) != binding["sha256"]:
            raise IntegrityError(f"required T10.3.6 artifact absent or drifted: {name}")
    if (_parent_root(root) / "collector.lock.json").exists():
        raise IntegrityError("T10.3.6 collector lock must be absent")
    diagnosis = _parent_diagnosis(root)
    expected = {key: SUPERSEDED_T10_3_6[key] for key in diagnosis}
    if diagnosis != expected:
        raise IntegrityError("T10.3.6 witness snapshot diverged")
    return diagnosis


def _code_hashes(root: Path) -> dict[str, str]:
    relative = (
        "theory/sage_t/goal_directed_v10_3_7.py",
        "theory/sage_t/t10_3_7_protocol.py",
        "theory/sage_t/t10_3_7_runtime.py",
        "theory/sage_t/goal_directed_v10_3_6.py",
        "theory/sage_t/t10_3_6_runtime.py",
        "theory/sage_t/progress_witness_v10.py",
        "tests/test_sage_t_goal_directed_v10_3_7.py",
        "tests/test_sage_t_t10_3_7_protocol.py",
        "tests/test_sage_t_t10_3_7_runtime.py",
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
    if phase == "witness-core":
        for game in CORE_GAMES:
            for index, seed in enumerate(WITNESS_SEEDS):
                rows.append(WorkSpec(phase, game, seed, "canonical_witness_diagnostic", index, WITNESS_ACTION_BUDGET))
        return tuple(rows)
    if phase == "discover-core":
        for game in CORE_GAMES:
            for index, seed in enumerate(DISCOVERY_SEEDS):
                rows.append(WorkSpec(phase, game, seed, "goal_directed_sage_t", index, CORE_ACTION_BUDGET))
        return tuple(rows)
    if phase == "reproduce-core":
        for game in CORE_GAMES:
            for index, seed in enumerate(REPRODUCTION_SEEDS):
                rows.append(WorkSpec(phase, game, seed, "goal_directed_sage_t", index, CORE_ACTION_BUDGET))
        return tuple(rows)
    if phase == "discover-sequence":
        for game in SEQUENCE_GAMES:
            for index, seed in enumerate(SEQUENCE_SEEDS):
                rows.append(WorkSpec(phase, game, seed, "goal_directed_sage_t", index, SEQUENCE_ACTION_BUDGET))
        return tuple(rows)
    if phase == "confirm":
        for game_index, game in enumerate(ALL_SOURCE_GAMES):
            budget = CORE_ACTION_BUDGET if game in CORE_GAMES else SEQUENCE_ACTION_BUDGET
            for seed_index, seed in enumerate(CONFIRMATION_SEEDS):
                arms = list(CONFIRMATION_ARMS)
                if (game_index + seed_index) % 2:
                    arms.reverse()
                for arm_index, arm in enumerate(arms):
                    rows.append(WorkSpec(phase, game, seed, arm, arm_index, budget))
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
        for phase in ("witness-core", "discover-core", "reproduce-core", "discover-sequence", "confirm")
    }
    return {
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
        "objective": "recover_exact_fresh_successor_execution_then_resume_functional_end_to_end_test",
        "parent_artifacts": PARENT_ARTIFACTS,
        "superseded_t10_3_6": {
            **SUPERSEDED_T10_3_6,
            "journal_digest": _parent_journal_digest(root),
            "diagnosis": diagnosis,
        },
        "canonical_witness_descriptors": WITNESS_PROGRAMS,
        "code_hashes": _code_hashes(root),
        "cli_phases": [
            "freeze", "status", "audit", "preflight", "witness-core",
            "discover-core", "reproduce-core", "discover-sequence", "compile",
            "confirm", "report",
        ],
        "exit_codes": {"success": 0, "integrity": 2, "scientific_gate": 3},
        "matrix": _matrix_payload(),
        "functional_contract": {
            "fresh_path_induced_per_reset": True,
            "fresh_path_held_ephemerally_during_option": True,
            "each_waypoint_reacquired_from_current_legal_actions": True,
            "mid_option_path_shortening_forbidden": True,
            "historical_grounded_actions_loaded": False,
            "blank_posterior_discovery": True,
            "level_increment_is_only_success_credit": True,
            "latency_is_telemetry_only": True,
            "no_latency_scientific_gate": True,
        },
        "gates": {
            "witness_level_each_core_game": True,
            "exact_ten_waypoint_su15_execution": True,
            "blank_discovery_level_each_core_game": True,
            "fresh_reproduction_level_each_core_game": True,
            "sequence_level_minimum_games": 1,
            "sequence_minimum_action_schemas": 2,
            "confirmation_total_level_advantage": 1,
        },
        "firewall": {
            "source_validation_opened": False,
            "ar25_opened": False,
            "holdout_opened": False,
            "production_authority": False,
            "automatic_retuning": False,
            "t10_3_6_events_training_authorized": False,
            "t10_0b_events_training_authorized": False,
            "t10_0b_program_prior_for_discovery_authorized": False,
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
            "format_version": "sage-t10.3.7-stable-successor-migration-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "superseded_protocol": "SAGE.T10.3.6",
            "superseded_status": SUPERSEDED_T10_3_6["status"],
            "parent_journal_digest": manifest["superseded_t10_3_6"]["journal_digest"],
            "diagnosis": manifest["superseded_t10_3_6"]["diagnosis"],
            "parent_events_used_for_training": 0,
            "parent_artifacts_mutated": False,
            "parent_physical_actions_replayed": 0,
            "correction": "retain_fresh_initial_successor_plan_ephemerally_and_reacquire_each_waypoint",
        },
        "receipt_checksum",
    )
    write_json_once(migration_path or root / DEFAULT_MIGRATION_PATH, migration)
    return manifest, migration


def load_manifest(root: Path, *, verify_code: bool = True) -> dict[str, Any]:
    root = root.resolve()
    path = root / DEFAULT_MANIFEST_PATH
    if not path.is_file():
        raise IntegrityError("T10.3.7 manifest has not been frozen")
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_signed(payload, "manifest_checksum")
    if payload.get("format_version") != FORMAT_VERSION:
        raise IntegrityError("T10.3.7 manifest format drifted")
    _verify_parent(root)
    if payload.get("superseded_t10_3_6", {}).get("journal_digest") != _parent_journal_digest(root):
        raise IntegrityError("T10.3.6 journal changed after supersession")
    if verify_code and payload.get("code_hashes") != _code_hashes(root):
        raise IntegrityError("T10.3.7 code hash drifted after freeze")
    migration_path = root / DEFAULT_MIGRATION_PATH
    if not migration_path.is_file():
        raise IntegrityError("T10.3.7 migration receipt is absent")
    migration = json.loads(migration_path.read_text(encoding="utf-8"))
    verify_signed(migration, "receipt_checksum")
    if migration.get("manifest_checksum") != payload.get("manifest_checksum"):
        raise IntegrityError("T10.3.7 migration receipt is detached")
    return payload


__all__ = [
    "ALL_SOURCE_GAMES", "CONFIRMATION_ARMS", "CONFIRMATION_SEEDS",
    "CORE_GAMES", "DEFAULT_MANIFEST_PATH", "DEFAULT_MIGRATION_PATH",
    "DEFAULT_OUTPUT_DIR", "DISCOVERY_SEEDS", "IntegrityError",
    "REPRODUCTION_SEEDS", "SEQUENCE_GAMES", "SEQUENCE_SEEDS",
    "ScientificGateMiss", "SUPERSEDED_T10_3_6", "TOTAL_MAXIMUM_ACTIONS",
    "TOTAL_RESETS", "WITNESS_PROGRAMS", "WITNESS_SEEDS", "WorkSpec",
    "build_manifest", "file_sha256", "freeze_manifest", "load_manifest",
    "maximum_actions_for_phase", "maximum_actions_for_specs",
    "reset_wall_seconds", "sha256_payload", "verify_signed", "work_specs",
    "write_json_once",
]
