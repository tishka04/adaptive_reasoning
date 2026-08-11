"""Frozen directional-progress protocol for SAGE.T10.3.10."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import t10_3_9_protocol as parent

FORMAT_VERSION = "sage-t10.3.10-directional-progress-protocol-v1"
MANIFEST_STATUS = "FROZEN_BEFORE_T10_3_10_SOURCE_ACTION"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "sage_t10_3_10_protocol_manifest.json"
)
DEFAULT_MIGRATION_PATH = Path(__file__).with_name(
    "sage_t10_3_10_migration_receipt.json"
)
DEFAULT_OUTPUT_DIR = (
    Path("training") / "sage_t" / "t10_3_10_directional_progress"
)

CORE_GAMES = parent.CORE_GAMES
SEQUENCE_GAMES = parent.SEQUENCE_GAMES
ALL_SOURCE_GAMES = (*CORE_GAMES, *SEQUENCE_GAMES)
DISCOVERY_SEEDS = (3391, 3392, 3393, 3394)
REPRODUCTION_SEEDS = (3401, 3402)
CONFIRMATION_SEEDS = (3411, 3412)
CONFIRMATION_ARMS = parent.CONFIRMATION_ARMS
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
    "t10_3_9_manifest": {
        "path": "theory/sage_t/sage_t10_3_9_protocol_manifest.json",
        "sha256": "23395435296edad851a9d1c2b31183aa8e59aa98e763376efff468be8320005a",
    },
    "t10_3_9_migration": {
        "path": "theory/sage_t/sage_t10_3_9_migration_receipt.json",
        "sha256": "ada9b900618a7746fc977dc003a36fac625ec359e9fdf5a95a81610f096278a0",
    },
    "t10_3_9_audit": {
        "path": (
            "training/sage_t/t10_3_9_causal_subgoal_composition/"
            "offline_audit.json"
        ),
        "sha256": "9128a2539348230d461902364b54a5c56a6bb992841151c8a18b4f2d121aa845",
    },
    "t10_3_9_preflight": {
        "path": (
            "training/sage_t/t10_3_9_causal_subgoal_composition/"
            "synthetic_preflight.json"
        ),
        "sha256": "3dc436e3bff4fd800be2955fed706060e56baf3685f7558d46d121acf4b4d0ca",
    },
}

SUPERSEDED_T10_3_9 = {
    "status": "SUPERSEDED_PARTIAL_EFFECT_CYCLE",
    "manifest_checksum": (
        "a6473f56f818b1968394279b8dc4d033fcd54d40a275c19722024d321d9ffff4"
    ),
    "intent_count": 153,
    "event_count": 153,
    "branch_count": 1,
    "unresolved_count": 0,
    "incomplete_work_count": 1,
    "completed_sequence_action_count": 96,
    "interrupted_sequence_action_count": 57,
    "sequence_level_count": 0,
    "maximum_identical_action_run": 22,
    "completed_reset_controller_cycle_p95_ms": 24818.78279999364,
    "used_for_training": False,
    "registry_used_as_prior": False,
    "mutated_by_t10_3_10": False,
    "physical_actions_replayed": 0,
}


def _signed(payload: Mapping[str, Any], checksum_field: str) -> dict[str, Any]:
    output = dict(payload)
    output[checksum_field] = sha256_payload(output)
    return output


def _parent_root(root: Path) -> Path:
    return (
        root
        / "training"
        / "sage_t"
        / "t10_3_9_causal_subgoal_composition"
    )


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


def _maximum_action_run(paths: Sequence[Path]) -> int:
    maximum = 0
    by_work: dict[str, list[tuple[int, str]]] = {}
    for path in paths:
        intent = _read_signed(path, "intent_checksum")
        by_work.setdefault(str(intent["work_id"]), []).append(
            (int(intent["step_index"]), str(intent["action"]["name"]))
        )
    for rows in by_work.values():
        previous = None
        run = 0
        for _, action in sorted(rows):
            if action == previous:
                run += 1
            else:
                previous = action
                run = 1
            maximum = max(maximum, run)
    return maximum


def _parent_diagnosis(root: Path) -> dict[str, Any]:
    manifest = parent.load_manifest(root)
    intents = _journal_files(root, "intents")
    events = _journal_files(root, "events")
    unresolved = _journal_files(root, "unresolved")
    receipts = [
        _read_signed(path, "receipt_checksum")
        for path in _journal_files(root, "branches")
        if path.name == "receipt.json"
    ]
    started = {path.parent.name for path in intents}
    complete = {str(row["work_id"]) for row in receipts}
    incomplete = started - complete
    completed_actions = sum(int(row.get("sealed_events", 0)) for row in receipts)
    return {
        "manifest_checksum": str(manifest["manifest_checksum"]),
        "intent_count": len(intents),
        "event_count": len(events),
        "branch_count": len(receipts),
        "unresolved_count": len(unresolved),
        "incomplete_work_count": len(incomplete),
        "completed_sequence_action_count": completed_actions,
        "interrupted_sequence_action_count": len(events) - completed_actions,
        "sequence_level_count": sum(
            int(row.get("level_delta", 0)) for row in receipts
        ),
        "maximum_identical_action_run": _maximum_action_run(intents),
        "completed_reset_controller_cycle_p95_ms": max(
            (float(row.get("controller_cycle_p95_ms", 0.0)) for row in receipts),
            default=0.0,
        ),
    }


def _verify_parent(root: Path) -> dict[str, Any]:
    parent.load_manifest(root)
    for name, binding in PARENT_ARTIFACTS.items():
        path = root / binding["path"]
        if not path.is_file() or file_sha256(path) != binding["sha256"]:
            raise IntegrityError(f"required T10.3.9 artifact absent or drifted: {name}")
    if (_parent_root(root) / "collector.lock.json").exists():
        raise IntegrityError("T10.3.9 collector lock must be absent")
    diagnosis = _parent_diagnosis(root)
    expected = {key: SUPERSEDED_T10_3_9[key] for key in diagnosis}
    if diagnosis != expected:
        raise IntegrityError("T10.3.9 interrupted diagnosis diverged")
    return diagnosis


def _code_hashes(root: Path) -> dict[str, str]:
    relative = (
        "theory/sage_t/goal_directed_v10_3_10.py",
        "theory/sage_t/t10_3_10_protocol.py",
        "theory/sage_t/t10_3_10_runtime.py",
        "theory/sage_t/goal_directed_v10_3_9.py",
        "theory/sage_t/t10_3_5_runtime.py",
        "tests/test_sage_t_goal_directed_v10_3_10.py",
        "tests/test_sage_t_t10_3_10_protocol.py",
        "tests/test_sage_t_t10_3_10_runtime.py",
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
        for game in SEQUENCE_GAMES:
            for index, seed in enumerate(DISCOVERY_SEEDS):
                rows.append(
                    WorkSpec(
                        phase,
                        game,
                        seed,
                        "goal_directed_sage_t",
                        index,
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
                        "goal_directed_sage_t",
                        index,
                        SEQUENCE_ACTION_BUDGET,
                    )
                )
        return tuple(rows)
    if phase == "confirm":
        for game_index, game in enumerate(ALL_SOURCE_GAMES):
            budget = (
                CORE_ACTION_BUDGET if game in CORE_GAMES else SEQUENCE_ACTION_BUDGET
            )
            for seed_index, seed in enumerate(CONFIRMATION_SEEDS):
                arms = list(CONFIRMATION_ARMS)
                if (game_index + seed_index) % 2:
                    arms.reverse()
                for arm_index, arm in enumerate(arms):
                    rows.append(
                        WorkSpec(phase, game, seed, arm, arm_index, budget)
                    )
        return tuple(rows)
    raise ValueError(f"unsupported physical phase: {phase}")


def maximum_actions_for_phase(phase: str) -> int:
    return sum(work.action_budget for work in work_specs(phase))


def maximum_actions_for_specs(specs: Sequence[WorkSpec]) -> int:
    return sum(int(item.action_budget) for item in specs)


def reset_wall_seconds(work: WorkSpec) -> float:
    return (
        CORE_RESET_WALL_SECONDS
        if work.game_id in CORE_GAMES
        else SEQUENCE_RESET_WALL_SECONDS
    )


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
        "total_maximum_actions": sum(
            row["maximum_actions"] for row in phases.values()
        ),
    }


TOTAL_RESETS = _matrix_payload()["total_resets"]
TOTAL_MAXIMUM_ACTIONS = _matrix_payload()["total_maximum_actions"]


def build_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    diagnosis = _verify_parent(root)
    core = {
        "format_version": FORMAT_VERSION,
        "status": MANIFEST_STATUS,
        "objective": "escape_repeated_effect_cycles_and_test_directional_subgoal_progress",
        "parent_artifacts": PARENT_ARTIFACTS,
        "superseded_t10_3_9": {
            **SUPERSEDED_T10_3_9,
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
            "directional_structural_gain_is_exploration_only": True,
            "raw_state_novelty_rewarded": False,
            "repeated_action_effect_stall_limit": 3,
            "maximum_planned_identical_action_run": 2,
            "maximum_directional_option_horizon": 6,
            "exact_state_cycle_abort": True,
            "transition_history_limit": 8,
            "relational_rule_verification_deferred": True,
            "mechanic_theory_updated_each_transition": True,
            "sage_t_posterior_updated_each_transition": True,
            "parent_events_diagnostic_only": True,
            "parent_registry_loaded": False,
            "t10_3_8_core_registry_prior_support_zero": True,
            "latency_is_telemetry_only": True,
        },
        "gates": {
            "discovery_sequence_level_total_minimum": 1,
            "discovery_mixed_winner_required": True,
            "reproduction_same_game_required": True,
            "reproduction_mixed_winner_required": True,
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
            "t10_3_9_events_training_authorized": False,
            "t10_3_9_registry_prior_authorized": False,
            "t10_3_8_core_registry_structural_prior_authorized": True,
        },
        "durability": {
            "intent_before_action": True,
            "event_immediate_seal": True,
            "physical_replay": False,
            "write_once": True,
            "interrupted_parent_intent_replayed": False,
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
            "format_version": "sage-t10.3.10-directional-progress-migration-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "superseded_protocol": "SAGE.T10.3.9",
            "superseded_status": SUPERSEDED_T10_3_9["status"],
            "parent_journal_digest": manifest["superseded_t10_3_9"][
                "journal_digest"
            ],
            "diagnosis": manifest["superseded_t10_3_9"]["diagnosis"],
            "parent_events_used_for_training": 0,
            "parent_registry_loaded": False,
            "parent_core_registry_loaded_as_structural_prior": True,
            "parent_core_registry_local_support": 0,
            "parent_artifacts_mutated": False,
            "parent_physical_actions_replayed": 0,
            "repairs": [
                "context_once_directional_structural_gain",
                "repeated_action_effect_stall_abort",
                "simulated_frontier_usage_updates",
                "two_action_run_cap",
                "bounded_transition_history",
                "deferred_online_relational_rule_verification",
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
        raise IntegrityError("T10.3.10 manifest has not been frozen")
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_signed(payload, "manifest_checksum")
    if payload.get("format_version") != FORMAT_VERSION:
        raise IntegrityError("T10.3.10 manifest format drifted")
    _verify_parent(root)
    if payload.get("superseded_t10_3_9", {}).get(
        "journal_digest"
    ) != _parent_journal_digest(root):
        raise IntegrityError("T10.3.9 journal changed after T10.3.10 freeze")
    if verify_code and payload.get("code_hashes") != _code_hashes(root):
        raise IntegrityError("T10.3.10 code hash drifted after freeze")
    migration_path = root / DEFAULT_MIGRATION_PATH
    if not migration_path.is_file():
        raise IntegrityError("T10.3.10 migration receipt is absent")
    migration = json.loads(migration_path.read_text(encoding="utf-8"))
    verify_signed(migration, "receipt_checksum")
    if migration.get("manifest_checksum") != payload.get("manifest_checksum"):
        raise IntegrityError("T10.3.10 migration receipt is detached")
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
    "DISCOVERY_SEEDS",
    "IntegrityError",
    "PARENT_ARTIFACTS",
    "REPRODUCTION_SEEDS",
    "SEQUENCE_ACTION_BUDGET",
    "SEQUENCE_GAMES",
    "SUPERSEDED_T10_3_9",
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
