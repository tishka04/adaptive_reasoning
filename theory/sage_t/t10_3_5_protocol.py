"""Frozen scheduled real-time continuation protocol for SAGE.T10.3.5."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import t10_3_4_protocol as parent

FORMAT_VERSION = "sage-t10.3.5-scheduled-real-time-recovery-protocol-v1"
MANIFEST_STATUS = "FROZEN_BEFORE_T10_3_5_SOURCE_ACTION"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name("sage_t10_3_5_protocol_manifest.json")
DEFAULT_MIGRATION_PATH = Path(__file__).with_name("sage_t10_3_5_migration_receipt.json")
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "t10_3_5_scheduled_real_time_recovery"

CORE_GAMES = parent.CORE_GAMES
SEQUENCE_GAMES = parent.SEQUENCE_GAMES
ALL_SOURCE_GAMES = (*CORE_GAMES, *SEQUENCE_GAMES)
DISCOVERY_SEEDS = (3201, 3202)
CONFIRMATION_SEEDS = (3211, 3212)
CONFIRMATION_ARMS = parent.CONFIRMATION_ARMS
CORE_ACTION_BUDGET = parent.CORE_ACTION_BUDGET
SEQUENCE_ACTION_BUDGET = parent.SEQUENCE_ACTION_BUDGET
CORE_DISCOVERY_ACTIONS = parent.CORE_DISCOVERY_ACTIONS
SEQUENCE_DISCOVERY_ACTIONS = parent.SEQUENCE_DISCOVERY_ACTIONS
CONFIRMATION_ACTIONS = parent.CONFIRMATION_ACTIONS
TOTAL_RESETS = parent.TOTAL_RESETS
TOTAL_MAXIMUM_ACTIONS = parent.TOTAL_MAXIMUM_ACTIONS
CORE_RESET_WALL_SECONDS = parent.CORE_RESET_WALL_SECONDS
SEQUENCE_RESET_WALL_SECONDS = parent.SEQUENCE_RESET_WALL_SECONDS
MAXIMUM_DECISION_P95_MS = parent.MAXIMUM_DECISION_P95_MS
MAXIMUM_CONTROLLER_CYCLE_P95_MS = 2500.0

IntegrityError = parent.IntegrityError
ScientificGateMiss = parent.ScientificGateMiss
sha256_payload = parent.sha256_payload
file_sha256 = parent.file_sha256
verify_signed = parent.verify_signed
write_json_once = parent.write_json_once

PARENT_ARTIFACTS = {
    "t10_3_4_manifest": {
        "path": "theory/sage_t/sage_t10_3_4_protocol_manifest.json",
        "sha256": "7af91b1d3799e9a04b9557fce4bd3683ca608e9a20d362e68ad8b1468f6caee7",
    },
    "t10_3_4_migration": {
        "path": "theory/sage_t/sage_t10_3_4_migration_receipt.json",
        "sha256": "0aae0b105ee299671b6c5b9747c45d2806077f895b3d9b532aa05a0d7b016ede",
    },
    "t10_3_4_audit": {
        "path": "training/sage_t/t10_3_4_bounded_compute_recovery/offline_audit.json",
        "sha256": "f72f5febf3106fa37fb4c62d8ab55ed0d8e3375847d1a394737fa969f5f7a098",
    },
    "t10_3_4_preflight": {
        "path": "training/sage_t/t10_3_4_bounded_compute_recovery/synthetic_preflight.json",
        "sha256": "a0766e3d9bb6a5de632d01a6bf24bd9eb696adddc94facba85bddd41e013f62f",
    },
    "t10_3_4_checkpoint": {
        "path": "training/sage_t/t10_3_4_bounded_compute_recovery/checkpoint.json",
        "sha256": "4be08a54338577ec672173bd1dc4c3b7faaddf1b2cce3b39d545cf9d922bebb2",
    },
    "t10_3_4_core_report": {
        "path": "training/sage_t/t10_3_4_bounded_compute_recovery/discovery_core_report.json",
        "sha256": "13b04047efcbddb82e0667364a9aaa147aa08e31643fc6b989b561874eca7332",
    },
    "t10_3_4_terminal_report": {
        "path": "training/sage_t/t10_3_4_bounded_compute_recovery/terminal_report.json",
        "sha256": "0e80941749bdd9e3d9ec573dad10e39273ea4fa4a4daaf208a0eaa6cba655daa",
    },
}

SUPERSEDED_T10_3_4 = {
    "status": "SUPERSEDED_COMPLETE_NEGATIVE_COMPUTE_BOUND",
    "intent_count": 126,
    "event_count": 126,
    "unresolved_count": 0,
    "branch_count": 4,
    "level_delta": 1,
    "game_over_count": 3,
    "decision_p95_ms": 5496.576999983517,
    "fast_path_decision_count": 28,
    "verdict": "BOUNDED_CORE_MISS",
    "used_for_training": False,
    "positive_witness_imported_as_prior": False,
    "mutated_by_t10_3_5": False,
    "physical_actions_replayed": 0,
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _signed(payload: Mapping[str, Any], checksum_field: str) -> dict[str, Any]:
    result = dict(payload)
    result[checksum_field] = sha256_payload(result)
    return result


def _parent_root(root: Path) -> Path:
    return root / "training" / "sage_t" / "t10_3_4_bounded_compute_recovery"


def _journal_files(root: Path, category: str) -> tuple[Path, ...]:
    directory = _parent_root(root) / "journal" / category
    return tuple(sorted(directory.rglob("*.json"))) if directory.exists() else ()


def _parent_journal_digest(root: Path) -> str:
    base = _parent_root(root) / "journal"
    rows = [
        {"relative_path": path.relative_to(base).as_posix(), "sha256": file_sha256(path)}
        for path in sorted(item for item in base.rglob("*.json") if item.is_file())
    ]
    return sha256_payload(rows)


def _load_rows(root: Path, category: str, checksum_field: str) -> list[dict[str, Any]]:
    rows = []
    for path in _journal_files(root, category):
        payload = json.loads(path.read_text(encoding="utf-8"))
        verify_signed(payload, checksum_field)
        rows.append(payload)
    return rows


def _parent_diagnosis(root: Path) -> dict[str, Any]:
    intents = _load_rows(root, "intents", "intent_checksum")
    events = _load_rows(root, "events", "event_checksum")
    branches = _load_rows(root, "branches", "receipt_checksum")
    core = json.loads(
        (_parent_root(root) / "discovery_core_report.json").read_text(encoding="utf-8")
    )
    verify_signed(core, "report_checksum")
    terminal = json.loads(
        (_parent_root(root) / "terminal_report.json").read_text(encoding="utf-8")
    )
    verify_signed(terminal, "report_checksum")
    return {
        "intent_count": len(intents),
        "event_count": len(events),
        "unresolved_count": len(_journal_files(root, "unresolved")),
        "branch_count": len(branches),
        "level_delta": sum(int(row.get("level_delta", 0)) for row in events),
        "game_over_count": sum(
            str(row.get("game_state_after", "")).upper() == "GAME_OVER"
            for row in events
        ),
        "decision_p95_ms": float(core.get("metrics", {}).get("decision_p95_ms", 0.0)),
        "fast_path_decision_count": int(
            core.get("metrics", {}).get("fast_path_decisions", 0)
        ),
        "verdict": str(terminal.get("verdict", "")),
    }


def _verify_parent(root: Path) -> dict[str, Any]:
    parent.load_manifest(root)
    for name, binding in PARENT_ARTIFACTS.items():
        path = root / str(binding["path"])
        if not path.is_file():
            raise IntegrityError(f"required parent artifact is absent: {name}")
        if file_sha256(path) != str(binding["sha256"]):
            raise IntegrityError(f"required parent artifact drifted: {name}")
    if (_parent_root(root) / "collector.lock.json").exists():
        raise IntegrityError("T10.3.4 collector lock must be absent before supersession")
    diagnosis = _parent_diagnosis(root)
    expected = {key: SUPERSEDED_T10_3_4[key] for key in diagnosis}
    if diagnosis != expected:
        raise IntegrityError("T10.3.4 terminal snapshot diverged")
    return diagnosis


def _code_hashes(root: Path) -> dict[str, str]:
    relative = (
        "theory/sage_t/goal_directed_v10_3_5.py",
        "theory/sage_t/t10_3_5_protocol.py",
        "theory/sage_t/t10_3_5_runtime.py",
        "theory/sage_t/goal_directed_v10_3_4.py",
        "theory/sage_t/t10_3_4_runtime.py",
        "theory/sage_t/goal_directed_v10_3_3.py",
        "theory/sage_t/goal_directed_v10_3_2.py",
        "theory/unified_cognitive_controller.py",
        "theory/sage_t/frame_adapters_v10_3.py",
        "theory/sage_t/progress_witness_v10.py",
        "tests/test_sage_t_goal_directed_v10_3_5.py",
        "tests/test_sage_t_t10_3_5_protocol.py",
        "tests/test_sage_t_t10_3_5_runtime.py",
    )
    output = {}
    for item in relative:
        path = root / item
        if not path.is_file():
            raise IntegrityError(f"protocol code dependency is absent: {item}")
        output[item] = file_sha256(path)
    return output


def _matrix_payload() -> dict[str, Any]:
    return {
        "discovery": {
            "core": {
                "games": list(CORE_GAMES),
                "seeds": list(DISCOVERY_SEEDS),
                "actions_per_reset": CORE_ACTION_BUDGET,
                "maximum_actions": CORE_DISCOVERY_ACTIONS,
            },
            "sequence": {
                "games": list(SEQUENCE_GAMES),
                "seeds": list(DISCOVERY_SEEDS),
                "actions_per_reset": SEQUENCE_ACTION_BUDGET,
                "maximum_actions": SEQUENCE_DISCOVERY_ACTIONS,
            },
        },
        "confirmation": {
            "games": list(ALL_SOURCE_GAMES),
            "seeds": list(CONFIRMATION_SEEDS),
            "arms": list(CONFIRMATION_ARMS),
            "core_actions_per_reset": CORE_ACTION_BUDGET,
            "sequence_actions_per_reset": SEQUENCE_ACTION_BUDGET,
            "maximum_actions": CONFIRMATION_ACTIONS,
            "counterbalanced": True,
        },
        "total_resets": TOTAL_RESETS,
        "total_maximum_actions": TOTAL_MAXIMUM_ACTIONS,
    }


def build_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    diagnosis = _verify_parent(root)
    core = {
        "format_version": FORMAT_VERSION,
        "status": MANIFEST_STATUS,
        "objective": "end_to_end_progress_under_scheduled_real_time_control",
        "parent_artifacts": PARENT_ARTIFACTS,
        "superseded_t10_3_4": {
            **SUPERSEDED_T10_3_4,
            "journal_digest": _parent_journal_digest(root),
            "diagnosis": diagnosis,
        },
        "code_hashes": _code_hashes(root),
        "cli_phases": [
            "freeze", "status", "audit", "preflight", "discover-core",
            "discover-sequence", "compile", "confirm", "report",
        ],
        "exit_codes": {"success": 0, "integrity": 2, "scientific_gate": 3},
        "matrix": _matrix_payload(),
        "scheduled_control": {
            "shared_legacy_proposal_each_action": True,
            "full_unified_decision_path_enabled": False,
            "full_unified_observation_path_enabled": False,
            "lightweight_effect_model_each_transition": True,
            "sage_t_posterior_each_transition": True,
            "active_option_fast_path": True,
            "productive_option_extension": True,
            "maximum_option_horizon": 32,
            "discovery_warmup_actions": 8,
            "exploration_actions_between_options": 8,
            "transition_history_limit": 32,
            "same_schedule_for_active_and_baseline": True,
            "stop_after_first_sealed_level": True,
            "core_reset_wall_seconds": CORE_RESET_WALL_SECONDS,
            "sequence_reset_wall_seconds": SEQUENCE_RESET_WALL_SECONDS,
            "stage_timing_required": True,
            "terminal_option_contradiction_recorded": True,
        },
        "gates": {
            "parent_negative_attested_but_not_reused": True,
            "preflight_ambiguous_parameterized_targets": True,
            "preflight_scheduled_active_and_baseline": True,
            "preflight_productive_option_extension": True,
            "core_progress_each_game_and_seed": True,
            "winning_decision_from_sage_t": True,
            "minimum_sage_t_physical_action_per_core_reset": 1,
            "sequence_progress_minimum_games": 1,
            "sequence_minimum_distinct_action_schemas": 2,
            "registry_independent_reproduction_count": 2,
            "confirmation_total_level_advantage": 1,
            "maximum_decision_p95_ms": MAXIMUM_DECISION_P95_MS,
            "maximum_controller_cycle_p95_ms": MAXIMUM_CONTROLLER_CYCLE_P95_MS,
            "structural_collision_policy": "fail_closed_if_observed_not_required_to_occur",
        },
        "firewall": {
            "source_validation_opened": False,
            "ar25_opened": False,
            "holdout_opened": False,
            "production_authority": False,
            "automatic_retuning": False,
            "t10_3_4_events_training_authorized": False,
            "t10_3_4_positive_witness_prior_authorized": False,
            "t10_3_4_physical_replay_authorized": False,
        },
        "durability": {
            "intent_before_action": True,
            "event_immediate_seal": True,
            "physical_replay": False,
            "single_live_inflight_intent": True,
            "write_once": True,
            "planned_early_stop_receipt": True,
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
    target = manifest_path or (root / DEFAULT_MANIFEST_PATH)
    migration_target = migration_path or (root / DEFAULT_MIGRATION_PATH)
    write_json_once(target, manifest)
    migration = _signed(
        {
            "format_version": "sage-t10.3.5-scheduled-real-time-migration-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "superseded_protocol": "SAGE.T10.3.4",
            "superseded_status": SUPERSEDED_T10_3_4["status"],
            "parent_journal_digest": manifest["superseded_t10_3_4"]["journal_digest"],
            "diagnosis": manifest["superseded_t10_3_4"]["diagnosis"],
            "parent_events_used_for_training": 0,
            "parent_positive_witness_imported_as_prior": False,
            "parent_artifacts_mutated": False,
            "parent_physical_actions_replayed": 0,
        },
        "receipt_checksum",
    )
    write_json_once(migration_target, migration)
    return manifest, migration


def load_manifest(root: Path, *, verify_code: bool = True) -> dict[str, Any]:
    root = root.resolve()
    path = root / DEFAULT_MANIFEST_PATH
    if not path.is_file():
        raise IntegrityError("T10.3.5 manifest has not been frozen")
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_signed(payload, "manifest_checksum")
    if payload.get("format_version") != FORMAT_VERSION:
        raise IntegrityError("T10.3.5 manifest format drifted")
    _verify_parent(root)
    if payload.get("superseded_t10_3_4", {}).get("journal_digest") != _parent_journal_digest(root):
        raise IntegrityError("T10.3.4 journal changed after supersession")
    if verify_code and payload.get("code_hashes") != _code_hashes(root):
        raise IntegrityError("T10.3.5 code hash drifted after freeze")
    migration_path = root / DEFAULT_MIGRATION_PATH
    if not migration_path.is_file():
        raise IntegrityError("T10.3.5 migration receipt is absent")
    migration = json.loads(migration_path.read_text(encoding="utf-8"))
    verify_signed(migration, "receipt_checksum")
    if migration.get("manifest_checksum") != payload.get("manifest_checksum"):
        raise IntegrityError("T10.3.5 migration receipt is detached")
    return payload


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
    if phase in {"discover-core", "discover-sequence"}:
        games = CORE_GAMES if phase == "discover-core" else SEQUENCE_GAMES
        budget = CORE_ACTION_BUDGET if phase == "discover-core" else SEQUENCE_ACTION_BUDGET
        for game in games:
            for seed in DISCOVERY_SEEDS:
                rows.append(WorkSpec(phase, game, seed, "goal_directed_sage_t", 0, budget))
        return tuple(rows)
    if phase == "confirm":
        for game_index, game in enumerate(ALL_SOURCE_GAMES):
            budget = CORE_ACTION_BUDGET if game in CORE_GAMES else SEQUENCE_ACTION_BUDGET
            for seed_index, seed in enumerate(CONFIRMATION_SEEDS):
                arms = list(CONFIRMATION_ARMS)
                if (game_index + seed_index) % 2:
                    arms.reverse()
                for arm in arms:
                    rows.append(WorkSpec(phase, game, seed, arm, 0, budget))
        return tuple(rows)
    raise ValueError("work phase must be discover-core, discover-sequence or confirm")


def maximum_actions_for_phase(phase: str) -> int:
    return {
        "discover-core": CORE_DISCOVERY_ACTIONS,
        "discover-sequence": SEQUENCE_DISCOVERY_ACTIONS,
        "confirm": CONFIRMATION_ACTIONS,
    }[phase]


def maximum_actions_for_specs(specs: Sequence[WorkSpec]) -> int:
    return sum(int(item.action_budget) for item in specs)


def reset_wall_seconds(work: WorkSpec) -> float:
    return CORE_RESET_WALL_SECONDS if work.game_id in CORE_GAMES else SEQUENCE_RESET_WALL_SECONDS


__all__ = [
    "ALL_SOURCE_GAMES", "CONFIRMATION_ARMS", "CONFIRMATION_SEEDS",
    "DEFAULT_MANIFEST_PATH", "DEFAULT_MIGRATION_PATH", "DEFAULT_OUTPUT_DIR",
    "DISCOVERY_SEEDS", "IntegrityError", "MAXIMUM_CONTROLLER_CYCLE_P95_MS",
    "MAXIMUM_DECISION_P95_MS", "ScientificGateMiss", "SUPERSEDED_T10_3_4",
    "TOTAL_MAXIMUM_ACTIONS", "TOTAL_RESETS", "WorkSpec", "build_manifest",
    "file_sha256", "freeze_manifest", "load_manifest", "maximum_actions_for_phase",
    "maximum_actions_for_specs", "reset_wall_seconds", "sha256_payload",
    "verify_signed", "work_specs", "write_json_once",
]
