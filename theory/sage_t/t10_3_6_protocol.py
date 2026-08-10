"""Frozen functional end-to-end source protocol for SAGE.T10.3.6."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import t10_3_5_protocol as parent

FORMAT_VERSION = "sage-t10.3.6-functional-end-to-end-source-protocol-v1"
MANIFEST_STATUS = "FROZEN_BEFORE_T10_3_6_SOURCE_ACTION"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name("sage_t10_3_6_protocol_manifest.json")
DEFAULT_MIGRATION_PATH = Path(__file__).with_name("sage_t10_3_6_migration_receipt.json")
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "t10_3_6_functional_end_to_end"

CORE_GAMES = parent.CORE_GAMES
SEQUENCE_GAMES = parent.SEQUENCE_GAMES
ALL_SOURCE_GAMES = (*CORE_GAMES, *SEQUENCE_GAMES)
WITNESS_SEEDS = (3221, 3222)
DISCOVERY_SEEDS = (3231, 3232, 3233, 3234)
REPRODUCTION_SEEDS = (3241, 3242)
SEQUENCE_SEEDS = (3251, 3252)
CONFIRMATION_SEEDS = (3261, 3262)
CONFIRMATION_ARMS = ("goal_directed_sage_t", "unified_sage_t_off")
WITNESS_ACTION_BUDGET = 16
CORE_ACTION_BUDGET = 32
SEQUENCE_ACTION_BUDGET = 64
CORE_RESET_WALL_SECONDS = 3600.0
SEQUENCE_RESET_WALL_SECONDS = 7200.0

WITNESS_PROGRAMS = {
    "lp85-305b61c3": {
        "macro_schema": "repeat_target",
        "horizon": 5,
        "program_hash": "8dd32112126b37500c26d3af574e8ccd0aff957fc690774004dbe8a16f20dcb8",
        "target_selector": "same_effect_distinct_target",
    },
    "su15-4c352900": {
        "macro_schema": "path_successor",
        "horizon": 10,
        "program_hash": "3308687661481d60477337c7f9ef608f9b00229fcf118a898915712cce738387",
        "target_selector": "successor_toward_enclosure",
    },
}

IntegrityError = parent.IntegrityError
ScientificGateMiss = parent.ScientificGateMiss
sha256_payload = parent.sha256_payload
file_sha256 = parent.file_sha256
verify_signed = parent.verify_signed
write_json_once = parent.write_json_once

PARENT_ARTIFACTS = {
    "t10_3_5_manifest": {
        "path": "theory/sage_t/sage_t10_3_5_protocol_manifest.json",
        "sha256": "0aaa718e39aab9887774b1fe300fceaf4d3aae2b9623a2c2e26627a3842eaf91",
    },
    "t10_3_5_migration": {
        "path": "theory/sage_t/sage_t10_3_5_migration_receipt.json",
        "sha256": "067b85316c1206ac64d7707731094ce13318e4849683c332d3313bb2ef09e535",
    },
    "t10_3_5_audit": {
        "path": "training/sage_t/t10_3_5_scheduled_real_time_recovery/offline_audit.json",
        "sha256": "2214069b03bc13cec3d75019f073d7ad4224eb23d6dfe3e29963aa54731c133b",
    },
    "t10_3_5_preflight": {
        "path": "training/sage_t/t10_3_5_scheduled_real_time_recovery/synthetic_preflight.json",
        "sha256": "2a914ff5bb79cfa7bda3ee6f333da42703f884bb1e8bcf9be29314214061b6eb",
    },
    "t10_3_5_checkpoint": {
        "path": "training/sage_t/t10_3_5_scheduled_real_time_recovery/checkpoint.json",
        "sha256": "135111692095d0bfec92abf51bdb7484d760742e1189fac4c6b329d6ec021b52",
    },
    "t10_3_5_core_registry": {
        "path": "training/sage_t/t10_3_5_scheduled_real_time_recovery/core_registry_candidates.json",
        "sha256": "5ae72114046c389aa01afb0d5332c3409bb143b7b406766a66d0b1fb88ea3825",
    },
    "t10_3_5_core_report": {
        "path": "training/sage_t/t10_3_5_scheduled_real_time_recovery/discovery_core_report.json",
        "sha256": "e6e6c514536a1cfb6a7fda8c1e82978a4a3abd4f72bbe21d174d1916edf78fc9",
    },
}

T10_0B_ARTIFACTS = {
    "manifest": {
        "path": "theory/sage_t/sage_t10_0b_progress_witness_manifest.json",
        "sha256": "c68afaf876865e85296dd0eb4112253f54c273fe752aff6da397105d499e4e29",
    },
    "report": {
        "path": "training/sage_t/progress_witness_v10_0b/report.json",
        "sha256": "998b8af56b9933383bfe9b37aa4e245cffad3fb635d0abc8a19c5817724c044f",
    },
}

SUPERSEDED_T10_3_5 = {
    "status": "SUPERSEDED_COMPLETE_NEGATIVE_FUNCTION_UNRESOLVED",
    "intent_count": 90,
    "event_count": 90,
    "unresolved_count": 0,
    "branch_count": 4,
    "level_delta": 0,
    "game_over_count": 4,
    "decision_p95_ms": 2176.762199989753,
    "controller_cycle_p95_ms": 3133.8912999781314,
    "verdict": "REAL_TIME_BOUND_MISS",
    "used_for_training": False,
    "positive_witness_imported_as_prior": False,
    "mutated_by_t10_3_6": False,
    "physical_actions_replayed": 0,
}


def _signed(payload: Mapping[str, Any], checksum_field: str) -> dict[str, Any]:
    result = dict(payload)
    result[checksum_field] = sha256_payload(result)
    return result


def _parent_root(root: Path) -> Path:
    return root / "training" / "sage_t" / "t10_3_5_scheduled_real_time_recovery"


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


def _read_signed(path: Path, checksum_field: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_signed(payload, checksum_field)
    return payload


def _parent_diagnosis(root: Path) -> dict[str, Any]:
    intents = _journal_files(root, "intents")
    events = tuple(_read_signed(path, "event_checksum") for path in _journal_files(root, "events"))
    receipts = tuple(_read_signed(path, "receipt_checksum") for path in _journal_files(root, "branches"))
    core = _read_signed(_parent_root(root) / "discovery_core_report.json", "report_checksum")
    return {
        "intent_count": len(intents),
        "event_count": len(events),
        "unresolved_count": len(_journal_files(root, "unresolved")),
        "branch_count": len(receipts),
        "level_delta": sum(int(row.get("level_delta", 0)) for row in events),
        "game_over_count": sum(
            str(row.get("game_state_after", "")).upper() == "GAME_OVER" for row in events
        ),
        "decision_p95_ms": float(core["metrics"]["decision_p95_ms"]),
        "controller_cycle_p95_ms": float(core["metrics"]["controller_cycle_p95_ms"]),
        "verdict": str(core["verdict"]),
    }


def _verify_bound_files(root: Path, bindings: Mapping[str, Mapping[str, str]]) -> None:
    for name, binding in bindings.items():
        path = root / binding["path"]
        if not path.is_file():
            raise IntegrityError(f"required artifact is absent: {name}")
        if file_sha256(path) != binding["sha256"]:
            raise IntegrityError(f"required artifact drifted: {name}")


def _verify_parent(root: Path) -> dict[str, Any]:
    parent.load_manifest(root)
    _verify_bound_files(root, PARENT_ARTIFACTS)
    _verify_bound_files(root, T10_0B_ARTIFACTS)
    if (_parent_root(root) / "collector.lock.json").exists():
        raise IntegrityError("T10.3.5 collector lock must be absent before supersession")
    diagnosis = _parent_diagnosis(root)
    expected = {key: SUPERSEDED_T10_3_5[key] for key in diagnosis}
    if diagnosis != expected:
        raise IntegrityError("T10.3.5 functional snapshot diverged")
    report = _read_signed(root / T10_0B_ARTIFACTS["report"]["path"], "report_checksum")
    if report.get("status") != "PASS_T10_0_AUTHORIZE_T10_1":
        raise IntegrityError("T10.0b positive witness is not terminal-positive")
    return diagnosis


def _code_hashes(root: Path) -> dict[str, str]:
    relative = (
        "theory/sage_t/goal_directed_v10_3_6.py",
        "theory/sage_t/t10_3_6_protocol.py",
        "theory/sage_t/t10_3_6_runtime.py",
        "theory/sage_t/goal_directed_v10_3_5.py",
        "theory/sage_t/t10_3_5_runtime.py",
        "theory/sage_t/progress_witness_v10.py",
        "theory/unified_cognitive_controller.py",
        "tests/test_sage_t_goal_directed_v10_3_6.py",
        "tests/test_sage_t_t10_3_6_protocol.py",
        "tests/test_sage_t_t10_3_6_runtime.py",
    )
    output = {}
    for item in relative:
        path = root / item
        if not path.is_file():
            raise IntegrityError(f"protocol dependency is absent: {item}")
        output[item] = file_sha256(path)
    return output


def _phase_actions(phase: str) -> int:
    return sum(item.action_budget for item in work_specs(phase))


def _matrix_payload() -> dict[str, Any]:
    phases = {
        phase: {
            "resets": len(work_specs(phase)),
            "maximum_actions": _phase_actions(phase),
        }
        for phase in (
            "witness-core", "discover-core", "reproduce-core",
            "discover-sequence", "confirm",
        )
    }
    return {
        "phases": phases,
        "total_resets": sum(row["resets"] for row in phases.values()),
        "total_maximum_actions": sum(row["maximum_actions"] for row in phases.values()),
        "core_action_budget": CORE_ACTION_BUDGET,
        "sequence_action_budget": SEQUENCE_ACTION_BUDGET,
    }


def build_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    diagnosis = _verify_parent(root)
    core = {
        "format_version": FORMAT_VERSION,
        "status": MANIFEST_STATUS,
        "objective": "prove_explore_induce_execute_reproduce_and_transfer_level_progress",
        "parent_artifacts": PARENT_ARTIFACTS,
        "t10_0b_witness_artifacts": T10_0B_ARTIFACTS,
        "canonical_witness_descriptors": WITNESS_PROGRAMS,
        "superseded_t10_3_5": {
            **SUPERSEDED_T10_3_5,
            "journal_digest": _parent_journal_digest(root),
            "diagnosis": diagnosis,
        },
        "code_hashes": _code_hashes(root),
        "cli_phases": [
            "freeze", "status", "audit", "preflight", "witness-core",
            "discover-core", "reproduce-core", "discover-sequence",
            "compile", "confirm", "report",
        ],
        "exit_codes": {"success": 0, "integrity": 2, "scientific_gate": 3},
        "matrix": _matrix_payload(),
        "functional_contract": {
            "canonical_witness_is_diagnostic_only": True,
            "historical_grounded_actions_loaded": False,
            "fresh_regrounding_every_reset": True,
            "blank_posterior_discovery": True,
            "level_increment_is_only_success_credit": True,
            "visual_change_is_not_progress": True,
            "state_cycle_aborts_option": True,
            "ambiguous_bindings_balanced_across_fresh_resets": True,
            "latency_is_telemetry_only": True,
            "no_latency_scientific_gate": True,
        },
        "gates": {
            "witness_level_each_core_game": True,
            "witness_winning_action_from_sage_t": True,
            "blank_discovery_level_each_core_game": True,
            "fresh_reproduction_level_each_core_game": True,
            "sequence_level_minimum_games": 1,
            "sequence_minimum_action_schemas": 2,
            "compiled_independent_support_minimum": 2,
            "confirmation_core_level_each_game": True,
            "confirmation_sequence_level_minimum": 1,
            "confirmation_total_level_advantage": 1,
        },
        "firewall": {
            "source_validation_opened": False,
            "ar25_opened": False,
            "holdout_opened": False,
            "production_authority": False,
            "automatic_retuning": False,
            "t10_3_5_events_training_authorized": False,
            "t10_0b_events_training_authorized": False,
            "t10_0b_program_prior_for_discovery_authorized": False,
        },
        "durability": {
            "intent_before_action": True,
            "event_immediate_seal": True,
            "physical_replay": False,
            "single_live_inflight_intent": True,
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
    target = manifest_path or root / DEFAULT_MANIFEST_PATH
    migration_target = migration_path or root / DEFAULT_MIGRATION_PATH
    write_json_once(target, manifest)
    migration = _signed(
        {
            "format_version": "sage-t10.3.6-functional-migration-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "superseded_protocol": "SAGE.T10.3.5",
            "superseded_status": SUPERSEDED_T10_3_5["status"],
            "parent_journal_digest": manifest["superseded_t10_3_5"]["journal_digest"],
            "diagnosis": diagnosis if (diagnosis := manifest["superseded_t10_3_5"]["diagnosis"]) else {},
            "parent_events_used_for_training": 0,
            "t10_0b_grounded_actions_imported": 0,
            "t10_0b_programs_authorized_for_diagnostic_witness_only": True,
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
        raise IntegrityError("T10.3.6 manifest has not been frozen")
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_signed(payload, "manifest_checksum")
    if payload.get("format_version") != FORMAT_VERSION:
        raise IntegrityError("T10.3.6 manifest format drifted")
    _verify_parent(root)
    if payload.get("superseded_t10_3_5", {}).get("journal_digest") != _parent_journal_digest(root):
        raise IntegrityError("T10.3.5 journal changed after supersession")
    if verify_code and payload.get("code_hashes") != _code_hashes(root):
        raise IntegrityError("T10.3.6 code hash drifted after freeze")
    migration_path = root / DEFAULT_MIGRATION_PATH
    if not migration_path.is_file():
        raise IntegrityError("T10.3.6 migration receipt is absent")
    migration = json.loads(migration_path.read_text(encoding="utf-8"))
    verify_signed(migration, "receipt_checksum")
    if migration.get("manifest_checksum") != payload.get("manifest_checksum"):
        raise IntegrityError("T10.3.6 migration receipt is detached")
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
    return _phase_actions(phase)


def maximum_actions_for_specs(specs: Sequence[WorkSpec]) -> int:
    return sum(int(item.action_budget) for item in specs)


def reset_wall_seconds(work: WorkSpec) -> float:
    return CORE_RESET_WALL_SECONDS if work.game_id in CORE_GAMES else SEQUENCE_RESET_WALL_SECONDS


TOTAL_RESETS = _matrix_payload()["total_resets"]
TOTAL_MAXIMUM_ACTIONS = _matrix_payload()["total_maximum_actions"]


__all__ = [
    "ALL_SOURCE_GAMES", "CONFIRMATION_ARMS", "CONFIRMATION_SEEDS",
    "CORE_GAMES", "DEFAULT_MANIFEST_PATH", "DEFAULT_MIGRATION_PATH",
    "DEFAULT_OUTPUT_DIR", "DISCOVERY_SEEDS", "IntegrityError",
    "REPRODUCTION_SEEDS", "SEQUENCE_GAMES", "SEQUENCE_SEEDS",
    "ScientificGateMiss", "SUPERSEDED_T10_3_5", "TOTAL_MAXIMUM_ACTIONS",
    "TOTAL_RESETS", "WITNESS_PROGRAMS", "WITNESS_SEEDS", "WorkSpec",
    "build_manifest", "file_sha256", "freeze_manifest", "load_manifest",
    "maximum_actions_for_phase", "maximum_actions_for_specs",
    "reset_wall_seconds", "sha256_payload", "verify_signed", "work_specs",
    "write_json_once",
]
