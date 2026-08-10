"""Frozen boolean-gate adjudication protocol for SAGE.T10.3.8."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import t10_3_7_protocol as parent

FORMAT_VERSION = "sage-t10.3.8-witness-gate-adjudication-protocol-v1"
MANIFEST_STATUS = "FROZEN_BEFORE_T10_3_8_SOURCE_ACTION"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name("sage_t10_3_8_protocol_manifest.json")
DEFAULT_MIGRATION_PATH = Path(__file__).with_name("sage_t10_3_8_migration_receipt.json")
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "t10_3_8_witness_gate_adjudication"

CORE_GAMES = parent.CORE_GAMES
SEQUENCE_GAMES = parent.SEQUENCE_GAMES
ALL_SOURCE_GAMES = parent.ALL_SOURCE_GAMES
WITNESS_PROGRAMS = parent.WITNESS_PROGRAMS
DISCOVERY_SEEDS = (3321, 3322, 3323, 3324)
REPRODUCTION_SEEDS = (3331, 3332)
SEQUENCE_SEEDS = (3341, 3342)
CONFIRMATION_SEEDS = (3351, 3352)
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
    "t10_3_7_manifest": {
        "path": "theory/sage_t/sage_t10_3_7_protocol_manifest.json",
        "sha256": "07875cec8bc03045862a688daad3be7a3bff009e1135b374fb04e694fd34c617",
    },
    "t10_3_7_migration": {
        "path": "theory/sage_t/sage_t10_3_7_migration_receipt.json",
        "sha256": "47d2196e50d0aefa7231c25971c6237ed5bb1aca0316d287da75c1eeaddeb975",
    },
    "t10_3_7_audit": {
        "path": "training/sage_t/t10_3_7_stable_successor_recovery/offline_audit.json",
        "sha256": "b56b45e86f1effd508de4f2a6e816530cbfdc588c3d72807e726d7c72b4b4062",
    },
    "t10_3_7_preflight": {
        "path": "training/sage_t/t10_3_7_stable_successor_recovery/synthetic_preflight.json",
        "sha256": "b4fa0abdea4a14aaee43635abeb48e71337b6c46519b9c3842d1b6020ff01bd4",
    },
    "t10_3_7_checkpoint": {
        "path": "training/sage_t/t10_3_7_stable_successor_recovery/checkpoint.json",
        "sha256": "c0f9edf3edbbea9738832826b36b55b0c724ab9019f69184562b6e3157e5e197",
    },
    "t10_3_7_witness_report": {
        "path": "training/sage_t/t10_3_7_stable_successor_recovery/canonical_witness_report.json",
        "sha256": "1c2d60c60ba2d8b68507784e8cff4c1c878fb819c081f0dfad41506d4f6b989f",
    },
}

SUPERSEDED_T10_3_7 = {
    "status": "ADJUDICATED_VALID_WITNESS_BOOLEAN_POLARITY_BUG",
    "intent_count": 38,
    "event_count": 38,
    "unresolved_count": 0,
    "branch_count": 4,
    "lp85_level_delta": 1,
    "su15_level_delta": 2,
    "false_checks": ("historical_grounded_actions_loaded",),
    "parent_passed": False,
    "parent_verdict": "CANONICAL_WITNESS_MISS",
    "used_for_training": False,
    "mutated_by_t10_3_8": False,
    "physical_actions_replayed": 0,
}


def _signed(payload: Mapping[str, Any], checksum_field: str) -> dict[str, Any]:
    result = dict(payload)
    result[checksum_field] = sha256_payload(result)
    return result


def _parent_root(root: Path) -> Path:
    return root / "training" / "sage_t" / "t10_3_7_stable_successor_recovery"


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
    report = _read_signed(_parent_root(root) / "canonical_witness_report.json", "report_checksum")
    levels = report.get("metrics", {}).get("levels", {})
    false_checks = tuple(
        sorted(key for key, value in report.get("checks", {}).items() if value is False)
    )
    return {
        "intent_count": len(_journal_files(root, "intents")),
        "event_count": len(_journal_files(root, "events")),
        "unresolved_count": len(_journal_files(root, "unresolved")),
        "branch_count": len(_journal_files(root, "branches")),
        "lp85_level_delta": int(levels.get("lp85-305b61c3", 0)),
        "su15_level_delta": int(levels.get("su15-4c352900", 0)),
        "false_checks": false_checks,
        "parent_passed": bool(report.get("passed")),
        "parent_verdict": str(report.get("verdict", "")),
    }


def _verify_parent(root: Path) -> dict[str, Any]:
    parent.load_manifest(root)
    for name, binding in PARENT_ARTIFACTS.items():
        path = root / binding["path"]
        if not path.is_file() or file_sha256(path) != binding["sha256"]:
            raise IntegrityError(f"required T10.3.7 artifact absent or drifted: {name}")
    if (_parent_root(root) / "collector.lock.json").exists():
        raise IntegrityError("T10.3.7 collector lock must be absent")
    diagnosis = _parent_diagnosis(root)
    expected = {key: SUPERSEDED_T10_3_7[key] for key in diagnosis}
    if diagnosis != expected:
        raise IntegrityError("T10.3.7 witness adjudication snapshot diverged")
    return diagnosis


def _code_hashes(root: Path) -> dict[str, str]:
    relative = (
        "theory/sage_t/t10_3_8_protocol.py",
        "theory/sage_t/t10_3_8_runtime.py",
        "theory/sage_t/goal_directed_v10_3_7.py",
        "theory/sage_t/t10_3_7_runtime.py",
        "theory/sage_t/t10_3_6_runtime.py",
        "tests/test_sage_t_t10_3_8_protocol.py",
        "tests/test_sage_t_t10_3_8_runtime.py",
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
        for phase in ("discover-core", "reproduce-core", "discover-sequence", "confirm")
    }
    return {
        "adjudication_physical_actions": 0,
        "adjudication_resets": 0,
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
        "objective": "adjudicate_expected_negative_boolean_then_resume_blank_functional_discovery",
        "parent_artifacts": PARENT_ARTIFACTS,
        "superseded_t10_3_7": {
            **SUPERSEDED_T10_3_7,
            "journal_digest": _parent_journal_digest(root),
            "diagnosis": diagnosis,
        },
        "canonical_witness_descriptors": WITNESS_PROGRAMS,
        "code_hashes": _code_hashes(root),
        "cli_phases": [
            "freeze", "status", "audit", "adjudicate", "discover-core",
            "reproduce-core", "discover-sequence", "compile", "confirm", "report",
        ],
        "exit_codes": {"success": 0, "integrity": 2, "scientific_gate": 3},
        "matrix": _matrix_payload(),
        "adjudication_contract": {
            "parent_report_immutable": True,
            "parent_false_field": "historical_grounded_actions_loaded",
            "normalized_positive_gate": "historical_grounded_actions_absent",
            "expected_parent_value": False,
            "normalized_gate_value": True,
            "all_other_parent_checks_required_true": True,
            "witness_recollection_authorized": False,
            "parent_physical_replay_authorized": False,
            "parent_events_fit_authorized": False,
        },
        "functional_contract": {
            "fresh_path_induced_per_reset": True,
            "fresh_path_held_ephemerally_during_option": True,
            "each_waypoint_reacquired_from_current_legal_actions": True,
            "blank_posterior_discovery": True,
            "level_increment_is_only_success_credit": True,
            "latency_is_telemetry_only": True,
            "no_latency_scientific_gate": True,
        },
        "firewall": {
            "source_validation_opened": False,
            "ar25_opened": False,
            "holdout_opened": False,
            "production_authority": False,
            "automatic_retuning": False,
            "t10_3_7_events_training_authorized": False,
            "t10_3_7_physical_replay_authorized": False,
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
            "format_version": "sage-t10.3.8-witness-gate-adjudication-migration-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "superseded_protocol": "SAGE.T10.3.7",
            "superseded_status": SUPERSEDED_T10_3_7["status"],
            "parent_journal_digest": manifest["superseded_t10_3_7"]["journal_digest"],
            "diagnosis": manifest["superseded_t10_3_7"]["diagnosis"],
            "parent_events_used_for_training": 0,
            "parent_artifacts_mutated": False,
            "parent_physical_actions_replayed": 0,
            "witness_recollection_actions": 0,
            "correction": "normalize_expected_false_safety_assertion_to_positive_absence_gate",
        },
        "receipt_checksum",
    )
    write_json_once(migration_path or root / DEFAULT_MIGRATION_PATH, migration)
    return manifest, migration


def load_manifest(root: Path, *, verify_code: bool = True) -> dict[str, Any]:
    root = root.resolve()
    path = root / DEFAULT_MANIFEST_PATH
    if not path.is_file():
        raise IntegrityError("T10.3.8 manifest has not been frozen")
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_signed(payload, "manifest_checksum")
    if payload.get("format_version") != FORMAT_VERSION:
        raise IntegrityError("T10.3.8 manifest format drifted")
    _verify_parent(root)
    if payload.get("superseded_t10_3_7", {}).get("journal_digest") != _parent_journal_digest(root):
        raise IntegrityError("T10.3.7 journal changed after adjudication")
    if verify_code and payload.get("code_hashes") != _code_hashes(root):
        raise IntegrityError("T10.3.8 code hash drifted after freeze")
    migration_path = root / DEFAULT_MIGRATION_PATH
    if not migration_path.is_file():
        raise IntegrityError("T10.3.8 migration receipt is absent")
    migration = json.loads(migration_path.read_text(encoding="utf-8"))
    verify_signed(migration, "receipt_checksum")
    if migration.get("manifest_checksum") != payload.get("manifest_checksum"):
        raise IntegrityError("T10.3.8 migration receipt is detached")
    return payload


__all__ = [
    "ALL_SOURCE_GAMES", "CONFIRMATION_ARMS", "CONFIRMATION_SEEDS",
    "CORE_GAMES", "DEFAULT_MANIFEST_PATH", "DEFAULT_MIGRATION_PATH",
    "DEFAULT_OUTPUT_DIR", "DISCOVERY_SEEDS", "IntegrityError",
    "REPRODUCTION_SEEDS", "SEQUENCE_GAMES", "SEQUENCE_SEEDS",
    "ScientificGateMiss", "SUPERSEDED_T10_3_7", "TOTAL_MAXIMUM_ACTIONS",
    "TOTAL_RESETS", "WITNESS_PROGRAMS", "WorkSpec", "build_manifest",
    "file_sha256", "freeze_manifest", "load_manifest",
    "maximum_actions_for_phase", "maximum_actions_for_specs",
    "reset_wall_seconds", "sha256_payload", "verify_signed", "work_specs",
    "write_json_once",
]
