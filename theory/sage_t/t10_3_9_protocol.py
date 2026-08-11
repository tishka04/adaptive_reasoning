"""Frozen causal-subgoal transfer protocol for SAGE.T10.3.9."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import t10_3_8_protocol as parent

FORMAT_VERSION = "sage-t10.3.9-causal-subgoal-transfer-protocol-v1"
MANIFEST_STATUS = "FROZEN_BEFORE_T10_3_9_SOURCE_ACTION"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name("sage_t10_3_9_protocol_manifest.json")
DEFAULT_MIGRATION_PATH = Path(__file__).with_name("sage_t10_3_9_migration_receipt.json")
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "t10_3_9_causal_subgoal_composition"

CORE_GAMES = parent.CORE_GAMES
SEQUENCE_GAMES = parent.SEQUENCE_GAMES
ALL_SOURCE_GAMES = (*CORE_GAMES, *SEQUENCE_GAMES)
DISCOVERY_SEEDS = (3361, 3362, 3363, 3364)
REPRODUCTION_SEEDS = (3371, 3372)
CONFIRMATION_SEEDS = (3381, 3382)
CONFIRMATION_ARMS = parent.CONFIRMATION_ARMS
CORE_ACTION_BUDGET = 32
SEQUENCE_ACTION_BUDGET = 96
CORE_RESET_WALL_SECONDS = parent.CORE_RESET_WALL_SECONDS
SEQUENCE_RESET_WALL_SECONDS = parent.SEQUENCE_RESET_WALL_SECONDS

IntegrityError = parent.IntegrityError
ScientificGateMiss = parent.ScientificGateMiss
sha256_payload = parent.sha256_payload
file_sha256 = parent.file_sha256
verify_signed = parent.verify_signed
write_json_once = parent.write_json_once

PARENT_ARTIFACTS = {
    "t10_3_8_manifest": {
        "path": "theory/sage_t/sage_t10_3_8_protocol_manifest.json",
        "sha256": "2e3319f2db15c3578fbcb646b3a82882aa3d259b3ad279b548b475f1d8129f37",
    },
    "t10_3_8_migration": {
        "path": "theory/sage_t/sage_t10_3_8_migration_receipt.json",
        "sha256": "88b3501d1efeeb11c43af942d1bfee584eddf9fa94399b4cea5cb88b6ed5b609",
    },
    "t10_3_8_audit": {
        "path": "training/sage_t/t10_3_8_witness_gate_adjudication/offline_audit.json",
        "sha256": "cf6ec7d439de7865b460f4a6532d4866fa99ce6fff4440e194e7b8c9d35eb2a3",
    },
    "t10_3_8_core": {
        "path": "training/sage_t/t10_3_8_witness_gate_adjudication/discovery_core_report.json",
        "sha256": "bb854bdf08e9374d3e2ba7ef20edd2f821b90e0c5f744319abb71796f61afe2e",
    },
    "t10_3_8_reproduction": {
        "path": "training/sage_t/t10_3_8_witness_gate_adjudication/reproduction_core_report.json",
        "sha256": "97479f5216c37b1f41c89d6ba53873d5f99ef2cc00496880207c2b177270ea1d",
    },
    "t10_3_8_core_registry": {
        "path": "training/sage_t/t10_3_8_witness_gate_adjudication/reproduced_core_registry.json",
        "sha256": "6346e98193c60b6444d414f27a5393677ef0529998ed0b2593281466816d5e4c",
    },
    "t10_3_8_sequence": {
        "path": "training/sage_t/t10_3_8_witness_gate_adjudication/discovery_sequence_report.json",
        "sha256": "374f0fb0d9e9eb82a742a7de88bd01915d51b9655cf94fc3b90f103f8605f5ac",
    },
    "t10_3_8_sequence_registry": {
        "path": "training/sage_t/t10_3_8_witness_gate_adjudication/sequence_registry_candidates.json",
        "sha256": "823f30d9883b10a308e63fa44e0e5d3b3ba46562644fa4eb2ef008adb8efd7a2",
    },
    "t10_3_8_terminal": {
        "path": "training/sage_t/t10_3_8_witness_gate_adjudication/terminal_report.json",
        "sha256": "b20da2dc357c3d1d8abf8432fff8f425a66f7d456dee611051864ecc64e68f68",
    },
}

SUPERSEDED_T10_3_8 = {
    "status": "SUPERSEDED_MIXED_SEQUENCE_MISS",
    "manifest_checksum": "1022e930fcb864bec21715ebc4a3b8049a122c05c9b754c259817043e8abdffc",
    "terminal_report_checksum": "a55fc77b4ca2296edc210c4c0e07eef6bcb8a4b9c227e17a21cdd4a18758abef",
    "verdict": "MIXED_SEQUENCE_MISS",
    "intent_count": 407,
    "event_count": 407,
    "branch_count": 18,
    "unresolved_count": 0,
    "sequence_action_count": 272,
    "sequence_branch_count": 6,
    "sequence_level_count": 0,
    "re86_controller_error_count": 2,
    "ls20_budget_exhaustion_count": 2,
    "sc25_terminal_count": 2,
    "used_for_training": False,
    "sequence_registry_used_as_prior": False,
    "mutated_by_t10_3_9": False,
    "physical_actions_replayed": 0,
}


def _signed(payload: Mapping[str, Any], checksum_field: str) -> dict[str, Any]:
    output = dict(payload)
    output[checksum_field] = sha256_payload(output)
    return output


def _parent_root(root: Path) -> Path:
    return root / "training" / "sage_t" / "t10_3_8_witness_gate_adjudication"


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
        {"relative_path": path.relative_to(base).as_posix(), "sha256": file_sha256(path)}
        for path in sorted(item for item in base.rglob("*.json") if item.is_file())
    ]
    return sha256_payload(rows)


def _parent_diagnosis(root: Path) -> dict[str, Any]:
    terminal = _read_signed(_parent_root(root) / "terminal_report.json", "report_checksum")
    sequence = _read_signed(
        _parent_root(root) / "discovery_sequence_report.json", "report_checksum"
    )
    receipts = [
        _read_signed(path, "receipt_checksum")
        for path in (_parent_root(root) / "journal" / "branches").rglob("receipt.json")
    ]
    sequence_rows = [row for row in receipts if row.get("phase") == "discover-sequence"]
    return {
        "manifest_checksum": str(terminal.get("manifest_checksum", "")),
        "terminal_report_checksum": str(terminal.get("report_checksum", "")),
        "verdict": str(terminal.get("verdict", "")),
        "intent_count": len(_journal_files(root, "intents")),
        "event_count": len(_journal_files(root, "events")),
        "branch_count": len(receipts),
        "unresolved_count": len(_journal_files(root, "unresolved")),
        "sequence_action_count": int(sequence.get("metrics", {}).get("actions", -1)),
        "sequence_branch_count": len(sequence_rows),
        "sequence_level_count": sum(
            int(row.get("level_delta", 0)) for row in sequence_rows
        ),
        "re86_controller_error_count": sum(
            bool(row.get("errors"))
            for row in sequence_rows
            if str(row.get("game_id", "")).startswith("re86-")
        ),
        "ls20_budget_exhaustion_count": sum(
            row.get("stop_reason") == "ACTION_BUDGET_EXHAUSTED"
            for row in sequence_rows
            if str(row.get("game_id", "")).startswith("ls20-")
        ),
        "sc25_terminal_count": sum(
            row.get("stop_reason") == "TERMINAL_STATE"
            for row in sequence_rows
            if str(row.get("game_id", "")).startswith("sc25-")
        ),
    }


def _verify_parent(root: Path) -> dict[str, Any]:
    parent.load_manifest(root)
    for name, binding in PARENT_ARTIFACTS.items():
        path = root / binding["path"]
        if not path.is_file() or file_sha256(path) != binding["sha256"]:
            raise IntegrityError(f"required T10.3.8 artifact absent or drifted: {name}")
    if (_parent_root(root) / "collector.lock.json").exists():
        raise IntegrityError("T10.3.8 collector lock must be absent")
    diagnosis = _parent_diagnosis(root)
    expected = {
        key: SUPERSEDED_T10_3_8[key]
        for key in diagnosis
    }
    if diagnosis != expected:
        raise IntegrityError("T10.3.8 terminal diagnosis diverged")
    return diagnosis


def _code_hashes(root: Path) -> dict[str, str]:
    relative = (
        "theory/sage_t/goal_directed_v10_3_9.py",
        "theory/sage_t/t10_3_9_protocol.py",
        "theory/sage_t/t10_3_9_runtime.py",
        "theory/sage_t/t10_3_5_runtime.py",
        "theory/sage_t/t10_3_6_runtime.py",
        "theory/sage_t/t10_3_8_runtime.py",
        "tests/test_sage_t_goal_directed_v10_3_9.py",
        "tests/test_sage_t_t10_3_9_protocol.py",
        "tests/test_sage_t_t10_3_9_runtime.py",
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
                    WorkSpec(phase, game, seed, "goal_directed_sage_t", index, SEQUENCE_ACTION_BUDGET)
                )
        return tuple(rows)
    if phase == "reproduce-sequence":
        for game in SEQUENCE_GAMES:
            for index, seed in enumerate(REPRODUCTION_SEEDS):
                rows.append(
                    WorkSpec(phase, game, seed, "goal_directed_sage_t", index, SEQUENCE_ACTION_BUDGET)
                )
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
        "objective": "repair_sequence_observation_and_induce_reset_local_causal_subgoals",
        "parent_artifacts": PARENT_ARTIFACTS,
        "superseded_t10_3_8": {
            **SUPERSEDED_T10_3_8,
            "journal_digest": _parent_journal_digest(root),
            "diagnosis": diagnosis,
        },
        "code_hashes": _code_hashes(root),
        "cli_phases": [
            "freeze", "status", "audit", "preflight", "discover-sequence",
            "reproduce-sequence", "compile", "confirm", "report",
        ],
        "exit_codes": {"success": 0, "integrity": 2, "scientific_gate": 3},
        "matrix": _matrix_payload(),
        "functional_contract": {
            "level_increment_is_only_success_credit": True,
            "effect_novelty_is_exploration_only": True,
            "effect_descriptor_total_for_extended_displacements": True,
            "seed_diversified_action_schema_order": True,
            "local_causal_effect_graph_ephemeral": True,
            "mixed_automata_induced_from_local_effects": True,
            "core_registry_prior_support_zero": True,
            "parent_sequence_events_diagnostic_only": True,
            "parent_sequence_registry_loaded": False,
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
        },
        "firewall": {
            "source_validation_opened": False,
            "ar25_opened": False,
            "holdout_opened": False,
            "production_authority": False,
            "automatic_retuning": False,
            "t10_3_8_sequence_events_training_authorized": False,
            "t10_3_8_sequence_registry_prior_authorized": False,
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
            "format_version": "sage-t10.3.9-causal-subgoal-migration-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "superseded_protocol": "SAGE.T10.3.8",
            "superseded_status": SUPERSEDED_T10_3_8["status"],
            "parent_journal_digest": manifest["superseded_t10_3_8"]["journal_digest"],
            "diagnosis": manifest["superseded_t10_3_8"]["diagnosis"],
            "parent_sequence_events_used_for_training": 0,
            "parent_sequence_registry_loaded": False,
            "parent_core_registry_loaded_as_structural_prior": True,
            "parent_core_registry_local_support": 0,
            "parent_artifacts_mutated": False,
            "parent_physical_actions_replayed": 0,
            "repairs": [
                "total_effect_descriptor",
                "seed_diversified_schema_probing",
                "reset_local_causal_effect_graph",
                "effect_frontier_mixed_automaton",
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
        raise IntegrityError("T10.3.9 manifest has not been frozen")
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_signed(payload, "manifest_checksum")
    if payload.get("format_version") != FORMAT_VERSION:
        raise IntegrityError("T10.3.9 manifest format drifted")
    _verify_parent(root)
    if payload.get("superseded_t10_3_8", {}).get("journal_digest") != _parent_journal_digest(root):
        raise IntegrityError("T10.3.8 journal changed after T10.3.9 freeze")
    if verify_code and payload.get("code_hashes") != _code_hashes(root):
        raise IntegrityError("T10.3.9 code hash drifted after freeze")
    migration_path = root / DEFAULT_MIGRATION_PATH
    if not migration_path.is_file():
        raise IntegrityError("T10.3.9 migration receipt is absent")
    migration = json.loads(migration_path.read_text(encoding="utf-8"))
    verify_signed(migration, "receipt_checksum")
    if migration.get("manifest_checksum") != payload.get("manifest_checksum"):
        raise IntegrityError("T10.3.9 migration receipt is detached")
    return payload


__all__ = [
    "ALL_SOURCE_GAMES", "CONFIRMATION_ARMS", "CONFIRMATION_SEEDS",
    "CORE_ACTION_BUDGET", "CORE_GAMES", "DEFAULT_MANIFEST_PATH",
    "DEFAULT_MIGRATION_PATH", "DEFAULT_OUTPUT_DIR", "DISCOVERY_SEEDS",
    "IntegrityError", "PARENT_ARTIFACTS", "REPRODUCTION_SEEDS",
    "SEQUENCE_ACTION_BUDGET", "SEQUENCE_GAMES", "ScientificGateMiss",
    "SUPERSEDED_T10_3_8", "TOTAL_MAXIMUM_ACTIONS", "TOTAL_RESETS",
    "WorkSpec", "build_manifest", "file_sha256", "freeze_manifest",
    "load_manifest", "maximum_actions_for_phase", "maximum_actions_for_specs",
    "reset_wall_seconds", "sha256_payload", "verify_signed", "work_specs",
    "write_json_once",
]
