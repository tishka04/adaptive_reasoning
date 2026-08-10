"""Frozen bounded-compute continuation protocol for SAGE.T10.3.4."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import t10_3_3_protocol as parent

FORMAT_VERSION = "sage-t10.3.4-bounded-compute-recovery-protocol-v1"
MANIFEST_STATUS = "FROZEN_BEFORE_T10_3_4_SOURCE_ACTION"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name("sage_t10_3_4_protocol_manifest.json")
DEFAULT_MIGRATION_PATH = Path(__file__).with_name("sage_t10_3_4_migration_receipt.json")
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "t10_3_4_bounded_compute_recovery"

CORE_GAMES = parent.CORE_GAMES
SEQUENCE_GAMES = parent.SEQUENCE_GAMES
ALL_SOURCE_GAMES = (*CORE_GAMES, *SEQUENCE_GAMES)
DISCOVERY_SEEDS = (3181, 3182)
CONFIRMATION_SEEDS = (3191, 3192)
CONFIRMATION_ARMS = parent.CONFIRMATION_ARMS
CORE_ACTION_BUDGET = parent.CORE_ACTION_BUDGET
SEQUENCE_ACTION_BUDGET = parent.SEQUENCE_ACTION_BUDGET
CORE_DISCOVERY_ACTIONS = 512
SEQUENCE_DISCOVERY_ACTIONS = 1536
CONFIRMATION_ACTIONS = 4096
TOTAL_RESETS = 30
TOTAL_MAXIMUM_ACTIONS = 6144
CORE_RESET_WALL_SECONDS = 900.0
SEQUENCE_RESET_WALL_SECONDS = 1800.0
MAXIMUM_DECISION_P95_MS = 2500.0

PARENT_ARTIFACTS = {
    "t10_3_3_manifest": {
        "path": "theory/sage_t/sage_t10_3_3_protocol_manifest.json",
        "sha256": "f5d2b283900327bdf954cb5e58d54e01dd0fbe1f6c6be6d1eba46ff6f3f883ef",
    },
    "t10_3_3_migration": {
        "path": "theory/sage_t/sage_t10_3_3_migration_receipt.json",
        "sha256": "d66f46e854493685ed7ca07bbd6553c7010a96dea09514be589502238c610118",
    },
    "t10_3_3_audit": {
        "path": "training/sage_t/t10_3_3_relational_binding_recovery/offline_audit.json",
        "sha256": "dd6130c54a234cb6a4cd82cd5301ed4e1746ea144c4312998c3f9787d9721808",
    },
    "t10_3_3_preflight": {
        "path": "training/sage_t/t10_3_3_relational_binding_recovery/synthetic_preflight.json",
        "sha256": "6a3582e84eab3dec75d7de92c8894587cdae801f20fb1b6f463e4960b624a891",
    },
    "t10_3_3_interrupted_checkpoint": {
        "path": "training/sage_t/t10_3_3_relational_binding_recovery/checkpoint.json",
        "sha256": "8eddcbc74fe9941d35e652755c580543f668203d2dae700225234ba08962712a",
    },
}

SUPERSEDED_T10_3_3 = {
    "status": "SUPERSEDED_PARTIAL_POSITIVE_COMPUTE_GROWTH",
    "intent_count": 76,
    "event_count": 76,
    "unresolved_count": 0,
    "branch_count": 0,
    "interrupted_work_count": 1,
    "action6_count": 76,
    "unchanged_frame_count": 64,
    "level_delta": 1,
    "sage_t_decision_count": 3,
    "candidate_count": 3,
    "candidate_success_count": 1,
    "candidate_contradiction_count": 2,
    "winning_step_index": 51,
    "winning_decision_source": "sage_t_joint_program",
    "winning_event_checksum": (
        "ccd3524d116e70427a6d18c8855ffbb2ca3362094a35a5c5e7f4a923acfee6a2"
    ),
    "used_for_training": False,
    "positive_witness_imported_as_prior": False,
    "mutated_by_t10_3_4": False,
    "physical_actions_replayed": 0,
}


class IntegrityError(RuntimeError):
    """Raised when frozen provenance or durable state diverges."""


class ScientificGateMiss(RuntimeError):
    """Raised when a preregistered scientific gate is negative."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signed(payload: Mapping[str, Any], checksum_field: str) -> dict[str, Any]:
    result = dict(payload)
    result[checksum_field] = sha256_payload(result)
    return result


def verify_signed(payload: Mapping[str, Any], checksum_field: str) -> None:
    expected = str(payload.get(checksum_field, ""))
    core = {key: value for key, value in payload.items() if key != checksum_field}
    if not expected or sha256_payload(core) != expected:
        raise IntegrityError(f"{checksum_field} mismatch")


def write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = _canonical(payload) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
    except FileExistsError:
        if path.read_text(encoding="utf-8") != encoded:
            raise IntegrityError(f"write-once artifact diverged: {path}") from None


def _parent_root(root: Path) -> Path:
    return root / "training" / "sage_t" / "t10_3_3_relational_binding_recovery"


def _parent_journal(root: Path) -> Path:
    return _parent_root(root) / "journal"


def _journal_files(root: Path, category: str) -> tuple[Path, ...]:
    directory = _parent_journal(root) / category
    return tuple(sorted(directory.rglob("*.json"))) if directory.exists() else ()


def _parent_journal_digest(root: Path) -> str:
    base = _parent_journal(root)
    rows = [
        {
            "relative_path": path.relative_to(base).as_posix(),
            "sha256": file_sha256(path),
        }
        for path in sorted(item for item in base.rglob("*.json") if item.is_file())
    ]
    return sha256_payload(rows)


def _load_rows(root: Path, category: str, checksum_field: str) -> list[dict[str, Any]]:
    rows = []
    for path in _journal_files(root, category):
        payload = json.loads(path.read_text(encoding="utf-8"))
        parent.verify_signed(payload, checksum_field)
        rows.append(payload)
    return rows


def _parent_diagnosis(root: Path) -> dict[str, Any]:
    intents = _load_rows(root, "intents", "intent_checksum")
    events = _load_rows(root, "events", "event_checksum")
    unresolved = _journal_files(root, "unresolved")
    branches = _journal_files(root, "branches")
    checkpoint_path = _parent_root(root) / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    parent.verify_signed(checkpoint, "checkpoint_checksum")
    programs = checkpoint.get("controller_registry", {}).get("programs", ())
    sources = Counter(str(row.get("decision_source", "")) for row in intents)
    level_events = [row for row in events if int(row.get("level_delta", 0)) > 0]
    winning = level_events[0] if len(level_events) == 1 else {}
    return {
        "intent_count": len(intents),
        "event_count": len(events),
        "unresolved_count": len(unresolved),
        "branch_count": len(branches),
        "interrupted_work_count": int(bool(intents) and not branches),
        "action6_count": sum(
            str(row.get("action", {}).get("name", "")) == "ACTION6" for row in intents
        ),
        "unchanged_frame_count": sum(
            row.get("frame_before_sha256") == row.get("frame_after_sha256")
            for row in events
        ),
        "level_delta": sum(int(row.get("level_delta", 0)) for row in events),
        "sage_t_decision_count": int(sources["sage_t_joint_program"]),
        "candidate_count": len(programs),
        "candidate_success_count": sum(
            len(row.get("success_attestations", ())) for row in programs
        ),
        "candidate_contradiction_count": sum(
            len(row.get("contradiction_attestations", ())) for row in programs
        ),
        "winning_step_index": int(winning.get("step_index", -1)),
        "winning_decision_source": str(winning.get("decision_source", "")),
        "winning_event_checksum": str(winning.get("event_checksum", "")),
        "decision_sources": dict(sorted(sources.items())),
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
        raise IntegrityError(
            "T10.3.3 collector lock must be absent before supersession"
        )
    diagnosis = _parent_diagnosis(root)
    expected = {
        key: SUPERSEDED_T10_3_3[key] for key in diagnosis if key != "decision_sources"
    }
    observed = {
        key: value for key, value in diagnosis.items() if key != "decision_sources"
    }
    if observed != expected:
        raise IntegrityError("T10.3.3 interrupted snapshot diverged")
    return diagnosis


def _code_hashes(root: Path) -> dict[str, str]:
    relative = (
        "theory/sage_t/goal_directed_v10_3_4.py",
        "theory/sage_t/t10_3_4_protocol.py",
        "theory/sage_t/t10_3_4_runtime.py",
        "theory/sage_t/goal_directed_v10_3_3.py",
        "theory/sage_t/t10_3_3_runtime.py",
        "theory/sage_t/goal_directed_v10_3_2.py",
        "theory/sage_t/t10_3_2_runtime.py",
        "theory/unified_cognitive_controller.py",
        "theory/unified_cognition_ab_benchmark.py",
        "theory/sage_t/frame_adapters_v10_3.py",
        "theory/sage_t/progress_witness_v10.py",
        "tests/test_sage_t_goal_directed_v10_3_4.py",
        "tests/test_sage_t_t10_3_4_protocol.py",
        "tests/test_sage_t_t10_3_4_runtime.py",
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
        "objective": "preserve_end_to_end_progress_under_bounded_compute",
        "parent_artifacts": PARENT_ARTIFACTS,
        "superseded_t10_3_3": {
            **SUPERSEDED_T10_3_3,
            "journal_digest": _parent_journal_digest(root),
            "decision_sources": diagnosis["decision_sources"],
        },
        "code_hashes": _code_hashes(root),
        "cli_phases": [
            "freeze",
            "status",
            "audit",
            "preflight",
            "discover-core",
            "discover-sequence",
            "compile",
            "confirm",
            "report",
        ],
        "exit_codes": {"success": 0, "integrity": 2, "scientific_gate": 3},
        "matrix": _matrix_payload(),
        "bounded_compute": {
            "stop_after_first_sealed_level": True,
            "same_profile_for_active_and_baseline": True,
            "active_option_fast_path": True,
            "discovery_warmup_actions": 8,
            "exploration_actions_between_options": 8,
            "transition_history_limit": 32,
            "operator_induction_interval": 8,
            "operator_planning_enabled": False,
            "long_horizon_growth_modules_enabled": False,
            "core_reset_wall_seconds": CORE_RESET_WALL_SECONDS,
            "sequence_reset_wall_seconds": SEQUENCE_RESET_WALL_SECONDS,
            "stage_timing_required": True,
        },
        "gates": {
            "parent_positive_witness_attested_but_not_reused": True,
            "preflight_ambiguous_parameterized_targets": True,
            "preflight_active_option_fast_path": True,
            "preflight_transition_history_bound": True,
            "preflight_early_success_stop": True,
            "core_progress_each_game_and_seed": True,
            "winning_decision_from_sage_t": True,
            "minimum_sage_t_physical_action_per_core_reset": 1,
            "sequence_progress_minimum_games": 1,
            "sequence_minimum_distinct_action_schemas": 2,
            "registry_independent_reproduction_count": 2,
            "confirmation_total_level_advantage": 1,
            "maximum_decision_p95_ms": MAXIMUM_DECISION_P95_MS,
        },
        "firewall": {
            "source_validation_opened": False,
            "ar25_opened": False,
            "holdout_opened": False,
            "production_authority": False,
            "automatic_retuning": False,
            "t10_3_3_events_training_authorized": False,
            "t10_3_3_positive_witness_prior_authorized": False,
            "t10_3_3_physical_replay_authorized": False,
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
            "format_version": "sage-t10.3.4-bounded-compute-migration-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "superseded_protocol": "SAGE.T10.3.3",
            "superseded_status": SUPERSEDED_T10_3_3["status"],
            "parent_journal_digest": manifest["superseded_t10_3_3"]["journal_digest"],
            "diagnosis": {
                key: manifest["superseded_t10_3_3"][key]
                for key in (
                    "intent_count",
                    "event_count",
                    "unchanged_frame_count",
                    "level_delta",
                    "sage_t_decision_count",
                    "winning_step_index",
                    "winning_decision_source",
                    "winning_event_checksum",
                    "candidate_count",
                    "candidate_success_count",
                    "candidate_contradiction_count",
                )
            },
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
        raise IntegrityError("T10.3.4 manifest has not been frozen")
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_signed(payload, "manifest_checksum")
    if payload.get("format_version") != FORMAT_VERSION:
        raise IntegrityError("T10.3.4 manifest format drifted")
    _verify_parent(root)
    if payload.get("superseded_t10_3_3", {}).get(
        "journal_digest"
    ) != _parent_journal_digest(root):
        raise IntegrityError("T10.3.3 journal changed after supersession")
    if verify_code and payload.get("code_hashes") != _code_hashes(root):
        raise IntegrityError("T10.3.4 code hash drifted after freeze")
    migration_path = root / DEFAULT_MIGRATION_PATH
    if not migration_path.is_file():
        raise IntegrityError("T10.3.4 migration receipt is absent")
    migration = json.loads(migration_path.read_text(encoding="utf-8"))
    verify_signed(migration, "receipt_checksum")
    if migration.get("manifest_checksum") != payload.get("manifest_checksum"):
        raise IntegrityError("T10.3.4 migration receipt is detached")
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
        budget = (
            CORE_ACTION_BUDGET if phase == "discover-core" else SEQUENCE_ACTION_BUDGET
        )
        for game in games:
            for seed in DISCOVERY_SEEDS:
                rows.append(
                    WorkSpec(phase, game, seed, "goal_directed_sage_t", 0, budget)
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
    return (
        CORE_RESET_WALL_SECONDS
        if work.game_id in CORE_GAMES
        else SEQUENCE_RESET_WALL_SECONDS
    )


__all__ = [
    "ALL_SOURCE_GAMES",
    "CONFIRMATION_ARMS",
    "CONFIRMATION_SEEDS",
    "CORE_GAMES",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_MIGRATION_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DISCOVERY_SEEDS",
    "MAXIMUM_DECISION_P95_MS",
    "SEQUENCE_GAMES",
    "SUPERSEDED_T10_3_3",
    "TOTAL_MAXIMUM_ACTIONS",
    "TOTAL_RESETS",
    "IntegrityError",
    "ScientificGateMiss",
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
