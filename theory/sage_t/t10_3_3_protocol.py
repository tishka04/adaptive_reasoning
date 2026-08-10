"""Frozen continuation protocol for T10.3.3 relational binding recovery."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import t10_3_2_protocol as parent

FORMAT_VERSION = "sage-t10.3.3-relational-binding-recovery-protocol-v1"
MANIFEST_STATUS = "FROZEN_BEFORE_T10_3_3_SOURCE_ACTION"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name("sage_t10_3_3_protocol_manifest.json")
DEFAULT_MIGRATION_PATH = Path(__file__).with_name("sage_t10_3_3_migration_receipt.json")
DEFAULT_OUTPUT_DIR = (
    Path("training") / "sage_t" / "t10_3_3_relational_binding_recovery"
)

CORE_GAMES = parent.CORE_GAMES
SEQUENCE_GAMES = parent.SEQUENCE_GAMES
ALL_SOURCE_GAMES = (*CORE_GAMES, *SEQUENCE_GAMES)
DISCOVERY_SEEDS = (3161, 3162)
CONFIRMATION_SEEDS = (3171, 3172)
CONFIRMATION_ARMS = parent.CONFIRMATION_ARMS
CORE_ACTION_BUDGET = parent.CORE_ACTION_BUDGET
SEQUENCE_ACTION_BUDGET = parent.SEQUENCE_ACTION_BUDGET
CORE_DISCOVERY_ACTIONS = 512
SEQUENCE_DISCOVERY_ACTIONS = 1536
CONFIRMATION_ACTIONS = 4096
TOTAL_RESETS = 30
TOTAL_MAXIMUM_ACTIONS = 6144

PARENT_ARTIFACTS = {
    "t10_3_2_manifest": {
        "path": "theory/sage_t/sage_t10_3_2_protocol_manifest.json",
        "sha256": "54f37c92fbe6fa69c0c674e20b56d45cd23b603f5c231f7f0709c51191321f72",
    },
    "t10_3_2_migration": {
        "path": "theory/sage_t/sage_t10_3_2_migration_receipt.json",
        "sha256": "e61b5c4947d6fb48a1bc3dbe1ac63cf193d5e886f4420154994ad33fd25835be",
    },
    "t10_3_2_audit": {
        "path": "training/sage_t/t10_3_2_end_to_end_convergence/offline_audit.json",
        "sha256": "51551b725bbff153f6a4a00672b5f51665e7cf3abcda805e5546e093645329f5",
    },
    "t10_3_2_preflight": {
        "path": "training/sage_t/t10_3_2_end_to_end_convergence/synthetic_preflight.json",
        "sha256": "117428c92d827d8376b63796b461167c9d1e8559d90935c86ed7ac55d01ac7bd",
    },
    "t10_3_2_interrupted_checkpoint": {
        "path": "training/sage_t/t10_3_2_end_to_end_convergence/checkpoint.json",
        "sha256": "bda6d26756a8343e4105bbe96a428c8322b2d1597d5402c31de81b74e7aaea79",
    },
}

SUPERSEDED_T10_3_2 = {
    "status": "SUPERSEDED_PARTIAL_BINDING_AMBIGUITY",
    "intent_count": 99,
    "event_count": 99,
    "unresolved_count": 0,
    "branch_count": 0,
    "interrupted_work_count": 1,
    "action6_count": 99,
    "unchanged_frame_count": 91,
    "level_delta": 0,
    "sage_t_decision_count": 0,
    "candidate_count": 8,
    "candidate_success_count": 0,
    "candidate_contradiction_count": 8,
    "used_for_training": False,
    "mutated_by_t10_3_3": False,
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


def _parent_journal(root: Path) -> Path:
    return (
        root
        / "training"
        / "sage_t"
        / "t10_3_2_end_to_end_convergence"
        / "journal"
    )


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
    checkpoint_path = (
        root / "training" / "sage_t" / "t10_3_2_end_to_end_convergence" / "checkpoint.json"
    )
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    parent.verify_signed(checkpoint, "checkpoint_checksum")
    programs = checkpoint.get("controller_registry", {}).get("programs", ())
    sources = Counter(str(row.get("decision_source", "")) for row in intents)
    return {
        "intent_count": len(intents),
        "event_count": len(events),
        "unresolved_count": len(unresolved),
        "branch_count": len(branches),
        "interrupted_work_count": int(bool(intents) and not branches),
        "action6_count": sum(
            str(row.get("action", {}).get("name", "")) == "ACTION6"
            for row in intents
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
    lock = (
        root
        / "training"
        / "sage_t"
        / "t10_3_2_end_to_end_convergence"
        / "collector.lock.json"
    )
    if lock.exists():
        raise IntegrityError("T10.3.2 collector lock must be absent before supersession")
    diagnosis = _parent_diagnosis(root)
    expected = {
        key: SUPERSEDED_T10_3_2[key]
        for key in diagnosis
        if key != "decision_sources"
    }
    observed = {key: value for key, value in diagnosis.items() if key != "decision_sources"}
    if observed != expected:
        raise IntegrityError("T10.3.2 interrupted snapshot diverged")
    return diagnosis


def _code_hashes(root: Path) -> dict[str, str]:
    relative = (
        "theory/sage_t/goal_directed_v10_3_3.py",
        "theory/sage_t/t10_3_3_protocol.py",
        "theory/sage_t/t10_3_3_runtime.py",
        "theory/sage_t/goal_directed_v10_3_2.py",
        "theory/sage_t/t10_3_2_runtime.py",
        "theory/unified_cognitive_controller.py",
        "theory/unified_cognition_ab_benchmark.py",
        "theory/sage_t/frame_adapters_v10_3.py",
        "theory/sage_t/progress_witness_v10.py",
        "tests/test_sage_t_goal_directed_v10_3_3.py",
        "tests/test_sage_t_t10_3_3_protocol.py",
        "tests/test_sage_t_t10_3_3_runtime.py",
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
        "objective": "recover_relational_binding_then_end_to_end_progress",
        "parent_artifacts": PARENT_ARTIFACTS,
        "superseded_t10_3_2": {
            **SUPERSEDED_T10_3_2,
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
        "binding_recovery": {
            "persistent_coordinates": False,
            "persistent_entity_identifiers": False,
            "branch_local_productive_anchor": True,
            "dynamic_successor_replanning": True,
            "unique_structural_binding_requires_unique_candidate": True,
            "proposal_reacquisition_before_local_support": True,
            "explicit_rejection_reasons": True,
            "protected_route_sterile_limit": 4,
            "symmetric_danger_veto_for_baseline_identical_action": True,
        },
        "gates": {
            "preflight_ambiguous_parameterized_targets": True,
            "preflight_path_length_10": True,
            "preflight_mixed_beyond_16": True,
            "core_progress_each_game_and_seed": True,
            "winning_decision_from_sage_t": True,
            "minimum_sage_t_physical_action_per_core_reset": 1,
            "sequence_progress_minimum_games": 1,
            "sequence_minimum_distinct_action_schemas": 2,
            "registry_independent_reproduction_count": 2,
            "confirmation_total_level_advantage": 1,
            "maximum_decision_p95_ms": 2500.0,
        },
        "firewall": {
            "source_validation_opened": False,
            "ar25_opened": False,
            "holdout_opened": False,
            "production_authority": False,
            "automatic_retuning": False,
            "t10_3_2_events_training_authorized": False,
            "t10_3_2_physical_replay_authorized": False,
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
    target = manifest_path or (root / DEFAULT_MANIFEST_PATH)
    migration_target = migration_path or (root / DEFAULT_MIGRATION_PATH)
    write_json_once(target, manifest)
    migration = _signed(
        {
            "format_version": "sage-t10.3.3-binding-recovery-migration-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "superseded_protocol": "SAGE.T10.3.2",
            "superseded_status": "SUPERSEDED_PARTIAL_BINDING_AMBIGUITY",
            "parent_journal_digest": manifest["superseded_t10_3_2"][
                "journal_digest"
            ],
            "diagnosis": {
                key: manifest["superseded_t10_3_2"][key]
                for key in (
                    "intent_count",
                    "event_count",
                    "action6_count",
                    "unchanged_frame_count",
                    "level_delta",
                    "sage_t_decision_count",
                    "candidate_count",
                    "candidate_success_count",
                    "candidate_contradiction_count",
                )
            },
            "parent_events_used_for_training": 0,
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
        raise IntegrityError("T10.3.3 manifest has not been frozen")
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_signed(payload, "manifest_checksum")
    if payload.get("format_version") != FORMAT_VERSION:
        raise IntegrityError("T10.3.3 manifest format drifted")
    _verify_parent(root)
    if payload.get("superseded_t10_3_2", {}).get(
        "journal_digest"
    ) != _parent_journal_digest(root):
        raise IntegrityError("T10.3.2 journal changed after supersession")
    if verify_code and payload.get("code_hashes") != _code_hashes(root):
        raise IntegrityError("T10.3.3 code hash drifted after freeze")
    migration_path = root / DEFAULT_MIGRATION_PATH
    if not migration_path.is_file():
        raise IntegrityError("T10.3.3 migration receipt is absent")
    migration = json.loads(migration_path.read_text(encoding="utf-8"))
    verify_signed(migration, "receipt_checksum")
    if migration.get("manifest_checksum") != payload.get("manifest_checksum"):
        raise IntegrityError("T10.3.3 migration receipt is detached")
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


__all__ = [
    "ALL_SOURCE_GAMES",
    "CONFIRMATION_ARMS",
    "CONFIRMATION_SEEDS",
    "CORE_GAMES",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_MIGRATION_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DISCOVERY_SEEDS",
    "SEQUENCE_GAMES",
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
    "sha256_payload",
    "verify_signed",
    "work_specs",
    "write_json_once",
]
