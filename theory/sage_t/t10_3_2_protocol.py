"""Frozen source-only protocol for SAGE.T10.3.2 end-to-end convergence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FORMAT_VERSION = "sage-t10.3.2-end-to-end-protocol-v1"
MANIFEST_STATUS = "FROZEN_BEFORE_T10_3_2_SOURCE_ACTION"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name("sage_t10_3_2_protocol_manifest.json")
DEFAULT_MIGRATION_PATH = Path(__file__).with_name("sage_t10_3_2_migration_receipt.json")
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "t10_3_2_end_to_end_convergence"

CORE_GAMES = ("lp85-305b61c3", "su15-4c352900")
SEQUENCE_GAMES = ("re86-4e57566e", "ls20-9607627b", "sc25-f9b21a2f")
ALL_SOURCE_GAMES = (*CORE_GAMES, *SEQUENCE_GAMES)
DISCOVERY_SEEDS = (3141, 3142)
CONFIRMATION_SEEDS = (3151, 3152)
CONFIRMATION_ARMS = ("goal_directed_sage_t", "unified_sage_t_off")
CORE_ACTION_BUDGET = 128
SEQUENCE_ACTION_BUDGET = 256
MAXIMUM_OPTION_HORIZON = 32
DISCOVERY_RESETS = 10
CONFIRMATION_RESETS = 20
TOTAL_RESETS = 30
CORE_DISCOVERY_ACTIONS = 512
SEQUENCE_DISCOVERY_ACTIONS = 1536
DISCOVERY_ACTIONS = CORE_DISCOVERY_ACTIONS + SEQUENCE_DISCOVERY_ACTIONS
CONFIRMATION_ACTIONS = 4096
TOTAL_MAXIMUM_ACTIONS = 6144

PARENT_ARTIFACTS = {
    "t9_4_active_report": {
        "path": "training/sage_t/active_v9_4/report.json",
        "sha256": "0c19e7ded9d648d02b4ee8d24391246aa90741a72ce3ca255ffe491bccace9e7",
    },
    "t9_5_source_validation_report": {
        "path": "training/sage_t/active_v9_5/report.json",
        "sha256": "750def40446babee5433a2e8a4d7ca01b1260444c5ba0cc1bfefa3a2510a956e",
    },
    "t9_6_abstention_report": {
        "path": "training/sage_t/productive_abstention_v9_6/report.json",
        "sha256": "77ee5cfb2c4137d23001e3aef05ce9b1fe39a106d8424d2e54a827910baab2c8",
    },
    "t10_0b_progress_report": {
        "path": "training/sage_t/progress_witness_v10_0b/report.json",
        "sha256": "998b8af56b9933383bfe9b37aa4e245cffad3fb635d0abc8a19c5817724c044f",
    },
    "t10_3_1_manifest": {
        "path": "theory/sage_t/sage_t10_3_1_protocol_manifest.json",
        "sha256": "99517d6c7d12564b0b5af2349b5fcb7e1af14b9af875f5d2ca316bd5bd42e6e2",
    },
    "t10_3_1_offline_audit": {
        "path": "training/sage_t/t10_3_1_goal_progress_correction/offline_audit.json",
        "sha256": "59675b64fa3435841824bdd43595efc463d3786bf02be857acc1b43d5794917c",
    },
}

SUPERSEDED_T10_3_1 = {
    "status": "SUPERSEDED_PARTIAL",
    "intent_count": 95,
    "event_count": 94,
    "unresolved_count": 0,
    "branch_count": 7,
    "interrupted_intent_count": 1,
    "used_for_training": False,
    "mutated_by_t10_3_2": False,
    "physical_actions_replayed": 0,
}


class IntegrityError(RuntimeError):
    """Raised when frozen provenance or write-once state diverges."""


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


def _count_parent_journal(root: Path) -> dict[str, int]:
    base = root / "training" / "sage_t" / "t10_3_1_goal_progress_correction" / "journal"

    def count(category: str) -> int:
        directory = base / category
        return len(tuple(directory.rglob("*.json"))) if directory.exists() else 0

    intents = count("intents")
    events = count("events")
    unresolved = count("unresolved")
    branches = count("branches")
    return {
        "intent_count": intents,
        "event_count": events,
        "unresolved_count": unresolved,
        "branch_count": branches,
        "interrupted_intent_count": intents - events - unresolved,
    }


def _parent_journal_digest(root: Path) -> str:
    base = root / "training" / "sage_t" / "t10_3_1_goal_progress_correction" / "journal"
    rows = []
    if base.exists():
        for path in sorted(item for item in base.rglob("*.json") if item.is_file()):
            rows.append(
                {
                    "relative_path": path.relative_to(base).as_posix(),
                    "sha256": file_sha256(path),
                }
            )
    return sha256_payload(rows)


def _verify_parent_artifacts(root: Path) -> None:
    for name, binding in PARENT_ARTIFACTS.items():
        path = root / str(binding["path"])
        if not path.is_file():
            raise IntegrityError(f"required parent artifact is absent: {name}")
        if file_sha256(path) != str(binding["sha256"]):
            raise IntegrityError(f"required parent artifact drifted: {name}")
    observed = _count_parent_journal(root)
    expected = {
        key: int(SUPERSEDED_T10_3_1[key])
        for key in observed
    }
    if observed != expected:
        raise IntegrityError("T10.3.1 partial journal snapshot diverged")


def _code_hashes(root: Path) -> dict[str, str]:
    relative = (
        "theory/sage_t/goal_directed_v10_3_2.py",
        "theory/sage_t/t10_3_2_protocol.py",
        "theory/sage_t/t10_3_2_runtime.py",
        "theory/unified_cognitive_controller.py",
        "theory/unified_cognition_ab_benchmark.py",
        "theory/sage_t/controller.py",
        "theory/sage_t/progress_witness_v10.py",
        "tests/test_sage_t_goal_directed_v10_3_2.py",
        "tests/test_sage_t_t10_3_2_protocol.py",
        "tests/test_sage_t_t10_3_2_runtime.py",
    )
    result = {}
    for item in relative:
        path = root / item
        if not path.is_file():
            raise IntegrityError(f"protocol code dependency is absent: {item}")
        result[item] = file_sha256(path)
    return result


def _matrix_payload() -> dict[str, Any]:
    return {
        "discovery": {
            "core": {
                "games": list(CORE_GAMES),
                "seeds": list(DISCOVERY_SEEDS),
                "resets_per_condition": 1,
                "actions_per_reset": CORE_ACTION_BUDGET,
                "maximum_actions": CORE_DISCOVERY_ACTIONS,
            },
            "sequence": {
                "games": list(SEQUENCE_GAMES),
                "seeds": list(DISCOVERY_SEEDS),
                "resets_per_condition": 1,
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
    _verify_parent_artifacts(root)
    core = {
        "format_version": FORMAT_VERSION,
        "status": MANIFEST_STATUS,
        "objective": "closed_loop_exploration_to_level_progress",
        "parent_artifacts": PARENT_ARTIFACTS,
        "superseded_t10_3_1": {
            **SUPERSEDED_T10_3_1,
            "journal_digest": _parent_journal_digest(root),
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
        "controller": {
            "integration": "UnifiedCognitiveController.sage_t_controller",
            "authority": "source_experimental_active",
            "baseline": "UnifiedCognitiveController_sage_t_off",
            "fresh_controller_per_reset": True,
            "transferred_support": 0,
            "discovery_warmup_actions": 32,
            "exploration_actions_between_options": 8,
            "maximum_option_horizon": MAXIMUM_OPTION_HORIZON,
            "maximum_sterile_option_actions": 4,
            "shortlist_horizon": 8,
            "shortlist_size": 8,
            "maximum_decision_p95_ms": 2500.0,
        },
        "persistent_program_forbidden_fields": [
            "game_id",
            "seed",
            "coordinates",
            "colors",
            "raw_grid",
            "entity_id",
            "object_id",
        ],
        "gates": {
            "preflight_environments": [
                "repeat_target",
                "path_length_10",
                "mixed_mode_switch_beyond_16",
            ],
            "core_progress_each_game_and_seed": True,
            "sequence_progress_minimum_games": 1,
            "sequence_minimum_distinct_action_schemas": 2,
            "registry_independent_reproduction_count": 2,
            "registry_controls": [
                "binding_swap",
                "order_permutation",
                "automaton_ablation",
            ],
            "confirmation_total_level_advantage": 1,
            "zero_illegal_actions": True,
            "zero_controller_errors": True,
            "game_over_nonincrease": True,
        },
        "firewall": {
            "source_validation_opened": False,
            "ar25_opened": False,
            "holdout_opened": False,
            "production_authority": False,
            "automatic_retuning": False,
        },
        "durability": {
            "intent_before_action": True,
            "event_immediate_seal": True,
            "physical_replay": False,
            "single_live_inflight_intent": True,
            "lock_fields": ["pid", "process_start", "nonce", "heartbeat"],
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
            "format_version": "sage-t10.3.2-supersession-receipt-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "superseded_protocol": "SAGE.T10.3.1",
            "superseded_status": "SUPERSEDED_PARTIAL",
            "parent_journal_digest": manifest["superseded_t10_3_1"][
                "journal_digest"
            ],
            "parent_data_used_for_training": False,
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
        raise IntegrityError("T10.3.2 manifest has not been frozen")
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_signed(payload, "manifest_checksum")
    if payload.get("format_version") != FORMAT_VERSION:
        raise IntegrityError("T10.3.2 manifest format drifted")
    _verify_parent_artifacts(root)
    if payload.get("superseded_t10_3_1", {}).get(
        "journal_digest"
    ) != _parent_journal_digest(root):
        raise IntegrityError("T10.3.1 parent journal changed after supersession")
    if verify_code and payload.get("code_hashes") != _code_hashes(root):
        raise IntegrityError("T10.3.2 code hash drifted after freeze")
    migration = json.loads((root / DEFAULT_MIGRATION_PATH).read_text(encoding="utf-8"))
    verify_signed(migration, "receipt_checksum")
    if migration.get("manifest_checksum") != payload.get("manifest_checksum"):
        raise IntegrityError("T10.3.2 migration receipt is detached")
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
