"""Preregistered closed-loop relational successor diagnostic for T10.3.12e."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import t10_3_12d_protocol as parent
from .closed_loop_successor_v10_3_12e import ARMS, sha256_payload

FORMAT_VERSION = "sage-t10.3.12e-closed-loop-successor-protocol-v1"
MANIFEST_STATUS = "FROZEN_BEFORE_T10_3_12E_DIAGNOSTIC_RESET"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name("sage_t10_3_12e_protocol_manifest.json")
DEFAULT_FREEZE_RECEIPT_PATH = Path(__file__).with_name("sage_t10_3_12e_freeze_receipt.json")
DEFAULT_OUTPUT_DIR = Path("training/sage_t/t10_3_12e_closed_loop_successor")

PARENT_OUTPUT_DIR = Path("training/sage_t/t10_3_12d_executor_correspondence")
PARENT_ARTIFACT_PATHS = {
    "manifest": Path("theory/sage_t/sage_t10_3_12d_protocol_manifest.json"),
    "freeze_receipt": Path("theory/sage_t/sage_t10_3_12d_freeze_receipt.json"),
    "registry": PARENT_OUTPUT_DIR / "executor_registry.json",
    "active": PARENT_OUTPUT_DIR / "active_executor_diagnostic.json",
    "adjudication": PARENT_OUTPUT_DIR / "executor_adjudication_report.json",
    "terminal": PARENT_OUTPUT_DIR / "terminal_report.json",
}
SOURCE_FACTOR_REGISTRY_PATH = (
    Path("training/sage_t/t10_3_12c_cross_game_factor_falsification")
    / "cross_game_factor_registry.json"
)
EXPECTED_PARENT = {
    "manifest_checksum": "9de2e6d2dce4131c9c3cc1fd1e6d1800476703af45c359584c2a2be20681167f",
    "registry_checksum": "6fcc91fc061ae1c12aada2b963705d8fa395af0d5e13e4b912ed7ffa6bcb8863",
    "active_report_checksum": "2793f87020c71bffa91061190b9546a2c2ce30673163c7e532ff6d36c16fc63a",
    "adjudication_report_checksum": "1cdab90e05690de0ae92cc69b6281284ec311033cdf16e9dba427b7cd1b256d2",
    "terminal_report_checksum": "8cfcdba345d74ae2c4d92bf50ec1ee5a95aba39a021e75f15193105707a136a5",
    "verdict": "STABLE_EXECUTOR_NO_PROGRESS",
    "authorized_actions": 91,
    "sealed_events": 91,
    "inflight_intents": 0,
    "unresolved_intents": 0,
    "receipt_count": 36,
    "distinct_target_initial_frames": 9,
}

TARGET_GAMES = parent.TARGET_GAMES
ACTION_BUDGET = 16
RESET_WALL_SECONDS = 180.0
TOTAL_RESETS = len(TARGET_GAMES) * len(ARMS)
TOTAL_MAXIMUM_ACTIONS = TOTAL_RESETS * ACTION_BUDGET

IntegrityError = parent.IntegrityError
ScientificGateMiss = parent.ScientificGateMiss
file_sha256 = parent.file_sha256
write_json_once = parent.write_json_once

ARTIFACT_CONTRACT = {
    "audit-parent": {
        "path": "parent_stable_executor_negative_audit.json",
        "checksum_field": "audit_checksum",
        "gate_field": "passed",
    },
    "audit-trajectories": {
        "path": "parent_closed_loop_motivation_audit.json",
        "checksum_field": "audit_checksum",
        "gate_field": "passed",
    },
    "preflight": {
        "path": "closed_loop_preflight.json",
        "checksum_field": "preflight_checksum",
        "gate_field": "passed",
    },
    "compile-programs": {
        "path": "closed_loop_registry.json",
        "checksum_field": "registry_checksum",
        "gate_field": None,
    },
    "active-diagnostic": {
        "path": "active_closed_loop_diagnostic.json",
        "checksum_field": "report_checksum",
        "gate_field": "collection_complete",
    },
    "adjudicate": {
        "path": "closed_loop_adjudication_report.json",
        "checksum_field": "report_checksum",
        "gate_field": None,
    },
    "report": {
        "path": "terminal_report.json",
        "checksum_field": "report_checksum",
        "gate_field": None,
    },
}


def verify_signed(payload: Mapping[str, Any], checksum_field: str) -> None:
    expected = str(payload.get(checksum_field, ""))
    core = {key: value for key, value in payload.items() if key != checksum_field}
    if not expected or sha256_payload(core) != expected:
        raise IntegrityError(f"invalid {checksum_field}")


def _signed(payload: Mapping[str, Any], checksum_field: str) -> dict[str, Any]:
    output = dict(payload)
    output[checksum_field] = sha256_payload(output)
    return output


def _parent_payload(root: Path, name: str, checksum_field: str) -> dict[str, Any]:
    path = root / PARENT_ARTIFACT_PATHS[name]
    if not path.is_file():
        raise IntegrityError(f"required T10.3.12d artifact is absent: {name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_signed(payload, checksum_field)
    return payload


def source_factor_payload(root: Path) -> dict[str, Any]:
    path = root / SOURCE_FACTOR_REGISTRY_PATH
    if not path.is_file():
        raise IntegrityError("required T10.3.12c source factor registry is absent")
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_signed(payload, "registry_checksum")
    return payload


def parent_journal_digest(root: Path) -> str:
    base = root / PARENT_OUTPUT_DIR / "journal"
    rows = [
        {"relative_path": path.relative_to(base).as_posix(), "sha256": file_sha256(path)}
        for path in sorted(item for item in base.rglob("*.json") if item.is_file())
    ]
    if not rows:
        raise IntegrityError("T10.3.12d parent journal is absent")
    return sha256_payload(rows)


def verify_parent(root: Path) -> dict[str, Any]:
    manifest = parent.load_manifest(root)
    registry = _parent_payload(root, "registry", "registry_checksum")
    active = _parent_payload(root, "active", "report_checksum")
    adjudication = _parent_payload(root, "adjudication", "report_checksum")
    terminal = _parent_payload(root, "terminal", "report_checksum")
    accounting = terminal.get("accounting", {})
    observed = {
        "manifest_checksum": manifest.get("manifest_checksum"),
        "registry_checksum": registry.get("registry_checksum"),
        "active_report_checksum": active.get("report_checksum"),
        "adjudication_report_checksum": adjudication.get("report_checksum"),
        "terminal_report_checksum": terminal.get("report_checksum"),
        "verdict": terminal.get("verdict"),
        "authorized_actions": accounting.get("authorized_actions"),
        "sealed_events": accounting.get("sealed_events"),
        "inflight_intents": accounting.get("inflight_intents"),
        "unresolved_intents": accounting.get("unresolved_intents"),
        "receipt_count": len(active.get("receipt_checksums", ())),
        "distinct_target_initial_frames": active.get("metrics", {}).get(
            "distinct_target_initial_frames"
        ),
    }
    if observed != EXPECTED_PARENT:
        raise IntegrityError("T10.3.12d parent state diverged from the frozen negative")
    if not accounting.get("equation_holds") or not accounting.get("inflight_valid"):
        raise IntegrityError("T10.3.12d accounting is not clean")
    if terminal.get("passed") is not False or terminal.get("program_promoted") is not False:
        raise IntegrityError("T10.3.12d terminal polarity drifted")
    if int(registry.get("local_support_total", -1)) != 0:
        raise IntegrityError("T10.3.12d registry contains target support")
    factor = source_factor_payload(root)
    if registry.get("parent_factor_registry_checksum") != factor.get("registry_checksum"):
        raise IntegrityError("T10.3.12d registry detached from source factor registry")
    return observed


def parent_artifact_bindings(root: Path) -> dict[str, dict[str, str]]:
    rows = {
        name: {"path": path.as_posix(), "sha256": file_sha256(root / path)}
        for name, path in PARENT_ARTIFACT_PATHS.items()
    }
    rows["source_factor_registry"] = {
        "path": SOURCE_FACTOR_REGISTRY_PATH.as_posix(),
        "sha256": file_sha256(root / SOURCE_FACTOR_REGISTRY_PATH),
    }
    return rows


def _code_hashes(root: Path) -> dict[str, str]:
    relative = (
        "theory/sage_t/closed_loop_successor_v10_3_12e.py",
        "theory/sage_t/t10_3_12e_protocol.py",
        "theory/sage_t/t10_3_12e_runtime.py",
        "tests/test_sage_t_closed_loop_successor_v10_3_12e.py",
        "tests/test_sage_t_t10_3_12e_protocol.py",
        "tests/test_sage_t_t10_3_12e_runtime.py",
        "reports/SAGE_T10_3_12E_CLOSED_LOOP_SUCCESSOR_PROTOCOL.md",
        "reports/SAGE_T10_3_12E_CLOSED_LOOP_SUCCESSOR_RUNBOOK.md",
    )
    output = {}
    for item in relative:
        path = root / item
        if not path.is_file():
            raise IntegrityError(f"T10.3.12e protocol dependency is absent: {item}")
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
    if phase != "active-diagnostic":
        raise ValueError(f"unsupported T10.3.12e physical phase: {phase}")
    rows = []
    for game_index, game in enumerate(TARGET_GAMES):
        rotation = game_index % len(ARMS)
        ordered_arms = ARMS[rotation:] + ARMS[:rotation]
        for arm in ordered_arms:
            rows.append(
                WorkSpec(
                    phase=phase,
                    game_id=game,
                    seed=3821 + game_index,
                    arm=arm,
                    reset_index=0,
                    action_budget=ACTION_BUDGET,
                )
            )
    return tuple(rows)


def maximum_actions_for_phase(phase: str) -> int:
    return sum(work.action_budget for work in work_specs(phase))


def maximum_actions_for_specs(specs: Sequence[WorkSpec]) -> int:
    return sum(int(work.action_budget) for work in specs)


def reset_wall_seconds(work: WorkSpec) -> float:
    if work.phase != "active-diagnostic":
        raise ValueError("reset wall budget is active-diagnostic only")
    return RESET_WALL_SECONDS


def build_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    parent_state = verify_parent(root)
    bindings = parent_artifact_bindings(root)
    core = {
        "format_version": FORMAT_VERSION,
        "status": MANIFEST_STATUS,
        "objective": "test_abstract_goal_anchor_with_dynamic_relational_successor_grounding",
        "hypothesis": (
            "a source-derived salient goal-end role combined with a reset-local visited "
            "relational frontier and current-state successor re-grounding yields terminal "
            "progress that frozen, stateless, and goal-end-swapped controls do not"
        ),
        "post_hoc_diagnostic": True,
        "claim_boundary": {
            "closed_loop_mechanism_diagnosable": True,
            "cross_game_generalization_proven": False,
            "factor_generalization_proven": False,
            "independent_confirmation": False,
            "sequence_composition_authorized": False,
            "production_authority": False,
        },
        "code_hashes": _code_hashes(root),
        "parent_state": parent_state,
        "parent_artifacts": bindings,
        "parent_journal_digest": parent_journal_digest(root),
        "cli_phases": [
            "freeze", "status", "audit-parent", "audit-trajectories", "preflight",
            "compile-programs", "active-diagnostic", "adjudicate", "report",
        ],
        "exit_codes": {"success": 0, "integrity": 2, "scientific_gate": 3},
        "artifact_contract": ARTIFACT_CONTRACT,
        "matrix": {
            "games": list(TARGET_GAMES),
            "games_already_observed_in_t10_3_12c_and_d": True,
            "diagnostic_only": True,
            "path_context_only": True,
            "non_path_context_policy": "uniform_zero_action_abstention",
            "arms": list(ARMS),
            "one_reset_per_game_arm": True,
            "labels_seed_environment": False,
            "arm_order": "game_rotated_latin_order",
            "resets": TOTAL_RESETS,
            "maximum_actions_per_reset": ACTION_BUDGET,
            "maximum_actions": TOTAL_MAXIMUM_ACTIONS,
            "maximum_legal_candidates_processed_per_decision": 512,
            "maximum_reset_wall_seconds": RESET_WALL_SECONDS,
            "maximum_global_wall_seconds": 7200,
            "maximum_artifact_bytes": 15 * 1024 * 1024,
        },
        "gates": {
            "parent_receipts": 36,
            "parent_path_applicable_games": 3,
            "parent_stable_successes": 0,
            "parent_changed_frame_events": 91,
            "preflight_cases": 16,
            "all_receipts_required": TOTAL_RESETS,
            "minimum_path_applicable_games": 3,
            "minimum_dynamic_success_games": 1,
            "minimum_dynamic_over_frozen_success_advantage": 1,
            "minimum_dynamic_over_stateless_success_advantage": 1,
            "minimum_dynamic_over_goal_swap_success_advantage": 1,
            "anchor_builds_per_applicable_reset": 1,
            "maximum_dynamic_grounding_misses": 0,
            "minimum_dynamic_exact_grounding_fraction": 1.0,
            "minimum_dynamic_frontier_advance_fraction": 1.0,
        },
        "firewall": {
            "new_games_opened": False,
            "sequence_games_opened": False,
            "source_validation_opened": False,
            "ar25_opened": False,
            "holdout_opened": False,
            "production_authority": False,
            "automatic_retuning": False,
            "legacy_fallback_authorized": False,
            "parent_actions_used_for_training": False,
            "parent_outcomes_used_for_post_hoc_diagnosis": True,
            "parent_grounded_paths_compiled_into_programs": False,
            "parent_action_checksums_compiled_into_programs": False,
        },
        "durability": {
            "intent_before_action": True,
            "event_immediate_seal": True,
            "physical_replay": False,
            "write_once": True,
            "fresh_controller_per_game_arm": True,
            "ephemeral_frontier_cleared_per_reset": True,
            "planned_abstention_is_complete_zero_or_bounded_action_result": True,
        },
        "negative_result_policy": {
            "no_post_freeze_repair": True,
            "terminal_level_delta_is_primary_success_signal": True,
            "frame_change_is_diagnostic_only": True,
            "same_panel_success_is_diagnostic_not_confirmatory": True,
            "no_program_promotion": True,
            "pass_authorizes_only_new_independent_preregistration": True,
        },
        "output_directory": DEFAULT_OUTPUT_DIR.as_posix(),
    }
    return _signed(core, "manifest_checksum")


def freeze_manifest(
    root: Path,
    *,
    manifest_path: Path | None = None,
    receipt_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    manifest = build_manifest(root)
    write_json_once(manifest_path or root / DEFAULT_MANIFEST_PATH, manifest)
    receipt = _signed(
        {
            "format_version": "sage-t10.3.12e-freeze-receipt-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "parent_terminal_checksum": manifest["parent_state"]["terminal_report_checksum"],
            "parent_journal_digest": manifest["parent_journal_digest"],
            "diagnostic_only": True,
            "maximum_actions": TOTAL_MAXIMUM_ACTIONS,
            "physical_actions_at_freeze": 0,
            "new_games_opened": False,
            "holdout_opened": False,
            "production_authority": False,
        },
        "receipt_checksum",
    )
    write_json_once(receipt_path or root / DEFAULT_FREEZE_RECEIPT_PATH, receipt)
    return manifest, receipt


def load_manifest(root: Path, *, verify_code: bool = True) -> dict[str, Any]:
    root = root.resolve()
    path = root / DEFAULT_MANIFEST_PATH
    if not path.is_file():
        raise IntegrityError("T10.3.12e manifest has not been frozen")
    payload = json.loads(path.read_text(encoding="utf-8"))
    verify_signed(payload, "manifest_checksum")
    if payload.get("format_version") != FORMAT_VERSION:
        raise IntegrityError("T10.3.12e manifest format drifted")
    if verify_parent(root) != payload.get("parent_state"):
        raise IntegrityError("T10.3.12d parent state changed after freeze")
    for name, binding in payload.get("parent_artifacts", {}).items():
        bound_path = root / str(binding["path"])
        if not bound_path.is_file() or file_sha256(bound_path) != binding["sha256"]:
            raise IntegrityError(f"frozen T10.3.12e parent binding drifted: {name}")
    if payload.get("parent_journal_digest") != parent_journal_digest(root):
        raise IntegrityError("T10.3.12d journal changed after T10.3.12e freeze")
    if verify_code and payload.get("code_hashes") != _code_hashes(root):
        raise IntegrityError("T10.3.12e code changed after freeze")
    receipt_path = root / DEFAULT_FREEZE_RECEIPT_PATH
    if not receipt_path.is_file():
        raise IntegrityError("T10.3.12e freeze receipt is absent")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    verify_signed(receipt, "receipt_checksum")
    if receipt.get("manifest_checksum") != payload.get("manifest_checksum"):
        raise IntegrityError("T10.3.12e freeze receipt is detached")
    return payload


__all__ = [
    "ACTION_BUDGET", "ARMS", "ARTIFACT_CONTRACT", "DEFAULT_FREEZE_RECEIPT_PATH",
    "DEFAULT_MANIFEST_PATH", "DEFAULT_OUTPUT_DIR", "EXPECTED_PARENT", "FORMAT_VERSION",
    "IntegrityError", "MANIFEST_STATUS", "PARENT_ARTIFACT_PATHS", "PARENT_OUTPUT_DIR",
    "RESET_WALL_SECONDS", "SOURCE_FACTOR_REGISTRY_PATH", "ScientificGateMiss",
    "TARGET_GAMES", "TOTAL_MAXIMUM_ACTIONS", "TOTAL_RESETS", "WorkSpec",
    "build_manifest", "file_sha256", "freeze_manifest", "load_manifest",
    "maximum_actions_for_phase", "maximum_actions_for_specs", "parent_artifact_bindings",
    "parent_journal_digest", "reset_wall_seconds", "sha256_payload",
    "source_factor_payload", "verify_parent", "verify_signed", "work_specs",
    "write_json_once",
]
