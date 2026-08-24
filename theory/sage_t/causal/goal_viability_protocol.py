"""Frozen T12.5b.5 protocol for goal-continuation viability."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .experiment import (
    _bound_path,
    _file_sha256,
    _git_state,
    _read_json,
    _signed,
    _verify_signed,
    _write_json_once,
)
from .goal_viability import ViabilityBranch, viability_branch
from .local_program_utility_protocol import (
    LOCAL_UTILITY_CODE_PATHS,
    load_local_program_utility_manifest,
    load_local_program_utility_receipt,
)

GOAL_VIABILITY_PROTOCOL_FORMAT = "sage-t12.5b.5-goal-viability-protocol-v1"
GOAL_VIABILITY_MANIFEST_FORMAT = "sage-t12.5b.5-goal-viability-manifest-v1"
GOAL_VIABILITY_RECEIPT_FORMAT = "sage-t12.5b.5-goal-viability-receipt-v1"

GOAL_VIABILITY_CODE_PATHS = tuple(
    dict.fromkeys(
        (
            *LOCAL_UTILITY_CODE_PATHS,
            "theory/sage_t/causal/goal_viability.py",
            "theory/sage_t/causal/goal_viability_protocol.py",
            "theory/sage_t/causal/goal_viability_experiment.py",
            "theory/sage_t/causal/goal_viability_cli.py",
        )
    )
)


def _canonical(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    )


def _checksum(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _resolve_bound(path: str, *, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _verified_artifact(meta: Mapping[str, Any], *, root: Path) -> dict[str, str]:
    path = _resolve_bound(str(meta["path"]), root=root)
    expected = str(meta["sha256"])
    if not path.is_file() or _file_sha256(path) != expected:
        raise ValueError(f"T12.5b.5 input artifact mismatch: {path}")
    return {"path": _bound_path(path, root=root), "sha256": expected}


@dataclass(frozen=True)
class GoalViabilityProtocol:
    """Immutable paired branch schedule and advancement firewall."""

    format_version: str = GOAL_VIABILITY_PROTOCOL_FORMAT
    parent_status: str = "FAIL_T12_5B_4_NO_LOCAL_PROGRESS_PROGRAM"
    parent_classification: str = "NO_LOCAL_PROGRESS_PROGRAM"
    calibration_lineage_seed: int = 8_701
    evaluation_lineage_seed: int = 8_705
    target_stage: int = 3
    candidate_first_actions: tuple[str, ...] = ("ACTION3", "ACTION4", "ACTION6")
    transport_first_actions: tuple[str, ...] = ("ACTION3", "ACTION4")
    goal_continuation: tuple[str, ...] = ("ACTION3", "ACTION3")
    repetitions_per_branch: int = 2
    maximum_calibration_sdk_calls: int = 1_000
    maximum_evaluation_sdk_calls: int = 750
    maximum_total_sdk_calls: int = 1_750
    maximum_wall_seconds_per_phase: int = 7_200
    maximum_artifact_bytes_per_phase: int = 3 * 1024 * 1024 * 1024
    allowed_effect_features: tuple[str, ...] = (
        "predicate_counts.adjacent",
        "predicate_counts.aligned",
        "predicate_counts.contact",
        "predicate_counts.near",
        "role_counts.clickable",
        "role_counts.movable",
    )
    persist_raw_frames: bool = False

    def __post_init__(self) -> None:
        for name in (
            "candidate_first_actions",
            "transport_first_actions",
            "goal_continuation",
            "allowed_effect_features",
        ):
            object.__setattr__(
                self,
                name,
                tuple(str(item) for item in getattr(self, name)),
            )
        expected = {
            "format_version": GOAL_VIABILITY_PROTOCOL_FORMAT,
            "parent_status": "FAIL_T12_5B_4_NO_LOCAL_PROGRESS_PROGRAM",
            "parent_classification": "NO_LOCAL_PROGRESS_PROGRAM",
            "calibration_lineage_seed": 8_701,
            "evaluation_lineage_seed": 8_705,
            "target_stage": 3,
            "candidate_first_actions": ("ACTION3", "ACTION4", "ACTION6"),
            "transport_first_actions": ("ACTION3", "ACTION4"),
            "goal_continuation": ("ACTION3", "ACTION3"),
            "repetitions_per_branch": 2,
            "maximum_calibration_sdk_calls": 1_000,
            "maximum_evaluation_sdk_calls": 750,
            "maximum_total_sdk_calls": 1_750,
            "maximum_wall_seconds_per_phase": 7_200,
            "maximum_artifact_bytes_per_phase": 3 * 1024 * 1024 * 1024,
            "allowed_effect_features": (
                "predicate_counts.adjacent",
                "predicate_counts.aligned",
                "predicate_counts.contact",
                "predicate_counts.near",
                "role_counts.clickable",
                "role_counts.movable",
            ),
            "persist_raw_frames": False,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"T12.5b.5 preregistered value changed: {name}")
        if not set(self.transport_first_actions).issubset(
            self.candidate_first_actions
        ):
            raise ValueError("T12.5b.5 transport catalogue exceeds calibration")

    @property
    def calibration_branches(self) -> tuple[ViabilityBranch, ...]:
        return tuple(
            viability_branch(
                action,
                goal_continuation=self.goal_continuation,
                transport_actions=self.transport_first_actions,
            )
            for action in self.candidate_first_actions
        )

    @property
    def transport_branches(self) -> tuple[ViabilityBranch, ...]:
        return tuple(
            branch for branch in self.calibration_branches if branch.transport_eligible
        )

    @property
    def expected_calibration_trials(self) -> int:
        return len(self.calibration_branches) * self.repetitions_per_branch

    @property
    def expected_evaluation_trials(self) -> int:
        return 2 * self.repetitions_per_branch

    @property
    def context_id(self) -> str:
        return "stage_3_goal_cursor"

    @property
    def checksum(self) -> str:
        return _checksum(asdict(self))


def _sealed_parent_diagnostic(
    *,
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, str]],
    protocol: GoalViabilityProtocol,
    root: Path,
) -> dict[str, Any]:
    option_path = _resolve_bound(str(manifest["inputs"]["minimal_option"]["path"]), root=root)
    option_payload = _read_json(option_path)
    option = dict(option_payload["option"])
    actions = tuple(str(step["action_name"]).upper() for step in option["steps"])
    expected_option = ("ACTION4", "ACTION4", "ACTION4", "ACTION3", "ACTION3")
    if actions != expected_option:
        raise ValueError("T12.5b.5 confirmed minimal option changed")
    if actions[protocol.target_stage :] != protocol.goal_continuation:
        raise ValueError("T12.5b.5 goal continuation changed")
    bindings = tuple(option_payload.get("context_bindings", ()))
    if {int(item["seed"]) for item in bindings} != {
        protocol.calibration_lineage_seed,
        protocol.evaluation_lineage_seed,
    }:
        raise ValueError("T12.5b.5 option is not confirmed on both route lineages")
    if any(int(item["target_level"]) != 1 for item in bindings):
        raise ValueError("T12.5b.5 option target level changed")

    trials_payload = _read_json(_resolve_bound(artifacts["trials"]["path"], root=root))
    fatal = tuple(
        row
        for row in trials_payload.get("trials", ())
        if str(row.get("program_id")) == "ACTION3>ACTION3"
        and int(row.get("lineage_seed", -1)) == protocol.calibration_lineage_seed
    )
    if len(fatal) != protocol.repetitions_per_branch:
        raise ValueError("T12.5b.5 sealed fatal-detour evidence changed")
    if not all(
        row.get("prefix_exact") is True
        and row.get("terminal_failure") is True
        and int(row.get("level_delta", 0)) == 0
        for row in fatal
    ):
        raise ValueError("T12.5b.5 parent no longer proves the fatal detour")

    checks = dict(receipt.get("metrics", {}).get("checks", {}))
    integrity_names = (
        "availability_is_deterministic",
        "context_replay_is_exact",
        "effects_are_deterministic",
        "effects_are_deterministic_when_complete",
        "fixed_program_schedule_completed",
        "outcomes_are_deterministic",
        "repetition_count_is_exact",
        "sdk_budget_respected",
        "wall_time_respected",
    )
    if not all(checks.get(name) is True for name in integrity_names):
        raise ValueError("T12.5b.5 requires an integrity-clean parent calibration")
    return {
        "calibration_branch_programs": [
            list(branch.program_actions) for branch in protocol.calibration_branches
        ],
        "context_id": protocol.context_id,
        "fatal_detour_action": "ACTION4",
        "fatal_detour_evidence_ids": [str(row["trial_id"]) for row in fatal],
        "goal_continuation": list(protocol.goal_continuation),
        "goal_continuation_source": {
            "context_binding_seeds": sorted(int(item["seed"]) for item in bindings),
            "minimal_option_checksum": str(option_payload["option_checksum"]),
            "minimum_successful_length": int(
                option_payload["minimality"]["minimum_successful_length"]
            ),
            "reproduction_count": int(option["reproduction_count"]),
            "target_level": 1,
        },
        "labels_use_future_level_delta_not_immediate_magnitude": True,
        "parent_integrity_checks": {name: True for name in integrity_names},
        "score_used_for_branch_selection": False,
        "target_stage": protocol.target_stage,
        "transport_branch_programs": [
            list(branch.program_actions) for branch in protocol.transport_branches
        ],
    }


def freeze_goal_viability(
    *,
    output_path: str | Path,
    parent_manifest_path: str | Path,
    parent_receipt_path: str | Path,
    root: str | Path | None = None,
    protocol: GoalViabilityProtocol | None = None,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    selected = protocol or GoalViabilityProtocol()
    parent_manifest_path = Path(parent_manifest_path).resolve()
    parent_receipt_path = Path(parent_receipt_path).resolve()
    parent_manifest = load_local_program_utility_manifest(
        parent_manifest_path, root=repo_root, verify_code=True
    )
    parent_receipt = load_local_program_utility_receipt(
        parent_receipt_path,
        manifest=parent_manifest,
        root=repo_root,
        require_passed=False,
        expected_phase="calibration",
    )
    metrics = dict(parent_receipt.get("metrics", {}))
    if parent_receipt.get("passed") is not False:
        raise ValueError("T12.5b.5 requires the sealed negative T12.5b.4 result")
    if parent_receipt.get("status") != selected.parent_status:
        raise ValueError("T12.5b.5 parent status changed")
    if metrics.get("classification") != selected.parent_classification:
        raise ValueError("T12.5b.5 parent classification changed")
    if metrics.get("evaluation_collection_authorized") is not False:
        raise ValueError("T12.5b.5 parent evaluation firewall changed")
    if parent_manifest.get("stage") != "source_train":
        raise ValueError("T12.5b.5 is restricted to source_train")

    parent_artifacts = {
        name: _verified_artifact(parent_receipt["artifacts"][name], root=repo_root)
        for name in ("programs", "report", "trials")
    }
    inputs = {
        name: _verified_artifact(parent_manifest["inputs"][name], root=repo_root)
        for name in (
            "applicability_trials",
            "minimal_option",
            "posterior",
            "program_registry",
            "witness_registry",
        )
    }
    selection = _sealed_parent_diagnostic(
        manifest=parent_manifest,
        receipt=parent_receipt,
        artifacts=parent_artifacts,
        protocol=selected,
        root=repo_root,
    )
    missing = [
        path for path in GOAL_VIABILITY_CODE_PATHS if not (repo_root / path).is_file()
    ]
    if missing:
        raise ValueError(f"T12.5b.5 code inventory is incomplete: {missing}")
    git = _git_state(repo_root)
    if git["dirty"] and not allow_dirty:
        raise ValueError("scientific freeze requires a clean worktree")
    authorized = bool(not git["dirty"])
    payload = {
        "format_version": GOAL_VIABILITY_MANIFEST_FORMAT,
        "status": "FROZEN_BEFORE_T12_5B_5_CALIBRATION",
        "stage": "source_train",
        "game_id": parent_manifest["game_id"],
        "protocol": asdict(selected),
        "protocol_checksum": selected.checksum,
        "parent": {
            "manifest": {
                "path": _bound_path(parent_manifest_path, root=repo_root),
                "sha256": _file_sha256(parent_manifest_path),
                "manifest_checksum": parent_manifest["manifest_checksum"],
            },
            "receipt": {
                "path": _bound_path(parent_receipt_path, root=repo_root),
                "sha256": _file_sha256(parent_receipt_path),
                "receipt_checksum": parent_receipt["receipt_checksum"],
                "status": parent_receipt["status"],
                "classification": metrics["classification"],
                "next_phase_authorized": False,
            },
            "artifacts": parent_artifacts,
        },
        "inputs": {
            **inputs,
            "successful_prefix_lengths": {
                str(seed): int(
                    parent_manifest["inputs"]["successful_prefix_lengths"][str(seed)]
                )
                for seed in (
                    selected.calibration_lineage_seed,
                    selected.evaluation_lineage_seed,
                )
            },
        },
        "selection": selection,
        "design": {
            "branch_labels_are_score_independent": True,
            "candidate_terminal_is_scientific_risk_not_integrity_failure": True,
            "goal_continuation_is_reacquired_from_live_legal_actions": True,
            "immediate_milestone_neutrality_is_not_viability": True,
            "parent_negative_result_preserved": True,
            "parent_next_phase_authorized": False,
            "separate_iteration_explicitly_user_scoped": True,
            "unavailable_action_is_missing_not_zero_effect": True,
        },
        "code_sha256": {
            path: _file_sha256(repo_root / path) for path in GOAL_VIABILITY_CODE_PATHS
        },
        "git": git,
        "scientific_claims_authorized": authorized,
        "firewall": {
            "calibration_collection_authorized": authorized,
            "evaluation_collection_authorized": False,
            "environment_collection_authorized": authorized,
            "causal_progress_control_authorized": False,
            "holdout_opened": False,
            "source_validation_opened": False,
            "production_authority": False,
            "neural_training_authorized": False,
            "neural_active_evaluation_authorized": False,
            "option_control_authorized": False,
            "t12_5c_control_freeze_authorized": False,
            "t12_6_freeze_authorized": False,
        },
        "claim_boundary": {
            "authorized": (
                "source-train goal-continuation viability calibration; evaluation "
                "requires a passed signed calibration receipt"
            ),
            "not_authorized": [
                "generic ARC-AGI improvement",
                "environment control",
                "target-game generalization",
                "source validation",
                "holdout performance",
                "neural training",
                "production authority",
            ],
        },
        "storage": {
            "maximum_artifact_bytes_per_phase": selected.maximum_artifact_bytes_per_phase,
            "maximum_calibration_sdk_calls": selected.maximum_calibration_sdk_calls,
            "maximum_evaluation_sdk_calls": selected.maximum_evaluation_sdk_calls,
            "maximum_total_sdk_calls": selected.maximum_total_sdk_calls,
            "maximum_wall_seconds_per_phase": selected.maximum_wall_seconds_per_phase,
            "persist_raw_frames": False,
            "hard_fail_before_write": True,
        },
    }
    manifest = _signed(payload, "manifest_checksum")
    _write_json_once(output_path, manifest)
    receipt = goal_viability_receipt(
        manifest=manifest,
        phase="freeze",
        passed=authorized,
        status="PASS_T12_5B_5_FREEZE" if authorized else "DIRTY_SMOKE_ONLY",
        metrics={
            "calibration_branch_count": len(selected.calibration_branches),
            "expected_calibration_trials": selected.expected_calibration_trials,
            "expected_evaluation_trials": selected.expected_evaluation_trials,
            "parent_negative_result_preserved": True,
            "transport_branch_count": len(selected.transport_branches),
        },
    )
    _write_json_once(Path(output_path).with_name("freeze_receipt.json"), receipt)
    return manifest


def load_goal_viability_manifest(
    path: str | Path,
    *,
    root: str | Path | None = None,
    verify_code: bool = True,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = _read_json(path)
    _verify_signed(manifest, "manifest_checksum")
    if manifest.get("format_version") != GOAL_VIABILITY_MANIFEST_FORMAT:
        raise ValueError("unsupported T12.5b.5 goal-viability manifest")
    protocol = GoalViabilityProtocol(**dict(manifest["protocol"]))
    if protocol.checksum != manifest.get("protocol_checksum"):
        raise ValueError("T12.5b.5 protocol checksum mismatch")
    metas: list[tuple[str, Mapping[str, Any]]] = []
    for name, meta in manifest.get("parent", {}).items():
        if name == "artifacts" and isinstance(meta, Mapping):
            metas.extend((f"parent.{key}", value) for key, value in meta.items())
        elif isinstance(meta, Mapping) and "path" in meta:
            metas.append((f"parent.{name}", meta))
    metas.extend(
        (f"inputs.{name}", meta)
        for name, meta in manifest.get("inputs", {}).items()
        if isinstance(meta, Mapping) and "path" in meta
    )
    for name, meta in metas:
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError(f"T12.5b.5 bound artifact mismatch: {name}")
    if verify_code:
        for relative, expected in manifest["code_sha256"].items():
            candidate = repo_root / relative
            if not candidate.is_file() or _file_sha256(candidate) != expected:
                raise ValueError(f"T12.5b.5 code checksum mismatch: {relative}")
    return manifest


def goal_viability_receipt(
    *,
    manifest: Mapping[str, Any],
    phase: str,
    passed: bool,
    status: str,
    metrics: Mapping[str, Any],
    artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return _signed(
        {
            "format_version": GOAL_VIABILITY_RECEIPT_FORMAT,
            "phase": str(phase),
            "passed": bool(passed),
            "status": str(status),
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "parent_t12_5b_4_receipt_checksum": manifest["parent"]["receipt"][
                "receipt_checksum"
            ],
            "metrics": dict(metrics),
            "artifacts": dict(artifacts or {}),
        },
        "receipt_checksum",
    )


def load_goal_viability_receipt(
    path: str | Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
    require_passed: bool = False,
    expected_phase: str | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    receipt = _read_json(path)
    _verify_signed(receipt, "receipt_checksum")
    if receipt.get("format_version") != GOAL_VIABILITY_RECEIPT_FORMAT:
        raise ValueError("unsupported T12.5b.5 goal-viability receipt")
    if manifest is not None and (
        receipt.get("manifest_checksum") != manifest.get("manifest_checksum")
        or receipt.get("protocol_checksum") != manifest.get("protocol_checksum")
    ):
        raise ValueError("T12.5b.5 receipt belongs to another manifest")
    if expected_phase is not None and receipt.get("phase") != expected_phase:
        raise ValueError("T12.5b.5 receipt phase mismatch")
    if require_passed and receipt.get("passed") is not True:
        raise ValueError(f"T12.5b.5 gate failed: {receipt.get('status')}")
    for name, meta in receipt.get("artifacts", {}).items():
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError(f"T12.5b.5 receipt artifact mismatch: {name}")
    return receipt


def load_signed_evaluation_registry(
    path: str | Path,
    *,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _read_json(path)
    _verify_signed(payload, "registry_checksum")
    if payload.get("format_version") != (
        "sage-t12.5b.5-goal-viability-evaluation-registry-v1"
    ):
        raise ValueError("unsupported T12.5b.5 evaluation registry")
    if payload.get("manifest_checksum") != manifest.get("manifest_checksum"):
        raise ValueError("T12.5b.5 evaluation registry belongs to another manifest")
    if payload.get("protocol_checksum") != manifest.get("protocol_checksum"):
        raise ValueError("T12.5b.5 evaluation registry protocol changed")
    return payload


__all__ = [
    "GOAL_VIABILITY_MANIFEST_FORMAT",
    "GOAL_VIABILITY_PROTOCOL_FORMAT",
    "GOAL_VIABILITY_RECEIPT_FORMAT",
    "GoalViabilityProtocol",
    "freeze_goal_viability",
    "goal_viability_receipt",
    "load_goal_viability_manifest",
    "load_goal_viability_receipt",
    "load_signed_evaluation_registry",
]
