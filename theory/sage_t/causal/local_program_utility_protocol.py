"""Frozen T12.5b.4 protocol for target-local short-program utility."""

from __future__ import annotations

import hashlib
import itertools
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
from .progress_contrast_protocol import (
    load_progress_contrast_manifest,
    load_progress_contrast_receipt,
)

LOCAL_UTILITY_PROTOCOL_FORMAT = "sage-t12.5b.4-local-program-utility-protocol-v1"
LOCAL_UTILITY_MANIFEST_FORMAT = "sage-t12.5b.4-local-program-utility-manifest-v1"
LOCAL_UTILITY_RECEIPT_FORMAT = "sage-t12.5b.4-local-program-utility-receipt-v1"

LOCAL_UTILITY_CODE_PATHS = (
    "theory/sage_t/causal/experiment.py",
    "theory/sage_t/causal/progress.py",
    "theory/sage_t/causal/progress_shadow.py",
    "theory/sage_t/causal/progress_shadow_experiment.py",
    "theory/sage_t/causal/progress_contrast_protocol.py",
    "theory/sage_t/causal/option_applicability_experiment.py",
    "theory/sage_t/causal/witness_experiment.py",
    "theory/sage_t/causal/witness_protocol.py",
    "theory/sage_t/causal/local_program_utility.py",
    "theory/sage_t/causal/local_program_utility_protocol.py",
    "theory/sage_t/causal/local_program_utility_experiment.py",
    "theory/sage_t/causal/local_program_utility_cli.py",
    "theory/sage/live_prefix_counterfactual_collector.py",
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
        raise ValueError(f"T12.5b.4 input artifact mismatch: {path}")
    return {"path": _bound_path(path, root=root), "sha256": expected}


@dataclass(frozen=True)
class LocalProgramUtilityProtocol:
    """Immutable T12.5b.4 schedule and scientific gates."""

    format_version: str = LOCAL_UTILITY_PROTOCOL_FORMAT
    parent_status: str = "FAIL_T12_5B_3_COLLECTION_INTEGRITY_GATE"
    parent_classification: str = "COLLECTION_INTEGRITY_FAILURE"
    calibration_lineage_seed: int = 8_701
    evaluation_lineage_seed: int = 8_705
    target_stage: int = 3
    detour_action: str = "ACTION4"
    detour_depth: int = 1
    candidate_actions: tuple[str, ...] = ("ACTION3", "ACTION4", "ACTION6")
    transport_actions: tuple[str, ...] = ("ACTION3", "ACTION4")
    program_lengths: tuple[int, ...] = (2, 3)
    repetitions_per_program: int = 2
    minimum_distractor_magnitude_gap: float = 1.0
    maximum_calibration_sdk_calls: int = 6_500
    maximum_evaluation_sdk_calls: int = 1_000
    maximum_total_sdk_calls: int = 7_500
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
        for name, caster in (
            ("candidate_actions", str),
            ("transport_actions", str),
            ("program_lengths", int),
            ("allowed_effect_features", str),
        ):
            object.__setattr__(
                self,
                name,
                tuple(caster(item) for item in getattr(self, name)),
            )
        object.__setattr__(self, "detour_action", str(self.detour_action).upper())
        expected = {
            "format_version": LOCAL_UTILITY_PROTOCOL_FORMAT,
            "parent_status": "FAIL_T12_5B_3_COLLECTION_INTEGRITY_GATE",
            "parent_classification": "COLLECTION_INTEGRITY_FAILURE",
            "calibration_lineage_seed": 8_701,
            "evaluation_lineage_seed": 8_705,
            "target_stage": 3,
            "detour_action": "ACTION4",
            "detour_depth": 1,
            "candidate_actions": ("ACTION3", "ACTION4", "ACTION6"),
            "transport_actions": ("ACTION3", "ACTION4"),
            "program_lengths": (2, 3),
            "repetitions_per_program": 2,
            "minimum_distractor_magnitude_gap": 1.0,
            "maximum_calibration_sdk_calls": 6_500,
            "maximum_evaluation_sdk_calls": 1_000,
            "maximum_total_sdk_calls": 7_500,
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
                raise ValueError(f"T12.5b.4 preregistered value changed: {name}")
        if not set(self.transport_actions).issubset(self.candidate_actions):
            raise ValueError("T12.5b.4 transport catalogue exceeds calibration")

    @property
    def calibration_programs(self) -> tuple[tuple[str, ...], ...]:
        return tuple(
            tuple(actions)
            for length in self.program_lengths
            for actions in itertools.product(self.candidate_actions, repeat=length)
        )

    @property
    def transport_programs(self) -> tuple[tuple[str, ...], ...]:
        return tuple(
            program
            for program in self.calibration_programs
            if set(program).issubset(self.transport_actions)
        )

    @property
    def expected_calibration_trials(self) -> int:
        return len(self.calibration_programs) * self.repetitions_per_program

    @property
    def expected_evaluation_trials(self) -> int:
        return 2 * self.repetitions_per_program

    @property
    def context_id(self) -> str:
        return "stage_3_action4_depth_1"

    @property
    def checksum(self) -> str:
        return _checksum(asdict(self))


def _parent_target_summary(
    *,
    receipt: Mapping[str, Any],
    affordances: Mapping[str, Any],
    trials: Mapping[str, Any],
    protocol: LocalProgramUtilityProtocol,
) -> dict[str, Any]:
    contexts = [
        dict(item)
        for item in affordances.get("contexts", ())
        if item.get("valid") is True
    ]
    common_depths = []
    for depth in (1, 2, 3):
        context_id = f"stage_3_action4_depth_{depth}"
        seeds = {
            int(item["lineage_seed"])
            for item in contexts
            if item.get("context_id") == context_id
        }
        if seeds == {
            protocol.calibration_lineage_seed,
            protocol.evaluation_lineage_seed,
        }:
            common_depths.append(depth)
    if not common_depths or min(common_depths) != protocol.detour_depth:
        raise ValueError("T12.5b.4 shallowest common context changed")

    parent_rows = [
        dict(item)
        for item in trials.get("trials", ())
        if item.get("context_id") == protocol.context_id
    ]
    expected_rows = 2 * len(protocol.candidate_actions) * 2
    if len(parent_rows) != expected_rows:
        raise ValueError("T12.5b.4 parent depth-1 trial matrix changed")
    if any(item.get("terminal_failure") for item in parent_rows):
        raise ValueError("T12.5b.4 selected parent context is terminal")

    rows = [
        dict(item)
        for item in affordances.get("affordances", ())
        if item.get("context_id") == protocol.context_id
    ]
    executable_by_seed = {
        seed: tuple(
            sorted(
                str(item["action_name"])
                for item in rows
                if int(item["lineage_seed"]) == seed and item.get("executable")
            )
        )
        for seed in (
            protocol.calibration_lineage_seed,
            protocol.evaluation_lineage_seed,
        )
    }
    if executable_by_seed[protocol.calibration_lineage_seed] != (
        "ACTION3",
        "ACTION4",
        "ACTION6",
    ):
        raise ValueError("T12.5b.4 calibration action catalogue changed")
    if executable_by_seed[protocol.evaluation_lineage_seed] != (
        "ACTION3",
        "ACTION4",
    ):
        raise ValueError("T12.5b.4 evaluation action catalogue changed")
    shared = tuple(
        sorted(
            set(executable_by_seed[protocol.calibration_lineage_seed])
            & set(executable_by_seed[protocol.evaluation_lineage_seed])
        )
    )
    if shared != protocol.transport_actions:
        raise ValueError("T12.5b.4 shared transport catalogue changed")
    if any(
        item.get("executable")
        and (
            any(bool(value) for value in item.get("milestone_signature", ()))
            or float(item.get("progress_gain") or 0.0) != 0.0
        )
        for item in rows
    ):
        raise ValueError("T12.5b.4 parent unexpectedly has one-step progress")

    metrics = dict(receipt.get("metrics", {}))
    return {
        "calibration_lineage_seed": protocol.calibration_lineage_seed,
        "calibration_parent_executable_actions": list(
            executable_by_seed[protocol.calibration_lineage_seed]
        ),
        "context_id": protocol.context_id,
        "criterion": (
            "unique shallowest common exact non-terminal context with at least "
            "two deterministic executable actions per lineage"
        ),
        "evaluation_lineage_seed": protocol.evaluation_lineage_seed,
        "evaluation_parent_executable_actions": list(
            executable_by_seed[protocol.evaluation_lineage_seed]
        ),
        "parent_hard_contrast_count": int(metrics.get("hard_contrast_count", -1)),
        "parent_progress_affordance_lineage_count": int(
            metrics.get("progress_affordance_lineage_count", -1)
        ),
        "parent_terminal_failures_at_selected_context": 0,
        "posterior_score_used_for_context_selection": False,
        "shared_transport_actions": list(shared),
    }


def freeze_local_program_utility(
    *,
    output_path: str | Path,
    parent_manifest_path: str | Path,
    parent_receipt_path: str | Path,
    root: str | Path | None = None,
    protocol: LocalProgramUtilityProtocol | None = None,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    selected = protocol or LocalProgramUtilityProtocol()
    parent_manifest_path = Path(parent_manifest_path).resolve()
    parent_receipt_path = Path(parent_receipt_path).resolve()
    parent_manifest = load_progress_contrast_manifest(
        parent_manifest_path,
        root=repo_root,
        verify_code=True,
    )
    parent_receipt = load_progress_contrast_receipt(
        parent_receipt_path,
        manifest=parent_manifest,
        root=repo_root,
        require_passed=False,
    )
    metrics = dict(parent_receipt.get("metrics", {}))
    if parent_receipt.get("passed") is not False:
        raise ValueError("T12.5b.4 requires the sealed negative T12.5b.3 result")
    if parent_receipt.get("status") != selected.parent_status:
        raise ValueError("T12.5b.4 parent status changed")
    if metrics.get("classification") != selected.parent_classification:
        raise ValueError("T12.5b.4 parent classification changed")
    checks = dict(metrics.get("checks", {}))
    failed_checks = tuple(sorted(name for name, passed in checks.items() if not passed))
    expected_failures = (
        "causal_ranking_beats_magnitude_on_hard_contrasts",
        "causal_ranking_is_perfect_on_hard_contrasts",
        "common_hard_contrast_context_exists",
        "hard_contrast_exists_in_every_lineage",
        "no_terminal_failures",
        "progress_affordance_binds_across_lineages",
        "progress_affordance_observed_in_both_lineages",
    )
    if failed_checks != expected_failures:
        raise ValueError("T12.5b.4 parent failure class changed")
    if parent_manifest.get("stage") != "source_train":
        raise ValueError("T12.5b.4 is restricted to source_train")

    parent_artifacts = {
        name: _verified_artifact(parent_receipt["artifacts"][name], root=repo_root)
        for name in ("affordances", "contrasts", "report", "trials")
    }
    affordance_payload = _read_json(
        _resolve_bound(parent_artifacts["affordances"]["path"], root=repo_root)
    )
    trials_payload = _read_json(
        _resolve_bound(parent_artifacts["trials"]["path"], root=repo_root)
    )
    selection = _parent_target_summary(
        receipt=parent_receipt,
        affordances=affordance_payload,
        trials=trials_payload,
        protocol=selected,
    )
    physical_inputs = {
        name: _verified_artifact(parent_manifest["inputs"][name], root=repo_root)
        for name in (
            "applicability_trials",
            "minimal_option",
            "posterior",
            "program_registry",
            "witness_registry",
        )
    }

    missing = [
        path for path in LOCAL_UTILITY_CODE_PATHS if not (repo_root / path).is_file()
    ]
    if missing:
        raise ValueError(f"T12.5b.4 code inventory is incomplete: {missing}")
    git = _git_state(repo_root)
    if git["dirty"] and not allow_dirty:
        raise ValueError("scientific freeze requires a clean worktree")
    authorized = bool(not git["dirty"])
    payload = {
        "format_version": LOCAL_UTILITY_MANIFEST_FORMAT,
        "status": "FROZEN_BEFORE_T12_5B_4_CALIBRATION",
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
            "failed_checks": list(failed_checks),
        },
        "inputs": {
            **physical_inputs,
            "successful_prefix_lengths": {
                str(seed): int(
                    parent_manifest["inputs"]["successful_prefix_lengths"][str(seed)]
                )
                for seed in (
                    selected.calibration_lineage_seed,
                    selected.evaluation_lineage_seed,
                )
            },
            "successful_anchor_hashes": {
                str(seed): str(
                    parent_manifest["inputs"]["successful_anchor_hashes"][str(seed)]
                )
                for seed in (
                    selected.calibration_lineage_seed,
                    selected.evaluation_lineage_seed,
                )
            },
        },
        "selection": selection,
        "design": {
            "bounded_program_lengths": list(selected.program_lengths),
            "calibration_before_evaluation": True,
            "candidate_terminal_is_scientific_risk_not_integrity_failure": True,
            "detour_terminal_invalidates_context": True,
            "labels_use_level_delta_and_terminal_state_not_causal_score": True,
            "parent_negative_result_preserved": True,
            "parent_next_phase_authorized": False,
            "posterior_frozen_before_calibration": True,
            "program_selection_independent_of_causal_score": True,
            "separate_iteration_explicitly_user_scoped": True,
            "unavailable_action_is_missing_not_zero_effect": True,
        },
        "code_sha256": {
            path: _file_sha256(repo_root / path) for path in LOCAL_UTILITY_CODE_PATHS
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
                "source-train target-local short-program calibration; evaluation "
                "requires a passed signed calibration receipt"
            ),
            "not_authorized": [
                "policy improvement",
                "environment control",
                "target-game generalization",
                "source validation",
                "holdout performance",
                "neural training",
                "production authority",
            ],
        },
        "storage": {
            "maximum_artifact_bytes_per_phase": (
                selected.maximum_artifact_bytes_per_phase
            ),
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
    receipt = local_program_utility_receipt(
        manifest=manifest,
        phase="freeze",
        passed=authorized,
        status="PASS_T12_5B_4_FREEZE" if authorized else "DIRTY_SMOKE_ONLY",
        metrics={
            "calibration_program_count": len(selected.calibration_programs),
            "expected_calibration_trials": selected.expected_calibration_trials,
            "expected_evaluation_trials": selected.expected_evaluation_trials,
            "parent_negative_result_preserved": True,
            "transport_program_count": len(selected.transport_programs),
        },
    )
    _write_json_once(Path(output_path).with_name("freeze_receipt.json"), receipt)
    return manifest


def load_local_program_utility_manifest(
    path: str | Path,
    *,
    root: str | Path | None = None,
    verify_code: bool = True,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = _read_json(path)
    _verify_signed(manifest, "manifest_checksum")
    if manifest.get("format_version") != LOCAL_UTILITY_MANIFEST_FORMAT:
        raise ValueError("unsupported T12.5b.4 local-utility manifest")
    protocol = LocalProgramUtilityProtocol(**dict(manifest["protocol"]))
    if protocol.checksum != manifest.get("protocol_checksum"):
        raise ValueError("T12.5b.4 protocol checksum mismatch")
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
            raise ValueError(f"T12.5b.4 bound artifact mismatch: {name}")
    if verify_code:
        for relative, expected in manifest["code_sha256"].items():
            candidate = repo_root / relative
            if not candidate.is_file() or _file_sha256(candidate) != expected:
                raise ValueError(f"T12.5b.4 code checksum mismatch: {relative}")
    return manifest


def local_program_utility_receipt(
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
            "format_version": LOCAL_UTILITY_RECEIPT_FORMAT,
            "phase": str(phase),
            "passed": bool(passed),
            "status": str(status),
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "parent_t12_5b_3_receipt_checksum": manifest["parent"]["receipt"][
                "receipt_checksum"
            ],
            "metrics": dict(metrics),
            "artifacts": dict(artifacts or {}),
        },
        "receipt_checksum",
    )


def load_local_program_utility_receipt(
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
    if receipt.get("format_version") != LOCAL_UTILITY_RECEIPT_FORMAT:
        raise ValueError("unsupported T12.5b.4 local-utility receipt")
    if manifest is not None and (
        receipt.get("manifest_checksum") != manifest.get("manifest_checksum")
        or receipt.get("protocol_checksum") != manifest.get("protocol_checksum")
    ):
        raise ValueError("T12.5b.4 receipt belongs to another manifest")
    if expected_phase is not None and receipt.get("phase") != expected_phase:
        raise ValueError("T12.5b.4 receipt phase mismatch")
    if require_passed and receipt.get("passed") is not True:
        raise ValueError(f"T12.5b.4 gate failed: {receipt.get('status')}")
    for name, meta in receipt.get("artifacts", {}).items():
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError(f"T12.5b.4 receipt artifact mismatch: {name}")
    return receipt


def load_signed_evaluation_registry(
    path: str | Path,
    *,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _read_json(path)
    _verify_signed(payload, "registry_checksum")
    if payload.get("format_version") != "sage-t12.5b.4-evaluation-registry-v1":
        raise ValueError("unsupported T12.5b.4 evaluation registry")
    if payload.get("manifest_checksum") != manifest.get("manifest_checksum"):
        raise ValueError("T12.5b.4 evaluation registry belongs to another manifest")
    if payload.get("protocol_checksum") != manifest.get("protocol_checksum"):
        raise ValueError("T12.5b.4 evaluation registry protocol changed")
    return payload


__all__ = [
    "LOCAL_UTILITY_MANIFEST_FORMAT",
    "LOCAL_UTILITY_PROTOCOL_FORMAT",
    "LOCAL_UTILITY_RECEIPT_FORMAT",
    "LocalProgramUtilityProtocol",
    "freeze_local_program_utility",
    "load_local_program_utility_manifest",
    "load_local_program_utility_receipt",
    "load_signed_evaluation_registry",
    "local_program_utility_receipt",
]
