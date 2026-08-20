"""Frozen prospective protocol for SAGE.T12.5b.3 progress contrasts."""

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
from .progress_discrimination_protocol import (
    load_progress_discrimination_manifest,
    load_progress_discrimination_receipt,
)
from .progress_shadow_protocol import load_progress_shadow_manifest

CONTRAST_PROTOCOL_FORMAT = "sage-t12.5b.3-progress-contrast-protocol-v1"
CONTRAST_MANIFEST_FORMAT = "sage-t12.5b.3-progress-contrast-manifest-v1"
CONTRAST_RECEIPT_FORMAT = "sage-t12.5b.3-progress-contrast-receipt-v1"

CONTRAST_CODE_PATHS = (
    "theory/sage_t/causal/experiment.py",
    "theory/sage_t/causal/progress.py",
    "theory/sage_t/causal/progress_shadow.py",
    "theory/sage_t/causal/progress_discrimination.py",
    "theory/sage_t/causal/progress_shadow_experiment.py",
    "theory/sage_t/causal/progress_contrast.py",
    "theory/sage_t/causal/progress_contrast_protocol.py",
    "theory/sage_t/causal/progress_contrast_experiment.py",
    "theory/sage_t/causal/progress_contrast_cli.py",
    "theory/sage_t/causal/option_applicability_experiment.py",
    "theory/sage_t/causal/witness_experiment.py",
    "theory/sage_t/causal/witness_protocol.py",
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


@dataclass(frozen=True)
class ProgressContrastProtocol:
    """Immutable T12.5b.3 schedule and scientific gate."""

    format_version: str = CONTRAST_PROTOCOL_FORMAT
    parent_status: str = "FAIL_T12_5B_2_INSUFFICIENT_DISCRIMINATIVE_CONTRASTS"
    parent_classification: str = "INSUFFICIENT_DISCRIMINATIVE_CONTRASTS"
    lineage_seeds: tuple[int, ...] = (8_701, 8_705)
    target_stage: int = 3
    detour_action: str = "ACTION4"
    detour_depths: tuple[int, ...] = (1, 2, 3)
    candidate_actions: tuple[str, ...] = (
        "ACTION3",
        "ACTION4",
        "ACTION6",
    )
    excluded_non_executable_actions: tuple[str, ...] = ("ACTION7",)
    allowed_effect_features: tuple[str, ...] = (
        "predicate_counts.adjacent",
        "predicate_counts.aligned",
        "predicate_counts.contact",
        "predicate_counts.near",
        "role_counts.clickable",
        "role_counts.movable",
    )
    repetitions_per_branch: int = 2
    minimum_executable_actions_per_context: int = 2
    minimum_valid_contexts_per_lineage: int = 1
    minimum_common_valid_contexts: int = 1
    minimum_distractor_magnitude_gap: float = 1.0
    minimum_hard_contrasts_per_lineage: int = 1
    minimum_common_hard_contrast_contexts: int = 1
    minimum_causal_hard_contrast_accuracy: float = 1.0
    minimum_hard_contrast_accuracy_gain: float = 0.5
    maximum_terminal_failures: int = 0
    maximum_sdk_calls: int = 3_500
    maximum_wall_seconds: int = 7_200
    maximum_artifact_bytes_per_run: int = 3 * 1024 * 1024 * 1024
    persist_raw_frames: bool = False

    def __post_init__(self) -> None:
        for name, caster in (
            ("lineage_seeds", int),
            ("detour_depths", int),
            ("candidate_actions", str),
            ("excluded_non_executable_actions", str),
            ("allowed_effect_features", str),
        ):
            object.__setattr__(
                self, name, tuple(caster(item) for item in getattr(self, name))
            )
        object.__setattr__(self, "detour_action", str(self.detour_action).upper())
        expected = {
            "format_version": CONTRAST_PROTOCOL_FORMAT,
            "parent_status": (
                "FAIL_T12_5B_2_INSUFFICIENT_DISCRIMINATIVE_CONTRASTS"
            ),
            "parent_classification": "INSUFFICIENT_DISCRIMINATIVE_CONTRASTS",
            "lineage_seeds": (8_701, 8_705),
            "target_stage": 3,
            "detour_action": "ACTION4",
            "detour_depths": (1, 2, 3),
            "candidate_actions": ("ACTION3", "ACTION4", "ACTION6"),
            "excluded_non_executable_actions": ("ACTION7",),
            "allowed_effect_features": (
                "predicate_counts.adjacent",
                "predicate_counts.aligned",
                "predicate_counts.contact",
                "predicate_counts.near",
                "role_counts.clickable",
                "role_counts.movable",
            ),
            "repetitions_per_branch": 2,
            "minimum_executable_actions_per_context": 2,
            "minimum_valid_contexts_per_lineage": 1,
            "minimum_common_valid_contexts": 1,
            "minimum_distractor_magnitude_gap": 1.0,
            "minimum_hard_contrasts_per_lineage": 1,
            "minimum_common_hard_contrast_contexts": 1,
            "minimum_causal_hard_contrast_accuracy": 1.0,
            "minimum_hard_contrast_accuracy_gain": 0.5,
            "maximum_terminal_failures": 0,
            "maximum_sdk_calls": 3_500,
            "maximum_wall_seconds": 7_200,
            "maximum_artifact_bytes_per_run": 3 * 1024 * 1024 * 1024,
            "persist_raw_frames": False,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"T12.5b.3 preregistered value changed: {name}")
        if self.detour_action not in self.candidate_actions:
            raise ValueError("T12.5b.3 detour action is outside the candidate catalog")

    @property
    def context_ids(self) -> tuple[str, ...]:
        return tuple(f"stage_3_action4_depth_{depth}" for depth in self.detour_depths)

    @property
    def expected_trial_count(self) -> int:
        return (
            len(self.lineage_seeds)
            * len(self.detour_depths)
            * len(self.candidate_actions)
            * self.repetitions_per_branch
        )

    @property
    def checksum(self) -> str:
        return _checksum(asdict(self))


def _resolve_bound(path: str, *, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _verified_artifact(meta: Mapping[str, Any], *, root: Path) -> dict[str, str]:
    path = _resolve_bound(str(meta["path"]), root=root)
    expected = str(meta["sha256"])
    if not path.is_file() or _file_sha256(path) != expected:
        raise ValueError(f"T12.5b.3 input artifact mismatch: {path}")
    return {"path": _bound_path(path, root=root), "sha256": expected}


def _target_selection_summary(
    affordance_payload: Mapping[str, Any], *, protocol: ProgressContrastProtocol
) -> dict[str, Any]:
    """Verify that stage 3 is the unique nearest observed magnitude contest."""

    rows = tuple(dict(item) for item in affordance_payload.get("affordances", ()))
    if not rows:
        raise ValueError("T12.5b.3 parent affordance registry is empty")
    per_stage: dict[int, dict[str, Any]] = {}
    milestone_count = 5
    for stage in range(milestone_count):
        wanted = tuple(index == stage for index in range(milestone_count))
        lineage_rows = []
        for seed in protocol.lineage_seeds:
            local = [
                item
                for item in rows
                if int(item.get("lineage_seed", -1)) == seed
                and int(item.get("stage", -1)) == stage
                and item.get("executable") is True
                and item.get("effect_deterministic") is True
            ]
            progress = [
                item
                for item in local
                if tuple(bool(value) for value in item["milestone_signature"])
                == wanted
            ]
            distractors = [
                item
                for item in local
                if tuple(bool(value) for value in item["milestone_signature"])
                != wanted
            ]
            if len(progress) != 1 or not distractors:
                raise ValueError(
                    f"T12.5b.3 cannot derive the parent contest at stage {stage}"
                )
            best = sorted(
                distractors,
                key=lambda item: (-float(item["magnitude"]), str(item["action_name"])),
            )[0]
            lineage_rows.append(
                {
                    "distractor_action": str(best["action_name"]),
                    "distractor_magnitude": float(best["magnitude"]),
                    "lineage_seed": seed,
                    "progress_action": str(progress[0]["action_name"]),
                    "progress_magnitude": float(progress[0]["magnitude"]),
                    "shortfall_to_hard_contrast": (
                        float(progress[0]["magnitude"])
                        + protocol.minimum_distractor_magnitude_gap
                        - float(best["magnitude"])
                    ),
                }
            )
        per_stage[stage] = {
            "lineages": lineage_rows,
            "maximum_shortfall": max(
                float(item["shortfall_to_hard_contrast"]) for item in lineage_rows
            ),
        }
    minimum_shortfall = min(
        float(item["maximum_shortfall"]) for item in per_stage.values()
    )
    nearest_stages = tuple(
        stage
        for stage, item in sorted(per_stage.items())
        if float(item["maximum_shortfall"]) == minimum_shortfall
    )
    if len(nearest_stages) != 1:
        raise ValueError("T12.5b.3 parent no longer has one unique nearest contest")
    nearest = nearest_stages[0]
    if nearest != protocol.target_stage:
        raise ValueError("T12.5b.3 target stage is no longer the nearest contest")
    target = per_stage[protocol.target_stage]
    if any(
        item["progress_action"] != "ACTION3"
        or item["distractor_action"] != protocol.detour_action
        for item in target["lineages"]
    ):
        raise ValueError("T12.5b.3 stage-3 action contest changed")
    return {
        "criterion": (
            "minimum cross-lineage shortfall to the preregistered larger-"
            "nonprogress magnitude contrast"
        ),
        "parent_causal_score_used_for_selection": False,
        "stage": protocol.target_stage,
        "detour_action": protocol.detour_action,
        "lineages": target["lineages"],
        "unique_nearest_stage": nearest,
    }


def freeze_progress_contrast(
    *,
    output_path: str | Path,
    parent_manifest_path: str | Path,
    parent_receipt_path: str | Path,
    root: str | Path | None = None,
    protocol: ProgressContrastProtocol | None = None,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    selected = protocol or ProgressContrastProtocol()
    parent_manifest_path = Path(parent_manifest_path).resolve()
    parent_receipt_path = Path(parent_receipt_path).resolve()
    parent_manifest = load_progress_discrimination_manifest(
        parent_manifest_path, root=repo_root, verify_code=True
    )
    parent_receipt = load_progress_discrimination_receipt(
        parent_receipt_path,
        manifest=parent_manifest,
        root=repo_root,
        require_passed=False,
    )
    if parent_receipt.get("passed") is not False:
        raise ValueError("T12.5b.3 requires the sealed negative T12.5b.2 audit")
    if parent_receipt.get("status") != selected.parent_status:
        raise ValueError("T12.5b.3 parent status changed")
    metrics = dict(parent_receipt.get("metrics", {}))
    if metrics.get("classification") != selected.parent_classification:
        raise ValueError("T12.5b.3 parent classification changed")
    if metrics.get("collection_freeze_authorized") is not True:
        raise ValueError("T12.5b.3 collection freeze is not authorized")
    checks = dict(metrics.get("checks", {}))
    failed_checks = tuple(sorted(name for name, passed in checks.items() if not passed))
    expected_failures = (
        "causal_ranking_beats_magnitude_on_hard_contrasts",
        "causal_ranking_is_perfect_on_hard_contrasts",
        "hard_contrast_exists_in_every_lineage",
    )
    if failed_checks != expected_failures:
        raise ValueError("T12.5b.3 parent failure class changed")
    if parent_manifest.get("stage") != "source_train":
        raise ValueError("T12.5b.3 is restricted to source_train")

    parent_artifacts = {
        name: _verified_artifact(parent_receipt["artifacts"][name], root=repo_root)
        for name in ("affordances", "contrasts", "report")
    }
    affordance_payload = _read_json(
        _resolve_bound(parent_artifacts["affordances"]["path"], root=repo_root)
    )
    selection = _target_selection_summary(affordance_payload, protocol=selected)

    shadow_manifest_path = _resolve_bound(
        str(parent_manifest["parent"]["manifest"]["path"]), root=repo_root
    )
    shadow_manifest = load_progress_shadow_manifest(
        shadow_manifest_path, root=repo_root, verify_code=True
    )
    if shadow_manifest.get("game_id") != parent_manifest.get("game_id"):
        raise ValueError("T12.5b.3 physical parent game changed")
    physical_inputs = {
        name: _verified_artifact(shadow_manifest["inputs"][name], root=repo_root)
        for name in (
            "applicability_trials",
            "minimal_option",
            "posterior",
            "program_registry",
            "witness_registry",
        )
    }

    missing = [path for path in CONTRAST_CODE_PATHS if not (repo_root / path).is_file()]
    if missing:
        raise ValueError(f"T12.5b.3 code inventory is incomplete: {missing}")
    git = _git_state(repo_root)
    if git["dirty"] and not allow_dirty:
        raise ValueError("scientific freeze requires a clean worktree")
    authorized = bool(not git["dirty"])
    payload = {
        "format_version": CONTRAST_MANIFEST_FORMAT,
        "status": "FROZEN_BEFORE_T12_5B_3_PROSPECTIVE_CONTRAST_COLLECTION",
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
            },
            "shadow_manifest": {
                "path": _bound_path(shadow_manifest_path, root=repo_root),
                "sha256": _file_sha256(shadow_manifest_path),
                "manifest_checksum": shadow_manifest["manifest_checksum"],
            },
            "failed_checks": list(failed_checks),
        },
        "inputs": {
            **parent_artifacts,
            **physical_inputs,
            "successful_prefix_lengths": {
                str(seed): int(
                    shadow_manifest["inputs"]["successful_prefix_lengths"][str(seed)]
                )
                for seed in selected.lineage_seeds
            },
            "successful_anchor_hashes": {
                str(seed): str(
                    shadow_manifest["inputs"]["successful_anchor_hashes"][str(seed)]
                )
                for seed in selected.lineage_seeds
            },
        },
        "selection": selection,
        "design": {
            "all_detour_depths_fixed_before_collection": True,
            "all_candidate_branches_fixed_before_collection": True,
            "candidate_actions_are_local_sdk_affordances": True,
            "detour_must_match_no_sealed_milestone": True,
            "every_candidate_executes_from_one_exact_detour_context": True,
            "hard_contrast_label_uses_milestone_semantics_not_causal_score": True,
            "posterior_is_frozen_before_collection": True,
            "posterior_never_selects_executed_actions": True,
            "unavailable_action_is_missing_not_zero_effect": True,
            "cross_lineage_binding_fields": ["stage", "milestone_signature"],
            "action_name_is_provenance_only": True,
            "parent_negative_result_is_preserved": True,
        },
        "code_sha256": {
            path: _file_sha256(repo_root / path) for path in CONTRAST_CODE_PATHS
        },
        "git": git,
        "scientific_claims_authorized": authorized,
        "firewall": {
            "prospective_contrast_collection_authorized": authorized,
            "environment_collection_authorized": authorized,
            "causal_progress_control_authorized": False,
            "holdout_opened": False,
            "source_validation_opened": False,
            "production_authority": False,
            "terminal_shield_production_authority": False,
            "neural_training_authorized": False,
            "neural_active_evaluation_authorized": False,
            "option_control_authorized": False,
            "t12_5c_control_freeze_authorized": False,
            "t12_6_freeze_authorized": False,
        },
        "claim_boundary": {
            "authorized": (
                "source-train prospective same-prefix detour contrast collection"
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
            "maximum_artifact_bytes_per_run": (
                selected.maximum_artifact_bytes_per_run
            ),
            "maximum_sdk_calls": selected.maximum_sdk_calls,
            "maximum_wall_seconds": selected.maximum_wall_seconds,
            "persist_raw_frames": False,
            "hard_fail_before_write": True,
        },
    }
    manifest = _signed(payload, "manifest_checksum")
    _write_json_once(output_path, manifest)
    receipt = progress_contrast_receipt(
        manifest=manifest,
        phase="freeze",
        passed=authorized,
        status="PASS_T12_5B_3_FREEZE" if authorized else "DIRTY_SMOKE_ONLY",
        metrics={
            "expected_trials": selected.expected_trial_count,
            "maximum_sdk_calls": selected.maximum_sdk_calls,
            "maximum_wall_seconds": selected.maximum_wall_seconds,
            "parent_negative_result_preserved": True,
        },
    )
    _write_json_once(Path(output_path).with_name("freeze_receipt.json"), receipt)
    return manifest


def load_progress_contrast_manifest(
    path: str | Path,
    *,
    root: str | Path | None = None,
    verify_code: bool = True,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = _read_json(path)
    _verify_signed(manifest, "manifest_checksum")
    if manifest.get("format_version") != CONTRAST_MANIFEST_FORMAT:
        raise ValueError("unsupported T12.5b.3 progress-contrast manifest")
    protocol = ProgressContrastProtocol(**dict(manifest["protocol"]))
    if protocol.checksum != manifest.get("protocol_checksum"):
        raise ValueError("T12.5b.3 protocol checksum mismatch")
    for name, meta in manifest["parent"].items():
        if name == "failed_checks" or not isinstance(meta, Mapping):
            continue
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError(f"T12.5b.3 parent artifact mismatch: {name}")
    for name, meta in manifest["inputs"].items():
        if not isinstance(meta, Mapping) or "path" not in meta:
            continue
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError(f"T12.5b.3 input artifact mismatch: {name}")
    if verify_code:
        for relative, expected in manifest["code_sha256"].items():
            candidate = repo_root / relative
            if not candidate.is_file() or _file_sha256(candidate) != expected:
                raise ValueError(f"T12.5b.3 code checksum mismatch: {relative}")
    return manifest


def progress_contrast_receipt(
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
            "format_version": CONTRAST_RECEIPT_FORMAT,
            "phase": str(phase),
            "passed": bool(passed),
            "status": str(status),
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "parent_t12_5b_2_receipt_checksum": manifest["parent"]["receipt"][
                "receipt_checksum"
            ],
            "metrics": dict(metrics),
            "artifacts": dict(artifacts or {}),
        },
        "receipt_checksum",
    )


def load_progress_contrast_receipt(
    path: str | Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
    require_passed: bool = False,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    receipt = _read_json(path)
    _verify_signed(receipt, "receipt_checksum")
    if receipt.get("format_version") != CONTRAST_RECEIPT_FORMAT:
        raise ValueError("unsupported T12.5b.3 progress-contrast receipt")
    if manifest is not None and (
        receipt.get("manifest_checksum") != manifest.get("manifest_checksum")
        or receipt.get("protocol_checksum") != manifest.get("protocol_checksum")
    ):
        raise ValueError("T12.5b.3 receipt belongs to another manifest")
    if require_passed and receipt.get("passed") is not True:
        raise ValueError(f"T12.5b.3 gate failed: {receipt.get('status')}")
    for name, meta in receipt.get("artifacts", {}).items():
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError(f"T12.5b.3 receipt artifact mismatch: {name}")
    return receipt


__all__ = [
    "ProgressContrastProtocol",
    "freeze_progress_contrast",
    "load_progress_contrast_manifest",
    "load_progress_contrast_receipt",
    "progress_contrast_receipt",
]
