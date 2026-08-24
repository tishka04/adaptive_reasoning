"""Frozen offline protocol for SAGE.T12.6 future-viability grounding."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
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
from .goal_cursor_control_protocol import (
    GOAL_CURSOR_CONTROL_CODE_PATHS,
    load_goal_cursor_control_manifest,
    load_goal_cursor_control_receipt,
)
from .hazard_diversity_protocol import (
    HAZARD_DIVERSITY_CODE_PATHS,
    load_hazard_diversity_manifest,
    load_hazard_diversity_receipt,
)
from .target_regrounding_protocol import (
    TARGET_REGROUNDING_CODE_PATHS,
    load_target_regrounding_manifest,
    load_target_regrounding_receipt,
)

FUTURE_VIABILITY_PROTOCOL_FORMAT = "sage-t12.6-future-viability-protocol-v1"
FUTURE_VIABILITY_MANIFEST_FORMAT = "sage-t12.6-future-viability-manifest-v1"
FUTURE_VIABILITY_RECEIPT_FORMAT = "sage-t12.6-future-viability-receipt-v1"

FUTURE_VIABILITY_CODE_PATHS = tuple(
    dict.fromkeys(
        (
            *GOAL_CURSOR_CONTROL_CODE_PATHS,
            *TARGET_REGROUNDING_CODE_PATHS,
            *HAZARD_DIVERSITY_CODE_PATHS,
            "theory/sage_t/causal/future_viability.py",
            "theory/sage_t/causal/future_viability_protocol.py",
            "theory/sage_t/causal/future_viability_experiment.py",
            "theory/sage_t/causal/future_viability_cli.py",
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


def _verified_artifact(meta: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    path = _resolve_bound(str(meta["path"]), root=root)
    expected = str(meta["sha256"])
    if not path.is_file() or _file_sha256(path) != expected:
        raise ValueError(f"T12.6 input artifact mismatch: {path}")
    return {
        **{
            key: value
            for key, value in meta.items()
            if key not in {"path", "sha256"}
        },
        "path": _bound_path(path, root=root),
        "sha256": expected,
    }


def _artifact_meta(path: Path, *, root: Path, **extra: Any) -> dict[str, Any]:
    return {
        **extra,
        "path": _bound_path(path, root=root),
        "sha256": _file_sha256(path),
    }


@dataclass(frozen=True)
class FutureViabilityProtocol:
    """Immutable chronological train/evaluation split and offline gates."""

    format_version: str = FUTURE_VIABILITY_PROTOCOL_FORMAT
    authority_parent_status: str = "PASS_T12_5C_GOAL_CURSOR_CONTROL_GATE"
    training_parent_status: str = "FAIL_T12_4A_4D_TARGET_WITNESS_GATE"
    evaluation_parent_status: str = "FAIL_T12_4A_4D_1_HAZARD_DIVERSITY_GATE"
    source_lineages: tuple[int, ...] = (8_701, 8_705)
    training_search_seeds: tuple[int, ...] = (9_101, 9_102, 9_103)
    evaluation_search_seeds: tuple[int, ...] = (9_201, 9_202, 9_203)
    training_arms: tuple[str, ...] = (
        "local_archive_control",
        "contract_regrounded",
    )
    evaluation_arms: tuple[str, ...] = (
        "local_archive_control",
        "diversity_control",
        "abstract_hazard_diversity",
    )
    future_horizon: int = 4
    local_radius: int = 7
    minimum_signature_support: int = 2
    binding_shift: int = 1
    minimum_compile_eligible_groups: int = 240
    minimum_compile_top1_accuracy: float = 0.75
    minimum_compile_gain_over_immediate: float = 0.10
    minimum_compile_gain_over_binding_swap: float = 0.30
    minimum_compile_signature_coverage: float = 0.45
    minimum_compile_lineage_accuracy: float = 0.70
    minimum_evaluation_eligible_groups: int = 250
    minimum_evaluation_top1_accuracy: float = 0.70
    minimum_evaluation_gain_over_immediate: float = 0.08
    minimum_evaluation_gain_over_binding_swap: float = 0.25
    minimum_evaluation_signature_coverage: float = 0.40
    minimum_evaluation_lineage_accuracy: float = 0.65
    maximum_sdk_calls: int = 0
    maximum_wall_seconds_per_phase: int = 3_600
    maximum_artifact_bytes_per_phase: int = 3 * 1024 * 1024 * 1024
    persist_archive_copies: bool = False

    def __post_init__(self) -> None:
        for name in (
            "source_lineages",
            "training_search_seeds",
            "evaluation_search_seeds",
            "training_arms",
            "evaluation_arms",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        expected = {
            "format_version": FUTURE_VIABILITY_PROTOCOL_FORMAT,
            "authority_parent_status": "PASS_T12_5C_GOAL_CURSOR_CONTROL_GATE",
            "training_parent_status": "FAIL_T12_4A_4D_TARGET_WITNESS_GATE",
            "evaluation_parent_status": "FAIL_T12_4A_4D_1_HAZARD_DIVERSITY_GATE",
            "source_lineages": (8_701, 8_705),
            "training_search_seeds": (9_101, 9_102, 9_103),
            "evaluation_search_seeds": (9_201, 9_202, 9_203),
            "training_arms": ("local_archive_control", "contract_regrounded"),
            "evaluation_arms": (
                "local_archive_control",
                "diversity_control",
                "abstract_hazard_diversity",
            ),
            "future_horizon": 4,
            "local_radius": 7,
            "minimum_signature_support": 2,
            "binding_shift": 1,
            "minimum_compile_eligible_groups": 240,
            "minimum_compile_top1_accuracy": 0.75,
            "minimum_compile_gain_over_immediate": 0.10,
            "minimum_compile_gain_over_binding_swap": 0.30,
            "minimum_compile_signature_coverage": 0.45,
            "minimum_compile_lineage_accuracy": 0.70,
            "minimum_evaluation_eligible_groups": 250,
            "minimum_evaluation_top1_accuracy": 0.70,
            "minimum_evaluation_gain_over_immediate": 0.08,
            "minimum_evaluation_gain_over_binding_swap": 0.25,
            "minimum_evaluation_signature_coverage": 0.40,
            "minimum_evaluation_lineage_accuracy": 0.65,
            "maximum_sdk_calls": 0,
            "maximum_wall_seconds_per_phase": 3_600,
            "maximum_artifact_bytes_per_phase": 3 * 1024 * 1024 * 1024,
            "persist_archive_copies": False,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"T12.6 preregistered value changed: {name}")
        if set(self.training_search_seeds) & set(self.evaluation_search_seeds):
            raise ValueError("T12.6 chronological corpora must be disjoint")

    @property
    def checksum(self) -> str:
        return _checksum(asdict(self))


def _require_integrity_checks(
    receipt: Mapping[str, Any],
    *,
    names: Iterable[str],
    parent_name: str,
) -> None:
    checks = dict(receipt.get("metrics", {}).get("checks", {}))
    missing = [name for name in names if checks.get(name) is not True]
    if missing:
        raise ValueError(f"T12.6 {parent_name} integrity checks failed: {missing}")


def freeze_future_viability(
    *,
    output_path: str | Path,
    authority_manifest_path: str | Path,
    authority_receipt_path: str | Path,
    training_manifest_path: str | Path,
    training_receipt_path: str | Path,
    evaluation_manifest_path: str | Path,
    evaluation_receipt_path: str | Path,
    root: str | Path | None = None,
    protocol: FutureViabilityProtocol | None = None,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    selected = protocol or FutureViabilityProtocol()
    authority_manifest_path = Path(authority_manifest_path).resolve()
    authority_receipt_path = Path(authority_receipt_path).resolve()
    training_manifest_path = Path(training_manifest_path).resolve()
    training_receipt_path = Path(training_receipt_path).resolve()
    evaluation_manifest_path = Path(evaluation_manifest_path).resolve()
    evaluation_receipt_path = Path(evaluation_receipt_path).resolve()

    authority_manifest = load_goal_cursor_control_manifest(
        authority_manifest_path, root=repo_root, verify_code=True
    )
    authority_receipt = load_goal_cursor_control_receipt(
        authority_receipt_path,
        manifest=authority_manifest,
        root=repo_root,
        require_passed=True,
        expected_phase="paired_control",
    )
    if authority_receipt.get("status") != selected.authority_parent_status:
        raise ValueError("T12.6 authority parent status changed")
    if authority_receipt.get("metrics", {}).get("t12_6_freeze_authorized") is not True:
        raise ValueError("T12.6 authority parent did not authorize this freeze")

    training_manifest = load_target_regrounding_manifest(
        training_manifest_path, root=repo_root, verify_code=False
    )
    training_receipt = load_target_regrounding_receipt(
        training_receipt_path,
        manifest=training_manifest,
        root=repo_root,
    )
    if training_receipt.get("status") != selected.training_parent_status:
        raise ValueError("T12.6 training parent status changed")
    if training_receipt.get("passed") is not False:
        raise ValueError("T12.6 training corpus must preserve its negative result")
    _require_integrity_checks(
        training_receipt,
        names=(
            "all_anchor_replays_exact",
            "all_archive_replays_exact",
            "contracted_option_blocked_at_every_anchor",
            "paired_candidate_catalogs_identical",
            "sdk_budget_respected",
        ),
        parent_name="training",
    )

    evaluation_manifest = load_hazard_diversity_manifest(
        evaluation_manifest_path, root=repo_root, verify_code=False
    )
    evaluation_receipt = load_hazard_diversity_receipt(
        evaluation_receipt_path,
        manifest=evaluation_manifest,
        root=repo_root,
    )
    if evaluation_receipt.get("status") != selected.evaluation_parent_status:
        raise ValueError("T12.6 evaluation parent status changed")
    if evaluation_receipt.get("passed") is not False:
        raise ValueError("T12.6 evaluation corpus must preserve its negative result")
    _require_integrity_checks(
        evaluation_receipt,
        names=(
            "all_anchor_replays_exact",
            "all_archive_replays_exact",
            "contracted_option_blocked_at_every_anchor",
            "diversity_arms_do_not_collapse",
            "paired_candidate_catalogs_identical",
            "sdk_budget_respected",
        ),
        parent_name="evaluation",
    )

    training_archives = [
        _verified_artifact(meta, root=repo_root)
        for meta in training_receipt["artifacts"]["archives"]
    ]
    evaluation_archives = [
        _verified_artifact(meta, root=repo_root)
        for meta in evaluation_receipt["artifacts"]["archives"]
    ]
    expected_training = (
        len(selected.training_search_seeds)
        * len(selected.source_lineages)
        * len(selected.training_arms)
    )
    expected_evaluation = (
        len(selected.evaluation_search_seeds)
        * len(selected.source_lineages)
        * len(selected.evaluation_arms)
    )
    if len(training_archives) != expected_training:
        raise ValueError("T12.6 training archive count changed")
    if len(evaluation_archives) != expected_evaluation:
        raise ValueError("T12.6 evaluation archive count changed")

    missing = [
        path for path in FUTURE_VIABILITY_CODE_PATHS if not (repo_root / path).is_file()
    ]
    if missing:
        raise ValueError(f"T12.6 code inventory is incomplete: {missing}")
    git = _git_state(repo_root)
    if git["dirty"] and not allow_dirty:
        raise ValueError("scientific freeze requires a clean worktree")
    authorized = bool(not git["dirty"])

    def parent(
        manifest_path: Path,
        manifest: Mapping[str, Any],
        receipt_path: Path,
        receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "manifest": _artifact_meta(
                manifest_path,
                root=repo_root,
                manifest_checksum=manifest["manifest_checksum"],
            ),
            "receipt": _artifact_meta(
                receipt_path,
                root=repo_root,
                receipt_checksum=receipt["receipt_checksum"],
                status=receipt["status"],
            ),
        }

    payload = {
        "format_version": FUTURE_VIABILITY_MANIFEST_FORMAT,
        "status": "FROZEN_BEFORE_T12_6_OFFLINE_COMPILE",
        "stage": "source_train_offline",
        "game_id": authority_manifest["game_id"],
        "protocol": asdict(selected),
        "protocol_checksum": selected.checksum,
        "authority_parent": parent(
            authority_manifest_path,
            authority_manifest,
            authority_receipt_path,
            authority_receipt,
        ),
        "training_parent": parent(
            training_manifest_path,
            training_manifest,
            training_receipt_path,
            training_receipt,
        ),
        "evaluation_parent": parent(
            evaluation_manifest_path,
            evaluation_manifest,
            evaluation_receipt_path,
            evaluation_receipt,
        ),
        "inputs": {
            "training_archives": training_archives,
            "evaluation_archives": evaluation_archives,
        },
        "design": {
            "archive_outcomes_are_never_relabelled_as_level_progress": True,
            "binding_swap_preserves_score_multiset": True,
            "chronological_training_evaluation_split": True,
            "evaluation_scores_are_frozen_before_evaluation": True,
            "future_label_excludes_immediate_edge": True,
            "historical_code_is_not_reexecuted": True,
            "historical_receipts_and_artifacts_are_hash_verified": True,
            "identity_and_absolute_coordinates_excluded_from_signature": True,
            "no_sdk_calls": True,
            "parent_negative_results_preserved": True,
        },
        "code_sha256": {
            path: _file_sha256(repo_root / path)
            for path in FUTURE_VIABILITY_CODE_PATHS
        },
        "git": git,
        "scientific_claims_authorized": authorized,
        "firewall": {
            "compile_authorized": authorized,
            "evaluation_authorized": False,
            "environment_collection_authorized": False,
            "source_validation_opened": False,
            "holdout_opened": False,
            "controller_authority": False,
            "neural_training_authorized": False,
            "production_authority": False,
            "t12_6b_physical_freeze_authorized": False,
        },
        "claim_boundary": {
            "authorized": (
                "offline temporal transfer of target-local future productive-reach "
                "ranking inside sealed bp35 level-1 archives"
            ),
            "not_authorized": [
                "level progress",
                "fresh physical control",
                "generic ARC-AGI improvement",
                "source validation",
                "holdout performance",
                "autonomous controller authority",
                "neural training",
                "production authority",
            ],
        },
        "storage": {
            "maximum_artifact_bytes_per_phase": (
                selected.maximum_artifact_bytes_per_phase
            ),
            "maximum_sdk_calls": 0,
            "maximum_wall_seconds_per_phase": selected.maximum_wall_seconds_per_phase,
            "persist_archive_copies": False,
            "hard_fail_before_write": True,
        },
    }
    manifest = _signed(payload, "manifest_checksum")
    _write_json_once(output_path, manifest)
    receipt = future_viability_receipt(
        manifest=manifest,
        phase="freeze",
        passed=authorized,
        status="PASS_T12_6_FREEZE" if authorized else "DIRTY_SMOKE_ONLY",
        metrics={
            "evaluation_archive_count": len(evaluation_archives),
            "maximum_sdk_calls": 0,
            "training_archive_count": len(training_archives),
        },
    )
    _write_json_once(Path(output_path).with_name("freeze_receipt.json"), receipt)
    return manifest


def _iter_bound_metas(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if "path" in value and "sha256" in value:
            yield value
        else:
            for nested in value.values():
                yield from _iter_bound_metas(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _iter_bound_metas(nested)


def load_future_viability_manifest(
    path: str | Path,
    *,
    root: str | Path | None = None,
    verify_code: bool = True,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = _read_json(path)
    _verify_signed(manifest, "manifest_checksum")
    if manifest.get("format_version") != FUTURE_VIABILITY_MANIFEST_FORMAT:
        raise ValueError("unsupported T12.6 future-viability manifest")
    protocol = FutureViabilityProtocol(**dict(manifest["protocol"]))
    if protocol.checksum != manifest.get("protocol_checksum"):
        raise ValueError("T12.6 protocol checksum mismatch")
    for meta in _iter_bound_metas(
        {
            "authority_parent": manifest["authority_parent"],
            "training_parent": manifest["training_parent"],
            "evaluation_parent": manifest["evaluation_parent"],
            "inputs": manifest["inputs"],
        }
    ):
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError(f"T12.6 bound artifact mismatch: {candidate}")
    if verify_code:
        for relative, expected in manifest["code_sha256"].items():
            candidate = repo_root / relative
            if not candidate.is_file() or _file_sha256(candidate) != expected:
                raise ValueError(f"T12.6 code checksum mismatch: {relative}")
    return manifest


def future_viability_receipt(
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
            "format_version": FUTURE_VIABILITY_RECEIPT_FORMAT,
            "phase": str(phase),
            "passed": bool(passed),
            "status": str(status),
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "authority_t12_5c_receipt_checksum": manifest["authority_parent"][
                "receipt"
            ]["receipt_checksum"],
            "metrics": dict(metrics),
            "artifacts": dict(artifacts or {}),
        },
        "receipt_checksum",
    )


def load_future_viability_receipt(
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
    if receipt.get("format_version") != FUTURE_VIABILITY_RECEIPT_FORMAT:
        raise ValueError("unsupported T12.6 future-viability receipt")
    if manifest is not None and (
        receipt.get("manifest_checksum") != manifest.get("manifest_checksum")
        or receipt.get("protocol_checksum") != manifest.get("protocol_checksum")
    ):
        raise ValueError("T12.6 receipt belongs to another manifest")
    if expected_phase is not None and receipt.get("phase") != expected_phase:
        raise ValueError("T12.6 receipt phase mismatch")
    if require_passed and receipt.get("passed") is not True:
        raise ValueError(f"T12.6 gate failed: {receipt.get('status')}")
    for name, meta in receipt.get("artifacts", {}).items():
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError(f"T12.6 receipt artifact mismatch: {name}")
    return receipt


__all__ = [
    "FUTURE_VIABILITY_MANIFEST_FORMAT",
    "FUTURE_VIABILITY_PROTOCOL_FORMAT",
    "FUTURE_VIABILITY_RECEIPT_FORMAT",
    "FutureViabilityProtocol",
    "freeze_future_viability",
    "future_viability_receipt",
    "load_future_viability_manifest",
    "load_future_viability_receipt",
]
