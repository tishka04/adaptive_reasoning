"""Frozen T12.6.1 protocol for hierarchical future-viability transfer."""

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
from .future_viability_diagnostic_protocol import (
    _load_parent_manifest_for_diagnostic,
    load_future_viability_diagnostic_manifest,
    load_future_viability_diagnostic_receipt,
)
from .future_viability_protocol import (
    FutureViabilityProtocol,
    _resolve_bound,
    load_future_viability_receipt,
)

HIERARCHY_PROTOCOL_FORMAT = "sage-t12.6.1-future-viability-hierarchy-protocol-v1"
HIERARCHY_MANIFEST_FORMAT = "sage-t12.6.1-future-viability-hierarchy-manifest-v1"
HIERARCHY_RECEIPT_FORMAT = "sage-t12.6.1-future-viability-hierarchy-receipt-v1"

HIERARCHY_CODE_PATHS = (
    "theory/sage_t/causal/experiment.py",
    "theory/sage_t/causal/archive.py",
    "theory/sage_t/causal/future_viability.py",
    "theory/sage_t/causal/future_viability_protocol.py",
    "theory/sage_t/causal/future_viability_diagnostic_protocol.py",
    "theory/sage_t/causal/hazard_diversity_model.py",
    "theory/sage_t/causal/future_viability_hierarchy.py",
    "theory/sage_t/causal/future_viability_hierarchy_protocol.py",
    "theory/sage_t/causal/future_viability_hierarchy_experiment.py",
    "theory/sage_t/causal/future_viability_hierarchy_cli.py",
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


def _artifact_meta(path: Path, *, root: Path, **extra: Any) -> dict[str, Any]:
    return {
        **extra,
        "path": _bound_path(path, root=root),
        "sha256": _file_sha256(path),
    }


def _verify_meta(meta: Mapping[str, Any], *, root: Path) -> Path:
    path = _resolve_bound(str(meta["path"]), root=root)
    if not path.is_file() or _file_sha256(path) != str(meta["sha256"]):
        raise ValueError(f"T12.6.1 bound artifact mismatch: {path}")
    return path


@dataclass(frozen=True)
class FutureViabilityHierarchyProtocol:
    """Immutable development and sealed-evaluation gates for T12.6.1."""

    format_version: str = HIERARCHY_PROTOCOL_FORMAT
    parent_compile_status: str = "FAIL_T12_6_FUTURE_VIABILITY_IDENTIFICATION_GATE"
    diagnostic_status: str = "PASS_T12_6A_DIAGNOSTIC_COMPLETE"
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
    minimum_compile_gain_over_immediate: float = 0.15
    minimum_compile_gain_over_binding_swap: float = 0.30
    minimum_compile_hierarchy_coverage: float = 0.80
    minimum_compile_coverage_gain_over_incumbent: float = 0.25
    minimum_compile_unique_top_rate: float = 0.90
    minimum_compile_recommendation_coverage: float = 0.75
    minimum_compile_lineage_fold_accuracy: float = 0.75
    minimum_compile_gain_over_incumbent: float = -0.02
    minimum_compile_worst_fold_gain_over_incumbent: float = 0.04
    minimum_evaluation_eligible_groups: int = 250
    minimum_evaluation_top1_accuracy: float = 0.70
    minimum_evaluation_gain_over_immediate: float = 0.10
    minimum_evaluation_gain_over_binding_swap: float = 0.25
    minimum_evaluation_gain_over_incumbent: float = 0.02
    minimum_evaluation_hierarchy_coverage: float = 0.70
    minimum_evaluation_coverage_gain_over_incumbent: float = 0.15
    minimum_evaluation_unique_top_rate: float = 0.85
    minimum_evaluation_recommendation_coverage: float = 0.60
    minimum_evaluation_lineage_accuracy: float = 0.65
    minimum_evaluation_worst_seed_gain_over_incumbent: float = 0.0
    minimum_evaluation_seed_wins_over_incumbent: int = 2
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
            "format_version": HIERARCHY_PROTOCOL_FORMAT,
            "parent_compile_status": "FAIL_T12_6_FUTURE_VIABILITY_IDENTIFICATION_GATE",
            "diagnostic_status": "PASS_T12_6A_DIAGNOSTIC_COMPLETE",
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
            "minimum_compile_gain_over_immediate": 0.15,
            "minimum_compile_gain_over_binding_swap": 0.30,
            "minimum_compile_hierarchy_coverage": 0.80,
            "minimum_compile_coverage_gain_over_incumbent": 0.25,
            "minimum_compile_unique_top_rate": 0.90,
            "minimum_compile_recommendation_coverage": 0.75,
            "minimum_compile_lineage_fold_accuracy": 0.75,
            "minimum_compile_gain_over_incumbent": -0.02,
            "minimum_compile_worst_fold_gain_over_incumbent": 0.04,
            "minimum_evaluation_eligible_groups": 250,
            "minimum_evaluation_top1_accuracy": 0.70,
            "minimum_evaluation_gain_over_immediate": 0.10,
            "minimum_evaluation_gain_over_binding_swap": 0.25,
            "minimum_evaluation_gain_over_incumbent": 0.02,
            "minimum_evaluation_hierarchy_coverage": 0.70,
            "minimum_evaluation_coverage_gain_over_incumbent": 0.15,
            "minimum_evaluation_unique_top_rate": 0.85,
            "minimum_evaluation_recommendation_coverage": 0.60,
            "minimum_evaluation_lineage_accuracy": 0.65,
            "minimum_evaluation_worst_seed_gain_over_incumbent": 0.0,
            "minimum_evaluation_seed_wins_over_incumbent": 2,
            "maximum_sdk_calls": 0,
            "maximum_wall_seconds_per_phase": 3_600,
            "maximum_artifact_bytes_per_phase": 3 * 1024 * 1024 * 1024,
            "persist_archive_copies": False,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"T12.6.1 preregistered value changed: {name}")
        if set(self.training_search_seeds) & set(self.evaluation_search_seeds):
            raise ValueError("T12.6.1 chronological corpora must be disjoint")

    @property
    def checksum(self) -> str:
        return _checksum(asdict(self))


def freeze_future_viability_hierarchy(
    *,
    output_path: str | Path,
    parent_manifest_path: str | Path,
    parent_compile_receipt_path: str | Path,
    diagnostic_manifest_path: str | Path,
    diagnostic_receipt_path: str | Path,
    root: str | Path | None = None,
    protocol: FutureViabilityHierarchyProtocol | None = None,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    selected = protocol or FutureViabilityHierarchyProtocol()
    parent_manifest_path = Path(parent_manifest_path).resolve()
    parent_compile_receipt_path = Path(parent_compile_receipt_path).resolve()
    diagnostic_manifest_path = Path(diagnostic_manifest_path).resolve()
    diagnostic_receipt_path = Path(diagnostic_receipt_path).resolve()
    parent = _load_parent_manifest_for_diagnostic(
        parent_manifest_path, root=repo_root
    )
    parent_protocol = FutureViabilityProtocol(**dict(parent["protocol"]))
    parent_receipt = load_future_viability_receipt(
        parent_compile_receipt_path,
        manifest=parent,
        root=repo_root,
        expected_phase="compile",
    )
    if parent_receipt.get("status") != selected.parent_compile_status:
        raise ValueError("T12.6.1 parent compile status changed")
    if parent_receipt.get("passed") is not False:
        raise ValueError("T12.6.1 requires the preserved negative parent")
    failed = {
        name
        for name, value in parent_receipt["metrics"]["checks"].items()
        if value is not True
    }
    if failed != {"every_compile_lineage_accuracy_sufficient"}:
        raise ValueError("T12.6.1 parent failure set changed")
    diagnostic_manifest = load_future_viability_diagnostic_manifest(
        diagnostic_manifest_path, root=repo_root
    )
    diagnostic_receipt = load_future_viability_diagnostic_receipt(
        diagnostic_receipt_path,
        manifest=diagnostic_manifest,
        root=repo_root,
        expected_phase="diagnostic",
    )
    if (
        diagnostic_receipt.get("status") != selected.diagnostic_status
        or diagnostic_receipt.get("passed") is not True
    ):
        raise ValueError("T12.6.1 diagnostic parent is not complete")
    if diagnostic_receipt.get("parent_t12_6_compile_receipt_checksum") != (
        parent_receipt["receipt_checksum"]
    ):
        raise ValueError("T12.6.1 diagnostic belongs to another parent")

    training_archives = []
    for meta in parent["inputs"]["training_archives"]:
        path = _verify_meta(meta, root=repo_root)
        training_archives.append(
            {
                **{
                    key: value
                    for key, value in meta.items()
                    if key not in {"path", "sha256"}
                },
                "path": _bound_path(path, root=repo_root),
                "sha256": str(meta["sha256"]),
            }
        )
    evaluation_archives = [dict(meta) for meta in parent["inputs"]["evaluation_archives"]]
    if len(training_archives) != 12 or len(evaluation_archives) != 18:
        raise ValueError("T12.6.1 archive registry changed")
    if (
        tuple(parent_protocol.training_search_seeds) != selected.training_search_seeds
        or tuple(parent_protocol.evaluation_search_seeds)
        != selected.evaluation_search_seeds
    ):
        raise ValueError("T12.6.1 parent split changed")
    missing_code = [
        path for path in HIERARCHY_CODE_PATHS if not (repo_root / path).is_file()
    ]
    if missing_code:
        raise ValueError(f"T12.6.1 code inventory incomplete: {missing_code}")
    git = _git_state(repo_root)
    if git["dirty"] and not allow_dirty:
        raise ValueError("scientific freeze requires a clean worktree")
    authorized = bool(not git["dirty"])

    payload = {
        "claim_boundary": {
            "authorized": (
                "chronological transfer of a frozen exact-to-composition "
                "future-viability hierarchy inside sealed bp35 level-1 archives"
            ),
            "not_authorized": [
                "level progress",
                "physical collection",
                "generic ARC-AGI improvement",
                "source validation",
                "holdout performance beyond registered archives",
                "controller authority",
                "neural training",
                "production authority",
            ],
        },
        "code_sha256": {
            path: _file_sha256(repo_root / path) for path in HIERARCHY_CODE_PATHS
        },
        "design": {
            "chronological_training_evaluation_split": True,
            "composition_discards_offsets_but_preserves_typed_local_multiset": True,
            "development_is_posthoc_to_t12_6": True,
            "evaluation_scores_frozen_before_evaluation": True,
            "exact_signature_precedes_composition_precedes_family": True,
            "future_label_unchanged_from_t12_6": True,
            "identity_and_absolute_coordinates_excluded": True,
            "incumbent_and_equal_capacity_controls_frozen": True,
            "no_sdk_calls": True,
        },
        "firewall": {
            "compile_authorized": authorized,
            "evaluation_authorized": False,
            "environment_collection_authorized": False,
            "source_validation_opened": False,
            "holdout_opened": False,
            "controller_authority": False,
            "neural_training_authorized": False,
            "production_authority": False,
            "t12_6_2_freeze_authorized": False,
        },
        "format_version": HIERARCHY_MANIFEST_FORMAT,
        "game_id": parent["game_id"],
        "git": git,
        "inputs": {
            "training_archives": training_archives,
            "evaluation_archives": evaluation_archives,
        },
        "parents": {
            "t12_6_manifest": _artifact_meta(
                parent_manifest_path,
                root=repo_root,
                manifest_checksum=parent["manifest_checksum"],
            ),
            "t12_6_compile_receipt": _artifact_meta(
                parent_compile_receipt_path,
                root=repo_root,
                receipt_checksum=parent_receipt["receipt_checksum"],
                status=parent_receipt["status"],
            ),
            "t12_6_compile_artifacts": {
                name: dict(meta) for name, meta in parent_receipt["artifacts"].items()
            },
            "t12_6a_manifest": _artifact_meta(
                diagnostic_manifest_path,
                root=repo_root,
                manifest_checksum=diagnostic_manifest["manifest_checksum"],
            ),
            "t12_6a_receipt": _artifact_meta(
                diagnostic_receipt_path,
                root=repo_root,
                receipt_checksum=diagnostic_receipt["receipt_checksum"],
                status=diagnostic_receipt["status"],
            ),
        },
        "protocol": asdict(selected),
        "protocol_checksum": selected.checksum,
        "scientific_claims_authorized": authorized,
        "stage": "source_train_then_sealed_evaluation",
        "status": "FROZEN_BEFORE_T12_6_1_OFFLINE_COMPILE",
        "storage": {
            "maximum_artifact_bytes_per_phase": selected.maximum_artifact_bytes_per_phase,
            "maximum_sdk_calls": 0,
            "maximum_wall_seconds_per_phase": selected.maximum_wall_seconds_per_phase,
            "persist_archive_copies": False,
        },
    }
    manifest = _signed(payload, "manifest_checksum")
    _write_json_once(output_path, manifest)
    receipt = hierarchy_receipt(
        manifest=manifest,
        phase="freeze",
        passed=authorized,
        status="PASS_T12_6_1_FREEZE" if authorized else "DIRTY_SMOKE_ONLY",
        metrics={
            "evaluation_archive_count": len(evaluation_archives),
            "evaluation_archive_payloads_loaded": 0,
            "training_archive_count": len(training_archives),
        },
    )
    _write_json_once(Path(output_path).with_name("freeze_receipt.json"), receipt)
    return manifest


def load_future_viability_hierarchy_manifest(
    path: str | Path,
    *,
    root: str | Path | None = None,
    verify_code: bool = True,
    open_evaluation: bool = False,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = _read_json(path)
    _verify_signed(manifest, "manifest_checksum")
    if manifest.get("format_version") != HIERARCHY_MANIFEST_FORMAT:
        raise ValueError("unsupported T12.6.1 hierarchy manifest")
    protocol = FutureViabilityHierarchyProtocol(**dict(manifest["protocol"]))
    if protocol.checksum != manifest.get("protocol_checksum"):
        raise ValueError("T12.6.1 hierarchy protocol checksum mismatch")
    for meta in manifest["inputs"]["training_archives"]:
        _verify_meta(meta, root=repo_root)
    for name, meta in manifest["parents"].items():
        if name == "t12_6_compile_artifacts":
            for nested in meta.values():
                _verify_meta(nested, root=repo_root)
        else:
            _verify_meta(meta, root=repo_root)
    if open_evaluation:
        for meta in manifest["inputs"]["evaluation_archives"]:
            _verify_meta(meta, root=repo_root)
    if verify_code:
        for relative, expected in manifest["code_sha256"].items():
            candidate = repo_root / relative
            if not candidate.is_file() or _file_sha256(candidate) != expected:
                raise ValueError(f"T12.6.1 code checksum mismatch: {relative}")
    return manifest


def hierarchy_receipt(
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
            "artifacts": dict(artifacts or {}),
            "format_version": HIERARCHY_RECEIPT_FORMAT,
            "manifest_checksum": manifest["manifest_checksum"],
            "metrics": dict(metrics),
            "parent_t12_6_compile_receipt_checksum": manifest["parents"][
                "t12_6_compile_receipt"
            ]["receipt_checksum"],
            "parent_t12_6a_receipt_checksum": manifest["parents"][
                "t12_6a_receipt"
            ]["receipt_checksum"],
            "passed": bool(passed),
            "phase": str(phase),
            "protocol_checksum": manifest["protocol_checksum"],
            "status": str(status),
        },
        "receipt_checksum",
    )


def load_hierarchy_receipt(
    path: str | Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
    expected_phase: str | None = None,
    require_passed: bool = False,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    receipt = _read_json(path)
    _verify_signed(receipt, "receipt_checksum")
    if receipt.get("format_version") != HIERARCHY_RECEIPT_FORMAT:
        raise ValueError("unsupported T12.6.1 hierarchy receipt")
    if manifest is not None and (
        receipt.get("manifest_checksum") != manifest.get("manifest_checksum")
        or receipt.get("protocol_checksum") != manifest.get("protocol_checksum")
    ):
        raise ValueError("T12.6.1 receipt belongs to another manifest")
    if expected_phase is not None and receipt.get("phase") != expected_phase:
        raise ValueError("T12.6.1 receipt phase mismatch")
    if require_passed and receipt.get("passed") is not True:
        raise ValueError(f"T12.6.1 gate failed: {receipt.get('status')}")
    for meta in receipt.get("artifacts", {}).values():
        _verify_meta(meta, root=repo_root)
    return receipt


__all__ = [
    "HIERARCHY_CODE_PATHS",
    "HIERARCHY_MANIFEST_FORMAT",
    "HIERARCHY_PROTOCOL_FORMAT",
    "HIERARCHY_RECEIPT_FORMAT",
    "FutureViabilityHierarchyProtocol",
    "freeze_future_viability_hierarchy",
    "hierarchy_receipt",
    "load_future_viability_hierarchy_manifest",
    "load_hierarchy_receipt",
]
