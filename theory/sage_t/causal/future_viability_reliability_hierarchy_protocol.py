"""Frozen source-train protocol for SAGE.T12.6.1c."""

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
from .future_viability_hierarchy_protocol import (
    FutureViabilityHierarchyProtocol,
    _verify_meta,
    load_future_viability_hierarchy_manifest,
    load_hierarchy_receipt,
)
from .future_viability_reliability_hierarchy import RELIABILITY_CANDIDATES
from .future_viability_seed_shift_diagnostic_protocol import (
    load_seed_shift_diagnostic_receipt,
)

RELIABILITY_PROTOCOL_FORMAT = (
    "sage-t12.6.1c-future-viability-reliability-protocol-v1"
)
RELIABILITY_MANIFEST_FORMAT = (
    "sage-t12.6.1c-future-viability-reliability-manifest-v1"
)
RELIABILITY_RECEIPT_FORMAT = (
    "sage-t12.6.1c-future-viability-reliability-receipt-v1"
)

RELIABILITY_CODE_PATHS = (
    "theory/sage_t/causal/experiment.py",
    "theory/sage_t/causal/archive.py",
    "theory/sage_t/causal/future_viability.py",
    "theory/sage_t/causal/hazard_diversity_model.py",
    "theory/sage_t/causal/future_viability_hierarchy.py",
    "theory/sage_t/causal/future_viability_hierarchy_protocol.py",
    "theory/sage_t/causal/future_viability_hierarchy_experiment.py",
    "theory/sage_t/causal/future_viability_seed_shift_diagnostic.py",
    "theory/sage_t/causal/future_viability_seed_shift_diagnostic_protocol.py",
    "theory/sage_t/causal/future_viability_reliability_hierarchy.py",
    "theory/sage_t/causal/future_viability_reliability_hierarchy_protocol.py",
    "theory/sage_t/causal/future_viability_reliability_hierarchy_experiment.py",
    "theory/sage_t/causal/future_viability_reliability_hierarchy_cli.py",
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


@dataclass(frozen=True)
class FutureViabilityReliabilityProtocol:
    """Immutable source-train selection rule and fail-closed compile gates."""

    format_version: str = RELIABILITY_PROTOCOL_FORMAT
    hierarchy_compile_status: str = "PASS_T12_6_1_COMPILE_GATE"
    seed_shift_diagnostic_status: str = "PASS_T12_6_1B_DIAGNOSTIC_COMPLETE"
    seed_shift_classification: str = (
        "POSTHOC_9202_HETEROGENEOUS_EXACT_SIGNATURE_MISRANKING_DOMINANT"
    )
    training_search_seeds: tuple[int, ...] = (9_101, 9_102, 9_103)
    source_lineages: tuple[int, ...] = (8_701, 8_705)
    training_arms: tuple[str, ...] = (
        "local_archive_control",
        "contract_regrounded",
    )
    future_horizon: int = 4
    local_radius: int = 7
    minimum_signature_support: int = 2
    reliability_candidates: tuple[str, ...] = (
        "exact_span2_range0",
        "exact_span2_range1",
        "exact_span2_range2",
    )
    candidate_selection_criterion: str = (
        "lexicographic(worst_fold_accuracy,micro_accuracy,"
        "gain_over_exact_first,coverage,frozen_candidate_precedence)"
    )
    binding_shift: int = 1
    minimum_compile_eligible_groups: int = 240
    minimum_compile_top1_accuracy: float = 0.75
    minimum_compile_worst_fold_accuracy: float = 0.75
    minimum_compile_lineage_fold_accuracy: float = 0.75
    minimum_compile_gain_over_immediate: float = 0.10
    minimum_compile_gain_over_binding_swap: float = 0.30
    minimum_compile_gain_over_exact_first: float = 0.0
    minimum_compile_hierarchy_coverage: float = 0.80
    minimum_compile_unique_top_rate: float = 0.90
    minimum_compile_recommendation_coverage: float = 0.75
    minimum_exact_rejection_exercised_rate: float = 0.50
    minimum_full_fit_reliable_exact_supports: int = 1
    maximum_sdk_calls: int = 0
    maximum_wall_seconds: int = 600
    maximum_artifact_bytes: int = 512 * 1024 * 1024
    persist_archive_copies: bool = False
    evaluation_archive_payloads_authorized: bool = False
    confirmatory_claim_authorized: bool = False
    physical_collection_authorized: bool = False

    def __post_init__(self) -> None:
        for name in (
            "training_search_seeds",
            "source_lineages",
            "training_arms",
            "reliability_candidates",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        expected = {
            "format_version": RELIABILITY_PROTOCOL_FORMAT,
            "hierarchy_compile_status": "PASS_T12_6_1_COMPILE_GATE",
            "seed_shift_diagnostic_status": "PASS_T12_6_1B_DIAGNOSTIC_COMPLETE",
            "seed_shift_classification": (
                "POSTHOC_9202_HETEROGENEOUS_EXACT_SIGNATURE_MISRANKING_DOMINANT"
            ),
            "training_search_seeds": (9_101, 9_102, 9_103),
            "source_lineages": (8_701, 8_705),
            "training_arms": ("local_archive_control", "contract_regrounded"),
            "future_horizon": 4,
            "local_radius": 7,
            "minimum_signature_support": 2,
            "reliability_candidates": (
                "exact_span2_range0",
                "exact_span2_range1",
                "exact_span2_range2",
            ),
            "candidate_selection_criterion": (
                "lexicographic(worst_fold_accuracy,micro_accuracy,"
                "gain_over_exact_first,coverage,frozen_candidate_precedence)"
            ),
            "binding_shift": 1,
            "minimum_compile_eligible_groups": 240,
            "minimum_compile_top1_accuracy": 0.75,
            "minimum_compile_worst_fold_accuracy": 0.75,
            "minimum_compile_lineage_fold_accuracy": 0.75,
            "minimum_compile_gain_over_immediate": 0.10,
            "minimum_compile_gain_over_binding_swap": 0.30,
            "minimum_compile_gain_over_exact_first": 0.0,
            "minimum_compile_hierarchy_coverage": 0.80,
            "minimum_compile_unique_top_rate": 0.90,
            "minimum_compile_recommendation_coverage": 0.75,
            "minimum_exact_rejection_exercised_rate": 0.50,
            "minimum_full_fit_reliable_exact_supports": 1,
            "maximum_sdk_calls": 0,
            "maximum_wall_seconds": 600,
            "maximum_artifact_bytes": 512 * 1024 * 1024,
            "persist_archive_copies": False,
            "evaluation_archive_payloads_authorized": False,
            "confirmatory_claim_authorized": False,
            "physical_collection_authorized": False,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"T12.6.1c preregistered value changed: {name}")
        if any(name not in RELIABILITY_CANDIDATES for name in self.reliability_candidates):
            raise ValueError("T12.6.1c reliability candidate registry changed")

    @property
    def checksum(self) -> str:
        return _checksum(asdict(self))


def freeze_future_viability_reliability_hierarchy(
    *,
    output_path: str | Path,
    hierarchy_manifest_path: str | Path,
    hierarchy_compile_receipt_path: str | Path,
    seed_shift_diagnostic_receipt_path: str | Path,
    root: str | Path | None = None,
    protocol: FutureViabilityReliabilityProtocol | None = None,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    selected = protocol or FutureViabilityReliabilityProtocol()
    hierarchy_manifest_path = Path(hierarchy_manifest_path).resolve()
    hierarchy_compile_receipt_path = Path(hierarchy_compile_receipt_path).resolve()
    seed_shift_diagnostic_receipt_path = Path(
        seed_shift_diagnostic_receipt_path
    ).resolve()

    hierarchy = load_future_viability_hierarchy_manifest(
        hierarchy_manifest_path,
        root=repo_root,
        open_evaluation=False,
    )
    hierarchy_protocol = FutureViabilityHierarchyProtocol(
        **dict(hierarchy["protocol"])
    )
    hierarchy_receipt = load_hierarchy_receipt(
        hierarchy_compile_receipt_path,
        manifest=hierarchy,
        root=repo_root,
        expected_phase="compile",
        require_passed=True,
    )
    if hierarchy_receipt.get("status") != selected.hierarchy_compile_status:
        raise ValueError("T12.6.1c requires the passed T12.6.1 compile")
    diagnostic_receipt = load_seed_shift_diagnostic_receipt(
        seed_shift_diagnostic_receipt_path,
        root=repo_root,
        expected_phase="diagnostic",
    )
    if (
        diagnostic_receipt.get("passed") is not True
        or diagnostic_receipt.get("status") != selected.seed_shift_diagnostic_status
        or diagnostic_receipt.get("metrics", {}).get("classification")
        != selected.seed_shift_classification
    ):
        raise ValueError("T12.6.1c requires the completed T12.6.1b attribution")
    if diagnostic_receipt.get("metrics", {}).get("confirmatory_claim_authorized"):
        raise ValueError("T12.6.1c parent diagnostic boundary changed")
    if tuple(hierarchy_protocol.training_search_seeds) != selected.training_search_seeds:
        raise ValueError("T12.6.1c training seed registry changed")

    training_archives = []
    for meta in hierarchy["inputs"]["training_archives"]:
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
    if len(training_archives) != 12:
        raise ValueError("T12.6.1c requires exactly 12 training archives")

    missing_code = [
        path for path in RELIABILITY_CODE_PATHS if not (repo_root / path).is_file()
    ]
    if missing_code:
        raise ValueError(f"T12.6.1c code inventory incomplete: {missing_code}")
    git = _git_state(repo_root)
    if git["dirty"] and not allow_dirty:
        raise ValueError("scientific freeze requires a clean worktree")
    authorized = bool(not git["dirty"])
    payload = {
        "claim_boundary": {
            "authorized": (
                "source-train selection and compilation of a reliability-gated "
                "exact-to-composition future-viability hierarchy on bp35"
            ),
            "not_authorized": [
                "reuse of evaluation archives 9201-9203 for selection",
                "confirmatory transfer claim",
                "new holdout performance",
                "physical collection",
                "generic ARC-AGI improvement",
                "source validation",
                "controller authority",
                "neural training",
                "production authority",
            ],
        },
        "code_sha256": {
            path: _file_sha256(repo_root / path) for path in RELIABILITY_CODE_PATHS
        },
        "design": {
            "candidate_family_frozen_before_compile": True,
            "candidate_selection_uses_training_crossfit_only": True,
            "development_is_posthoc_to_t12_6_1b": True,
            "evaluation_archive_registry_imported": False,
            "evaluation_archive_payloads_authorized": False,
            "exact_requires_cross_seed_support_and_bounded_label_range": True,
            "exact_rejection_falls_back_to_composition_then_family": True,
            "future_confirmation_requires_new_archives": True,
            "same_archive_reconfirmation_forbidden": True,
            "zero_sdk_calls": True,
        },
        "firewall": {
            "compile_authorized": authorized,
            "evaluation_authorized": False,
            "environment_collection_authorized": False,
            "source_validation_opened": False,
            "new_holdout_opened": False,
            "controller_authority": False,
            "neural_training_authorized": False,
            "production_authority": False,
            "new_archive_protocol_freeze_authorized": False,
        },
        "format_version": RELIABILITY_MANIFEST_FORMAT,
        "game_id": hierarchy["game_id"],
        "git": git,
        "inputs": {"training_archives": training_archives},
        "parents": {
            "hierarchy_compile_receipt": _artifact_meta(
                hierarchy_compile_receipt_path,
                root=repo_root,
                receipt_checksum=hierarchy_receipt["receipt_checksum"],
                status=hierarchy_receipt["status"],
            ),
            "hierarchy_manifest": _artifact_meta(
                hierarchy_manifest_path,
                root=repo_root,
                manifest_checksum=hierarchy["manifest_checksum"],
            ),
            "seed_shift_diagnostic_receipt": _artifact_meta(
                seed_shift_diagnostic_receipt_path,
                root=repo_root,
                classification=diagnostic_receipt["metrics"]["classification"],
                receipt_checksum=diagnostic_receipt["receipt_checksum"],
                status=diagnostic_receipt["status"],
            ),
        },
        "protocol": asdict(selected),
        "protocol_checksum": selected.checksum,
        "scientific_claims_authorized": False,
        "stage": "source_train_reliability_development",
        "status": "FROZEN_BEFORE_T12_6_1C_SOURCE_TRAIN_COMPILE",
        "storage": {
            "maximum_artifact_bytes": selected.maximum_artifact_bytes,
            "maximum_sdk_calls": 0,
            "maximum_wall_seconds": selected.maximum_wall_seconds,
            "persist_archive_copies": False,
        },
    }
    manifest = _signed(payload, "manifest_checksum")
    _write_json_once(output_path, manifest)
    receipt = reliability_hierarchy_receipt(
        manifest=manifest,
        phase="freeze",
        passed=authorized,
        status=("PASS_T12_6_1C_FREEZE" if authorized else "DIRTY_SMOKE_ONLY"),
        metrics={
            "evaluation_archive_count": 0,
            "evaluation_archive_payloads_loaded": 0,
            "sdk_calls_used": 0,
            "training_archive_count": len(training_archives),
        },
    )
    _write_json_once(Path(output_path).with_name("freeze_receipt.json"), receipt)
    return manifest


def load_future_viability_reliability_manifest(
    path: str | Path,
    *,
    root: str | Path | None = None,
    verify_code: bool = True,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = _read_json(path)
    _verify_signed(manifest, "manifest_checksum")
    if manifest.get("format_version") != RELIABILITY_MANIFEST_FORMAT:
        raise ValueError("unsupported T12.6.1c reliability manifest")
    protocol = FutureViabilityReliabilityProtocol(**dict(manifest["protocol"]))
    if protocol.checksum != manifest.get("protocol_checksum"):
        raise ValueError("T12.6.1c protocol checksum mismatch")
    if set(manifest.get("inputs", {})) != {"training_archives"}:
        raise ValueError("T12.6.1c manifest imported a forbidden corpus")
    for meta in manifest["inputs"]["training_archives"]:
        _verify_meta(meta, root=repo_root)
    for meta in manifest["parents"].values():
        _verify_meta(meta, root=repo_root)
    if verify_code:
        for relative, expected in manifest["code_sha256"].items():
            candidate = repo_root / relative
            if not candidate.is_file() or _file_sha256(candidate) != expected:
                raise ValueError(f"T12.6.1c code checksum mismatch: {relative}")
    return manifest


def reliability_hierarchy_receipt(
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
            "format_version": RELIABILITY_RECEIPT_FORMAT,
            "manifest_checksum": manifest["manifest_checksum"],
            "metrics": dict(metrics),
            "parent_hierarchy_compile_receipt_checksum": manifest["parents"][
                "hierarchy_compile_receipt"
            ]["receipt_checksum"],
            "parent_seed_shift_diagnostic_receipt_checksum": manifest["parents"][
                "seed_shift_diagnostic_receipt"
            ]["receipt_checksum"],
            "passed": bool(passed),
            "phase": str(phase),
            "protocol_checksum": manifest["protocol_checksum"],
            "status": str(status),
        },
        "receipt_checksum",
    )


def load_reliability_hierarchy_receipt(
    path: str | Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
    expected_phase: str | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    receipt = _read_json(path)
    _verify_signed(receipt, "receipt_checksum")
    if receipt.get("format_version") != RELIABILITY_RECEIPT_FORMAT:
        raise ValueError("unsupported T12.6.1c reliability receipt")
    if manifest is not None and (
        receipt.get("manifest_checksum") != manifest.get("manifest_checksum")
        or receipt.get("protocol_checksum") != manifest.get("protocol_checksum")
    ):
        raise ValueError("T12.6.1c receipt belongs to another manifest")
    if expected_phase is not None and receipt.get("phase") != expected_phase:
        raise ValueError("T12.6.1c receipt phase mismatch")
    for meta in receipt.get("artifacts", {}).values():
        _verify_meta(meta, root=repo_root)
    return receipt


__all__ = [
    "RELIABILITY_CODE_PATHS",
    "RELIABILITY_MANIFEST_FORMAT",
    "RELIABILITY_PROTOCOL_FORMAT",
    "RELIABILITY_RECEIPT_FORMAT",
    "FutureViabilityReliabilityProtocol",
    "freeze_future_viability_reliability_hierarchy",
    "load_future_viability_reliability_manifest",
    "load_reliability_hierarchy_receipt",
    "reliability_hierarchy_receipt",
]
