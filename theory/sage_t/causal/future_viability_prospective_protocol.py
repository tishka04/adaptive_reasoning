"""Frozen prospective confirmation protocol for SAGE.T12.6.1d."""

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
from .future_viability_hierarchy_protocol import _verify_meta
from .future_viability_reliability_hierarchy_protocol import (
    FutureViabilityReliabilityProtocol,
    load_future_viability_reliability_manifest,
    load_reliability_hierarchy_receipt,
)
from .hazard_diversity_protocol import (
    HazardDiversityProtocol,
    load_hazard_diversity_manifest,
    load_hazard_diversity_receipt,
)

PROSPECTIVE_PROTOCOL_FORMAT = "sage-t12.6.1d-future-viability-confirmation-protocol-v1"
PROSPECTIVE_MANIFEST_FORMAT = "sage-t12.6.1d-future-viability-confirmation-manifest-v1"
PROSPECTIVE_RECEIPT_FORMAT = "sage-t12.6.1d-future-viability-confirmation-receipt-v1"

PROSPECTIVE_CODE_PATHS = (
    "theory/sage_t/causal/experiment.py",
    "theory/sage_t/causal/archive.py",
    "theory/sage_t/causal/graph_experiment.py",
    "theory/sage_t/causal/target_regrounding_experiment.py",
    "theory/sage_t/causal/hazard_diversity_model.py",
    "theory/sage_t/causal/hazard_diversity_protocol.py",
    "theory/sage_t/causal/hazard_diversity_experiment.py",
    "theory/sage_t/causal/future_viability.py",
    "theory/sage_t/causal/future_viability_hierarchy.py",
    "theory/sage_t/causal/future_viability_reliability_hierarchy.py",
    "theory/sage_t/causal/future_viability_reliability_hierarchy_protocol.py",
    "theory/sage_t/causal/future_viability_reliability_hierarchy_experiment.py",
    "theory/sage_t/causal/future_viability_prospective_confirmation.py",
    "theory/sage_t/causal/future_viability_prospective_protocol.py",
    "theory/sage_t/causal/future_viability_prospective_experiment.py",
    "theory/sage_t/causal/future_viability_prospective_cli.py",
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
class FutureViabilityProspectiveProtocol:
    """Immutable collection matrix, evaluator and superiority gates."""

    format_version: str = PROSPECTIVE_PROTOCOL_FORMAT
    reliability_compile_status: str = "PASS_T12_6_1C_SOURCE_TRAIN_COMPILE_GATE"
    reliability_selected_candidate: str = "exact_span2_range0"
    hazard_compile_status: str = "PASS_T12_4A_4D_1_HAZARD_COMPILE_GATE"
    prospective_search_seeds: tuple[int, ...] = (9_301, 9_302, 9_303)
    pilot_search_seeds: tuple[int, ...] = (9_301,)
    completion_search_seeds: tuple[int, ...] = (9_302, 9_303)
    source_lineages: tuple[int, ...] = (8_701, 8_705)
    search_arms: tuple[str, ...] = (
        "local_archive_control",
        "diversity_control",
        "abstract_hazard_diversity",
    )
    burst_schedule: tuple[int, ...] = (4, 8, 16)
    future_horizon: int = 4
    local_radius: int = 7
    binding_shift: int = 1
    sdk_calls_per_archive: int = 2_048
    maximum_total_sdk_calls: int = 38_000
    maximum_excursions_per_archive: int = 64
    maximum_cells_per_archive: int = 10_000
    expected_archive_count: int = 18
    minimum_unique_archive_count: int = 12
    minimum_eligible_groups: int = 250
    minimum_top1_accuracy: float = 0.70
    minimum_gain_over_immediate: float = 0.10
    minimum_gain_over_binding_swap: float = 0.25
    minimum_gain_over_exact_first: float = 0.02
    minimum_seed_wins_over_exact_first: int = 2
    minimum_worst_seed_gain_over_exact_first: float = 0.0
    minimum_lineage_accuracy: float = 0.65
    minimum_hierarchy_coverage: float = 0.70
    minimum_unique_top_rate: float = 0.85
    minimum_recommendation_coverage: float = 0.60
    minimum_exact_rejection_exercised_rate: float = 0.25
    bootstrap_repetitions: int = 10_000
    bootstrap_seed: int = 1_261
    bootstrap_lower_quantile: float = 0.05
    minimum_bootstrap_gain_lower_bound: float = 0.0
    maximum_artifact_bytes: int = 1024 * 1024 * 1024
    maximum_wall_seconds_per_batch: int = 14_400
    maximum_offline_wall_seconds: int = 1_800
    persist_raw_frames: bool = False
    model_refit_authorized: bool = False
    old_evaluation_archives_authorized: bool = False
    controller_authority: bool = False
    neural_training_authorized: bool = False
    production_authority: bool = False

    def __post_init__(self) -> None:
        for name in (
            "prospective_search_seeds",
            "pilot_search_seeds",
            "completion_search_seeds",
            "source_lineages",
            "burst_schedule",
        ):
            object.__setattr__(
                self, name, tuple(int(value) for value in getattr(self, name))
            )
        object.__setattr__(
            self, "search_arms", tuple(str(value) for value in self.search_arms)
        )
        expected = {
            "format_version": PROSPECTIVE_PROTOCOL_FORMAT,
            "reliability_compile_status": ("PASS_T12_6_1C_SOURCE_TRAIN_COMPILE_GATE"),
            "reliability_selected_candidate": "exact_span2_range0",
            "hazard_compile_status": "PASS_T12_4A_4D_1_HAZARD_COMPILE_GATE",
            "prospective_search_seeds": (9_301, 9_302, 9_303),
            "pilot_search_seeds": (9_301,),
            "completion_search_seeds": (9_302, 9_303),
            "source_lineages": (8_701, 8_705),
            "search_arms": (
                "local_archive_control",
                "diversity_control",
                "abstract_hazard_diversity",
            ),
            "burst_schedule": (4, 8, 16),
            "future_horizon": 4,
            "local_radius": 7,
            "binding_shift": 1,
            "sdk_calls_per_archive": 2_048,
            "maximum_total_sdk_calls": 38_000,
            "maximum_excursions_per_archive": 64,
            "maximum_cells_per_archive": 10_000,
            "expected_archive_count": 18,
            "minimum_unique_archive_count": 12,
            "minimum_eligible_groups": 250,
            "minimum_top1_accuracy": 0.70,
            "minimum_gain_over_immediate": 0.10,
            "minimum_gain_over_binding_swap": 0.25,
            "minimum_gain_over_exact_first": 0.02,
            "minimum_seed_wins_over_exact_first": 2,
            "minimum_worst_seed_gain_over_exact_first": 0.0,
            "minimum_lineage_accuracy": 0.65,
            "minimum_hierarchy_coverage": 0.70,
            "minimum_unique_top_rate": 0.85,
            "minimum_recommendation_coverage": 0.60,
            "minimum_exact_rejection_exercised_rate": 0.25,
            "bootstrap_repetitions": 10_000,
            "bootstrap_seed": 1_261,
            "bootstrap_lower_quantile": 0.05,
            "minimum_bootstrap_gain_lower_bound": 0.0,
            "maximum_artifact_bytes": 1024 * 1024 * 1024,
            "maximum_wall_seconds_per_batch": 14_400,
            "maximum_offline_wall_seconds": 1_800,
            "persist_raw_frames": False,
            "model_refit_authorized": False,
            "old_evaluation_archives_authorized": False,
            "controller_authority": False,
            "neural_training_authorized": False,
            "production_authority": False,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"T12.6.1d preregistered value changed: {name}")
        if set(self.pilot_search_seeds) & set(self.completion_search_seeds):
            raise ValueError("T12.6.1d collection batches overlap")
        if set(self.pilot_search_seeds) | set(self.completion_search_seeds) != set(
            self.prospective_search_seeds
        ):
            raise ValueError("T12.6.1d collection batches are incomplete")

    @property
    def checksum(self) -> str:
        return _checksum(asdict(self))


def freeze_future_viability_prospective_confirmation(
    *,
    output_path: str | Path,
    reliability_manifest_path: str | Path,
    reliability_compile_receipt_path: str | Path,
    hazard_manifest_path: str | Path,
    hazard_compile_receipt_path: str | Path,
    root: str | Path | None = None,
    protocol: FutureViabilityProspectiveProtocol | None = None,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    selected = protocol or FutureViabilityProspectiveProtocol()
    reliability_manifest_path = Path(reliability_manifest_path).resolve()
    reliability_compile_receipt_path = Path(reliability_compile_receipt_path).resolve()
    hazard_manifest_path = Path(hazard_manifest_path).resolve()
    hazard_compile_receipt_path = Path(hazard_compile_receipt_path).resolve()

    reliability_manifest = load_future_viability_reliability_manifest(
        reliability_manifest_path, root=repo_root
    )
    reliability_protocol = FutureViabilityReliabilityProtocol(
        **dict(reliability_manifest["protocol"])
    )
    reliability_receipt = load_reliability_hierarchy_receipt(
        reliability_compile_receipt_path,
        manifest=reliability_manifest,
        root=repo_root,
        expected_phase="compile",
    )
    if (
        reliability_receipt.get("passed") is not True
        or reliability_receipt.get("status") != selected.reliability_compile_status
        or reliability_receipt.get("metrics", {}).get("selected_candidate")
        != selected.reliability_selected_candidate
        or not reliability_receipt.get("metrics", {}).get(
            "new_archive_protocol_freeze_authorized", False
        )
    ):
        raise ValueError("T12.6.1d requires the qualified T12.6.1c model")
    model_meta = reliability_receipt.get("artifacts", {}).get("models")
    if not model_meta:
        raise ValueError("T12.6.1d parent model bundle is missing")
    model_path = _verify_meta(model_meta, root=repo_root)
    model_payload = _read_json(model_path)
    _verify_signed(model_payload, "bundle_checksum")
    model_bundle_checksum = str(model_payload["bundle_checksum"])

    hazard_manifest = load_hazard_diversity_manifest(
        hazard_manifest_path, root=repo_root, verify_code=False
    )
    hazard_protocol = HazardDiversityProtocol(**dict(hazard_manifest["protocol"]))
    hazard_receipt = load_hazard_diversity_receipt(
        hazard_compile_receipt_path,
        manifest=hazard_manifest,
        root=repo_root,
    )
    if (
        hazard_receipt.get("passed") is not True
        or hazard_receipt.get("status") != selected.hazard_compile_status
        or hazard_receipt.get("phase") != "compile"
    ):
        raise ValueError("T12.6.1d requires the passed hazard collector compile")
    if tuple(hazard_protocol.search_arms) != selected.search_arms:
        raise ValueError("T12.6.1d collector arm registry changed")
    if tuple(hazard_protocol.source_lineages) != selected.source_lineages:
        raise ValueError("T12.6.1d collector lineage registry changed")
    if set(selected.prospective_search_seeds) & (
        set(reliability_protocol.training_search_seeds)
        | set(hazard_protocol.active_search_seeds)
        | set(hazard_protocol.compile_search_seeds)
    ):
        raise ValueError("T12.6.1d prospective seeds are not fresh")

    missing = [
        path for path in PROSPECTIVE_CODE_PATHS if not (repo_root / path).is_file()
    ]
    if missing:
        raise ValueError(f"T12.6.1d code inventory incomplete: {missing}")
    git = _git_state(repo_root)
    if git["dirty"] and not allow_dirty:
        raise ValueError("scientific freeze requires a clean worktree")
    authorized = bool(not git["dirty"])
    payload = {
        "claim_boundary": {
            "authorized": (
                "fresh-seed bp35 prospective confirmation of the frozen "
                "T12.6.1c reliability-gated future-viability hierarchy"
            ),
            "not_authorized": [
                "reuse of 9201-9203",
                "model refit or recalibration",
                "target-game generalization",
                "generic ARC-AGI improvement",
                "source validation",
                "holdout opening",
                "controller authority",
                "neural training",
                "production authority",
            ],
        },
        "code_sha256": {
            path: _file_sha256(repo_root / path) for path in PROSPECTIVE_CODE_PATHS
        },
        "design": {
            "archive_content_deduplicated_before_scoring": True,
            "collection_batches_are_outcome_blind": True,
            "exact_state_is_decision_unit": True,
            "exact_state_future_graph_is_label_source": True,
            "label_blind_prediction_commitment_precedes_adjudication": True,
            "novelty_excluded_from_oracle": True,
            "same_exact_action_transition_conflict_fails_integrity": True,
            "same_model_bundle_reused_without_refit": True,
            "seed_blocked_bootstrap": True,
            "staged_collection_without_intermediate_scoring": True,
        },
        "firewall": {
            "preflight_authorized": authorized,
            "pilot_collection_authorized": False,
            "completion_collection_authorized": False,
            "collection_seal_authorized": False,
            "prediction_authorized": False,
            "adjudication_authorized": False,
            "t12_6_2_freeze_authorized": False,
            "source_validation_opened": False,
            "holdout_opened": False,
            "controller_authority": False,
            "neural_training_authorized": False,
            "production_authority": False,
        },
        "format_version": PROSPECTIVE_MANIFEST_FORMAT,
        "game_id": reliability_manifest["game_id"],
        "git": git,
        "parents": {
            "hazard_compile_receipt": _artifact_meta(
                hazard_compile_receipt_path,
                root=repo_root,
                receipt_checksum=hazard_receipt["receipt_checksum"],
                status=hazard_receipt["status"],
            ),
            "hazard_manifest": _artifact_meta(
                hazard_manifest_path,
                root=repo_root,
                manifest_checksum=hazard_manifest["manifest_checksum"],
            ),
            "reliability_compile_receipt": _artifact_meta(
                reliability_compile_receipt_path,
                root=repo_root,
                receipt_checksum=reliability_receipt["receipt_checksum"],
                status=reliability_receipt["status"],
            ),
            "reliability_manifest": _artifact_meta(
                reliability_manifest_path,
                root=repo_root,
                manifest_checksum=reliability_manifest["manifest_checksum"],
            ),
            "reliability_model_bundle": _artifact_meta(
                model_path,
                root=repo_root,
                bundle_checksum=model_bundle_checksum,
            ),
        },
        "protocol": asdict(selected),
        "protocol_checksum": selected.checksum,
        "scientific_claims_authorized": False,
        "stage": "fresh_seed_source_train_prospective_confirmation",
        "status": "FROZEN_BEFORE_T12_6_1D_PREFLIGHT",
        "storage": {
            "maximum_artifact_bytes": selected.maximum_artifact_bytes,
            "maximum_sdk_calls": selected.maximum_total_sdk_calls,
            "persist_raw_frames": False,
        },
    }
    manifest = _signed(payload, "manifest_checksum")
    _write_json_once(output_path, manifest)
    receipt = prospective_receipt(
        manifest=manifest,
        phase="freeze",
        passed=authorized,
        status="PASS_T12_6_1D_FREEZE" if authorized else "DIRTY_SMOKE_ONLY",
        metrics={
            "old_evaluation_archive_count": 0,
            "prospective_search_seeds": list(selected.prospective_search_seeds),
            "sdk_calls_used": 0,
        },
    )
    _write_json_once(Path(output_path).with_name("freeze_receipt.json"), receipt)
    return manifest


def load_future_viability_prospective_manifest(
    path: str | Path,
    *,
    root: str | Path | None = None,
    verify_code: bool = True,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = _read_json(path)
    _verify_signed(manifest, "manifest_checksum")
    if manifest.get("format_version") != PROSPECTIVE_MANIFEST_FORMAT:
        raise ValueError("unsupported T12.6.1d prospective manifest")
    protocol = FutureViabilityProspectiveProtocol(**dict(manifest["protocol"]))
    if protocol.checksum != manifest.get("protocol_checksum"):
        raise ValueError("T12.6.1d protocol checksum mismatch")
    if "inputs" in manifest:
        raise ValueError("T12.6.1d freeze imported prospective archives")
    for meta in manifest["parents"].values():
        _verify_meta(meta, root=repo_root)
    if verify_code:
        for relative, expected in manifest["code_sha256"].items():
            candidate = repo_root / relative
            if not candidate.is_file() or _file_sha256(candidate) != expected:
                raise ValueError(f"T12.6.1d code checksum mismatch: {relative}")
    return manifest


def prospective_receipt(
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
            "format_version": PROSPECTIVE_RECEIPT_FORMAT,
            "manifest_checksum": manifest["manifest_checksum"],
            "metrics": dict(metrics),
            "parent_hazard_compile_receipt_checksum": manifest["parents"][
                "hazard_compile_receipt"
            ]["receipt_checksum"],
            "parent_reliability_compile_receipt_checksum": manifest["parents"][
                "reliability_compile_receipt"
            ]["receipt_checksum"],
            "passed": bool(passed),
            "phase": str(phase),
            "protocol_checksum": manifest["protocol_checksum"],
            "status": str(status),
        },
        "receipt_checksum",
    )


def _verify_artifact_tree(value: Any, *, root: Path) -> None:
    if isinstance(value, Mapping):
        if "path" in value and "sha256" in value:
            _verify_meta(value, root=root)
        else:
            for nested in value.values():
                _verify_artifact_tree(nested, root=root)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _verify_artifact_tree(nested, root=root)


def load_prospective_receipt(
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
    if receipt.get("format_version") != PROSPECTIVE_RECEIPT_FORMAT:
        raise ValueError("unsupported T12.6.1d prospective receipt")
    if manifest is not None and (
        receipt.get("manifest_checksum") != manifest.get("manifest_checksum")
        or receipt.get("protocol_checksum") != manifest.get("protocol_checksum")
    ):
        raise ValueError("T12.6.1d receipt belongs to another manifest")
    if expected_phase is not None and receipt.get("phase") != expected_phase:
        raise ValueError("T12.6.1d receipt phase mismatch")
    if require_passed and receipt.get("passed") is not True:
        raise ValueError(f"T12.6.1d gate failed: {receipt.get('status')}")
    _verify_artifact_tree(receipt.get("artifacts", {}), root=repo_root)
    return receipt


__all__ = [
    "PROSPECTIVE_CODE_PATHS",
    "PROSPECTIVE_MANIFEST_FORMAT",
    "PROSPECTIVE_PROTOCOL_FORMAT",
    "PROSPECTIVE_RECEIPT_FORMAT",
    "FutureViabilityProspectiveProtocol",
    "freeze_future_viability_prospective_confirmation",
    "load_future_viability_prospective_manifest",
    "load_prospective_receipt",
    "prospective_receipt",
]
