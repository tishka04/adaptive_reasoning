"""Frozen post-hoc protocol for the SAGE.T12.6.1a conflict audit."""

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
    _signed,
    _verify_signed,
    _write_json_once,
)
from .future_viability_conflict_diagnostic import CONSOLIDATION_POLICIES
from .future_viability_hierarchy_protocol import (
    FutureViabilityHierarchyProtocol,
    _verify_meta,
    load_future_viability_hierarchy_manifest,
    load_hierarchy_receipt,
)

CONFLICT_PROTOCOL_FORMAT = (
    "sage-t12.6.1a-future-viability-conflict-protocol-v1"
)
CONFLICT_MANIFEST_FORMAT = (
    "sage-t12.6.1a-future-viability-conflict-manifest-v1"
)
CONFLICT_RECEIPT_FORMAT = (
    "sage-t12.6.1a-future-viability-conflict-receipt-v1"
)

CONFLICT_DIAGNOSTIC_CODE_PATHS = (
    "theory/sage_t/causal/experiment.py",
    "theory/sage_t/causal/archive.py",
    "theory/sage_t/causal/future_viability.py",
    "theory/sage_t/causal/hazard_diversity_model.py",
    "theory/sage_t/causal/future_viability_hierarchy.py",
    "theory/sage_t/causal/future_viability_hierarchy_protocol.py",
    "theory/sage_t/causal/future_viability_hierarchy_experiment.py",
    "theory/sage_t/causal/future_viability_conflict_diagnostic.py",
    "theory/sage_t/causal/future_viability_conflict_diagnostic_protocol.py",
    "theory/sage_t/causal/future_viability_conflict_diagnostic_experiment.py",
    "theory/sage_t/causal/future_viability_conflict_diagnostic_cli.py",
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
class FutureViabilityConflictDiagnosticProtocol:
    """Immutable explanatory axes for the already-open T12.6.1 evaluation."""

    format_version: str = CONFLICT_PROTOCOL_FORMAT
    parent_status: str = "FAIL_T12_6_1_EVALUATION_INTEGRITY_GATE"
    parent_classification: str = "EVALUATION_INTEGRITY_FAILURE"
    expected_parent_conflicts: int = 37
    expected_parent_eligible_groups: int = 366
    expected_parent_future_hits: int = 239
    expected_conflicted_archive_conditions: int = 12
    expected_unique_conflicted_archive_payloads: int = 8
    expected_future_label_conflicts: int = 6
    expected_immediate_label_conflicts: int = 37
    expected_difference_pattern_counts: tuple[tuple[str, int], ...] = (
        ("novel", 29),
        ("novel+target_cell_id", 2),
        ("terminal+novel+target_cell_id", 6),
    )
    consolidation_policies: tuple[str, ...] = CONSOLIDATION_POLICIES
    parent_reproduction_tolerance: float = 1e-12
    maximum_sdk_calls: int = 0
    maximum_wall_seconds: int = 600
    maximum_artifact_bytes: int = 512 * 1024 * 1024
    persist_archive_copies: bool = False
    confirmatory_claim_authorized: bool = False
    same_archive_reconfirmation_authorized: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "expected_difference_pattern_counts",
            tuple(tuple(value) for value in self.expected_difference_pattern_counts),
        )
        object.__setattr__(
            self, "consolidation_policies", tuple(self.consolidation_policies)
        )
        expected = {
            "format_version": CONFLICT_PROTOCOL_FORMAT,
            "parent_status": "FAIL_T12_6_1_EVALUATION_INTEGRITY_GATE",
            "parent_classification": "EVALUATION_INTEGRITY_FAILURE",
            "expected_parent_conflicts": 37,
            "expected_parent_eligible_groups": 366,
            "expected_parent_future_hits": 239,
            "expected_conflicted_archive_conditions": 12,
            "expected_unique_conflicted_archive_payloads": 8,
            "expected_future_label_conflicts": 6,
            "expected_immediate_label_conflicts": 37,
            "expected_difference_pattern_counts": (
                ("novel", 29),
                ("novel+target_cell_id", 2),
                ("terminal+novel+target_cell_id", 6),
            ),
            "consolidation_policies": CONSOLIDATION_POLICIES,
            "parent_reproduction_tolerance": 1e-12,
            "maximum_sdk_calls": 0,
            "maximum_wall_seconds": 600,
            "maximum_artifact_bytes": 512 * 1024 * 1024,
            "persist_archive_copies": False,
            "confirmatory_claim_authorized": False,
            "same_archive_reconfirmation_authorized": False,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"T12.6.1a preregistered value changed: {name}")

    @property
    def checksum(self) -> str:
        return _checksum(asdict(self))


def freeze_future_viability_conflict_diagnostic(
    *,
    output_path: str | Path,
    parent_manifest_path: str | Path,
    parent_evaluation_receipt_path: str | Path,
    root: str | Path | None = None,
    protocol: FutureViabilityConflictDiagnosticProtocol | None = None,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    selected = protocol or FutureViabilityConflictDiagnosticProtocol()
    parent_manifest_path = Path(parent_manifest_path).resolve()
    parent_evaluation_receipt_path = Path(
        parent_evaluation_receipt_path
    ).resolve()
    parent = load_future_viability_hierarchy_manifest(
        parent_manifest_path,
        root=repo_root,
        open_evaluation=True,
    )
    parent_protocol = FutureViabilityHierarchyProtocol(**dict(parent["protocol"]))
    parent_receipt = load_hierarchy_receipt(
        parent_evaluation_receipt_path,
        manifest=parent,
        root=repo_root,
        expected_phase="evaluation",
    )
    metrics = dict(parent_receipt.get("metrics", {}))
    if (
        parent_receipt.get("passed") is not False
        or parent_receipt.get("status") != selected.parent_status
        or metrics.get("classification") != selected.parent_classification
    ):
        raise ValueError("T12.6.1a requires the preserved integrity-failure parent")
    if metrics.get("checks", {}).get("duplicate_action_conflicts_absent") is not False:
        raise ValueError("T12.6.1a parent conflict gate changed")
    if any(
        metrics.get("checks", {}).get(name) is not True
        for name in (
            "all_archive_conditions_present",
            "all_evaluation_search_seeds_present",
            "all_source_lineages_present",
            "sdk_budget_respected",
            "wall_time_respected",
        )
    ):
        raise ValueError("T12.6.1a parent has another integrity failure")
    if (
        int(metrics.get("duplicate_action_conflicts", -1))
        != selected.expected_parent_conflicts
        or int(metrics.get("eligible_groups", -1))
        != selected.expected_parent_eligible_groups
        or int(metrics.get("future_binding_hits", -1))
        != selected.expected_parent_future_hits
    ):
        raise ValueError("T12.6.1a parent aggregate changed")

    evaluation_archives = []
    for meta in parent["inputs"]["evaluation_archives"]:
        path = _verify_meta(meta, root=repo_root)
        evaluation_archives.append(
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
    expected_conditions = (
        len(parent_protocol.evaluation_search_seeds)
        * len(parent_protocol.source_lineages)
        * len(parent_protocol.evaluation_arms)
    )
    if len(evaluation_archives) != expected_conditions:
        raise ValueError("T12.6.1a evaluation archive registry changed")

    missing_code = [
        path
        for path in CONFLICT_DIAGNOSTIC_CODE_PATHS
        if not (repo_root / path).is_file()
    ]
    if missing_code:
        raise ValueError(f"T12.6.1a code inventory incomplete: {missing_code}")
    git = _git_state(repo_root)
    if git["dirty"] and not allow_dirty:
        raise ValueError("scientific freeze requires a clean worktree")
    authorized = bool(not git["dirty"])

    payload = {
        "claim_boundary": {
            "authorized": (
                "post-hoc explanation and fixed-policy sensitivity of the "
                "already-open T12.6.1 evaluation conflicts"
            ),
            "not_authorized": [
                "repair or replacement of the T12.6.1 receipt",
                "confirmatory reuse of 9201-9203",
                "T12.6.2 freeze",
                "level progress",
                "physical collection",
                "generic ARC-AGI improvement",
                "source validation",
                "new holdout performance",
                "controller authority",
                "neural training",
                "production authority",
            ],
        },
        "code_sha256": {
            path: _file_sha256(repo_root / path)
            for path in CONFLICT_DIAGNOSTIC_CODE_PATHS
        },
        "design": {
            "archive_graph_held_raw_to_isolate_observed_edge_consolidation": True,
            "axes_are_posthoc_to_conflict_inspection": True,
            "fixed_models_and_parent_gates_reused_without_refit": True,
            "future_label_envelopes_are_diagnostic_not_deployment_policies": True,
            "parent_receipt_is_immutable": True,
            "same_archive_reconfirmation_forbidden": True,
            "threshold_retuning_forbidden": True,
            "zero_sdk_calls": True,
        },
        "firewall": {
            "diagnostic_authorized": authorized,
            "environment_collection_authorized": False,
            "source_validation_opened": False,
            "new_holdout_opened": False,
            "controller_authority": False,
            "neural_training_authorized": False,
            "production_authority": False,
            "t12_6_2_freeze_authorized": False,
            "new_archive_protocol_freeze_authorized": False,
        },
        "format_version": CONFLICT_MANIFEST_FORMAT,
        "game_id": parent["game_id"],
        "git": git,
        "inputs": {"evaluation_archives": evaluation_archives},
        "parent": {
            "evaluation_artifacts": {
                name: dict(meta)
                for name, meta in parent_receipt["artifacts"].items()
            },
            "evaluation_receipt": _artifact_meta(
                parent_evaluation_receipt_path,
                root=repo_root,
                receipt_checksum=parent_receipt["receipt_checksum"],
                status=parent_receipt["status"],
            ),
            "manifest": _artifact_meta(
                parent_manifest_path,
                root=repo_root,
                manifest_checksum=parent["manifest_checksum"],
            ),
        },
        "parent_protocol": asdict(parent_protocol),
        "protocol": asdict(selected),
        "protocol_checksum": selected.checksum,
        "scientific_claims_authorized": False,
        "stage": "opened_evaluation_posthoc_conflict_diagnostic",
        "status": "FROZEN_BEFORE_T12_6_1A_POSTHOC_DIAGNOSTIC",
        "storage": {
            "maximum_artifact_bytes": selected.maximum_artifact_bytes,
            "maximum_sdk_calls": 0,
            "maximum_wall_seconds": selected.maximum_wall_seconds,
            "persist_archive_copies": False,
        },
    }
    manifest = _signed(payload, "manifest_checksum")
    _write_json_once(output_path, manifest)
    receipt = conflict_diagnostic_receipt(
        manifest=manifest,
        phase="freeze",
        passed=authorized,
        status=(
            "PASS_T12_6_1A_DIAGNOSTIC_FREEZE"
            if authorized
            else "DIRTY_SMOKE_ONLY"
        ),
        metrics={
            "evaluation_archive_count": len(evaluation_archives),
            "parent_conflict_count": int(metrics["duplicate_action_conflicts"]),
            "sdk_calls_used": 0,
        },
    )
    _write_json_once(Path(output_path).with_name("freeze_receipt.json"), receipt)
    return manifest


def load_future_viability_conflict_diagnostic_manifest(
    path: str | Path,
    *,
    root: str | Path | None = None,
    verify_code: bool = True,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _verify_signed(manifest, "manifest_checksum")
    if manifest.get("format_version") != CONFLICT_MANIFEST_FORMAT:
        raise ValueError("unsupported T12.6.1a conflict manifest")
    protocol = FutureViabilityConflictDiagnosticProtocol(**dict(manifest["protocol"]))
    if protocol.checksum != manifest.get("protocol_checksum"):
        raise ValueError("T12.6.1a protocol checksum mismatch")
    for meta in manifest["inputs"]["evaluation_archives"]:
        _verify_meta(meta, root=repo_root)
    _verify_meta(manifest["parent"]["manifest"], root=repo_root)
    _verify_meta(manifest["parent"]["evaluation_receipt"], root=repo_root)
    for meta in manifest["parent"]["evaluation_artifacts"].values():
        _verify_meta(meta, root=repo_root)
    if verify_code:
        for relative, expected in manifest["code_sha256"].items():
            candidate = repo_root / relative
            if not candidate.is_file() or _file_sha256(candidate) != expected:
                raise ValueError(f"T12.6.1a code checksum mismatch: {relative}")
    return manifest


def conflict_diagnostic_receipt(
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
            "format_version": CONFLICT_RECEIPT_FORMAT,
            "manifest_checksum": manifest["manifest_checksum"],
            "metrics": dict(metrics),
            "parent_t12_6_1_evaluation_receipt_checksum": manifest["parent"][
                "evaluation_receipt"
            ]["receipt_checksum"],
            "passed": bool(passed),
            "phase": str(phase),
            "protocol_checksum": manifest["protocol_checksum"],
            "status": str(status),
        },
        "receipt_checksum",
    )


def load_conflict_diagnostic_receipt(
    path: str | Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
    expected_phase: str | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    _verify_signed(receipt, "receipt_checksum")
    if receipt.get("format_version") != CONFLICT_RECEIPT_FORMAT:
        raise ValueError("unsupported T12.6.1a conflict receipt")
    if manifest is not None and (
        receipt.get("manifest_checksum") != manifest.get("manifest_checksum")
        or receipt.get("protocol_checksum") != manifest.get("protocol_checksum")
    ):
        raise ValueError("T12.6.1a receipt belongs to another manifest")
    if expected_phase is not None and receipt.get("phase") != expected_phase:
        raise ValueError("T12.6.1a receipt phase mismatch")
    for meta in receipt.get("artifacts", {}).values():
        _verify_meta(meta, root=repo_root)
    return receipt


__all__ = [
    "CONFLICT_DIAGNOSTIC_CODE_PATHS",
    "CONFLICT_MANIFEST_FORMAT",
    "CONFLICT_PROTOCOL_FORMAT",
    "CONFLICT_RECEIPT_FORMAT",
    "FutureViabilityConflictDiagnosticProtocol",
    "conflict_diagnostic_receipt",
    "freeze_future_viability_conflict_diagnostic",
    "load_conflict_diagnostic_receipt",
    "load_future_viability_conflict_diagnostic_manifest",
]
