"""Frozen post-hoc protocol for the SAGE.T12.6.1b seed-shift audit."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
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
from .future_viability_conflict_diagnostic_protocol import (
    load_conflict_diagnostic_receipt,
    load_future_viability_conflict_diagnostic_manifest,
)
from .future_viability_hierarchy_protocol import (
    FutureViabilityHierarchyProtocol,
    _verify_meta,
    load_future_viability_hierarchy_manifest,
    load_hierarchy_receipt,
)
from .future_viability_seed_shift_diagnostic import SEED_SHIFT_DIAGNOSTIC_AXES

SEED_SHIFT_PROTOCOL_FORMAT = (
    "sage-t12.6.1b-future-viability-seed-shift-protocol-v1"
)
SEED_SHIFT_MANIFEST_FORMAT = (
    "sage-t12.6.1b-future-viability-seed-shift-manifest-v1"
)
SEED_SHIFT_RECEIPT_FORMAT = (
    "sage-t12.6.1b-future-viability-seed-shift-receipt-v1"
)

SEED_SHIFT_CODE_PATHS = (
    "theory/sage_t/causal/experiment.py",
    "theory/sage_t/causal/archive.py",
    "theory/sage_t/causal/future_viability.py",
    "theory/sage_t/causal/hazard_diversity_model.py",
    "theory/sage_t/causal/future_viability_hierarchy.py",
    "theory/sage_t/causal/future_viability_hierarchy_protocol.py",
    "theory/sage_t/causal/future_viability_hierarchy_experiment.py",
    "theory/sage_t/causal/future_viability_conflict_diagnostic.py",
    "theory/sage_t/causal/future_viability_conflict_diagnostic_protocol.py",
    "theory/sage_t/causal/future_viability_seed_shift_diagnostic.py",
    "theory/sage_t/causal/future_viability_seed_shift_diagnostic_protocol.py",
    "theory/sage_t/causal/future_viability_seed_shift_diagnostic_experiment.py",
    "theory/sage_t/causal/future_viability_seed_shift_diagnostic_cli.py",
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
class FutureViabilitySeedShiftDiagnosticProtocol:
    """Immutable attribution axes for the already-observed 9202 miss."""

    format_version: str = SEED_SHIFT_PROTOCOL_FORMAT
    hierarchy_status: str = "FAIL_T12_6_1_EVALUATION_INTEGRITY_GATE"
    conflict_diagnostic_status: str = "PASS_T12_6_1A_DIAGNOSTIC_COMPLETE"
    conflict_diagnostic_classification: str = (
        "POSTHOC_TRANSFER_GATE_MISS_ACROSS_REGISTERED_CONSOLIDATIONS"
    )
    focal_search_seed: int = 9_202
    reference_search_seeds: tuple[int, ...] = (9_201, 9_203)
    training_search_seeds: tuple[int, ...] = (9_101, 9_102, 9_103)
    expected_focal_eligible_groups: int = 92
    expected_focal_hits: int = 36
    expected_focal_accuracy: float = 36 / 92
    evaluation_consolidation_policy: str = "parent_order"
    diagnostic_axes: tuple[str, ...] = SEED_SHIFT_DIAGNOSTIC_AXES
    maximum_sdk_calls: int = 0
    maximum_wall_seconds: int = 600
    maximum_artifact_bytes: int = 512 * 1024 * 1024
    persist_archive_copies: bool = False
    confirmatory_claim_authorized: bool = False
    model_or_descriptor_change_authorized: bool = False

    def __post_init__(self) -> None:
        for name in (
            "reference_search_seeds",
            "training_search_seeds",
            "diagnostic_axes",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        expected = {
            "format_version": SEED_SHIFT_PROTOCOL_FORMAT,
            "hierarchy_status": "FAIL_T12_6_1_EVALUATION_INTEGRITY_GATE",
            "conflict_diagnostic_status": "PASS_T12_6_1A_DIAGNOSTIC_COMPLETE",
            "conflict_diagnostic_classification": (
                "POSTHOC_TRANSFER_GATE_MISS_ACROSS_REGISTERED_CONSOLIDATIONS"
            ),
            "focal_search_seed": 9_202,
            "reference_search_seeds": (9_201, 9_203),
            "training_search_seeds": (9_101, 9_102, 9_103),
            "expected_focal_eligible_groups": 92,
            "expected_focal_hits": 36,
            "expected_focal_accuracy": 36 / 92,
            "evaluation_consolidation_policy": "parent_order",
            "diagnostic_axes": SEED_SHIFT_DIAGNOSTIC_AXES,
            "maximum_sdk_calls": 0,
            "maximum_wall_seconds": 600,
            "maximum_artifact_bytes": 512 * 1024 * 1024,
            "persist_archive_copies": False,
            "confirmatory_claim_authorized": False,
            "model_or_descriptor_change_authorized": False,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"T12.6.1b preregistered value changed: {name}")

    @property
    def checksum(self) -> str:
        return _checksum(asdict(self))


def freeze_future_viability_seed_shift_diagnostic(
    *,
    output_path: str | Path,
    hierarchy_manifest_path: str | Path,
    hierarchy_evaluation_receipt_path: str | Path,
    conflict_manifest_path: str | Path,
    conflict_diagnostic_receipt_path: str | Path,
    root: str | Path | None = None,
    protocol: FutureViabilitySeedShiftDiagnosticProtocol | None = None,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    selected = protocol or FutureViabilitySeedShiftDiagnosticProtocol()
    hierarchy_manifest_path = Path(hierarchy_manifest_path).resolve()
    hierarchy_evaluation_receipt_path = Path(
        hierarchy_evaluation_receipt_path
    ).resolve()
    conflict_manifest_path = Path(conflict_manifest_path).resolve()
    conflict_diagnostic_receipt_path = Path(
        conflict_diagnostic_receipt_path
    ).resolve()

    hierarchy = load_future_viability_hierarchy_manifest(
        hierarchy_manifest_path, root=repo_root, open_evaluation=True
    )
    hierarchy_protocol = FutureViabilityHierarchyProtocol(
        **dict(hierarchy["protocol"])
    )
    hierarchy_receipt = load_hierarchy_receipt(
        hierarchy_evaluation_receipt_path,
        manifest=hierarchy,
        root=repo_root,
        expected_phase="evaluation",
    )
    if (
        hierarchy_receipt.get("passed") is not False
        or hierarchy_receipt.get("status") != selected.hierarchy_status
    ):
        raise ValueError("T12.6.1b requires the preserved T12.6.1 failure")

    conflict_manifest = load_future_viability_conflict_diagnostic_manifest(
        conflict_manifest_path, root=repo_root
    )
    conflict_receipt = load_conflict_diagnostic_receipt(
        conflict_diagnostic_receipt_path,
        manifest=conflict_manifest,
        root=repo_root,
        expected_phase="diagnostic",
    )
    if (
        conflict_receipt.get("passed") is not True
        or conflict_receipt.get("status") != selected.conflict_diagnostic_status
        or conflict_receipt.get("metrics", {}).get("classification")
        != selected.conflict_diagnostic_classification
    ):
        raise ValueError("T12.6.1b requires the completed T12.6.1a diagnostic")
    if conflict_receipt.get(
        "parent_t12_6_1_evaluation_receipt_checksum"
    ) != hierarchy_receipt.get("receipt_checksum"):
        raise ValueError("T12.6.1b parents do not share the same evaluation")
    focal = conflict_receipt["metrics"]["policy_results"][
        selected.evaluation_consolidation_policy
    ]["per_search_seed"][str(selected.focal_search_seed)]
    if (
        int(focal["eligible_groups"]) != selected.expected_focal_eligible_groups
        or int(focal["future_binding_hits"]) != selected.expected_focal_hits
        or abs(
            float(focal["future_binding_top1_accuracy"])
            - selected.expected_focal_accuracy
        )
        > 1e-12
    ):
        raise ValueError("T12.6.1b focal aggregate changed")
    if tuple(hierarchy_protocol.training_search_seeds) != (
        selected.training_search_seeds
    ):
        raise ValueError("T12.6.1b training seed registry changed")

    def bound_archives(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        output = []
        for meta in values:
            path = _verify_meta(meta, root=repo_root)
            output.append(
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
        return output

    training_archives = bound_archives(hierarchy["inputs"]["training_archives"])
    evaluation_archives = bound_archives(
        hierarchy["inputs"]["evaluation_archives"]
    )
    if len(training_archives) != 12 or len(evaluation_archives) != 18:
        raise ValueError("T12.6.1b archive registry changed")

    missing_code = [
        path for path in SEED_SHIFT_CODE_PATHS if not (repo_root / path).is_file()
    ]
    if missing_code:
        raise ValueError(f"T12.6.1b code inventory incomplete: {missing_code}")
    git = _git_state(repo_root)
    if git["dirty"] and not allow_dirty:
        raise ValueError("scientific freeze requires a clean worktree")
    authorized = bool(not git["dirty"])
    payload = {
        "claim_boundary": {
            "authorized": (
                "post-hoc attribution of the frozen T12.6.1 parent-order "
                "ranking collapse on evaluation seed 9202"
            ),
            "not_authorized": [
                "model or descriptor selection",
                "threshold retuning",
                "repair of T12.6.1",
                "confirmatory reuse of 9201-9203",
                "T12.6.2 freeze",
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
            path: _file_sha256(repo_root / path) for path in SEED_SHIFT_CODE_PATHS
        },
        "design": {
            "axes_are_frozen_before_individual_9202_error_inspection": True,
            "conflict_policy_is_parent_order": True,
            "evaluation_labels_are_used_for_attribution_only": True,
            "fixed_model_and_support_hierarchy": True,
            "leave_one_training_seed_out_is_sensitivity_not_model_selection": True,
            "posthoc_only": True,
            "same_archive_reconfirmation_forbidden": True,
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
            "future_protocol_freeze_authorized": False,
        },
        "format_version": SEED_SHIFT_MANIFEST_FORMAT,
        "game_id": hierarchy["game_id"],
        "git": git,
        "inputs": {
            "evaluation_archives": evaluation_archives,
            "training_archives": training_archives,
        },
        "parents": {
            "conflict_diagnostic_artifacts": {
                name: dict(meta)
                for name, meta in conflict_receipt["artifacts"].items()
            },
            "conflict_diagnostic_manifest": _artifact_meta(
                conflict_manifest_path,
                root=repo_root,
                manifest_checksum=conflict_manifest["manifest_checksum"],
            ),
            "conflict_diagnostic_receipt": _artifact_meta(
                conflict_diagnostic_receipt_path,
                root=repo_root,
                receipt_checksum=conflict_receipt["receipt_checksum"],
                status=conflict_receipt["status"],
            ),
            "hierarchy_evaluation_artifacts": {
                name: dict(meta)
                for name, meta in hierarchy_receipt["artifacts"].items()
            },
            "hierarchy_evaluation_receipt": _artifact_meta(
                hierarchy_evaluation_receipt_path,
                root=repo_root,
                receipt_checksum=hierarchy_receipt["receipt_checksum"],
                status=hierarchy_receipt["status"],
            ),
            "hierarchy_manifest": _artifact_meta(
                hierarchy_manifest_path,
                root=repo_root,
                manifest_checksum=hierarchy["manifest_checksum"],
            ),
        },
        "protocol": asdict(selected),
        "protocol_checksum": selected.checksum,
        "scientific_claims_authorized": False,
        "stage": "opened_evaluation_posthoc_seed_shift_diagnostic",
        "status": "FROZEN_BEFORE_T12_6_1B_POSTHOC_DIAGNOSTIC",
        "storage": {
            "maximum_artifact_bytes": selected.maximum_artifact_bytes,
            "maximum_sdk_calls": 0,
            "maximum_wall_seconds": selected.maximum_wall_seconds,
            "persist_archive_copies": False,
        },
    }
    manifest = _signed(payload, "manifest_checksum")
    _write_json_once(output_path, manifest)
    receipt = seed_shift_diagnostic_receipt(
        manifest=manifest,
        phase="freeze",
        passed=authorized,
        status=(
            "PASS_T12_6_1B_DIAGNOSTIC_FREEZE"
            if authorized
            else "DIRTY_SMOKE_ONLY"
        ),
        metrics={
            "evaluation_archive_count": len(evaluation_archives),
            "focal_eligible_groups": int(focal["eligible_groups"]),
            "focal_future_binding_hits": int(focal["future_binding_hits"]),
            "sdk_calls_used": 0,
            "training_archive_count": len(training_archives),
        },
    )
    _write_json_once(Path(output_path).with_name("freeze_receipt.json"), receipt)
    return manifest


def load_future_viability_seed_shift_diagnostic_manifest(
    path: str | Path,
    *,
    root: str | Path | None = None,
    verify_code: bool = True,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    _verify_signed(manifest, "manifest_checksum")
    if manifest.get("format_version") != SEED_SHIFT_MANIFEST_FORMAT:
        raise ValueError("unsupported T12.6.1b seed-shift manifest")
    protocol = FutureViabilitySeedShiftDiagnosticProtocol(**dict(manifest["protocol"]))
    if protocol.checksum != manifest.get("protocol_checksum"):
        raise ValueError("T12.6.1b protocol checksum mismatch")
    for values in manifest["inputs"].values():
        for meta in values:
            _verify_meta(meta, root=repo_root)
    for name, meta in manifest["parents"].items():
        if name.endswith("_artifacts"):
            for nested in meta.values():
                _verify_meta(nested, root=repo_root)
        else:
            _verify_meta(meta, root=repo_root)
    if verify_code:
        for relative, expected in manifest["code_sha256"].items():
            candidate = repo_root / relative
            if not candidate.is_file() or _file_sha256(candidate) != expected:
                raise ValueError(f"T12.6.1b code checksum mismatch: {relative}")
    return manifest


def seed_shift_diagnostic_receipt(
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
            "format_version": SEED_SHIFT_RECEIPT_FORMAT,
            "manifest_checksum": manifest["manifest_checksum"],
            "metrics": dict(metrics),
            "parent_t12_6_1_evaluation_receipt_checksum": manifest["parents"][
                "hierarchy_evaluation_receipt"
            ]["receipt_checksum"],
            "parent_t12_6_1a_diagnostic_receipt_checksum": manifest["parents"][
                "conflict_diagnostic_receipt"
            ]["receipt_checksum"],
            "passed": bool(passed),
            "phase": str(phase),
            "protocol_checksum": manifest["protocol_checksum"],
            "status": str(status),
        },
        "receipt_checksum",
    )


def load_seed_shift_diagnostic_receipt(
    path: str | Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
    expected_phase: str | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    _verify_signed(receipt, "receipt_checksum")
    if receipt.get("format_version") != SEED_SHIFT_RECEIPT_FORMAT:
        raise ValueError("unsupported T12.6.1b seed-shift receipt")
    if manifest is not None and (
        receipt.get("manifest_checksum") != manifest.get("manifest_checksum")
        or receipt.get("protocol_checksum") != manifest.get("protocol_checksum")
    ):
        raise ValueError("T12.6.1b receipt belongs to another manifest")
    if expected_phase is not None and receipt.get("phase") != expected_phase:
        raise ValueError("T12.6.1b receipt phase mismatch")
    for meta in receipt.get("artifacts", {}).values():
        _verify_meta(meta, root=repo_root)
    return receipt


__all__ = [
    "SEED_SHIFT_CODE_PATHS",
    "SEED_SHIFT_MANIFEST_FORMAT",
    "SEED_SHIFT_PROTOCOL_FORMAT",
    "SEED_SHIFT_RECEIPT_FORMAT",
    "FutureViabilitySeedShiftDiagnosticProtocol",
    "freeze_future_viability_seed_shift_diagnostic",
    "load_future_viability_seed_shift_diagnostic_manifest",
    "load_seed_shift_diagnostic_receipt",
    "seed_shift_diagnostic_receipt",
]
