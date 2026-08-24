"""Frozen post-hoc-only protocol for the SAGE.T12.6a diagnostic."""

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
from .future_viability_protocol import (
    FUTURE_VIABILITY_MANIFEST_FORMAT,
    FutureViabilityProtocol,
    _resolve_bound,
    load_future_viability_receipt,
)

FUTURE_VIABILITY_DIAGNOSTIC_PROTOCOL_FORMAT = (
    "sage-t12.6a-future-viability-diagnostic-protocol-v1"
)
FUTURE_VIABILITY_DIAGNOSTIC_MANIFEST_FORMAT = (
    "sage-t12.6a-future-viability-diagnostic-manifest-v1"
)
FUTURE_VIABILITY_DIAGNOSTIC_RECEIPT_FORMAT = (
    "sage-t12.6a-future-viability-diagnostic-receipt-v1"
)

FUTURE_VIABILITY_DIAGNOSTIC_CODE_PATHS = (
    "theory/sage_t/causal/experiment.py",
    "theory/sage_t/causal/future_viability.py",
    "theory/sage_t/causal/future_viability_protocol.py",
    "theory/sage_t/causal/future_viability_diagnostic.py",
    "theory/sage_t/causal/future_viability_diagnostic_protocol.py",
    "theory/sage_t/causal/future_viability_diagnostic_experiment.py",
    "theory/sage_t/causal/future_viability_diagnostic_cli.py",
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
        raise ValueError(f"T12.6a bound artifact mismatch: {path}")
    return path


@dataclass(frozen=True)
class FutureViabilityDiagnosticProtocol:
    """Immutable axes for an explanatory audit of one registered fold miss."""

    format_version: str = FUTURE_VIABILITY_DIAGNOSTIC_PROTOCOL_FORMAT
    parent_status: str = "FAIL_T12_6_FUTURE_VIABILITY_IDENTIFICATION_GATE"
    parent_classification: str = "FUTURE_VIABILITY_NOT_IDENTIFIED"
    parent_failed_check: str = "every_compile_lineage_accuracy_sufficient"
    focal_search_seed: int = 9_103
    focal_lineage_seed: int = 8_701
    reference_lineage_seed: int = 8_705
    reference_search_seeds: tuple[int, ...] = (9_101, 9_102)
    expected_focal_eligible_groups: int = 43
    expected_focal_hits: int = 30
    diagnostic_axes: tuple[str, ...] = (
        "support_tier",
        "score_ties",
        "local_signature_label_heterogeneity",
        "lineage_conditioning",
        "archive_arm_conditioning",
    )
    maximum_sdk_calls: int = 0
    maximum_wall_seconds: int = 600
    maximum_artifact_bytes: int = 512 * 1024 * 1024
    persist_archive_copies: bool = False
    evaluation_archive_payloads_authorized: bool = False
    confirmatory_claim_authorized: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "reference_search_seeds", tuple(self.reference_search_seeds)
        )
        object.__setattr__(self, "diagnostic_axes", tuple(self.diagnostic_axes))
        expected = {
            "format_version": FUTURE_VIABILITY_DIAGNOSTIC_PROTOCOL_FORMAT,
            "parent_status": "FAIL_T12_6_FUTURE_VIABILITY_IDENTIFICATION_GATE",
            "parent_classification": "FUTURE_VIABILITY_NOT_IDENTIFIED",
            "parent_failed_check": "every_compile_lineage_accuracy_sufficient",
            "focal_search_seed": 9_103,
            "focal_lineage_seed": 8_701,
            "reference_lineage_seed": 8_705,
            "reference_search_seeds": (9_101, 9_102),
            "expected_focal_eligible_groups": 43,
            "expected_focal_hits": 30,
            "diagnostic_axes": (
                "support_tier",
                "score_ties",
                "local_signature_label_heterogeneity",
                "lineage_conditioning",
                "archive_arm_conditioning",
            ),
            "maximum_sdk_calls": 0,
            "maximum_wall_seconds": 600,
            "maximum_artifact_bytes": 512 * 1024 * 1024,
            "persist_archive_copies": False,
            "evaluation_archive_payloads_authorized": False,
            "confirmatory_claim_authorized": False,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"T12.6a preregistered value changed: {name}")

    @property
    def checksum(self) -> str:
        return _checksum(asdict(self))


def _load_parent_manifest_for_diagnostic(
    path: Path, *, root: Path
) -> dict[str, Any]:
    manifest = _read_json(path)
    _verify_signed(manifest, "manifest_checksum")
    if manifest.get("format_version") != FUTURE_VIABILITY_MANIFEST_FORMAT:
        raise ValueError("T12.6a parent manifest format changed")
    parent_protocol = FutureViabilityProtocol(**dict(manifest["protocol"]))
    if parent_protocol.checksum != manifest.get("protocol_checksum"):
        raise ValueError("T12.6a parent protocol checksum mismatch")
    for relative, expected in manifest.get("code_sha256", {}).items():
        candidate = root / str(relative)
        if not candidate.is_file() or _file_sha256(candidate) != str(expected):
            raise ValueError(f"T12.6a parent code checksum mismatch: {relative}")
    return manifest


def freeze_future_viability_diagnostic(
    *,
    output_path: str | Path,
    parent_manifest_path: str | Path,
    parent_compile_receipt_path: str | Path,
    root: str | Path | None = None,
    protocol: FutureViabilityDiagnosticProtocol | None = None,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    selected = protocol or FutureViabilityDiagnosticProtocol()
    parent_manifest_path = Path(parent_manifest_path).resolve()
    parent_compile_receipt_path = Path(parent_compile_receipt_path).resolve()
    parent_manifest = _load_parent_manifest_for_diagnostic(
        parent_manifest_path, root=repo_root
    )
    parent_receipt = load_future_viability_receipt(
        parent_compile_receipt_path,
        manifest=parent_manifest,
        root=repo_root,
        expected_phase="compile",
    )
    if parent_receipt.get("status") != selected.parent_status:
        raise ValueError("T12.6a parent status changed")
    if parent_receipt.get("passed") is not False:
        raise ValueError("T12.6a requires the preserved negative compile receipt")
    metrics = dict(parent_receipt.get("metrics", {}))
    if metrics.get("classification") != selected.parent_classification:
        raise ValueError("T12.6a parent classification changed")
    checks = dict(metrics.get("checks", {}))
    failed = {name for name, value in checks.items() if value is not True}
    if failed != {selected.parent_failed_check}:
        raise ValueError(f"T12.6a parent failure set changed: {sorted(failed)}")

    crossfit_meta = parent_receipt["artifacts"]["crossfit"]
    crossfit_path = _verify_meta(crossfit_meta, root=repo_root)
    crossfit = _read_json(crossfit_path)["crossfit"]
    focal_fold = next(
        (
            fold
            for fold in crossfit["folds"]
            if int(fold["holdout_search_seed"]) == selected.focal_search_seed
        ),
        None,
    )
    if focal_fold is None:
        raise ValueError("T12.6a focal fold is missing")
    focal = focal_fold["metrics"]["per_lineage"][str(selected.focal_lineage_seed)]
    if (
        int(focal["eligible_groups"]) != selected.expected_focal_eligible_groups
        or int(focal["future_binding_hits"]) != selected.expected_focal_hits
    ):
        raise ValueError("T12.6a focal miss changed")

    training_archives = []
    for meta in parent_manifest["inputs"]["training_archives"]:
        archive_path = _verify_meta(meta, root=repo_root)
        training_archives.append(
            {
                **{
                    key: value
                    for key, value in meta.items()
                    if key not in {"path", "sha256"}
                },
                "path": _bound_path(archive_path, root=repo_root),
                "sha256": str(meta["sha256"]),
            }
        )
    if len(training_archives) != 12:
        raise ValueError("T12.6a training archive count changed")

    missing_code = [
        relative
        for relative in FUTURE_VIABILITY_DIAGNOSTIC_CODE_PATHS
        if not (repo_root / relative).is_file()
    ]
    if missing_code:
        raise ValueError(f"T12.6a code inventory incomplete: {missing_code}")
    git = _git_state(repo_root)
    if git["dirty"] and not allow_dirty:
        raise ValueError("scientific freeze requires a clean worktree")
    authorized = bool(not git["dirty"])

    payload = {
        "claim_boundary": {
            "authorized": (
                "post-hoc explanation of the sealed T12.6 compile miss on "
                "source-train archives only"
            ),
            "not_authorized": [
                "T12.6 gate repair",
                "sealed evaluation access",
                "level progress",
                "physical collection",
                "generic ARC-AGI improvement",
                "source validation",
                "holdout performance",
                "controller authority",
                "neural training",
                "production authority",
            ],
        },
        "code_sha256": {
            relative: _file_sha256(repo_root / relative)
            for relative in FUTURE_VIABILITY_DIAGNOSTIC_CODE_PATHS
        },
        "design": {
            "axes_frozen_before_individual_error_inspection": True,
            "evaluation_archive_payloads_excluded": True,
            "parent_gate_is_immutable": True,
            "posthoc_only": True,
            "same_version_rerun_forbidden": True,
            "threshold_retuning_forbidden": True,
            "zero_sdk_calls": True,
        },
        "firewall": {
            "diagnostic_authorized": authorized,
            "evaluation_authorized": False,
            "environment_collection_authorized": False,
            "source_validation_opened": False,
            "holdout_opened": False,
            "controller_authority": False,
            "neural_training_authorized": False,
            "production_authority": False,
            "future_protocol_freeze_authorized": False,
        },
        "format_version": FUTURE_VIABILITY_DIAGNOSTIC_MANIFEST_FORMAT,
        "game_id": parent_manifest["game_id"],
        "git": git,
        "inputs": {"training_archives": training_archives},
        "parent": {
            "compile_artifacts": {
                name: dict(meta)
                for name, meta in parent_receipt["artifacts"].items()
            },
            "compile_receipt": _artifact_meta(
                parent_compile_receipt_path,
                root=repo_root,
                receipt_checksum=parent_receipt["receipt_checksum"],
                status=parent_receipt["status"],
            ),
            "manifest": _artifact_meta(
                parent_manifest_path,
                root=repo_root,
                manifest_checksum=parent_manifest["manifest_checksum"],
            ),
        },
        "protocol": asdict(selected),
        "protocol_checksum": selected.checksum,
        "scientific_claims_authorized": False,
        "stage": "source_train_posthoc_diagnostic",
        "status": "FROZEN_BEFORE_T12_6A_POSTHOC_DIAGNOSTIC",
        "storage": {
            "maximum_artifact_bytes": selected.maximum_artifact_bytes,
            "maximum_sdk_calls": 0,
            "maximum_wall_seconds": selected.maximum_wall_seconds,
            "persist_archive_copies": False,
        },
    }
    manifest = _signed(payload, "manifest_checksum")
    _write_json_once(output_path, manifest)
    receipt = future_viability_diagnostic_receipt(
        manifest=manifest,
        phase="freeze",
        passed=authorized,
        status=(
            "PASS_T12_6A_DIAGNOSTIC_FREEZE" if authorized else "DIRTY_SMOKE_ONLY"
        ),
        metrics={
            "evaluation_archive_payloads_loaded": 0,
            "focal_eligible_groups": int(focal["eligible_groups"]),
            "focal_future_binding_hits": int(focal["future_binding_hits"]),
            "training_archive_count": len(training_archives),
        },
    )
    _write_json_once(Path(output_path).with_name("freeze_receipt.json"), receipt)
    return manifest


def load_future_viability_diagnostic_manifest(
    path: str | Path,
    *,
    root: str | Path | None = None,
    verify_code: bool = True,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = _read_json(path)
    _verify_signed(manifest, "manifest_checksum")
    if manifest.get("format_version") != FUTURE_VIABILITY_DIAGNOSTIC_MANIFEST_FORMAT:
        raise ValueError("unsupported T12.6a diagnostic manifest")
    protocol = FutureViabilityDiagnosticProtocol(**dict(manifest["protocol"]))
    if protocol.checksum != manifest.get("protocol_checksum"):
        raise ValueError("T12.6a diagnostic protocol checksum mismatch")
    for meta in manifest["inputs"]["training_archives"]:
        _verify_meta(meta, root=repo_root)
    _verify_meta(manifest["parent"]["manifest"], root=repo_root)
    _verify_meta(manifest["parent"]["compile_receipt"], root=repo_root)
    for meta in manifest["parent"]["compile_artifacts"].values():
        _verify_meta(meta, root=repo_root)
    if verify_code:
        for relative, expected in manifest["code_sha256"].items():
            candidate = repo_root / relative
            if not candidate.is_file() or _file_sha256(candidate) != expected:
                raise ValueError(f"T12.6a code checksum mismatch: {relative}")
    return manifest


def future_viability_diagnostic_receipt(
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
            "format_version": FUTURE_VIABILITY_DIAGNOSTIC_RECEIPT_FORMAT,
            "manifest_checksum": manifest["manifest_checksum"],
            "metrics": dict(metrics),
            "parent_t12_6_compile_receipt_checksum": manifest["parent"][
                "compile_receipt"
            ]["receipt_checksum"],
            "passed": bool(passed),
            "phase": str(phase),
            "protocol_checksum": manifest["protocol_checksum"],
            "status": str(status),
        },
        "receipt_checksum",
    )


def load_future_viability_diagnostic_receipt(
    path: str | Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
    expected_phase: str | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    receipt = _read_json(path)
    _verify_signed(receipt, "receipt_checksum")
    if receipt.get("format_version") != FUTURE_VIABILITY_DIAGNOSTIC_RECEIPT_FORMAT:
        raise ValueError("unsupported T12.6a diagnostic receipt")
    if manifest is not None and (
        receipt.get("manifest_checksum") != manifest.get("manifest_checksum")
        or receipt.get("protocol_checksum") != manifest.get("protocol_checksum")
    ):
        raise ValueError("T12.6a receipt belongs to another manifest")
    if expected_phase is not None and receipt.get("phase") != expected_phase:
        raise ValueError("T12.6a receipt phase mismatch")
    for meta in receipt.get("artifacts", {}).values():
        _verify_meta(meta, root=repo_root)
    return receipt


__all__ = [
    "FUTURE_VIABILITY_DIAGNOSTIC_CODE_PATHS",
    "FUTURE_VIABILITY_DIAGNOSTIC_MANIFEST_FORMAT",
    "FUTURE_VIABILITY_DIAGNOSTIC_PROTOCOL_FORMAT",
    "FUTURE_VIABILITY_DIAGNOSTIC_RECEIPT_FORMAT",
    "FutureViabilityDiagnosticProtocol",
    "freeze_future_viability_diagnostic",
    "future_viability_diagnostic_receipt",
    "load_future_viability_diagnostic_manifest",
    "load_future_viability_diagnostic_receipt",
]
