"""Frozen protocol and signed artifacts for SAGE.T12.3c replay lineage."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from theory.sage11.splits import SAGE11_SPLITS

from .archive import GoExploreArchive
from .contracts import GroundedAction
from .experiment import (
    _bound_path,
    _file_sha256,
    _git_state,
    _read_json,
    _signed,
    _verify_signed,
    _write_json_once,
)
from .shield_protocol import (
    TerminalShieldProtocol,
    load_shield_manifest,
    load_shield_receipt,
)

LINEAGE_PROTOCOL_FORMAT = "sage-t12.3c-replay-lineage-protocol-v1"
LINEAGE_REGISTRY_FORMAT = "sage-t12.3c-replay-audit-registry-v1"
LINEAGE_MANIFEST_FORMAT = "sage-t12.3c-replay-lineage-manifest-v1"
LINEAGE_RECEIPT_FORMAT = "sage-t12.3c-replay-lineage-receipt-v1"

LINEAGE_CODE_PATHS = (
    "theory/sage_t/causal/lineage_archive.py",
    "theory/sage_t/causal/lineage_protocol.py",
    "theory/sage_t/causal/lineage_experiment.py",
    "theory/sage_t/causal/lineage_experiment_cli.py",
    "theory/sage_t/causal/archive.py",
    "theory/sage_t/causal/graph_experiment.py",
    "theory/sage_t/causal/burst_experiment.py",
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
class ReplayAuditCase:
    case_id: str
    case_kind: str
    game_id: str
    source_seed: int
    source_arm: str
    source_archive_path: str
    source_archive_sha256: str
    cell_id: str
    exact_hash: str
    prefix_id: str
    replay_failures: int
    actions: tuple[GroundedAction, ...]
    expected_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(
            self, "expected_hashes", tuple(str(value) for value in self.expected_hashes)
        )
        if self.case_kind not in {"failed", "matched_control"}:
            raise ValueError("unsupported replay audit case kind")
        if not self.actions or len(self.expected_hashes) != len(self.actions) + 1:
            raise ValueError("replay audit case needs one hash per prefix state")
        if self.expected_hashes[-1] != self.exact_hash:
            raise ValueError("replay audit case final hash mismatch")
        if self.case_kind == "failed" and self.replay_failures <= 0:
            raise ValueError("failed audit case lacks an observed replay failure")
        if self.case_kind == "matched_control" and self.replay_failures != 0:
            raise ValueError("matched replay control is not failure-free")

    @property
    def depth(self) -> int:
        return len(self.actions)

    @property
    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "case_kind": self.case_kind,
            "game_id": self.game_id,
            "source_seed": self.source_seed,
            "source_arm": self.source_arm,
            "source_archive_path": self.source_archive_path,
            "source_archive_sha256": self.source_archive_sha256,
            "cell_id": self.cell_id,
            "exact_hash": self.exact_hash,
            "prefix_id": self.prefix_id,
            "replay_failures": self.replay_failures,
            "actions": [
                {
                    "action_name": action.action_name,
                    "action_data": dict(action.action_data),
                }
                for action in self.actions
            ],
            "expected_hashes": list(self.expected_hashes),
        }

    @property
    def case_checksum(self) -> str:
        return _checksum(self.unsigned_payload)

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_payload, "case_checksum": self.case_checksum}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReplayAuditCase:
        case = cls(
            case_id=str(payload["case_id"]),
            case_kind=str(payload["case_kind"]),
            game_id=str(payload["game_id"]),
            source_seed=int(payload["source_seed"]),
            source_arm=str(payload["source_arm"]),
            source_archive_path=str(payload["source_archive_path"]),
            source_archive_sha256=str(payload["source_archive_sha256"]),
            cell_id=str(payload["cell_id"]),
            exact_hash=str(payload["exact_hash"]),
            prefix_id=str(payload["prefix_id"]),
            replay_failures=int(payload["replay_failures"]),
            actions=tuple(
                GroundedAction(
                    str(item["action_name"]),
                    dict(item.get("action_data", {}) or {}),
                )
                for item in payload["actions"]
            ),
            expected_hashes=tuple(str(value) for value in payload["expected_hashes"]),
        )
        if case.case_checksum != str(payload.get("case_checksum", "")):
            raise ValueError("replay audit case checksum mismatch")
        return case


@dataclass(frozen=True)
class ReplayLineageProtocol:
    format_version: str = LINEAGE_PROTOCOL_FORMAT
    audit_repetitions: int = 3
    maximum_audit_cases: int = 36
    maximum_audit_depth: int = 40
    minimum_failed_audit_cases: int = 12
    evaluation_seeds: tuple[int, ...] = (6803, 7101, 7102)
    evaluation_arms: tuple[str, ...] = (
        "shortest_prefix_control",
        "lineage_preserving",
    )
    burst_schedule: tuple[int, ...] = (4, 8, 16)
    sdk_calls_per_evaluation_arm: int = 3_500
    maximum_total_sdk_calls: int = 30_000
    maximum_artifact_bytes_per_run: int = 3 * 1024 * 1024 * 1024
    maximum_cells: int = 50_000
    minimum_treatment_replay_exact_rate: float = 0.95
    minimum_calibration_seed_replay_gain: float = 0.02
    minimum_coverage_ratio: float = 0.80
    maximum_progress_regression_seeds: int = 1
    minimum_reproduced_parent_failures: int = 1
    split_checksum: str = field(default_factory=lambda: SAGE11_SPLITS.checksum)

    def __post_init__(self) -> None:
        for name in ("evaluation_seeds", "burst_schedule"):
            object.__setattr__(
                self, name, tuple(int(value) for value in getattr(self, name))
            )
        object.__setattr__(
            self,
            "evaluation_arms",
            tuple(str(value) for value in self.evaluation_arms),
        )
        if self.format_version != LINEAGE_PROTOCOL_FORMAT:
            raise ValueError("unsupported T12.3c replay-lineage protocol")
        fixed = {
            "audit_repetitions": (self.audit_repetitions, 3),
            "maximum_audit_cases": (self.maximum_audit_cases, 36),
            "maximum_audit_depth": (self.maximum_audit_depth, 40),
            "minimum_failed_audit_cases": (self.minimum_failed_audit_cases, 12),
            "sdk_calls_per_evaluation_arm": (
                self.sdk_calls_per_evaluation_arm,
                3_500,
            ),
            "maximum_total_sdk_calls": (self.maximum_total_sdk_calls, 30_000),
            "maximum_artifact_bytes_per_run": (
                self.maximum_artifact_bytes_per_run,
                3 * 1024 * 1024 * 1024,
            ),
            "maximum_cells": (self.maximum_cells, 50_000),
            "minimum_treatment_replay_exact_rate": (
                self.minimum_treatment_replay_exact_rate,
                0.95,
            ),
            "minimum_calibration_seed_replay_gain": (
                self.minimum_calibration_seed_replay_gain,
                0.02,
            ),
            "minimum_coverage_ratio": (self.minimum_coverage_ratio, 0.80),
            "maximum_progress_regression_seeds": (
                self.maximum_progress_regression_seeds,
                1,
            ),
            "minimum_reproduced_parent_failures": (
                self.minimum_reproduced_parent_failures,
                1,
            ),
        }
        for name, (observed, expected) in fixed.items():
            if observed != expected:
                raise ValueError(f"T12.3c preregistered value changed: {name}")
        if self.evaluation_seeds != (6803, 7101, 7102):
            raise ValueError("T12.3c evaluation seeds are frozen")
        if self.evaluation_arms != (
            "shortest_prefix_control",
            "lineage_preserving",
        ):
            raise ValueError("T12.3c evaluation arms are frozen")
        if self.burst_schedule != (4, 8, 16):
            raise ValueError("T12.3c burst schedule is frozen")
        audit_upper_bound = (
            self.maximum_audit_cases
            * self.audit_repetitions
            * (1 + self.maximum_audit_depth)
        )
        evaluation_upper_bound = (
            len(self.evaluation_seeds)
            * len(self.evaluation_arms)
            * self.sdk_calls_per_evaluation_arm
        )
        if audit_upper_bound + evaluation_upper_bound > self.maximum_total_sdk_calls:
            raise ValueError("T12.3c planned calls exceed the global SDK budget")

    @property
    def checksum(self) -> str:
        return _checksum(asdict(self))


def _resolve_bound(path: str, *, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _parent_failure_is_replay_only(
    manifest: Mapping[str, Any], receipt: Mapping[str, Any]
) -> bool:
    if receipt.get("passed") is not False:
        return False
    if receipt.get("status") != "FAIL_T12_3B_TERMINAL_SHIELD_GATE":
        return False
    protocol = TerminalShieldProtocol(**dict(manifest["protocol"]))
    metrics = dict(receipt.get("metrics", {}))
    shield = dict(metrics.get("source_shield", {}))
    exact_rate = float(metrics.get("minimum_evaluation_replay_exact_rate", 1.0))
    checks = (
        int(metrics.get("confirmed_terminal_candidates", 0))
        >= protocol.minimum_confirmed_terminal_traces,
        float(metrics.get("terminal_confirmation_rate", 0.0))
        >= protocol.minimum_terminal_confirmation_rate,
        int(metrics.get("confirmed_terminal_source_groups", 0))
        == len(protocol.source_seeds) * len(protocol.source_arms),
        bool(shield.get("multi_step_hazard_observed", False)),
        int(shield.get("confirmed_unsafe_actions", 0)) >= 1,
        int(metrics.get("witness_confirmations", -1))
        == int(metrics.get("expected_witness_confirmations", -2)),
        int(metrics.get("witness_vetoes", -1)) == 0,
        bool(metrics.get("all_witness_actions_protected", False)),
        int(metrics.get("evaluation_vetoes", 0)) >= protocol.minimum_vetoes,
        float(metrics.get("terminal_rate_ratio", float("inf")))
        <= protocol.maximum_terminal_rate_ratio,
        int(metrics.get("terminal_regression_seeds", 99))
        <= protocol.maximum_terminal_regression_seeds,
        float(metrics.get("coverage_ratio", 0.0)) >= protocol.minimum_coverage_ratio,
        float(metrics.get("minimum_per_seed_coverage_ratio", 0.0))
        >= protocol.minimum_coverage_ratio,
        int(metrics.get("treatment_progress_edges", -1))
        >= int(metrics.get("control_progress_edges", 0)),
        int(metrics.get("progress_regression_seeds", 99)) == 0,
        int(metrics.get("sdk_calls", protocol.maximum_total_sdk_calls + 1))
        <= protocol.maximum_total_sdk_calls,
    )
    return bool(
        all(checks) and exact_rate < protocol.minimum_evaluation_replay_exact_rate
    )


def _case_from_variant(
    *,
    archive: GoExploreArchive,
    archive_meta: Mapping[str, Any],
    game_id: str,
    seed: int,
    arm: str,
    cell_id: str,
    variant: Any,
    kind: str,
) -> ReplayAuditCase:
    actions = archive.prefixes.actions(variant.prefix_id)
    edges = archive.path_edges(variant)
    if len(actions) != len(edges):
        raise ValueError("T12.3c source prefix/action path mismatch")
    if any(action.key != edge.action.key for action, edge in zip(actions, edges)):
        raise ValueError("T12.3c source prefix/edge action mismatch")
    expected = (edges[0].source_exact_hash,) + tuple(
        edge.target_exact_hash for edge in edges
    )
    token = _checksum(
        {
            "kind": kind,
            "archive": archive_meta["sha256"],
            "cell": cell_id,
            "prefix": variant.prefix_id,
        }
    )[:20]
    return ReplayAuditCase(
        case_id=f"replay_{kind}_{token}",
        case_kind=kind,
        game_id=game_id,
        source_seed=seed,
        source_arm=arm,
        source_archive_path=str(archive_meta["path"]),
        source_archive_sha256=str(archive_meta["sha256"]),
        cell_id=cell_id,
        exact_hash=variant.exact_hash,
        prefix_id=variant.prefix_id,
        replay_failures=variant.replay_failures,
        actions=actions,
        expected_hashes=expected,
    )


def extract_replay_audit_cases(
    paired_evaluation: Mapping[str, Any],
    *,
    protocol: ReplayLineageProtocol,
) -> tuple[ReplayAuditCase, ...]:
    failed: list[ReplayAuditCase] = []
    matched: list[ReplayAuditCase] = []
    for condition in paired_evaluation.get("conditions", ()):
        game_id = str(condition["game_id"])
        seed = int(condition["seed"])
        for arm, raw_arm in sorted(dict(condition["arms"]).items()):
            meta = dict(raw_arm["archive"])
            archive_path = Path(str(meta["path"]))
            if not archive_path.is_file() or _file_sha256(archive_path) != meta["sha256"]:
                raise ValueError("T12.3c source archive checksum mismatch")
            archive = GoExploreArchive.from_dict(_read_json(archive_path))
            for cell_id, cell in sorted(archive.cells.items()):
                bad = sorted(
                    (
                        item
                        for item in cell.variants.values()
                        if 0 < item.replay_failures
                        and archive.prefixes.depth(item.prefix_id)
                        <= protocol.maximum_audit_depth
                    ),
                    key=lambda item: (archive.prefixes.depth(item.prefix_id), item.exact_hash),
                )
                good = tuple(
                    item
                    for item in cell.variants.values()
                    if item.replay_failures == 0
                    and 0 < archive.prefixes.depth(item.prefix_id)
                    <= protocol.maximum_audit_depth
                )
                for variant in bad:
                    failed.append(
                        _case_from_variant(
                            archive=archive,
                            archive_meta=meta,
                            game_id=game_id,
                            seed=seed,
                            arm=arm,
                            cell_id=cell_id,
                            variant=variant,
                            kind="failed",
                        )
                    )
                    if good:
                        control = min(
                            good,
                            key=lambda item: (
                                abs(
                                    archive.prefixes.depth(item.prefix_id)
                                    - archive.prefixes.depth(variant.prefix_id)
                                ),
                                archive.prefixes.depth(item.prefix_id),
                                item.exact_hash,
                            ),
                        )
                        matched.append(
                            _case_from_variant(
                                archive=archive,
                                archive_meta=meta,
                                game_id=game_id,
                                seed=seed,
                                arm=arm,
                                cell_id=cell_id,
                                variant=control,
                                kind="matched_control",
                            )
                        )
    failed = sorted(failed, key=lambda item: item.case_id)
    matched = sorted(matched, key=lambda item: item.case_id)
    cases = tuple((failed + matched)[: protocol.maximum_audit_cases])
    if len(failed) < protocol.minimum_failed_audit_cases:
        raise ValueError("T12.3c lacks enough parent replay failures")
    if {item.source_arm for item in failed} != {
        "burst_control",
        "burst_terminal_shield",
    }:
        raise ValueError("T12.3c replay failures are not present in both parent arms")
    if not matched or len(cases) > protocol.maximum_audit_cases:
        raise ValueError("T12.3c matched replay audit registry is invalid")
    return cases


def freeze_lineage_experiment(
    *,
    output_path: str | Path,
    audit_registry_path: str | Path,
    parent_manifest_path: str | Path,
    parent_receipt_path: str | Path,
    root: str | Path | None = None,
    allow_dirty: bool = False,
    protocol: ReplayLineageProtocol | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    selected = protocol or ReplayLineageProtocol()
    parent_manifest = load_shield_manifest(
        parent_manifest_path, root=repo_root, verify_code=False
    )
    parent_receipt = load_shield_receipt(
        parent_receipt_path, manifest=parent_manifest, root=repo_root
    )
    if parent_manifest.get("stage") != "source_train":
        raise ValueError("T12.3c is restricted to source_train")
    if not _parent_failure_is_replay_only(parent_manifest, parent_receipt):
        raise ValueError("T12.3c requires the replay-only T12.3b gate failure")
    evaluation_meta = dict(parent_receipt["artifacts"]["paired_evaluation"])
    evaluation_path = _resolve_bound(str(evaluation_meta["path"]), root=repo_root)
    cases = extract_replay_audit_cases(
        _read_json(evaluation_path), protocol=selected
    )
    registry = _signed(
        {
            "format_version": LINEAGE_REGISTRY_FORMAT,
            "protocol_checksum": selected.checksum,
            "parent_t12_3b_receipt_checksum": parent_receipt["receipt_checksum"],
            "cases": [item.to_dict() for item in cases],
        },
        "registry_checksum",
    )
    missing = [path for path in LINEAGE_CODE_PATHS if not (repo_root / path).is_file()]
    if missing:
        raise ValueError(f"T12.3c code inventory is incomplete: {missing}")
    git = _git_state(repo_root)
    if git["dirty"] and not allow_dirty:
        raise ValueError("scientific freeze requires a clean worktree")
    authorized = bool(not git["dirty"] and parent_manifest.get("scientific_claims_authorized"))
    _write_json_once(audit_registry_path, registry)
    payload = {
        "format_version": LINEAGE_MANIFEST_FORMAT,
        "status": "FROZEN_BEFORE_T12_3C_REPLAY_LINEAGE",
        "stage": "source_train",
        "game_id": parent_manifest["game_id"],
        "protocol": asdict(selected),
        "protocol_checksum": selected.checksum,
        "audit_registry": {
            "path": _bound_path(audit_registry_path, root=repo_root),
            "sha256": _file_sha256(audit_registry_path),
            "registry_checksum": registry["registry_checksum"],
        },
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
                "passed": False,
                "status": "FAIL_T12_3B_TERMINAL_SHIELD_GATE",
                "failure_class": "REPLAY_EXACT_ONLY",
            },
        },
        "code_sha256": {
            path: _file_sha256(repo_root / path) for path in LINEAGE_CODE_PATHS
        },
        "git": git,
        "scientific_claims_authorized": authorized,
        "firewall": {
            "holdout_opened": False,
            "source_validation_opened": False,
            "production_authority": False,
            "replay_lineage_experiment_authorized": authorized,
            "t12_3b_child_rerun_authorized": False,
            "terminal_shield_production_authority": False,
            "neural_training_authorized": False,
            "option_extraction_authorized": False,
        },
        "storage": {
            "maximum_artifact_bytes_per_run": selected.maximum_artifact_bytes_per_run,
            "persist_raw_frames": False,
            "hard_fail_before_write": True,
        },
    }
    manifest = _signed(payload, "manifest_checksum")
    _write_json_once(output_path, manifest)
    freeze_receipt = lineage_phase_receipt(
        manifest=manifest,
        phase="freeze",
        passed=authorized,
        status="PASS_T12_3C_FREEZE" if authorized else "DIRTY_SMOKE_ONLY",
        metrics={
            "audit_cases": len(cases),
            "failed_cases": sum(item.case_kind == "failed" for item in cases),
            "matched_controls": sum(
                item.case_kind == "matched_control" for item in cases
            ),
        },
    )
    _write_json_once(Path(output_path).with_name("freeze_receipt.json"), freeze_receipt)
    return manifest


def load_lineage_registry(
    path: str | Path, *, protocol: ReplayLineageProtocol | None = None
) -> tuple[dict[str, Any], tuple[ReplayAuditCase, ...]]:
    payload = _read_json(path)
    _verify_signed(payload, "registry_checksum")
    if payload.get("format_version") != LINEAGE_REGISTRY_FORMAT:
        raise ValueError("unsupported T12.3c replay audit registry")
    selected = protocol or ReplayLineageProtocol()
    if payload.get("protocol_checksum") != selected.checksum:
        raise ValueError("T12.3c replay audit registry protocol mismatch")
    cases = tuple(ReplayAuditCase.from_dict(dict(item)) for item in payload["cases"])
    if not cases or len(cases) > selected.maximum_audit_cases:
        raise ValueError("T12.3c replay audit registry is incomplete")
    return payload, cases


def load_lineage_manifest(
    path: str | Path,
    *,
    root: str | Path | None = None,
    verify_code: bool = True,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = _read_json(path)
    _verify_signed(manifest, "manifest_checksum")
    if manifest.get("format_version") != LINEAGE_MANIFEST_FORMAT:
        raise ValueError("unsupported T12.3c replay-lineage manifest")
    protocol = ReplayLineageProtocol(**dict(manifest["protocol"]))
    if protocol.checksum != manifest.get("protocol_checksum"):
        raise ValueError("T12.3c protocol checksum mismatch")
    registry_meta = dict(manifest["audit_registry"])
    registry_path = _resolve_bound(str(registry_meta["path"]), root=repo_root)
    if not registry_path.is_file() or _file_sha256(registry_path) != registry_meta["sha256"]:
        raise ValueError("T12.3c replay audit registry checksum mismatch")
    registry, _ = load_lineage_registry(registry_path, protocol=protocol)
    if registry["registry_checksum"] != registry_meta["registry_checksum"]:
        raise ValueError("T12.3c replay audit registry signature mismatch")
    for name in ("manifest", "receipt"):
        meta = dict(manifest["parent"][name])
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError(f"T12.3c parent {name} checksum mismatch")
    if verify_code:
        for relative, expected in dict(manifest["code_sha256"]).items():
            candidate = repo_root / relative
            if not candidate.is_file() or _file_sha256(candidate) != expected:
                raise ValueError(f"T12.3c code checksum mismatch: {relative}")
    return manifest


def lineage_phase_receipt(
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
            "format_version": LINEAGE_RECEIPT_FORMAT,
            "phase": str(phase),
            "passed": bool(passed),
            "status": str(status),
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "parent_t12_3b_receipt_checksum": manifest["parent"]["receipt"][
                "receipt_checksum"
            ],
            "metrics": dict(metrics),
            "artifacts": dict(artifacts or {}),
        },
        "receipt_checksum",
    )


def load_lineage_receipt(
    path: str | Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    receipt = _read_json(path)
    _verify_signed(receipt, "receipt_checksum")
    if receipt.get("format_version") != LINEAGE_RECEIPT_FORMAT:
        raise ValueError("unsupported T12.3c receipt")
    if manifest is not None:
        if receipt.get("manifest_checksum") != manifest.get("manifest_checksum"):
            raise ValueError("T12.3c receipt belongs to another manifest")
        if receipt.get("protocol_checksum") != manifest.get("protocol_checksum"):
            raise ValueError("T12.3c receipt belongs to another protocol")
    for name, raw_meta in dict(receipt.get("artifacts", {})).items():
        meta = dict(raw_meta)
        artifact = _resolve_bound(str(meta.get("path", "")), root=repo_root)
        if not artifact.is_file() or _file_sha256(artifact) != meta.get("sha256"):
            raise ValueError(f"T12.3c receipt artifact checksum mismatch: {name}")
        if name == "paired_evaluation":
            evaluation = _read_json(artifact)
            for condition in evaluation.get("conditions", ()):
                for arm_name, arm in dict(condition.get("arms", {})).items():
                    for artifact_name in ("archive", "excursions"):
                        nested = dict(arm[artifact_name])
                        nested_path = _resolve_bound(str(nested["path"]), root=repo_root)
                        if not nested_path.is_file() or _file_sha256(nested_path) != nested["sha256"]:
                            raise ValueError(
                                "T12.3c paired artifact checksum mismatch: "
                                f"{arm_name}:{artifact_name}"
                            )
    return receipt


__all__ = [
    "ReplayAuditCase",
    "ReplayLineageProtocol",
    "extract_replay_audit_cases",
    "freeze_lineage_experiment",
    "lineage_phase_receipt",
    "load_lineage_manifest",
    "load_lineage_receipt",
    "load_lineage_registry",
]
