"""Frozen T12.4a.2 protocol for exact reconfirmation of two progress routes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from theory.sage11.splits import SAGE11_SPLITS

from .archive import GoExploreArchive
from .calibration_protocol import (
    load_calibration_manifest,
    load_calibration_receipt,
)
from .experiment import (
    _bound_path,
    _file_sha256,
    _git_state,
    _read_json,
    _signed,
    _verify_signed,
    _write_json_once,
)
from .witness_protocol import ProgressWitness, WitnessStep, common_action_suffix

RECONFIRM_PROTOCOL_FORMAT = "sage-t12.4a.2-witness-reconfirmation-protocol-v1"
RECONFIRM_REGISTRY_FORMAT = "sage-t12.4a.2-witness-registry-v1"
RECONFIRM_MANIFEST_FORMAT = "sage-t12.4a.2-witness-manifest-v1"
RECONFIRM_RECEIPT_FORMAT = "sage-t12.4a.2-witness-receipt-v1"

RECONFIRM_CODE_PATHS = (
    "theory/sage_t/causal/witness_reconfirmation_protocol.py",
    "theory/sage_t/causal/witness_reconfirmation_experiment.py",
    "theory/sage_t/causal/witness_reconfirmation_cli.py",
    "theory/sage_t/causal/witness_protocol.py",
    "theory/sage_t/causal/witness_experiment.py",
    "theory/sage_t/causal/calibration_protocol.py",
    "theory/sage_t/causal/archive.py",
    "theory/sage_t/causal/graph_experiment.py",
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
class WitnessReconfirmationProtocol:
    format_version: str = RECONFIRM_PROTOCOL_FORMAT
    source_seeds: tuple[int, ...] = (8_701, 8_705)
    expected_route_lengths: tuple[int, ...] = (64, 61)
    expected_target_level: int = 1
    expected_common_suffix: tuple[str, ...] = (
        "ACTION3",
        "ACTION4",
        "ACTION4",
        "ACTION4",
        "ACTION3",
        "ACTION3",
    )
    repetitions_per_route: int = 3
    repetitions_per_suffix_branch: int = 3
    minimum_successful_route_replays: int = 3
    minimum_successful_suffix_replays: int = 3
    minimum_paired_contrasts: int = 3
    minimum_step_exact_rate: float = 1.0
    maximum_witness_steps: int = 128
    maximum_sdk_calls: int = 2_048
    maximum_artifact_bytes_per_run: int = 3 * 1024 * 1024 * 1024
    split_checksum: str = field(default_factory=lambda: SAGE11_SPLITS.checksum)

    def __post_init__(self) -> None:
        for name in (
            "source_seeds",
            "expected_route_lengths",
            "expected_common_suffix",
        ):
            caster = str if name == "expected_common_suffix" else int
            object.__setattr__(
                self,
                name,
                tuple(caster(value) for value in getattr(self, name)),
            )
        if self.format_version != RECONFIRM_PROTOCOL_FORMAT:
            raise ValueError("unsupported T12.4a.2 witness protocol")
        expected = {
            "source_seeds": (8_701, 8_705),
            "expected_route_lengths": (64, 61),
            "expected_target_level": 1,
            "expected_common_suffix": (
                "ACTION3",
                "ACTION4",
                "ACTION4",
                "ACTION4",
                "ACTION3",
                "ACTION3",
            ),
            "repetitions_per_route": 3,
            "repetitions_per_suffix_branch": 3,
            "minimum_successful_route_replays": 3,
            "minimum_successful_suffix_replays": 3,
            "minimum_paired_contrasts": 3,
            "minimum_step_exact_rate": 1.0,
            "maximum_witness_steps": 128,
            "maximum_sdk_calls": 2_048,
            "maximum_artifact_bytes_per_run": 3 * 1024 * 1024 * 1024,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"T12.4a.2 preregistered value changed: {name}")

    @property
    def checksum(self) -> str:
        return _checksum(asdict(self))


def _resolve_bound(path: str, *, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _load_archive(meta: Mapping[str, Any]) -> GoExploreArchive:
    path = Path(str(meta["path"]))
    if not path.is_file() or _file_sha256(path) != str(meta["sha256"]):
        raise ValueError(f"T12.4a.2 archive checksum mismatch: {path}")
    return GoExploreArchive.from_dict(_read_json(path))


def _archive_artifacts(collection_receipt: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    collection_meta = dict(collection_receipt["artifacts"]["collection"])
    collection_path = Path(str(collection_meta["path"]))
    if not collection_path.is_file() or _file_sha256(collection_path) != str(
        collection_meta["sha256"]
    ):
        raise ValueError("T12.4a.2 collection checksum mismatch")
    collection = _read_json(collection_path)
    return tuple(
        {
            **dict(condition["archive"]),
            "game_id": str(condition["game_id"]),
            "seed": int(condition["seed"]),
            "split": str(condition["split"]),
        }
        for condition in collection.get("conditions", ())
    )


def extract_reconfirmation_witnesses(
    archive_artifacts: Sequence[Mapping[str, Any]],
    *,
    protocol: WitnessReconfirmationProtocol | None = None,
) -> tuple[ProgressWitness, ...]:
    selected = protocol or WitnessReconfirmationProtocol()
    expected_lengths = dict(zip(selected.source_seeds, selected.expected_route_lengths))
    witnesses = []
    all_progress = []
    for raw_meta in archive_artifacts:
        meta = dict(raw_meta)
        archive = _load_archive(meta)
        progress_edges = [
            edge
            for edge in sorted(archive.edges.values(), key=lambda item: item.ordinal)
            if edge.level_delta > 0 or edge.success
        ]
        all_progress.extend((int(meta["seed"]), edge.edge_id) for edge in progress_edges)
        seed = int(meta["seed"])
        if seed not in expected_lengths:
            continue
        if str(meta.get("arm", "")) != "lineage_shield_control":
            raise ValueError("T12.4a.2 source arm is not lineage_shield_control")
        if len(progress_edges) != 1:
            raise ValueError("T12.4a.2 requires one progress edge per source seed")
        edge = progress_edges[0]
        target_cell = archive.cells[edge.target_cell_id]
        target_variant = target_cell.variants[edge.target_exact_hash]
        path = archive.path_edges(target_variant)
        if len(path) != expected_lengths[seed]:
            raise ValueError("T12.4a.2 route length differs from the opened diagnosis")
        token = _checksum(
            {
                "seed": seed,
                "archive": meta["sha256"],
                "edge": edge.edge_id,
                "route": [item.action.key for item in path],
            }
        )[:20]
        witnesses.append(
            ProgressWitness(
                witness_id=f"witness_t12_4a_2_{token}",
                game_id=str(meta["game_id"]),
                source_seed=seed,
                source_arm=str(meta["arm"]),
                source_archive_sha256=str(meta["sha256"]),
                source_progress_edge_id=edge.edge_id,
                initial_exact_hash=path[0].source_exact_hash,
                initial_level=archive.cells[path[0].source_cell_id].level,
                target_exact_hash=path[-1].target_exact_hash,
                target_level=target_cell.level,
                steps=tuple(
                    WitnessStep(
                        expected_source_hash=item.source_exact_hash,
                        action=item.action,
                        expected_target_hash=item.target_exact_hash,
                        level_delta=item.level_delta,
                        terminal=item.terminal,
                        success=item.success,
                    )
                    for item in path
                ),
            )
        )
    if {seed for seed, _ in all_progress} != set(selected.source_seeds):
        raise ValueError("T12.4a.2 collection has unexpected progress provenance")
    if len(all_progress) != 2 or len(witnesses) != 2:
        raise ValueError("T12.4a.2 requires exactly two progress edges")
    return tuple(sorted(witnesses, key=lambda item: item.source_seed))


def _validate_witness_pair(
    witnesses: Sequence[ProgressWitness],
    protocol: WitnessReconfirmationProtocol,
) -> tuple[Any, ...]:
    if len(witnesses) != 2:
        raise ValueError("T12.4a.2 requires two witnesses")
    if tuple(item.source_seed for item in witnesses) != protocol.source_seeds:
        raise ValueError("T12.4a.2 source seeds differ")
    if tuple(len(item.steps) for item in witnesses) != protocol.expected_route_lengths:
        raise ValueError("T12.4a.2 route lengths differ")
    if len({item.game_id for item in witnesses}) != 1:
        raise ValueError("T12.4a.2 witnesses belong to different games")
    if len({item.initial_exact_hash for item in witnesses}) != 1:
        raise ValueError("T12.4a.2 initial hashes differ")
    if len({item.target_exact_hash for item in witnesses}) != 1:
        raise ValueError("T12.4a.2 target hashes differ")
    if {item.target_level for item in witnesses} != {protocol.expected_target_level}:
        raise ValueError("T12.4a.2 target level differs")
    if any(len(item.steps) > protocol.maximum_witness_steps for item in witnesses):
        raise ValueError("T12.4a.2 witness exceeds the maximum route length")
    suffix = common_action_suffix(witnesses)
    if tuple(action.action_name for action in suffix) != protocol.expected_common_suffix:
        raise ValueError("T12.4a.2 common suffix differs")
    if any(action.action_data for action in suffix):
        raise ValueError("T12.4a.2 common suffix unexpectedly has parameters")
    if any(item.steps[-1].level_delta != 1 for item in witnesses):
        raise ValueError("T12.4a.2 progress is not localized to the final action")
    if any(any(step.level_delta for step in item.steps[:-1]) for item in witnesses):
        raise ValueError("T12.4a.2 route progresses before the final action")
    return suffix


def _registry_payload(
    *,
    witnesses: Sequence[ProgressWitness],
    protocol: WitnessReconfirmationProtocol,
    parent_receipt_checksum: str,
    collection_receipt_checksum: str,
) -> dict[str, Any]:
    suffix = _validate_witness_pair(witnesses, protocol)
    return {
        "format_version": RECONFIRM_REGISTRY_FORMAT,
        "protocol_checksum": protocol.checksum,
        "parent_t12_4a_1_receipt_checksum": parent_receipt_checksum,
        "collection_receipt_checksum": collection_receipt_checksum,
        "witnesses": [item.to_dict() for item in witnesses],
        "common_initial_exact_hash": witnesses[0].initial_exact_hash,
        "common_target_exact_hash": witnesses[0].target_exact_hash,
        "common_target_level": witnesses[0].target_level,
        "common_suffix": [
            {
                "action_name": action.action_name,
                "action_data": dict(action.action_data),
            }
            for action in suffix
        ],
    }


def _parent_is_global_calibration_failure(
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> bool:
    protocol = dict(manifest["protocol"])
    metrics = dict(receipt.get("metrics", {}))
    return bool(
        receipt.get("passed") is False
        and receipt.get("phase") == "train"
        and receipt.get("status") == "FAIL_T12_4A_1_CALIBRATION_GATE"
        and float(metrics.get("uncalibrated_maximum_ece", 1.0))
        <= float(protocol["maximum_pooled_ece"])
        and float(metrics.get("calibration_ece_improvement", 0.0))
        < float(protocol["minimum_calibration_ece_improvement"])
        and float(metrics.get("calibrated_brier_regression", 0.0))
        > float(protocol["maximum_calibrated_brier_regression"])
    )


def freeze_witness_reconfirmation(
    *,
    output_path: str | Path,
    witness_registry_path: str | Path,
    parent_manifest_path: str | Path,
    parent_receipt_path: str | Path,
    collection_receipt_path: str | Path,
    root: str | Path | None = None,
    allow_dirty: bool = False,
    protocol: WitnessReconfirmationProtocol | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    selected = protocol or WitnessReconfirmationProtocol()
    parent_manifest = load_calibration_manifest(parent_manifest_path, root=repo_root)
    parent_receipt = load_calibration_receipt(
        parent_receipt_path,
        manifest=parent_manifest,
        root=repo_root,
    )
    collection_receipt = load_calibration_receipt(
        collection_receipt_path,
        manifest=parent_manifest,
        root=repo_root,
        require_passed=True,
    )
    if not _parent_is_global_calibration_failure(parent_manifest, parent_receipt):
        raise ValueError("T12.4a.2 requires the global calibration transport failure")
    if collection_receipt.get("status") != "PASS_T12_4A_1_COLLECTION_GATE":
        raise ValueError("T12.4a.2 requires the passed T12.4a.1 collection")
    if parent_receipt.get("parent_receipt_checksum") != collection_receipt.get(
        "receipt_checksum"
    ):
        raise ValueError("T12.4a.2 collection is not the parent training input")
    if parent_manifest.get("stage") != "source_train":
        raise ValueError("T12.4a.2 is restricted to source_train")
    archives = _archive_artifacts(collection_receipt)
    witnesses = extract_reconfirmation_witnesses(archives, protocol=selected)
    _validate_witness_pair(witnesses, selected)
    missing = [path for path in RECONFIRM_CODE_PATHS if not (repo_root / path).is_file()]
    if missing:
        raise ValueError(f"T12.4a.2 code inventory is incomplete: {missing}")
    git = _git_state(repo_root)
    if git["dirty"] and not allow_dirty:
        raise ValueError("scientific freeze requires a clean worktree")
    authorized = bool(
        not git["dirty"] and parent_manifest.get("scientific_claims_authorized", False)
    )
    registry = _signed(
        _registry_payload(
            witnesses=witnesses,
            protocol=selected,
            parent_receipt_checksum=parent_receipt["receipt_checksum"],
            collection_receipt_checksum=collection_receipt["receipt_checksum"],
        ),
        "registry_checksum",
    )
    _write_json_once(witness_registry_path, registry)
    payload = {
        "format_version": RECONFIRM_MANIFEST_FORMAT,
        "status": "FROZEN_BEFORE_T12_4A_2_WITNESS_RECONFIRMATION",
        "stage": "source_train",
        "game_id": witnesses[0].game_id,
        "protocol": asdict(selected),
        "protocol_checksum": selected.checksum,
        "witness_registry": {
            "path": _bound_path(witness_registry_path, root=repo_root),
            "sha256": _file_sha256(witness_registry_path),
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
                "status": "FAIL_T12_4A_1_CALIBRATION_GATE",
                "failure_class": "GLOBAL_CALIBRATOR_TRANSPORT_FAILURE",
            },
            "collection_receipt": {
                "path": _bound_path(collection_receipt_path, root=repo_root),
                "sha256": _file_sha256(collection_receipt_path),
                "receipt_checksum": collection_receipt["receipt_checksum"],
                "passed": True,
                "status": "PASS_T12_4A_1_COLLECTION_GATE",
            },
        },
        "code_sha256": {
            path: _file_sha256(repo_root / path) for path in RECONFIRM_CODE_PATHS
        },
        "git": git,
        "scientific_claims_authorized": authorized,
        "firewall": {
            "holdout_opened": False,
            "source_validation_opened": False,
            "production_authority": False,
            "terminal_shield_production_authority": False,
            "neural_training_authorized": False,
            "neural_active_evaluation_authorized": False,
            "calibration_training_authorized": False,
            "witness_reconfirmation_authorized": authorized,
            "option_extraction_authorized": False,
            "t12_4a_3_option_freeze_authorized": False,
            "t12_4b_freeze_authorized": False,
            "t12_5_freeze_authorized": False,
        },
        "storage": {
            "maximum_artifact_bytes_per_run": selected.maximum_artifact_bytes_per_run,
            "persist_raw_frames": False,
            "hard_fail_before_write": True,
        },
    }
    manifest = _signed(payload, "manifest_checksum")
    _write_json_once(output_path, manifest)
    receipt = reconfirmation_phase_receipt(
        manifest=manifest,
        phase="freeze",
        passed=authorized,
        status="PASS_T12_4A_2_FREEZE" if authorized else "DIRTY_SMOKE_ONLY",
        metrics={
            "witnesses": len(witnesses),
            "route_lengths": [len(item.steps) for item in witnesses],
            "common_suffix_length": len(selected.expected_common_suffix),
        },
    )
    _write_json_once(Path(output_path).with_name("freeze_receipt.json"), receipt)
    return manifest


def load_reconfirmation_registry(
    path: str | Path,
    *,
    protocol: WitnessReconfirmationProtocol | None = None,
) -> tuple[dict[str, Any], tuple[ProgressWitness, ...]]:
    payload = _read_json(path)
    _verify_signed(payload, "registry_checksum")
    if payload.get("format_version") != RECONFIRM_REGISTRY_FORMAT:
        raise ValueError("unsupported T12.4a.2 witness registry")
    selected = protocol or WitnessReconfirmationProtocol()
    if payload.get("protocol_checksum") != selected.checksum:
        raise ValueError("T12.4a.2 registry protocol mismatch")
    witnesses = tuple(
        ProgressWitness.from_dict(dict(item)) for item in payload.get("witnesses", ())
    )
    _validate_witness_pair(witnesses, selected)
    if payload.get("common_initial_exact_hash") != witnesses[0].initial_exact_hash:
        raise ValueError("T12.4a.2 registry initial hash mismatch")
    if payload.get("common_target_exact_hash") != witnesses[0].target_exact_hash:
        raise ValueError("T12.4a.2 registry target hash mismatch")
    suffix = tuple(
        str(dict(item)["action_name"]) for item in payload.get("common_suffix", ())
    )
    if suffix != selected.expected_common_suffix:
        raise ValueError("T12.4a.2 registry suffix mismatch")
    return payload, witnesses


def load_reconfirmation_manifest(
    path: str | Path,
    *,
    root: str | Path | None = None,
    verify_code: bool = True,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = _read_json(path)
    _verify_signed(manifest, "manifest_checksum")
    if manifest.get("format_version") != RECONFIRM_MANIFEST_FORMAT:
        raise ValueError("unsupported T12.4a.2 manifest")
    protocol = WitnessReconfirmationProtocol(**dict(manifest["protocol"]))
    if protocol.checksum != manifest.get("protocol_checksum"):
        raise ValueError("T12.4a.2 protocol checksum mismatch")
    for key in ("manifest", "receipt", "collection_receipt"):
        meta = dict(manifest["parent"][key])
        candidate = _resolve_bound(str(meta["path"]), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta["sha256"]:
            raise ValueError(f"T12.4a.2 parent artifact mismatch: {key}")
    registry_meta = dict(manifest["witness_registry"])
    registry_path = _resolve_bound(str(registry_meta["path"]), root=repo_root)
    if not registry_path.is_file() or _file_sha256(registry_path) != registry_meta["sha256"]:
        raise ValueError("T12.4a.2 witness registry mismatch")
    registry, _ = load_reconfirmation_registry(registry_path, protocol=protocol)
    if registry.get("registry_checksum") != registry_meta.get("registry_checksum"):
        raise ValueError("T12.4a.2 registry checksum binding mismatch")
    if verify_code:
        for relative, expected in dict(manifest["code_sha256"]).items():
            candidate = repo_root / relative
            if not candidate.is_file() or _file_sha256(candidate) != expected:
                raise ValueError(f"T12.4a.2 code checksum mismatch: {relative}")
    return manifest


def reconfirmation_phase_receipt(
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
            "format_version": RECONFIRM_RECEIPT_FORMAT,
            "phase": str(phase),
            "passed": bool(passed),
            "status": str(status),
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "parent_t12_4a_1_receipt_checksum": manifest["parent"]["receipt"][
                "receipt_checksum"
            ],
            "collection_receipt_checksum": manifest["parent"][
                "collection_receipt"
            ]["receipt_checksum"],
            "metrics": dict(metrics),
            "artifacts": dict(artifacts or {}),
        },
        "receipt_checksum",
    )


def load_reconfirmation_receipt(
    path: str | Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    receipt = _read_json(path)
    _verify_signed(receipt, "receipt_checksum")
    if receipt.get("format_version") != RECONFIRM_RECEIPT_FORMAT:
        raise ValueError("unsupported T12.4a.2 receipt")
    if manifest is not None:
        if receipt.get("manifest_checksum") != manifest.get("manifest_checksum"):
            raise ValueError("T12.4a.2 receipt belongs to another manifest")
        if receipt.get("protocol_checksum") != manifest.get("protocol_checksum"):
            raise ValueError("T12.4a.2 receipt belongs to another protocol")
    for name, raw_meta in dict(receipt.get("artifacts", {})).items():
        meta = dict(raw_meta)
        candidate = _resolve_bound(str(meta.get("path", "")), root=repo_root)
        if not candidate.is_file() or _file_sha256(candidate) != meta.get("sha256"):
            raise ValueError(f"T12.4a.2 receipt artifact mismatch: {name}")
    return receipt


__all__ = [
    "WitnessReconfirmationProtocol",
    "extract_reconfirmation_witnesses",
    "freeze_witness_reconfirmation",
    "load_reconfirmation_manifest",
    "load_reconfirmation_receipt",
    "load_reconfirmation_registry",
    "reconfirmation_phase_receipt",
]
