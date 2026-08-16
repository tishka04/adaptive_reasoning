"""Offline execution of the SAGE.T12.5b.2 discrimination audit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .experiment import RunStorageBudget, _file_sha256, _read_json, _write_json_once
from .progress import CausalProgressProgram
from .progress_discrimination import audit_progress_discrimination
from .progress_discrimination_protocol import (
    ProgressDiscriminationProtocol,
    _resolve_bound,
    load_progress_discrimination_manifest,
    load_progress_discrimination_receipt,
    progress_discrimination_receipt,
)
from .progress_shadow import posterior_from_snapshot

DISCRIMINATION_REPORT_FORMAT = "sage-t12.5b.2-discrimination-report-v1"


def _ordered_milestones(
    programs: Sequence[CausalProgressProgram],
):  # type: ignore[no-untyped-def]
    candidates = [
        item.milestones for item in programs if item.progress_kind == "ordered_effects"
    ]
    if not candidates:
        raise ValueError("T12.5b.2 has no ordered progress program")
    semantic = [tuple(item.semantic_payload for item in group) for group in candidates]
    if any(value != semantic[0] for value in semantic[1:]):
        raise ValueError("T12.5b.2 ordered programs disagree on milestone semantics")
    return tuple(candidates[0])


def _load_inputs(
    manifest: Mapping[str, Any], *, root: Path
) -> tuple[
    tuple[Mapping[str, Any], ...],
    Mapping[str, Any],
    tuple[CausalProgressProgram, ...],
]:
    trials_payload = _read_json(
        _resolve_bound(str(manifest["inputs"]["trials"]["path"]), root=root)
    )
    posterior_payload = _read_json(
        _resolve_bound(str(manifest["inputs"]["posterior"]["path"]), root=root)
    )
    registry_payload = _read_json(
        _resolve_bound(
            str(manifest["inputs"]["program_registry"]["path"]), root=root
        )
    )
    trials = tuple(dict(item) for item in trials_payload.get("trials", ()))
    programs = tuple(
        CausalProgressProgram.from_dict(dict(item))
        for item in registry_payload.get("programs", ())
    )
    if not trials or not programs:
        raise ValueError("T12.5b.2 sealed inputs are empty")
    return trials, posterior_payload, programs


def _parent_trial_count(manifest: Mapping[str, Any], *, root: Path) -> int:
    parent_manifest = _read_json(
        _resolve_bound(str(manifest["parent"]["manifest"]["path"]), root=root)
    )
    protocol = dict(parent_manifest["protocol"])
    return (
        len(protocol["lineage_seeds"])
        * len(protocol["stages"])
        * len(protocol["candidate_actions"])
        * int(protocol["repetitions_per_branch"])
    )


def run_progress_discrimination_audit(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_progress_discrimination_manifest(
        manifest_path, root=repo_root, verify_code=True
    )
    if not manifest["firewall"].get(
        "affordance_discrimination_audit_authorized", False
    ):
        raise ValueError("T12.5b.2 manifest does not authorize the offline audit")
    protocol = ProgressDiscriminationProtocol(**dict(manifest["protocol"]))
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable run: {destination}")
    storage = RunStorageBudget(
        destination, protocol.maximum_artifact_bytes_per_run
    )
    trials, posterior_payload, programs = _load_inputs(manifest, root=repo_root)
    posterior = posterior_from_snapshot(posterior_payload)
    milestones = _ordered_milestones(programs)
    if len(milestones) != len(protocol.stages):
        raise ValueError("T12.5b.2 milestone count changed")
    parent_manifest = _read_json(
        _resolve_bound(str(manifest["parent"]["manifest"]["path"]), root=repo_root)
    )
    parent_protocol = dict(parent_manifest["protocol"])
    audit = audit_progress_discrimination(
        trials=trials,
        features=tuple(parent_protocol["allowed_effect_features"]),
        posterior=posterior,
        milestones=milestones,
        lineage_seeds=protocol.lineage_seeds,
        stages=protocol.stages,
        expected_actions=protocol.expected_actions,
        repetitions_per_branch=protocol.repetitions_per_branch,
        induction_lineage_seed=protocol.induction_lineage_seed,
        confirmation_lineage_seed=protocol.confirmation_lineage_seed,
        minimum_distractor_magnitude_gap=(
            protocol.minimum_distractor_magnitude_gap
        ),
    )
    metrics = dict(audit["metrics"])
    contrasts = tuple(audit["contrast_registry"]["hard_contrasts"])
    per_lineage = {
        str(seed): sum(int(item["lineage_seed"]) == seed for item in contrasts)
        for seed in protocol.lineage_seeds
    }
    hard_causal_accuracy = (
        sum(
            float(item["progress_gain"])
            > float(item["distractor_progress_gain"])
            for item in contrasts
        )
        / len(contrasts)
        if contrasts
        else 0.0
    )
    hard_magnitude_accuracy = (
        sum(
            float(item["progress_magnitude"])
            > float(item["distractor_magnitude"])
            for item in contrasts
        )
        / len(contrasts)
        if contrasts
        else 0.0
    )
    hard_accuracy_gain = hard_causal_accuracy - hard_magnitude_accuracy
    metrics.update(
        {
            "hard_contrast_accuracy_gain": hard_accuracy_gain,
            "hard_contrast_causal_accuracy": hard_causal_accuracy,
            "hard_contrast_magnitude_accuracy": hard_magnitude_accuracy,
            "hard_contrasts_per_lineage": per_lineage,
            "parent_trial_count": len(trials),
            "sdk_calls_used": 0,
        }
    )
    failed_parent_checks = tuple(sorted(manifest["parent"]["failed_checks"]))
    expected_failed = tuple(sorted(protocol.authorized_parent_failed_checks))
    integrity_checks = {
        "parent_negative_failure_class_preserved": failed_parent_checks
        == expected_failed,
        "parent_trial_count_exact": len(trials)
        == _parent_trial_count(manifest, root=repo_root),
        "all_prefixes_exact": metrics["exact_prefix_rate"] == 1.0,
        "repetition_count_exact": metrics["repetition_count_is_exact"],
        "local_availability_is_deterministic": metrics[
            "availability_is_deterministic"
        ],
        "executable_effects_are_deterministic": metrics[
            "effect_is_deterministic_when_executable"
        ],
        "local_candidate_sets_are_nontrivial": metrics[
            "minimum_executable_actions_per_context"
        ]
        >= protocol.minimum_executable_actions_per_context,
        "progress_action_is_locally_executable": metrics[
            "progress_action_executable_in_every_context"
        ],
        "progress_affordance_transports_semantically": metrics[
            "affordance_binding_coverage"
        ]
        >= protocol.minimum_affordance_binding_coverage,
        "no_environment_calls": metrics["sdk_calls_used"]
        == protocol.maximum_sdk_calls,
    }
    discrimination_checks = {
        "hard_contrast_exists_in_every_lineage": (
            metrics["hard_contrast_lineage_count"]
            >= protocol.minimum_hard_contrast_lineages
            and all(
                count >= protocol.minimum_hard_contrasts_per_lineage
                for count in per_lineage.values()
            )
        ),
        "causal_ranking_is_perfect_on_hard_contrasts": (
            bool(contrasts)
            and hard_causal_accuracy
            >= protocol.minimum_causal_hard_contrast_accuracy
        ),
        "causal_ranking_beats_magnitude_on_hard_contrasts": (
            bool(contrasts)
            and hard_accuracy_gain
            >= protocol.minimum_hard_contrast_accuracy_gain
        ),
    }
    checks = {**integrity_checks, **discrimination_checks}
    integrity_passed = all(integrity_checks.values())
    discrimination_passed = all(discrimination_checks.values())
    passed = integrity_passed and discrimination_passed
    if not integrity_passed:
        classification = "AUDIT_INTEGRITY_FAILURE"
        status = "FAIL_T12_5B_2_AUDIT_INTEGRITY_GATE"
    elif not contrasts:
        classification = "INSUFFICIENT_DISCRIMINATIVE_CONTRASTS"
        status = "FAIL_T12_5B_2_INSUFFICIENT_DISCRIMINATIVE_CONTRASTS"
    elif not discrimination_passed:
        classification = "CAUSAL_PROGRESS_NOT_DISCRIMINATIVE"
        status = "FAIL_T12_5B_2_DISCRIMINATION_GATE"
    else:
        classification = "CAUSAL_PROGRESS_DISCRIMINATES_FROM_MAGNITUDE"
        status = "PASS_T12_5B_2_DISCRIMINATION_GATE"
    collection_freeze_authorized = bool(
        integrity_passed
        and classification == "INSUFFICIENT_DISCRIMINATIVE_CONTRASTS"
    )
    metrics.update(
        {
            "checks": checks,
            "classification": classification,
            "collection_freeze_authorized": collection_freeze_authorized,
            "parent_negative_result_preserved": True,
        }
    )

    affordance_path = destination / "affordance_registry.json"
    contrast_path = destination / "hard_contrast_registry.json"
    report_path = destination / "discrimination_report.json"
    receipt_path = destination / "discrimination_receipt.json"
    affordance_payload = {
        **dict(audit["affordance_registry"]),
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
    }
    contrast_payload = {
        **dict(audit["contrast_registry"]),
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
    }
    _write_json_once(
        affordance_path, affordance_payload, storage_budget=storage
    )
    _write_json_once(contrast_path, contrast_payload, storage_budget=storage)
    report = {
        "claim_boundary": manifest["claim_boundary"],
        "format_version": DISCRIMINATION_REPORT_FORMAT,
        "manifest_checksum": manifest["manifest_checksum"],
        "metrics": metrics,
        "passed": passed,
        "protocol_checksum": manifest["protocol_checksum"],
        "status": status,
    }
    _write_json_once(report_path, report, storage_budget=storage)
    metrics["storage"] = storage.snapshot()
    checks["storage_within_budget"] = metrics["storage"]["within_budget"]
    receipt = progress_discrimination_receipt(
        manifest=manifest,
        phase="offline_audit",
        passed=passed and checks["storage_within_budget"],
        status=status if checks["storage_within_budget"] else "FAIL_T12_5B_2_STORAGE_GATE",
        metrics=metrics,
        artifacts={
            "affordances": {
                "path": str(affordance_path.resolve()),
                "sha256": _file_sha256(affordance_path),
            },
            "contrasts": {
                "path": str(contrast_path.resolve()),
                "sha256": _file_sha256(contrast_path),
            },
            "report": {
                "path": str(report_path.resolve()),
                "sha256": _file_sha256(report_path),
            },
        },
    )
    _write_json_once(receipt_path, receipt, storage_budget=storage)
    return receipt


def progress_discrimination_status(
    *,
    manifest_path: str | Path,
    receipt_path: str | Path | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_progress_discrimination_manifest(
        manifest_path, root=repo_root
    )
    receipt = (
        None
        if receipt_path is None or not Path(receipt_path).is_file()
        else load_progress_discrimination_receipt(
            receipt_path, manifest=manifest, root=repo_root
        )
    )
    collection_freeze_authorized = bool(
        receipt
        and receipt.get("metrics", {}).get("collection_freeze_authorized") is True
        and receipt.get("status")
        == "FAIL_T12_5B_2_INSUFFICIENT_DISCRIMINATIVE_CONTRASTS"
    )
    return {
        "claim_boundary": manifest["claim_boundary"],
        "firewall": {
            "affordance_discrimination_audit_authorized": True,
            "causal_progress_control_authorized": False,
            "environment_collection_authorized": False,
            "holdout_opened": False,
            "neural_active_evaluation_authorized": False,
            "neural_training_authorized": False,
            "option_control_authorized": False,
            "production_authority": False,
            "source_validation_opened": False,
            "t12_5b_3_collection_freeze_authorized": collection_freeze_authorized,
            "t12_5c_control_freeze_authorized": False,
            "t12_6_freeze_authorized": False,
            "terminal_shield_production_authority": False,
        },
        "format_version": "sage-t12.5b.2-discrimination-status-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "next_phase_authorized": collection_freeze_authorized,
        "parent_t12_5b_status": manifest["parent"]["receipt"]["status"],
        "protocol_checksum": manifest["protocol_checksum"],
        "receipt": (
            None
            if receipt is None
            else {
                "classification": receipt.get("metrics", {}).get("classification"),
                "passed": receipt["passed"],
                "phase": receipt["phase"],
                "receipt_checksum": receipt["receipt_checksum"],
                "status": receipt["status"],
            }
        ),
    }


__all__ = [
    "progress_discrimination_status",
    "run_progress_discrimination_audit",
]
