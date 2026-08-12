"""T12.3d confirmed replay controls and prospective lineage comparison."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from theory.m1.polymorphic_a25_adapter import _step_env_action
from theory.m2.m3_execution_smoke import _reset_env
from theory.sage.live_prefix_counterfactual_collector import (
    select_live_action,
    state_signature_from_frame,
)

from .experiment import RunStorageBudget, _file_sha256, _write_json_once
from .graph_experiment import _make_env, _write_archive
from .lineage_experiment import run_lineage_burst_arm
from .provenance_protocol import (
    ConfirmedControlProtocol,
    ConfirmedReplayControl,
    load_provenance_manifest,
    load_provenance_receipt,
    load_provenance_registry,
    provenance_phase_receipt,
)

EnvFactory = Callable[[str], Any]


@dataclass(frozen=True)
class ConfirmedControlTrial:
    control_id: str
    witness_id: str
    repetition: int
    exact: bool
    calls: int
    first_divergence_step: int | None
    first_divergence_kind: str
    expected_hash: str
    observed_hash: str
    events: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_id": self.control_id,
            "witness_id": self.witness_id,
            "repetition": self.repetition,
            "exact": self.exact,
            "calls": self.calls,
            "first_divergence_step": self.first_divergence_step,
            "first_divergence_kind": self.first_divergence_kind,
            "expected_hash": self.expected_hash,
            "observed_hash": self.observed_hash,
            "events": [dict(item) for item in self.events],
        }


def replay_confirmed_control(
    *,
    control: ConfirmedReplayControl,
    repetition: int,
    environments_dir: str | Path,
    env_factory: EnvFactory | None = None,
) -> ConfirmedControlTrial:
    env = _make_env(control.game_id, environments_dir, env_factory)
    frame = _reset_env(env)
    calls = 1
    observed = state_signature_from_frame(frame)
    expected = control.expected_hashes[0]
    divergence_step: int | None = None
    divergence_kind = ""
    events: list[dict[str, Any]] = [
        {
            "step": 0,
            "kind": "reset",
            "expected_hash": expected,
            "observed_hash": observed,
            "exact": observed == expected,
        }
    ]
    if observed != expected:
        divergence_step = 0
        divergence_kind = "reset_hash"
    else:
        for index, action in enumerate(control.actions, start=1):
            selected = select_live_action(
                env,
                action.action_name,
                action_args=action.action_data,
            )
            expected = control.expected_hashes[index]
            if selected is None:
                divergence_step = index
                divergence_kind = "action_unavailable"
                observed = state_signature_from_frame(frame)
                events.append(
                    {
                        "step": index,
                        "kind": "action_unavailable",
                        "action_key": action.key,
                        "expected_hash": expected,
                        "observed_hash": observed,
                        "exact": False,
                    }
                )
                break
            frame = _step_env_action(env, selected)
            calls += 1
            observed = state_signature_from_frame(frame)
            exact = observed == expected
            events.append(
                {
                    "step": index,
                    "kind": "action",
                    "action_key": action.key,
                    "expected_hash": expected,
                    "observed_hash": observed,
                    "exact": exact,
                }
            )
            if not exact:
                divergence_step = index
                divergence_kind = "state_hash"
                break
    return ConfirmedControlTrial(
        control_id=control.control_id,
        witness_id=control.witness_id,
        repetition=int(repetition),
        exact=divergence_step is None,
        calls=calls,
        first_divergence_step=divergence_step,
        first_divergence_kind=divergence_kind,
        expected_hash=expected,
        observed_hash=observed,
        events=tuple(events),
    )


def _aggregate_gate(
    *,
    protocol: ConfirmedControlProtocol,
    controls: Sequence[ConfirmedReplayControl],
    control_trials: Sequence[ConfirmedControlTrial],
    conditions: Sequence[Mapping[str, Any]],
    sdk_calls: int,
) -> tuple[bool, dict[str, Any]]:
    unique_control_ids = {item.control_id for item in controls}
    unique_route_checksums = {item.route_checksum for item in controls}
    expected_trials = len(controls) * protocol.control_repetitions
    exact_control_trials = sum(item.exact for item in control_trials)
    confirmed_control_exact_rate = (
        0.0 if not control_trials else exact_control_trials / len(control_trials)
    )
    per_control = []
    for control in controls:
        trials = [item for item in control_trials if item.control_id == control.control_id]
        per_control.append(
            {
                "control_id": control.control_id,
                "witness_id": control.witness_id,
                "route_checksum": control.route_checksum,
                "depth": control.depth,
                "prior_route_confirmations": control.prior_route_confirmations,
                "trials": len(trials),
                "exact_trials": sum(item.exact for item in trials),
            }
        )

    per_seed = []
    minimum_treatment_exact = 1.0
    minimum_coverage_ratio = float("inf")
    replay_regression_seeds = 0
    progress_regression_seeds = 0
    lineage_attached = 0
    rebases_avoided = 0
    lineage_rebased = 0
    for condition in conditions:
        seed = int(condition["seed"])
        arms = dict(condition["arms"])
        control_metrics = dict(arms["shortest_prefix_control"]["metrics"])
        treatment_metrics = dict(arms["lineage_preserving"]["metrics"])
        control_exact = float(control_metrics["replay_exact_rate"])
        treatment_exact = float(treatment_metrics["replay_exact_rate"])
        exact_delta = treatment_exact - control_exact
        control_coverage = float(
            control_metrics["symbolic_cells_per_1000_sdk_calls"]
        )
        treatment_coverage = float(
            treatment_metrics["symbolic_cells_per_1000_sdk_calls"]
        )
        coverage_ratio = (
            1.0 if control_coverage <= 0.0 else treatment_coverage / control_coverage
        )
        control_progress = int(control_metrics["progress_edges"])
        treatment_progress = int(treatment_metrics["progress_edges"])
        minimum_treatment_exact = min(minimum_treatment_exact, treatment_exact)
        minimum_coverage_ratio = min(minimum_coverage_ratio, coverage_ratio)
        replay_regression_seeds += int(
            exact_delta < -protocol.maximum_per_seed_replay_regression
        )
        progress_regression_seeds += int(treatment_progress < control_progress)
        lineage_attached += int(treatment_metrics["lineage_attached_transitions"])
        rebases_avoided += int(treatment_metrics["shortest_prefix_rebases_avoided"])
        lineage_rebased += int(treatment_metrics["lineage_rebased_transitions"])
        per_seed.append(
            {
                "seed": seed,
                "control_replay_exact_rate": control_exact,
                "treatment_replay_exact_rate": treatment_exact,
                "replay_delta": exact_delta,
                "control_coverage": control_coverage,
                "treatment_coverage": treatment_coverage,
                "coverage_ratio": coverage_ratio,
                "control_progress": control_progress,
                "treatment_progress": treatment_progress,
                "rebases_avoided": int(
                    treatment_metrics["shortest_prefix_rebases_avoided"]
                ),
            }
        )
    if minimum_coverage_ratio == float("inf"):
        minimum_coverage_ratio = 0.0
    metrics = {
        "unique_confirmed_controls": len(unique_control_ids),
        "unique_route_checksums": len(unique_route_checksums),
        "expected_control_trials": expected_trials,
        "control_trials": len(control_trials),
        "exact_control_trials": exact_control_trials,
        "confirmed_control_exact_rate": confirmed_control_exact_rate,
        "per_control": per_control,
        "minimum_treatment_replay_exact_rate": minimum_treatment_exact,
        "replay_regression_seeds": replay_regression_seeds,
        "minimum_per_seed_coverage_ratio": minimum_coverage_ratio,
        "progress_regression_seeds": progress_regression_seeds,
        "lineage_attached_transitions": lineage_attached,
        "shortest_prefix_rebases_avoided": rebases_avoided,
        "lineage_rebased_transitions": lineage_rebased,
        "sdk_calls": sdk_calls,
        "maximum_total_sdk_calls": protocol.maximum_total_sdk_calls,
        "per_seed": per_seed,
    }
    passed = bool(
        len(unique_control_ids) == protocol.expected_unique_controls
        and len(unique_route_checksums) == protocol.expected_unique_controls
        and len(control_trials) == expected_trials
        and all(
            item.prior_route_confirmations
            >= protocol.minimum_prior_route_confirmations
            for item in controls
        )
        and confirmed_control_exact_rate
        >= protocol.minimum_confirmed_control_exact_rate
        and minimum_treatment_exact >= protocol.minimum_treatment_replay_exact_rate
        and replay_regression_seeds == 0
        and minimum_coverage_ratio >= protocol.minimum_coverage_ratio
        and progress_regression_seeds <= protocol.maximum_progress_regression_seeds
        and lineage_attached > 0
        and rebases_avoided >= protocol.minimum_rebases_avoided
        and lineage_rebased == 0
        and sdk_calls <= protocol.maximum_total_sdk_calls
    )
    return passed, metrics


def _registry_path(manifest: Mapping[str, Any]) -> Path:
    path = Path(str(manifest["control_registry"]["path"]))
    return path if path.is_absolute() else Path(__file__).resolve().parents[3] / path


def run_provenance_experiment(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    environments_dir: str | Path = "environment_files",
    env_factory: EnvFactory | None = None,
) -> dict[str, Any]:
    manifest = load_provenance_manifest(
        manifest_path, verify_code=env_factory is None
    )
    if not manifest.get("scientific_claims_authorized", False):
        raise ValueError("T12.3d run requires a clean scientific freeze")
    if not manifest.get("firewall", {}).get(
        "confirmed_control_experiment_authorized", False
    ):
        raise ValueError("T12.3d confirmed-control experiment is not authorized")
    protocol = ConfirmedControlProtocol(**dict(manifest["protocol"]))
    _, controls = load_provenance_registry(_registry_path(manifest), protocol=protocol)
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable run: {destination}")
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes_per_run)

    control_trials = []
    for control in controls:
        for repetition in range(protocol.control_repetitions):
            control_trials.append(
                replay_confirmed_control(
                    control=control,
                    repetition=repetition,
                    environments_dir=environments_dir,
                    env_factory=env_factory,
                )
            )

    conditions = []
    archive_artifacts = []
    game_id = str(manifest["game_id"])
    for seed in protocol.evaluation_seeds:
        arms = {}
        for arm in protocol.evaluation_arms:
            run = run_lineage_burst_arm(
                game_id=game_id,
                seed=seed,
                sdk_call_budget=protocol.sdk_calls_per_evaluation_arm,
                burst_schedule=protocol.burst_schedule,
                preserve_lineage=arm == "lineage_preserving",
                environments_dir=environments_dir,
                env_factory=env_factory,
                maximum_cells=protocol.maximum_cells,
            )
            archive_path = destination / game_id / str(seed) / f"{arm}.json"
            artifact = _write_archive(
                archive_path, run.archive, storage_budget=storage
            )
            artifact.update({"game_id": game_id, "seed": seed, "arm": arm})
            archive_artifacts.append(artifact)
            excursions_path = (
                destination / game_id / str(seed) / f"{arm}_excursions.json"
            )
            _write_json_once(
                excursions_path,
                {
                    "format_version": "sage-t12.3d-provenance-excursions-v1",
                    "game_id": game_id,
                    "seed": seed,
                    "arm": arm,
                    "excursions": [item.to_dict() for item in run.excursions],
                },
                storage_budget=storage,
            )
            arms[arm] = {
                "metrics": run.metrics(),
                "archive": artifact,
                "excursions": {
                    "path": str(excursions_path.resolve()),
                    "sha256": _file_sha256(excursions_path),
                },
            }
        conditions.append({"game_id": game_id, "seed": seed, "arms": arms})

    sdk_calls = sum(item.calls for item in control_trials) + sum(
        int(arm["metrics"]["sdk_calls"])
        for condition in conditions
        for arm in condition["arms"].values()
    )
    passed, metrics = _aggregate_gate(
        protocol=protocol,
        controls=controls,
        control_trials=control_trials,
        conditions=conditions,
        sdk_calls=sdk_calls,
    )
    controls_path = destination / "confirmed_control_trials.json"
    _write_json_once(
        controls_path,
        {
            "format_version": "sage-t12.3d-confirmed-control-trials-v1",
            "trials": [item.to_dict() for item in control_trials],
        },
        storage_budget=storage,
    )
    evaluation_path = destination / "paired_evaluation.json"
    _write_json_once(
        evaluation_path,
        {
            "format_version": "sage-t12.3d-paired-provenance-evaluation-v1",
            "conditions": conditions,
            "archives": archive_artifacts,
        },
        storage_budget=storage,
    )
    status = (
        "PASS_T12_3D_CONFIRMED_CONTROL_GATE"
        if passed
        else "FAIL_T12_3D_CONFIRMED_CONTROL_GATE"
    )
    report = {
        "format_version": "sage-t12.3d-confirmed-control-report-v1",
        "status": status,
        "passed": passed,
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "parent_t12_3c_receipt_checksum": manifest["parent"]["receipt"][
            "receipt_checksum"
        ],
        "metrics": metrics,
        "conditions": conditions,
        "storage": storage.snapshot(),
    }
    report_path = destination / "provenance_report.json"
    _write_json_once(report_path, report, storage_budget=storage)
    receipt = provenance_phase_receipt(
        manifest=manifest,
        phase="confirmed_control",
        passed=passed,
        status=status,
        metrics=metrics,
        artifacts={
            "confirmed_control_trials": {
                "path": str(controls_path.resolve()),
                "sha256": _file_sha256(controls_path),
            },
            "paired_evaluation": {
                "path": str(evaluation_path.resolve()),
                "sha256": _file_sha256(evaluation_path),
            },
            "report": {
                "path": str(report_path.resolve()),
                "sha256": _file_sha256(report_path),
            },
            "control_registry": dict(manifest["control_registry"]),
        },
    )
    _write_json_once(
        destination / "provenance_receipt.json", receipt, storage_budget=storage
    )
    return report


def provenance_experiment_status(
    *,
    manifest_path: str | Path,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest = load_provenance_manifest(manifest_path)
    receipt = (
        None
        if receipt_path is None
        else load_provenance_receipt(receipt_path, manifest=manifest)
    )
    passed = bool(
        manifest.get("scientific_claims_authorized", False)
        and receipt is not None
        and receipt.get("passed") is True
        and receipt.get("status") == "PASS_T12_3D_CONFIRMED_CONTROL_GATE"
    )
    return {
        "format_version": "sage-t12.3d-confirmed-control-status-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "parent_t12_3c_status": manifest["parent"]["receipt"]["status"],
        "parent_failure_class": manifest["parent"]["receipt"]["failure_class"],
        "receipt": (
            None
            if receipt is None
            else {
                "phase": receipt["phase"],
                "passed": receipt["passed"],
                "status": receipt["status"],
                "receipt_checksum": receipt["receipt_checksum"],
            }
        ),
        "next_phase_authorized": passed,
        "firewall": {
            **dict(manifest["firewall"]),
            "t12_3b_child_rerun_authorized": passed,
        },
    }


__all__ = [
    "ConfirmedControlTrial",
    "provenance_experiment_status",
    "replay_confirmed_control",
    "run_provenance_experiment",
]
