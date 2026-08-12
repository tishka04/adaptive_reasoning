"""Exact, repeated replay of the two T12.2 progress witnesses."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from theory.m1.polymorphic_a25_adapter import _step_env_action
from theory.m2.m3_execution_smoke import _reset_env
from theory.real_env_option_adapter import snapshot_frame
from theory.sage.live_prefix_counterfactual_collector import (
    select_live_action,
    state_signature_from_frame,
)

from .experiment import RunStorageBudget, _file_sha256, _write_json_once
from .graph_experiment import _make_env
from .witness_protocol import (
    ProgressWitness,
    WitnessConfirmProtocol,
    WitnessStep,
    load_witness_manifest,
    load_witness_receipt,
    load_witness_registry,
    witness_phase_receipt,
)

EnvFactory = Callable[[str], Any]


class SdkCallBudget:
    def __init__(self, maximum_calls: int) -> None:
        self.maximum_calls = max(1, int(maximum_calls))
        self.used_calls = 0

    def consume(self, calls: int = 1) -> None:
        additional = max(0, int(calls))
        if self.used_calls + additional > self.maximum_calls:
            raise RuntimeError(
                "T12.3a SDK call budget exceeded: "
                f"used={self.used_calls} additional={additional} "
                f"maximum={self.maximum_calls}"
            )
        self.used_calls += additional

    def snapshot(self) -> dict[str, Any]:
        return {
            "maximum_sdk_calls": self.maximum_calls,
            "used_sdk_calls": self.used_calls,
            "remaining_sdk_calls": self.maximum_calls - self.used_calls,
            "within_budget": self.used_calls <= self.maximum_calls,
        }


@dataclass(frozen=True)
class ReplayTrial:
    witness_id: str
    trial_type: str
    repetition: int
    exact: bool
    initial_exact: bool
    observed_progress: bool
    expected_progress: bool
    final_exact_hash: str
    final_level: int
    first_divergence: str
    events: tuple[Mapping[str, Any], ...]

    @property
    def confirmed(self) -> bool:
        return bool(self.exact and self.observed_progress == self.expected_progress)

    def to_dict(self) -> dict[str, Any]:
        return {
            "witness_id": self.witness_id,
            "trial_type": self.trial_type,
            "repetition": self.repetition,
            "exact": self.exact,
            "initial_exact": self.initial_exact,
            "observed_progress": self.observed_progress,
            "expected_progress": self.expected_progress,
            "confirmed": self.confirmed,
            "final_exact_hash": self.final_exact_hash,
            "final_level": self.final_level,
            "first_divergence": self.first_divergence,
            "events": [dict(item) for item in self.events],
        }


def _execute_expected_steps(
    *,
    env: Any,
    frame: Any,
    steps: Sequence[WitnessStep],
    phase: str,
    start_index: int,
    budget: SdkCallBudget,
) -> tuple[Any, list[dict[str, Any]], str]:
    events = []
    divergence = ""
    for offset, step in enumerate(steps):
        observed_source = state_signature_from_frame(frame)
        if observed_source != step.expected_source_hash:
            divergence = f"{phase}:source:{start_index + offset}"
            events.append(
                {
                    "kind": "source_check",
                    "phase": phase,
                    "step_index": start_index + offset,
                    "expected_hash": step.expected_source_hash,
                    "observed_hash": observed_source,
                    "exact": False,
                }
            )
            break
        selected = select_live_action(
            env,
            step.action.action_name,
            action_args=step.action.action_data,
        )
        if selected is None:
            divergence = f"{phase}:action_unavailable:{start_index + offset}"
            events.append(
                {
                    "kind": "transition",
                    "phase": phase,
                    "step_index": start_index + offset,
                    "expected_source_hash": step.expected_source_hash,
                    "observed_source_hash": observed_source,
                    "action": {
                        "action_name": step.action.action_name,
                        "action_data": dict(step.action.action_data),
                    },
                    "expected_target_hash": step.expected_target_hash,
                    "observed_target_hash": "",
                    "exact": False,
                    "reason": "action_unavailable",
                }
            )
            break
        budget.consume()
        next_frame = _step_env_action(env, selected)
        observed_target = state_signature_from_frame(next_frame)
        exact = observed_target == step.expected_target_hash
        events.append(
            {
                "kind": "transition",
                "phase": phase,
                "step_index": start_index + offset,
                "expected_source_hash": step.expected_source_hash,
                "observed_source_hash": observed_source,
                "action": {
                    "action_name": step.action.action_name,
                    "action_data": dict(step.action.action_data),
                },
                "expected_target_hash": step.expected_target_hash,
                "observed_target_hash": observed_target,
                "exact": exact,
                "levels_completed": int(snapshot_frame(next_frame).levels_completed),
            }
        )
        frame = next_frame
        if not exact:
            divergence = f"{phase}:target:{start_index + offset}"
            break
    return frame, events, divergence


def replay_trial(
    *,
    witness: ProgressWitness,
    trial_type: str,
    repetition: int,
    budget: SdkCallBudget,
    environments_dir: str | Path,
    env_factory: EnvFactory | None,
    suffix_length: int = 3,
) -> ReplayTrial:
    if trial_type not in {"full_route", "common_suffix", "delete_last_suffix_action"}:
        raise ValueError(f"unsupported T12.3a trial type: {trial_type}")
    env = _make_env(witness.game_id, environments_dir, env_factory)
    budget.consume()
    frame = _reset_env(env)
    initial_hash = state_signature_from_frame(frame)
    initial_exact = initial_hash == witness.initial_exact_hash
    events: list[dict[str, Any]] = [
        {
            "kind": "reset",
            "phase": "initial",
            "expected_hash": witness.initial_exact_hash,
            "observed_hash": initial_hash,
            "exact": initial_exact,
            "levels_completed": int(snapshot_frame(frame).levels_completed),
        }
    ]
    divergence = "" if initial_exact else "initial:reset"
    if trial_type == "full_route":
        prefix: tuple[WitnessStep, ...] = ()
        branch = witness.steps
        expected_progress = True
    else:
        prefix = witness.steps[:-suffix_length]
        suffix = witness.steps[-suffix_length:]
        branch = suffix if trial_type == "common_suffix" else suffix[:-1]
        expected_progress = trial_type == "common_suffix"
    if initial_exact:
        frame, prefix_events, prefix_divergence = _execute_expected_steps(
            env=env,
            frame=frame,
            steps=prefix,
            phase="prefix",
            start_index=0,
            budget=budget,
        )
        events.extend(prefix_events)
        divergence = prefix_divergence
    if initial_exact and not divergence:
        frame, branch_events, branch_divergence = _execute_expected_steps(
            env=env,
            frame=frame,
            steps=branch,
            phase=("route" if trial_type == "full_route" else "suffix_branch"),
            start_index=len(prefix),
            budget=budget,
        )
        events.extend(branch_events)
        divergence = branch_divergence
    final_snapshot = snapshot_frame(frame)
    final_hash = state_signature_from_frame(frame)
    observed_progress = int(final_snapshot.levels_completed) > witness.initial_level
    exact = bool(initial_exact and not divergence)
    if trial_type in {"full_route", "common_suffix"}:
        exact = bool(exact and final_hash == witness.target_exact_hash)
        if not exact and not divergence:
            divergence = f"{trial_type}:final_target"
    return ReplayTrial(
        witness_id=witness.witness_id,
        trial_type=trial_type,
        repetition=repetition,
        exact=exact,
        initial_exact=initial_exact,
        observed_progress=observed_progress,
        expected_progress=expected_progress,
        final_exact_hash=final_hash,
        final_level=int(final_snapshot.levels_completed),
        first_divergence=divergence,
        events=tuple(events),
    )


def _observed_prefix_hash(trial: ReplayTrial) -> str:
    prefix = [
        event
        for event in trial.events
        if event.get("kind") == "transition" and event.get("phase") == "prefix"
    ]
    if prefix:
        return str(prefix[-1].get("observed_target_hash", ""))
    reset = next(
        (event for event in trial.events if event.get("kind") == "reset"),
        {},
    )
    return str(reset.get("observed_hash", ""))


def _intervention_bundles(
    *,
    witnesses: Sequence[ProgressWitness],
    trials: Sequence[ReplayTrial],
    suffix_length: int,
) -> list[dict[str, Any]]:
    bundles = []
    for witness in witnesses:
        expected_prefix_hash = (
            witness.initial_exact_hash
            if len(witness.steps) == suffix_length
            else witness.steps[-suffix_length - 1].expected_target_hash
        )
        selected = [trial for trial in trials if trial.witness_id == witness.witness_id]
        for repetition in sorted({trial.repetition for trial in selected}):
            branches = {
                trial.trial_type: trial
                for trial in selected
                if trial.repetition == repetition
                and trial.trial_type in {"common_suffix", "delete_last_suffix_action"}
            }
            if set(branches) != {"common_suffix", "delete_last_suffix_action"}:
                continue
            prefix_hashes = {
                trial_type: _observed_prefix_hash(trial)
                for trial_type, trial in branches.items()
            }
            branch_prefix_exact = {
                trial_type: bool(
                    trial.initial_exact
                    and len(
                        [
                            event
                            for event in trial.events
                            if event.get("kind") == "transition"
                            and event.get("phase") == "prefix"
                        ]
                    )
                    == len(witness.steps) - suffix_length
                    and all(
                        bool(event.get("exact"))
                        for event in trial.events
                        if event.get("phase") == "prefix"
                    )
                )
                for trial_type, trial in branches.items()
            }
            paired_prefix_exact = bool(
                all(branch_prefix_exact.values())
                and len(set(prefix_hashes.values())) == 1
                and next(iter(prefix_hashes.values())) == expected_prefix_hash
            )
            bundles.append(
                {
                    "bundle_id": f"{witness.witness_id}:pair:{repetition}",
                    "witness_id": witness.witness_id,
                    "repetition": repetition,
                    "prefix_length": len(witness.steps) - suffix_length,
                    "expected_prefix_hash": expected_prefix_hash,
                    "observed_prefix_hashes": prefix_hashes,
                    "paired_prefix_exact": paired_prefix_exact,
                    "branches": {
                        trial_type: {
                            "confirmed": trial.confirmed,
                            "prefix_exact": branch_prefix_exact[trial_type],
                            "expected_progress": trial.expected_progress,
                            "observed_progress": trial.observed_progress,
                            "final_exact_hash": trial.final_exact_hash,
                            "first_divergence": trial.first_divergence,
                        }
                        for trial_type, trial in sorted(branches.items())
                    },
                    "paired_contrast_confirmed": bool(
                        paired_prefix_exact
                        and all(trial.confirmed for trial in branches.values())
                    ),
                }
            )
    return bundles


def _metrics(
    *,
    witnesses: Sequence[ProgressWitness],
    trials: Sequence[ReplayTrial],
    protocol: WitnessConfirmProtocol,
    budget: SdkCallBudget,
    intervention_bundles: Sequence[Mapping[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    per_witness = []
    for witness in witnesses:
        selected = [trial for trial in trials if trial.witness_id == witness.witness_id]
        route = [trial for trial in selected if trial.trial_type == "full_route"]
        suffix = [trial for trial in selected if trial.trial_type == "common_suffix"]
        deletion = [
            trial
            for trial in selected
            if trial.trial_type == "delete_last_suffix_action"
        ]
        paired = [
            bundle
            for bundle in intervention_bundles
            if bundle.get("witness_id") == witness.witness_id
        ]
        per_witness.append(
            {
                "witness_id": witness.witness_id,
                "source_seed": witness.source_seed,
                "source_arm": witness.source_arm,
                "route_length": len(witness.steps),
                "route_confirmations": sum(trial.confirmed for trial in route),
                "suffix_confirmations": sum(trial.confirmed for trial in suffix),
                "deletion_control_exact_no_progress": sum(
                    trial.confirmed for trial in deletion
                ),
                "deletion_control_progresses": sum(
                    trial.observed_progress for trial in deletion
                ),
                "paired_contrast_confirmations": sum(
                    bool(bundle.get("paired_contrast_confirmed")) for bundle in paired
                ),
                "paired_prefix_exact": sum(
                    bool(bundle.get("paired_prefix_exact")) for bundle in paired
                ),
                "initial_exact_trials": sum(trial.initial_exact for trial in selected),
                "trials": len(selected),
            }
        )
    comparison_events = [
        event
        for trial in trials
        for event in trial.events
        if event.get("kind") in {"reset", "source_check", "transition"}
    ]
    exact_comparisons = sum(bool(event.get("exact")) for event in comparison_events)
    step_exact_rate = (
        0.0 if not comparison_events else exact_comparisons / len(comparison_events)
    )
    divergences = Counter(
        trial.first_divergence for trial in trials if trial.first_divergence
    )
    metrics = {
        "per_witness": per_witness,
        "witnesses": len(witnesses),
        "common_initial_exact_hash": witnesses[0].initial_exact_hash,
        "common_target_exact_hash": witnesses[0].target_exact_hash,
        "common_target_level": witnesses[0].target_level,
        "step_comparisons": len(comparison_events),
        "exact_step_comparisons": exact_comparisons,
        "step_exact_rate": step_exact_rate,
        "divergences": dict(sorted(divergences.items())),
        "sdk_budget": budget.snapshot(),
    }
    passed = bool(
        all(
            item["route_confirmations"] >= protocol.minimum_successful_route_replays
            and item["suffix_confirmations"]
            >= protocol.minimum_successful_suffix_replays
            and item["deletion_control_exact_no_progress"]
            >= protocol.minimum_successful_suffix_replays
            and item["deletion_control_progresses"] == 0
            and item["paired_contrast_confirmations"]
            >= protocol.minimum_paired_contrasts
            and item["initial_exact_trials"] == item["trials"]
            for item in per_witness
        )
        and step_exact_rate >= protocol.minimum_step_exact_rate
        and budget.used_calls <= protocol.maximum_sdk_calls
    )
    return passed, metrics


def _resolve_registry_path(manifest: Mapping[str, Any]) -> Path:
    path = Path(str(manifest["witness_registry"]["path"]))
    return path if path.is_absolute() else Path(__file__).resolve().parents[3] / path


def run_witness_confirmation(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    environments_dir: str | Path = "environment_files",
    env_factory: EnvFactory | None = None,
) -> dict[str, Any]:
    manifest = load_witness_manifest(manifest_path, verify_code=env_factory is None)
    protocol = WitnessConfirmProtocol(**dict(manifest["protocol"]))
    _, witnesses = load_witness_registry(
        _resolve_registry_path(manifest), protocol=protocol
    )
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable run: {destination}")
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes_per_run)
    budget = SdkCallBudget(protocol.maximum_sdk_calls)
    trials = []
    for witness in witnesses:
        for repetition in range(protocol.repetitions_per_route):
            trials.append(
                replay_trial(
                    witness=witness,
                    trial_type="full_route",
                    repetition=repetition,
                    budget=budget,
                    environments_dir=environments_dir,
                    env_factory=env_factory,
                )
            )
        for trial_type in ("common_suffix", "delete_last_suffix_action"):
            for repetition in range(protocol.repetitions_per_suffix_branch):
                trials.append(
                    replay_trial(
                        witness=witness,
                        trial_type=trial_type,
                        repetition=repetition,
                        budget=budget,
                        environments_dir=environments_dir,
                        env_factory=env_factory,
                        suffix_length=len(protocol.expected_common_suffix),
                    )
                )
    intervention_bundles = _intervention_bundles(
        witnesses=witnesses,
        trials=trials,
        suffix_length=len(protocol.expected_common_suffix),
    )
    passed, metrics = _metrics(
        witnesses=witnesses,
        trials=trials,
        protocol=protocol,
        budget=budget,
        intervention_bundles=intervention_bundles,
    )
    trial_path = destination / "replay_trials.json"
    _write_json_once(
        trial_path,
        {
            "format_version": "sage-t12.3a-replay-trials-v1",
            "trials": [trial.to_dict() for trial in trials],
        },
        storage_budget=storage,
    )
    bundle_path = destination / "intervention_bundles.json"
    _write_json_once(
        bundle_path,
        {
            "format_version": "sage-t12.3a-exact-prefix-intervention-bundles-v1",
            "bundles": intervention_bundles,
        },
        storage_budget=storage,
    )
    report = {
        "format_version": "sage-t12.3a-witness-report-v1",
        "status": (
            "PASS_T12_3A_WITNESS_GATE" if passed else "FAIL_T12_3A_WITNESS_GATE"
        ),
        "passed": passed,
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "parent_t12_2_receipt_checksum": manifest["parent"]["receipt"][
            "receipt_checksum"
        ],
        "metrics": metrics,
        "storage": storage.snapshot(),
    }
    report_path = destination / "witness_report.json"
    _write_json_once(report_path, report, storage_budget=storage)
    receipt = witness_phase_receipt(
        manifest=manifest,
        phase="witness_confirmation",
        passed=passed,
        status=report["status"],
        metrics=metrics,
        artifacts={
            "trials": {
                "path": str(trial_path.resolve()),
                "sha256": _file_sha256(trial_path),
            },
            "report": {
                "path": str(report_path.resolve()),
                "sha256": _file_sha256(report_path),
            },
            "intervention_bundles": {
                "path": str(bundle_path.resolve()),
                "sha256": _file_sha256(bundle_path),
            },
            "witness_registry": dict(manifest["witness_registry"]),
        },
    )
    _write_json_once(
        destination / "witness_receipt.json", receipt, storage_budget=storage
    )
    return report


def witness_experiment_status(
    *, manifest_path: str | Path, receipt_path: str | Path | None = None
) -> dict[str, Any]:
    manifest = load_witness_manifest(manifest_path)
    receipt = (
        None
        if receipt_path is None
        else load_witness_receipt(receipt_path, manifest=manifest)
    )
    return {
        "format_version": "sage-t12.3a-witness-status-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "parent_t12_2_status": manifest["parent"]["receipt"]["status"],
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
        "next_phase_authorized": bool(
            manifest.get("scientific_claims_authorized", False)
            and receipt is not None
            and receipt.get("passed") is True
        ),
        "firewall": dict(manifest["firewall"]),
    }


__all__ = [
    "ReplayTrial",
    "SdkCallBudget",
    "replay_trial",
    "run_witness_confirmation",
    "witness_experiment_status",
]
