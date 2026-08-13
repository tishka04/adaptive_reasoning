"""Exact-prefix paired execution for T12.4a.4 multi-level option transfer."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from theory.sage.live_prefix_counterfactual_collector import (
    _step_env_action,
    select_live_action,
    snapshot_frame,
    state_signature_from_frame,
)

from .contracts import GroundedAction
from .experiment import RunStorageBudget, _file_sha256, _write_json_once
from .graph_experiment import _is_terminal, _make_env
from .option_minimization_experiment import _load_contextual_option
from .option_transfer_protocol import (
    OptionTransferProtocol,
    _checksum,
    _resolve_bound,
    load_option_transfer_manifest,
    load_option_transfer_receipt,
    option_transfer_receipt,
)
from .options import MinimalCausalOption
from .witness_experiment import _execute_expected_steps, _reset_env
from .witness_protocol import ProgressWitness, WitnessStep
from .witness_reconfirmation_protocol import load_reconfirmation_registry

EnvFactory = Callable[[str], Any]


@dataclass
class TransferSdkBudget:
    maximum: int
    used: int = 0

    def consume(self, count: int = 1) -> None:
        additional = max(0, int(count))
        if self.used + additional > self.maximum:
            raise RuntimeError(
                "T12.4a.4 SDK call budget exceeded: "
                f"used={self.used} additional={additional} maximum={self.maximum}"
            )
        self.used += additional

    def snapshot(self) -> dict[str, Any]:
        return {
            "maximum_sdk_calls": self.maximum,
            "remaining_sdk_calls": self.maximum - self.used,
            "used_sdk_calls": self.used,
            "within_budget": self.used <= self.maximum,
        }


@dataclass(frozen=True)
class TransferTrial:
    trial_id: str
    stage_index: int
    branch_name: str
    repetition: int
    lineage_seed: int
    anchor_level: int
    expected_anchor_hash: str
    observed_anchor_hash: str
    prefix_exact: bool
    first_divergence: str
    anchor_comparisons: int
    anchor_trace_checksum: str
    action_names: tuple[str, ...]
    branch_available: bool
    executed_action_count: int
    branch_trace: tuple[Mapping[str, Any], ...]
    branch_trace_checksum: str
    progressed: bool
    level_delta: int
    progress_action_count: int
    final_level: int
    final_exact_hash: str
    terminal: bool
    terminal_failure: bool
    sdk_calls_after: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_names": list(self.action_names),
            "anchor_comparisons": self.anchor_comparisons,
            "anchor_level": self.anchor_level,
            "anchor_trace_checksum": self.anchor_trace_checksum,
            "branch_available": self.branch_available,
            "branch_name": self.branch_name,
            "branch_trace": [dict(item) for item in self.branch_trace],
            "branch_trace_checksum": self.branch_trace_checksum,
            "executed_action_count": self.executed_action_count,
            "expected_anchor_hash": self.expected_anchor_hash,
            "final_exact_hash": self.final_exact_hash,
            "final_level": self.final_level,
            "first_divergence": self.first_divergence,
            "level_delta": self.level_delta,
            "lineage_seed": self.lineage_seed,
            "observed_anchor_hash": self.observed_anchor_hash,
            "prefix_exact": self.prefix_exact,
            "progress_action_count": self.progress_action_count,
            "progressed": self.progressed,
            "repetition": self.repetition,
            "sdk_calls_after": self.sdk_calls_after,
            "stage_index": self.stage_index,
            "terminal": self.terminal,
            "terminal_failure": self.terminal_failure,
            "trial_id": self.trial_id,
        }


def _load_inputs(
    manifest: Mapping[str, Any],
    *,
    root: Path,
) -> tuple[tuple[ProgressWitness, ...], MinimalCausalOption]:
    witness_path = _resolve_bound(
        str(manifest["inputs"]["witness_registry"]["path"]),
        root=root,
    )
    _, witnesses = load_reconfirmation_registry(witness_path)
    option_path = _resolve_bound(
        str(manifest["inputs"]["minimal_option"]["path"]),
        root=root,
    )
    contextual = _load_contextual_option(option_path)
    option = MinimalCausalOption.from_dict(contextual["option"])
    if option.checksum != manifest["inputs"]["option_checksum"]:
        raise ValueError("T12.4a.4 option checksum differs from manifest")
    return witnesses, option


def _branch_actions(
    option: MinimalCausalOption,
    protocol: OptionTransferProtocol,
) -> dict[str, tuple[GroundedAction, ...]]:
    full = tuple(step.materialize(_empty_state()) for step in option.steps)
    action_names = tuple(action.action_name for action in full)
    if action_names != protocol.expected_option_actions:
        raise ValueError("T12.4a.4 materialized option differs from protocol")
    action4_index = action_names.index("ACTION4")
    action3_index = action_names.index("ACTION3")
    branches = {
        "option_full": full,
        "delete_action4": (
            full[:action4_index] + full[action4_index + 1 :]
        ),
        "delete_action3": (
            full[:action3_index] + full[action3_index + 1 :]
        ),
        "reverse": tuple(reversed(full)),
        "null": (),
    }
    if tuple(branches) != protocol.branch_names:
        raise ValueError("T12.4a.4 branch construction differs from protocol")
    return branches


def _empty_state():
    from theory.sage_t.contracts import AbstractState

    return AbstractState()


def _run_trial(
    *,
    game_id: str,
    environments_dir: str | Path,
    env_factory: EnvFactory | None,
    route: Sequence[WitnessStep],
    branch: Sequence[GroundedAction],
    branch_name: str,
    stage_index: int,
    repetition: int,
    lineage_seed: int,
    expected_anchor_hash: str,
    anchor_level: int,
    budget: TransferSdkBudget,
) -> TransferTrial:
    env = _make_env(game_id, environments_dir, env_factory)
    budget.consume()
    frame = _reset_env(env)
    initial_hash = state_signature_from_frame(frame)
    initial_exact = bool(route and initial_hash == route[0].expected_source_hash)
    anchor_events: list[dict[str, Any]] = []
    divergence = "" if initial_exact else "reset:initial"
    if initial_exact:
        frame, anchor_events, divergence = _execute_expected_steps(
            env=env,
            frame=frame,
            steps=route,
            phase=f"transfer_stage_{stage_index}_anchor",
            start_index=0,
            budget=budget,
        )
    observed_anchor_hash = state_signature_from_frame(frame)
    anchor_snapshot = snapshot_frame(frame)
    observed_anchor_level = int(anchor_snapshot.levels_completed)
    prefix_exact = bool(
        initial_exact
        and not divergence
        and observed_anchor_hash == expected_anchor_hash
        and observed_anchor_level == anchor_level
    )

    trace: list[dict[str, Any]] = []
    branch_available = prefix_exact
    progressed = False
    progress_action_count = 0
    if prefix_exact:
        for position, action in enumerate(branch):
            before_hash = state_signature_from_frame(frame)
            before_snapshot = snapshot_frame(frame)
            selected = select_live_action(
                env,
                action.action_name,
                action_args=action.action_data,
            )
            if selected is None:
                branch_available = False
                trace.append(
                    {
                        "action_data": dict(action.action_data),
                        "action_name": action.action_name,
                        "available": False,
                        "position": position,
                        "source_exact_hash": before_hash,
                    }
                )
                break
            budget.consume()
            frame = _step_env_action(env, selected)
            after_snapshot = snapshot_frame(frame)
            after_hash = state_signature_from_frame(frame)
            step_delta = max(
                0,
                int(after_snapshot.levels_completed)
                - int(before_snapshot.levels_completed),
            )
            terminal = _is_terminal(after_snapshot.game_state)
            trace.append(
                {
                    "action_data": dict(action.action_data),
                    "action_name": action.action_name,
                    "available": True,
                    "level_delta": step_delta,
                    "position": position,
                    "source_exact_hash": before_hash,
                    "target_exact_hash": after_hash,
                    "target_level": int(after_snapshot.levels_completed),
                    "terminal": terminal,
                }
            )
            if int(after_snapshot.levels_completed) > anchor_level:
                progressed = True
                progress_action_count = position + 1
                break
            if terminal:
                break

    final_snapshot = snapshot_frame(frame)
    final_level = int(final_snapshot.levels_completed)
    final_hash = state_signature_from_frame(frame)
    level_delta = max(0, final_level - anchor_level)
    terminal = _is_terminal(final_snapshot.game_state)
    terminal_failure = bool(terminal and level_delta == 0)
    return TransferTrial(
        trial_id=(
            f"stage_{stage_index}_{branch_name}_seed_{lineage_seed}_rep_{repetition}"
        ),
        stage_index=stage_index,
        branch_name=branch_name,
        repetition=repetition,
        lineage_seed=lineage_seed,
        anchor_level=anchor_level,
        expected_anchor_hash=expected_anchor_hash,
        observed_anchor_hash=observed_anchor_hash,
        prefix_exact=prefix_exact,
        first_divergence=divergence,
        anchor_comparisons=len(anchor_events),
        anchor_trace_checksum=_checksum(anchor_events),
        action_names=tuple(action.action_name for action in branch),
        branch_available=branch_available,
        executed_action_count=len(trace),
        branch_trace=tuple(trace),
        branch_trace_checksum=_checksum(trace),
        progressed=progressed,
        level_delta=level_delta,
        progress_action_count=progress_action_count,
        final_level=final_level,
        final_exact_hash=final_hash,
        terminal=terminal,
        terminal_failure=terminal_failure,
        sdk_calls_after=budget.used,
    )


def _stage_summary(
    trials: Sequence[TransferTrial],
    *,
    protocol: OptionTransferProtocol,
) -> dict[str, Any]:
    by_branch = {
        branch: tuple(trial for trial in trials if trial.branch_name == branch)
        for branch in protocol.branch_names
    }
    full = by_branch["option_full"]
    controls = tuple(
        trial
        for branch in protocol.branch_names
        if branch != "option_full"
        for trial in by_branch[branch]
    )
    branch_metrics = {}
    for branch, selected in by_branch.items():
        branch_metrics[branch] = {
            "available_trials": sum(trial.branch_available for trial in selected),
            "deterministic": len(
                {trial.branch_trace_checksum for trial in selected}
            )
            == 1,
            "final_exact_hashes": sorted(
                {trial.final_exact_hash for trial in selected}
            ),
            "lineage_counts": dict(
                sorted(Counter(trial.lineage_seed for trial in selected).items())
            ),
            "prefix_exact_trials": sum(trial.prefix_exact for trial in selected),
            "progressions": sum(trial.progressed for trial in selected),
            "terminal_failures": sum(trial.terminal_failure for trial in selected),
            "trials": len(selected),
        }
    full_trace_deterministic = len(
        {trial.branch_trace_checksum for trial in full}
    ) == 1
    checks = {
        "all_branches_repeated": all(
            len(selected) == protocol.repetitions_per_branch
            for selected in by_branch.values()
        ),
        "all_prefixes_exact": all(trial.prefix_exact for trial in trials),
        "controls_do_not_progress": not any(trial.progressed for trial in controls),
        "controls_have_no_terminal_failure": not any(
            trial.terminal_failure for trial in controls
        ),
        "full_option_available": all(trial.branch_available for trial in full),
        "full_option_deterministic": full_trace_deterministic,
        "full_option_progresses": all(trial.progressed for trial in full),
        "full_option_progresses_on_final_action": (
            not protocol.require_progress_on_final_option_action
            or all(
                trial.progress_action_count == len(protocol.expected_option_actions)
                for trial in full
            )
        ),
        "full_option_reaches_one_exact_target": len(
            {trial.final_exact_hash for trial in full}
        )
        == 1,
        "full_option_unit_level_delta": (
            not protocol.require_unit_level_delta
            or all(trial.level_delta == 1 for trial in full)
        ),
        "lineages_strictly_paired": all(
            tuple(trial.lineage_seed for trial in by_branch[branch])
            == protocol.lineage_schedule
            for branch in protocol.branch_names
        ),
        "null_control_preserves_anchor": all(
            trial.final_exact_hash == trial.expected_anchor_hash
            for trial in by_branch["null"]
        ),
    }
    passed = all(checks.values())
    return {
        "branch_metrics": branch_metrics,
        "checks": checks,
        "passed": passed,
        "stage_index": trials[0].stage_index,
        "source_level": trials[0].anchor_level,
        "target_exact_hash": full[0].final_exact_hash if passed else None,
        "target_level": full[0].final_level if passed else None,
        "target_terminal": bool(passed and any(trial.terminal for trial in full)),
    }


def _extension_from_trial(trial: TransferTrial) -> tuple[WitnessStep, ...]:
    if not trial.progressed:
        raise ValueError("T12.4a.4 cannot extend a non-progressing trial")
    return tuple(
        WitnessStep(
            expected_source_hash=str(item["source_exact_hash"]),
            action=GroundedAction(
                str(item["action_name"]),
                dict(item.get("action_data", {}) or {}),
            ),
            expected_target_hash=str(item["target_exact_hash"]),
            level_delta=int(item.get("level_delta", 0)),
            terminal=bool(item.get("terminal", False)),
            success=int(item.get("level_delta", 0)) > 0,
        )
        for item in trial.branch_trace
    )


def run_option_transfer(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    environments_dir: str | Path = "environment_files",
    env_factory: EnvFactory | None = None,
) -> dict[str, Any]:
    manifest = load_option_transfer_manifest(
        manifest_path,
        verify_code=env_factory is None,
    )
    if not manifest["firewall"]["option_transfer_experiment_authorized"]:
        raise ValueError("T12.4a.4 transfer experiment is not authorized")
    protocol = OptionTransferProtocol(**dict(manifest["protocol"]))
    root = Path(__file__).resolve().parents[3]
    witnesses, option = _load_inputs(manifest, root=root)
    routes = {item.source_seed: tuple(item.steps) for item in witnesses}
    branches = _branch_actions(option, protocol)
    expected_anchor_hash = str(manifest["inputs"]["entry_exact_hash"])
    anchor_level = int(manifest["inputs"]["entry_level"])
    extensions: tuple[WitnessStep, ...] = ()
    budget = TransferSdkBudget(protocol.maximum_sdk_calls)
    trials: list[TransferTrial] = []
    stages: list[dict[str, Any]] = []
    confirmed_levels = 0

    for stage_index in range(1, protocol.maximum_transfer_levels + 1):
        stage_trials: list[TransferTrial] = []
        for branch_name in protocol.branch_names:
            for repetition, lineage_seed in enumerate(protocol.lineage_schedule):
                route = routes[lineage_seed] + extensions
                trial = _run_trial(
                    game_id=str(manifest["game_id"]),
                    environments_dir=environments_dir,
                    env_factory=env_factory,
                    route=route,
                    branch=branches[branch_name],
                    branch_name=branch_name,
                    stage_index=stage_index,
                    repetition=repetition,
                    lineage_seed=lineage_seed,
                    expected_anchor_hash=expected_anchor_hash,
                    anchor_level=anchor_level,
                    budget=budget,
                )
                stage_trials.append(trial)
                trials.append(trial)
        stage = _stage_summary(stage_trials, protocol=protocol)
        stages.append(stage)
        if not stage["passed"]:
            break
        confirmed_levels += 1
        first_full = next(
            trial for trial in stage_trials if trial.branch_name == "option_full"
        )
        extensions = extensions + _extension_from_trial(first_full)
        expected_anchor_hash = str(stage["target_exact_hash"])
        anchor_level = int(stage["target_level"])
        if stage["target_terminal"]:
            break

    prefix_exact_trials = sum(trial.prefix_exact for trial in trials)
    terminal_failures = sum(trial.terminal_failure for trial in trials)
    lineage_pairing = all(
        tuple(
            trial.lineage_seed
            for trial in trials
            if trial.stage_index == stage["stage_index"]
            and trial.branch_name == branch
        )
        == protocol.lineage_schedule
        for stage in stages
        for branch in protocol.branch_names
    )
    checks = {
        "all_executed_prefixes_exact": prefix_exact_trials == len(trials),
        "minimum_transferred_levels_reached": (
            confirmed_levels >= protocol.minimum_transferred_levels
        ),
        "no_terminal_failures": (
            terminal_failures <= protocol.maximum_terminal_failures
        ),
        "posterior_owner_mass_preserved_in_shadow": (
            float(manifest["inputs"]["posterior_owner_mass"]) + 1e-12
            >= protocol.minimum_posterior_owner_mass
        ),
        "sdk_budget_respected": budget.used <= protocol.maximum_sdk_calls,
        "strict_lineage_pairing": lineage_pairing,
    }
    passed = all(checks.values())
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable run: {destination}")
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes_per_run)
    trials_path = destination / "transfer_trials.json"
    bundles_path = destination / "intervention_bundles.json"
    report_path = destination / "transfer_report.json"
    receipt_path = destination / "transfer_receipt.json"
    _write_json_once(
        trials_path,
        {
            "format_version": "sage-t12.4a.4-transfer-trials-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "trials": [trial.to_dict() for trial in trials],
        },
        storage_budget=storage,
    )
    _write_json_once(
        bundles_path,
        {
            "format_version": "sage-t12.4a.4-intervention-bundles-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "stages": stages,
        },
        storage_budget=storage,
    )
    metrics = {
        "attempted_transfer_levels": len(stages),
        "branch_count": len(trials),
        "checks": checks,
        "confirmed_transfer_levels": confirmed_levels,
        "entry_level": int(manifest["inputs"]["entry_level"]),
        "final_confirmed_level": (
            int(manifest["inputs"]["entry_level"]) + confirmed_levels
        ),
        "option_action_count": len(option.steps),
        "prefix_exact_trials": prefix_exact_trials,
        "sdk_calls": budget.snapshot(),
        "stage_results": stages,
        "storage": storage.snapshot(),
        "terminal_failures": terminal_failures,
        "trial_count": len(trials),
    }
    report = {
        "format_version": "sage-t12.4a.4-option-transfer-report-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "metrics": metrics,
        "passed": passed,
        "protocol_checksum": manifest["protocol_checksum"],
        "status": (
            "PASS_T12_4A_4_OPTION_TRANSFER_GATE"
            if passed
            else "FAIL_T12_4A_4_OPTION_TRANSFER_GATE"
        ),
        "storage": storage.snapshot(),
    }
    _write_json_once(report_path, report, storage_budget=storage)
    receipt = option_transfer_receipt(
        manifest=manifest,
        phase="option_transfer",
        passed=passed,
        status=report["status"],
        metrics=metrics,
        artifacts={
            "bundles": {
                "path": str(bundles_path.resolve()),
                "sha256": _file_sha256(bundles_path),
            },
            "report": {
                "path": str(report_path.resolve()),
                "sha256": _file_sha256(report_path),
            },
            "trials": {
                "path": str(trials_path.resolve()),
                "sha256": _file_sha256(trials_path),
            },
        },
    )
    _write_json_once(receipt_path, receipt, storage_budget=storage)
    return receipt


def option_transfer_status(
    *,
    manifest_path: str | Path,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest = load_option_transfer_manifest(manifest_path)
    receipt = None
    if receipt_path is not None and Path(receipt_path).is_file():
        receipt = load_option_transfer_receipt(
            receipt_path,
            manifest=manifest,
        )
    passed = bool(
        receipt
        and receipt.get("passed")
        and receipt.get("phase") == "option_transfer"
        and receipt.get("status") == "PASS_T12_4A_4_OPTION_TRANSFER_GATE"
    )
    return {
        "firewall": {
            "holdout_opened": False,
            "neural_active_evaluation_authorized": False,
            "option_control_authorized": False,
            "option_transfer_experiment_authorized": manifest["firewall"][
                "option_transfer_experiment_authorized"
            ],
            "production_authority": False,
            "source_validation_opened": False,
            "t12_4a_5_option_control_freeze_authorized": passed,
            "t12_4b_freeze_authorized": False,
            "t12_5_freeze_authorized": False,
            "terminal_shield_production_authority": False,
        },
        "format_version": "sage-t12.4a.4-option-transfer-status-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "next_phase_authorized": passed,
        "parent_t12_4a_3_status": manifest["parent"]["compile_receipt"]["status"],
        "protocol_checksum": manifest["protocol_checksum"],
        "receipt": receipt,
    }


__all__ = ["option_transfer_status", "run_option_transfer"]
