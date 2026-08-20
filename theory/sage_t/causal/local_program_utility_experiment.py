"""Two-phase physical experiment for SAGE.T12.5b.4 local program utility."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
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
from .local_program_utility import (
    audit_calibration_trials,
    audit_evaluation_trials,
    evaluation_registry_payload,
    program_id,
)
from .local_program_utility_protocol import (
    LocalProgramUtilityProtocol,
    _checksum,
    load_local_program_utility_manifest,
    load_local_program_utility_receipt,
    load_signed_evaluation_registry,
    local_program_utility_receipt,
)
from .option_applicability_experiment import _structured_delta
from .progress import ProgressMilestone
from .progress_shadow import (
    posterior_from_snapshot,
    progress_signature,
    projected_effect_step,
)
from .progress_shadow_experiment import (
    _descriptor,
    _expected_stage_hashes,
    _load_inputs,
    _materialize_option,
    _ordered_milestones,
)
from .witness_experiment import _execute_expected_steps, _reset_env
from .witness_protocol import ProgressWitness

EnvFactory = Callable[[str], Any]
CALIBRATION_TRIALS_FORMAT = "sage-t12.5b.4-calibration-trials-v1"
EVALUATION_TRIALS_FORMAT = "sage-t12.5b.4-evaluation-trials-v1"


@dataclass
class LocalUtilityBudget:
    maximum_sdk_calls: int
    maximum_wall_seconds: int
    used_sdk_calls: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def consume(self, count: int = 1, *, reason: str = "replay") -> None:
        additional = max(0, int(count))
        if self.used_sdk_calls + additional > self.maximum_sdk_calls:
            raise RuntimeError(
                "T12.5b.4 SDK call budget exceeded: "
                f"used={self.used_sdk_calls} additional={additional} "
                f"maximum={self.maximum_sdk_calls} reason={reason}"
            )
        if self.elapsed_seconds > self.maximum_wall_seconds:
            raise RuntimeError(
                "T12.5b.4 wall-time budget exceeded before environment action"
            )
        self.used_sdk_calls += additional

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, time.monotonic() - self.started_at)

    def snapshot(self) -> dict[str, Any]:
        elapsed = self.elapsed_seconds
        return {
            "elapsed_seconds": elapsed,
            "maximum_sdk_calls": self.maximum_sdk_calls,
            "maximum_wall_seconds": self.maximum_wall_seconds,
            "remaining_sdk_calls": self.maximum_sdk_calls - self.used_sdk_calls,
            "used_sdk_calls": self.used_sdk_calls,
            "within_sdk_budget": self.used_sdk_calls <= self.maximum_sdk_calls,
            "within_wall_time": elapsed <= self.maximum_wall_seconds,
        }


@dataclass(frozen=True)
class LocalProgramTrial:
    trial_id: str
    phase: str
    lineage_seed: int
    context_id: str
    program_id: str
    program_actions: tuple[str, ...]
    repetition: int
    original_prefix_exact: bool
    detour_available: bool
    detour_neutral: bool
    detour_terminal: bool
    detour_context_hash: str
    prefix_exact: bool
    prefix_steps: tuple[Mapping[str, Any], ...]
    candidate_steps: tuple[Mapping[str, Any], ...]
    executed_action_count: int
    program_complete: bool
    level_delta: int
    progressed: bool
    terminal: bool
    terminal_failure: bool
    terminal_state: str
    sdk_calls_after: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_steps": [dict(item) for item in self.candidate_steps],
            "context_id": self.context_id,
            "detour_available": self.detour_available,
            "detour_context_hash": self.detour_context_hash,
            "detour_neutral": self.detour_neutral,
            "detour_terminal": self.detour_terminal,
            "executed_action_count": self.executed_action_count,
            "level_delta": self.level_delta,
            "lineage_seed": self.lineage_seed,
            "original_prefix_exact": self.original_prefix_exact,
            "phase": self.phase,
            "prefix_exact": self.prefix_exact,
            "prefix_steps": [dict(item) for item in self.prefix_steps],
            "program_actions": list(self.program_actions),
            "program_complete": self.program_complete,
            "program_id": self.program_id,
            "progressed": self.progressed,
            "repetition": self.repetition,
            "sdk_calls_after": self.sdk_calls_after,
            "terminal": self.terminal,
            "terminal_failure": self.terminal_failure,
            "terminal_state": self.terminal_state,
            "trial_id": self.trial_id,
        }


def _state_name(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw)


def _execute_program_trial(
    *,
    phase: str,
    game_id: str,
    environments_dir: str | Path,
    env_factory: EnvFactory | None,
    witness: ProgressWitness,
    witness_prefix_length: int,
    option_actions: Sequence[GroundedAction],
    target_stage: int,
    detour_action: str,
    detour_depth: int,
    actions: Sequence[str],
    repetition: int,
    expected_stage_hash: str,
    features: Sequence[str],
    milestones: Sequence[ProgressMilestone],
    budget: LocalUtilityBudget,
) -> LocalProgramTrial:
    normalized_actions = tuple(str(item).upper() for item in actions)
    identifier = program_id(normalized_actions)
    env = _make_env(game_id, environments_dir, env_factory)
    budget.consume(reason="reset")
    frame = _reset_env(env)
    initial_hash = state_signature_from_frame(frame)
    initial_exact = bool(
        witness.steps and initial_hash == witness.steps[0].expected_source_hash
    )
    divergence = "" if initial_exact else "reset:initial"
    route = tuple(witness.steps[: int(witness_prefix_length)])
    if initial_exact:
        frame, _, divergence = _execute_expected_steps(
            env=env,
            frame=frame,
            steps=route,
            phase=f"t12_5b_4_{phase}_{witness.source_seed}",
            start_index=0,
            budget=budget,
        )

    prefix_steps: list[dict[str, Any]] = []
    prefix_available = bool(initial_exact and not divergence)
    if prefix_available:
        for position, action in enumerate(option_actions[:target_stage]):
            before = _descriptor(frame)
            selected = select_live_action(
                env,
                action.action_name,
                action_args=action.action_data,
            )
            if selected is None:
                prefix_available = False
                break
            budget.consume(reason="stage_prefix")
            frame = _step_env_action(env, selected)
            after = _descriptor(frame)
            prefix_steps.append(
                projected_effect_step(
                    {
                        "action_name": action.action_name,
                        "available": True,
                        "delta": _structured_delta(before, after),
                        "position": position,
                    },
                    features=features,
                )
            )
    original_prefix_exact = bool(
        prefix_available
        and state_signature_from_frame(frame) == expected_stage_hash
        and len(prefix_steps) == int(target_stage)
    )

    detour_steps: list[dict[str, Any]] = []
    detour_available = original_prefix_exact
    detour_neutral = original_prefix_exact
    detour_terminal = False
    if original_prefix_exact:
        for index in range(int(detour_depth)):
            before = _descriptor(frame)
            selected = select_live_action(env, detour_action, action_args={})
            if selected is None:
                detour_available = False
                break
            budget.consume(reason="detour")
            frame = _step_env_action(env, selected)
            after = _descriptor(frame)
            step = projected_effect_step(
                {
                    "action_name": detour_action,
                    "available": True,
                    "delta": _structured_delta(before, after),
                    "position": target_stage + index,
                },
                features=features,
            )
            detour_steps.append(step)
            if any(progress_signature(step, milestones)):
                detour_neutral = False
            if _is_terminal(snapshot_frame(frame).game_state):
                detour_terminal = True
                break
    detour_available = bool(
        detour_available and len(detour_steps) == int(detour_depth)
    )
    detour_context_hash = state_signature_from_frame(frame)
    prefix_steps.extend(detour_steps)
    preliminary_context_valid = bool(
        original_prefix_exact
        and detour_available
        and detour_neutral
        and not detour_terminal
    )

    before_snapshot = snapshot_frame(frame)
    before_level = int(before_snapshot.levels_completed)
    candidate_steps: list[dict[str, Any]] = []
    executed = 0
    if preliminary_context_valid:
        for index, action_name in enumerate(normalized_actions):
            before = _descriptor(frame)
            selected = select_live_action(env, action_name, action_args={})
            if selected is None:
                candidate_steps.append(
                    projected_effect_step(
                        {
                            "action_name": action_name,
                            "available": False,
                            "delta": {"mechanism": {}},
                            "position": len(prefix_steps) + index,
                        },
                        features=features,
                    )
                )
                break
            budget.consume(reason="candidate_program")
            frame = _step_env_action(env, selected)
            executed += 1
            after = _descriptor(frame)
            candidate_steps.append(
                projected_effect_step(
                    {
                        "action_name": action_name,
                        "available": True,
                        "delta": _structured_delta(before, after),
                        "position": len(prefix_steps) + index,
                    },
                    features=features,
                )
            )
            snapshot = snapshot_frame(frame)
            if (
                _is_terminal(snapshot.game_state)
                or int(snapshot.levels_completed) > before_level
            ):
                break
    after_snapshot = snapshot_frame(frame)
    level_delta = int(after_snapshot.levels_completed) - before_level
    terminal = _is_terminal(after_snapshot.game_state)
    terminal_failure = bool(terminal and level_delta <= 0)
    complete = executed == len(normalized_actions)
    return LocalProgramTrial(
        trial_id=(
            f"{phase}_lineage_{witness.source_seed}_{identifier.lower()}_"
            f"rep_{int(repetition)}"
        ),
        phase=str(phase),
        lineage_seed=int(witness.source_seed),
        context_id="stage_3_action4_depth_1",
        program_id=identifier,
        program_actions=normalized_actions,
        repetition=int(repetition),
        original_prefix_exact=original_prefix_exact,
        detour_available=detour_available,
        detour_neutral=detour_neutral,
        detour_terminal=detour_terminal,
        detour_context_hash=detour_context_hash,
        prefix_exact=preliminary_context_valid,
        prefix_steps=tuple(prefix_steps),
        candidate_steps=tuple(candidate_steps),
        executed_action_count=executed,
        program_complete=complete,
        level_delta=level_delta,
        progressed=level_delta > 0,
        terminal=terminal,
        terminal_failure=terminal_failure,
        terminal_state=_state_name(after_snapshot.game_state),
        sdk_calls_after=budget.used_sdk_calls,
    )


def _annotate_context_exactness(
    trials: Sequence[LocalProgramTrial],
) -> tuple[LocalProgramTrial, ...]:
    hashes = {item.detour_context_hash for item in trials}
    prefixes = {_checksum([dict(step) for step in item.prefix_steps]) for item in trials}
    exact = bool(
        len(hashes) == 1
        and len(prefixes) == 1
        and all(item.prefix_exact for item in trials)
    )
    return tuple(replace(item, prefix_exact=exact) for item in trials)


def _runtime_inputs(
    manifest: Mapping[str, Any],
    *,
    protocol: LocalProgramUtilityProtocol,
    root: Path,
) -> tuple[
    dict[int, ProgressWitness],
    tuple[GroundedAction, ...],
    Any,
    tuple[ProgressMilestone, ...],
    dict[tuple[int, int], str],
]:
    witnesses, option, posterior_payload, programs, applicability = _load_inputs(
        manifest,
        root=root,
    )
    by_seed = {int(item.source_seed): item for item in witnesses}
    expected_seeds = {
        protocol.calibration_lineage_seed,
        protocol.evaluation_lineage_seed,
    }
    if set(by_seed) != expected_seeds:
        raise ValueError("T12.5b.4 witness lineage set changed")
    option_actions = _materialize_option(option)
    if tuple(item.action_name for item in option_actions) != (
        "ACTION4",
        "ACTION4",
        "ACTION4",
        "ACTION3",
        "ACTION3",
    ):
        raise ValueError("T12.5b.4 parent option changed")
    milestones = _ordered_milestones(programs)
    if len(milestones) != 5:
        raise ValueError("T12.5b.4 milestone count changed")
    posterior = posterior_from_snapshot(posterior_payload)
    expected_hashes = _expected_stage_hashes(
        applicability,
        lineage_seeds=tuple(sorted(expected_seeds)),
    )
    return by_seed, option_actions, posterior, milestones, expected_hashes


def _collect_programs(
    *,
    phase: str,
    manifest: Mapping[str, Any],
    protocol: LocalProgramUtilityProtocol,
    lineage_seed: int,
    programs: Sequence[Sequence[str]],
    environments_dir: str | Path,
    env_factory: EnvFactory | None,
    budget: LocalUtilityBudget,
    root: Path,
) -> tuple[tuple[LocalProgramTrial, ...], Any]:
    witnesses, option_actions, posterior, milestones, expected_hashes = (
        _runtime_inputs(manifest, protocol=protocol, root=root)
    )
    witness = witnesses[int(lineage_seed)]
    prefix_length = int(
        manifest["inputs"]["successful_prefix_lengths"][str(lineage_seed)]
    )
    trials = tuple(
        _execute_program_trial(
            phase=phase,
            game_id=str(manifest["game_id"]),
            environments_dir=environments_dir,
            env_factory=env_factory,
            witness=witness,
            witness_prefix_length=prefix_length,
            option_actions=option_actions,
            target_stage=protocol.target_stage,
            detour_action=protocol.detour_action,
            detour_depth=protocol.detour_depth,
            actions=actions,
            repetition=repetition,
            expected_stage_hash=expected_hashes[(lineage_seed, protocol.target_stage)],
            features=protocol.allowed_effect_features,
            milestones=milestones,
            budget=budget,
        )
        for actions in programs
        for repetition in range(protocol.repetitions_per_program)
    )
    return _annotate_context_exactness(trials), posterior


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _file_sha256(path)}


def run_local_program_calibration(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    environments_dir: str | Path = "environment_files",
    env_factory: EnvFactory | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_local_program_utility_manifest(
        manifest_path,
        root=repo_root,
        verify_code=env_factory is None,
    )
    if not manifest["firewall"].get("calibration_collection_authorized", False):
        raise ValueError("T12.5b.4 manifest does not authorize calibration")
    protocol = LocalProgramUtilityProtocol(**dict(manifest["protocol"]))
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable run: {destination}")
    storage = RunStorageBudget(
        destination,
        protocol.maximum_artifact_bytes_per_phase,
    )
    budget = LocalUtilityBudget(
        maximum_sdk_calls=protocol.maximum_calibration_sdk_calls,
        maximum_wall_seconds=protocol.maximum_wall_seconds_per_phase,
    )
    trials, posterior = _collect_programs(
        phase="calibration",
        manifest=manifest,
        protocol=protocol,
        lineage_seed=protocol.calibration_lineage_seed,
        programs=protocol.calibration_programs,
        environments_dir=environments_dir,
        env_factory=env_factory,
        budget=budget,
        root=repo_root,
    )
    audit = audit_calibration_trials(
        trials=[item.to_dict() for item in trials],
        expected_programs=protocol.calibration_programs,
        repetitions_per_program=protocol.repetitions_per_program,
        transport_actions=protocol.transport_actions,
        features=protocol.allowed_effect_features,
        posterior=posterior,
        minimum_distractor_magnitude_gap=(
            protocol.minimum_distractor_magnitude_gap
        ),
    )
    metrics = dict(audit["metrics"])
    sdk = budget.snapshot()
    integrity_checks = {
        "availability_is_deterministic": metrics["availability_is_deterministic"],
        "context_replay_is_exact": metrics["context_replay_is_exact"],
        "effects_are_deterministic": metrics["effects_are_deterministic"],
        "effects_are_deterministic_when_complete": metrics[
            "effects_are_deterministic_when_complete"
        ],
        "fixed_program_schedule_completed": metrics[
            "fixed_program_schedule_completed"
        ],
        "outcomes_are_deterministic": metrics["outcomes_are_deterministic"],
        "repetition_count_is_exact": metrics["repetition_count_is_exact"],
        "sdk_budget_respected": sdk["within_sdk_budget"],
        "wall_time_respected": sdk["within_wall_time"],
    }
    scientific_checks = {
        "causal_gain_beats_registered_distractor": metrics[
            "causal_contrast_correct"
        ],
        "hard_utility_contrast_exists": metrics["hard_utility_contrast_count"] == 1,
        "transport_safe_progress_program_exists": metrics[
            "transport_safe_progress_program_count"
        ]
        >= 1,
    }
    integrity_passed = all(integrity_checks.values())
    progress_passed = scientific_checks["transport_safe_progress_program_exists"]
    contrast_passed = scientific_checks["hard_utility_contrast_exists"]
    causal_passed = scientific_checks["causal_gain_beats_registered_distractor"]
    passed = bool(
        integrity_passed and progress_passed and contrast_passed and causal_passed
    )
    if not integrity_passed:
        classification = "CALIBRATION_INTEGRITY_FAILURE"
        status = "FAIL_T12_5B_4_CALIBRATION_INTEGRITY_GATE"
    elif not progress_passed:
        classification = "NO_LOCAL_PROGRESS_PROGRAM"
        status = "FAIL_T12_5B_4_NO_LOCAL_PROGRESS_PROGRAM"
    elif not contrast_passed:
        classification = "NO_HARD_LOCAL_UTILITY_CONTRAST"
        status = "FAIL_T12_5B_4_NO_HARD_UTILITY_CONTRAST"
    elif not causal_passed:
        classification = "CAUSAL_LOCAL_UTILITY_NOT_DISCRIMINATIVE"
        status = "FAIL_T12_5B_4_CAUSAL_UTILITY_GATE"
    else:
        classification = "CAUSAL_LOCAL_UTILITY_CALIBRATED"
        status = "PASS_T12_5B_4_CALIBRATION_GATE"
    metrics.update(
        {
            "checks": {**integrity_checks, **scientific_checks},
            "classification": classification,
            "evaluation_collection_authorized": passed,
            "sdk_calls": sdk,
        }
    )

    trials_path = destination / "calibration_trials.json"
    registry_path = destination / "local_program_registry.json"
    report_path = destination / "calibration_report.json"
    evaluation_path = destination / "evaluation_registry.json"
    _write_json_once(
        trials_path,
        {
            "format_version": CALIBRATION_TRIALS_FORMAT,
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "trials": [item.to_dict() for item in trials],
        },
        storage_budget=storage,
    )
    program_registry = {
        **audit["program_registry"],
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
    }
    _write_json_once(registry_path, program_registry, storage_budget=storage)
    artifacts = {
        "programs": _artifact(registry_path),
        "trials": _artifact(trials_path),
    }
    if passed:
        evidence_checksum = _checksum(
            {
                "programs_sha256": artifacts["programs"]["sha256"],
                "selection": audit["selection"],
                "trials_sha256": artifacts["trials"]["sha256"],
            }
        )
        evaluation_registry = evaluation_registry_payload(
            manifest_checksum=manifest["manifest_checksum"],
            protocol_checksum=manifest["protocol_checksum"],
            calibration_evidence_checksum=evidence_checksum,
            selection=audit["selection"],
        )
        _write_json_once(
            evaluation_path,
            evaluation_registry,
            storage_budget=storage,
        )
        artifacts["evaluation_registry"] = _artifact(evaluation_path)
    storage_snapshot = storage.snapshot()
    metrics["storage"] = storage_snapshot
    report = {
        "claim_boundary": manifest["claim_boundary"],
        "format_version": "sage-t12.5b.4-calibration-report-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "metrics": metrics,
        "passed": passed,
        "protocol_checksum": manifest["protocol_checksum"],
        "selection": audit["selection"],
        "status": status,
    }
    _write_json_once(report_path, report, storage_budget=storage)
    artifacts["report"] = _artifact(report_path)
    receipt = local_program_utility_receipt(
        manifest=manifest,
        phase="calibration",
        passed=passed,
        status=status,
        metrics=metrics,
        artifacts=artifacts,
    )
    _write_json_once(
        destination / "calibration_receipt.json",
        receipt,
        storage_budget=storage,
    )
    return receipt


def run_local_program_evaluation(
    *,
    manifest_path: str | Path,
    calibration_receipt_path: str | Path,
    output_dir: str | Path,
    environments_dir: str | Path = "environment_files",
    env_factory: EnvFactory | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_local_program_utility_manifest(
        manifest_path,
        root=repo_root,
        verify_code=env_factory is None,
    )
    calibration = load_local_program_utility_receipt(
        calibration_receipt_path,
        manifest=manifest,
        root=repo_root,
        require_passed=True,
        expected_phase="calibration",
    )
    if calibration.get("status") != "PASS_T12_5B_4_CALIBRATION_GATE":
        raise ValueError("T12.5b.4 evaluation requires the calibration pass")
    registry_meta = calibration["artifacts"].get("evaluation_registry")
    if not isinstance(registry_meta, Mapping):
        raise ValueError("T12.5b.4 calibration did not seal an evaluation registry")
    registry_path = Path(str(registry_meta["path"]))
    evaluation_registry = load_signed_evaluation_registry(
        registry_path,
        manifest=manifest,
    )
    protocol = LocalProgramUtilityProtocol(**dict(manifest["protocol"]))
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable run: {destination}")
    storage = RunStorageBudget(
        destination,
        protocol.maximum_artifact_bytes_per_phase,
    )
    budget = LocalUtilityBudget(
        maximum_sdk_calls=protocol.maximum_evaluation_sdk_calls,
        maximum_wall_seconds=protocol.maximum_wall_seconds_per_phase,
    )
    registered = dict(evaluation_registry["programs"])
    programs = (
        tuple(registered["progress"]["program_actions"]),
        tuple(registered["distractor"]["program_actions"]),
    )
    trials, posterior = _collect_programs(
        phase="evaluation",
        manifest=manifest,
        protocol=protocol,
        lineage_seed=protocol.evaluation_lineage_seed,
        programs=programs,
        environments_dir=environments_dir,
        env_factory=env_factory,
        budget=budget,
        root=repo_root,
    )
    audit = audit_evaluation_trials(
        trials=[item.to_dict() for item in trials],
        evaluation_registry=evaluation_registry,
        repetitions_per_program=protocol.repetitions_per_program,
        transport_actions=protocol.transport_actions,
        features=protocol.allowed_effect_features,
        posterior=posterior,
    )
    metrics = dict(audit["metrics"])
    sdk = budget.snapshot()
    calibration_sdk_calls = int(
        calibration.get("metrics", {}).get("sdk_calls", {}).get("used_sdk_calls", 0)
    )
    total_sdk_calls = calibration_sdk_calls + int(sdk["used_sdk_calls"])
    integrity_checks = {
        "availability_is_deterministic": metrics["availability_is_deterministic"],
        "context_replay_is_exact": metrics["context_replay_is_exact"],
        "effects_are_deterministic": metrics["effects_are_deterministic"],
        "effects_are_deterministic_when_complete": metrics[
            "effects_are_deterministic_when_complete"
        ],
        "fixed_program_schedule_completed": metrics[
            "fixed_program_schedule_completed"
        ],
        "outcomes_are_deterministic": metrics["outcomes_are_deterministic"],
        "repetition_count_is_exact": metrics["repetition_count_is_exact"],
        "sdk_budget_respected": sdk["within_sdk_budget"],
        "total_sdk_budget_respected": total_sdk_calls
        <= protocol.maximum_total_sdk_calls,
        "wall_time_respected": sdk["within_wall_time"],
    }
    scientific_checks = {
        "causal_utility_transferred": metrics["causal_utility_transferred"],
        "distractor_stable_safe_nonprogress": metrics[
            "distractor_stable_safe_nonprogress"
        ],
        "progress_program_transferred": metrics["progress_program_transferred"],
    }
    integrity_passed = all(integrity_checks.values())
    progress_passed = scientific_checks["progress_program_transferred"]
    distractor_passed = scientific_checks["distractor_stable_safe_nonprogress"]
    causal_passed = scientific_checks["causal_utility_transferred"]
    passed = bool(
        integrity_passed and progress_passed and distractor_passed and causal_passed
    )
    if not integrity_passed:
        classification = "EVALUATION_INTEGRITY_FAILURE"
        status = "FAIL_T12_5B_4_EVALUATION_INTEGRITY_GATE"
    elif not progress_passed:
        classification = "LOCAL_PROGRESS_PROGRAM_DID_NOT_TRANSFER"
        status = "FAIL_T12_5B_4_PROGRESS_PROGRAM_TRANSFER_GATE"
    elif not distractor_passed:
        classification = "LOCAL_UTILITY_DISTRACTOR_UNSTABLE"
        status = "FAIL_T12_5B_4_DISTRACTOR_STABILITY_GATE"
    elif not causal_passed:
        classification = "CAUSAL_LOCAL_UTILITY_DID_NOT_TRANSFER"
        status = "FAIL_T12_5B_4_CAUSAL_UTILITY_TRANSFER_GATE"
    else:
        classification = "CAUSAL_LOCAL_PROGRAM_UTILITY_TRANSFERS"
        status = "PASS_T12_5B_4_LOCAL_PROGRAM_UTILITY_GATE"
    metrics.update(
        {
            "checks": {**integrity_checks, **scientific_checks},
            "classification": classification,
            "sdk_calls": sdk,
            "t12_5c_control_freeze_authorized": passed,
            "total_sdk_calls": total_sdk_calls,
        }
    )

    trials_path = destination / "evaluation_trials.json"
    registry_out_path = destination / "evaluation_program_registry.json"
    report_path = destination / "evaluation_report.json"
    _write_json_once(
        trials_path,
        {
            "format_version": EVALUATION_TRIALS_FORMAT,
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "trials": [item.to_dict() for item in trials],
        },
        storage_budget=storage,
    )
    registry_out = {
        **audit["program_registry"],
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "source_evaluation_registry_checksum": evaluation_registry[
            "registry_checksum"
        ],
    }
    _write_json_once(registry_out_path, registry_out, storage_budget=storage)
    metrics["storage"] = storage.snapshot()
    report = {
        "claim_boundary": manifest["claim_boundary"],
        "format_version": "sage-t12.5b.4-evaluation-report-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "metrics": metrics,
        "passed": passed,
        "protocol_checksum": manifest["protocol_checksum"],
        "registered_distractor": audit["registered_distractor"],
        "registered_progress": audit["registered_progress"],
        "status": status,
    }
    _write_json_once(report_path, report, storage_budget=storage)
    artifacts = {
        "calibration_receipt": _artifact(Path(calibration_receipt_path)),
        "evaluation_registry": _artifact(registry_path),
        "programs": _artifact(registry_out_path),
        "report": _artifact(report_path),
        "trials": _artifact(trials_path),
    }
    receipt = local_program_utility_receipt(
        manifest=manifest,
        phase="evaluation",
        passed=passed,
        status=status,
        metrics=metrics,
        artifacts=artifacts,
    )
    _write_json_once(
        destination / "evaluation_receipt.json",
        receipt,
        storage_budget=storage,
    )
    return receipt


def local_program_utility_status(
    *,
    manifest_path: str | Path,
    calibration_receipt_path: str | Path | None = None,
    evaluation_receipt_path: str | Path | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_local_program_utility_manifest(
        manifest_path,
        root=repo_root,
    )
    calibration = (
        None
        if calibration_receipt_path is None
        or not Path(calibration_receipt_path).is_file()
        else load_local_program_utility_receipt(
            calibration_receipt_path,
            manifest=manifest,
            root=repo_root,
            expected_phase="calibration",
        )
    )
    evaluation = (
        None
        if evaluation_receipt_path is None
        or not Path(evaluation_receipt_path).is_file()
        else load_local_program_utility_receipt(
            evaluation_receipt_path,
            manifest=manifest,
            root=repo_root,
            expected_phase="evaluation",
        )
    )
    calibration_passed = bool(
        calibration
        and calibration.get("passed") is True
        and calibration.get("status") == "PASS_T12_5B_4_CALIBRATION_GATE"
    )
    evaluation_passed = bool(
        evaluation
        and evaluation.get("passed") is True
        and evaluation.get("status")
        == "PASS_T12_5B_4_LOCAL_PROGRAM_UTILITY_GATE"
    )
    calibration_ready = bool(
        calibration is None
        and manifest["firewall"].get("calibration_collection_authorized", False)
    )
    evaluation_ready = bool(calibration_passed and evaluation is None)
    return {
        "claim_boundary": manifest["claim_boundary"],
        "firewall": {
            "calibration_collection_authorized": calibration_ready,
            "evaluation_collection_authorized": evaluation_ready,
            "environment_collection_authorized": (
                calibration_ready or evaluation_ready
            ),
            "causal_progress_control_authorized": False,
            "holdout_opened": False,
            "neural_active_evaluation_authorized": False,
            "neural_training_authorized": False,
            "option_control_authorized": False,
            "production_authority": False,
            "source_validation_opened": False,
            "t12_5c_control_freeze_authorized": evaluation_passed,
            "t12_6_freeze_authorized": False,
        },
        "format_version": "sage-t12.5b.4-local-program-utility-status-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "next_phase_authorized": evaluation_passed,
        "parent_t12_5b_3_status": manifest["parent"]["receipt"]["status"],
        "protocol_checksum": manifest["protocol_checksum"],
        "calibration_receipt": (
            None
            if calibration is None
            else {
                "classification": calibration.get("metrics", {}).get(
                    "classification"
                ),
                "passed": calibration["passed"],
                "receipt_checksum": calibration["receipt_checksum"],
                "status": calibration["status"],
            }
        ),
        "evaluation_receipt": (
            None
            if evaluation is None
            else {
                "classification": evaluation.get("metrics", {}).get(
                    "classification"
                ),
                "passed": evaluation["passed"],
                "receipt_checksum": evaluation["receipt_checksum"],
                "status": evaluation["status"],
            }
        ),
    }


__all__ = [
    "LocalProgramTrial",
    "LocalUtilityBudget",
    "local_program_utility_status",
    "run_local_program_calibration",
    "run_local_program_evaluation",
]
