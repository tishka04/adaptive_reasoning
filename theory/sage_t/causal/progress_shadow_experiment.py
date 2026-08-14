"""Exact-prefix observed-effect shadow experiment for SAGE.T12.5b."""

from __future__ import annotations

from collections import defaultdict
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
from .experiment import RunStorageBudget, _file_sha256, _read_json, _write_json_once
from .graph_experiment import _is_terminal, _make_env, _symbolic_state
from .option_applicability_experiment import _state_descriptor, _structured_delta
from .option_minimization_experiment import _load_contextual_option
from .options import MinimalCausalOption
from .progress import CausalProgressProgram, ProgressMilestone
from .progress_shadow import (
    EmpiricalActionEffectModel,
    build_shadow_ranking,
    posterior_from_snapshot,
    progress_signature,
    projected_effect_step,
    projection_vector,
    reciprocal_rank,
)
from .progress_shadow_protocol import (
    ProgressShadowProtocol,
    _checksum,
    _resolve_bound,
    load_progress_shadow_manifest,
    load_progress_shadow_receipt,
    progress_shadow_receipt,
)
from .witness_experiment import _execute_expected_steps, _reset_env
from .witness_protocol import ProgressWitness
from .witness_reconfirmation_protocol import load_reconfirmation_registry

EnvFactory = Callable[[str], Any]
SHADOW_TRIALS_FORMAT = "sage-t12.5b-progress-shadow-trials-v2"
SHADOW_REPORT_FORMAT = "sage-t12.5b-progress-shadow-report-v2"
SHADOW_RANKINGS_FORMAT = "sage-t12.5b-progress-shadow-rankings-v2"


@dataclass
class ShadowSdkBudget:
    maximum: int
    used: int = 0

    def consume(self, count: int = 1, *, reason: str = "replay") -> None:
        additional = max(0, int(count))
        if self.used + additional > self.maximum:
            raise RuntimeError(
                "T12.5b SDK call budget exceeded: "
                f"used={self.used} additional={additional} "
                f"maximum={self.maximum} reason={reason}"
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
class ProgressShadowTrial:
    trial_id: str
    lineage_seed: int
    stage: int
    action_name: str
    repetition: int
    prefix_exact: bool
    branch_available: bool
    expected_stage_hash: str
    observed_stage_hash: str
    prefix_steps: tuple[Mapping[str, Any], ...]
    candidate_step: Mapping[str, Any]
    level_delta: int
    terminal: bool
    terminal_failure: bool
    sdk_calls_after: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_name": self.action_name,
            "branch_available": self.branch_available,
            "candidate_step": dict(self.candidate_step),
            "expected_stage_hash": self.expected_stage_hash,
            "level_delta": self.level_delta,
            "lineage_seed": self.lineage_seed,
            "observed_stage_hash": self.observed_stage_hash,
            "prefix_exact": self.prefix_exact,
            "prefix_steps": [dict(item) for item in self.prefix_steps],
            "repetition": self.repetition,
            "sdk_calls_after": self.sdk_calls_after,
            "stage": self.stage,
            "terminal": self.terminal,
            "terminal_failure": self.terminal_failure,
            "trial_id": self.trial_id,
        }


def _load_inputs(
    manifest: Mapping[str, Any], *, root: Path
) -> tuple[
    tuple[ProgressWitness, ...],
    MinimalCausalOption,
    Mapping[str, Any],
    tuple[CausalProgressProgram, ...],
    Mapping[str, Any],
]:
    witness_path = _resolve_bound(
        str(manifest["inputs"]["witness_registry"]["path"]), root=root
    )
    _, witnesses = load_reconfirmation_registry(witness_path)
    option_payload = _load_contextual_option(
        _resolve_bound(str(manifest["inputs"]["minimal_option"]["path"]), root=root)
    )
    option = MinimalCausalOption.from_dict(dict(option_payload["option"]))
    posterior = _read_json(
        _resolve_bound(str(manifest["inputs"]["posterior"]["path"]), root=root)
    )
    registry = _read_json(
        _resolve_bound(
            str(manifest["inputs"]["program_registry"]["path"]), root=root
        )
    )
    progress_programs = tuple(
        CausalProgressProgram.from_dict(dict(item))
        for item in registry.get("programs", ())
    )
    applicability = _read_json(
        _resolve_bound(
            str(manifest["inputs"]["applicability_trials"]["path"]), root=root
        )
    )
    return witnesses, option, posterior, progress_programs, applicability


def _ordered_milestones(
    programs: Sequence[CausalProgressProgram],
) -> tuple[ProgressMilestone, ...]:
    candidates = [
        item.milestones for item in programs if item.progress_kind == "ordered_effects"
    ]
    if not candidates:
        raise ValueError("T12.5b has no ordered progress program")
    semantic = [tuple(item.semantic_payload for item in group) for group in candidates]
    if any(value != semantic[0] for value in semantic[1:]):
        raise ValueError("T12.5b ordered owners disagree on milestone semantics")
    return tuple(candidates[0])


def _expected_stage_hashes(
    applicability: Mapping[str, Any], *, lineage_seeds: Sequence[int]
) -> dict[tuple[int, int], str]:
    output: dict[tuple[int, int], str] = {}
    for seed in lineage_seeds:
        candidates = [
            dict(item)
            for item in applicability.get("trials", ())
            if int(item.get("lineage_seed", -1)) == int(seed)
            and item.get("context_name") == "successful_level0"
            and item.get("branch_name") == "option_full"
            and item.get("prefix_exact")
            and item.get("branch_available")
            and item.get("progressed")
        ]
        if not candidates:
            raise ValueError(f"T12.5b has no successful stage evidence for {seed}")
        hashes_by_stage: dict[int, set[str]] = defaultdict(set)
        for item in candidates:
            for step in item.get("trace", ()):
                hashes_by_stage[int(step["position"])].add(
                    str(step["source_state"]["exact_hash"])
                )
        if set(hashes_by_stage) != {0, 1, 2, 3, 4} or any(
            len(values) != 1 for values in hashes_by_stage.values()
        ):
            raise ValueError(f"T12.5b stage hashes are not deterministic for {seed}")
        for stage, values in hashes_by_stage.items():
            output[(int(seed), stage)] = next(iter(values))
    return output


def _materialize_option(option: MinimalCausalOption) -> tuple[GroundedAction, ...]:
    from theory.sage_t.contracts import AbstractState

    empty = AbstractState()
    return tuple(step.materialize(empty) for step in option.steps)


def _descriptor(frame: Any) -> dict[str, Any]:
    snapshot = snapshot_frame(frame)
    return _state_descriptor(
        _symbolic_state(frame),
        exact_hash=state_signature_from_frame(frame),
        level=int(snapshot.levels_completed),
        game_state=str(snapshot.game_state),
    )


def _execute_stage_branch(
    *,
    game_id: str,
    environments_dir: str | Path,
    env_factory: EnvFactory | None,
    witness: ProgressWitness,
    witness_prefix_length: int,
    option_actions: Sequence[GroundedAction],
    stage: int,
    branch_action: str,
    repetition: int,
    expected_stage_hash: str,
    features: Sequence[str],
    budget: ShadowSdkBudget,
) -> ProgressShadowTrial:
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
            phase=f"t12_5b_anchor_{witness.source_seed}",
            start_index=0,
            budget=budget,
        )

    prefix_steps: list[dict[str, Any]] = []
    prefix_available = bool(initial_exact and not divergence)
    if prefix_available:
        for position, action in enumerate(option_actions[:stage]):
            before = _descriptor(frame)
            selected = select_live_action(
                env, action.action_name, action_args=action.action_data
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
    observed_stage_hash = state_signature_from_frame(frame)
    prefix_exact = bool(
        prefix_available
        and observed_stage_hash == expected_stage_hash
        and len(prefix_steps) == int(stage)
    )
    before_snapshot = snapshot_frame(frame)
    before_level = int(before_snapshot.levels_completed)
    before_state = _descriptor(frame)
    selected = (
        select_live_action(env, branch_action, action_args={}) if prefix_exact else None
    )
    branch_available = selected is not None
    if selected is None:
        candidate_step = projected_effect_step(
            {
                "action_name": branch_action,
                "available": False,
                "delta": {"mechanism": {}},
                "position": stage,
            },
            features=features,
        )
        after_snapshot = before_snapshot
    else:
        budget.consume(reason="candidate_branch")
        frame = _step_env_action(env, selected)
        after_snapshot = snapshot_frame(frame)
        after_state = _descriptor(frame)
        candidate_step = projected_effect_step(
            {
                "action_name": branch_action,
                "available": True,
                "delta": _structured_delta(before_state, after_state),
                "position": stage,
            },
            features=features,
        )
    level_delta = int(after_snapshot.levels_completed) - before_level
    terminal = _is_terminal(after_snapshot.game_state)
    return ProgressShadowTrial(
        trial_id=(
            f"lineage_{witness.source_seed}_stage_{stage}_"
            f"{branch_action.lower()}_rep_{repetition}"
        ),
        lineage_seed=int(witness.source_seed),
        stage=int(stage),
        action_name=str(branch_action).upper(),
        repetition=int(repetition),
        prefix_exact=prefix_exact,
        branch_available=branch_available,
        expected_stage_hash=expected_stage_hash,
        observed_stage_hash=observed_stage_hash,
        prefix_steps=tuple(prefix_steps),
        candidate_step=candidate_step,
        level_delta=level_delta,
        terminal=terminal,
        terminal_failure=bool(terminal and level_delta <= 0),
        sdk_calls_after=budget.used,
    )


def _collect_lineage(
    *,
    manifest: Mapping[str, Any],
    protocol: ProgressShadowProtocol,
    witness: ProgressWitness,
    option_actions: Sequence[GroundedAction],
    expected_hashes: Mapping[tuple[int, int], str],
    environments_dir: str | Path,
    env_factory: EnvFactory | None,
    budget: ShadowSdkBudget,
) -> tuple[ProgressShadowTrial, ...]:
    trials = []
    seed = int(witness.source_seed)
    prefix_length = int(manifest["inputs"]["successful_prefix_lengths"][str(seed)])
    for stage in protocol.stages:
        for action in protocol.candidate_actions:
            for repetition in range(protocol.repetitions_per_branch):
                trials.append(
                    _execute_stage_branch(
                        game_id=str(manifest["game_id"]),
                        environments_dir=environments_dir,
                        env_factory=env_factory,
                        witness=witness,
                        witness_prefix_length=prefix_length,
                        option_actions=option_actions,
                        stage=stage,
                        branch_action=action,
                        repetition=repetition,
                        expected_stage_hash=expected_hashes[(seed, stage)],
                        features=protocol.allowed_effect_features,
                        budget=budget,
                    )
                )
    return tuple(trials)


def _determinism_rate(
    trials: Sequence[ProgressShadowTrial], *, features: Sequence[str]
) -> float:
    groups: dict[tuple[int, int, str], set[tuple[int, ...]]] = defaultdict(set)
    for trial in trials:
        groups[(trial.lineage_seed, trial.stage, trial.action_name)].add(
            projection_vector(trial.candidate_step, features=features)
        )
    return sum(len(values) == 1 for values in groups.values()) / max(1, len(groups))


def _confirmation_effect_metrics(
    *,
    model: EmpiricalActionEffectModel,
    trials: Sequence[ProgressShadowTrial],
    milestones: Sequence[ProgressMilestone],
) -> dict[str, float]:
    vector_matches = []
    signature_matches = []
    for trial in trials:
        expected = model.prediction(trial.stage, trial.action_name)
        observed_vector = projection_vector(
            trial.candidate_step, features=model.features
        )
        predicted_step = model.predicted_step(trial.stage, trial.action_name)
        vector_matches.append(observed_vector == expected.projection)
        signature_matches.append(
            progress_signature(trial.candidate_step, milestones)
            == progress_signature(predicted_step, milestones)
        )
    return {
        "exact_projection_transport_rate": sum(vector_matches)
        / max(1, len(vector_matches)),
        "milestone_signature_transport_rate": sum(signature_matches)
        / max(1, len(signature_matches)),
    }


def _ranking_metrics(
    ranking_plan: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    methods = (
        "causal_progress",
        "change_only",
        "magnitude_only",
        "lexicographic",
        "action_only",
    )
    per_method: dict[str, dict[str, float]] = {}
    for method in methods:
        reciprocal = [
            reciprocal_rank(item["rankings"][method], item["expected_action"])
            for _, item in sorted(ranking_plan.items())
        ]
        top1 = [
            bool(item["rankings"][method][0] == item["expected_action"])
            for _, item in sorted(ranking_plan.items())
        ]
        per_method[method] = {
            "mean_reciprocal_rank": sum(reciprocal) / len(reciprocal),
            "top1_accuracy": sum(top1) / len(top1),
        }
    return {
        "minimum_causal_margin": min(
            float(item["causal_margin"]) for item in ranking_plan.values()
        ),
        "per_method": per_method,
    }


def _observed_ranking_metrics(
    *,
    posterior: Any,
    trials: Sequence[ProgressShadowTrial],
    protocol: ProgressShadowProtocol,
) -> dict[str, Any]:
    rankings = {}
    progress_gains = []
    for stage in protocol.stages:
        stage_trials = [trial for trial in trials if trial.stage == stage]
        representative = {
            action: next(
                trial
                for trial in stage_trials
                if trial.action_name == action and trial.repetition == 0
            )
            for action in protocol.candidate_actions
        }
        prefix_checksums = {
            _checksum([dict(item) for item in trial.prefix_steps])
            for trial in representative.values()
        }
        if len(prefix_checksums) != 1:
            raise ValueError("T12.5b confirmation branches do not share a prefix")
        prefix = next(iter(representative.values())).prefix_steps
        scores = {
            action: posterior.expected_potential(
                (*prefix, trial.candidate_step)
            )
            for action, trial in representative.items()
        }
        ranking = tuple(
            name for name, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        )
        expected = protocol.expected_actions[stage]
        progress_gains.append(
            scores[expected] - posterior.expected_potential(prefix)
        )
        rankings[stage] = {
            "expected_action": expected,
            "ranking": list(ranking),
            "scores": scores,
        }
    reciprocal = [
        reciprocal_rank(item["ranking"], item["expected_action"])
        for item in rankings.values()
    ]
    return {
        "expected_action_progress_gains": progress_gains,
        "mean_reciprocal_rank": sum(reciprocal) / len(reciprocal),
        "rankings": rankings,
        "top1_accuracy": sum(
            item["ranking"][0] == item["expected_action"]
            for item in rankings.values()
        )
        / len(rankings),
    }


def run_progress_shadow(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    environments_dir: str | Path = "environment_files",
    env_factory: EnvFactory | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_progress_shadow_manifest(
        manifest_path, root=repo_root, verify_code=env_factory is None
    )
    if not manifest["firewall"].get(
        "causal_progress_shadow_collection_authorized", False
    ):
        raise ValueError("T12.5b manifest does not authorize shadow collection")
    protocol = ProgressShadowProtocol(**dict(manifest["protocol"]))
    witnesses, option, posterior_payload, progress_programs, applicability = (
        _load_inputs(manifest, root=repo_root)
    )
    witnesses_by_seed = {int(item.source_seed): item for item in witnesses}
    if set(witnesses_by_seed) != set(protocol.lineage_seeds):
        raise ValueError("T12.5b witness lineage set changed")
    option_actions = _materialize_option(option)
    if tuple(item.action_name for item in option_actions) != protocol.expected_actions:
        raise ValueError("T12.5b option actions changed")
    milestones = _ordered_milestones(progress_programs)
    if len(milestones) != len(protocol.stages):
        raise ValueError("T12.5b milestone count changed")
    posterior = posterior_from_snapshot(posterior_payload)
    expected_hashes = _expected_stage_hashes(
        applicability, lineage_seeds=protocol.lineage_seeds
    )
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable run: {destination}")
    storage = RunStorageBudget(
        destination, protocol.maximum_artifact_bytes_per_run
    )
    budget = ShadowSdkBudget(protocol.maximum_sdk_calls)

    induction = _collect_lineage(
        manifest=manifest,
        protocol=protocol,
        witness=witnesses_by_seed[protocol.induction_lineage_seed],
        option_actions=option_actions,
        expected_hashes=expected_hashes,
        environments_dir=environments_dir,
        env_factory=env_factory,
        budget=budget,
    )
    model = EmpiricalActionEffectModel.fit(
        [item.to_dict() for item in induction],
        features=protocol.allowed_effect_features,
        induction_lineage_seed=protocol.induction_lineage_seed,
        expected_stages=protocol.stages,
        candidate_actions=protocol.candidate_actions,
    )
    ranking_plan = {
        stage: build_shadow_ranking(
            posterior=posterior,
            model=model,
            stage=stage,
            candidate_actions=protocol.candidate_actions,
            expected_actions=protocol.expected_actions,
        )
        for stage in protocol.stages
    }
    ranking_plan_checksum_before_confirmation = _checksum(ranking_plan)

    confirmation = _collect_lineage(
        manifest=manifest,
        protocol=protocol,
        witness=witnesses_by_seed[protocol.confirmation_lineage_seed],
        option_actions=option_actions,
        expected_hashes=expected_hashes,
        environments_dir=environments_dir,
        env_factory=env_factory,
        budget=budget,
    )
    trials = (*induction, *confirmation)
    ranking_metrics = _ranking_metrics(ranking_plan)
    transport = _confirmation_effect_metrics(
        model=model, trials=confirmation, milestones=milestones
    )
    observed = _observed_ranking_metrics(
        posterior=posterior, trials=confirmation, protocol=protocol
    )
    exact_prefix_rate = sum(item.prefix_exact for item in trials) / len(trials)
    availability_rate = sum(item.branch_available for item in trials) / len(trials)
    determinism_rate = _determinism_rate(
        trials, features=protocol.allowed_effect_features
    )
    terminal_failures = sum(item.terminal_failure for item in trials)
    causal_metrics = ranking_metrics["per_method"]["causal_progress"]
    non_action_baseline_mrr = max(
        ranking_metrics["per_method"][name]["mean_reciprocal_rank"]
        for name in ("change_only", "magnitude_only", "lexicographic")
    )
    checks = {
        "all_stage_prefixes_exact": exact_prefix_rate
        >= protocol.minimum_exact_prefix_rate,
        "all_candidate_actions_available": availability_rate
        >= protocol.minimum_branch_availability_rate,
        "all_effects_deterministic_within_lineage": determinism_rate
        >= protocol.minimum_effect_determinism_rate,
        "confirmation_milestone_signature_transports": transport[
            "milestone_signature_transport_rate"
        ]
        == 1.0,
        "causal_shadow_top1_is_perfect": causal_metrics["top1_accuracy"]
        >= protocol.minimum_causal_top1_accuracy,
        "causal_shadow_mrr_is_perfect": causal_metrics["mean_reciprocal_rank"]
        >= protocol.minimum_causal_mean_reciprocal_rank,
        "causal_margins_are_positive": ranking_metrics["minimum_causal_margin"]
        >= protocol.minimum_positive_margin,
        "causal_ranking_beats_non_goal_baselines": (
            causal_metrics["mean_reciprocal_rank"] - non_action_baseline_mrr
            >= protocol.minimum_baseline_mrr_gain
        ),
        "observed_confirmation_ranking_is_perfect": observed["top1_accuracy"]
        == 1.0
        and observed["mean_reciprocal_rank"] == 1.0,
        "expected_observed_actions_increase_progress": all(
            value > 0.0 for value in observed["expected_action_progress_gains"]
        ),
        "ranking_plan_frozen_before_confirmation": _checksum(ranking_plan)
        == ranking_plan_checksum_before_confirmation,
        "ranking_never_controlled_collection": manifest["design"][
            "rankings_never_select_executed_actions"
        ]
        is True,
        "trial_count_exact": len(trials) == protocol.expected_trial_count,
        "sdk_budget_respected": budget.used <= protocol.maximum_sdk_calls,
        "no_terminal_failures": terminal_failures
        <= protocol.maximum_terminal_failures,
    }
    passed = all(checks.values())
    trials_path = destination / "shadow_trials.json"
    model_path = destination / "effect_model.json"
    rankings_path = destination / "shadow_rankings.json"
    report_path = destination / "shadow_report.json"
    receipt_path = destination / "shadow_receipt.json"
    _write_json_once(
        trials_path,
        {
            "format_version": SHADOW_TRIALS_FORMAT,
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "trials": [item.to_dict() for item in trials],
        },
        storage_budget=storage,
    )
    _write_json_once(model_path, model.to_dict(), storage_budget=storage)
    _write_json_once(
        rankings_path,
        {
            "format_version": SHADOW_RANKINGS_FORMAT,
            "manifest_checksum": manifest["manifest_checksum"],
            "model_checksum": model.model_checksum,
            "plan_checksum": ranking_plan_checksum_before_confirmation,
            "protocol_checksum": manifest["protocol_checksum"],
            "rankings": {str(key): value for key, value in ranking_plan.items()},
        },
        storage_budget=storage,
    )
    metrics = {
        "branch_availability_rate": availability_rate,
        "checks": checks,
        "effect_determinism_rate": determinism_rate,
        "effect_transport": transport,
        "exact_prefix_rate": exact_prefix_rate,
        "observed_confirmation": observed,
        "ranking": ranking_metrics,
        "ranking_plan_checksum": ranking_plan_checksum_before_confirmation,
        "sdk_calls": budget.snapshot(),
        "terminal_failures": terminal_failures,
        "trial_count": len(trials),
    }
    status = (
        "PASS_T12_5B_PROGRESS_SHADOW_GATE"
        if passed
        else "FAIL_T12_5B_PROGRESS_SHADOW_GATE"
    )
    report = {
        "claim_boundary": manifest["claim_boundary"],
        "format_version": SHADOW_REPORT_FORMAT,
        "manifest_checksum": manifest["manifest_checksum"],
        "metrics": metrics,
        "passed": passed,
        "protocol_checksum": manifest["protocol_checksum"],
        "status": status,
    }
    _write_json_once(report_path, report, storage_budget=storage)
    metrics["storage"] = storage.snapshot()
    checks["storage_within_budget"] = metrics["storage"]["within_budget"]
    receipt = progress_shadow_receipt(
        manifest=manifest,
        phase="shadow_ranking",
        passed=passed and checks["storage_within_budget"],
        status=status if checks["storage_within_budget"] else "FAIL_T12_5B_STORAGE_GATE",
        metrics=metrics,
        artifacts={
            "effect_model": {
                "path": str(model_path.resolve()),
                "sha256": _file_sha256(model_path),
            },
            "rankings": {
                "path": str(rankings_path.resolve()),
                "sha256": _file_sha256(rankings_path),
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


def progress_shadow_status(
    *,
    manifest_path: str | Path,
    receipt_path: str | Path | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_progress_shadow_manifest(manifest_path, root=repo_root)
    receipt = (
        None
        if receipt_path is None or not Path(receipt_path).is_file()
        else load_progress_shadow_receipt(
            receipt_path, manifest=manifest, root=repo_root
        )
    )
    passed = bool(
        receipt
        and receipt.get("passed") is True
        and receipt.get("status") == "PASS_T12_5B_PROGRESS_SHADOW_GATE"
    )
    return {
        "claim_boundary": manifest["claim_boundary"],
        "firewall": {
            "causal_progress_control_authorized": False,
            "causal_progress_shadow_collection_authorized": True,
            "holdout_opened": False,
            "neural_active_evaluation_authorized": False,
            "neural_training_authorized": False,
            "option_control_authorized": False,
            "production_authority": False,
            "source_validation_opened": False,
            "t12_5c_control_freeze_authorized": passed,
            "t12_6_freeze_authorized": False,
            "terminal_shield_production_authority": False,
        },
        "format_version": "sage-t12.5b-progress-shadow-status-v2",
        "manifest_checksum": manifest["manifest_checksum"],
        "next_phase_authorized": passed,
        "parent_t12_5_status": manifest["parent"]["receipt"]["status"],
        "protocol_checksum": manifest["protocol_checksum"],
        "receipt": (
            None
            if receipt is None
            else {
                "passed": receipt["passed"],
                "phase": receipt["phase"],
                "receipt_checksum": receipt["receipt_checksum"],
                "status": receipt["status"],
            }
        ),
    }


__all__ = ["progress_shadow_status", "run_progress_shadow"]
