"""Prospective same-prefix contrast collection for SAGE.T12.5b.3."""

from __future__ import annotations

import time
from collections import defaultdict
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
from .option_applicability_experiment import _structured_delta
from .progress import ProgressMilestone
from .progress_contrast import audit_prospective_progress_contrasts
from .progress_contrast_protocol import (
    ProgressContrastProtocol,
    load_progress_contrast_manifest,
    load_progress_contrast_receipt,
    progress_contrast_receipt,
)
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
CONTRAST_TRIALS_FORMAT = "sage-t12.5b.3-progress-contrast-trials-v1"
CONTRAST_REPORT_FORMAT = "sage-t12.5b.3-progress-contrast-report-v1"


@dataclass
class ContrastCollectionBudget:
    maximum_sdk_calls: int
    maximum_wall_seconds: int
    used_sdk_calls: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def consume(self, count: int = 1, *, reason: str = "replay") -> None:
        additional = max(0, int(count))
        if self.used_sdk_calls + additional > self.maximum_sdk_calls:
            raise RuntimeError(
                "T12.5b.3 SDK call budget exceeded: "
                f"used={self.used_sdk_calls} additional={additional} "
                f"maximum={self.maximum_sdk_calls} reason={reason}"
            )
        if self.elapsed_seconds > self.maximum_wall_seconds:
            raise RuntimeError(
                "T12.5b.3 wall-time budget exceeded before environment action"
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
class ProgressContrastTrial:
    trial_id: str
    lineage_seed: int
    stage: int
    context_id: str
    detour_action: str
    detour_depth: int
    action_name: str
    repetition: int
    original_prefix_exact: bool
    detour_available: bool
    detour_neutral: bool
    detour_terminal: bool
    detour_context_hash: str
    prefix_exact: bool
    branch_available: bool
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
            "context_id": self.context_id,
            "detour_action": self.detour_action,
            "detour_available": self.detour_available,
            "detour_context_hash": self.detour_context_hash,
            "detour_depth": self.detour_depth,
            "detour_neutral": self.detour_neutral,
            "detour_terminal": self.detour_terminal,
            "level_delta": self.level_delta,
            "lineage_seed": self.lineage_seed,
            "original_prefix_exact": self.original_prefix_exact,
            "prefix_exact": self.prefix_exact,
            "prefix_steps": [dict(item) for item in self.prefix_steps],
            "repetition": self.repetition,
            "sdk_calls_after": self.sdk_calls_after,
            "stage": self.stage,
            "terminal": self.terminal,
            "terminal_failure": self.terminal_failure,
            "trial_id": self.trial_id,
        }


def _execute_detour_branch(
    *,
    game_id: str,
    environments_dir: str | Path,
    env_factory: EnvFactory | None,
    witness: ProgressWitness,
    witness_prefix_length: int,
    option_actions: Sequence[GroundedAction],
    target_stage: int,
    detour_action: str,
    detour_depth: int,
    branch_action: str,
    repetition: int,
    expected_stage_hash: str,
    features: Sequence[str],
    milestones: Sequence[ProgressMilestone],
    budget: ContrastCollectionBudget,
) -> ProgressContrastTrial:
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
            phase=f"t12_5b_3_anchor_{witness.source_seed}",
            start_index=0,
            budget=budget,
        )

    prefix_steps: list[dict[str, Any]] = []
    prefix_available = bool(initial_exact and not divergence)
    if prefix_available:
        for position, action in enumerate(option_actions[:target_stage]):
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
            snapshot = snapshot_frame(frame)
            if _is_terminal(snapshot.game_state):
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
    before_state = _descriptor(frame)
    selected = (
        select_live_action(env, branch_action, action_args={})
        if preliminary_context_valid
        else None
    )
    branch_available = selected is not None
    if selected is None:
        candidate_step = projected_effect_step(
            {
                "action_name": branch_action,
                "available": False,
                "delta": {"mechanism": {}},
                "position": len(prefix_steps),
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
                "position": len(prefix_steps),
            },
            features=features,
        )
    level_delta = int(after_snapshot.levels_completed) - before_level
    terminal = _is_terminal(after_snapshot.game_state)
    context_id = f"stage_3_action4_depth_{int(detour_depth)}"
    return ProgressContrastTrial(
        trial_id=(
            f"lineage_{witness.source_seed}_{context_id}_"
            f"{branch_action.lower()}_rep_{repetition}"
        ),
        lineage_seed=int(witness.source_seed),
        stage=int(target_stage),
        context_id=context_id,
        detour_action=str(detour_action).upper(),
        detour_depth=int(detour_depth),
        action_name=str(branch_action).upper(),
        repetition=int(repetition),
        original_prefix_exact=original_prefix_exact,
        detour_available=detour_available,
        detour_neutral=detour_neutral,
        detour_terminal=detour_terminal,
        detour_context_hash=detour_context_hash,
        prefix_exact=preliminary_context_valid,
        branch_available=branch_available,
        prefix_steps=tuple(prefix_steps),
        candidate_step=candidate_step,
        level_delta=level_delta,
        terminal=terminal,
        terminal_failure=bool(
            detour_terminal or (terminal and level_delta <= 0)
        ),
        sdk_calls_after=budget.used_sdk_calls,
    )


def _annotate_context_exactness(
    trials: Sequence[ProgressContrastTrial],
) -> tuple[ProgressContrastTrial, ...]:
    groups: dict[tuple[int, str], list[ProgressContrastTrial]] = defaultdict(list)
    for trial in trials:
        groups[(trial.lineage_seed, trial.context_id)].append(trial)
    valid: dict[tuple[int, str], bool] = {}
    for key, records in groups.items():
        hashes = {item.detour_context_hash for item in records}
        prefixes = {
            json_key([dict(step) for step in item.prefix_steps]) for item in records
        }
        valid[key] = bool(
            len(hashes) == 1
            and len(prefixes) == 1
            and all(item.prefix_exact for item in records)
        )
    return tuple(
        replace(
            item,
            prefix_exact=bool(valid[(item.lineage_seed, item.context_id)]),
        )
        for item in trials
    )


def json_key(payload: Any) -> str:
    import json

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    )


def _collect_lineage(
    *,
    manifest: Mapping[str, Any],
    protocol: ProgressContrastProtocol,
    witness: ProgressWitness,
    option_actions: Sequence[GroundedAction],
    expected_stage_hash: str,
    milestones: Sequence[ProgressMilestone],
    environments_dir: str | Path,
    env_factory: EnvFactory | None,
    budget: ContrastCollectionBudget,
) -> tuple[ProgressContrastTrial, ...]:
    seed = int(witness.source_seed)
    prefix_length = int(manifest["inputs"]["successful_prefix_lengths"][str(seed)])
    trials = []
    for depth in protocol.detour_depths:
        for action in protocol.candidate_actions:
            for repetition in range(protocol.repetitions_per_branch):
                trials.append(
                    _execute_detour_branch(
                        game_id=str(manifest["game_id"]),
                        environments_dir=environments_dir,
                        env_factory=env_factory,
                        witness=witness,
                        witness_prefix_length=prefix_length,
                        option_actions=option_actions,
                        target_stage=protocol.target_stage,
                        detour_action=protocol.detour_action,
                        detour_depth=depth,
                        branch_action=action,
                        repetition=repetition,
                        expected_stage_hash=expected_stage_hash,
                        features=protocol.allowed_effect_features,
                        milestones=milestones,
                        budget=budget,
                    )
                )
    return _annotate_context_exactness(trials)


def run_progress_contrast_collection(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    environments_dir: str | Path = "environment_files",
    env_factory: EnvFactory | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_progress_contrast_manifest(
        manifest_path, root=repo_root, verify_code=env_factory is None
    )
    if not manifest["firewall"].get(
        "prospective_contrast_collection_authorized", False
    ):
        raise ValueError("T12.5b.3 manifest does not authorize collection")
    protocol = ProgressContrastProtocol(**dict(manifest["protocol"]))
    witnesses, option, posterior_payload, progress_programs, applicability = (
        _load_inputs(manifest, root=repo_root)
    )
    witnesses_by_seed = {int(item.source_seed): item for item in witnesses}
    if set(witnesses_by_seed) != set(protocol.lineage_seeds):
        raise ValueError("T12.5b.3 witness lineage set changed")
    option_actions = _materialize_option(option)
    expected_actions = tuple(item.action_name for item in option_actions)
    if expected_actions != ("ACTION4", "ACTION4", "ACTION4", "ACTION3", "ACTION3"):
        raise ValueError("T12.5b.3 parent option changed")
    milestones = _ordered_milestones(progress_programs)
    if len(milestones) != 5:
        raise ValueError("T12.5b.3 milestone count changed")
    posterior = posterior_from_snapshot(posterior_payload)
    expected_hashes = _expected_stage_hashes(
        applicability, lineage_seeds=protocol.lineage_seeds
    )
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable run: {destination}")
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes_per_run)
    budget = ContrastCollectionBudget(
        maximum_sdk_calls=protocol.maximum_sdk_calls,
        maximum_wall_seconds=protocol.maximum_wall_seconds,
    )
    trials = tuple(
        item
        for seed in protocol.lineage_seeds
        for item in _collect_lineage(
            manifest=manifest,
            protocol=protocol,
            witness=witnesses_by_seed[seed],
            option_actions=option_actions,
            expected_stage_hash=expected_hashes[(seed, protocol.target_stage)],
            milestones=milestones,
            environments_dir=environments_dir,
            env_factory=env_factory,
            budget=budget,
        )
    )
    audit = audit_prospective_progress_contrasts(
        trials=[item.to_dict() for item in trials],
        features=protocol.allowed_effect_features,
        posterior=posterior,
        milestones=milestones,
        lineage_seeds=protocol.lineage_seeds,
        target_stage=protocol.target_stage,
        context_ids=protocol.context_ids,
        candidate_actions=protocol.candidate_actions,
        repetitions_per_branch=protocol.repetitions_per_branch,
        minimum_distractor_magnitude_gap=(
            protocol.minimum_distractor_magnitude_gap
        ),
    )
    metrics = dict(audit["metrics"])
    budget_snapshot = budget.snapshot()
    metrics["sdk_calls"] = budget_snapshot
    metrics["parent_negative_result_preserved"] = True
    valid_per_lineage = {
        int(seed): int(metrics["valid_contexts_per_lineage"].get(str(seed), 0))
        for seed in protocol.lineage_seeds
    }
    integrity_checks = {
        "all_original_stage_prefixes_exact": (
            metrics["original_prefix_exact_rate"] == 1.0
        ),
        "branch_availability_is_deterministic": metrics[
            "availability_is_deterministic"
        ],
        "candidate_effects_are_deterministic": metrics[
            "effect_is_deterministic_when_executable"
        ],
        "fixed_candidate_schedule_completed": (
            metrics["trial_count"] == protocol.expected_trial_count
        ),
        "no_terminal_failures": (
            metrics["terminal_failures"] <= protocol.maximum_terminal_failures
        ),
        "parent_negative_result_preserved": True,
        "posterior_never_controlled_collection": manifest["design"][
            "posterior_never_selects_executed_actions"
        ],
        "repetition_count_exact": metrics["repetition_count_is_exact"],
        "sdk_budget_respected": budget_snapshot["within_sdk_budget"],
        "wall_time_respected": budget_snapshot["within_wall_time"],
    }
    context_checks = {
        "common_valid_detour_context_exists": (
            metrics["common_valid_context_count"]
            >= protocol.minimum_common_valid_contexts
        ),
        "minimum_local_candidates_available": (
            metrics["minimum_executable_actions_per_valid_context"]
            >= protocol.minimum_executable_actions_per_context
        ),
        "valid_detour_context_in_every_lineage": all(
            count >= protocol.minimum_valid_contexts_per_lineage
            for count in valid_per_lineage.values()
        ),
    }
    progress_checks = {
        "progress_affordance_binds_across_lineages": (
            metrics["affordance_binding_count"] >= 1
        ),
        "progress_affordance_observed_in_both_lineages": (
            metrics["progress_affordance_lineage_count"]
            == len(protocol.lineage_seeds)
        ),
    }
    hard_per_lineage = {
        int(seed): int(metrics["hard_contrasts_per_lineage"].get(int(seed), 0))
        for seed in protocol.lineage_seeds
    }
    contrast_availability_checks = {
        "common_hard_contrast_context_exists": (
            metrics["common_hard_contrast_context_count"]
            >= protocol.minimum_common_hard_contrast_contexts
        ),
        "hard_contrast_exists_in_every_lineage": all(
            count >= protocol.minimum_hard_contrasts_per_lineage
            for count in hard_per_lineage.values()
        ),
    }
    discrimination_checks = {
        "causal_ranking_beats_magnitude_on_hard_contrasts": (
            metrics["hard_contrast_count"] > 0
            and metrics["hard_contrast_accuracy_gain"]
            >= protocol.minimum_hard_contrast_accuracy_gain
        ),
        "causal_ranking_is_perfect_on_hard_contrasts": (
            metrics["hard_contrast_count"] > 0
            and metrics["causal_hard_contrast_accuracy"]
            >= protocol.minimum_causal_hard_contrast_accuracy
        ),
    }
    checks = {
        **integrity_checks,
        **context_checks,
        **progress_checks,
        **contrast_availability_checks,
        **discrimination_checks,
    }
    integrity_passed = all(integrity_checks.values())
    context_passed = all(context_checks.values())
    progress_passed = all(progress_checks.values())
    contrast_availability_passed = all(contrast_availability_checks.values())
    discrimination_passed = all(discrimination_checks.values())
    passed = bool(
        integrity_passed
        and context_passed
        and progress_passed
        and contrast_availability_passed
        and discrimination_passed
    )
    if not integrity_passed:
        classification = "COLLECTION_INTEGRITY_FAILURE"
        status = "FAIL_T12_5B_3_COLLECTION_INTEGRITY_GATE"
    elif not context_passed:
        classification = "NO_REPRODUCIBLE_NEUTRAL_DETOUR_CONTEXT"
        status = "FAIL_T12_5B_3_DETOUR_CONTEXT_GATE"
    elif not progress_passed:
        classification = "NO_TRANSPORTED_PROGRESS_AFFORDANCE"
        status = "FAIL_T12_5B_3_PROGRESS_AFFORDANCE_GATE"
    elif not contrast_availability_passed:
        classification = "INSUFFICIENT_PROSPECTIVE_DISCRIMINATIVE_CONTRASTS"
        status = "FAIL_T12_5B_3_INSUFFICIENT_DISCRIMINATIVE_CONTRASTS"
    elif not discrimination_passed:
        classification = "CAUSAL_PROGRESS_NOT_PROSPECTIVELY_DISCRIMINATIVE"
        status = "FAIL_T12_5B_3_DISCRIMINATION_GATE"
    else:
        classification = "CAUSAL_PROGRESS_PROSPECTIVELY_DISCRIMINATES"
        status = "PASS_T12_5B_3_PROSPECTIVE_CONTRAST_GATE"
    metrics.update(
        {
            "checks": checks,
            "classification": classification,
            "t12_5c_control_freeze_authorized": passed,
        }
    )

    trials_path = destination / "contrast_trials.json"
    affordance_path = destination / "affordance_registry.json"
    contrast_path = destination / "hard_contrast_registry.json"
    report_path = destination / "contrast_report.json"
    receipt_path = destination / "contrast_receipt.json"
    _write_json_once(
        trials_path,
        {
            "format_version": CONTRAST_TRIALS_FORMAT,
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
            "trials": [item.to_dict() for item in trials],
        },
        storage_budget=storage,
    )
    _write_json_once(
        affordance_path,
        {
            **dict(audit["affordance_registry"]),
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
        },
        storage_budget=storage,
    )
    _write_json_once(
        contrast_path,
        {
            **dict(audit["contrast_registry"]),
            "manifest_checksum": manifest["manifest_checksum"],
            "protocol_checksum": manifest["protocol_checksum"],
        },
        storage_budget=storage,
    )
    metrics["storage"] = storage.snapshot()
    report = {
        "claim_boundary": manifest["claim_boundary"],
        "format_version": CONTRAST_REPORT_FORMAT,
        "manifest_checksum": manifest["manifest_checksum"],
        "metrics": metrics,
        "passed": passed,
        "protocol_checksum": manifest["protocol_checksum"],
        "status": status,
    }
    _write_json_once(report_path, report, storage_budget=storage)
    receipt = progress_contrast_receipt(
        manifest=manifest,
        phase="prospective_contrast_collection",
        passed=passed,
        status=status,
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
            "trials": {
                "path": str(trials_path.resolve()),
                "sha256": _file_sha256(trials_path),
            },
        },
    )
    _write_json_once(receipt_path, receipt, storage_budget=storage)
    return receipt


def progress_contrast_status(
    *,
    manifest_path: str | Path,
    receipt_path: str | Path | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(root).resolve() if root else Path(__file__).resolve().parents[3]
    manifest = load_progress_contrast_manifest(manifest_path, root=repo_root)
    receipt = (
        None
        if receipt_path is None or not Path(receipt_path).is_file()
        else load_progress_contrast_receipt(
            receipt_path, manifest=manifest, root=repo_root
        )
    )
    passed = bool(
        receipt
        and receipt.get("passed") is True
        and receipt.get("status")
        == "PASS_T12_5B_3_PROSPECTIVE_CONTRAST_GATE"
    )
    collection_ready = bool(
        receipt is None
        and manifest["firewall"].get(
            "prospective_contrast_collection_authorized", False
        )
    )
    return {
        "claim_boundary": manifest["claim_boundary"],
        "firewall": {
            "prospective_contrast_collection_authorized": collection_ready,
            "environment_collection_authorized": collection_ready,
            "causal_progress_control_authorized": False,
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
        "format_version": "sage-t12.5b.3-progress-contrast-status-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "next_phase_authorized": passed,
        "parent_t12_5b_2_status": manifest["parent"]["receipt"]["status"],
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
    "ContrastCollectionBudget",
    "ProgressContrastTrial",
    "progress_contrast_status",
    "run_progress_contrast_collection",
]
