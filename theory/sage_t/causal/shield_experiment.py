"""Bounded T12.3b terminal confirmation and paired shield evaluation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from theory.m1.polymorphic_a25_adapter import _step_env_action
from theory.m2.m3_execution_smoke import _reset_env
from theory.real_env_option_adapter import snapshot_frame
from theory.sage.live_prefix_counterfactual_collector import (
    select_live_action,
    state_signature_from_frame,
)
from theory.unified_cognition_ab_benchmark import _is_terminal

from .archive import GoExploreArchive
from .burst_experiment import BurstExcursion, BurstRun
from .experiment import RunStorageBudget, _file_sha256, _write_json_once
from .graph_experiment import (
    _grounded_actions,
    _make_env,
    _record_root,
    _restore_variant,
    _symbolic_state,
    _write_archive,
)
from .shield_model import ProgressProtectedTerminalShield
from .shield_protocol import (
    ProtectedActionSpec,
    TerminalShieldProtocol,
    TerminalTraceCandidate,
    load_shield_manifest,
    load_shield_receipt,
    load_shield_registry,
    shield_phase_receipt,
)
from .terminal_shield import MultiStepTerminalShield
from .witness_protocol import (
    ProgressWitness,
    load_witness_manifest,
    load_witness_registry,
)

EnvFactory = Callable[[str], Any]


@dataclass(frozen=True)
class TerminalConfirmation:
    candidate_id: str
    repetition: int
    exact: bool
    terminal_failure: bool
    confirmed: bool
    calls: int
    final_exact_hash: str
    first_divergence: str
    events: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "repetition": self.repetition,
            "exact": self.exact,
            "terminal_failure": self.terminal_failure,
            "confirmed": self.confirmed,
            "calls": self.calls,
            "final_exact_hash": self.final_exact_hash,
            "first_divergence": self.first_divergence,
            "events": [dict(item) for item in self.events],
        }


@dataclass(frozen=True)
class WitnessShieldTrial:
    witness_id: str
    repetition: int
    exact: bool
    progressed: bool
    all_actions_protected: bool
    vetoed_actions: int
    calls: int
    final_exact_hash: str
    first_divergence: str
    events: tuple[Mapping[str, Any], ...]

    @property
    def confirmed(self) -> bool:
        return bool(
            self.exact
            and self.progressed
            and self.all_actions_protected
            and self.vetoed_actions == 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "witness_id": self.witness_id,
            "repetition": self.repetition,
            "exact": self.exact,
            "progressed": self.progressed,
            "all_actions_protected": self.all_actions_protected,
            "vetoed_actions": self.vetoed_actions,
            "confirmed": self.confirmed,
            "calls": self.calls,
            "final_exact_hash": self.final_exact_hash,
            "first_divergence": self.first_divergence,
            "events": [dict(item) for item in self.events],
        }


def confirm_terminal_candidate(
    *,
    candidate: TerminalTraceCandidate,
    repetition: int,
    environments_dir: str | Path,
    env_factory: EnvFactory | None = None,
) -> TerminalConfirmation:
    env = _make_env(candidate.game_id, environments_dir, env_factory)
    frame = _reset_env(env)
    calls = 1
    initial_hash = state_signature_from_frame(frame)
    initial_exact = initial_hash == candidate.initial_exact_hash
    events: list[dict[str, Any]] = [
        {
            "kind": "reset",
            "expected_hash": candidate.initial_exact_hash,
            "observed_hash": initial_hash,
            "exact": initial_exact,
        }
    ]
    divergence = "" if initial_exact else "initial:reset"
    if initial_exact:
        for index, edge in enumerate(candidate.edges):
            observed_source = state_signature_from_frame(frame)
            if observed_source != edge.source_exact_hash:
                divergence = f"source:{index}"
                events.append(
                    {
                        "kind": "source_check",
                        "step_index": index,
                        "expected_hash": edge.source_exact_hash,
                        "observed_hash": observed_source,
                        "exact": False,
                    }
                )
                break
            selected = select_live_action(
                env,
                edge.action.action_name,
                action_args=edge.action.action_data,
            )
            if selected is None:
                divergence = f"action_unavailable:{index}"
                events.append(
                    {
                        "kind": "transition",
                        "step_index": index,
                        "action_key": edge.action.key,
                        "expected_target_hash": edge.target_exact_hash,
                        "observed_target_hash": "",
                        "exact": False,
                        "reason": "action_unavailable",
                    }
                )
                break
            frame = _step_env_action(env, selected)
            calls += 1
            observed_target = state_signature_from_frame(frame)
            exact = observed_target == edge.target_exact_hash
            events.append(
                {
                    "kind": "transition",
                    "step_index": index,
                    "action_key": edge.action.key,
                    "expected_target_hash": edge.target_exact_hash,
                    "observed_target_hash": observed_target,
                    "exact": exact,
                }
            )
            if not exact:
                divergence = f"target:{index}"
                break
    final = snapshot_frame(frame)
    final_hash = state_signature_from_frame(frame)
    terminal_failure = bool(
        _is_terminal(final.game_state)
        and int(final.levels_completed) == candidate.terminal_source_level
        and str(final.game_state).upper() not in {"WIN", "WON", "VICTORY"}
    )
    exact = bool(
        initial_exact
        and not divergence
        and final_hash == candidate.edges[-1].target_exact_hash
    )
    return TerminalConfirmation(
        candidate_id=candidate.candidate_id,
        repetition=repetition,
        exact=exact,
        terminal_failure=terminal_failure,
        confirmed=bool(exact and terminal_failure),
        calls=calls,
        final_exact_hash=final_hash,
        first_divergence=divergence,
        events=tuple(events),
    )


def build_progress_protected_shield(
    *,
    candidates: Sequence[TerminalTraceCandidate],
    confirmations: Sequence[TerminalConfirmation],
    protected_actions: Sequence[ProtectedActionSpec],
    protocol: TerminalShieldProtocol,
) -> tuple[ProgressProtectedTerminalShield, tuple[str, ...]]:
    confirmed_ids = {
        candidate.candidate_id
        for candidate in candidates
        if sum(
            item.confirmed
            for item in confirmations
            if item.candidate_id == candidate.candidate_id
        )
        >= protocol.terminal_confirmation_repetitions
    }
    base = MultiStepTerminalShield(
        horizon=protocol.terminal_horizon,
        minimum_support=protocol.terminal_minimum_support,
    )
    for candidate in candidates:
        if candidate.candidate_id in confirmed_ids:
            base.record_terminal_trace(
                candidate.edges,
                exact_replay_confirmed=True,
            )
    witness_ids = sorted(
        {witness_id for item in protected_actions for witness_id in item.witness_ids}
    )
    shield = ProgressProtectedTerminalShield(
        base=base,
        protected_pairs=tuple(
            (item.cell_id, item.action_key) for item in protected_actions
        ),
        witness_ids=witness_ids,
    )
    return shield, tuple(sorted(confirmed_ids))


def replay_witness_with_shield(
    *,
    witness: ProgressWitness,
    shield_payload: Mapping[str, Any],
    repetition: int,
    environments_dir: str | Path,
    env_factory: EnvFactory | None = None,
) -> WitnessShieldTrial:
    shield = ProgressProtectedTerminalShield.from_dict(shield_payload)
    env = _make_env(witness.game_id, environments_dir, env_factory)
    frame = _reset_env(env)
    calls = 1
    initial_hash = state_signature_from_frame(frame)
    initial_exact = initial_hash == witness.initial_exact_hash
    events: list[dict[str, Any]] = [
        {
            "kind": "reset",
            "expected_hash": witness.initial_exact_hash,
            "observed_hash": initial_hash,
            "exact": initial_exact,
        }
    ]
    divergence = "" if initial_exact else "initial:reset"
    vetoes = 0
    all_protected = True
    if initial_exact:
        for index, step in enumerate(witness.steps):
            observed_source = state_signature_from_frame(frame)
            if observed_source != step.expected_source_hash:
                divergence = f"source:{index}"
                break
            snapshot = snapshot_frame(frame)
            legal = _grounded_actions(env)
            cell_id = GoExploreArchive.cell_key(
                _symbolic_state(frame),
                level=int(snapshot.levels_completed),
                legal_actions=legal,
            )
            protected = shield.is_protected(cell_id, step.action)
            allowed = shield.allows(cell_id, step.action)
            all_protected = bool(all_protected and protected)
            vetoes += int(not allowed)
            event = {
                "kind": "shield_decision",
                "step_index": index,
                "cell_id": cell_id,
                "action_key": step.action.key,
                "protected": protected,
                "allowed": allowed,
            }
            events.append(event)
            if not protected:
                divergence = f"unprotected:{index}"
                break
            if not allowed:
                divergence = f"veto:{index}"
                break
            selected = select_live_action(
                env,
                step.action.action_name,
                action_args=step.action.action_data,
            )
            if selected is None:
                divergence = f"action_unavailable:{index}"
                break
            frame = _step_env_action(env, selected)
            calls += 1
            observed_target = state_signature_from_frame(frame)
            exact = observed_target == step.expected_target_hash
            events.append(
                {
                    "kind": "transition",
                    "step_index": index,
                    "expected_target_hash": step.expected_target_hash,
                    "observed_target_hash": observed_target,
                    "exact": exact,
                }
            )
            if not exact:
                divergence = f"target:{index}"
                break
    final = snapshot_frame(frame)
    final_hash = state_signature_from_frame(frame)
    progressed = int(final.levels_completed) > witness.initial_level
    exact = bool(
        initial_exact and not divergence and final_hash == witness.target_exact_hash
    )
    return WitnessShieldTrial(
        witness_id=witness.witness_id,
        repetition=repetition,
        exact=exact,
        progressed=progressed,
        all_actions_protected=all_protected,
        vetoed_actions=vetoes,
        calls=calls,
        final_exact_hash=final_hash,
        first_divergence=divergence,
        events=tuple(events),
    )


def run_shielded_burst_arm(
    *,
    game_id: str,
    seed: int,
    sdk_call_budget: int,
    burst_schedule: tuple[int, ...],
    environments_dir: str | Path,
    env_factory: EnvFactory | None = None,
    shield: ProgressProtectedTerminalShield | None = None,
    maximum_cells: int = 50_000,
) -> tuple[BurstRun, ProgressProtectedTerminalShield | None]:
    if tuple(int(value) for value in burst_schedule) != (4, 8, 16):
        raise ValueError("T12.3b burst runner requires the 4/8/16 schedule")
    archive = GoExploreArchive(maximum_cells=maximum_cells, seed=seed)
    env = _make_env(game_id, environments_dir, env_factory)
    frame = _reset_env(env)
    archive.sdk_calls = 1
    _record_root(archive, env, frame)
    excursions = []
    excursion_index = 0
    while archive.sdk_calls < sdk_call_budget:
        cell = archive.select_cell(
            remaining_sdk_calls=sdk_call_budget - archive.sdk_calls
        )
        if cell is None:
            break
        variant = cell.best_variant(archive.prefixes)
        env, frame, exact, restoration_calls = _restore_variant(
            archive=archive,
            variant=variant,
            game_id=game_id,
            environments_dir=environments_dir,
            env_factory=env_factory,
        )
        archive.sdk_calls += restoration_calls
        archive.note_replay(exact=exact)
        horizon = int(burst_schedule[excursion_index % len(burst_schedule)])
        executed = 0
        progress_edges = 0
        terminal_failures = 0
        reason = "burst_complete"
        if not exact:
            updated = replace(variant, replay_failures=variant.replay_failures + 1)
            cell.variants[variant.exact_hash] = updated
            if updated.replay_failures >= 2:
                cell.blocked = True
            reason = "restore_mismatch"
        else:
            source_cell = cell
            source_exact_hash = variant.exact_hash
            for _ in range(horizon):
                if archive.sdk_calls >= sdk_call_budget:
                    reason = "sdk_budget"
                    break
                before = snapshot_frame(frame)
                if _is_terminal(before.game_state):
                    reason = "terminal_source"
                    break
                candidates = _grounded_actions(env)
                action = archive.choose_action(
                    source_cell,
                    candidates,
                    shield=shield,
                )
                if action is None:
                    source_cell.blocked = True
                    reason = "no_shield_allowed_action" if shield else "no_legal_action"
                    break
                selected = select_live_action(
                    env,
                    action.action_name,
                    action_args=action.action_data,
                )
                if selected is None:
                    source_cell.action_attempts[action.key] = (
                        source_cell.action_attempts.get(action.key, 0) + 1
                    )
                    reason = "action_unavailable"
                    break
                after_frame = _step_env_action(env, selected)
                archive.sdk_calls += 1
                executed += 1
                after = snapshot_frame(
                    after_frame,
                    fallback_available_actions=before.available_actions,
                )
                target_hash = state_signature_from_frame(after_frame)
                level_delta = max(
                    0,
                    int(after.levels_completed) - int(before.levels_completed),
                )
                success = bool(
                    level_delta > 0
                    or str(after.game_state).upper() in {"WIN", "WON", "VICTORY"}
                )
                terminal = _is_terminal(after.game_state)
                edge = archive.add_transition(
                    source_cell_id=source_cell.cell_id,
                    source_exact_hash=source_exact_hash,
                    action=action,
                    target_state=_symbolic_state(after_frame),
                    target_exact_hash=target_hash,
                    target_level=int(after.levels_completed),
                    target_legal_actions=_grounded_actions(env),
                    terminal=terminal,
                    success=success,
                    changed=source_exact_hash != target_hash,
                )
                progress_edges += int(edge.level_delta > 0 or edge.success)
                terminal_failures += int(edge.terminal and not edge.success)
                frame = after_frame
                source_cell = archive.cells[edge.target_cell_id]
                source_exact_hash = edge.target_exact_hash
                if edge.level_delta > 0 or edge.success:
                    reason = "progress"
                    break
                if edge.terminal:
                    reason = "terminal_failure"
                    break
        excursions.append(
            BurstExcursion(
                excursion_index=excursion_index,
                requested_horizon=horizon,
                executed_actions=executed,
                restoration_calls=restoration_calls,
                exact_restoration=exact,
                start_cell_id=cell.cell_id,
                start_exact_hash=variant.exact_hash,
                stopped_reason=reason,
                progress_edges=progress_edges,
                terminal_failures=terminal_failures,
            )
        )
        excursion_index += 1
    return BurstRun(archive=archive, excursions=tuple(excursions)), shield


def _aggregate_gate(
    *,
    protocol: TerminalShieldProtocol,
    confirmations: Sequence[TerminalConfirmation],
    candidates: Sequence[TerminalTraceCandidate],
    shield: ProgressProtectedTerminalShield,
    witness_trials: Sequence[WitnessShieldTrial],
    conditions: Sequence[Mapping[str, Any]],
    sdk_calls: int,
) -> tuple[bool, dict[str, Any]]:
    confirmed_candidates = {
        item.candidate_id for item in confirmations if item.confirmed
    }
    confirmed_source_groups = {
        (candidate.source_seed, candidate.source_arm)
        for candidate in candidates
        if candidate.candidate_id in confirmed_candidates
    }
    confirmation_rate = len(confirmed_candidates) / max(1, len(candidates))
    per_seed = []
    control_terminal = 0
    treatment_terminal = 0
    control_actions = 0
    treatment_actions = 0
    control_cells = 0
    treatment_cells = 0
    control_sdk = 0
    treatment_sdk = 0
    control_progress = 0
    treatment_progress = 0
    terminal_regression_seeds = 0
    progress_regression_seeds = 0
    vetoes = 0
    minimum_replay_exact_rate = 1.0
    minimum_per_seed_coverage_ratio = float("inf")
    for condition in conditions:
        control = condition["arms"]["burst_control"]
        treatment = condition["arms"]["burst_terminal_shield"]
        control_metrics = control["metrics"]
        treatment_metrics = treatment["metrics"]
        c_actions = int(control_metrics["exploration_actions"])
        t_actions = int(treatment_metrics["exploration_actions"])
        c_terminal = int(control_metrics["terminal_edges"])
        t_terminal = int(treatment_metrics["terminal_edges"])
        c_rate = c_terminal / max(1, c_actions)
        t_rate = t_terminal / max(1, t_actions)
        terminal_regression_seeds += int(t_rate > c_rate)
        progress_regression_seeds += int(
            int(treatment_metrics["progress_edges"])
            < int(control_metrics["progress_edges"])
        )
        control_terminal += c_terminal
        treatment_terminal += t_terminal
        control_actions += c_actions
        treatment_actions += t_actions
        control_cells += int(control_metrics["symbolic_cells"])
        treatment_cells += int(treatment_metrics["symbolic_cells"])
        control_sdk += int(control_metrics["sdk_calls"])
        treatment_sdk += int(treatment_metrics["sdk_calls"])
        control_progress += int(control_metrics["progress_edges"])
        treatment_progress += int(treatment_metrics["progress_edges"])
        vetoes += int(treatment["shield_metrics"].get("vetoes", 0))
        minimum_replay_exact_rate = min(
            minimum_replay_exact_rate,
            float(control_metrics["replay_exact_rate"]),
            float(treatment_metrics["replay_exact_rate"]),
        )
        per_seed_coverage_ratio = float(
            treatment_metrics["symbolic_cells_per_1000_sdk_calls"]
        ) / max(
            1e-12,
            float(control_metrics["symbolic_cells_per_1000_sdk_calls"]),
        )
        minimum_per_seed_coverage_ratio = min(
            minimum_per_seed_coverage_ratio,
            per_seed_coverage_ratio,
        )
        per_seed.append(
            {
                "seed": condition["seed"],
                "control_terminal_rate": c_rate,
                "treatment_terminal_rate": t_rate,
                "control_progress": int(control_metrics["progress_edges"]),
                "treatment_progress": int(treatment_metrics["progress_edges"]),
                "control_coverage": float(
                    control_metrics["symbolic_cells_per_1000_sdk_calls"]
                ),
                "treatment_coverage": float(
                    treatment_metrics["symbolic_cells_per_1000_sdk_calls"]
                ),
                "coverage_ratio": per_seed_coverage_ratio,
                "vetoes": int(treatment["shield_metrics"].get("vetoes", 0)),
            }
        )
    control_terminal_rate = control_terminal / max(1, control_actions)
    treatment_terminal_rate = treatment_terminal / max(1, treatment_actions)
    terminal_rate_ratio = (
        1.0
        if control_terminal_rate == 0.0 and treatment_terminal_rate == 0.0
        else (
            1_000_000_000.0
            if control_terminal_rate == 0.0
            else treatment_terminal_rate / control_terminal_rate
        )
    )
    control_coverage = 1000.0 * control_cells / max(1, control_sdk)
    treatment_coverage = 1000.0 * treatment_cells / max(1, treatment_sdk)
    coverage_ratio = treatment_coverage / max(1e-12, control_coverage)
    witness_confirmations = sum(item.confirmed for item in witness_trials)
    expected_witness_confirmations = 2 * protocol.witness_repetitions
    shield_metrics = shield.metrics()
    metrics = {
        "confirmed_terminal_candidates": len(confirmed_candidates),
        "terminal_candidates": len(candidates),
        "terminal_confirmation_rate": confirmation_rate,
        "confirmed_terminal_source_groups": len(confirmed_source_groups),
        "expected_terminal_source_groups": len(protocol.source_seeds)
        * len(protocol.source_arms),
        "terminal_confirmation_divergences": dict(
            sorted(
                Counter(
                    item.first_divergence
                    for item in confirmations
                    if item.first_divergence
                ).items()
            )
        ),
        "source_shield": shield_metrics,
        "witness_confirmations": witness_confirmations,
        "expected_witness_confirmations": expected_witness_confirmations,
        "witness_vetoes": sum(item.vetoed_actions for item in witness_trials),
        "all_witness_actions_protected": all(
            item.all_actions_protected for item in witness_trials
        ),
        "control_terminal_rate": control_terminal_rate,
        "treatment_terminal_rate": treatment_terminal_rate,
        "terminal_rate_ratio": terminal_rate_ratio,
        "terminal_regression_seeds": terminal_regression_seeds,
        "progress_regression_seeds": progress_regression_seeds,
        "control_coverage": control_coverage,
        "treatment_coverage": treatment_coverage,
        "coverage_ratio": coverage_ratio,
        "minimum_per_seed_coverage_ratio": minimum_per_seed_coverage_ratio,
        "control_progress_edges": control_progress,
        "treatment_progress_edges": treatment_progress,
        "evaluation_vetoes": vetoes,
        "minimum_evaluation_replay_exact_rate": minimum_replay_exact_rate,
        "sdk_calls": sdk_calls,
        "maximum_total_sdk_calls": protocol.maximum_total_sdk_calls,
        "per_seed": per_seed,
    }
    passed = bool(
        len(confirmed_candidates) >= protocol.minimum_confirmed_terminal_traces
        and confirmation_rate >= protocol.minimum_terminal_confirmation_rate
        and len(confirmed_source_groups)
        == len(protocol.source_seeds) * len(protocol.source_arms)
        and shield_metrics["multi_step_hazard_observed"]
        and shield_metrics["confirmed_unsafe_actions"] >= 1
        and witness_confirmations == expected_witness_confirmations
        and metrics["witness_vetoes"] == 0
        and metrics["all_witness_actions_protected"]
        and vetoes >= protocol.minimum_vetoes
        and terminal_rate_ratio <= protocol.maximum_terminal_rate_ratio
        and terminal_regression_seeds <= protocol.maximum_terminal_regression_seeds
        and coverage_ratio >= protocol.minimum_coverage_ratio
        and minimum_per_seed_coverage_ratio >= protocol.minimum_coverage_ratio
        and treatment_progress >= control_progress
        and progress_regression_seeds == 0
        and minimum_replay_exact_rate >= protocol.minimum_evaluation_replay_exact_rate
        and sdk_calls <= protocol.maximum_total_sdk_calls
    )
    return passed, metrics


def _resolve_registry_path(manifest: Mapping[str, Any]) -> Path:
    path = Path(str(manifest["terminal_registry"]["path"]))
    return path if path.is_absolute() else Path(__file__).resolve().parents[3] / path


def _load_parent_witnesses(
    manifest: Mapping[str, Any],
) -> tuple[ProgressWitness, ...]:
    root = Path(__file__).resolve().parents[3]
    parent_path = Path(str(manifest["parent"]["manifest"]["path"]))
    if not parent_path.is_absolute():
        parent_path = root / parent_path
    parent = load_witness_manifest(parent_path, root=root)
    registry_path = Path(str(parent["witness_registry"]["path"]))
    if not registry_path.is_absolute():
        registry_path = root / registry_path
    _, witnesses = load_witness_registry(registry_path)
    return witnesses


def run_shield_experiment(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    environments_dir: str | Path = "environment_files",
    env_factory: EnvFactory | None = None,
) -> dict[str, Any]:
    manifest = load_shield_manifest(
        manifest_path,
        verify_code=env_factory is None,
    )
    protocol = TerminalShieldProtocol(**dict(manifest["protocol"]))
    _, candidates, protected_actions = load_shield_registry(
        _resolve_registry_path(manifest),
        protocol=protocol,
    )
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable run: {destination}")
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes_per_run)

    confirmations = []
    for candidate in candidates:
        for repetition in range(protocol.terminal_confirmation_repetitions):
            confirmations.append(
                confirm_terminal_candidate(
                    candidate=candidate,
                    repetition=repetition,
                    environments_dir=environments_dir,
                    env_factory=env_factory,
                )
            )
    shield, confirmed_ids = build_progress_protected_shield(
        candidates=candidates,
        confirmations=confirmations,
        protected_actions=protected_actions,
        protocol=protocol,
    )
    shield_payload = shield.to_dict()

    witnesses = _load_parent_witnesses(manifest)
    witness_trials = []
    for witness in witnesses:
        for repetition in range(protocol.witness_repetitions):
            witness_trials.append(
                replay_witness_with_shield(
                    witness=witness,
                    shield_payload=shield_payload,
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
            arm_shield = (
                None
                if arm == "burst_control"
                else ProgressProtectedTerminalShield.from_dict(shield_payload)
            )
            run, used_shield = run_shielded_burst_arm(
                game_id=game_id,
                seed=seed,
                sdk_call_budget=protocol.sdk_calls_per_evaluation_arm,
                burst_schedule=protocol.burst_schedule,
                environments_dir=environments_dir,
                env_factory=env_factory,
                shield=arm_shield,
                maximum_cells=protocol.maximum_cells,
            )
            archive_path = destination / game_id / str(seed) / f"{arm}.json"
            artifact = _write_archive(
                archive_path,
                run.archive,
                storage_budget=storage,
            )
            artifact.update({"game_id": game_id, "seed": seed, "arm": arm})
            archive_artifacts.append(artifact)
            excursion_path = (
                destination / game_id / str(seed) / f"{arm}_excursions.json"
            )
            _write_json_once(
                excursion_path,
                {
                    "format_version": "sage-t12.3b-burst-excursions-v1",
                    "game_id": game_id,
                    "seed": seed,
                    "arm": arm,
                    "excursions": [item.to_dict() for item in run.excursions],
                },
                storage_budget=storage,
            )
            excursions_meta = {
                "path": str(excursion_path.resolve()),
                "sha256": _file_sha256(excursion_path),
            }
            arms[arm] = {
                "metrics": run.metrics(),
                "shield_metrics": (
                    {} if used_shield is None else used_shield.metrics()
                ),
                "archive": artifact,
                "excursions": excursions_meta,
            }
        conditions.append({"game_id": game_id, "seed": seed, "arms": arms})

    sdk_calls = (
        sum(item.calls for item in confirmations)
        + sum(item.calls for item in witness_trials)
        + sum(
            int(arm["metrics"]["sdk_calls"])
            for condition in conditions
            for arm in condition["arms"].values()
        )
    )
    passed, metrics = _aggregate_gate(
        protocol=protocol,
        confirmations=confirmations,
        candidates=candidates,
        shield=shield,
        witness_trials=witness_trials,
        conditions=conditions,
        sdk_calls=sdk_calls,
    )

    confirmation_path = destination / "terminal_confirmations.json"
    _write_json_once(
        confirmation_path,
        {
            "format_version": "sage-t12.3b-terminal-confirmations-v1",
            "confirmed_candidate_ids": list(confirmed_ids),
            "confirmations": [item.to_dict() for item in confirmations],
        },
        storage_budget=storage,
    )
    shield_path = destination / "terminal_shield.json"
    _write_json_once(shield_path, shield_payload, storage_budget=storage)
    witness_path = destination / "witness_non_regression.json"
    _write_json_once(
        witness_path,
        {
            "format_version": "sage-t12.3b-witness-non-regression-v1",
            "trials": [item.to_dict() for item in witness_trials],
        },
        storage_budget=storage,
    )
    evaluation_path = destination / "paired_evaluation.json"
    _write_json_once(
        evaluation_path,
        {
            "format_version": "sage-t12.3b-paired-shield-evaluation-v1",
            "conditions": conditions,
            "archives": archive_artifacts,
        },
        storage_budget=storage,
    )
    report = {
        "format_version": "sage-t12.3b-terminal-shield-report-v1",
        "status": (
            "PASS_T12_3B_TERMINAL_SHIELD_GATE"
            if passed
            else "FAIL_T12_3B_TERMINAL_SHIELD_GATE"
        ),
        "passed": passed,
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "parent_t12_3a_receipt_checksum": manifest["parent"]["receipt"][
            "receipt_checksum"
        ],
        "metrics": metrics,
        "conditions": conditions,
        "storage": storage.snapshot(),
    }
    report_path = destination / "shield_report.json"
    _write_json_once(report_path, report, storage_budget=storage)
    receipt = shield_phase_receipt(
        manifest=manifest,
        phase="terminal_shield",
        passed=passed,
        status=report["status"],
        metrics=metrics,
        artifacts={
            "terminal_confirmations": {
                "path": str(confirmation_path.resolve()),
                "sha256": _file_sha256(confirmation_path),
            },
            "terminal_shield": {
                "path": str(shield_path.resolve()),
                "sha256": _file_sha256(shield_path),
            },
            "witness_non_regression": {
                "path": str(witness_path.resolve()),
                "sha256": _file_sha256(witness_path),
            },
            "paired_evaluation": {
                "path": str(evaluation_path.resolve()),
                "sha256": _file_sha256(evaluation_path),
            },
            "report": {
                "path": str(report_path.resolve()),
                "sha256": _file_sha256(report_path),
            },
            "terminal_registry": dict(manifest["terminal_registry"]),
        },
    )
    _write_json_once(
        destination / "shield_receipt.json",
        receipt,
        storage_budget=storage,
    )
    return report


def shield_experiment_status(
    *,
    manifest_path: str | Path,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest = load_shield_manifest(manifest_path)
    receipt = (
        None
        if receipt_path is None
        else load_shield_receipt(receipt_path, manifest=manifest)
    )
    return {
        "format_version": "sage-t12.3b-terminal-shield-status-v1",
        "manifest_checksum": manifest["manifest_checksum"],
        "protocol_checksum": manifest["protocol_checksum"],
        "parent_t12_3a_status": manifest["parent"]["receipt"]["status"],
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
    "TerminalConfirmation",
    "WitnessShieldTrial",
    "build_progress_protected_shield",
    "confirm_terminal_candidate",
    "replay_witness_with_shield",
    "run_shield_experiment",
    "run_shielded_burst_arm",
    "shield_experiment_status",
]
