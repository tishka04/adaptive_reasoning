"""Paired target-local search for SAGE.T12.4a.4d.

Both search arms start from the same exact level-1 state, expose the same
grounded action catalogue, use the same terminal shield and receive the same
SDK budget.  The treatment changes only deterministic action ordering: sparse
guard mismatches from T12.4a.4c are combined with branch-local object roles.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from theory.m1.polymorphic_a25_adapter import _step_env_action
from theory.m2.m3_execution_smoke import _reset_env
from theory.real_env_option_adapter import snapshot_frame
from theory.sage.live_prefix_counterfactual_collector import (
    select_live_action,
    state_signature_from_frame,
)
from theory.sage_t.contracts import AbstractState
from theory.unified_cognition_ab_benchmark import _is_terminal

from .archive import ROOT_PREFIX_ID, ArchiveEdge, _action_payload, _digest
from .contracts import GroundedAction, causal_program_from_dict
from .experiment import (
    RunStorageBudget,
    _file_sha256,
    _read_json,
    _signed,
    _write_json_once,
)
from .graph_experiment import (
    _grounded_actions,
    _intervention_bundles,
    _make_env,
    _restore_variant,
    _symbolic_state,
    _write_archive,
)
from .lineage_archive import LineagePreservingArchive
from .option_applicability_experiment import _state_descriptor
from .option_contracts import ContractedCausalOption, ContractedOptionProvider
from .shield_model import ProgressProtectedTerminalShield
from .target_regrounding_protocol import (
    TargetRegroundingProtocol,
    _checksum,
    load_target_regrounding_manifest,
    load_target_regrounding_receipt,
    target_regrounding_receipt,
)
from .witness_protocol import ProgressWitness, WitnessStep
from .witness_reconfirmation_protocol import load_reconfirmation_registry

EnvFactory = Callable[[str], Any]


class AnchoredLineageArchive(LineagePreservingArchive):
    """Lineage archive whose immutable replay prefix predates local edges."""

    def __init__(
        self,
        *,
        anchor_prefix_depth: int,
        maximum_cells: int,
        seed: int,
    ) -> None:
        super().__init__(maximum_cells=maximum_cells, seed=seed)
        self.anchor_prefix_depth = max(0, int(anchor_prefix_depth))

    def add_anchored_transition(
        self,
        *,
        source_cell_id: str,
        source_exact_hash: str,
        source_prefix_id: str,
        source_path_edge_ids: Sequence[str],
        action: GroundedAction,
        target_state: AbstractState,
        target_exact_hash: str,
        target_level: int,
        target_legal_actions: Sequence[GroundedAction],
        terminal: bool,
        success: bool,
        changed: bool,
    ) -> ArchiveEdge:
        source = self.cells[source_cell_id]
        representative = source.variants.get(source_exact_hash)
        if representative is None:
            raise KeyError(f"unknown exact anchored source: {source_exact_hash}")
        actual_path = tuple(str(value) for value in source_path_edge_ids)
        actual_prefix = str(source_prefix_id)
        relative_depth = self.prefixes.depth(actual_prefix) - self.anchor_prefix_depth
        if relative_depth != len(actual_path):
            raise ValueError("anchored prefix depth/path lineage mismatch")
        if actual_path:
            previous = self.edges.get(actual_path[-1])
            if previous is None or previous.target_exact_hash != source_exact_hash:
                raise ValueError("anchored lineage does not end at its source")
        if representative.prefix_id != actual_prefix:
            self.shortest_prefix_rebases_avoided += 1

        prefix_id = self.prefixes.extend(actual_prefix, action)
        target_cell_id = self.cell_key(
            target_state,
            level=target_level,
            legal_actions=target_legal_actions,
        )
        novel = target_cell_id not in self.cells
        ordinal = self._edge_observations
        edge_id = "edge_" + _digest(
            {
                "source": source_cell_id,
                "source_exact": source_exact_hash,
                "source_prefix": actual_prefix,
                "action": _action_payload(action),
                "target_exact": target_exact_hash,
                "ordinal": ordinal,
            }
        )
        self._edge_observations += 1
        edge = ArchiveEdge(
            edge_id=edge_id,
            ordinal=ordinal,
            source_cell_id=source_cell_id,
            source_exact_hash=source_exact_hash,
            action=action,
            target_cell_id=target_cell_id,
            target_exact_hash=str(target_exact_hash),
            level_delta=max(0, int(target_level) - int(source.level)),
            terminal=bool(terminal),
            success=bool(success),
            changed=bool(changed),
            novel=bool(novel),
            prefix_id=prefix_id,
        )
        self.edges[edge.edge_id] = edge
        source.expansions += 1
        source.action_attempts[action.key] = source.action_attempts.get(action.key, 0) + 1
        self.observe_state(
            state=target_state,
            exact_hash=target_exact_hash,
            level=target_level,
            legal_actions=target_legal_actions,
            prefix_id=prefix_id,
            path_edge_ids=actual_path + (edge.edge_id,),
            terminal=terminal,
        )
        self.lineage_attached_transitions += 1
        return edge

    def to_dict(self) -> dict[str, object]:
        return {**super().to_dict(), "anchor_prefix_depth": self.anchor_prefix_depth}


class ContractRegroundingScorer:
    """Identity-free contract mismatch plus branch-local role grounding."""

    def __init__(self, registry: ContractedCausalOption) -> None:
        self.registry = registry
        self.contracted_actions = frozenset(
            item.action_name for item in registry.effect_contracts
        )

    @staticmethod
    def _mechanism_descriptor(state: AbstractState) -> dict[str, Any]:
        return {
            "mechanism": {
                "predicate_counts": dict(
                    Counter(item.predicate for item in state.true_facts)
                ),
                "role_counts": dict(
                    Counter(role for entity in state.entities for role in entity.roles)
                ),
            }
        }

    def score(
        self,
        state: AbstractState,
        action: GroundedAction,
    ) -> tuple[float, float]:
        descriptor = self._mechanism_descriptor(state)
        mismatches = [
            spec.atoms[0]
            for spec in self.registry.initiation_specs
            if not spec.matches(descriptor)
        ]
        normalized_mismatch = sum(
            abs(atom.observed_value(descriptor) - atom.expected_value)
            / max(1.0, abs(float(atom.expected_value)))
            for atom in mismatches
        ) / max(1, len(mismatches))
        unseen_action = float(action.action_name not in self.contracted_actions)
        relevant_roles = {
            atom.key
            for atom in mismatches
            if atom.family == "role_counts"
        }
        role_bonus = 0.0
        locality = 0.0
        data = dict(action.action_data)
        if "x" in data and "y" in data:
            try:
                x, y = float(data["x"]), float(data["y"])
            except (TypeError, ValueError):
                x = y = math.inf
            nearest = None
            for entity in state.entities:
                if entity.center is None:
                    continue
                distance = math.hypot(x - entity.center[0], y - entity.center[1])
                if nearest is None or distance < nearest[0]:
                    nearest = (distance, entity)
            if nearest is not None:
                distance, entity = nearest
                locality = 1.0 / (1.0 + distance)
                role_bonus = float(bool(set(entity.roles) & relevant_roles))
        novelty = 2.0 * unseen_action + role_bonus + locality + normalized_mismatch
        predicted_change = 1.0 + unseen_action + 0.5 * role_bonus
        return predicted_change, novelty


@dataclass(frozen=True)
class TargetLocalRun:
    archive: AnchoredLineageArchive
    arm: str
    lineage_seed: int
    search_seed: int
    entry_exact: bool
    entry_hash: str
    entry_descriptor: Mapping[str, Any]
    candidate_catalog_checksum: str
    applicable_mass: float
    materialized_option_actions: int
    excursions: int
    progress_edge_id: str | None
    progress_suffix: tuple[GroundedAction, ...]
    progress_sdk_calls: int | None

    def metrics(self) -> dict[str, Any]:
        metrics = dict(self.archive.metrics())
        actions = int(metrics["edges"])
        terminal = int(metrics["terminal_edges"])
        return {
            **metrics,
            "arm": self.arm,
            "candidate_catalog_checksum": self.candidate_catalog_checksum,
            "entry_exact": self.entry_exact,
            "entry_hash": self.entry_hash,
            "excursions": self.excursions,
            "exploration_actions": actions,
            "first_progress_sdk_calls": self.progress_sdk_calls,
            "materialized_option_actions": self.materialized_option_actions,
            "option_applicable_mass": self.applicable_mass,
            "progress_suffix_length": len(self.progress_suffix),
            "terminal_failure_rate": terminal / max(1, actions),
        }


def _resolve(path: str, *, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _load_frozen_inputs(
    manifest: Mapping[str, Any],
    *,
    root: Path,
) -> tuple[
    tuple[ProgressWitness, ...],
    ContractedCausalOption,
    Any,
    ProgressProtectedTerminalShield,
]:
    _, witnesses = load_reconfirmation_registry(
        _resolve(str(manifest["inputs"]["witness_registry"]["path"]), root=root)
    )
    registry = ContractedCausalOption.from_dict(
        _read_json(
            _resolve(
                str(manifest["inputs"]["contracted_option_registry"]["path"]),
                root=root,
            )
        )
    )
    programs_payload = _read_json(
        _resolve(
            str(manifest["inputs"]["contracted_option_programs"]["path"]),
            root=root,
        )
    )
    programs = {
        program.canonical_hash: program
        for program in (
            causal_program_from_dict(dict(item))
            for item in programs_payload.get("programs", ())
        )
    }
    snapshot_payload = _read_json(
        _resolve(
            str(manifest["inputs"]["contracted_posterior"]["path"]),
            root=root,
        )
    )
    snapshot = dict(snapshot_payload["posterior"])
    particles = []
    for item in snapshot.get("particles", ()):
        program_hash = str(item["program_hash"])
        if program_hash not in programs:
            raise ValueError("T12.4a.4d posterior references an unknown program")
        particles.append(
            SimpleNamespace(
                probability=float(item["probability"]),
                program=programs[program_hash],
            )
        )
    if len(particles) != len(registry.owner_program_hashes):
        raise ValueError("T12.4a.4d contracted posterior is incomplete")
    posterior = SimpleNamespace(particles=tuple(particles))
    shield = ProgressProtectedTerminalShield.from_dict(
        _read_json(
            _resolve(
                str(manifest["inputs"]["terminal_shield"]["path"]),
                root=root,
            )
        )
    )
    return tuple(sorted(witnesses, key=lambda item: item.source_seed)), registry, posterior, shield


def _replay_anchor(
    *,
    witness: ProgressWitness,
    archive: AnchoredLineageArchive,
    game_id: str,
    environments_dir: str | Path,
    env_factory: EnvFactory | None,
) -> tuple[Any, Any, bool, int, str, str]:
    env = _make_env(game_id, environments_dir, env_factory)
    frame = _reset_env(env)
    calls = 1
    exact = state_signature_from_frame(frame) == witness.initial_exact_hash
    prefix_id = ROOT_PREFIX_ID
    for step in witness.steps:
        if not exact or state_signature_from_frame(frame) != step.expected_source_hash:
            exact = False
            break
        selected = select_live_action(
            env,
            step.action.action_name,
            action_args=step.action.action_data,
        )
        if selected is None:
            exact = False
            break
        frame = _step_env_action(env, selected)
        calls += 1
        prefix_id = archive.prefixes.extend(prefix_id, step.action)
        if state_signature_from_frame(frame) != step.expected_target_hash:
            exact = False
            break
    entry_hash = state_signature_from_frame(frame)
    if exact:
        exact = entry_hash == witness.target_exact_hash
    snapshot = snapshot_frame(frame)
    if exact:
        cell, _ = archive.observe_state(
            state=_symbolic_state(frame),
            exact_hash=entry_hash,
            level=int(snapshot.levels_completed),
            legal_actions=_grounded_actions(env),
            prefix_id=prefix_id,
            path_edge_ids=(),
            terminal=_is_terminal(snapshot.game_state),
        )
        cell.blocked = False
        return env, frame, True, calls, cell.cell_id, entry_hash
    return env, frame, False, calls, "", entry_hash


def _catalog_checksum(actions: Sequence[GroundedAction]) -> str:
    return _checksum(sorted(action.key for action in actions))


def run_target_local_arm(
    *,
    game_id: str,
    witness: ProgressWitness,
    registry: ContractedCausalOption,
    posterior: Any,
    shield: ProgressProtectedTerminalShield,
    arm: str,
    search_seed: int,
    sdk_call_budget: int,
    maximum_excursions: int,
    maximum_cells: int,
    burst_schedule: tuple[int, ...],
    environments_dir: str | Path,
    env_factory: EnvFactory | None = None,
) -> TargetLocalRun:
    if arm not in {"local_archive_control", "contract_regrounded"}:
        raise ValueError(f"unsupported target-local arm: {arm}")
    archive = AnchoredLineageArchive(
        anchor_prefix_depth=len(witness.steps),
        maximum_cells=maximum_cells,
        seed=search_seed,
    )
    env, frame, entry_exact, calls, _, entry_hash = _replay_anchor(
        witness=witness,
        archive=archive,
        game_id=game_id,
        environments_dir=environments_dir,
        env_factory=env_factory,
    )
    archive.sdk_calls = calls
    if not entry_exact:
        return TargetLocalRun(
            archive=archive,
            arm=arm,
            lineage_seed=witness.source_seed,
            search_seed=search_seed,
            entry_exact=False,
            entry_hash=entry_hash,
            entry_descriptor={},
            candidate_catalog_checksum="",
            applicable_mass=1.0,
            materialized_option_actions=0,
            excursions=0,
            progress_edge_id=None,
            progress_suffix=(),
            progress_sdk_calls=None,
        )
    entry_snapshot = snapshot_frame(frame)
    entry_state = _symbolic_state(frame)
    descriptor = _state_descriptor(
        entry_state,
        exact_hash=entry_hash,
        level=int(entry_snapshot.levels_completed),
        game_state=str(entry_snapshot.game_state),
    )
    provider = ContractedOptionProvider(registry)
    applicable_mass = provider.applicable_mass(descriptor, posterior)
    materialized = provider.materialize(descriptor, entry_state, posterior)
    catalog = _grounded_actions(env)
    catalog_checksum = _catalog_checksum(catalog)
    scorer = ContractRegroundingScorer(registry)

    excursion_index = 0
    progress_edge_id = None
    progress_suffix: tuple[GroundedAction, ...] = ()
    progress_sdk_calls = None
    while (
        archive.sdk_calls < sdk_call_budget
        and excursion_index < maximum_excursions
        and progress_edge_id is None
    ):
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
        if not exact:
            updated = replace(variant, replay_failures=variant.replay_failures + 1)
            cell.variants[variant.exact_hash] = updated
            if updated.replay_failures >= 2:
                cell.blocked = True
            excursion_index += 1
            continue
        source_cell = cell
        source_hash = variant.exact_hash
        source_prefix_id = variant.prefix_id
        source_path_ids = variant.path_edge_ids
        horizon = int(burst_schedule[excursion_index % len(burst_schedule)])
        for _ in range(horizon):
            if archive.sdk_calls >= sdk_call_budget:
                break
            before = snapshot_frame(frame)
            if _is_terminal(before.game_state):
                break
            candidates = _grounded_actions(env)
            action = archive.choose_action(
                source_cell,
                candidates,
                shield=shield,
                novelty_scorer=(
                    scorer if arm == "contract_regrounded" else None
                ),
            )
            if action is None:
                source_cell.blocked = True
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
                break
            after_frame = _step_env_action(env, selected)
            archive.sdk_calls += 1
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
            edge = archive.add_anchored_transition(
                source_cell_id=source_cell.cell_id,
                source_exact_hash=source_hash,
                source_prefix_id=source_prefix_id,
                source_path_edge_ids=source_path_ids,
                action=action,
                target_state=_symbolic_state(after_frame),
                target_exact_hash=target_hash,
                target_level=int(after.levels_completed),
                target_legal_actions=_grounded_actions(env),
                terminal=terminal,
                success=success,
                changed=source_hash != target_hash,
            )
            source_prefix_id = edge.prefix_id
            source_path_ids = source_path_ids + (edge.edge_id,)
            frame = after_frame
            source_cell = archive.cells[edge.target_cell_id]
            source_hash = edge.target_exact_hash
            if edge.level_delta > 0 or edge.success:
                progress_edge_id = edge.edge_id
                actions = archive.prefixes.actions(edge.prefix_id)
                progress_suffix = tuple(actions[len(witness.steps) :])
                progress_sdk_calls = archive.sdk_calls
                break
            if edge.terminal:
                break
        excursion_index += 1
    return TargetLocalRun(
        archive=archive,
        arm=arm,
        lineage_seed=witness.source_seed,
        search_seed=search_seed,
        entry_exact=True,
        entry_hash=entry_hash,
        entry_descriptor=descriptor,
        candidate_catalog_checksum=catalog_checksum,
        applicable_mass=applicable_mass,
        materialized_option_actions=len(materialized),
        excursions=excursion_index,
        progress_edge_id=progress_edge_id,
        progress_suffix=progress_suffix,
        progress_sdk_calls=progress_sdk_calls,
    )


def _confirm_suffix(
    *,
    witness: ProgressWitness,
    suffix: Sequence[GroundedAction],
    repetition: int,
    game_id: str,
    environments_dir: str | Path,
    env_factory: EnvFactory | None,
) -> dict[str, Any]:
    env = _make_env(game_id, environments_dir, env_factory)
    frame = _reset_env(env)
    calls = 1
    prefix_exact = state_signature_from_frame(frame) == witness.initial_exact_hash
    for step in witness.steps:
        if not prefix_exact or state_signature_from_frame(frame) != step.expected_source_hash:
            prefix_exact = False
            break
        selected = select_live_action(
            env,
            step.action.action_name,
            action_args=step.action.action_data,
        )
        if selected is None:
            prefix_exact = False
            break
        frame = _step_env_action(env, selected)
        calls += 1
        if state_signature_from_frame(frame) != step.expected_target_hash:
            prefix_exact = False
            break
    prefix_exact = bool(
        prefix_exact and state_signature_from_frame(frame) == witness.target_exact_hash
    )
    available = prefix_exact
    events = []
    entry_level = int(snapshot_frame(frame).levels_completed)
    for position, action in enumerate(suffix):
        if not available:
            break
        before = snapshot_frame(frame)
        before_hash = state_signature_from_frame(frame)
        selected = select_live_action(
            env,
            action.action_name,
            action_args=action.action_data,
        )
        if selected is None:
            available = False
            break
        frame = _step_env_action(env, selected)
        calls += 1
        after = snapshot_frame(frame, fallback_available_actions=before.available_actions)
        after_hash = state_signature_from_frame(frame)
        level_delta = max(
            0,
            int(after.levels_completed) - int(before.levels_completed),
        )
        events.append(
            {
                "action": {
                    "action_name": action.action_name,
                    "action_data": dict(action.action_data),
                },
                "expected_source_hash": before_hash,
                "expected_target_hash": after_hash,
                "level_delta": level_delta,
                "position": position,
                "success": bool(level_delta > 0),
                "terminal": _is_terminal(after.game_state),
            }
        )
        if level_delta > 0 or _is_terminal(after.game_state):
            break
    final = snapshot_frame(frame)
    return {
        "available": available and len(events) == len(suffix),
        "calls": calls,
        "events": events,
        "final_exact_hash": state_signature_from_frame(frame),
        "final_level": int(final.levels_completed),
        "lineage_seed": witness.source_seed,
        "prefix_exact": prefix_exact,
        "progressed": int(final.levels_completed) > entry_level,
        "repetition": repetition,
        "terminal_failure": bool(
            _is_terminal(final.game_state)
            and int(final.levels_completed) <= entry_level
        ),
    }


def _discovered_witnesses(
    *,
    witnesses: Sequence[ProgressWitness],
    trials: Sequence[Mapping[str, Any]],
    archive_sha256: str,
    progress_edge_id: str,
    source_arm: str,
) -> tuple[ProgressWitness, ...]:
    output = []
    for witness in witnesses:
        trial = next(
            item
            for item in trials
            if int(item["lineage_seed"]) == witness.source_seed
            and int(item["repetition"]) == 0
        )
        suffix_steps = tuple(
            WitnessStep.from_dict(dict(item)) for item in trial["events"]
        )
        token = _checksum(
            {
                "lineage": witness.source_seed,
                "route": witness.route_checksum,
                "suffix": [item.to_dict() for item in suffix_steps],
            }
        )[:20]
        output.append(
            ProgressWitness(
                witness_id=f"witness_t12_4a_4d_{token}",
                game_id=witness.game_id,
                source_seed=witness.source_seed,
                source_arm=source_arm,
                source_archive_sha256=archive_sha256,
                source_progress_edge_id=progress_edge_id,
                initial_exact_hash=witness.initial_exact_hash,
                initial_level=witness.initial_level,
                target_exact_hash=str(trial["final_exact_hash"]),
                target_level=int(trial["final_level"]),
                steps=witness.steps + suffix_steps,
            )
        )
    return tuple(output)


def _gate(
    *,
    protocol: TargetRegroundingProtocol,
    conditions: Sequence[Mapping[str, Any]],
    confirmation_trials: Sequence[Mapping[str, Any]],
    candidate: TargetLocalRun | None,
    total_sdk_calls: int,
) -> tuple[bool, bool, dict[str, Any]]:
    paired_catalogs = all(
        condition["arms"]["local_archive_control"]["metrics"][
            "candidate_catalog_checksum"
        ]
        == condition["arms"]["contract_regrounded"]["metrics"][
            "candidate_catalog_checksum"
        ]
        for condition in conditions
    )
    all_arms = [
        arm
        for condition in conditions
        for arm in condition["arms"].values()
    ]
    all_entries_exact = all(arm["metrics"]["entry_exact"] for arm in all_arms)
    all_replays_exact = all(
        float(arm["metrics"]["replay_exact_rate"]) == 1.0 for arm in all_arms
    )
    all_options_blocked = all(
        int(arm["metrics"]["materialized_option_actions"]) == 0
        and 1.0 - float(arm["metrics"]["option_applicable_mass"])
        >= protocol.minimum_contract_block_margin
        for arm in all_arms
    )
    within_per_arm_budget = all(
        int(arm["metrics"]["sdk_calls"]) <= protocol.sdk_calls_per_search_arm
        for arm in all_arms
    )
    progress = {
        arm: sum(
            int(condition["arms"][arm]["metrics"]["progress_edges"])
            for condition in conditions
        )
        for arm in protocol.search_arms
    }
    terminal = {
        arm: sum(
            int(condition["arms"][arm]["metrics"]["terminal_edges"])
            for condition in conditions
        )
        for arm in protocol.search_arms
    }
    actions = {
        arm: sum(
            int(condition["arms"][arm]["metrics"]["exploration_actions"])
            for condition in conditions
        )
        for arm in protocol.search_arms
    }
    terminal_rates = {
        arm: terminal[arm] / max(1, actions[arm]) for arm in protocol.search_arms
    }
    final_hashes = {str(item["final_exact_hash"]) for item in confirmation_trials}
    expected_confirmations = (
        len(protocol.source_lineages) * protocol.confirmation_repetitions_per_lineage
    )
    confirmation_exact_rate = sum(
        bool(item["prefix_exact"]) for item in confirmation_trials
    ) / max(1, expected_confirmations)
    confirmed = bool(
        len(confirmation_trials) == expected_confirmations
        and confirmation_exact_rate >= protocol.minimum_confirmation_exact_rate
        and all(
            item["prefix_exact"]
            and item["available"]
            and item["progressed"]
            and not item["terminal_failure"]
            for item in confirmation_trials
        )
        and len(final_hashes) == 1
    )
    control_calls = [
        int(value)
        for condition in conditions
        if (
            value := condition["arms"]["local_archive_control"]["metrics"][
                "first_progress_sdk_calls"
            ]
        )
        is not None
    ]
    treatment_calls = [
        int(value)
        for condition in conditions
        if (
            value := condition["arms"]["contract_regrounded"]["metrics"][
                "first_progress_sdk_calls"
            ]
        )
        is not None
    ]
    paired_efficiency_wins = 0
    paired_efficiency_losses = 0
    paired_efficiency_ties = 0
    for condition in conditions:
        control_call = condition["arms"]["local_archive_control"]["metrics"][
            "first_progress_sdk_calls"
        ]
        treatment_call = condition["arms"]["contract_regrounded"]["metrics"][
            "first_progress_sdk_calls"
        ]
        if treatment_call is not None and control_call is None:
            paired_efficiency_wins += 1
        elif treatment_call is None and control_call is not None:
            paired_efficiency_losses += 1
        elif treatment_call is not None and control_call is not None:
            if int(treatment_call) < int(control_call):
                paired_efficiency_wins += 1
            elif int(treatment_call) > int(control_call):
                paired_efficiency_losses += 1
            else:
                paired_efficiency_ties += 1
    guidance_passed = bool(
        progress["contract_regrounded"] >= protocol.minimum_progress_edges
        and terminal_rates["contract_regrounded"]
        <= terminal_rates["local_archive_control"] + 1e-12
        and paired_efficiency_wins >= 1
        and paired_efficiency_losses == 0
    )
    checks = {
        "all_anchor_replays_exact": all_entries_exact,
        "all_archive_replays_exact": all_replays_exact,
        "contracted_option_blocked_at_every_anchor": all_options_blocked,
        "discovered_suffix_within_bound": bool(
            candidate
            and 0 < len(candidate.progress_suffix)
            <= protocol.maximum_witness_suffix_actions
        ),
        "exact_cross_lineage_confirmation": confirmed,
        "paired_candidate_catalogs_identical": paired_catalogs,
        "progress_witness_discovered": candidate is not None,
        "sdk_budget_respected": (
            within_per_arm_budget
            and total_sdk_calls <= protocol.maximum_total_sdk_calls
        ),
        "terminal_failure_rate_bounded": max(terminal_rates.values(), default=0.0)
        <= protocol.maximum_terminal_failure_rate,
    }
    metrics = {
        "actions": actions,
        "checks": checks,
        "confirmation_count": len(confirmation_trials),
        "confirmation_exact_rate": confirmation_exact_rate,
        "expected_confirmation_count": expected_confirmations,
        "final_hash_count": len(final_hashes),
        "guidance_claim_passed": guidance_passed,
        "minimum_control_progress_sdk_calls": min(control_calls, default=None),
        "minimum_treatment_progress_sdk_calls": min(treatment_calls, default=None),
        "paired_efficiency_losses": paired_efficiency_losses,
        "paired_efficiency_ties": paired_efficiency_ties,
        "paired_efficiency_wins": paired_efficiency_wins,
        "progress_edges": progress,
        "sdk_calls_used": total_sdk_calls,
        "terminal_edges": terminal,
        "terminal_failure_rates": terminal_rates,
    }
    return all(checks.values()), guidance_passed, metrics


def run_target_regrounding_experiment(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    environments_dir: str | Path = "environment_files",
    env_factory: EnvFactory | None = None,
) -> dict[str, Any]:
    manifest = load_target_regrounding_manifest(
        manifest_path,
        verify_code=env_factory is None,
    )
    if not manifest.get("scientific_claims_authorized", False):
        raise ValueError("T12.4a.4d run requires a clean scientific freeze")
    if not manifest["firewall"]["target_regrounding_experiment_authorized"]:
        raise ValueError("T12.4a.4d target re-grounding is not authorized")
    protocol = TargetRegroundingProtocol(**dict(manifest["protocol"]))
    root = Path(__file__).resolve().parents[3]
    witnesses, registry, posterior, frozen_shield = _load_frozen_inputs(
        manifest,
        root=root,
    )
    destination = Path(output_dir)
    if destination.exists() and any(destination.rglob("*")):
        raise FileExistsError(f"refusing to append to immutable run: {destination}")
    storage = RunStorageBudget(destination, protocol.maximum_artifact_bytes_per_run)
    conditions = []
    runs: list[tuple[TargetLocalRun, dict[str, Any]]] = []
    archive_artifacts = []
    bundles = []
    for search_seed in protocol.search_seeds:
        for witness in witnesses:
            arms = {}
            for arm in protocol.search_arms:
                shield = ProgressProtectedTerminalShield.from_dict(
                    frozen_shield.to_dict()
                )
                run = run_target_local_arm(
                    game_id=str(manifest["game_id"]),
                    witness=witness,
                    registry=registry,
                    posterior=posterior,
                    shield=shield,
                    arm=arm,
                    search_seed=search_seed,
                    sdk_call_budget=protocol.sdk_calls_per_search_arm,
                    maximum_excursions=protocol.maximum_excursions_per_arm,
                    maximum_cells=protocol.maximum_cells,
                    burst_schedule=protocol.burst_schedule,
                    environments_dir=environments_dir,
                    env_factory=env_factory,
                )
                path = (
                    destination
                    / str(manifest["game_id"])
                    / str(search_seed)
                    / str(witness.source_seed)
                    / f"{arm}.json"
                )
                artifact = _write_archive(path, run.archive, storage_budget=storage)
                artifact.update(
                    {
                        "arm": arm,
                        "lineage_seed": witness.source_seed,
                        "search_seed": search_seed,
                    }
                )
                archive_artifacts.append(artifact)
                metrics = run.metrics()
                arms[arm] = {
                    "artifact": artifact,
                    "metrics": metrics,
                    "shield_metrics": shield.metrics(),
                }
                runs.append((run, artifact))
                for bundle_index, item in enumerate(
                    _intervention_bundles(
                        run.archive,
                        game_id=str(manifest["game_id"]),
                        seed=search_seed,
                    )
                ):
                    bundles.append(
                        {
                            **item,
                            "arm": arm,
                            "bundle_id": (
                                f"bundle_{manifest['game_id']}_{search_seed}_"
                                f"{witness.source_seed}_{arm}_{bundle_index:06d}"
                            ),
                            "lineage_seed": witness.source_seed,
                            "source_bundle_id": item["bundle_id"],
                        }
                    )
            conditions.append(
                {
                    "arms": arms,
                    "game_id": manifest["game_id"],
                    "lineage_seed": witness.source_seed,
                    "search_seed": search_seed,
                }
            )

    candidates = [
        (run, artifact)
        for run, artifact in runs
        if run.progress_edge_id is not None and run.progress_suffix
    ]
    selected = min(
        candidates,
        key=lambda item: (
            int(item[0].progress_sdk_calls or 10**9),
            len(item[0].progress_suffix),
            item[0].search_seed,
            item[0].lineage_seed,
            item[0].arm,
        ),
        default=None,
    )
    confirmation_trials = []
    if selected is not None:
        candidate, _ = selected
        for witness in witnesses:
            for repetition in range(protocol.confirmation_repetitions_per_lineage):
                confirmation_trials.append(
                    _confirm_suffix(
                        witness=witness,
                        suffix=candidate.progress_suffix,
                        repetition=repetition,
                        game_id=str(manifest["game_id"]),
                        environments_dir=environments_dir,
                        env_factory=env_factory,
                    )
                )
    total_sdk_calls = sum(run.archive.sdk_calls for run, _ in runs) + sum(
        int(item["calls"]) for item in confirmation_trials
    )
    passed, guidance_passed, gate_metrics = _gate(
        protocol=protocol,
        conditions=conditions,
        confirmation_trials=confirmation_trials,
        candidate=None if selected is None else selected[0],
        total_sdk_calls=total_sdk_calls,
    )

    trials_path = destination / "confirmation_trials.json"
    bundles_path = destination / "intervention_bundles.json"
    registry_path = destination / "progress_witnesses.sealed.json"
    report_path = destination / "target_regrounding_report.json"
    _write_json_once(
        trials_path,
        {
            "format_version": "sage-t12.4a.4d-confirmation-trials-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "trials": confirmation_trials,
        },
        storage_budget=storage,
    )
    _write_json_once(
        bundles_path,
        {
            "format_version": "sage-t12.4a.4d-intervention-bundles-v1",
            "bundles": bundles,
            "manifest_checksum": manifest["manifest_checksum"],
        },
        storage_budget=storage,
    )
    discovered: tuple[ProgressWitness, ...] = ()
    selected_payload = None
    if passed and selected is not None:
        candidate, artifact = selected
        discovered = _discovered_witnesses(
            witnesses=witnesses,
            trials=confirmation_trials,
            archive_sha256=str(artifact["sha256"]),
            progress_edge_id=str(candidate.progress_edge_id),
            source_arm=candidate.arm,
        )
        selected_payload = {
            "arm": candidate.arm,
            "archive_sha256": artifact["sha256"],
            "lineage_seed": candidate.lineage_seed,
            "progress_edge_id": candidate.progress_edge_id,
            "search_seed": candidate.search_seed,
            "suffix": [
                {
                    "action_name": action.action_name,
                    "action_data": dict(action.action_data),
                }
                for action in candidate.progress_suffix
            ],
        }
    witness_registry = _signed(
        {
            "format_version": "sage-t12.4a.4d-progress-witness-registry-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "parent_t12_4a_4c_receipt_checksum": manifest["parent"]["receipt"][
                "receipt_checksum"
            ],
            "protocol_checksum": manifest["protocol_checksum"],
            "selected_discovery": selected_payload,
            "witnesses": [item.to_dict() for item in discovered],
        },
        "registry_checksum",
    )
    _write_json_once(registry_path, witness_registry, storage_budget=storage)
    metrics = {
        **gate_metrics,
        "blocked_option_shadow_control": protocol.blocked_option_shadow_control,
        "discovered_witness_count": len(discovered),
        "exact_prefix_intervention_bundles": len(bundles),
        "paired_condition_count": len(conditions),
        "selected_discovery": selected_payload,
        "storage": storage.snapshot(),
    }
    status = (
        "PASS_T12_4A_4D_TARGET_WITNESS_GATE"
        if passed
        else "FAIL_T12_4A_4D_TARGET_WITNESS_GATE"
    )
    report = {
        "conditions": conditions,
        "format_version": "sage-t12.4a.4d-target-regrounding-report-v1",
        "guidance_claim_authorized": guidance_passed,
        "manifest_checksum": manifest["manifest_checksum"],
        "metrics": metrics,
        "passed": passed,
        "protocol_checksum": manifest["protocol_checksum"],
        "status": status,
        "storage": storage.snapshot(),
    }
    _write_json_once(report_path, report, storage_budget=storage)
    receipt = target_regrounding_receipt(
        manifest=manifest,
        phase="target_regrounding",
        passed=passed,
        status=status,
        metrics=metrics,
        artifacts={
            "archives": archive_artifacts,
            "confirmation_trials": {
                "path": str(trials_path.resolve()),
                "sha256": _file_sha256(trials_path),
            },
            "intervention_bundles": {
                "path": str(bundles_path.resolve()),
                "sha256": _file_sha256(bundles_path),
            },
            "progress_witness_registry": {
                "path": str(registry_path.resolve()),
                "sha256": _file_sha256(registry_path),
            },
            "report": {
                "path": str(report_path.resolve()),
                "sha256": _file_sha256(report_path),
            },
        },
    )
    _write_json_once(
        destination / "target_regrounding_receipt.json",
        receipt,
        storage_budget=storage,
    )
    return receipt


def target_regrounding_status(
    *,
    manifest_path: str | Path,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest = load_target_regrounding_manifest(manifest_path)
    receipt = None
    if receipt_path is not None and Path(receipt_path).is_file():
        receipt = load_target_regrounding_receipt(
            receipt_path,
            manifest=manifest,
        )
    passed = bool(
        receipt
        and receipt.get("passed") is True
        and receipt.get("phase") == "target_regrounding"
        and receipt.get("status") == "PASS_T12_4A_4D_TARGET_WITNESS_GATE"
    )
    guidance = bool(
        passed and receipt.get("metrics", {}).get("guidance_claim_passed", False)
    )
    return {
        "firewall": {
            "holdout_opened": False,
            "neural_active_evaluation_authorized": False,
            "neural_training_authorized": False,
            "option_control_authorized": False,
            "production_authority": False,
            "source_validation_opened": False,
            "t12_4a_4e_option_extraction_freeze_authorized": passed,
            "t12_4b_freeze_authorized": False,
            "t12_5_freeze_authorized": False,
            "target_regrounding_experiment_authorized": manifest["firewall"][
                "target_regrounding_experiment_authorized"
            ],
            "terminal_shield_production_authority": False,
        },
        "format_version": "sage-t12.4a.4d-target-regrounding-status-v1",
        "guidance_claim_authorized": guidance,
        "manifest_checksum": manifest["manifest_checksum"],
        "next_phase_authorized": passed,
        "parent_t12_4a_4c_status": manifest["parent"]["receipt"]["status"],
        "protocol_checksum": manifest["protocol_checksum"],
        "receipt": receipt,
    }


__all__ = [
    "AnchoredLineageArchive",
    "ContractRegroundingScorer",
    "TargetLocalRun",
    "run_target_local_arm",
    "run_target_regrounding_experiment",
    "target_regrounding_status",
]
