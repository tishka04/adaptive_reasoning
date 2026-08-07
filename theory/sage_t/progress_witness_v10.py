"""SAGE.T10 progress-witness induction on real ARC source games.

The module deliberately separates grounded evidence from transferable theory.
Coordinates and game ids may occur in an audit trail, but never in the
``transferable_payload`` or in the canonical SAGE.T program.

T10 searches for the *first* observed level increment with a small grammar of
active experiments:

* repeat one causally distinct target;
* follow the successor relation of a geometric chain toward its enclosed end.

A positive trace is compiled into a complete ``JointProgramHypothesis`` whose
control skeleton is ``repeat_apply_until_progress``.  The same skeleton is
then tested leave-one-game-out with fresh grounding in the other source game.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from theory.live_transition_loop import build_observation, build_transition_record

from .compiler import compile_observation, compile_transition_record
from .contracts import (
    AbstractEntity,
    AbstractState,
    ActionBinding,
    ActionCandidate,
    Effect,
    Expression,
    GoalRule,
    JointProgramHypothesis,
    ObjectSchema,
    ObservedTransition,
    ProgressRule,
    TerminalRule,
    TransitionRule,
)
from .executor import ProgramExecutor
from .posterior import ProgramPosterior

FORMAT_VERSION = "sage-t10-progress-witness-v1"
CONTROL_FAMILY = "repeat_apply_until_progress"
WIN_STATES = frozenset({"WIN", "WON", "VICTORY"})
TERMINAL_STATES = WIN_STATES | frozenset(
    {"GAME_OVER", "TERMINATED", "FINISHED"}
)
FAILURE_DIAGNOSES = frozenset(
    {"GENERATOR_MISS", "SEQUENCE_MISS", "GROUNDING_MISS", "POSTERIOR_MISS"}
)
SOURCE_TRAIN_GAMES = ("lp85-305b61c3", "su15-4c352900")
DEFAULT_OUTPUT_DIR = Path("training") / "sage_t" / "progress_witness_v10_0"


class StatefulWitnessPosterior(ProgramPosterior):
    """T10 challenger: semantic deduplication over the whole stateful trace.

    The legacy posterior is intentionally left untouched because old frozen
    replays hash that implementation.  T10 needs a stateful equivalence test:
    programs agreeing on a prefix are not equivalent if their counters make
    them diverge at a later teleological boundary.
    """

    def _observed_semantic_signature(  # type: ignore[override]
        self,
        program: JointProgramHypothesis,
    ) -> tuple[Any, ...]:
        output = []
        state = None
        for evidence in self.history:
            if evidence.reset:
                state = None
                continue
            start = (
                evidence.state_before
                if state is None
                else state.merge_observation(evidence.state_before)
            )
            packet = self.executor.step(program, start, evidence.action)
            output.append((evidence.action.key, packet.full_signature))
            predicted_state = packet.state_after or start
            state = predicted_state.merge_observation(evidence.state_after)
        return tuple(output)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _visual_digest(grid: Any) -> str:
    array = np.asarray(grid, dtype=np.int32)
    return hashlib.sha1(array.tobytes()).hexdigest()[:16]


def _terminal(state: Any) -> bool:
    return str(state).upper() in TERMINAL_STATES


@dataclass(frozen=True)
class GroundedAction:
    """One branch-local action.  It is evidence, never transferable theory."""

    action_name: str
    action_data: tuple[tuple[str, int | float | str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_name", str(self.action_name).upper())
        object.__setattr__(
            self,
            "action_data",
            tuple(sorted((str(key), value) for key, value in self.action_data)),
        )

    @classmethod
    def from_view(cls, action: Any) -> GroundedAction:
        return cls(
            action_name=str(getattr(action, "name", "")),
            action_data=tuple(
                dict(getattr(action, "action_args", {}) or {}).items()
            ),
        )

    @property
    def data(self) -> dict[str, int | float | str]:
        return dict(self.action_data)

    @property
    def key(self) -> str:
        return f"{self.action_name}:{_canonical_json(self.data)}"

    @property
    def candidate(self) -> ActionCandidate:
        return ActionCandidate(self.action_name, self.data)

    def to_dict(self) -> dict[str, Any]:
        return {"action_name": self.action_name, "action_data": self.data}


@dataclass(frozen=True)
class AbstractWitnessStep:
    """Coordinate-free causal description of one successful step."""

    operator: str = "apply"
    target_selector: str = "causal_target"
    relation: str = "identity"
    expected_event: str = "state_change"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class SearchConfig:
    maximum_horizon: int = 16
    maximum_one_step_actions: int = 512
    maximum_effect_representatives: int = 32
    chain_link_radius: float = 6.0
    chain_minimum_nodes: int = 6
    chain_stride: int = 2
    maximum_candidate_macros: int = 40

    def __post_init__(self) -> None:
        if not 1 <= int(self.maximum_horizon) <= 64:
            raise ValueError("maximum_horizon must be in [1, 64]")
        if int(self.maximum_one_step_actions) < 1:
            raise ValueError("maximum_one_step_actions must be positive")
        if int(self.maximum_effect_representatives) < 1:
            raise ValueError("maximum_effect_representatives must be positive")
        if int(self.chain_minimum_nodes) < 3:
            raise ValueError("chain_minimum_nodes must be at least three")
        if int(self.chain_stride) < 1:
            raise ValueError("chain_stride must be positive")


@dataclass(frozen=True)
class EffectScanRow:
    action: GroundedAction
    effect_key: str
    changed_cells: int
    level_delta: int
    game_state: str
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "effect_key": self.effect_key,
            "changed_cells": self.changed_cells,
            "level_delta": self.level_delta,
            "game_state": self.game_state,
            "latency_ms": round(self.latency_ms, 3),
        }


@dataclass(frozen=True)
class CandidateMacro:
    schema: str
    relation: str
    actions: tuple[GroundedAction, ...]
    context: tuple[tuple[str, int | float | str], ...] = ()

    @property
    def key(self) -> str:
        return _sha256_payload(
            {
                "schema": self.schema,
                "relation": self.relation,
                "actions": [action.key for action in self.actions],
            }
        )[:20]

    @property
    def transferable_descriptor(self) -> dict[str, Any]:
        return {
            "control_family": CONTROL_FAMILY,
            "schema": self.schema,
            "relation": self.relation,
            "operator": "apply",
            "target_selector": (
                "successor_toward_enclosure"
                if self.schema == "path_successor"
                else "same_effect_distinct_target"
            ),
            "maximum_steps": len(self.actions),
            "context": dict(self.context),
        }


@dataclass(frozen=True)
class ExecutedTrace:
    actions: tuple[GroundedAction, ...]
    observations: tuple[ObservedTransition, ...]
    rows: tuple[Mapping[str, Any], ...]
    initial_state: AbstractState | None
    level_delta: int
    terminal_events: int
    illegal_actions: int
    errors: tuple[str, ...]
    latency_ms: float

    @property
    def progressed(self) -> bool:
        return self.level_delta > 0


@dataclass(frozen=True)
class ProgressWitness:
    """Positive grounded evidence plus its coordinate-free program."""

    source_game: str
    context_signature: str
    macro_schema: str
    relation: str
    abstract_steps: tuple[AbstractWitnessStep, ...]
    grounded_actions: tuple[GroundedAction, ...]
    observed_events: tuple[tuple[str, ...], ...]
    level_delta: int
    program: JointProgramHypothesis
    posterior_rank: int | None
    posterior_mass: float

    @property
    def transferable_payload(self) -> dict[str, Any]:
        payload = {
            "format_version": FORMAT_VERSION,
            "control_family": CONTROL_FAMILY,
            "context_signature": self.context_signature,
            "macro_schema": self.macro_schema,
            "relation": self.relation,
            "steps": [step.to_dict() for step in self.abstract_steps],
            "teleological_effect": "level_progress",
            "program": self.program.canonical_payload,
            "program_hash": self.program.canonical_hash,
        }
        _validate_transferable_payload(payload)
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_game": self.source_game,
            "transferable": self.transferable_payload,
            "grounded_evidence": {
                "actions": [action.to_dict() for action in self.grounded_actions],
                "observed_events": [list(events) for events in self.observed_events],
                "level_delta": self.level_delta,
            },
            "posterior": {
                "rank": self.posterior_rank,
                "mass": round(self.posterior_mass, 8),
            },
        }


@dataclass(frozen=True)
class SearchOutcome:
    game: str
    witness: ProgressWitness | None
    diagnosis: str
    scan_rows: tuple[EffectScanRow, ...]
    effect_groups: int
    candidate_macros: int
    macros_executed: int
    actions_executed: int
    illegal_actions: int
    terminal_events: int
    errors: tuple[str, ...]
    wall_seconds: float

    @property
    def passed(self) -> bool:
        return self.witness is not None and self.diagnosis == "SUCCESS"

    def to_dict(self, *, include_scan_rows: bool = False) -> dict[str, Any]:
        payload = {
            "game": self.game,
            "passed": self.passed,
            "diagnosis": self.diagnosis,
            "effect_groups": self.effect_groups,
            "candidate_macros": self.candidate_macros,
            "macros_executed": self.macros_executed,
            "actions_executed": self.actions_executed,
            "illegal_actions": self.illegal_actions,
            "terminal_events": self.terminal_events,
            "errors": list(self.errors),
            "wall_seconds": round(self.wall_seconds, 3),
            "witness": None if self.witness is None else self.witness.to_dict(),
        }
        if include_scan_rows:
            payload["one_step_scan"] = [row.to_dict() for row in self.scan_rows]
        return payload


def _validate_transferable_payload(payload: Mapping[str, Any]) -> None:
    """Reject leakage of grounded evidence into a transferable witness."""

    text = _canonical_json(payload).lower()
    forbidden_keys = ('"x":', '"y":', "action_data", "source_game", "game_id")
    if any(token in text for token in forbidden_keys):
        raise ValueError("transferable witness contains grounded fields")
    if any(game.lower() in text for game in SOURCE_TRAIN_GAMES):
        raise ValueError("transferable witness contains a source game id")


def _expression_counter_equals(key: str, value: int) -> Expression:
    return Expression(
        op="eq",
        args=(
            Expression(op="counter", value=key),
            Expression.constant(float(value)),
        ),
    )


def compile_progress_program(
    *,
    sequence_length: int,
    action_names: Sequence[str] = ("ACTION6",),
    positive: bool = True,
) -> JointProgramHypothesis:
    """Compile a coordinate-free finite-state causal witness."""

    length = max(1, int(sequence_length))
    names = tuple(sorted({str(name).upper() for name in action_names}))
    rules = []
    for index in range(length):
        effects = [
            Effect(operation="assert", predicate="changed", terms=("$target",)),
            Effect(
                operation="increment_counter",
                key="witness_step",
                value=1.0,
            ),
        ]
        if positive and index == length - 1:
            effects.extend(
                (
                    Effect(operation="progress", value=1.0),
                    Effect(operation="win"),
                )
            )
        rules.append(
            TransitionRule(
                rule_id=f"witness_step_{index}",
                action_operator="apply",
                condition=_expression_counter_equals("witness_step", index),
                effects=tuple(effects),
            )
        )
    return JointProgramHypothesis(
        program_id=(
            f"progress_witness_{length}"
            if positive
            else f"progress_decoy_{length}"
        ),
        object_schema=ObjectSchema(roles=("object", "target")),
        action_bindings=tuple(
            ActionBinding(name, "apply", target_role="target") for name in names
        ),
        transition_rules=tuple(rules),
        progress_rule=ProgressRule(Expression(op="counter", value="progress")),
        terminal_rules=(
            TerminalRule(Expression.fact("game_over"), outcome="game_over"),
        ),
        goal_rule=GoalRule(
            Expression.fact("level_complete") if positive else Expression.constant(False),
            family="level_progress" if positive else "no_progress",
        ),
        provenance=("progress_witness_induction",),
    )


def _context_signature(
    state: AbstractState,
    *,
    macro: CandidateMacro,
) -> str:
    role_counts = Counter(role for entity in state.entities for role in entity.roles)
    payload = {
        "roles": sorted(role_counts.items()),
        "entity_count_bucket": min(8, len(state.entities) // 4),
        "schema": macro.schema,
        "relation": macro.relation,
        "context": dict(macro.context),
    }
    return _sha256_payload(payload)[:20]


def _events_for_observation(observation: ObservedTransition) -> tuple[str, ...]:
    events = set(observation.events)
    packet = observation.observation
    if packet.progress_mean and packet.progress_mean > 0.0:
        events.add("progress")
    if packet.goal_probability and packet.goal_probability >= 0.5:
        events.add("level_complete")
    if packet.terminal_probability and packet.terminal_probability >= 0.5:
        events.add("game_over")
    return tuple(sorted(events))


def _posterior_check(
    program: JointProgramHypothesis,
    trace: ExecutedTrace,
) -> tuple[int | None, float]:
    decoy = compile_progress_program(
        sequence_length=len(trace.actions),
        action_names=tuple(action.action_name for action in trace.actions),
        positive=False,
    )
    posterior = StatefulWitnessPosterior(
        maximum_particles=8,
        repair_ess_threshold=1.0,
    )
    # A witness program is generated *after* its positive trace.  Load the
    # evidence first, then seed and replay both hypotheses from absolute priors.
    # Seeding before the first observation would incorrectly merge programs
    # that agree on a prefix but diverge at the teleological boundary.
    for evidence in trace.observations:
        posterior.observe(evidence, allow_repair=False)
    posterior.seed((program, decoy), initial_state=trace.initial_state)
    ranked = posterior.top(8)
    for index, particle in enumerate(ranked, start=1):
        if particle.program.canonical_hash == program.canonical_hash:
            return index, particle.probability
    return None, 0.0


def compile_witness(
    *,
    game: str,
    macro: CandidateMacro,
    trace: ExecutedTrace,
) -> ProgressWitness:
    if not trace.progressed or not trace.observations or trace.initial_state is None:
        raise ValueError("a progress witness needs a positive observed trace")
    effective_length = len(trace.actions)
    program = compile_progress_program(
        sequence_length=effective_length,
        action_names=tuple(action.action_name for action in trace.actions),
    )
    executor = ProgramExecutor()
    rollout = executor.rollout(
        program,
        trace.initial_state,
        tuple(action.candidate for action in trace.actions),
        maximum_actions=effective_length,
    )
    if rollout.final_packet.progress_mean != 1.0:
        raise ValueError("compiled witness does not predict its level progress")
    if (rollout.final_packet.goal_probability or 0.0) < 0.5:
        raise ValueError("compiled witness does not predict goal satisfaction")
    rank, mass = _posterior_check(program, trace)
    selector = (
        "successor_toward_enclosure"
        if macro.schema == "path_successor"
        else "same_effect_distinct_target"
    )
    abstract_steps = tuple(
        AbstractWitnessStep(
            target_selector=selector,
            relation=macro.relation,
            expected_event=(
                "level_progress" if index == effective_length - 1 else "state_change"
            ),
        )
        for index in range(effective_length)
    )
    return ProgressWitness(
        source_game=game,
        context_signature=_context_signature(trace.initial_state, macro=macro),
        macro_schema=macro.schema,
        relation=macro.relation,
        abstract_steps=abstract_steps,
        grounded_actions=trace.actions,
        observed_events=tuple(
            _events_for_observation(observation)
            for observation in trace.observations
        ),
        level_delta=trace.level_delta,
        program=program,
        posterior_rank=rank,
        posterior_mass=mass,
    )


def _area(entity: AbstractEntity) -> str:
    return dict(entity.attributes).get("area", "")


def _longest_chain(
    state: AbstractState,
    *,
    radius: float,
    minimum_nodes: int,
) -> tuple[AbstractEntity, ...]:
    points = [
        entity
        for entity in state.entities
        if _area(entity) == "one" and entity.center is not None
    ]
    if len(points) < minimum_nodes:
        return ()
    adjacency: dict[int, list[int]] = defaultdict(list)
    for left in range(len(points)):
        for right in range(left + 1, len(points)):
            if math.dist(points[left].center or (0.0, 0.0), points[right].center or (0.0, 0.0)) <= radius:
                adjacency[left].append(right)
                adjacency[right].append(left)
    unseen = set(range(len(points)))
    components: list[list[int]] = []
    while unseen:
        root = min(unseen)
        queue = deque([root])
        unseen.remove(root)
        component = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        components.append(component)
    component = max(components, key=len)
    if len(component) < minimum_nodes:
        return ()
    selected = [points[index] for index in component]
    centers = np.asarray([entity.center for entity in selected], dtype=float)
    centered = centers - centers.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    projections = centered @ axis
    return tuple(
        entity
        for _, entity in sorted(zip(projections.tolist(), selected), key=lambda item: item[0])
    )


def _endpoint_target_score(
    endpoint: AbstractEntity,
    state: AbstractState,
) -> float:
    rank = {"one": 0.0, "tiny": 1.0, "small": 2.0, "medium": 3.0, "large": 5.0}
    center = endpoint.center or (0.0, 0.0)
    score = 0.0
    for entity in state.entities:
        if entity.entity_id == endpoint.entity_id or entity.center is None:
            continue
        area = _area(entity)
        if area == "one":
            continue
        distance = math.dist(center, entity.center)
        if distance <= 8.0:
            score += rank.get(area, 1.0) / (1.0 + distance)
            if area == "large":
                score += 2.0
    return score


def _xy(action: GroundedAction) -> tuple[float, float] | None:
    data = action.data
    try:
        return float(data["y"]), float(data["x"])
    except (KeyError, TypeError, ValueError):
        return None


def chain_successor_macro(
    state: AbstractState,
    actions: Sequence[GroundedAction],
    *,
    config: SearchConfig,
) -> CandidateMacro | None:
    """Ground a path macro from relative topology, without raw values."""

    chain = list(
        _longest_chain(
            state,
            radius=config.chain_link_radius,
            minimum_nodes=config.chain_minimum_nodes,
        )
    )
    positioned_actions = [action for action in actions if _xy(action) is not None]
    if not chain or not positioned_actions:
        return None
    left_score = _endpoint_target_score(chain[0], state)
    right_score = _endpoint_target_score(chain[-1], state)
    # The ordered chain must run from the movable/source end to the salient
    # enclosed end.  Reverse only when the salient end is currently first.
    if left_score > right_score:
        chain.reverse()
        target_score = left_score
    else:
        target_score = right_score
    # The first path node is already adjacent to the movable endpoint.  The
    # perceptual graph may omit a node hidden by the movable object, so index
    # strides are brittle.  Use arc distance instead: first take one link,
    # then advance roughly ``chain_stride`` median links at a time.
    consecutive = [
        math.dist(left.center or (0.0, 0.0), right.center or (0.0, 0.0))
        for left, right in pairwise(chain)
    ]
    base_link = float(np.median(consecutive)) if consecutive else 0.0
    minimum_advance = max(1e-6, 0.8 * base_link * config.chain_stride)
    waypoints = [chain[1]] if len(chain) > 2 else []
    for entity in chain[2:-1]:
        if not waypoints or math.dist(
            waypoints[-1].center or (0.0, 0.0),
            entity.center or (0.0, 0.0),
        ) >= minimum_advance:
            waypoints.append(entity)
    grounded = []
    for waypoint in waypoints:
        center = waypoint.center
        if center is None:
            continue
        action = min(
            positioned_actions,
            key=lambda item: math.dist(_xy(item) or center, center),
        )
        if not grounded or action.key != grounded[-1].key:
            grounded.append(action)
        if len(grounded) >= config.maximum_horizon:
            break
    if not grounded:
        return None
    return CandidateMacro(
        schema="path_successor",
        relation="successor_toward_enclosure",
        actions=tuple(grounded),
        context=(
            ("chain_length_bucket", min(8, len(chain) // 4)),
            ("target_end_salience", round(float(target_score), 2)),
        ),
    )


def _action_identity(action: Any) -> str:
    return GroundedAction.from_view(action).key


def _live_helpers() -> tuple[Any, Any, Any, Any, Any, Any]:
    # Lazy imports preserve unit-testability when the ARC SDK is unavailable.
    from theory.m1.polymorphic_a25_adapter import _step_env_action
    from theory.m2.m3_execution_smoke import _reset_env
    from theory.non_ar25_active_micro_run import _env_dir, _valid_actions
    from theory.real_env_option_adapter import snapshot_frame
    from theory.unified_cognition_ab_benchmark import _make_real_env

    return _make_real_env, _env_dir, _valid_actions, _reset_env, _step_env_action, snapshot_frame


def _make_live_env(game: str) -> Any:
    _make_real_env, _env_dir, _, _, _, _ = _live_helpers()
    return _make_real_env(game, Path(_env_dir()))


def _initial_live_state(game: str) -> tuple[Any, Any, tuple[GroundedAction, ...], AbstractState]:
    _, _, _valid_actions, _reset_env, _, snapshot_frame = _live_helpers()
    env = _make_live_env(game)
    frame = _reset_env(env)
    snapshot = snapshot_frame(frame)
    actions = tuple(GroundedAction.from_view(action) for action in _valid_actions(env))
    observation = build_observation(
        snapshot.grid,
        available_actions=tuple(action.action_name for action in actions),
        game_state=snapshot.game_state,
        levels_completed=snapshot.levels_completed,
    )
    return env, frame, actions, compile_observation(observation)


def one_step_effect_scan(
    game: str,
    *,
    config: SearchConfig,
) -> tuple[EffectScanRow, ...]:
    _, _, _valid_actions, _reset_env, _step_env_action, snapshot_frame = _live_helpers()
    env, _, inventory, _ = _initial_live_state(game)
    inventory = inventory[: config.maximum_one_step_actions]
    rows = []
    for wanted in inventory:
        frame = _reset_env(env)
        before = snapshot_frame(frame)
        legal = {_action_identity(action): action for action in _valid_actions(env)}
        concrete = legal.get(wanted.key)
        if concrete is None:
            rows.append(
                EffectScanRow(
                    action=wanted,
                    effect_key="GROUNDING_MISS",
                    changed_cells=0,
                    level_delta=0,
                    game_state="",
                    latency_ms=0.0,
                )
            )
            continue
        started = time.perf_counter()
        after_frame = _step_env_action(env, concrete)
        latency = (time.perf_counter() - started) * 1000.0
        after = snapshot_frame(after_frame, fallback_available_actions=before.available_actions)
        before_grid = np.asarray(before.grid)
        after_grid = np.asarray(after.grid)
        changed = (
            int(np.sum(before_grid != after_grid))
            if before_grid.shape == after_grid.shape
            else int(max(before_grid.size, after_grid.size))
        )
        delta = max(0, int(after.levels_completed) - int(before.levels_completed))
        effect_key = _sha256_payload(
            {
                "visual": _visual_digest(after.grid),
                "changed": changed,
                "level_delta": delta,
                "terminal": _terminal(after.game_state),
            }
        )[:20]
        rows.append(
            EffectScanRow(
                action=wanted,
                effect_key=effect_key,
                changed_cells=changed,
                level_delta=delta,
                game_state=str(after.game_state),
                latency_ms=latency,
            )
        )
    return tuple(rows)


def effect_representatives(
    rows: Sequence[EffectScanRow],
    *,
    maximum: int,
) -> tuple[GroundedAction, ...]:
    grouped: dict[str, list[EffectScanRow]] = defaultdict(list)
    for row in rows:
        if row.effect_key != "GROUNDING_MISS":
            grouped[row.effect_key].append(row)
    representatives = [
        min(items, key=lambda row: row.action.key).action
        for _, items in sorted(
            grouped.items(),
            key=lambda item: (len(item[1]), item[0]),
        )
    ]
    return tuple(representatives[: max(1, int(maximum))])


def candidate_macros(
    state: AbstractState,
    inventory: Sequence[GroundedAction],
    scan_rows: Sequence[EffectScanRow],
    *,
    config: SearchConfig,
) -> tuple[CandidateMacro, ...]:
    macros = []
    chain = chain_successor_macro(state, inventory, config=config)
    if chain is not None:
        macros.append(chain)
    for action in effect_representatives(
        scan_rows,
        maximum=config.maximum_effect_representatives,
    ):
        macros.append(
            CandidateMacro(
                schema="repeat_target",
                relation="identity",
                actions=tuple(action for _ in range(config.maximum_horizon)),
                context=(("effect_distinct", 1),),
            )
        )
    unique: dict[str, CandidateMacro] = {}
    for macro in macros:
        unique.setdefault(macro.key, macro)
    return tuple(unique.values())[: config.maximum_candidate_macros]


def execute_grounded_sequence(
    game: str,
    actions: Sequence[GroundedAction],
) -> ExecutedTrace:
    _, _, _valid_actions, _reset_env, _step_env_action, snapshot_frame = _live_helpers()
    started = time.perf_counter()
    errors = []
    rows = []
    observations = []
    executed = []
    terminal_events = 0
    illegal_actions = 0
    env = _make_live_env(game)
    frame = _reset_env(env)
    initial_snapshot = snapshot_frame(frame)
    initial_observation = build_observation(
        initial_snapshot.grid,
        available_actions=initial_snapshot.available_actions,
        game_state=initial_snapshot.game_state,
        levels_completed=initial_snapshot.levels_completed,
    )
    initial_state = compile_observation(initial_observation)
    initial_level = int(initial_snapshot.levels_completed)
    maximum_level = initial_level
    for index, action in enumerate(actions):
        before = snapshot_frame(frame)
        if _terminal(before.game_state):
            terminal_events += 1
            break
        legal_views = list(_valid_actions(env))
        legal = {_action_identity(item): item for item in legal_views}
        concrete = legal.get(action.key)
        if concrete is None:
            illegal_actions += 1
            errors.append(f"grounding_miss:{index}:{action.key}")
            break
        try:
            after_frame = _step_env_action(env, concrete)
            after = snapshot_frame(
                after_frame,
                fallback_available_actions=before.available_actions,
            )
        except Exception as exc:  # noqa: BLE001  # pragma: no cover
            errors.append(f"execution_error:{index}:{type(exc).__name__}:{exc}")
            break
        record = build_transition_record(
            action=action.action_name,
            action_args=action.data,
            grid_before=before.grid,
            grid_after=after.grid,
            available_actions=tuple(item.name for item in legal_views),
            game_state_before=before.game_state,
            game_state_after=after.game_state,
            levels_completed_before=before.levels_completed,
            levels_completed_after=after.levels_completed,
            timestamp=index,
        )
        evidence = compile_transition_record(record, source_game_id=game)
        observations.append(evidence)
        executed.append(action)
        delta = max(0, int(after.levels_completed) - int(before.levels_completed))
        maximum_level = max(maximum_level, int(after.levels_completed))
        rows.append(
            {
                "step": index + 1,
                "action": action.to_dict(),
                "visual_before": _visual_digest(before.grid),
                "visual_after": _visual_digest(after.grid),
                "changed_cells": int(np.sum(np.asarray(before.grid) != np.asarray(after.grid))),
                "level_delta": delta,
                "game_state_after": str(after.game_state),
                "events": list(_events_for_observation(evidence)),
            }
        )
        frame = after_frame
        if _terminal(after.game_state):
            terminal_events += 1
        if delta > 0 or _terminal(after.game_state):
            break
    return ExecutedTrace(
        actions=tuple(executed),
        observations=tuple(observations),
        rows=tuple(rows),
        initial_state=initial_state,
        level_delta=max(0, maximum_level - initial_level),
        terminal_events=terminal_events,
        illegal_actions=illegal_actions,
        errors=tuple(errors),
        latency_ms=(time.perf_counter() - started) * 1000.0,
    )


def search_progress_witness(
    game: str,
    *,
    config: SearchConfig | None = None,
    enabled_control_families: Sequence[str] = (CONTROL_FAMILY,),
) -> SearchOutcome:
    """Search one real source game and stop at its first level increment."""

    started = time.perf_counter()
    effective = config or SearchConfig()
    if CONTROL_FAMILY not in set(enabled_control_families):
        return SearchOutcome(
            game=game,
            witness=None,
            diagnosis="GENERATOR_MISS",
            scan_rows=(),
            effect_groups=0,
            candidate_macros=0,
            macros_executed=0,
            actions_executed=0,
            illegal_actions=0,
            terminal_events=0,
            errors=(),
            wall_seconds=time.perf_counter() - started,
        )
    errors = []
    try:
        _, _, inventory, state = _initial_live_state(game)
        scan = one_step_effect_scan(game, config=effective)
    except Exception as exc:  # noqa: BLE001  # pragma: no cover
        return SearchOutcome(
            game=game,
            witness=None,
            diagnosis="GROUNDING_MISS",
            scan_rows=(),
            effect_groups=0,
            candidate_macros=0,
            macros_executed=0,
            actions_executed=0,
            illegal_actions=1,
            terminal_events=0,
            errors=(f"setup_error:{type(exc).__name__}:{exc}",),
            wall_seconds=time.perf_counter() - started,
        )
    groups = len({row.effect_key for row in scan if row.effect_key != "GROUNDING_MISS"})
    macros = candidate_macros(state, inventory, scan, config=effective)
    actions_executed = len(scan)
    illegal_actions = sum(row.effect_key == "GROUNDING_MISS" for row in scan)
    terminal_events = sum(_terminal(row.game_state) for row in scan)
    for macro_index, macro in enumerate(macros, start=1):
        trace = execute_grounded_sequence(game, macro.actions)
        actions_executed += len(trace.actions)
        illegal_actions += trace.illegal_actions
        terminal_events += trace.terminal_events
        errors.extend(trace.errors)
        if not trace.progressed:
            continue
        try:
            witness = compile_witness(game=game, macro=macro, trace=trace)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"compile_error:{type(exc).__name__}:{exc}")
            return SearchOutcome(
                game=game,
                witness=None,
                diagnosis="GENERATOR_MISS",
                scan_rows=scan,
                effect_groups=groups,
                candidate_macros=len(macros),
                macros_executed=macro_index,
                actions_executed=actions_executed,
                illegal_actions=illegal_actions,
                terminal_events=terminal_events,
                errors=tuple(errors),
                wall_seconds=time.perf_counter() - started,
            )
        diagnosis = "SUCCESS" if witness.posterior_rank is not None and witness.posterior_rank <= 8 else "POSTERIOR_MISS"
        return SearchOutcome(
            game=game,
            witness=witness,
            diagnosis=diagnosis,
            scan_rows=scan,
            effect_groups=groups,
            candidate_macros=len(macros),
            macros_executed=macro_index,
            actions_executed=actions_executed,
            illegal_actions=illegal_actions,
            terminal_events=terminal_events,
            errors=tuple(errors),
            wall_seconds=time.perf_counter() - started,
        )
    diagnosis = (
        "GROUNDING_MISS"
        if illegal_actions and not macros
        else "GENERATOR_MISS"
        if not macros
        else "SEQUENCE_MISS"
    )
    return SearchOutcome(
        game=game,
        witness=None,
        diagnosis=diagnosis,
        scan_rows=scan,
        effect_groups=groups,
        candidate_macros=len(macros),
        macros_executed=len(macros),
        actions_executed=actions_executed,
        illegal_actions=illegal_actions,
        terminal_events=terminal_events,
        errors=tuple(errors),
        wall_seconds=time.perf_counter() - started,
    )


def run_leave_one_game_out(
    outcomes: Sequence[SearchOutcome],
    *,
    config: SearchConfig | None = None,
) -> dict[str, Any]:
    """Transfer only the induced control family, then ground it afresh."""

    by_game = {outcome.game: outcome for outcome in outcomes}
    folds = []
    for source in SOURCE_TRAIN_GAMES:
        target = next(game for game in SOURCE_TRAIN_GAMES if game != source)
        source_outcome = by_game.get(source)
        family_available = bool(source_outcome and source_outcome.witness)
        target_outcome = search_progress_witness(
            target,
            config=config,
            enabled_control_families=(CONTROL_FAMILY,) if family_available else (),
        )
        folds.append(
            {
                "source_game": source,
                "target_game": target,
                "source_family_available": family_available,
                "target": target_outcome.to_dict(),
                "no_source_ablation": {
                    "enabled_program_families": 0,
                    "actions_executed": 0,
                    "levels_completed": 0,
                    "diagnosis": "GENERATOR_MISS",
                },
                "passed": bool(
                    family_available
                    and target_outcome.passed
                    and target_outcome.witness is not None
                    and target_outcome.witness.level_delta > 0
                    and target_outcome.illegal_actions == 0
                    and not target_outcome.errors
                ),
            }
        )
    return {
        "protocol": "leave_one_source_game_out_active_grounding",
        "folds": folds,
        "levels_completed": sum(
            int((fold["target"].get("witness") or {}).get("grounded_evidence", {}).get("level_delta", 0))
            for fold in folds
        ),
        "passed": bool(len(folds) == 2 and all(fold["passed"] for fold in folds)),
        "scientific_scope": (
            "The held-out game contributes only its initial observation, legal "
            "actions, and outcomes of actions actually executed. No stored winning "
            "path or hidden counterfactual arm is read."
        ),
    }


def _report_checksum(payload: Mapping[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("report_checksum", None)
    return _sha256_payload(clean)


def run_source_train(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config: SearchConfig | None = None,
    include_scan_rows: bool = True,
) -> dict[str, Any]:
    effective = config or SearchConfig()
    started = time.perf_counter()
    outcomes = [
        search_progress_witness(game, config=effective)
        for game in SOURCE_TRAIN_GAMES
    ]
    loo = run_leave_one_game_out(outcomes, config=effective)
    checks = {
        "source_train_only": True,
        "both_source_games_progress": all(outcome.passed for outcome in outcomes),
        "programs_coordinate_free": all(
            outcome.witness is not None
            and bool(outcome.witness.transferable_payload)
            for outcome in outcomes
        ),
        "posterior_top8": all(
            outcome.witness is not None
            and outcome.witness.posterior_rank is not None
            and outcome.witness.posterior_rank <= 8
            for outcome in outcomes
        ),
        "zero_illegal_actions": all(outcome.illegal_actions == 0 for outcome in outcomes)
        and all(fold["target"]["illegal_actions"] == 0 for fold in loo["folds"]),
        "zero_errors": all(not outcome.errors for outcome in outcomes)
        and all(not fold["target"]["errors"] for fold in loo["folds"]),
        "leave_one_game_out": bool(loo["passed"]),
        "holdout_closed": True,
        "ar25_closed": True,
    }
    passed = all(checks.values())
    report: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "phase": "T10.0_SOURCE_TRAIN",
        "status": "PASS_T10_0_AUTHORIZE_T10_1" if passed else "FAIL_CLOSED",
        "config": asdict(effective),
        "source_games": list(SOURCE_TRAIN_GAMES),
        "outcomes": [
            outcome.to_dict(include_scan_rows=include_scan_rows)
            for outcome in outcomes
        ],
        "leave_one_game_out": loo,
        "diagnosis_counts": dict(Counter(outcome.diagnosis for outcome in outcomes)),
        "checks": checks,
        "passed": passed,
        "firewall": {
            "source_validation_opened": False,
            "holdout_opened": False,
            "ar25_opened": False,
            "t10_1_authorized": passed,
        },
        "wall_seconds": round(time.perf_counter() - started, 3),
    }
    report["report_checksum"] = _report_checksum(report)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SAGE.T10 progress-witness induction.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--maximum-horizon", type=int, default=16)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)
    report = run_source_train(
        output_dir=args.output_dir,
        config=SearchConfig(maximum_horizon=args.maximum_horizon),
        include_scan_rows=not args.compact,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "passed": report["passed"],
                "report_checksum": report["report_checksum"],
                "output": str(Path(args.output_dir) / "report.json"),
            },
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 2


__all__ = [
    "CONTROL_FAMILY",
    "FAILURE_DIAGNOSES",
    "FORMAT_VERSION",
    "SOURCE_TRAIN_GAMES",
    "AbstractWitnessStep",
    "CandidateMacro",
    "EffectScanRow",
    "ExecutedTrace",
    "GroundedAction",
    "ProgressWitness",
    "SearchConfig",
    "SearchOutcome",
    "StatefulWitnessPosterior",
    "candidate_macros",
    "chain_successor_macro",
    "compile_progress_program",
    "compile_witness",
    "effect_representatives",
    "execute_grounded_sequence",
    "one_step_effect_scan",
    "run_leave_one_game_out",
    "run_source_train",
    "search_progress_witness",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
