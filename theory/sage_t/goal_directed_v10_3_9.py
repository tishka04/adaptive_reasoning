"""Causal-subgoal composition controller for SAGE.T10.3.9.

T10.3.8 established blank-posterior discovery and fresh reproduction on the
two core games, but its sequence panel exposed two independent defects:

* an effect descriptor could raise ``ValueError`` while unpacking an unusual
  actor displacement; and
* mixed discovery eventually became a fixed, seed-insensitive cycle over the
  legal action names.

This module keeps the existing SAGE.T authority and registry contracts.  It
replaces only the online option inducer and the discovery policy.  The new
inducer builds an ephemeral action/effect graph, probes every legal schema in a
seed-diversified order, and composes a bounded mixed option toward under-seen
effect modes.  Level progress remains the only success credit.  Coordinates,
colors, entity identities, raw grids, and the exploration seed never enter a
transferable program.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from typing import Any

from v3.schemas import TransitionRecord

from .contracts import AbstractState, ActionCandidate
from .goal_directed_v10_3_2 import (
    MAXIMUM_OPTION_HORIZON,
    GoalDirectedOption,
    OptionStep,
)
from .goal_directed_v10_3_3 import UNIQUE_ACTION_SCHEMA
from .goal_directed_v10_3_6 import (
    BALANCED_CAUSAL_BINDING,
)
from .goal_directed_v10_3_7 import StableFreshPathSageTController

FORMAT_VERSION = "sage-t10.3.9-causal-subgoal-composition-v1"
CAUSAL_FRONTIER_SOURCE = "reset_local_causal_subgoal_frontier"
CAUSAL_PROBE_SOURCE = "seed_diversified_causal_schema_probe"
REPRODUCTION_SOURCE = "fresh_mixed_registry_reproduction"
PROBES_PER_ACTION_SCHEMA = 2
MINIMUM_MIXED_HORIZON = 16


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _bounded_len(value: Any, maximum: int) -> int:
    try:
        return min(maximum, len(value))
    except (TypeError, ValueError):
        return 0


def _safe_axis(displacement: Any) -> str:
    """Describe actor motion without assuming an exact tuple arity.

    The frozen T10.3.8 descriptor used ``dy, dx = displacement``.  A valid
    transition carrying an extended displacement payload therefore aborted
    observation on RE86.  Only a coarse axis is needed by the transferable
    effect contract, so malformed or extended payloads are safely classified.
    """

    if displacement is None:
        return "none"
    try:
        values = tuple(displacement)
    except (TypeError, ValueError):
        return "unknown"
    if len(values) < 2:
        return "unknown"
    try:
        dy = int(values[-2])
        dx = int(values[-1])
    except (TypeError, ValueError, OverflowError):
        return "unknown"
    if abs(dy) > abs(dx):
        return "vertical"
    if abs(dx) > 0:
        return "horizontal"
    return "none"


def robust_effect_descriptor(record: Any) -> dict[str, Any]:
    """Return a bounded, identity-free transition descriptor.

    This function is deliberately total for structurally valid transition
    objects.  It never serializes the frame, colors, coordinates, or object
    identities; collection persists only its digest in option steps.
    """

    diff = record.diff
    progressed = bool(
        getattr(diff, "level_complete", False)
        or int(getattr(record.obs_after, "levels_completed", 0))
        > int(getattr(record.obs_before, "levels_completed", 0))
    )
    terminal = bool(getattr(diff, "game_over", False))
    noop = bool(getattr(diff, "is_noop", False))
    moved = _bounded_len(getattr(diff, "moved_objects", ()), 4)
    created = _bounded_len(getattr(diff, "created_objects", ()), 4)
    removed = _bounded_len(getattr(diff, "removed_objects", ()), 4)
    try:
        changed = min(8, max(0, int(getattr(diff, "num_changed", 0))) // 4)
    except (TypeError, ValueError, OverflowError):
        changed = 0
    if progressed:
        mode = "level_progress"
    elif terminal:
        mode = "terminal"
    elif noop:
        mode = "noop"
    elif created or removed:
        mode = "object_set_change"
    elif moved or _safe_axis(getattr(diff, "player_displacement", None)) != "none":
        mode = "motion"
    else:
        mode = "structural_change"
    return {
        "mode": mode,
        "noop": noop,
        "terminal": terminal,
        "level_progress": progressed,
        "moved_bucket": moved,
        "created_bucket": created,
        "removed_bucket": removed,
        "changed_bucket": changed,
        "actor_axis": _safe_axis(getattr(diff, "player_displacement", None)),
    }


def robust_effect_signature(record: Any) -> str:
    return _sha(robust_effect_descriptor(record))


class CausalSubgoalAutomatonInducer:
    """Build a reset-local causal effect graph and compose novelty frontiers."""

    def __init__(self) -> None:
        self._branch: deque[tuple[OptionStep, str, bool]] = deque(
            maxlen=MAXIMUM_OPTION_HORIZON
        )
        self._effects_by_action: dict[str, Counter[str]] = defaultdict(Counter)
        self._modes_by_action: dict[str, Counter[str]] = defaultdict(Counter)
        self._transitions: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
        self._effect_visits: Counter[str] = Counter()
        self._action_uses: Counter[str] = Counter()
        self._terminal_uses: Counter[str] = Counter()
        self._noop_uses: Counter[str] = Counter()
        self._last_effect = "branch_start"
        self._successful_options: list[GoalDirectedOption] = []
        self._observation_rejections = 0

    def start_branch(self) -> None:
        self._branch.clear()
        self._last_effect = "branch_start"

    @property
    def action_uses(self) -> Mapping[str, int]:
        return dict(self._action_uses)

    def observe(
        self,
        record: TransitionRecord,
        *,
        selected_step: OptionStep | None,
        active_option: GoalDirectedOption | None,
    ) -> GoalDirectedOption | None:
        action_name = str(record.action.name).strip().upper()
        if not action_name.startswith("ACTION"):
            self._observation_rejections += 1
            return None
        descriptor = robust_effect_descriptor(record)
        effect = _sha(descriptor)
        try:
            base_step = selected_step or OptionStep(action_name=action_name)
            step = OptionStep(
                action_name=base_step.action_name,
                binding_method=base_step.binding_method,
                structural_signature=base_step.structural_signature,
                expected_effect=effect,
            )
        except (TypeError, ValueError):
            self._observation_rejections += 1
            return None
        productive = not descriptor["noop"] and not descriptor["terminal"]
        self._branch.append((step, effect, productive))
        self._action_uses[action_name] += 1
        self._effects_by_action[action_name][effect] += 1
        self._modes_by_action[action_name][str(descriptor["mode"])] += 1
        self._transitions[self._last_effect][(action_name, effect)] += 1
        self._effect_visits[effect] += 1
        if descriptor["terminal"]:
            self._terminal_uses[action_name] += 1
        if descriptor["noop"]:
            self._noop_uses[action_name] += 1
        self._last_effect = effect

        if not descriptor["level_progress"]:
            return None
        if active_option is not None:
            observed = tuple(self._branch)
            learned_steps = tuple(
                OptionStep(
                    action_name=item.action_name,
                    binding_method=item.binding_method,
                    structural_signature=item.structural_signature,
                    expected_effect=(
                        observed[index][1]
                        if index < len(observed)
                        else item.expected_effect
                    ),
                )
                for index, item in enumerate(active_option.steps)
            )
            learned = GoalDirectedOption(
                schema=(
                    "mixed_automaton"
                    if len({item.action_name for item in learned_steps}) >= 2
                    else active_option.schema
                ),
                steps=learned_steps,
                source="causal_subgoal_level_reproduction",
            )
        else:
            suffix = self._causal_suffix()
            learned = GoalDirectedOption(
                schema=(
                    "mixed_automaton"
                    if len({item.action_name for item in suffix}) >= 2
                    else "repeat_target"
                ),
                steps=suffix,
                source="causal_subgoal_level_induction",
            )
        self._successful_options.append(learned)
        return learned

    def _causal_suffix(self) -> tuple[OptionStep, ...]:
        rows = tuple(self._branch)
        if not rows:
            return (OptionStep("ACTION1"),)
        start = 0
        for index, (_, _, productive) in enumerate(rows):
            if productive:
                start = index
                break
        suffix = tuple(row[0] for row in rows[start:])
        return suffix or (rows[-1][0],)

    def _predicted_effect(self, action_name: str) -> str:
        effects = self._effects_by_action.get(action_name)
        if not effects:
            return "causal_frontier_unseen"
        return min(
            effects,
            key=lambda token: (self._effect_visits[token], -effects[token], token),
        )

    def compose_frontier(
        self,
        legal_action_names: Sequence[str],
        *,
        rotation: int,
        horizon: int = MINIMUM_MIXED_HORIZON,
    ) -> GoalDirectedOption | None:
        legal = tuple(
            sorted(
                {
                    str(item).strip().upper()
                    for item in legal_action_names
                    if str(item).strip().upper().startswith("ACTION")
                }
            )
        )
        if len(legal) < 2:
            return None
        offset = int(rotation) % len(legal)
        rotated = legal[offset:] + legal[:offset]
        length = max(2, min(MAXIMUM_OPTION_HORIZON, int(horizon)))
        current = self._last_effect
        steps: list[OptionStep] = []
        for index in range(length):
            ranked = sorted(
                rotated,
                key=lambda action: (
                    self._transitions[current][(action, self._predicted_effect(action))],
                    self._terminal_uses[action],
                    self._noop_uses[action],
                    self._action_uses[action],
                    -len(self._effects_by_action[action]),
                    (rotated.index(action) - index) % len(rotated),
                ),
            )
            action = ranked[0]
            effect = self._predicted_effect(action)
            steps.append(
                OptionStep(
                    action,
                    binding_method=UNIQUE_ACTION_SCHEMA,
                    expected_effect=effect,
                )
            )
            current = effect
        if len({step.action_name for step in steps}) < 2:
            alternate = next(action for action in rotated if action != steps[0].action_name)
            steps[1] = OptionStep(
                alternate,
                binding_method=UNIQUE_ACTION_SCHEMA,
                expected_effect=self._predicted_effect(alternate),
            )
        return GoalDirectedOption(
            schema="mixed_automaton",
            steps=tuple(steps),
            source=CAUSAL_FRONTIER_SOURCE,
        )

    def mixed_candidates(
        self,
        legal_action_names: Sequence[str],
        *,
        subgoal_action_names: Sequence[str] = (),
    ) -> tuple[GoalDirectedOption, ...]:
        # Compatibility with the inherited controller API.  Explicit
        # subgoals are prepended only when they are legal; they never carry
        # grounded action data.
        legal = tuple(dict.fromkeys(str(item).strip().upper() for item in legal_action_names))
        preferred = tuple(
            name
            for name in dict.fromkeys(str(item).strip().upper() for item in subgoal_action_names)
            if name in legal
        )
        order = (*preferred, *(name for name in legal if name not in preferred))
        option = self.compose_frontier(order, rotation=0)
        return () if option is None else (option,)

    def summary(self) -> dict[str, Any]:
        return {
            "branch_steps": len(self._branch),
            "action_schema_count": len(self._effects_by_action),
            "effect_mode_count": len(self._effect_visits),
            "effect_graph_edges": sum(len(rows) for rows in self._transitions.values()),
            "successful_option_count": len(self._successful_options),
            "observation_rejections": self._observation_rejections,
            "action_uses": dict(self._action_uses),
            "terminal_uses": dict(self._terminal_uses),
            "noop_uses": dict(self._noop_uses),
        }


class CausalSubgoalSageTController(StableFreshPathSageTController):
    """Seed-diversified controller that composes reset-local causal subgoals."""

    def __init__(
        self,
        *args: Any,
        exploration_seed: int = 0,
        reproduce_mixed_registry: bool = False,
        **kwargs: Any,
    ) -> None:
        # Consecutive preregistered seeds must induce consecutive rotations;
        # hashing first can accidentally collapse them modulo a small action
        # alphabet.  The integer is branch-local and is never serialized.
        self._exploration_rotation = abs(int(exploration_seed))
        kwargs.setdefault("exploration_offset", self._exploration_rotation)
        kwargs.setdefault("prefer_mixed", True)
        super().__init__(*args, **kwargs)
        self.inducer = CausalSubgoalAutomatonInducer()
        self.reproduce_mixed_registry = bool(reproduce_mixed_registry)
        self._schema_probe_attempts: Counter[str] = Counter()
        self._causal_frontier_trials = 0
        self._registry_reproduction_trials = 0

    def start_branch(self, *, regime_index: int | None = None) -> None:
        super().start_branch(regime_index=regime_index)
        self._schema_probe_attempts.clear()

    def _rotated_names(self, candidates: Sequence[ActionCandidate]) -> tuple[str, ...]:
        names = tuple(sorted({candidate.action_name for candidate in candidates}))
        if not names:
            return ()
        offset = self._exploration_rotation % len(names)
        return names[offset:] + names[:offset]

    @staticmethod
    def _binding_method(
        action_name: str,
        candidates: Sequence[ActionCandidate],
    ) -> str | None:
        rows = [candidate for candidate in candidates if candidate.action_name == action_name]
        if len(rows) == 1:
            return UNIQUE_ACTION_SCHEMA
        if rows and all(candidate.action_data for candidate in rows):
            return BALANCED_CAUSAL_BINDING
        return None

    def _probe_option(
        self,
        candidates: Sequence[ActionCandidate],
    ) -> GoalDirectedOption | None:
        names = self._rotated_names(candidates)
        available = [
            name
            for name in names
            if self._schema_probe_attempts[name] < PROBES_PER_ACTION_SCHEMA
            and self._binding_method(name, candidates) is not None
        ]
        if not available:
            return None
        action_name = min(
            available,
            key=lambda name: (self._schema_probe_attempts[name], names.index(name)),
        )
        self._schema_probe_attempts[action_name] += 1
        return GoalDirectedOption(
            schema="repeat_target",
            steps=(
                OptionStep(
                    action_name,
                    binding_method=str(self._binding_method(action_name, candidates)),
                    expected_effect="reset_local_causal_probe",
                ),
            ),
            source=CAUSAL_PROBE_SOURCE,
        )

    def _registry_mixed_option(self) -> GoalDirectedOption | None:
        candidates = [
            option
            for option in self.registry.reproduction_candidates()
            if option.mixed and option.option_id not in self._tried_option_ids
        ]
        if not candidates:
            return None
        assessed = self.evaluator.assess(
            tuple(candidates),
            registry=self.registry,
            tried_option_ids=self._tried_option_ids,
        )
        if not assessed:
            return None
        self._registry_reproduction_trials += 1
        option = assessed[0].option
        return GoalDirectedOption(
            schema="mixed_automaton",
            steps=option.steps,
            initiation=option.initiation,
            termination=option.termination,
            source=REPRODUCTION_SOURCE,
        )

    def _choose_option(
        self,
        state: AbstractState,
        candidates: Sequence[ActionCandidate],
        *,
        goal_hypotheses: Sequence[Any] = (),
    ) -> GoalDirectedOption | None:
        if self.phase == "confirmation":
            return super()._choose_option(
                state,
                candidates,
                goal_hypotheses=goal_hypotheses,
            )
        if self.reproduce_mixed_registry:
            reproduced = self._registry_mixed_option()
            if reproduced is not None:
                return reproduced
        probe = self._probe_option(candidates)
        if probe is not None:
            return probe
        names = self._rotated_names(candidates)
        option = self.inducer.compose_frontier(
            names,
            rotation=self._exploration_rotation + self._causal_frontier_trials,
            horizon=min(
                MAXIMUM_OPTION_HORIZON,
                MINIMUM_MIXED_HORIZON + 8 * min(2, self._causal_frontier_trials),
            ),
        )
        if option is not None:
            rebound = []
            for step in option.steps:
                method = self._binding_method(step.action_name, candidates)
                if method is None:
                    return None
                rebound.append(
                    OptionStep(
                        step.action_name,
                        binding_method=method,
                        expected_effect=step.expected_effect,
                    )
                )
            self._causal_frontier_trials += 1
            return GoalDirectedOption(
                schema="mixed_automaton",
                steps=tuple(rebound),
                source=CAUSAL_FRONTIER_SOURCE,
            )
        return super()._choose_option(
            state,
            candidates,
            goal_hypotheses=goal_hypotheses,
        )

    def observe_transition(self, record: TransitionRecord) -> None:
        # The replaced inducer makes the known T10.3.8 effect descriptor path
        # total.  Any remaining exception is intentionally allowed to fail the
        # scientific gate instead of silently losing a posterior update.
        super().observe_transition(record)

    def summary(self) -> Mapping[str, Any]:
        base = dict(super().summary())
        base.update(
            {
                "format_version": FORMAT_VERSION,
                "causal_subgoal_composition": True,
                "effect_descriptor_total": True,
                "seed_diversified_exploration": True,
                "exploration_seed_persisted": False,
                "schema_probe_attempts": dict(self._schema_probe_attempts),
                "causal_frontier_trials": self._causal_frontier_trials,
                "registry_reproduction_trials": self._registry_reproduction_trials,
                "reproduce_mixed_registry": self.reproduce_mixed_registry,
                "causal_inducer": self.inducer.summary(),
            }
        )
        return base


__all__ = [
    "CAUSAL_FRONTIER_SOURCE",
    "CAUSAL_PROBE_SOURCE",
    "CausalSubgoalAutomatonInducer",
    "CausalSubgoalSageTController",
    "FORMAT_VERSION",
    "MINIMUM_MIXED_HORIZON",
    "PROBES_PER_ACTION_SCHEMA",
    "REPRODUCTION_SOURCE",
    "robust_effect_descriptor",
    "robust_effect_signature",
]
