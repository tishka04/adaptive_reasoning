"""Small abstract world model for semantic trajectory rollouts."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from v3.schemas import TransitionRecord

from .compiler import CompiledSemanticOption
from .scene_graph import SceneGraph, build_scene_graph


@dataclass(frozen=True)
class SemanticTrajectoryStep:
    option: CompiledSemanticOption
    state_before: frozenset[str]
    state_after: frozenset[str]
    transition_probability: float
    uncertainty: float


@dataclass(frozen=True)
class SemanticTrajectory:
    steps: tuple[SemanticTrajectoryStep, ...]
    final_state: frozenset[str]
    probability: float
    uncertainty: float

    @property
    def first_option(self) -> CompiledSemanticOption:
        if not self.steps:
            raise ValueError("empty trajectory has no first option")
        return self.steps[0].option

    @property
    def length(self) -> int:
        return len(self.steps)


class SemanticWorldModel:
    """Beta-smoothed effect model over grounded semantic options."""

    def __init__(self, *, action_key_mode: str = "grounded") -> None:
        if action_key_mode not in {"grounded", "name"}:
            raise ValueError("action_key_mode must be either 'grounded' or 'name'")
        self.action_key_mode = action_key_mode
        self._support: Counter[tuple[str, str]] = Counter()
        self._refutations: Counter[tuple[str, str]] = Counter()
        self.observations = 0

    def _effect_key(
        self,
        option: CompiledSemanticOption,
        effect: str,
    ) -> tuple[str, str]:
        action = (
            option.action_key
            if self.action_key_mode == "grounded"
            else option.action_name.strip().upper()
        )
        return action, effect

    def effect_probability(
        self,
        option: CompiledSemanticOption,
        effect: str,
    ) -> float:
        effect_name = str(effect).split("|", 1)[0]
        if effect_name in option.effect_probabilities:
            return float(option.effect_probabilities[effect_name])
        key = self._effect_key(option, effect)
        support = self._support[key]
        refutations = self._refutations[key]
        prior_weight = 2.0
        return (support + 1.0 + prior_weight * float(option.confidence)) / (
            support + refutations + 2.0 + prior_weight
        )

    def effect_uncertainty(
        self,
        option: CompiledSemanticOption,
        effect: str,
    ) -> float:
        key = self._effect_key(option, effect)
        evidence = self._support[key] + self._refutations[key]
        return 1.0 / math.sqrt(evidence + 1.0)

    def observe(
        self,
        option: CompiledSemanticOption,
        observed_effects: Iterable[str],
    ) -> None:
        observed = set(observed_effects)
        for effect in option.asserted_effects:
            key = self._effect_key(option, effect)
            if effect in observed:
                self._support[key] += 1
            else:
                self._refutations[key] += 1
        self.observations += 1

    def rollout(
        self,
        *,
        initial_state: Iterable[str],
        options: Sequence[CompiledSemanticOption],
        maximum_depth: int = 3,
        beam_width: int = 8,
    ) -> tuple[SemanticTrajectory, ...]:
        """Beam-search abstract trajectories; no environment is stepped."""
        initial = frozenset(initial_state)
        beam = (
            SemanticTrajectory(
                steps=(),
                final_state=initial,
                probability=1.0,
                uncertainty=0.0,
            ),
        )
        completed = []
        for _ in range(max(1, int(maximum_depth))):
            expanded = []
            for trajectory in beam:
                for option in options:
                    if not set(option.preconditions).issubset(trajectory.final_state):
                        continue
                    probabilities = [
                        self.effect_probability(option, effect)
                        for effect in option.asserted_effects
                    ]
                    uncertainties = [
                        self.effect_uncertainty(option, effect)
                        for effect in option.asserted_effects
                    ]
                    transition_probability = (
                        sum(probabilities) / len(probabilities)
                        if probabilities
                        else 0.0
                    )
                    transition_uncertainty = (
                        sum(uncertainties) / len(uncertainties)
                        if uncertainties
                        else 1.0
                    )
                    next_state = set(trajectory.final_state)
                    next_state.difference_update(option.retracted_effects)
                    next_state.update(option.asserted_effects)
                    step = SemanticTrajectoryStep(
                        option=option,
                        state_before=trajectory.final_state,
                        state_after=frozenset(next_state),
                        transition_probability=transition_probability,
                        uncertainty=transition_uncertainty,
                    )
                    expanded.append(
                        SemanticTrajectory(
                            steps=trajectory.steps + (step,),
                            final_state=frozenset(next_state),
                            probability=(
                                trajectory.probability
                                * max(1e-6, transition_probability)
                            ),
                            uncertainty=(
                                trajectory.uncertainty + transition_uncertainty
                            )
                            / (trajectory.length + 1),
                        )
                    )
            if not expanded:
                break
            expanded.sort(
                key=lambda item: (
                    item.probability - 0.1 * item.uncertainty - 0.01 * item.length
                ),
                reverse=True,
            )
            beam = tuple(expanded[: max(1, int(beam_width))])
            completed.extend(beam)
        return tuple(completed)

    def summary(self) -> dict[str, int | str]:
        return {
            "action_key_mode": self.action_key_mode,
            "observations": self.observations,
            "supported_effects": sum(self._support.values()),
            "refuted_effects": sum(self._refutations.values()),
        }


def observed_semantic_effects(record: TransitionRecord) -> frozenset[str]:
    """Derive semantic outcomes only from an observed before/after record."""
    before: SceneGraph = build_scene_graph(record.obs_before)
    after: SceneGraph = build_scene_graph(record.obs_after)
    effects = set(after.state_predicates - before.state_predicates)
    diff = record.diff
    if not diff.is_noop:
        effects.add("changed|-|-|")
    if diff.player_displacement is not None:
        for entity in before.entities_for_role("player"):
            effects.add(f"moved|{entity.entity_id}|-|")
    if diff.level_complete:
        effects.add("level_complete|-|-|")
        effects.add("progress|-|-|")
    if diff.game_over:
        effects.add("game_over|-|-|")
    return frozenset(effects)


__all__ = [
    "SemanticTrajectory",
    "SemanticTrajectoryStep",
    "SemanticWorldModel",
    "observed_semantic_effects",
]
