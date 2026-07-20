"""Active, bounded generation and discrimination of online goal hypotheses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from v3.schemas import GameObservation, Operator

from .online_relational_option import relation_holds
from .online_terminal_objective import (
    OnlineTerminalObjectiveStore,
    TerminalObjectiveAssessment,
    TerminalObjectiveHypothesis,
    TerminalObjectiveStatus,
)
from .promoted_relational_rule import PromotedRelationalRule


RELATION_PREDICATES = ("adjacent_to", "aligned_with", "same_shape")


@dataclass(frozen=True)
class GeneratedGoalHypothesis:
    """One measurable counterfactual goal; generation is never proof."""

    objective_id: str
    family: str
    source_color: int | None
    target_color: int | None
    predicate: str
    supporting_rule_keys: Tuple[str, ...]
    supporting_actions: Tuple[str, ...]
    generation_reason: str
    prior_priority: float = 0.0


@dataclass(frozen=True)
class ObjectiveExperimentChoice:
    """A primitive intervention selected to discriminate online goals."""

    action_name: str
    action_data: Dict[str, Any]
    intervention_id: str
    objective_id: str
    competing_objective_ids: Tuple[str, ...]
    predicted_reduction_objective_ids: Tuple[str, ...]
    ablation_of_objective_id: str = ""
    expected_divergence: float = 0.0
    is_probe: bool = True
    reason: str = ""


class GoalHypothesisGenerator:
    """Generate a small diverse bank from live structure and learned mechanics."""

    def __init__(
        self,
        *,
        max_candidates_total: int = 10,
        max_candidates_per_family: int = 2,
        max_colors: int = 5,
    ) -> None:
        self.max_candidates_total = max(1, int(max_candidates_total))
        self.max_candidates_per_family = max(1, int(max_candidates_per_family))
        self.max_colors = max(2, int(max_colors))

    def generate(
        self,
        *,
        observation: GameObservation,
        rules: Sequence[PromotedRelationalRule],
        available_actions: Sequence[str],
    ) -> List[GeneratedGoalHypothesis]:
        actions = tuple(sorted({_normalize_action(action) for action in available_actions}))
        colors = _rank_object_colors(observation)[: self.max_colors]
        if not colors or not actions:
            return []
        by_family: Dict[str, List[GeneratedGoalHypothesis]] = {
            family: [] for family in ("appear", "break", "exhaust", "reach", "convert")
        }
        by_family["appear"].extend(
            self._relation_candidates(
                family="appear",
                observation=observation,
                colors=colors,
                rules=rules,
                actions=actions,
            )
        )
        by_family["break"].extend(
            self._relation_candidates(
                family="break",
                observation=observation,
                colors=colors,
                rules=rules,
                actions=actions,
            )
        )
        by_family["exhaust"].extend(
            self._exhaust_candidates(observation, colors, rules, actions)
        )
        by_family["reach"].extend(
            self._reach_candidates(observation, colors, actions)
        )
        by_family["convert"].extend(
            self._convert_candidates(colors, rules, actions)
        )

        selected: List[GeneratedGoalHypothesis] = []
        for family in ("appear", "break", "exhaust", "reach", "convert"):
            unique = _dedupe_candidates(by_family[family])
            ranked = sorted(
                unique,
                key=lambda item: (item.prior_priority, item.objective_id),
                reverse=True,
            )
            selected.extend(ranked[: self.max_candidates_per_family])
        selected.sort(
            key=lambda item: (item.prior_priority, item.family, item.objective_id),
            reverse=True,
        )
        return selected[: self.max_candidates_total]

    def generate_from_transition(
        self,
        *,
        observation_before: GameObservation,
        observation_after: GameObservation,
        action_name: str,
        rules: Sequence[PromotedRelationalRule] = (),
        max_candidates: int = 6,
    ) -> List[GeneratedGoalHypothesis]:
        """Turn a real non-noop effect into new measurable goal candidates.

        These candidates describe transformations suggested by the observed
        color flow, relation toggle, or approach.  They remain hypotheses: only
        later online distance reductions and terminal outcomes can support them.
        """
        before_grid = np.asarray(observation_before.raw_grid, dtype=np.int32)
        after_grid = np.asarray(observation_after.raw_grid, dtype=np.int32)
        if before_grid.shape != after_grid.shape or np.array_equal(
            before_grid,
            after_grid,
        ):
            return []
        action = _normalize_action(action_name)
        before_counts = _color_counts(before_grid)
        after_counts = _color_counts(after_grid)
        background_colors = {
            color
            for color, count in (*before_counts.items(), *after_counts.items())
            if count >= 0.5 * before_grid.size
        }
        colors = sorted(set(before_counts) | set(after_counts))
        decreased = sorted(
            (
                (color, before_counts[color] - after_counts.get(color, 0))
                for color in colors
                if color not in background_colors
                if before_counts.get(color, 0) > after_counts.get(color, 0)
            ),
            key=lambda item: (item[1], item[0]),
            reverse=True,
        )
        increased = sorted(
            (
                (color, after_counts[color] - before_counts.get(color, 0))
                for color in colors
                if color not in background_colors
                if after_counts.get(color, 0) > before_counts.get(color, 0)
            ),
            key=lambda item: (item[1], item[0]),
            reverse=True,
        )
        candidates: List[GeneratedGoalHypothesis] = []
        for source, _ in decreased[:3]:
            matching = [
                rule for rule in rules
                if rule.source_color == int(source) and rule.action == action
            ]
            candidates.append(GeneratedGoalHypothesis(
                objective_id=f"terminal::exhaust::color{int(source)}",
                family="exhaust",
                source_color=int(source),
                target_color=None,
                predicate="object_count_equals_zero",
                supporting_rule_keys=tuple(sorted(rule.key for rule in matching)),
                supporting_actions=(action,),
                generation_reason="observed_effect_color_depletion",
                prior_priority=0.9,
            ))
        for (source, _), (target, _) in (
            (left, right)
            for left in decreased[:3]
            for right in increased[:3]
            if left[0] != right[0]
        ):
            matching = [
                rule for rule in rules
                if rule.family == "color_transform"
                and rule.source_color == int(source)
                and rule.target_color == int(target)
                and rule.action == action
            ]
            candidates.append(GeneratedGoalHypothesis(
                objective_id=(
                    f"terminal::convert::{int(source)}_to_{int(target)}"
                ),
                family="convert",
                source_color=int(source),
                target_color=int(target),
                predicate="source_target_color_transform",
                supporting_rule_keys=tuple(sorted(rule.key for rule in matching)),
                supporting_actions=(action,),
                generation_reason="observed_effect_color_flow",
                prior_priority=1.1 if matching else 1.0,
            ))

        changed_colors = {
            int(color) for color, _ in (*decreased[:3], *increased[:3])
        }
        structural_colors = list(dict.fromkeys((
            *changed_colors,
            *(
                color
                for color in _rank_object_colors(observation_after)
                if color not in background_colors
            ),
        )))
        for first, second in combinations(structural_colors[:5], 2):
            if not ({int(first), int(second)} & changed_colors):
                continue
            pair = tuple(sorted((int(first), int(second))))
            for predicate in RELATION_PREDICATES:
                held_before = relation_holds(
                    observation_before,
                    predicate,
                    pair[0],
                    pair[1],
                )
                holds_after = relation_holds(
                    observation_after,
                    predicate,
                    pair[0],
                    pair[1],
                )
                if held_before == holds_after:
                    continue
                family = "break" if holds_after else "appear"
                candidates.append(GeneratedGoalHypothesis(
                    objective_id=(
                        f"terminal::{family}::{predicate}::"
                        f"colors{pair[0]}_{pair[1]}"
                    ),
                    family=family,
                    source_color=pair[0],
                    target_color=pair[1],
                    predicate=predicate,
                    supporting_rule_keys=(),
                    supporting_actions=(action,),
                    generation_reason="observed_effect_relation_toggle",
                    prior_priority=0.85,
                ))

        if (
            observation_before.best_player is not None
            and observation_after.best_player is not None
        ):
            for color in _rank_object_colors(observation_after)[: self.max_colors]:
                before_distance = _reach_distance(observation_before, int(color))
                after_distance = _reach_distance(observation_after, int(color))
                if (
                    before_distance is None
                    or after_distance is None
                    or after_distance >= before_distance
                ):
                    continue
                candidates.append(GeneratedGoalHypothesis(
                    objective_id=f"terminal::reach::color{int(color)}",
                    family="reach",
                    source_color=int(observation_after.best_player.value),
                    target_color=int(color),
                    predicate="player_adjacent_to_target",
                    supporting_rule_keys=(),
                    supporting_actions=(action,),
                    generation_reason="observed_effect_approach",
                    prior_priority=0.95,
                ))

        ranked = sorted(
            _dedupe_candidates(candidates),
            key=lambda item: (item.prior_priority, item.objective_id),
            reverse=True,
        )
        return ranked[: max(1, int(max_candidates))]

    def _relation_candidates(
        self,
        *,
        family: str,
        observation: GameObservation,
        colors: Sequence[int],
        rules: Sequence[PromotedRelationalRule],
        actions: Sequence[str],
    ) -> List[GeneratedGoalHypothesis]:
        result: List[GeneratedGoalHypothesis] = []
        rule_predicates = [
            rule.predicate for rule in rules if rule.family == "relation"
        ]
        predicates = tuple(dict.fromkeys((*rule_predicates, *RELATION_PREDICATES)))
        for first, second in combinations(colors, 2):
            pair = tuple(sorted((int(first), int(second))))
            for predicate in predicates:
                holds = relation_holds(
                    observation,
                    predicate,
                    pair[0],
                    pair[1],
                )
                if family == "appear" and holds:
                    continue
                if family == "break" and not holds:
                    continue
                matching = [
                    rule
                    for rule in rules
                    if rule.family == "relation"
                    and rule.predicate == predicate
                    and rule.target_color is not None
                    and set((rule.source_color, rule.target_color)) == set(pair)
                ]
                supporting_actions = _supporting_actions(
                    matching,
                    actions,
                    allow_click_probe=True,
                )
                if not supporting_actions:
                    continue
                directed = any(
                    rule.expected_outcome == (
                        "appears" if family == "appear" else "broken"
                    )
                    for rule in matching
                )
                result.append(GeneratedGoalHypothesis(
                    objective_id=(
                        f"terminal::{family}::{predicate}::"
                        f"colors{pair[0]}_{pair[1]}"
                    ),
                    family=family,
                    source_color=pair[0],
                    target_color=pair[1],
                    predicate=predicate,
                    supporting_rule_keys=tuple(sorted(rule.key for rule in matching)),
                    supporting_actions=supporting_actions,
                    generation_reason=(
                        "directed_relation_mechanic"
                        if directed
                        else "live_relation_counterfactual"
                    ),
                    prior_priority=0.9 if directed else (0.55 if matching else 0.25),
                ))
        return result

    def _exhaust_candidates(
        self,
        observation: GameObservation,
        colors: Sequence[int],
        rules: Sequence[PromotedRelationalRule],
        actions: Sequence[str],
    ) -> List[GeneratedGoalHypothesis]:
        player_color = (
            None if observation.best_player is None else observation.best_player.value
        )
        result = []
        for color in colors:
            if player_color is not None and int(color) == int(player_color):
                continue
            matching = [rule for rule in rules if rule.source_color == int(color)]
            supporting_actions = _supporting_actions(
                matching,
                actions,
                allow_click_probe=True,
            )
            if not supporting_actions:
                continue
            directed = any(
                rule.family == "color_transform" for rule in matching
            )
            result.append(GeneratedGoalHypothesis(
                objective_id=f"terminal::exhaust::color{int(color)}",
                family="exhaust",
                source_color=int(color),
                target_color=None,
                predicate="object_count_equals_zero",
                supporting_rule_keys=tuple(sorted(rule.key for rule in matching)),
                supporting_actions=supporting_actions,
                generation_reason=(
                    "reducible_object_class" if directed else "testable_object_class"
                ),
                prior_priority=0.75 if directed else 0.2,
            ))
        return result

    def _reach_candidates(
        self,
        observation: GameObservation,
        colors: Sequence[int],
        actions: Sequence[str],
    ) -> List[GeneratedGoalHypothesis]:
        if observation.best_player is None:
            return []
        movement_actions = tuple(action for action in actions if action != "ACTION6")
        if not movement_actions:
            return []
        player_color = int(observation.best_player.value)
        return [
            GeneratedGoalHypothesis(
                objective_id=f"terminal::reach::color{int(color)}",
                family="reach",
                source_color=player_color,
                target_color=int(color),
                predicate="player_adjacent_to_target",
                supporting_rule_keys=(),
                supporting_actions=movement_actions,
                generation_reason="live_reachable_object_counterfactual",
                prior_priority=0.3,
            )
            for color in colors
            if int(color) != player_color
        ]

    def _convert_candidates(
        self,
        colors: Sequence[int],
        rules: Sequence[PromotedRelationalRule],
        actions: Sequence[str],
    ) -> List[GeneratedGoalHypothesis]:
        result: List[GeneratedGoalHypothesis] = []
        informed_pairs = [
            (rule.source_color, int(rule.target_color))
            for rule in rules
            if rule.target_color is not None
            and rule.family == "color_transform"
        ]
        pairs = tuple(dict.fromkeys((*informed_pairs, *permutations(colors, 2))))
        for source, target in pairs:
            matching = [
                rule
                for rule in rules
                if rule.source_color == int(source)
                and rule.target_color == int(target)
                and rule.family == "color_transform"
            ]
            supporting_actions = _supporting_actions(
                matching,
                actions,
                allow_click_probe=True,
            )
            if not supporting_actions:
                continue
            result.append(GeneratedGoalHypothesis(
                objective_id=f"terminal::convert::{int(source)}_to_{int(target)}",
                family="convert",
                source_color=int(source),
                target_color=int(target),
                predicate="source_target_color_transform",
                supporting_rule_keys=tuple(sorted(rule.key for rule in matching)),
                supporting_actions=supporting_actions,
                generation_reason=(
                    "directed_color_mechanic"
                    if matching
                    else "testable_color_conversion"
                ),
                prior_priority=0.95 if matching else 0.15,
            ))
        return result


class ObjectiveDiscriminatingExperimentDesigner:
    """Choose selective interventions and terminal-only ablations."""

    def design(
        self,
        *,
        observation: GameObservation,
        store: OnlineTerminalObjectiveStore,
        safe_actions: Sequence[str],
        click_actions: Sequence[Any] = (),
        operators: Iterable[Operator] = (),
        preferred_objective_id: str = "",
        intervention_utilities: Mapping[str, float] | None = None,
        allow_ablation: bool = True,
        require_selectable: bool = True,
    ) -> ObjectiveExperimentChoice | None:
        assessments = {
            objective.objective_id: store.assess_objective(objective, observation)
            for objective in store.objectives()
        }
        selectable = {
            key: assessment
            for key, assessment in assessments.items()
            if assessment.selectable
        }
        preferred = str(preferred_objective_id)
        if preferred:
            objective = store.objective(preferred)
            assessment = assessments.get(preferred)
            if (
                objective is None
                or assessment is None
                or assessment.distance is None
                or assessment.distance <= 0.0
                or assessment.status == TerminalObjectiveStatus.REFUTED
                or (require_selectable and not assessment.selectable)
            ):
                return None
            selectable = {preferred: assessment}
        if not selectable and not store.ablation_targets():
            return None
        interventions = _candidate_interventions(
            observation=observation,
            objectives=store.objectives(),
            safe_actions=safe_actions,
            click_actions=click_actions,
            operators=operators,
        )
        if not interventions:
            return None

        ablation = None
        if allow_ablation and not preferred:
            ablation = self._design_ablation(
                observation=observation,
                store=store,
                selectable=selectable,
                interventions=interventions,
            )
        if ablation is not None:
            return ablation
        if not selectable:
            return None

        choices: List[Tuple[Tuple[float, ...], ObjectiveExperimentChoice]] = []
        learned_utilities = dict(intervention_utilities or {})
        for intervention in interventions:
            affected = tuple(
                objective_id
                for objective_id in intervention.predicted_reductions
                if objective_id in selectable
                and not store.intervention_is_unsafe(
                    objective_id, intervention.intervention_id
                )
            )
            if not affected:
                continue
            primary = (
                preferred
                if preferred and preferred in affected
                else max(affected, key=lambda key: selectable[key].priority)
            )
            unaffected = tuple(key for key in selectable if key not in affected)
            competitor = (
                max(unaffected, key=lambda key: selectable[key].priority)
                if unaffected else ""
            )
            competing = tuple(
                key for key in (primary, competitor) if key
            )
            divergence = 2.0 if competitor else (1.0 / max(1, len(affected)))
            assessment = selectable[primary]
            effective_priority = assessment.priority
            if effective_priority == float("-inf"):
                objective = store.objective(primary)
                effective_priority = (
                    0.0 if objective is None else objective.prior_priority
                ) + 1.0 / (1.0 + float(assessment.distance or 0.0))
            choice = ObjectiveExperimentChoice(
                action_name=intervention.action_name,
                action_data=dict(intervention.action_data),
                intervention_id=intervention.intervention_id,
                objective_id=primary,
                competing_objective_ids=competing,
                predicted_reduction_objective_ids=affected,
                expected_divergence=divergence,
                is_probe=(
                    assessment.status
                    != TerminalObjectiveStatus.TERMINAL_SUPPORTED
                    if preferred else assessment.is_probe
                ),
                reason=(
                    "effect-guided causal intervention"
                    if learned_utilities.get(
                        semantic_intervention_signature(
                            intervention.action_name,
                            intervention.action_data,
                            observation,
                        ),
                        0.0,
                    ) > 0.0
                    else (
                        "selective terminal-goal discriminator"
                        if competitor
                        else "bounded measurable terminal-goal probe"
                    )
                ),
            )
            learned_utility = learned_utilities.get(
                semantic_intervention_signature(
                    intervention.action_name,
                    intervention.action_data,
                    observation,
                ),
                0.0,
            )
            choices.append((
                (
                    int(assessment.status == TerminalObjectiveStatus.TERMINAL_SUPPORTED),
                    float(learned_utility),
                    divergence,
                    effective_priority,
                    -len(affected),
                ),
                choice,
            ))
        return None if not choices else max(choices, key=lambda item: item[0])[1]

    def _design_ablation(
        self,
        *,
        observation: GameObservation,
        store: OnlineTerminalObjectiveStore,
        selectable: Mapping[str, TerminalObjectiveAssessment],
        interventions: Sequence["_CandidateIntervention"],
    ) -> ObjectiveExperimentChoice | None:
        for target in store.ablation_targets():
            target_distance = target.distance(observation)
            if target_distance is None or target_distance <= 0.0:
                continue
            alternatives = []
            for intervention in interventions:
                if intervention.intervention_id in target.terminal_interventions:
                    continue
                if store.intervention_is_unsafe(
                    target.objective_id, intervention.intervention_id
                ):
                    continue
                affected = tuple(
                    objective_id
                    for objective_id in intervention.predicted_reductions
                    if objective_id in selectable
                    and objective_id != target.objective_id
                )
                omits_target = target.objective_id not in intervention.predicted_reductions
                alternatives.append((
                    (int(omits_target), int(bool(affected)), -len(affected)),
                    intervention,
                    affected,
                ))
            if not alternatives:
                continue
            _, intervention, affected = max(alternatives, key=lambda item: item[0])
            primary = affected[0] if affected else ""
            return ObjectiveExperimentChoice(
                action_name=intervention.action_name,
                action_data=dict(intervention.action_data),
                intervention_id=intervention.intervention_id,
                objective_id=primary,
                competing_objective_ids=tuple(
                    dict.fromkeys((target.objective_id, *affected))
                ),
                predicted_reduction_objective_ids=affected,
                ablation_of_objective_id=target.objective_id,
                expected_divergence=3.0,
                is_probe=True,
                reason="terminal-only ablation of a provisionally supported goal",
            )
        return None


@dataclass(frozen=True)
class _CandidateIntervention:
    action_name: str
    action_data: Dict[str, Any]
    intervention_id: str
    predicted_reductions: Tuple[str, ...]


def intervention_id(action_name: str, action_data: Mapping[str, Any]) -> str:
    return f"{_normalize_action(action_name)}::{json.dumps(dict(action_data), sort_keys=True, separators=(',', ':'))}"


def semantic_intervention_signature(
    action_name: str,
    action_data: Mapping[str, Any],
    observation: GameObservation,
) -> str:
    """Return a position-invariant identity for transferable interventions."""
    normalized = _normalize_action(action_name)
    if normalized != "ACTION6":
        return normalized
    color = _clicked_color(observation, action_data)
    return f"ACTION6::color:{'unknown' if color is None else int(color)}"


def _candidate_interventions(
    *,
    observation: GameObservation,
    objectives: Sequence[TerminalObjectiveHypothesis],
    safe_actions: Sequence[str],
    click_actions: Sequence[Any],
    operators: Iterable[Operator],
) -> List[_CandidateIntervention]:
    concrete: List[Tuple[str, Dict[str, Any]]] = []
    for action in safe_actions:
        name = _normalize_action(action)
        if name == "ACTION6":
            concrete.extend(
                (name, dict(getattr(item, "action_args", {}) or {}))
                for item in click_actions
            )
        else:
            concrete.append((name, {}))
    movement = _movement_predictions(operators)
    result = []
    for action_name, action_data in concrete:
        identity = intervention_id(action_name, action_data)
        clicked_color = _clicked_color(observation, action_data)
        reductions = []
        for objective in objectives:
            if action_name not in objective.supporting_actions:
                continue
            if _intervention_can_reduce(
                objective,
                observation=observation,
                action_name=action_name,
                clicked_color=clicked_color,
                displacement=movement.get(action_name),
            ):
                reductions.append(objective.objective_id)
        result.append(_CandidateIntervention(
            action_name=action_name,
            action_data=action_data,
            intervention_id=identity,
            predicted_reductions=tuple(sorted(reductions)),
        ))
    return result


def _intervention_can_reduce(
    objective: TerminalObjectiveHypothesis,
    *,
    observation: GameObservation,
    action_name: str,
    clicked_color: int | None,
    displacement: Tuple[int, int] | None,
) -> bool:
    if objective.family in {"exhaust", "convert"}:
        return action_name != "ACTION6" or clicked_color == objective.source_color
    if objective.family in {"appear", "break"}:
        return action_name != "ACTION6" or clicked_color in {
            objective.source_color,
            objective.target_color,
        }
    if objective.family != "reach":
        return False
    if displacement is None or observation.best_player is None:
        return True  # Unknown movement semantics: testable, never proof.
    before = objective.distance(observation)
    if before is None or objective.target_color is None:
        return False
    player_row, player_col = observation.best_player.position
    projected = (player_row + displacement[0], player_col + displacement[1])
    targets = [
        obj for obj in observation.objects if obj.value == objective.target_color
    ]
    if not targets:
        return False
    after = max(0, min(
        abs(int(row) - int(projected[0])) + abs(int(col) - int(projected[1]))
        for obj in targets
        for row, col in obj.cells
    ) - 1)
    return float(after) < before


def _movement_predictions(
    operators: Iterable[Operator],
) -> Dict[str, Tuple[int, int]]:
    result = {}
    for operator in operators:
        if str(getattr(operator.kind, "value", operator.kind)) != "move":
            continue
        action = _normalize_action(operator.primitive_action or "")
        if not action:
            continue
        result[action] = (
            int(operator.parameters.get("dy", 0)),
            int(operator.parameters.get("dx", 0)),
        )
    return result


def _clicked_color(
    observation: GameObservation,
    action_data: Mapping[str, Any],
) -> int | None:
    try:
        x = int(action_data["x"])
        y = int(action_data["y"])
    except (KeyError, TypeError, ValueError):
        return None
    grid = observation.raw_grid
    if not (0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]):
        return None
    return int(grid[y, x])


def _rank_object_colors(observation: GameObservation) -> List[int]:
    counts: Dict[int, Tuple[int, int]] = {}
    for obj in observation.objects:
        objects, area = counts.get(int(obj.value), (0, 0))
        counts[int(obj.value)] = (objects + 1, area + int(obj.area))
    return [
        color
        for color, _ in sorted(
            counts.items(),
            key=lambda item: (item[1][0], item[1][1], -item[0]),
            reverse=True,
        )
    ]


def _color_counts(grid: np.ndarray) -> Dict[int, int]:
    values, counts = np.unique(np.asarray(grid, dtype=np.int32), return_counts=True)
    return {
        int(value): int(count)
        for value, count in zip(values.tolist(), counts.tolist())
    }


def _reach_distance(
    observation: GameObservation,
    target_color: int,
) -> float | None:
    if observation.best_player is None:
        return None
    targets = [
        obj for obj in observation.objects if obj.value == int(target_color)
    ]
    if not targets:
        return None
    player_row, player_col = observation.best_player.position
    distance = min(
        abs(int(row) - int(player_row)) + abs(int(col) - int(player_col))
        for obj in targets
        for row, col in obj.cells
    )
    return float(max(0, distance - 1))


def _supporting_actions(
    rules: Sequence[PromotedRelationalRule],
    available_actions: Sequence[str],
    *,
    allow_click_probe: bool,
) -> Tuple[str, ...]:
    legal = set(available_actions)
    actions = {
        _normalize_action(rule.action)
        for rule in rules
        if _normalize_action(rule.action) in legal
    }
    if allow_click_probe and "ACTION6" in legal:
        actions.add("ACTION6")
    return tuple(sorted(actions))


def _dedupe_candidates(
    candidates: Iterable[GeneratedGoalHypothesis],
) -> List[GeneratedGoalHypothesis]:
    by_id: Dict[str, GeneratedGoalHypothesis] = {}
    for candidate in candidates:
        existing = by_id.get(candidate.objective_id)
        if existing is None or candidate.prior_priority > existing.prior_priority:
            by_id[candidate.objective_id] = candidate
    return list(by_id.values())


def _normalize_action(action: Any) -> str:
    raw = getattr(action, "name", action)
    text = str(raw or "").strip().upper()
    return text.split(".")[-1]


__all__ = [
    "GeneratedGoalHypothesis",
    "GoalHypothesisGenerator",
    "ObjectiveDiscriminatingExperimentDesigner",
    "ObjectiveExperimentChoice",
    "intervention_id",
]
