"""Compile promoted relational rules into live, revisable execution options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from v3.schemas import GameObservation

from .epistemic_metrics import HypothesisStatus
from .promoted_relational_rule import PromotedRelationalRule


@dataclass(frozen=True)
class CompiledRelationalOption:
    """A confirmed rule with explicit initiation and termination predicates."""

    option_id: str
    rule_key: str
    action: str
    source_color: int
    target_color: int | None
    family: str
    predicate: str
    expected_outcome: str
    required_predicates: Tuple[str, ...]
    postconditions: Tuple[str, ...]


@dataclass(frozen=True)
class OptionAssessment:
    """Current applicability of one compiled option."""

    ready: bool
    already_satisfied: bool = False
    present_predicates: Tuple[str, ...] = ()
    missing_predicates: Tuple[str, ...] = ()


@dataclass(frozen=True)
class FunctionalOptionProgress:
    """Directed outcome of an option, separate from raw visual change."""

    expected_outcome_observed: bool
    functional_progress: bool
    level_progressed: bool
    visual_change: bool
    visual_only_change: bool
    source_reduction: int = 0
    target_gain: int = 0
    source_to_target_pixels: int = 0
    relation_appeared: bool = False
    relation_preserved: bool = False
    signals: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_outcome_observed": self.expected_outcome_observed,
            "functional_progress": self.functional_progress,
            "level_progressed": self.level_progressed,
            "visual_change": self.visual_change,
            "visual_only_change": self.visual_only_change,
            "source_reduction": self.source_reduction,
            "target_gain": self.target_gain,
            "source_to_target_pixels": self.source_to_target_pixels,
            "relation_appeared": self.relation_appeared,
            "relation_preserved": self.relation_preserved,
            "signals": list(self.signals),
        }


@dataclass
class OptionOutcomeStats:
    executions: int = 0
    expected_successes: int = 0
    functional_successes: int = 0
    level_successes: int = 0
    visual_only_outcomes: int = 0
    contradictions: int = 0
    preparation_uses: int = 0

    @property
    def expected_rate(self) -> float:
        return self.expected_successes / self.executions if self.executions else 0.0

    @property
    def functional_rate(self) -> float:
        return self.functional_successes / self.executions if self.executions else 0.0

    @property
    def level_rate(self) -> float:
        return self.level_successes / self.executions if self.executions else 0.0

    @property
    def visual_only_rate(self) -> float:
        return self.visual_only_outcomes / self.executions if self.executions else 0.0


class OptionExecutionMemory:
    """Value options from directed progress, never from pixel change alone."""

    def __init__(self) -> None:
        self._by_rule: Dict[str, OptionOutcomeStats] = {}

    def record(
        self,
        rule_key: str,
        progress: FunctionalOptionProgress,
        *,
        used_as_preparation: bool = False,
    ) -> None:
        stats = self._by_rule.setdefault(str(rule_key), OptionOutcomeStats())
        stats.executions += 1
        stats.expected_successes += int(progress.expected_outcome_observed)
        stats.functional_successes += int(progress.functional_progress)
        stats.level_successes += int(progress.level_progressed)
        stats.visual_only_outcomes += int(progress.visual_only_change)
        stats.contradictions += int(not progress.expected_outcome_observed)
        stats.preparation_uses += int(used_as_preparation)

    def stats_for(self, rule_key: str) -> OptionOutcomeStats:
        return self._by_rule.get(str(rule_key), OptionOutcomeStats())

    def value(self, rule: PromotedRelationalRule) -> float:
        stats = self.stats_for(rule.key)
        evidence_value = 0.45 * rule.confidence
        if stats.executions == 0:
            return evidence_value + 0.05
        return (
            evidence_value
            + 0.20 * stats.functional_rate
            + 0.20 * stats.level_rate
            + 0.10 * stats.expected_rate
            - 0.15 * stats.visual_only_rate
            - 0.20 * (stats.contradictions / stats.executions)
            - 0.01 * min(5, stats.executions)
        )

    def is_sterile(self, rule_key: str, *, min_executions: int = 3) -> bool:
        """True when a mechanically valid option has produced no goal progress."""
        stats = self.stats_for(rule_key)
        return bool(
            stats.executions >= max(1, int(min_executions))
            and stats.functional_successes == 0
            and stats.level_successes == 0
        )

    def summary(self) -> Dict[str, Any]:
        total = sum(stats.executions for stats in self._by_rule.values())
        return {
            "rules_executed": len(self._by_rule),
            "executions": total,
            "expected_successes": sum(
                stats.expected_successes for stats in self._by_rule.values()
            ),
            "functional_successes": sum(
                stats.functional_successes for stats in self._by_rule.values()
            ),
            "level_successes": sum(
                stats.level_successes for stats in self._by_rule.values()
            ),
            "visual_only_outcomes": sum(
                stats.visual_only_outcomes for stats in self._by_rule.values()
            ),
            "contradictions": sum(
                stats.contradictions for stats in self._by_rule.values()
            ),
            "preparation_uses": sum(
                stats.preparation_uses for stats in self._by_rule.values()
            ),
            "sterile_rule_keys": sorted(
                rule_key
                for rule_key in self._by_rule
                if self.is_sterile(rule_key)
            ),
        }


class OnlineRelationalOptionCompiler:
    """Turn confirmed directed rules into composable online options."""

    def compile(self, rules: Iterable[PromotedRelationalRule]) -> List[CompiledRelationalOption]:
        options = [
            _compile_rule(rule)
            for rule in rules
            if rule.status == HypothesisStatus.CONFIRMED and rule.goal_relevant
        ]
        return sorted(options, key=lambda option: option.option_id)

    def assess(
        self,
        option: CompiledRelationalOption,
        observation: GameObservation,
    ) -> OptionAssessment:
        present = _live_predicates(option, observation)
        already_satisfied = _postcondition_satisfied(option, present)
        missing = tuple(
            predicate
            for predicate in option.required_predicates
            if predicate not in present
        )
        return OptionAssessment(
            ready=not missing and not already_satisfied,
            already_satisfied=already_satisfied,
            present_predicates=tuple(sorted(present)),
            missing_predicates=missing,
        )

    def preparation_chain(
        self,
        target: CompiledRelationalOption,
        options: Sequence[CompiledRelationalOption],
        observation: GameObservation,
    ) -> List[CompiledRelationalOption]:
        """Find a one-step confirmed option that can establish a missing input."""
        assessment = self.assess(target, observation)
        if assessment.ready or assessment.already_satisfied:
            return []
        missing = set(assessment.missing_predicates)
        for candidate in options:
            if candidate.rule_key == target.rule_key:
                continue
            if not (missing & set(candidate.postconditions)):
                continue
            candidate_assessment = self.assess(candidate, observation)
            if candidate_assessment.ready:
                return [candidate, target]
        return []


def observe_option_progress(
    update: Any,
    rule: PromotedRelationalRule,
) -> FunctionalOptionProgress:
    """Measure a rule application using directed, rule-specific signals."""
    record = update.record
    before = np.asarray(record.obs_before.raw_grid, dtype=np.int32)
    after = np.asarray(record.obs_after.raw_grid, dtype=np.int32)
    source = int(rule.source_color)
    target = None if rule.target_color is None else int(rule.target_color)
    source_before = int(np.sum(before == source))
    source_after = int(np.sum(after == source))
    target_before = 0 if target is None else int(np.sum(before == target))
    target_after = 0 if target is None else int(np.sum(after == target))
    source_to_target = 0
    if target is not None and before.shape == after.shape:
        source_to_target = int(np.sum((before == source) & (after == target)))

    observed_outcome = observed_rule_outcome(update, rule)
    expected = observed_outcome == rule.expected_outcome
    before_relation = False
    after_relation = False
    if rule.family == "relation" and target is not None:
        before_relation = relation_holds(
            record.obs_before,
            rule.predicate,
            source,
            target,
        )
        after_relation = relation_holds(
            record.obs_after,
            rule.predicate,
            source,
            target,
        )
    relation_appeared = not before_relation and after_relation
    relation_preserved = before_relation and after_relation
    level_progressed = bool(
        record.obs_after.levels_completed > record.obs_before.levels_completed
        or record.diff.level_complete
    )
    directed_transform = bool(
        rule.family == "color_transform"
        and target is not None
        and source_to_target > 0
        and source_after < source_before
    )
    directed_relation = bool(
        rule.family == "relation"
        and (
            (rule.expected_outcome == "appears" and relation_appeared)
            or (rule.expected_outcome == "preserved" and level_progressed and relation_preserved)
        )
    )
    functional = bool(level_progressed or directed_transform or directed_relation)
    visual_change = bool(record.diff.num_changed > 0)
    signals: List[str] = []
    if level_progressed:
        signals.append("level_progressed")
    if directed_transform:
        signals.append("source_transformed_to_target")
    if relation_appeared:
        signals.append("target_relation_appeared")
    if relation_preserved:
        signals.append("target_relation_preserved")
    if visual_change and not functional:
        signals.append("visual_change_only")
    return FunctionalOptionProgress(
        expected_outcome_observed=expected,
        functional_progress=functional,
        level_progressed=level_progressed,
        visual_change=visual_change,
        visual_only_change=bool(visual_change and not functional),
        source_reduction=source_before - source_after,
        target_gain=target_after - target_before,
        source_to_target_pixels=source_to_target,
        relation_appeared=relation_appeared,
        relation_preserved=relation_preserved,
        signals=tuple(signals),
    )


def observed_rule_outcome(update: Any, rule: PromotedRelationalRule) -> str:
    record = update.record
    before = np.asarray(record.obs_before.raw_grid, dtype=np.int32)
    after = np.asarray(record.obs_after.raw_grid, dtype=np.int32)
    if rule.family == "color_transform" and rule.target_color is not None:
        mask = (before != after) & (before == rule.source_color)
        if not bool(mask.any()):
            return "none"
        values, counts = np.unique(after[mask], return_counts=True)
        return f"{rule.source_color}->{int(values[int(np.argmax(counts))])}"
    if rule.family == "object_count":
        changed = (
            len(record.obs_before.objects) != len(record.obs_after.objects)
            or bool(record.diff.created_objects)
            or bool(record.diff.removed_objects)
        )
        return "changed" if changed else "stable"
    if rule.family == "effect_scope":
        if not record.diff.changed_cells:
            return "none"
        if record.action.x is None or record.action.y is None:
            return "global" if record.diff.num_changed >= 10 else "local"
        local = sum(
            1
            for row, col in record.diff.changed_cells
            if max(
                abs(int(row) - int(record.action.y)),
                abs(int(col) - int(record.action.x)),
            )
            <= 8
        )
        return "local" if local / max(1, record.diff.num_changed) >= 0.8 else "global"
    if rule.family == "relation" and rule.target_color is not None:
        before_holds = relation_holds(
            record.obs_before,
            rule.predicate,
            rule.source_color,
            rule.target_color,
        )
        after_holds = relation_holds(
            record.obs_after,
            rule.predicate,
            rule.source_color,
            rule.target_color,
        )
        if not before_holds and after_holds:
            return "appears"
        if before_holds and not after_holds:
            return "broken"
        if before_holds and after_holds:
            return "preserved"
        return "absent"
    return "unobservable"


def relation_holds(
    observation: GameObservation,
    predicate: str,
    source_color: int,
    target_color: int,
) -> bool:
    source = [obj for obj in observation.objects if obj.value == int(source_color)]
    target = [obj for obj in observation.objects if obj.value == int(target_color)]
    if not source or not target:
        return False
    predicate = str(predicate).lower()
    if predicate == "paired_with":
        return True
    if predicate == "same_shape":
        return any(
            first.shape_signature == second.shape_signature and first.area == second.area
            for first in source
            for second in target
        )
    if predicate == "aligned_with":
        return any(
            int(round(first.center[0])) == int(round(second.center[0]))
            or int(round(first.center[1])) == int(round(second.center[1]))
            for first in source
            for second in target
        )
    if predicate == "adjacent_to":
        return any(
            _objects_adjacent(first.cells, second.cells)
            for first in source
            for second in target
        )
    return False


def _compile_rule(rule: PromotedRelationalRule) -> CompiledRelationalOption:
    required = [f"color_{rule.source_color}_present"]
    postconditions: List[str] = []
    if rule.target_color is not None:
        required.append(f"color_{rule.target_color}_present")
    if rule.family == "relation" and rule.target_color is not None:
        relation = _relation_predicate(rule)
        if rule.expected_outcome == "appears":
            required.append(f"{relation}_absent")
            postconditions.append(f"{relation}_present")
        elif rule.expected_outcome == "preserved":
            required.append(f"{relation}_present")
            postconditions.append(f"{relation}_present")
    if rule.family == "color_transform" and rule.target_color is not None:
        postconditions.extend(
            (
                f"color_{rule.target_color}_present",
                f"color_{rule.source_color}_transformed_to_{rule.target_color}",
            )
        )
        # The target color is an effect, not a prerequisite, for transforms.
        required = [f"color_{rule.source_color}_present"]
    return CompiledRelationalOption(
        option_id=f"option::{rule.key}",
        rule_key=rule.key,
        action=rule.action,
        source_color=rule.source_color,
        target_color=rule.target_color,
        family=rule.family,
        predicate=rule.predicate,
        expected_outcome=rule.expected_outcome,
        required_predicates=tuple(required),
        postconditions=tuple(postconditions),
    )


def _live_predicates(
    option: CompiledRelationalOption,
    observation: GameObservation,
) -> set[str]:
    predicates = {
        f"color_{obj.value}_present" for obj in observation.objects
    }
    if option.target_color is not None and option.family == "relation":
        name = _relation_predicate(option)
        holds = relation_holds(
            observation,
            option.predicate,
            option.source_color,
            option.target_color,
        )
        predicates.add(f"{name}_{'present' if holds else 'absent'}")
    return predicates


def _postcondition_satisfied(
    option: CompiledRelationalOption,
    present: set[str],
) -> bool:
    if option.family == "relation" and option.expected_outcome == "appears":
        return bool(option.postconditions) and all(
            predicate in present for predicate in option.postconditions
        )
    return False


def _relation_predicate(option_or_rule: Any) -> str:
    target = getattr(option_or_rule, "target_color", None)
    return (
        f"relation_{str(getattr(option_or_rule, 'predicate', 'paired_with')).lower()}_"
        f"colors{int(getattr(option_or_rule, 'source_color'))}_{int(target)}"
    )


def _objects_adjacent(
    first_cells: Sequence[Tuple[int, int]],
    second_cells: Sequence[Tuple[int, int]],
) -> bool:
    second = set(second_cells)
    return any(
        (row + drow, col + dcol) in second
        for row, col in first_cells
        for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1))
    )


__all__ = [
    "CompiledRelationalOption",
    "FunctionalOptionProgress",
    "OnlineRelationalOptionCompiler",
    "OptionAssessment",
    "OptionExecutionMemory",
    "OptionOutcomeStats",
    "observe_option_progress",
    "observed_rule_outcome",
    "relation_holds",
]
