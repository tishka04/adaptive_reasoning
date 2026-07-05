"""Minimal semi-MDP options induced from confirmed theory.

An option is not a smarter scorer. It is a named abstraction with explicit
initiation, a fixed policy action, and a termination test grounded in the
same correspondence observations used by the belief loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .correspondence_hypothesis import (
    CorrespondenceObservation,
    CorrespondenceRule,
    normalize_pair_colors,
)
from .epistemic_metrics import EpistemicScore, HypothesisStatus, MechanicsOracle
from .live_transition_loop import LiveTransitionBeliefLoop
from .mechanic_hypothesis import GameTheory
from .precondition_hypothesis import (
    PreconditionHypothesis,
    PreconditionObservation,
    extract_precondition_predicates,
    target_action_from_rule_key,
)


@dataclass(frozen=True)
class TheoryOptionInvocation:
    """One verified execution of a theory option's policy action."""

    option_name: str
    target_rule: str
    precondition_key: str
    policy_action: str
    step: int
    actual_action: str
    predicates_present: List[str] = field(default_factory=list)
    termination: str = "unknown"
    match_delta: float = 0.0
    global_delta: float = 0.0
    level_complete: bool = False

    @property
    def success(self) -> bool:
        return self.termination == "success"

    @property
    def contradiction(self) -> bool:
        return self.termination == "contradiction"


@dataclass(frozen=True)
class TheoryOption:
    """A semi-MDP option backed by a confirmed correspondence rule."""

    name: str
    target_rule: str
    initiation_predicate: str
    precondition_key: str
    policy_action: str
    pair_colors: tuple[int, int]
    termination_signals: tuple[str, ...] = (
        "correspondence_count_improved",
        "level_progressed",
        "contradiction",
    )

    def can_initiate(
        self,
        theory: GameTheory,
        predicates_present: Iterable[str],
        available_actions: Sequence[str] | None = None,
    ) -> bool:
        if available_actions is not None and self.policy_action not in available_actions:
            return False
        for hypothesis in theory.preconditions_for_rule(self.target_rule):
            if hypothesis.key != self.precondition_key:
                continue
            return hypothesis.is_applicable(predicates_present)
        return False

    def policy(self) -> str:
        return self.policy_action

    def observe_termination(
        self,
        *,
        step: int,
        actual_action: str,
        predicates_present: Iterable[str],
        update: Any,
    ) -> TheoryOptionInvocation | None:
        actual = _normalize_action_name(actual_action)
        if actual != self.policy_action:
            return None
        observation = CorrespondenceObservation.from_transition(
            update.record,
            pair_colors=self.pair_colors,
        )
        succeeded = bool(observation.improves or update.effect.level_complete)
        return TheoryOptionInvocation(
            option_name=self.name,
            target_rule=self.target_rule,
            precondition_key=self.precondition_key,
            policy_action=self.policy_action,
            step=int(step),
            actual_action=actual,
            predicates_present=sorted(str(item) for item in predicates_present),
            termination="success" if succeeded else "contradiction",
            match_delta=observation.match_delta,
            global_delta=observation.global_delta,
            level_complete=bool(update.effect.level_complete),
        )


@dataclass
class TheoryOptionRun:
    """Result of replaying a trace with theory options enabled."""

    loop: LiveTransitionBeliefLoop
    invocations: List[TheoryOptionInvocation] = field(default_factory=list)
    options_seen: List[str] = field(default_factory=list)

    @property
    def option_invocations(self) -> int:
        return len(self.invocations)

    @property
    def successful_invocations(self) -> List[TheoryOptionInvocation]:
        return [invocation for invocation in self.invocations if invocation.success]

    @property
    def contradicted_invocations(self) -> List[TheoryOptionInvocation]:
        return [
            invocation for invocation in self.invocations
            if invocation.contradiction
        ]

    @property
    def termination_success_rate(self) -> float:
        if not self.invocations:
            return 0.0
        return len(self.successful_invocations) / len(self.invocations)

    def score(self, oracle: MechanicsOracle) -> EpistemicScore:
        return self.loop.score(oracle)

    def summary(self) -> Dict[str, Any]:
        return {
            "transitions": self.loop.transition_count,
            "options_seen": sorted(set(self.options_seen)),
            "option_invocations": self.option_invocations,
            "option_successes": len(self.successful_invocations),
            "option_contradictions": len(self.contradicted_invocations),
            "termination_success_rate": round(self.termination_success_rate, 4),
            "correspondence_rules": [
                rule.key for rule in self.loop.theory.correspondence_rules()
            ],
            "preconditions_confirmed": [
                hyp.key for hyp in self.loop.theory.precondition_hypotheses()
                if hyp.status == HypothesisStatus.CONFIRMED
            ],
        }


def build_options_from_theory(theory: GameTheory) -> List[TheoryOption]:
    """Promote confirmed correspondence + ready precondition into options."""
    options: List[TheoryOption] = []
    for rule in _ranked_correspondence_rules(theory):
        for precondition in theory.preconditions_for_rule(rule.key):
            if precondition.status != HypothesisStatus.CONFIRMED:
                continue
            if precondition.predicate != "ready_to_validate_correspondence":
                continue
            options.append(_option_from_rule(rule, precondition))
    return options


def run_option_trace_file(
    trace_path: Path,
    *,
    game_id: str = "",
    task_program_path: Path | None = None,
    max_steps: int | None = None,
    background_value: int | None = None,
    infer_players: bool = True,
    verify_every: int = 1,
    correspondence_pair_colors: tuple[int, int] = (10, 11),
) -> TheoryOptionRun:
    """Replay a trace and verify invocations of induced theory options."""
    pair_colors = normalize_pair_colors(correspondence_pair_colors)
    loop = LiveTransitionBeliefLoop(
        game_id=game_id,
        background_value=background_value,
        infer_players=infer_players,
        verify_every=verify_every,
        correspondence_pair_colors=pair_colors,
    )
    if task_program_path is not None:
        loop.seed_task_program(task_program_path)

    invocations: List[TheoryOptionInvocation] = []
    options_seen: List[str] = []
    steps_seen = 0
    previous_levels = 0
    previous_action = ""
    recent_actions: List[str] = []
    recent_correspondence_successes: List[bool] = []

    with open(Path(trace_path), "r", encoding="utf-8") as handle:
        for line in handle:
            if max_steps is not None and steps_seen >= max_steps:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            action = _normalize_action_name(item.get("action"))
            if action == "RESET":
                continue
            before = item.get("frame_before")
            after = item.get("frame_after")
            if before is None or after is None:
                continue

            available = _normalize_actions(item.get("available_actions") or [])
            loop.theory.seed_actions(available)
            features_by_rule = _precondition_features_for_current_rules(
                loop.theory,
                before,
                previous_action=previous_action,
                recent_actions=recent_actions,
                recent_correspondence_successes=recent_correspondence_successes,
            )
            option_candidates = [
                option for option in build_options_from_theory(loop.theory)
                if option.can_initiate(
                    loop.theory,
                    features_by_rule.get(option.target_rule, set()),
                    available_actions=available,
                )
            ]
            option = option_candidates[0] if option_candidates else None
            if option is not None:
                options_seen.append(option.name)

            update = loop.observe_grids(
                action=action,
                action_args=item.get("action_args"),
                grid_before=before,
                grid_after=after,
                available_actions=available,
                game_state_after=item.get("game_state_after", "NOT_FINISHED"),
                levels_completed_before=previous_levels,
                levels_completed_after=int(item.get("levels_completed_after", 0) or 0),
                timestamp=int(item.get("step", steps_seen) or 0),
                was_experiment=True,
            )

            if option is not None:
                invocation = option.observe_termination(
                    step=int(item.get("step", steps_seen) or 0),
                    actual_action=action,
                    predicates_present=features_by_rule.get(option.target_rule, set()),
                    update=update,
                )
                if invocation is not None:
                    invocations.append(invocation)

            correspondence_successes = _observe_preconditions_from_transition(
                loop.theory,
                action=action,
                precondition_features=features_by_rule,
                succeeded_by_rule=_correspondence_success_by_rule(
                    loop.theory,
                    update,
                    pair_colors=pair_colors,
                ),
                was_experiment=True,
            )
            recent_correspondence_successes.append(any(correspondence_successes))
            recent_correspondence_successes = recent_correspondence_successes[-4:]
            previous_levels = int(item.get("levels_completed_after", previous_levels) or 0)
            previous_action = action
            recent_actions.append(action)
            recent_actions = recent_actions[-4:]
            steps_seen += 1

    return TheoryOptionRun(
        loop=loop,
        invocations=invocations,
        options_seen=options_seen,
    )


def _option_from_rule(
    rule: CorrespondenceRule,
    precondition: PreconditionHypothesis,
) -> TheoryOption:
    first, second = rule.pair_colors
    return TheoryOption(
        name=f"validate_correspondence_colors{first}_{second}",
        target_rule=rule.key,
        initiation_predicate=precondition.predicate,
        precondition_key=precondition.key,
        policy_action=rule.action or target_action_from_rule_key(rule.key),
        pair_colors=rule.pair_colors,
    )


def _precondition_features_for_current_rules(
    theory: GameTheory,
    grid: Any,
    *,
    previous_action: str,
    recent_actions: Iterable[str],
    recent_correspondence_successes: Iterable[bool],
) -> Dict[str, set[str]]:
    features: Dict[str, set[str]] = {}
    for rule in theory.correspondence_rules():
        _ensure_default_preconditions(theory, rule.key)
        features[rule.key] = extract_precondition_predicates(
            grid,
            target_rule=rule.key,
            pair_colors=rule.pair_colors,
            previous_action=previous_action,
            recent_actions=recent_actions,
            recent_correspondence_successes=recent_correspondence_successes,
        )
    return features


def _ensure_default_preconditions(theory: GameTheory, target_rule: str) -> None:
    if theory.preconditions_for_rule(target_rule):
        return
    theory.add_precondition(PreconditionHypothesis(
        target_rule=target_rule,
        predicate="ready_to_validate_correspondence",
    ))


def _correspondence_success_by_rule(
    theory: GameTheory,
    update: Any,
    *,
    pair_colors: tuple[int, int],
) -> Dict[str, bool]:
    results: Dict[str, bool] = {}
    for rule in theory.correspondence_rules():
        observation = CorrespondenceObservation.from_transition(
            update.record,
            pair_colors=rule.pair_colors or pair_colors,
        )
        results[rule.key] = bool(observation.improves or update.effect.level_complete)
    return results


def _observe_preconditions_from_transition(
    theory: GameTheory,
    *,
    action: str,
    precondition_features: Dict[str, set[str]],
    succeeded_by_rule: Dict[str, bool],
    was_experiment: bool,
) -> List[bool]:
    successes: List[bool] = []
    for target_rule, predicates in precondition_features.items():
        succeeded = bool(succeeded_by_rule.get(target_rule, False))
        observation = PreconditionObservation(
            target_rule=target_rule,
            action=action,
            predicates_present=set(predicates),
            succeeded=succeeded,
        )
        theory.observe_precondition(observation, was_experiment=was_experiment)
        if observation.target_action_executed:
            successes.append(succeeded)
    return successes


def _ranked_correspondence_rules(theory: GameTheory) -> List[CorrespondenceRule]:
    return sorted(
        theory.correspondence_rules(),
        key=lambda rule: (rule.confidence, rule.support, rule.key),
        reverse=True,
    )


def _normalize_actions(actions: Iterable[Any]) -> List[str]:
    normalized: List[str] = []
    for action in actions:
        name = _normalize_action_name(action)
        if name and name not in normalized:
            normalized.append(name)
    return normalized


def _normalize_action_name(action: Any) -> str:
    raw = str(action or "").strip().upper()
    if raw.isdigit():
        return f"ACTION{raw}"
    return raw
