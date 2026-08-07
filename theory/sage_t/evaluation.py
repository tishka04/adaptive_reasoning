"""Counterfactual replay evaluation and promotion gates for SAGE.T."""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import mean, stdev

from .contracts import AbstractState, ObservedTransition, PredictionPacket
from .decision import CounterfactualDecisionEngine
from .executor import ProgramExecutor
from .posterior import (
    DEFAULT_CHANNEL_WEIGHTS,
    ProgramPosterior,
    packet_log_likelihood,
)
from .synthesis import AssembledProgram


@dataclass(frozen=True)
class CounterfactualPanel:
    """Arms sharing the exact same abstract pre-state."""

    panel_id: str
    state: AbstractState
    arms: tuple[ObservedTransition, ...]
    source_game: str = ""
    split: str = "source"

    def __post_init__(self) -> None:
        if len(self.arms) < 2:
            raise ValueError("a counterfactual panel needs at least two arms")
        if len({arm.action.key for arm in self.arms}) < 2:
            raise ValueError("counterfactual arms need distinct legal actions")
        if any(arm.state_before.signature != self.state.signature for arm in self.arms):
            raise ValueError("all counterfactual arms must share one pre-state")
        if str(self.split).strip().lower() != "source":
            raise ValueError("SAGE.T counterfactual evaluation is source-only")


@dataclass(frozen=True)
class ReplayPanelResult:
    panel_id: str
    source_game: str
    revealed_actions: tuple[str, ...]
    held_out_log_likelihood: float
    entropy_reduction: float
    execution_errors: int = 0


@dataclass(frozen=True)
class ReplayEvaluation:
    condition: str
    panels: tuple[ReplayPanelResult, ...]
    illegal_actions: int = 0
    forbidden_fields: int = 0

    @property
    def mean_log_likelihood(self) -> float:
        values = [
            panel.held_out_log_likelihood
            for panel in self.panels
            if math.isfinite(panel.held_out_log_likelihood)
        ]
        return mean(values) if values else float("-inf")

    @property
    def mean_entropy_reduction(self) -> float:
        if not self.panels:
            return 0.0
        return mean(panel.entropy_reduction for panel in self.panels)

    @property
    def execution_errors(self) -> int:
        return sum(panel.execution_errors for panel in self.panels)


@dataclass(frozen=True)
class CounterfactualGateReport:
    passed: bool
    checks: Mapping[str, bool]
    metrics: Mapping[str, float]
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActivePairResult:
    game_id: str
    seed: int
    sage_t_levels_completed_delta: int
    baseline_levels_completed_delta: int
    sage_t_actions: int
    baseline_actions: int
    sage_t_resets: int = 14
    baseline_resets: int = 14
    sage_t_game_overs: int = 0
    baseline_game_overs: int = 0
    illegal_actions: int = 0
    controller_errors: int = 0
    safety_veto_violations: int = 0
    protected_competence_violations: int = 0

    @property
    def paired_progress_delta(self) -> float:
        sage_actions = max(1, int(self.sage_t_actions))
        baseline_actions = max(1, int(self.baseline_actions))
        return (
            float(self.sage_t_levels_completed_delta) / sage_actions
            - float(self.baseline_levels_completed_delta) / baseline_actions
        )


@dataclass(frozen=True)
class ActiveGateReport:
    passed: bool
    checks: Mapping[str, bool]
    metrics: Mapping[str, float]
    reasons: tuple[str, ...] = ()


@dataclass
class SageTCounterfactualEvaluator:
    """Reveal at most five agent-chosen arms and score the untouched arms."""

    executor: ProgramExecutor = field(default_factory=ProgramExecutor)
    decision_engine: CounterfactualDecisionEngine = field(
        default_factory=CounterfactualDecisionEngine
    )
    maximum_reveals: int = 5

    def evaluate(
        self,
        programs: Sequence[AssembledProgram],
        panels: Sequence[CounterfactualPanel],
        *,
        condition: str = "joint",
        selection: str = "information",
        seed: int = 0,
    ) -> ReplayEvaluation:
        weights = _condition_weights(condition)
        results = []
        rng = random.Random(seed)
        illegal_actions = 0
        for panel in panels:
            posterior = ProgramPosterior(
                executor=self.executor,
                channel_weights=weights,
            )
            posterior.seed(programs, initial_state=panel.state)
            initial_entropy = posterior.entropy
            remaining = {arm.action.key: arm for arm in panel.arms}
            revealed = []
            errors = 0
            reveal_budget = min(
                max(0, int(self.maximum_reveals)),
                max(0, len(remaining) - 1),
            )
            for _ in range(reveal_budget):
                if selection == "random":
                    selected_key = rng.choice(tuple(sorted(remaining)))
                else:
                    decision = self.decision_engine.decide(
                        posterior,
                        panel.state,
                        tuple(arm.action for arm in remaining.values()),
                    )
                    if decision.action is None:
                        break
                    selected_key = decision.action.key
                    if selected_key not in remaining:
                        errors += 1
                        illegal_actions += 1
                        break
                evidence = remaining.pop(selected_key)
                posterior.observe(evidence)
                revealed.append(selected_key)
            held_out_scores = []
            for evidence in remaining.values():
                terms = []
                for particle in posterior.particles:
                    try:
                        prediction = self.executor.step(
                            particle.program,
                            evidence.state_before,
                            evidence.action,
                        )
                        likelihood = packet_log_likelihood(
                            prediction,
                            evidence.observation,
                            channel_weights=weights,
                        )
                    except (TypeError, ValueError, RuntimeError):
                        errors += 1
                        continue
                    terms.append(particle.log_weight + likelihood)
                if terms:
                    held_out_scores.append(_logsumexp(terms))
            results.append(
                ReplayPanelResult(
                    panel_id=panel.panel_id,
                    source_game=panel.source_game,
                    revealed_actions=tuple(revealed),
                    held_out_log_likelihood=mean(held_out_scores)
                    if held_out_scores
                    else float("-inf"),
                    entropy_reduction=initial_entropy - posterior.entropy,
                    execution_errors=errors,
                )
            )
        return ReplayEvaluation(
            condition=condition,
            panels=tuple(results),
            illegal_actions=illegal_actions,
            forbidden_fields=count_forbidden_program_fields(programs),
        )

    def evaluate_required_conditions(
        self,
        programs: Sequence[AssembledProgram],
        panels: Sequence[CounterfactualPanel],
        *,
        seed: int = 0,
    ) -> Mapping[str, ReplayEvaluation]:
        return {
            "joint": self.evaluate(
                programs,
                panels,
                condition="joint",
                selection="information",
                seed=seed,
            ),
            "dynamics_only": self.evaluate(
                programs,
                panels,
                condition="dynamics_only",
                selection="information",
                seed=seed,
            ),
            "random": self.evaluate(
                programs,
                panels,
                condition="joint",
                selection="random",
                seed=seed,
            ),
            "action_only": self.evaluate_action_only(
                panels,
                seed=seed,
            ),
        }

    def evaluate_action_only(
        self,
        panels: Sequence[CounterfactualPanel],
        *,
        seed: int = 0,
    ) -> ReplayEvaluation:
        """Non-program baseline conditioned only on the local action name."""

        rng = random.Random(seed)
        learned: dict[str, list[PredictionPacket]] = defaultdict(list)
        results = []
        for panel in panels:
            remaining = {arm.action.key: arm for arm in panel.arms}
            reveal_budget = min(
                max(0, int(self.maximum_reveals)),
                max(0, len(remaining) - 1),
            )
            revealed = []
            for _ in range(reveal_budget):
                selected_key = rng.choice(tuple(sorted(remaining)))
                evidence = remaining.pop(selected_key)
                learned[evidence.action.action_name].append(evidence.observation)
                revealed.append(selected_key)
            scores = []
            for evidence in remaining.values():
                predicted = _mean_packet(learned.get(evidence.action.action_name, ()))
                scores.append(
                    packet_log_likelihood(
                        predicted,
                        evidence.observation,
                    )
                )
            results.append(
                ReplayPanelResult(
                    panel_id=panel.panel_id,
                    source_game=panel.source_game,
                    revealed_actions=tuple(revealed),
                    held_out_log_likelihood=(mean(scores) if scores else float("-inf")),
                    entropy_reduction=0.0,
                )
            )
        return ReplayEvaluation(
            condition="action_only",
            panels=tuple(results),
        )


def counterfactual_gate(
    evaluations: Mapping[str, ReplayEvaluation],
    *,
    baseline_log_likelihood: float,
    paired_interval_lower: float,
    non_negative_games: int,
) -> CounterfactualGateReport:
    joint = evaluations["joint"]
    dynamics = evaluations["dynamics_only"]
    random_result = evaluations["random"]
    action_only = evaluations.get("action_only")
    checks = {
        "better_than_baseline": (
            joint.mean_log_likelihood > float(baseline_log_likelihood)
            and float(paired_interval_lower) > 0.0
        ),
        "information_beats_random": (
            joint.mean_entropy_reduction > random_result.mean_entropy_reduction
        ),
        "joint_beats_dynamics_only": (
            joint.mean_log_likelihood > dynamics.mean_log_likelihood
        ),
        "joint_beats_action_only": (
            action_only is None
            or joint.mean_log_likelihood > action_only.mean_log_likelihood
        ),
        "two_of_three_games_non_negative": int(non_negative_games) >= 2,
        "zero_forbidden_fields": joint.forbidden_fields == 0,
        "zero_illegal_actions": joint.illegal_actions == 0,
        "zero_execution_errors": joint.execution_errors == 0,
    }
    reasons = tuple(name for name, passed in checks.items() if not passed)
    return CounterfactualGateReport(
        passed=all(checks.values()),
        checks=checks,
        metrics={
            "joint_log_likelihood": joint.mean_log_likelihood,
            "dynamics_only_log_likelihood": dynamics.mean_log_likelihood,
            "random_entropy_reduction": random_result.mean_entropy_reduction,
            "joint_entropy_reduction": joint.mean_entropy_reduction,
            "action_only_log_likelihood": (
                float("nan") if action_only is None else action_only.mean_log_likelihood
            ),
            "paired_interval_lower": float(paired_interval_lower),
        },
        reasons=reasons,
    )


def active_progress_gate(
    pairs: Sequence[ActivePairResult],
    *,
    configuration_frozen: bool,
    expected_games: Sequence[str] = ("re86", "ls20", "sc25"),
    expected_seeds_per_game: int = 5,
    expected_actions: int = 1000,
    expected_resets: int = 14,
) -> ActiveGateReport:
    """Strict paired promotion gate; it never opens or reads the holdout."""

    deltas = [pair.paired_progress_delta for pair in pairs]
    average = mean(deltas) if deltas else float("-inf")
    standard_error = (
        stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) >= 2 else float("inf")
    )
    interval_lower = average - 1.96 * standard_error
    expected = {str(game) for game in expected_games}
    counts = {game: sum(pair.game_id == game for pair in pairs) for game in expected}
    game_means = {
        game: mean(pair.paired_progress_delta for pair in pairs if pair.game_id == game)
        if counts[game]
        else float("-inf")
        for game in expected
    }
    checks = {
        "configuration_frozen": bool(configuration_frozen),
        "complete_paired_design": (
            {pair.game_id for pair in pairs} == expected
            and all(count == int(expected_seeds_per_game) for count in counts.values())
            and all(
                len(
                    {
                        pair.seed
                        for pair in pairs
                        if pair.game_id == game
                    }
                )
                == int(expected_seeds_per_game)
                for game in expected
            )
        ),
        "fixed_action_and_reset_budgets": all(
            pair.sage_t_actions == int(expected_actions)
            and pair.baseline_actions == int(expected_actions)
            and pair.sage_t_resets == int(expected_resets)
            and pair.baseline_resets == int(expected_resets)
            for pair in pairs
        ),
        "positive_95pct_lower_bound": interval_lower > 0.0,
        "two_of_three_games_non_negative": (
            sum(value >= 0.0 for value in game_means.values()) >= 2
        ),
        "zero_illegal_actions": (sum(pair.illegal_actions for pair in pairs) == 0),
        "zero_controller_errors": (sum(pair.controller_errors for pair in pairs) == 0),
        "game_over_not_worse": (
            sum(pair.sage_t_game_overs for pair in pairs)
            <= sum(pair.baseline_game_overs for pair in pairs)
        ),
        "all_safety_vetoes_respected": (
            sum(pair.safety_veto_violations for pair in pairs) == 0
        ),
        "all_protected_competences_respected": (
            sum(pair.protected_competence_violations for pair in pairs) == 0
        ),
    }
    reasons = tuple(name for name, passed in checks.items() if not passed)
    return ActiveGateReport(
        passed=all(checks.values()),
        checks=checks,
        metrics={
            "paired_progress_mean": average,
            "paired_interval_lower_95": interval_lower,
            "sage_t_game_overs": float(sum(pair.sage_t_game_overs for pair in pairs)),
            "baseline_game_overs": float(
                sum(pair.baseline_game_overs for pair in pairs)
            ),
            **{
                f"{game}_paired_progress_mean": value
                for game, value in game_means.items()
            },
        },
        reasons=reasons,
    )


def panels_from_transitions(
    transitions: Sequence[ObservedTransition],
    *,
    source_game: str = "",
) -> tuple[CounterfactualPanel, ...]:
    grouped: dict[str, list[ObservedTransition]] = {}
    for transition in transitions:
        grouped.setdefault(
            transition.state_before.signature,
            [],
        ).append(transition)
    return tuple(
        CounterfactualPanel(
            panel_id=f"panel_{index:04d}",
            state=items[0].state_before,
            arms=tuple(items),
            source_game=source_game,
        )
        for index, items in enumerate(grouped.values())
        if len({item.action.key for item in items}) >= 2
    )


def panels_from_binding_pairs(
    pairs: Sequence[object],
    *,
    maximum_panels: int | None = None,
) -> tuple[CounterfactualPanel, ...]:
    """Adapt frozen SAGE12 V4.3 same-prestate pairs without opening holdouts."""

    from theory.live_transition_loop import build_transition_record

    from .compiler import compile_transition_record

    if maximum_panels is not None and int(maximum_panels) <= 0:
        return ()
    panels = []
    for pair in pairs:
        split = str(getattr(pair, "source_split", ""))
        if split not in {"source_train", "source_validation"}:
            raise ValueError("SAGE.T V4.3 adapter is source-only")
        arms = []
        for branch_name in ("left", "right"):
            branch = getattr(pair, branch_name)
            trace = branch.trace
            record = build_transition_record(
                action=trace.selected_action_name,
                action_args=dict(getattr(trace, "selected_action_data", {})),
                grid_before=trace.frame_before,
                grid_after=trace.frame_after,
                available_actions=tuple(trace.available_action_names),
                game_state_before=getattr(
                    trace,
                    "game_state_before",
                    "NOT_FINISHED",
                ),
                game_state_after=getattr(
                    trace,
                    "game_state_after",
                    "NOT_FINISHED",
                ),
                levels_completed_before=int(
                    getattr(trace, "levels_completed_before", 0)
                ),
                levels_completed_after=int(getattr(trace, "levels_completed_after", 0)),
                timestamp=int(getattr(trace, "step_index", 0)),
            )
            arms.append(
                compile_transition_record(
                    record,
                    source_game_id=str(getattr(pair, "game_id", "")),
                )
            )
        panels.append(
            CounterfactualPanel(
                panel_id=str(
                    getattr(
                        pair,
                        "pair_digest",
                        f"binding_pair_{len(panels):06d}",
                    )
                ),
                state=arms[0].state_before,
                arms=tuple(arms),
                source_game=str(getattr(pair, "game_id", "")),
            )
        )
        if maximum_panels is not None and len(panels) >= max(0, int(maximum_panels)):
            break
    return tuple(panels)


def count_forbidden_program_fields(
    programs: Sequence[AssembledProgram],
) -> int:
    """Audit executable canonical payloads for transfer-prohibited constants."""

    def strings(value: object):
        if isinstance(value, str):
            yield value
        elif isinstance(value, Mapping):
            for key, item in value.items():
                yield str(key)
                yield from strings(item)
        elif isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes),
        ):
            for item in value:
                yield from strings(item)

    count = 0
    for assembled in programs:
        for raw in strings(assembled.program.canonical_payload):
            value = raw.lower()
            if (
                (
                    value != "game_over"
                    and value.startswith(("game_", "color_", "pixel_", "value_"))
                )
                or re.fullmatch(r"e(?:_player|[0-9]+(?::.*)?)", raw)
                or re.fullmatch(r"-?[0-9]+,-?[0-9]+", raw)
            ):
                count += 1
    return count


def _condition_weights(condition: str) -> Mapping[str, float]:
    normalized = str(condition).strip().lower()
    if normalized == "joint":
        return DEFAULT_CHANNEL_WEIGHTS
    if normalized == "dynamics_only":
        return {
            **DEFAULT_CHANNEL_WEIGHTS,
            "progress": 0.0,
            "goal": 0.0,
        }
    raise ValueError(f"unsupported replay condition: {condition}")


def _mean_packet(packets: Sequence[PredictionPacket]) -> PredictionPacket:
    if not packets:
        return PredictionPacket()
    known = set(packets[0].known_channels)
    for packet in packets[1:]:
        known.intersection_update(packet.known_channels)

    def events(attribute: str) -> dict[str, float]:
        keys = {key for packet in packets for key in dict(getattr(packet, attribute))}
        return {
            key: mean(
                float(dict(getattr(packet, attribute)).get(key, 0.05))
                for packet in packets
            )
            for key in keys
        }

    def scalar(attribute: str) -> float | None:
        values = [
            float(value)
            for packet in packets
            for value in (getattr(packet, attribute),)
            if value is not None
        ]
        return mean(values) if values else None

    return PredictionPacket(
        object_deltas=events("object_deltas"),
        relation_deltas=events("relation_deltas"),
        topology_deltas=events("topology_deltas"),
        progress_mean=scalar("progress_mean"),
        terminal_probability=scalar("terminal_probability"),
        goal_probability=scalar("goal_probability"),
        known_channels=frozenset(known),
    )


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        return float("-inf")
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


__all__ = [
    "ActiveGateReport",
    "ActivePairResult",
    "CounterfactualGateReport",
    "CounterfactualPanel",
    "ReplayEvaluation",
    "ReplayPanelResult",
    "SageTCounterfactualEvaluator",
    "active_progress_gate",
    "count_forbidden_program_fields",
    "counterfactual_gate",
    "panels_from_binding_pairs",
    "panels_from_transitions",
]
