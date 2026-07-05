"""A31 stress test for the A30 multi-game evaluation harness.

A31 does not add mechanics. It repeatedly runs the A30 per-game evaluation
under increasing experiment budgets, records typed failures, and exposes where
the current architecture stops producing useful evidence.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

from .multi_game_evaluation import (
    DEFAULT_TRACES_DIR,
    EvaluationResult,
    EvaluationTrace,
    evaluate_game,
)

DEFAULT_STRESS_BUDGETS = (5, 10, 25, 50)
DEFAULT_MAX_TRACES = 0
PredicateGenerator = Callable[..., Iterable[str]]
AnchorExpander = Callable[..., Iterable[Any]]
CandidateRanker = Callable[..., Sequence[Any]]


@dataclass
class StressCurvePoint:
    """A31 metrics for one game trace at one experiment budget."""

    game_id: str
    trace_path: Path
    experiment_budget: int
    attempts: int = 0
    experiments_run: int = 0
    hypotheses_confirmed: int = 0
    hypotheses_refuted: int = 0
    useful_new_states: int = 0
    negative_memory_contexts_avoided: int = 0
    wrong_confirmations: int = 0
    skips: int = 0
    failures: int = 0
    failure_types: Dict[str, int] = field(default_factory=dict)
    trace_support_counted_as_proof: bool = False
    stopped_reason: str = ""

    @property
    def confirmations_per_experiment(self) -> float:
        return _safe_ratio(self.hypotheses_confirmed, self.experiments_run)

    @property
    def refutations_per_experiment(self) -> float:
        return _safe_ratio(self.hypotheses_refuted, self.experiments_run)

    @property
    def useful_new_states_per_experiment(self) -> float:
        return _safe_ratio(self.useful_new_states, self.experiments_run)

    @property
    def updates_per_experiment(self) -> float:
        return _safe_ratio(
            self.hypotheses_confirmed + self.hypotheses_refuted,
            self.experiments_run,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "trace_path": str(self.trace_path),
            "experiment_budget": self.experiment_budget,
            "attempts": self.attempts,
            "experiments_run": self.experiments_run,
            "hypotheses_confirmed": self.hypotheses_confirmed,
            "hypotheses_refuted": self.hypotheses_refuted,
            "confirmations_per_experiment": round(
                self.confirmations_per_experiment,
                4,
            ),
            "refutations_per_experiment": round(
                self.refutations_per_experiment,
                4,
            ),
            "updates_per_experiment": round(self.updates_per_experiment, 4),
            "useful_new_states": self.useful_new_states,
            "useful_new_states_per_experiment": round(
                self.useful_new_states_per_experiment,
                4,
            ),
            "negative_memory_contexts_avoided": (
                self.negative_memory_contexts_avoided
            ),
            "wrong_confirmations": self.wrong_confirmations,
            "skips": self.skips,
            "failures": self.failures,
            "failure_types": dict(sorted(self.failure_types.items())),
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "stopped_reason": self.stopped_reason,
        }


@dataclass
class StressBudgetSummary:
    """Aggregate curve point for one experiment budget across traces."""

    experiment_budget: int
    points: List[StressCurvePoint] = field(default_factory=list)

    @property
    def traces_evaluated(self) -> int:
        return len(self.points)

    @property
    def experiments_run(self) -> int:
        return sum(point.experiments_run for point in self.points)

    @property
    def hypotheses_confirmed(self) -> int:
        return sum(point.hypotheses_confirmed for point in self.points)

    @property
    def hypotheses_refuted(self) -> int:
        return sum(point.hypotheses_refuted for point in self.points)

    @property
    def useful_new_states(self) -> int:
        return sum(point.useful_new_states for point in self.points)

    @property
    def negative_memory_contexts_avoided(self) -> int:
        return sum(point.negative_memory_contexts_avoided for point in self.points)

    @property
    def wrong_confirmations(self) -> int:
        return sum(point.wrong_confirmations for point in self.points)

    @property
    def skips(self) -> int:
        return sum(point.skips for point in self.points)

    @property
    def failures(self) -> int:
        return sum(point.failures for point in self.points)

    @property
    def failure_types(self) -> Dict[str, int]:
        return _merge_counters(point.failure_types for point in self.points)

    @property
    def confirmations_per_experiment(self) -> float:
        return _safe_ratio(self.hypotheses_confirmed, self.experiments_run)

    @property
    def refutations_per_experiment(self) -> float:
        return _safe_ratio(self.hypotheses_refuted, self.experiments_run)

    @property
    def useful_new_states_per_experiment(self) -> float:
        return _safe_ratio(self.useful_new_states, self.experiments_run)

    @property
    def updates_per_experiment(self) -> float:
        return _safe_ratio(
            self.hypotheses_confirmed + self.hypotheses_refuted,
            self.experiments_run,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_budget": self.experiment_budget,
            "traces_evaluated": self.traces_evaluated,
            "experiments_run": self.experiments_run,
            "hypotheses_confirmed": self.hypotheses_confirmed,
            "hypotheses_refuted": self.hypotheses_refuted,
            "confirmations_per_experiment": round(
                self.confirmations_per_experiment,
                4,
            ),
            "refutations_per_experiment": round(
                self.refutations_per_experiment,
                4,
            ),
            "updates_per_experiment": round(self.updates_per_experiment, 4),
            "useful_new_states": self.useful_new_states,
            "useful_new_states_per_experiment": round(
                self.useful_new_states_per_experiment,
                4,
            ),
            "negative_memory_contexts_avoided": (
                self.negative_memory_contexts_avoided
            ),
            "wrong_confirmations": self.wrong_confirmations,
            "skips": self.skips,
            "failures": self.failures,
            "failure_types": dict(sorted(self.failure_types.items())),
        }


@dataclass
class MultiGameStressTestResult:
    """A31 result: full stress curve across budgets and traces."""

    budgets: List[int] = field(default_factory=list)
    traces: List[EvaluationTrace] = field(default_factory=list)
    curve_points: List[StressCurvePoint] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def trace_count(self) -> int:
        return len(self.traces)

    @property
    def game_count(self) -> int:
        return len({trace.game_id for trace in self.traces})

    @property
    def budget_summaries(self) -> List[StressBudgetSummary]:
        summaries: List[StressBudgetSummary] = []
        for budget in self.budgets:
            summaries.append(
                StressBudgetSummary(
                    experiment_budget=budget,
                    points=[
                        point
                        for point in self.curve_points
                        if point.experiment_budget == budget
                    ],
                )
            )
        return summaries

    @property
    def experiments_run(self) -> int:
        return sum(point.experiments_run for point in self.curve_points)

    @property
    def hypotheses_confirmed(self) -> int:
        return sum(point.hypotheses_confirmed for point in self.curve_points)

    @property
    def hypotheses_refuted(self) -> int:
        return sum(point.hypotheses_refuted for point in self.curve_points)

    @property
    def useful_new_states(self) -> int:
        return sum(point.useful_new_states for point in self.curve_points)

    @property
    def negative_memory_contexts_avoided(self) -> int:
        return sum(
            point.negative_memory_contexts_avoided for point in self.curve_points
        )

    @property
    def wrong_confirmations(self) -> int:
        return sum(point.wrong_confirmations for point in self.curve_points)

    @property
    def failure_types(self) -> Dict[str, int]:
        return _merge_counters(point.failure_types for point in self.curve_points)

    @property
    def wrong_confirmations_zero(self) -> bool:
        return self.wrong_confirmations == 0

    @property
    def failures_are_typed(self) -> bool:
        return bool(
            self.failure_types or all(point.failures == 0 for point in self.curve_points)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "budgets": list(self.budgets),
            "trace_count": self.trace_count,
            "game_count": self.game_count,
            "experiments_run": self.experiments_run,
            "hypotheses_confirmed": self.hypotheses_confirmed,
            "hypotheses_refuted": self.hypotheses_refuted,
            "useful_new_states": self.useful_new_states,
            "negative_memory_contexts_avoided": (
                self.negative_memory_contexts_avoided
            ),
            "wrong_confirmations": self.wrong_confirmations,
            "wrong_confirmations_zero": self.wrong_confirmations_zero,
            "failures_are_typed": self.failures_are_typed,
            "failure_types": dict(sorted(self.failure_types.items())),
            "traces": [trace.to_dict() for trace in self.traces],
            "budget_summaries": [
                summary.to_dict() for summary in self.budget_summaries
            ],
            "curve_points": [point.to_dict() for point in self.curve_points],
            "errors": list(self.errors),
        }


def run_multi_game_stress_test(
    *,
    traces_dir: Path | str = DEFAULT_TRACES_DIR,
    trace_paths: Sequence[Path | str] = (),
    budgets: Sequence[int] = DEFAULT_STRESS_BUDGETS,
    max_traces: int = DEFAULT_MAX_TRACES,
    include_ar25: bool = True,
    latest_per_game: bool = False,
    environments_dir: Path | str | None = None,
    max_candidates: int = 20,
    min_pixel_support: int = 1,
    run_memory_comparison: bool = True,
    max_attempts_per_budget: int | None = None,
    stagnation_limit: int = 2,
    ar25_revise_every: int = 20,
    predicate_generator: PredicateGenerator | None = None,
    anchor_expander: AnchorExpander | None = None,
    candidate_ranker: CandidateRanker | None = None,
    preferred_predicates: Sequence[str] | None = None,
) -> MultiGameStressTestResult:
    """Run A30 repeatedly across traces and experiment budgets."""
    selected_budgets = _normalize_budgets(budgets)
    traces = select_stress_traces(
        traces_dir=traces_dir,
        trace_paths=trace_paths,
        max_traces=max_traces,
        include_ar25=include_ar25,
        latest_per_game=latest_per_game,
    )
    result = MultiGameStressTestResult(
        budgets=selected_budgets,
        traces=traces,
    )
    for budget in selected_budgets:
        for trace in traces:
            point = run_stress_curve_point(
                trace,
                experiment_budget=budget,
                environments_dir=environments_dir,
                max_candidates=max_candidates,
                min_pixel_support=min_pixel_support,
                run_memory_comparison=run_memory_comparison,
                max_attempts=max_attempts_per_budget,
                stagnation_limit=stagnation_limit,
                ar25_revise_every=ar25_revise_every,
                predicate_generator=predicate_generator,
                anchor_expander=anchor_expander,
                candidate_ranker=candidate_ranker,
                preferred_predicates=preferred_predicates,
            )
            result.curve_points.append(point)
    if not traces:
        result.errors.append("no_stress_traces_selected")
    return result


def run_stress_curve_point(
    trace: EvaluationTrace,
    *,
    experiment_budget: int,
    environments_dir: Path | str | None = None,
    max_candidates: int = 20,
    min_pixel_support: int = 1,
    run_memory_comparison: bool = True,
    max_attempts: int | None = None,
    stagnation_limit: int = 2,
    ar25_revise_every: int = 20,
    predicate_generator: PredicateGenerator | None = None,
    anchor_expander: AnchorExpander | None = None,
    candidate_ranker: CandidateRanker | None = None,
    preferred_predicates: Sequence[str] | None = None,
) -> StressCurvePoint:
    """Repeatedly run A30's per-game evaluation until budget or blockage."""
    target_budget = max(1, int(experiment_budget))
    attempt_limit = (
        max_attempts
        if max_attempts is not None
        else max(1, target_budget)
    )
    point = StressCurvePoint(
        game_id=trace.game_id,
        trace_path=trace.trace_path,
        experiment_budget=target_budget,
    )
    repeated_blocker = ""
    repeated_blocker_count = 0

    while point.experiments_run < target_budget and point.attempts < attempt_limit:
        point.attempts += 1
        attempt = evaluate_game(
            trace.game_id,
            trace.trace_path,
            environments_dir=environments_dir,
            max_candidates=max_candidates,
            min_pixel_support=min_pixel_support,
            run_memory_comparison=run_memory_comparison,
            ar25_budget=target_budget,
            ar25_revise_every=ar25_revise_every,
            predicate_generator=predicate_generator,
            anchor_expander=anchor_expander,
            candidate_ranker=candidate_ranker,
            preferred_predicates=preferred_predicates,
        )
        _accumulate_attempt(point, attempt)

        blocker = _dominant_blocker(attempt)
        if attempt.experiments_run <= 0:
            point.skips += 1
        if blocker:
            if blocker == repeated_blocker:
                repeated_blocker_count += 1
            else:
                repeated_blocker = blocker
                repeated_blocker_count = 1
            if repeated_blocker_count >= max(1, int(stagnation_limit)):
                point.stopped_reason = f"blocked:{blocker}"
                break
        elif attempt.experiments_run <= 0:
            point.stopped_reason = "blocked:no_experiments_produced"
            break
        else:
            repeated_blocker = ""
            repeated_blocker_count = 0

    if not point.stopped_reason:
        point.stopped_reason = (
            "budget_reached"
            if point.experiments_run >= target_budget
            else "max_attempts_reached"
        )
    return point


def select_stress_traces(
    *,
    traces_dir: Path | str = DEFAULT_TRACES_DIR,
    trace_paths: Sequence[Path | str] = (),
    max_traces: int = DEFAULT_MAX_TRACES,
    include_ar25: bool = True,
    latest_per_game: bool = False,
) -> List[EvaluationTrace]:
    """Select stress traces, optionally keeping only the latest per game."""
    paths = (
        [Path(path) for path in trace_paths]
        if trace_paths
        else sorted(Path(traces_dir).glob("*.steps.jsonl"))
    )
    traces: List[EvaluationTrace] = []
    for path in paths:
        game_id = _game_id_from_trace(path)
        if not game_id:
            continue
        if not include_ar25 and game_id.startswith("ar25"):
            continue
        traces.append(EvaluationTrace(game_id=game_id, trace_path=path))

    if latest_per_game:
        by_game: dict[str, EvaluationTrace] = {}
        for trace in traces:
            previous = by_game.get(trace.game_id)
            if previous is None or trace.trace_path.name > previous.trace_path.name:
                by_game[trace.game_id] = trace
        traces = [by_game[game_id] for game_id in sorted(by_game)]

    limit = int(max_traces)
    if limit > 0:
        return traces[:limit]
    return traces


def classify_failure(error: str) -> str:
    """Normalize runner errors into A31 blockage types."""
    text = str(error or "").strip()
    lowered = text.lower()
    if not lowered:
        return ""
    if "not_enough_relation_candidates" in lowered:
        return "not_enough_relation_candidates"
    if "no_live_compatible_hypothesis" in lowered:
        return "no_live_compatible_hypothesis"
    if "no_functional_progress" in lowered:
        return "no_functional_progress"
    if "local_relation_agenda_not_observed" in lowered:
        return "relation_agenda_not_observed"
    if "relation_agenda_not_observed" in lowered:
        return "relation_agenda_not_observed"
    if "env_setup_failed" in lowered:
        return "env_setup_failed"
    if "env_step_failed" in lowered or "env_step_returned_none" in lowered:
        return "env_step_failed"
    if "no_target_relation_experiment" in lowered:
        return "no_target_relation_experiment"
    if "missing_active_transition" in lowered:
        return "missing_active_transition"
    if "target_hypothesis_resolved_before_observation" in lowered:
        return "hypothesis_resolved_before_observation"
    if ":" in text:
        return text.split(":", 1)[-1].strip() or "unknown_failure"
    return text


def _accumulate_attempt(
    point: StressCurvePoint,
    attempt: EvaluationResult,
) -> None:
    point.experiments_run += int(attempt.experiments_run)
    point.hypotheses_confirmed += int(attempt.hypotheses_confirmed)
    point.hypotheses_refuted += int(attempt.hypotheses_refuted)
    point.useful_new_states += int(attempt.useful_new_states)
    point.negative_memory_contexts_avoided += int(
        attempt.negative_memory_contexts_avoided
    )
    point.wrong_confirmations += int(attempt.wrong_confirmations)
    point.trace_support_counted_as_proof = bool(
        point.trace_support_counted_as_proof
        or attempt.trace_support_counted_as_proof
    )
    for error in attempt.errors:
        kind = classify_failure(error)
        if not kind:
            continue
        point.failures += 1
        point.failure_types[kind] = point.failure_types.get(kind, 0) + 1


def _dominant_blocker(attempt: EvaluationResult) -> str:
    for error in attempt.errors:
        kind = classify_failure(error)
        if kind:
            return kind
    if attempt.experiments_run <= 0:
        return "no_experiments_produced"
    return ""


def _normalize_budgets(budgets: Sequence[int]) -> List[int]:
    result: List[int] = []
    for budget in budgets:
        value = max(1, int(budget))
        if value not in result:
            result.append(value)
    return result or list(DEFAULT_STRESS_BUDGETS)


def _merge_counters(counters: Iterable[Dict[str, int]]) -> Dict[str, int]:
    merged: Counter[str] = Counter()
    for counter in counters:
        merged.update({str(key): int(value) for key, value in counter.items()})
    return dict(merged)


def _game_id_from_trace(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                game_id = str(item.get("game_id", "")).strip()
                if game_id:
                    return game_id
                break
    except OSError:
        return ""
    return path.name.split(".")[0]


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _parse_budgets(raw: str) -> List[int]:
    return [
        int(item.strip())
        for item in str(raw).split(",")
        if item.strip()
    ]


def _parse_paths(values: Sequence[str]) -> List[Path]:
    return [Path(value) for value in values if value]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run A31 stress test over the A30 evaluation harness."
    )
    parser.add_argument("--traces-dir", type=Path, default=DEFAULT_TRACES_DIR)
    parser.add_argument("--trace-path", action="append", default=[])
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_STRESS_BUDGETS)))
    parser.add_argument("--max-traces", type=int, default=DEFAULT_MAX_TRACES)
    parser.add_argument("--exclude-ar25", action="store_true")
    parser.add_argument("--latest-per-game", action="store_true")
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--min-pixel-support", type=int, default=1)
    parser.add_argument("--skip-memory-comparison", action="store_true")
    parser.add_argument("--max-attempts-per-budget", type=int, default=None)
    parser.add_argument("--stagnation-limit", type=int, default=2)
    parser.add_argument("--ar25-revise-every", type=int, default=20)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_multi_game_stress_test(
        traces_dir=args.traces_dir,
        trace_paths=_parse_paths(args.trace_path),
        budgets=_parse_budgets(args.budgets),
        max_traces=args.max_traces,
        include_ar25=not args.exclude_ar25,
        latest_per_game=args.latest_per_game,
        environments_dir=args.environments_dir,
        max_candidates=args.max_candidates,
        min_pixel_support=args.min_pixel_support,
        run_memory_comparison=not args.skip_memory_comparison,
        max_attempts_per_budget=args.max_attempts_per_budget,
        stagnation_limit=args.stagnation_limit,
        ar25_revise_every=args.ar25_revise_every,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
