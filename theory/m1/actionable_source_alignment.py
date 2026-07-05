"""M1.3d-b short experiment-preparation search.

The search is intentionally conservative: it tries short offline preparation
sequences that make a blocked source selectable for the intended action. A
found precondition is still UNRESOLVED and is not counted as proof.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from theory.non_ar25_active_micro_run import (
    _configure_offline_env,
    _env_dir,
    _valid_actions,
)
from theory.real_env_option_adapter import snapshot_frame

from .live_anchor_ranking import _source_colors_by_action
from .source_reachability import (
    DEFAULT_SOURCE_REACHABILITY_OUTPUT_PATH,
    SourceAlignmentProblem,
    source_alignment_problem_from_dict,
)

DEFAULT_ACTIONABLE_ALIGNMENT_OUTPUT_PATH = (
    Path("diagnostics") / "m1" / "actionable_source_alignment_pretest.json"
)


@dataclass(frozen=True)
class PreparationAction:
    """One concrete action in an experimental preparation sequence."""

    name: str
    action_args: Dict[str, Any] = field(default_factory=dict)

    @property
    def signature(self) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
        return (
            self.name,
            tuple(sorted((str(key), str(value)) for key, value in self.action_args.items())),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "action_args": dict(self.action_args),
        }


@dataclass(frozen=True)
class ExperimentalPrecondition:
    """A candidate preparation that may make a source testable."""

    game_id: str
    source_problem: SourceAlignmentProblem
    prep_actions: Tuple[PreparationAction, ...]
    intended_effect: str = "make_source_actionable"
    then_test_pair: Tuple[str, int, int] = ("", 0, 0)
    source_becomes_actionable: bool = False
    target_still_present: bool = False
    final_available_sources: Tuple[int, ...] = ()
    prep_length: int = 0
    evaluated_nodes: int = 0
    status: str = "UNRESOLVED"
    trace_support_counted_as_proof: bool = False
    prior_counted_as_proof: bool = False

    def to_dict(self) -> Dict[str, Any]:
        action, source, target = self.then_test_pair
        return {
            "game_id": self.game_id,
            "source_problem": self.source_problem.to_dict(),
            "prep_actions": [action_item.to_dict() for action_item in self.prep_actions],
            "intended_effect": self.intended_effect,
            "then_test_pair": [action, int(source), int(target)],
            "source_becomes_actionable": bool(self.source_becomes_actionable),
            "target_still_present": bool(self.target_still_present),
            "final_available_sources": list(self.final_available_sources),
            "prep_length": int(self.prep_length),
            "evaluated_nodes": int(self.evaluated_nodes),
            "status": self.status,
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "prior_counted_as_proof": self.prior_counted_as_proof,
        }


@dataclass(frozen=True)
class SearchState:
    """A replayed environment state used by the bounded preparation search."""

    grid: np.ndarray
    actions: Tuple[Any, ...]


@dataclass(frozen=True)
class SearchResult:
    """Result of trying to prepare one SourceAlignmentProblem."""

    problem: SourceAlignmentProblem
    precondition: ExperimentalPrecondition | None
    evaluated_nodes: int
    exhausted: bool
    error: str = ""

    @property
    def found(self) -> bool:
        return self.precondition is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem": self.problem.to_dict(),
            "precondition": (
                self.precondition.to_dict() if self.precondition is not None else None
            ),
            "found": self.found,
            "evaluated_nodes": int(self.evaluated_nodes),
            "exhausted": bool(self.exhausted),
            "error": self.error,
        }


SequenceEvaluator = Callable[[Sequence[Any]], SearchState | None]


def load_source_alignment_problems(
    path: str | Path = DEFAULT_SOURCE_REACHABILITY_OUTPUT_PATH,
    *,
    source_scope: str = "ranked_new_pairs",
) -> Tuple[SourceAlignmentProblem, ...]:
    """Load typed source alignment problems from M1.3d-a output."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    key = (
        "ranked_new_pairs_problems"
        if source_scope == "ranked_new_pairs"
        else "ranked_pairs_problems"
    )
    if key not in payload:
        raise ValueError(f"source reachability payload has no {key!r} field")
    return tuple(
        source_alignment_problem_from_dict(item)
        for item in payload.get(key, []) or []
    )


def find_experimental_precondition(
    problem: SourceAlignmentProblem,
    *,
    evaluate_sequence: SequenceEvaluator,
    max_depth: int = 3,
    max_branching: int = 4,
    max_nodes: int = 85,
) -> SearchResult:
    """Search for a short sequence making the problem source actionable."""
    queue: List[Tuple[Any, ...]] = [tuple()]
    seen: set[Tuple[Tuple[str, Tuple[Tuple[str, str], ...]], ...]] = {tuple()}
    evaluated = 0
    last_error = ""
    while queue and evaluated < max(1, int(max_nodes)):
        sequence = queue.pop(0)
        state = evaluate_sequence(sequence)
        evaluated += 1
        if state is None:
            last_error = "sequence_evaluation_failed"
            continue
        if sequence and _source_actionable(problem, state):
            return SearchResult(
                problem=problem,
                precondition=_precondition_from_sequence(
                    problem,
                    sequence=sequence,
                    state=state,
                    evaluated_nodes=evaluated,
                ),
                evaluated_nodes=evaluated,
                exhausted=False,
            )
        if len(sequence) >= max(0, int(max_depth)):
            continue
        for action in _select_preparation_actions(
            state.actions,
            max_branching=max_branching,
        ):
            next_sequence = sequence + (action,)
            signature = tuple(_action_signature(item) for item in next_sequence)
            if signature in seen:
                continue
            seen.add(signature)
            queue.append(next_sequence)
    return SearchResult(
        problem=problem,
        precondition=None,
        evaluated_nodes=evaluated,
        exhausted=not queue,
        error=last_error,
    )


def run_actionable_source_alignment_pretest(
    *,
    source_reachability_path: str | Path = DEFAULT_SOURCE_REACHABILITY_OUTPUT_PATH,
    source_scope: str = "ranked_new_pairs",
    environments_dir: str | Path | None = None,
    max_depth: int = 3,
    max_branching: int = 4,
    max_nodes_per_problem: int = 85,
) -> Dict[str, Any]:
    """Run the M1.3d-b offline pretest over source alignment problems."""
    problems = load_source_alignment_problems(
        source_reachability_path,
        source_scope=source_scope,
    )
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    evaluators: Dict[str, ArcSequenceEvaluator] = {}
    results: List[SearchResult] = []
    for problem in problems:
        evaluator = evaluators.get(problem.game_id)
        if evaluator is None:
            evaluator = ArcSequenceEvaluator(problem.game_id, environments_dir=env_dir)
            evaluators[problem.game_id] = evaluator
        results.append(
            find_experimental_precondition(
                problem,
                evaluate_sequence=evaluator,
                max_depth=max_depth,
                max_branching=max_branching,
                max_nodes=max_nodes_per_problem,
            )
        )
    return {
        "config": {
            "source_reachability_path": str(source_reachability_path),
            "source_scope": source_scope,
            "environments_dir": str(env_dir),
            "max_depth": int(max_depth),
            "max_branching": int(max_branching),
            "max_nodes_per_problem": int(max_nodes_per_problem),
        },
        "summary": summarize_pretest_results(results),
        "results": [result.to_dict() for result in results],
        "preconditions": [
            result.precondition.to_dict()
            for result in results
            if result.precondition is not None
        ],
        "status": "UNRESOLVED",
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
    }


def summarize_pretest_results(results: Sequence[SearchResult]) -> Dict[str, Any]:
    """Aggregate M1.3d-b pretest metrics."""
    found = [result.precondition for result in results if result.precondition is not None]
    lengths = [precondition.prep_length for precondition in found]
    per_game: Dict[str, Dict[str, int]] = {}
    for result in results:
        row = per_game.setdefault(
            result.problem.game_id,
            {
                "problems_total": 0,
                "preconditions_found": 0,
                "source_becomes_actionable": 0,
                "target_still_present": 0,
            },
        )
        row["problems_total"] += 1
        if result.precondition is not None:
            row["preconditions_found"] += 1
            row["source_becomes_actionable"] += int(
                result.precondition.source_becomes_actionable
            )
            row["target_still_present"] += int(result.precondition.target_still_present)
    return {
        "problems_total": len(results),
        "preconditions_found": len(found),
        "mean_prep_length": round(
            sum(lengths) / len(lengths),
            4,
        )
        if lengths
        else 0.0,
        "source_becomes_actionable": sum(
            int(precondition.source_becomes_actionable)
            for precondition in found
        ),
        "target_still_present": sum(
            int(precondition.target_still_present)
            for precondition in found
        ),
        "per_game": {game_id: row for game_id, row in sorted(per_game.items())},
    }


def write_actionable_source_alignment_pretest(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_ACTIONABLE_ALIGNMENT_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class ArcSequenceEvaluator:
    """Replay preparation sequences from RESET in the offline ARC env."""

    def __init__(self, game_id: str, *, environments_dir: str | Path) -> None:
        self.game_id = str(game_id)
        self.environments_dir = Path(environments_dir)
        self._cache: Dict[Tuple[Tuple[str, Tuple[Tuple[str, str], ...]], ...], SearchState | None] = {}
        _configure_offline_env(self.environments_dir)

    def __call__(self, sequence: Sequence[Any]) -> SearchState | None:
        signature = tuple(_action_signature(action) for action in sequence)
        if signature in self._cache:
            return self._cache[signature]
        try:
            state = self._evaluate_uncached(sequence)
        except Exception:
            state = None
        self._cache[signature] = state
        return state

    def _evaluate_uncached(self, sequence: Sequence[Any]) -> SearchState:
        from arc_agi import Arcade, OperationMode
        from arcengine import GameAction

        arc = Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=str(self.environments_dir),
        )
        env = arc.make(self.game_id)
        frame = env.step(GameAction.RESET)
        fallback_actions = _valid_actions(env)
        for action in sequence:
            frame = _step_env_action(env, action)
            fallback_actions = _valid_actions(env)
            if frame is None:
                break
        snapshot = snapshot_frame(frame, fallback_available_actions=fallback_actions)
        return SearchState(
            grid=np.asarray(snapshot.grid, dtype=np.int32),
            actions=tuple(_valid_actions(env)),
        )


def _source_actionable(problem: SourceAlignmentProblem, state: SearchState) -> bool:
    sources = _source_colors_by_action(state.grid, state.actions)
    return int(problem.desired_source_color) in sources.get(str(problem.action), set())


def _precondition_from_sequence(
    problem: SourceAlignmentProblem,
    *,
    sequence: Sequence[Any],
    state: SearchState,
    evaluated_nodes: int,
) -> ExperimentalPrecondition:
    sources_by_action = _source_colors_by_action(state.grid, state.actions)
    final_sources = tuple(
        sorted(int(value) for value in sources_by_action.get(str(problem.action), set()))
    )
    return ExperimentalPrecondition(
        game_id=problem.game_id,
        source_problem=problem,
        prep_actions=tuple(
            PreparationAction(name=action.name, action_args=dict(action.action_args))
            for action in sequence
        ),
        then_test_pair=(
            problem.action,
            int(problem.desired_source_color),
            int(problem.target_color),
        ),
        source_becomes_actionable=int(problem.desired_source_color) in final_sources,
        target_still_present=bool(np.any(state.grid == int(problem.target_color))),
        final_available_sources=final_sources,
        prep_length=len(sequence),
        evaluated_nodes=int(evaluated_nodes),
    )


def _select_preparation_actions(
    actions: Sequence[Any],
    *,
    max_branching: int,
) -> Tuple[Any, ...]:
    selected: List[Any] = []
    seen: set[Tuple[str, Tuple[Tuple[str, str], ...]]] = set()
    ordered = sorted(
        actions,
        key=lambda action: (
            str(getattr(action, "name", "")) == "RESET",
            str(getattr(action, "name", "")),
            tuple(sorted((str(k), str(v)) for k, v in getattr(action, "action_args", {}).items())),
        ),
    )
    for action in ordered:
        name = str(getattr(action, "name", ""))
        if name == "RESET":
            continue
        prep_action = PreparationAction(
            name=name,
            action_args=dict(getattr(action, "action_args", {}) or {}),
        )
        if prep_action.signature in seen:
            continue
        seen.add(prep_action.signature)
        selected.append(action)
        if len(selected) >= max(1, int(max_branching)):
            break
    return tuple(selected)


def _action_signature(action: Any) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
    if isinstance(action, PreparationAction):
        return action.signature
    return (
        str(getattr(action, "name", "")),
        tuple(
            sorted(
                (str(key), str(value))
                for key, value in dict(
                    getattr(action, "action_args", {}) or {}
                ).items()
            )
        ),
    )


def _step_env_action(env: Any, action: Any) -> Any:
    action_args = dict(getattr(action, "action_args", {}) or {})
    raw_action = getattr(action, "raw_action", action)
    if action_args:
        return env.step(raw_action, data=action_args)
    return env.step(raw_action)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run M1.3d-b actionable source alignment pretest.",
    )
    parser.add_argument(
        "--source-reachability",
        type=Path,
        default=DEFAULT_SOURCE_REACHABILITY_OUTPUT_PATH,
    )
    parser.add_argument(
        "--source-scope",
        choices=("ranked_new_pairs", "ranked_pairs"),
        default="ranked_new_pairs",
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-branching", type=int, default=4)
    parser.add_argument("--max-nodes-per-problem", type=int, default=85)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_ACTIONABLE_ALIGNMENT_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_actionable_source_alignment_pretest(
        source_reachability_path=args.source_reachability,
        source_scope=args.source_scope,
        environments_dir=args.environments_dir,
        max_depth=args.max_depth,
        max_branching=args.max_branching,
        max_nodes_per_problem=args.max_nodes_per_problem,
    )
    write_actionable_source_alignment_pretest(payload, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "summary": payload["summary"],
                "status": "UNRESOLVED",
                "trace_support_counted_as_proof": False,
                "prior_counted_as_proof": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
