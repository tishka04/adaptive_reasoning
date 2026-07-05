"""A13 active non-ar25 correspondence micro-run.

This harness takes an unresolved A12 correspondence candidate, executes one
real environment action, builds a live TransitionRecord, and revises only from
the observed source-target predicate. It is deliberately not a planner and it
does not try to win the game.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np

from .correspondence_hypothesis import CorrespondenceHypothesis
from .cross_game_correspondence_discovery import (
    DiscoveredCorrespondenceCandidate,
    _source_target_predicates,
    discover_cross_game_correspondences,
)
from .epistemic_metrics import (
    EpistemicScore,
    HypothesisRecord,
    HypothesisStatus,
    MechanicsOracle,
    score_beliefs,
)
from .live_transition_loop import LiveTransitionBeliefLoop, LiveTransitionUpdate
from .mechanic_hypothesis import GameTheory
from .real_env_option_adapter import snapshot_frame

DEFAULT_GAME_ID = "ft09-0d8bbf25"
DEFAULT_TRACE_PATH = Path("human_traces/ft09-0d8bbf25.20260617-142428.steps.jsonl")


@dataclass(frozen=True)
class ActiveExperimentAction:
    """One concrete env action selected for an A13 micro-experiment."""

    name: str
    raw_action: Any
    action_args: dict[str, Any] = field(default_factory=dict)
    selection_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "action_args": dict(self.action_args),
            "selection_reason": self.selection_reason,
        }


@dataclass(frozen=True)
class NonAr25ActiveMicroRunEvent:
    """One live hypothesis test event."""

    step: int
    action: str
    action_args: dict[str, Any] = field(default_factory=dict)
    candidate_key: str = ""
    status_before: str = HypothesisStatus.UNRESOLVED.value
    status_after: str = HypothesisStatus.UNRESOLVED.value
    levels_before: int = 0
    levels_after: int = 0
    changed_pixels: int = 0
    pair_change_pixels: int = 0
    observed_predicates: List[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class NonAr25ActiveMicroRunResult:
    """Summary of an A13 active non-ar25 hypothesis test."""

    game_id: str
    trace_path: Path
    candidate: DiscoveredCorrespondenceCandidate | None = None
    selected_action: ActiveExperimentAction | None = None
    status_before: HypothesisStatus = HypothesisStatus.UNRESOLVED
    status_after: HypothesisStatus = HypothesisStatus.UNRESOLVED
    status_reason: str = ""
    transition_update: LiveTransitionUpdate | None = None
    score: EpistemicScore | None = None
    events: List[NonAr25ActiveMicroRunEvent] = field(default_factory=list)
    discovered_candidates: int = 0
    env_actions: int = 0
    trace_dependent: bool = False
    error: str = ""

    @property
    def candidate_key(self) -> str:
        return "" if self.candidate is None else self.candidate.key

    @property
    def active_transition_non_ar25(self) -> bool:
        return self.game_id != "ar25-e3c63847" and self.transition_count > 0

    @property
    def transition_count(self) -> int:
        return 0 if self.transition_update is None else 1

    @property
    def hypothesis_status_changed(self) -> bool:
        return self.status_before != self.status_after

    @property
    def evidence_permits_status_change(self) -> bool:
        if not self.hypothesis_status_changed:
            return True
        return self.status_reason in {
            "observed_source_target_color_transform",
            "observed_different_transition",
        }

    @property
    def pair_change_pixels(self) -> int:
        if not self.events:
            return 0
        return self.events[-1].pair_change_pixels

    @property
    def observed_predicates(self) -> List[str]:
        if not self.events:
            return []
        return list(self.events[-1].observed_predicates)

    @property
    def wrong_confirmations(self) -> int:
        if self.status_after == HypothesisStatus.CONFIRMED and self.pair_change_pixels <= 0:
            return 1
        return 0

    def record(self) -> HypothesisRecord | None:
        if self.candidate is None:
            return None
        return _record_from_revision(
            self.candidate,
            self.status_after,
            experiments_spent=self.env_actions,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "trace_path": str(self.trace_path),
            "candidate_key": self.candidate_key,
            "candidate": self.candidate.to_dict() if self.candidate else None,
            "selected_action": (
                self.selected_action.to_dict() if self.selected_action else None
            ),
            "active_transition_non_ar25": self.active_transition_non_ar25,
            "transitions": self.transition_count,
            "env_actions": self.env_actions,
            "trace_dependent": self.trace_dependent,
            "status_before": self.status_before.value,
            "status_after": self.status_after.value,
            "status_reason": self.status_reason,
            "hypothesis_status_changed": self.hypothesis_status_changed,
            "evidence_permits_status_change": self.evidence_permits_status_change,
            "wrong_confirmations": self.wrong_confirmations,
            "discovered_candidates": self.discovered_candidates,
            "score": self.score.to_dict() if self.score is not None else None,
            "events": [event.__dict__ for event in self.events],
            "error": self.error,
        }


def run_non_ar25_active_micro_run(
    *,
    game_id: str = DEFAULT_GAME_ID,
    trace_path: Path | str = DEFAULT_TRACE_PATH,
    environments_dir: Path | str | None = None,
    max_candidates: int = 20,
    min_pixel_support: int = 1,
    preferred_pair_colors: Tuple[int, int] | None = None,
    min_pair_change_pixels: int = 4,
) -> NonAr25ActiveMicroRunResult:
    """Run one active non-ar25 correspondence hypothesis test."""
    path = Path(trace_path)
    result = NonAr25ActiveMicroRunResult(game_id=game_id, trace_path=path)

    discovery = discover_cross_game_correspondences(
        path,
        game_id=game_id,
        min_pixel_support=min_pixel_support,
        top_k=max_candidates,
    )
    result.discovered_candidates = len(discovery.candidates)
    if not discovery.candidates:
        result.error = "no_unresolved_candidates"
        return result

    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    _configure_offline_env(env_dir)
    try:
        from arc_agi import Arcade, OperationMode
        from arcengine import GameAction

        arc = Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=str(env_dir),
        )
        env = arc.make(game_id)
        current_frame = env.step(GameAction.RESET)
    except Exception as exc:  # pragma: no cover - integration failure path
        result.error = f"env_setup_failed: {exc}"
        return result

    before = snapshot_frame(current_frame)
    candidate, selected_action = _select_live_test(
        discovery.candidates,
        before.grid,
        env,
        preferred_pair_colors=preferred_pair_colors,
    )
    if candidate is None or selected_action is None:
        result.error = "no_live_testable_candidate"
        return result

    result.candidate = candidate
    result.selected_action = selected_action

    theory = GameTheory(game_id)
    theory.seed_actions(before.available_actions)
    theory.add_semantic_hypotheses(
        correspondence=[_active_hypothesis_from_candidate(candidate)]
    )
    loop = LiveTransitionBeliefLoop(
        game_id,
        theory=theory,
        available_actions=before.available_actions,
        infer_players=False,
        correspondence_pair_colors=candidate.pair_colors,
    )

    try:
        after_frame = _step_env_with_action(env, selected_action)
    except Exception as exc:  # pragma: no cover - integration failure path
        result.error = f"env_step_failed:{selected_action.name}: {exc}"
        return result
    if after_frame is None:
        result.error = f"env_step_returned_none:{selected_action.name}"
        return result

    after = snapshot_frame(
        after_frame,
        fallback_available_actions=before.available_actions,
    )
    update = loop.observe_grids(
        action=selected_action.name,
        action_args=selected_action.action_args,
        grid_before=before.grid,
        grid_after=after.grid,
        available_actions=before.available_actions or after.available_actions,
        game_state_before=before.game_state,
        game_state_after=after.game_state,
        levels_completed_before=before.levels_completed,
        levels_completed_after=after.levels_completed,
        timestamp=0,
        was_experiment=True,
    )
    result.transition_update = update
    result.env_actions = 1

    changed_pixels = int(np.sum(before.grid != after.grid))
    pair_change_pixels = _pair_change_pixels(
        before.grid,
        after.grid,
        candidate.pair_colors,
    )
    observed_predicates = _observed_predicates(
        before.grid,
        after.grid,
        candidate,
        pair_change_pixels=pair_change_pixels,
        min_pair_change_pixels=min_pair_change_pixels,
    )
    status_after, reason = _revise_from_live_evidence(
        changed_pixels=changed_pixels,
        pair_change_pixels=pair_change_pixels,
        min_pair_change_pixels=min_pair_change_pixels,
    )
    result.status_after = status_after
    result.status_reason = reason
    result.events.append(
        NonAr25ActiveMicroRunEvent(
            step=0,
            action=selected_action.name,
            action_args=dict(selected_action.action_args),
            candidate_key=candidate.key,
            status_before=result.status_before.value,
            status_after=status_after.value,
            levels_before=before.levels_completed,
            levels_after=after.levels_completed,
            changed_pixels=changed_pixels,
            pair_change_pixels=pair_change_pixels,
            observed_predicates=observed_predicates,
            reason=reason,
        )
    )

    record = result.record()
    result.score = score_beliefs(
        [record] if record is not None else [],
        MechanicsOracle(game_id),
        experiment_actions=max(1, result.env_actions),
    )
    return result


def _select_live_test(
    candidates: Sequence[DiscoveredCorrespondenceCandidate],
    grid: np.ndarray,
    env: Any,
    *,
    preferred_pair_colors: Tuple[int, int] | None,
) -> tuple[DiscoveredCorrespondenceCandidate | None, ActiveExperimentAction | None]:
    ordered = list(candidates)
    if preferred_pair_colors is not None:
        preferred = tuple(int(value) for value in preferred_pair_colors)
        ordered.sort(key=lambda candidate: candidate.pair_colors != preferred)

    valid_actions = _valid_actions(env)
    for candidate in ordered:
        if candidate.to_hypothesis().status != HypothesisStatus.UNRESOLVED:
            continue
        for action in valid_actions:
            if action.name != candidate.action:
                continue
            if _action_pixel_matches_source(grid, action, candidate.source_color):
                return candidate, ActiveExperimentAction(
                    name=action.name,
                    raw_action=action.raw_action,
                    action_args=dict(action.action_args),
                    selection_reason=(
                        "live_action_pixel_matches_candidate_source_color"
                    ),
                )
        fallback = _first_unparameterized_action_named(valid_actions, candidate.action)
        if fallback is not None and _color_present(grid, candidate.source_color):
            return candidate, ActiveExperimentAction(
                name=fallback.name,
                raw_action=fallback.raw_action,
                action_args=dict(fallback.action_args),
                selection_reason="candidate_source_color_present",
            )
    return None, None


def _revise_from_live_evidence(
    *,
    changed_pixels: int,
    pair_change_pixels: int,
    min_pair_change_pixels: int,
) -> tuple[HypothesisStatus, str]:
    if pair_change_pixels >= min_pair_change_pixels:
        return (
            HypothesisStatus.CONFIRMED,
            "observed_source_target_color_transform",
        )
    if changed_pixels > 0:
        return HypothesisStatus.REFUTED, "observed_different_transition"
    return HypothesisStatus.UNRESOLVED, "no_observable_change"


def _observed_predicates(
    before: np.ndarray,
    after: np.ndarray,
    candidate: DiscoveredCorrespondenceCandidate,
    *,
    pair_change_pixels: int,
    min_pair_change_pixels: int,
) -> List[str]:
    if pair_change_pixels < min_pair_change_pixels:
        return []
    return _dedupe(
        _source_target_predicates(
            before,
            after,
            source_color=candidate.source_color,
            target_color=candidate.target_color,
        )
    )


def _active_hypothesis_from_candidate(
    candidate: DiscoveredCorrespondenceCandidate,
) -> CorrespondenceHypothesis:
    return CorrespondenceHypothesis(
        action=candidate.action,
        relation=candidate.relation,
        pair_colors=candidate.pair_colors,
        statement=(
            f"active test of {candidate.action} {candidate.relation} "
            f"colors {candidate.source_color}->{candidate.target_color}"
        ),
        status=HypothesisStatus.UNRESOLVED,
    )


def _record_from_revision(
    candidate: DiscoveredCorrespondenceCandidate,
    status: HypothesisStatus,
    *,
    experiments_spent: int,
) -> HypothesisRecord:
    return HypothesisRecord(
        key=candidate.key,
        description=(
            f"active non-ar25 test of {candidate.action} "
            f"{candidate.source_color}->{candidate.target_color}"
        ),
        status=status,
        support=1 if status == HypothesisStatus.CONFIRMED else 0,
        contradictions=1 if status == HypothesisStatus.REFUTED else 0,
        experiments_spent=int(experiments_spent),
    )


@dataclass(frozen=True)
class _ActionView:
    name: str
    raw_action: Any
    action_args: dict[str, Any] = field(default_factory=dict)


def _valid_actions(env: Any) -> List[_ActionView]:
    if hasattr(env, "_game") and hasattr(env._game, "_get_valid_actions"):
        actions = list(env._game._get_valid_actions())
        if actions:
            return [_view_action(action) for action in actions]
    raw_actions = list(getattr(env, "action_space", []) or [])
    return [_view_action(action) for action in raw_actions]


def _view_action(action: Any) -> _ActionView:
    raw = getattr(action, "id", action)
    data = dict(getattr(action, "data", None) or {})
    return _ActionView(name=_normalize_action_name(raw), raw_action=raw, action_args=data)


def _first_unparameterized_action_named(
    actions: Sequence[_ActionView],
    name: str,
) -> _ActionView | None:
    wanted = str(name).upper()
    for action in actions:
        if action.name == wanted and not action.action_args:
            return action
    return None


def _action_pixel_matches_source(
    grid: np.ndarray,
    action: _ActionView,
    source_color: int,
) -> bool:
    if "x" not in action.action_args or "y" not in action.action_args:
        return False
    x = _safe_int(action.action_args.get("x"))
    y = _safe_int(action.action_args.get("y"))
    if x is None or y is None:
        return False
    if not (0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]):
        return False
    return int(grid[y, x]) == int(source_color)


def _color_present(grid: np.ndarray, color: int) -> bool:
    return bool(np.any(grid == int(color)))


def _pair_change_pixels(
    before: np.ndarray,
    after: np.ndarray,
    pair_colors: Tuple[int, int],
) -> int:
    source, target = (int(pair_colors[0]), int(pair_colors[1]))
    mask = (before != after) & (before == source) & (after == target)
    return int(np.sum(mask))


def _step_env_with_action(env: Any, action: ActiveExperimentAction) -> Any:
    if action.action_args:
        return env.step(action.raw_action, data=dict(action.action_args))
    return env.step(action.raw_action)


def _env_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "environment_files"


def _configure_offline_env(environments_dir: Path) -> None:
    os.environ["OPERATION_MODE"] = "offline"
    os.environ["ENVIRONMENTS_DIR"] = str(environments_dir)
    os.environ.setdefault("ARC_API_KEY", "test")
    os.environ.setdefault("RECORDINGS_DIR", "recordings")
    os.environ["TQDM_DISABLE"] = "1"


def _normalize_action_name(action: Any) -> str:
    if action is None:
        return ""
    if isinstance(action, (int, np.integer)):
        if int(action) == 0:
            return "RESET"
        return f"ACTION{int(action)}"
    name = getattr(action, "name", None)
    if name:
        return str(name).strip().upper()
    raw = str(action).strip().upper()
    if "." in raw:
        raw = raw.split(".")[-1]
    if raw.isdigit():
        if int(raw) == 0:
            return "RESET"
        return f"ACTION{raw}"
    return raw


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _dedupe(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _parse_pair(raw: str) -> Tuple[int, int] | None:
    if not raw:
        return None
    left, _, right = raw.replace(":", ",").partition(",")
    if not left or not right:
        raise ValueError(f"expected COLOR_A,COLOR_B pair, got {raw!r}")
    return (int(left), int(right))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run A13 active non-ar25 correspondence micro-run."
    )
    parser.add_argument("--game-id", default=DEFAULT_GAME_ID)
    parser.add_argument("--trace-path", type=Path, default=DEFAULT_TRACE_PATH)
    parser.add_argument("--environments-dir", type=Path, default=_env_dir())
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--min-pixel-support", type=int, default=1)
    parser.add_argument(
        "--preferred-pair",
        default="",
        help="Optional source,target color pair, for example 9,8.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_non_ar25_active_micro_run(
        game_id=args.game_id,
        trace_path=args.trace_path,
        environments_dir=args.environments_dir,
        max_candidates=args.max_candidates,
        min_pixel_support=args.min_pixel_support,
        preferred_pair_colors=_parse_pair(args.preferred_pair),
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
