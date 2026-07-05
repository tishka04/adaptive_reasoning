"""Replay and verify human traces against the local ARC-AGI-3 environment.

This is intentionally narrower than the full AdaptiveReasoning launcher.
It answers the causal question first:

1. Pick a human episode for a game, preferring WIN episodes when present.
2. Replay its recorded actions exactly from a fresh environment.
3. Report the first mismatch in frame/action/state/level alignment.
4. Optionally compare the seeded GameMemory ranker with the human actions.

Example:
    python trace_replay_verifier.py --game ar25 --compare-memory-ranker
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_DIR = PROJECT_ROOT / "environment_files"
AGENTS_DIR = PROJECT_ROOT / "ARC-AGI-3-Agents"
BUNDLED_SITE_PACKAGES = AGENTS_DIR / ".venv" / "Lib" / "site-packages"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "diagnostics" / "trace_replay_verifier"

os.environ.setdefault("OPERATION_MODE", "offline")
os.environ.setdefault("ENVIRONMENTS_DIR", str(ENV_DIR))
os.environ.setdefault("ARC_API_KEY", "test")
os.environ.setdefault("RECORDINGS_DIR", str(PROJECT_ROOT / "recordings"))
os.environ.setdefault("TQDM_DISABLE", "1")

for _mod in ("torch",):
    try:
        __import__(_mod)
    except Exception:
        pass

if importlib.util.find_spec("pydantic") is None:
    import adaptive_reasoning_compat.pydantic as _compat_pydantic

    sys.modules.setdefault("pydantic", _compat_pydantic)

if importlib.util.find_spec("dotenv") is None:
    _dotenv = types.ModuleType("dotenv")

    def _load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False

    _dotenv.load_dotenv = _load_dotenv
    sys.modules.setdefault("dotenv", _dotenv)

for _path in (BUNDLED_SITE_PACKAGES, PROJECT_ROOT, AGENTS_DIR):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


try:
    from arc_agi import Arcade, OperationMode  # noqa: E402
    from arcengine import GameAction  # noqa: E402
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Could not import the bundled ARC-AGI-3 environment. "
        f"Tried {BUNDLED_SITE_PACKAGES}."
    ) from exc

from human_trace import build_prior_pack, load_traces, seed_game_memory  # noqa: E402
from human_trace.schema import EpisodeRecord, StepRecord  # noqa: E402
from v4_1_reasoning_system.arc_agi.game_memory import GameMemory  # noqa: E402


INT_TO_ACTION_NAME = {
    0: "RESET",
    1: "ACTION1",
    2: "ACTION2",
    3: "ACTION3",
    4: "ACTION4",
    5: "ACTION5",
    6: "ACTION6",
    7: "ACTION7",
}


@dataclass
class SelectedEpisode:
    game_id: str
    episode_id: str
    selection_reason: str
    episode: Optional[EpisodeRecord]
    steps: List[StepRecord]


@dataclass
class ReplayMismatch:
    step_index: int
    trace_step: int
    action: str
    kind: str
    expected: Any
    actual: Any
    detail: str = ""


@dataclass
class ReplayReport:
    game: str
    resolved_game_id: str
    episode_id: str
    selected_reason: str
    trace_final_state: str
    trace_levels_completed: int
    trace_steps: int
    replayed_steps: int
    exact_replay_ok: bool
    replay_final_state: str
    replay_levels_completed: int
    replay_won: bool
    first_mismatch: Optional[ReplayMismatch]
    warnings: List[str]
    memory_ranker: Optional[Dict[str, Any]] = None
    agent_policy: Optional[Dict[str, Any]] = None


def _resolve_full_game_id(arc: Arcade, game_id: str) -> str:
    if "-" in game_id:
        return game_id
    for env in arc.get_environments():
        if env.game_id.startswith(game_id + "-") or env.game_id == game_id:
            return env.game_id
    return game_id


def _state_name(state: Any) -> str:
    return state.name if hasattr(state, "name") else str(state)


def _primary_grid(raw_frame: Any) -> List[List[int]]:
    if raw_frame is None or getattr(raw_frame, "frame", None) is None:
        return [[0]]
    frame = raw_frame.frame
    if not frame:
        return [[0]]
    grid = frame[-1]
    if hasattr(grid, "tolist"):
        return grid.tolist()
    return [[int(v) for v in row] for row in grid]


def _grid_summary(grid: Sequence[Sequence[int]]) -> Dict[str, Any]:
    arr = np.array(grid, dtype=np.int32)
    if arr.ndim != 2:
        return {"shape": list(arr.shape), "hash": None}
    values, counts = np.unique(arr, return_counts=True)
    hist = {
        str(int(v)): int(c)
        for v, c in sorted(zip(values, counts), key=lambda item: int(item[0]))
    }
    return {
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "hash": hashlib.sha1(arr.tobytes()).hexdigest()[:16],
        "hist": hist,
    }


def _first_cell_diff(
    expected: Sequence[Sequence[int]],
    actual: Sequence[Sequence[int]],
) -> Optional[Dict[str, Any]]:
    exp = np.array(expected, dtype=np.int32)
    act = np.array(actual, dtype=np.int32)
    if exp.shape != act.shape:
        return {
            "shape_expected": list(exp.shape),
            "shape_actual": list(act.shape),
        }
    diff = np.argwhere(exp != act)
    if diff.size == 0:
        return None
    y, x = diff[0]
    return {
        "y": int(y),
        "x": int(x),
        "expected": int(exp[y, x]),
        "actual": int(act[y, x]),
        "num_diff_cells": int(diff.shape[0]),
    }


def _grids_equal(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> bool:
    return np.array_equal(np.array(a, dtype=np.int32), np.array(b, dtype=np.int32))


def _available_action_names(actions: Optional[Iterable[Any]]) -> List[str]:
    out: List[str] = []
    for action in actions or []:
        if isinstance(action, int):
            name = INT_TO_ACTION_NAME.get(action, f"ACTION{action}")
        else:
            name = action.name if hasattr(action, "name") else str(action)
        if name != "RESET":
            out.append(name)
    return out


def _action_enum(action_name: str) -> GameAction:
    try:
        return GameAction.from_name(action_name)
    except Exception:
        return getattr(GameAction, action_name)


def _step_action_data(game_id: str, step: StepRecord) -> Dict[str, Any]:
    data: Dict[str, Any] = {"game_id": game_id}
    if step.action_args:
        data.update(step.action_args)
    return data


def _get_episode_field(ep: Optional[EpisodeRecord], steps: Sequence[StepRecord], field: str) -> Any:
    if ep is not None:
        return getattr(ep, field)
    if field == "final_state":
        return steps[-1].game_state_after if steps else "UNKNOWN"
    if field == "levels_completed":
        return max((s.levels_completed_after for s in steps), default=0)
    if field == "n_steps":
        return len(steps)
    return None


def _select_episode(
    entries: List[Dict[str, object]],
    episode_id: Optional[str],
    require_win: bool,
) -> SelectedEpisode:
    candidates: List[Tuple[Optional[EpisodeRecord], List[StepRecord]]] = []
    for bucket in entries:
        ep = bucket.get("episode")
        steps = list(bucket.get("steps") or [])
        if episode_id:
            bucket_ep_id = getattr(ep, "episode_id", None) if ep is not None else None
            step_ep_id = steps[0].episode_id if steps else None
            if episode_id not in (bucket_ep_id, step_ep_id):
                continue
        candidates.append((ep, steps))  # type: ignore[arg-type]

    if not candidates:
        target = f"episode_id={episode_id!r}" if episode_id else "any episode"
        raise ValueError(f"No trace episode matched {target}")

    def score(item: Tuple[Optional[EpisodeRecord], List[StepRecord]]) -> Tuple[int, int, int]:
        ep, steps = item
        final_state = _get_episode_field(ep, steps, "final_state")
        levels = int(_get_episode_field(ep, steps, "levels_completed") or 0)
        n_steps = int(_get_episode_field(ep, steps, "n_steps") or len(steps))
        return (1 if final_state == "WIN" else 0, levels, n_steps)

    winners = [item for item in candidates if _get_episode_field(item[0], item[1], "final_state") == "WIN"]
    if require_win and not winners:
        best = max(candidates, key=score)
        best_ep, best_steps = best
        best_levels = _get_episode_field(best_ep, best_steps, "levels_completed")
        raise ValueError(
            "No winning human trace is available. "
            f"Best episode reaches levels_completed={best_levels}."
        )

    selected_ep, selected_steps = max(winners or candidates, key=score)
    reason = "winning_trace" if winners else "best_progress_no_win_trace"
    selected_id = (
        selected_ep.episode_id
        if selected_ep is not None
        else selected_steps[0].episode_id
        if selected_steps
        else "unknown"
    )
    game_id = (
        selected_ep.game_id
        if selected_ep is not None
        else selected_steps[0].game_id
        if selected_steps
        else "unknown"
    )
    return SelectedEpisode(
        game_id=game_id,
        episode_id=selected_id,
        selection_reason=reason,
        episode=selected_ep,
        steps=selected_steps,
    )


def _load_selected_episode(
    traces_dir: Path,
    requested_game: str,
    resolved_game_id: str,
    episode_id: Optional[str],
    require_win: bool,
) -> SelectedEpisode:
    corpus = load_traces(traces_dir)
    if resolved_game_id in corpus.by_game:
        entries = corpus.by_game[resolved_game_id]
    else:
        entries = []
        for gid, eps in corpus.by_game.items():
            if gid.startswith(requested_game):
                entries = eps
                break
    if not entries:
        raise ValueError(
            f"No traces matching {requested_game!r} or {resolved_game_id!r} under {traces_dir}"
        )
    return _select_episode(entries, episode_id=episode_id, require_win=require_win)


def replay_episode(
    arc: Arcade,
    full_game_id: str,
    selection: SelectedEpisode,
    *,
    max_steps: Optional[int] = None,
) -> Tuple[ReplayReport, List[Dict[str, Any]]]:
    env = arc.make(full_game_id)
    if env is None:
        raise ValueError(f"Could not make environment for {full_game_id}")

    warnings: List[str] = []
    trace_steps = selection.steps[: max_steps or None]
    current_raw = getattr(env, "observation_space", None)
    current_grid = _primary_grid(current_raw) if current_raw is not None else [[0]]
    current_available = list(getattr(current_raw, "available_actions", []) or [])

    replayed = 0
    final_state = _state_name(getattr(current_raw, "state", "NOT_PLAYED"))
    final_levels = int(getattr(current_raw, "levels_completed", 0) or 0)
    first_mismatch: Optional[ReplayMismatch] = None
    trace_rows: List[Dict[str, Any]] = []

    for idx, step in enumerate(trace_steps):
        expected_before = step.frame_before
        if step.action != "RESET" and not _grids_equal(current_grid, expected_before):
            first_mismatch = ReplayMismatch(
                step_index=idx,
                trace_step=step.step,
                action=step.action,
                kind="frame_before_mismatch",
                expected=_grid_summary(expected_before),
                actual=_grid_summary(current_grid),
                detail=json.dumps(_first_cell_diff(expected_before, current_grid)),
            )
            break

        if step.action != "RESET":
            actual_available = set(_available_action_names(current_available))
            expected_available = set(_available_action_names(step.available_actions))
            if actual_available != expected_available:
                warnings.append(
                    f"step {step.step}: available actions differ "
                    f"expected={sorted(expected_available)} actual={sorted(actual_available)}"
                )

        action = _action_enum(step.action)
        try:
            frame_after = env.step(action, data=_step_action_data(full_game_id, step))
        except TypeError:
            frame_after = env.step(action)
        except Exception as exc:
            first_mismatch = ReplayMismatch(
                step_index=idx,
                trace_step=step.step,
                action=step.action,
                kind="env_step_error",
                expected="step executes",
                actual=f"{type(exc).__name__}: {exc}",
            )
            break

        if frame_after is None:
            first_mismatch = ReplayMismatch(
                step_index=idx,
                trace_step=step.step,
                action=step.action,
                kind="env_step_returned_none",
                expected="FrameDataRaw",
                actual=None,
            )
            break

        actual_after = _primary_grid(frame_after)
        actual_state = _state_name(getattr(frame_after, "state", "UNKNOWN"))
        actual_levels = int(getattr(frame_after, "levels_completed", 0) or 0)
        replayed += 1

        trace_rows.append(
            {
                "idx": idx,
                "trace_step": step.step,
                "action": step.action,
                "action_args": step.action_args,
                "state": actual_state,
                "levels": actual_levels,
            }
        )

        if not _grids_equal(actual_after, step.frame_after):
            first_mismatch = ReplayMismatch(
                step_index=idx,
                trace_step=step.step,
                action=step.action,
                kind="frame_after_mismatch",
                expected=_grid_summary(step.frame_after),
                actual=_grid_summary(actual_after),
                detail=json.dumps(_first_cell_diff(step.frame_after, actual_after)),
            )
            final_state = actual_state
            final_levels = actual_levels
            break

        if actual_state != step.game_state_after:
            first_mismatch = ReplayMismatch(
                step_index=idx,
                trace_step=step.step,
                action=step.action,
                kind="state_mismatch",
                expected=step.game_state_after,
                actual=actual_state,
            )
            final_state = actual_state
            final_levels = actual_levels
            break

        if actual_levels != step.levels_completed_after:
            first_mismatch = ReplayMismatch(
                step_index=idx,
                trace_step=step.step,
                action=step.action,
                kind="levels_completed_mismatch",
                expected=step.levels_completed_after,
                actual=actual_levels,
            )
            final_state = actual_state
            final_levels = actual_levels
            break

        current_raw = frame_after
        current_grid = actual_after
        current_available = list(getattr(frame_after, "available_actions", []) or [])
        final_state = actual_state
        final_levels = actual_levels

    trace_final = str(_get_episode_field(selection.episode, selection.steps, "final_state"))
    trace_levels = int(_get_episode_field(selection.episode, selection.steps, "levels_completed") or 0)
    exact_ok = first_mismatch is None and replayed == len(trace_steps)

    report = ReplayReport(
        game=selection.game_id,
        resolved_game_id=full_game_id,
        episode_id=selection.episode_id,
        selected_reason=selection.selection_reason,
        trace_final_state=trace_final,
        trace_levels_completed=trace_levels,
        trace_steps=len(selection.steps),
        replayed_steps=replayed,
        exact_replay_ok=exact_ok,
        replay_final_state=final_state,
        replay_levels_completed=final_levels,
        replay_won=final_state == "WIN",
        first_mismatch=first_mismatch,
        warnings=warnings,
    )
    return report, trace_rows


def compare_memory_ranker(
    traces_dir: Path,
    selection: SelectedEpisode,
) -> Dict[str, Any]:
    corpus = load_traces(traces_dir)
    pack = build_prior_pack(corpus, selection.game_id)
    memory = GameMemory()
    seed_stats = seed_game_memory(pack, memory)

    first_divergence: Optional[Dict[str, Any]] = None
    checked = 0
    matched_top = 0
    matched_top3 = 0

    for idx, step in enumerate(selection.steps):
        if step.action == "RESET":
            continue
        available = _available_action_names(step.available_actions)
        if not available:
            available = [f"ACTION{i}" for i in range(1, 8)]
        ranked = memory.rank_actions(available)
        checked += 1
        if ranked and ranked[0] == step.action:
            matched_top += 1
        if step.action in ranked[:3]:
            matched_top3 += 1
        if first_divergence is None and (not ranked or ranked[0] != step.action):
            first_divergence = {
                "step_index": idx,
                "trace_step": step.step,
                "human_action": step.action,
                "human_action_args": step.action_args,
                "top_ranked": ranked[0] if ranked else None,
                "top3_ranked": ranked[:3],
                "human_action_rank": ranked.index(step.action) + 1
                if step.action in ranked
                else None,
                "available": available,
            }

    return {
        "checked_non_reset_steps": checked,
        "top1_matches": matched_top,
        "top1_match_rate": matched_top / max(checked, 1),
        "top3_matches": matched_top3,
        "top3_match_rate": matched_top3 / max(checked, 1),
        "first_divergence": first_divergence,
        "seed_stats": seed_stats,
        "memory_summary": memory.summary(),
    }


def _compact_action_args(data: Optional[Dict[str, Any]]) -> Optional[Dict[str, int]]:
    if not data:
        return None
    out: Dict[str, int] = {}
    for key in ("x", "y"):
        if key in data and data[key] is not None:
            out[key] = int(data[key])
    return out or None


def _extract_game_action_args(action: GameAction) -> Optional[Dict[str, int]]:
    raw = getattr(action, "action_data", None)
    if raw is None:
        return None
    try:
        data = raw.model_dump()
    except Exception:
        try:
            data = dict(raw)
        except Exception:
            return None
    return _compact_action_args(data)


def _compact_reasoning_meta(action: GameAction) -> Dict[str, Any]:
    meta = getattr(action, "reasoning", {}) or {}
    keep = (
        "phase",
        "goal",
        "subgoal",
        "strategy",
        "trajectory_source",
        "trajectory_score",
        "action_counter",
        "level",
    )
    return {key: meta.get(key) for key in keep if key in meta}


def compare_agent_policy(
    arc: Arcade,
    full_game_id: str,
    traces_dir: Path,
    selection: SelectedEpisode,
    *,
    reasoning_mode: str,
    ablation_stage: str,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    from agents import AVAILABLE_AGENTS  # noqa: WPS433
    from human_trace import HumanTraceMemory, seed_cross_game_memory  # noqa: WPS433
    from v4_1_reasoning_system.arc_agi.associative_memory import CrossGameMemory  # noqa: WPS433

    corpus = load_traces(traces_dir)
    pack = build_prior_pack(corpus, selection.game_id)
    cross_game = CrossGameMemory()
    seed_cross_game_memory(pack, cross_game)

    env = arc.make(full_game_id)
    if env is None:
        raise ValueError(f"Could not make environment for {full_game_id}")

    agent_cls = AVAILABLE_AGENTS["adaptivereasoning"]
    original_reasoning_mode = getattr(agent_cls, "REASONING_MODE", "full")
    original_ablation_stage = getattr(agent_cls, "ABLATION_STAGE", None)
    try:
        if hasattr(agent_cls, "REASONING_MODE"):
            agent_cls.REASONING_MODE = reasoning_mode
        if hasattr(agent_cls, "ABLATION_STAGE"):
            agent_cls.ABLATION_STAGE = ablation_stage

        agent = agent_cls(
            card_id=None,
            game_id=full_game_id,
            agent_name="adaptivereasoning",
            ROOT_URL="",
            record=False,
            arc_env=env,
            cross_game=cross_game,
            arcade=arc,
        )
        agent.reasoning.set_human_trace_memory(HumanTraceMemory.from_prior_pack(pack))
        seed_stats = seed_game_memory(pack, agent.memory)

        steps = selection.steps[: max_steps or None]
        checked = 0
        matched_actions = 0
        first_divergence: Optional[Dict[str, Any]] = None
        current_raw = getattr(env, "observation_space", None)

        for idx, step in enumerate(steps):
            if step.action == "RESET":
                current_raw = env.step(
                    _action_enum(step.action),
                    data=_step_action_data(full_game_id, step),
                )
                if current_raw is not None:
                    frame = agent._convert_raw_frame_data(current_raw)
                    agent.append_frame(frame)
                    if frame.frame is not None:
                        agent._prev_grid = agent.analyzer.parse_frame(frame.frame)
                    agent._prev_levels = int(frame.levels_completed or 0)
                    agent._prev_obs = None
                    agent._prev_strategy = None
                    agent._prev_trajectory = None
                    agent._prev_goal_context = None
                    agent._last_action_name = "RESET"
                    agent._needs_reset = False
                    agent._game_started = True
                    agent.action_counter += 1
                continue

            if current_raw is None:
                first_divergence = {
                    "step_index": idx,
                    "trace_step": step.step,
                    "kind": "missing_current_frame",
                }
                break

            latest = agent._convert_raw_frame_data(current_raw)
            chosen = agent.choose_action(agent.frames, latest)
            chosen_name = chosen.name if hasattr(chosen, "name") else str(chosen)
            chosen_args = _extract_game_action_args(chosen) if chosen_name == "ACTION6" else None
            human_args = _compact_action_args(step.action_args)
            checked += 1

            same_action = chosen_name == step.action
            same_args = chosen_name != "ACTION6" or chosen_args == human_args
            if same_action and same_args:
                matched_actions += 1
            elif first_divergence is None:
                first_divergence = {
                    "step_index": idx,
                    "trace_step": step.step,
                    "kind": "agent_policy_divergence",
                    "human_action": step.action,
                    "human_action_args": human_args,
                    "agent_action": chosen_name,
                    "agent_action_args": chosen_args,
                    "reasoning": _compact_reasoning_meta(chosen),
                }
                break

            current_raw = env.step(
                _action_enum(step.action),
                data=_step_action_data(full_game_id, step),
            )
            if current_raw is not None:
                frame = agent._convert_raw_frame_data(current_raw)
                agent.append_frame(frame)
                agent.action_counter += 1

        return {
            "reasoning_mode": reasoning_mode,
            "ablation_stage": ablation_stage,
            "checked_non_reset_steps": checked,
            "matched_actions": matched_actions,
            "match_rate": matched_actions / max(checked, 1),
            "first_divergence": first_divergence,
            "seed_stats": seed_stats,
        }
    finally:
        if hasattr(agent_cls, "REASONING_MODE"):
            agent_cls.REASONING_MODE = original_reasoning_mode
        if hasattr(agent_cls, "ABLATION_STAGE"):
            agent_cls.ABLATION_STAGE = original_ablation_stage


def _report_to_dict(report: ReplayReport) -> Dict[str, Any]:
    data = asdict(report)
    if report.first_mismatch is not None:
        data["first_mismatch"] = asdict(report.first_mismatch)
    return data


def _write_report(path: Path, report: ReplayReport, trace_rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "report": _report_to_dict(report),
        "replayed_prefix": trace_rows[:50],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _print_report(report: ReplayReport, output: Optional[Path]) -> None:
    print("=" * 88)
    print("Trace replay verifier")
    print("=" * 88)
    print(f"game:        {report.game}")
    print(f"resolved:    {report.resolved_game_id}")
    print(f"episode:     {report.episode_id} ({report.selected_reason})")
    print(
        "trace:       "
        f"final={report.trace_final_state} "
        f"levels={report.trace_levels_completed} "
        f"steps={report.trace_steps}"
    )
    print(
        "replay:      "
        f"ok={report.exact_replay_ok} "
        f"final={report.replay_final_state} "
        f"levels={report.replay_levels_completed} "
        f"steps={report.replayed_steps}"
    )
    if report.trace_final_state != "WIN":
        print("note:        no winning trace was selected; this verifies best progress only.")
    if report.first_mismatch is None:
        print("mismatch:    none")
    else:
        mm = report.first_mismatch
        print(
            "mismatch:    "
            f"step_index={mm.step_index} trace_step={mm.trace_step} "
            f"action={mm.action} kind={mm.kind}"
        )
        print(f"expected:    {mm.expected}")
        print(f"actual:      {mm.actual}")
        if mm.detail:
            print(f"detail:      {mm.detail}")
    if report.warnings:
        print(f"warnings:    {len(report.warnings)}")
        for warning in report.warnings[:5]:
            print(f"  - {warning}")
        if len(report.warnings) > 5:
            print(f"  - ... {len(report.warnings) - 5} more")
    if report.memory_ranker:
        ranker = report.memory_ranker
        print("memory:      GameMemory ranker")
        print(
            "             "
            f"top1={ranker['top1_matches']}/{ranker['checked_non_reset_steps']} "
            f"({ranker['top1_match_rate']:.1%}), "
            f"top3={ranker['top3_matches']}/{ranker['checked_non_reset_steps']} "
            f"({ranker['top3_match_rate']:.1%})"
        )
        div = ranker.get("first_divergence")
        if div:
            print(
                "             "
                f"first_divergence step={div['trace_step']} "
                f"human={div['human_action']} "
                f"top={div['top_ranked']} "
                f"top3={div['top3_ranked']} "
                f"human_rank={div['human_action_rank']}"
            )
        else:
            print("             no ranker divergence")
    if report.agent_policy:
        policy = report.agent_policy
        print("agent:       aligned policy compare")
        if policy.get("skipped"):
            print(f"             skipped={policy['skipped']}")
        else:
            print(
                "             "
                f"mode={policy['reasoning_mode']} stage={policy['ablation_stage']} "
                f"matches={policy['matched_actions']}/{policy['checked_non_reset_steps']} "
                f"({policy['match_rate']:.1%})"
            )
            div = policy.get("first_divergence")
            if div:
                print(
                    "             "
                    f"first_divergence step={div['trace_step']} "
                    f"human={div.get('human_action')} "
                    f"agent={div.get('agent_action')} "
                    f"kind={div.get('kind')}"
                )
                if div.get("reasoning"):
                    print(f"             reasoning={div['reasoning']}")
            else:
                print("             no policy divergence")
    if output is not None:
        print(f"json:        {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="ar25")
    parser.add_argument("--traces", type=Path, default=PROJECT_ROOT / "human_traces")
    parser.add_argument("--episode-id", default=None)
    parser.add_argument(
        "--require-win",
        action="store_true",
        help="Fail if no WIN episode exists instead of falling back to best progress.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Replay only the first N trace steps for a quick smoke test.",
    )
    parser.add_argument(
        "--compare-memory-ranker",
        action="store_true",
        help="After exact replay, compare seeded GameMemory.rank_actions with human actions.",
    )
    parser.add_argument(
        "--compare-agent-policy",
        action="store_true",
        help=(
            "Compare the actual AdaptiveReasoning policy on human-aligned states. "
            "The verifier still executes the human actions to preserve alignment."
        ),
    )
    parser.add_argument("--agent-reasoning-mode", default="symbolic_core")
    parser.add_argument("--agent-ablation-stage", default="game_memory")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write a machine-readable report. Defaults to diagnostics/trace_replay_verifier.",
    )
    args = parser.parse_args()

    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ENV_DIR),
    )
    full_game_id = _resolve_full_game_id(arc, args.game)
    selection = _load_selected_episode(
        args.traces,
        requested_game=args.game,
        resolved_game_id=full_game_id,
        episode_id=args.episode_id,
        require_win=args.require_win,
    )

    report, trace_rows = replay_episode(
        arc,
        full_game_id,
        selection,
        max_steps=args.max_steps,
    )

    if args.compare_memory_ranker:
        report.memory_ranker = compare_memory_ranker(args.traces, selection)
    if args.compare_agent_policy:
        if report.exact_replay_ok:
            report.agent_policy = compare_agent_policy(
                arc,
                full_game_id,
                args.traces,
                selection,
                reasoning_mode=args.agent_reasoning_mode,
                ablation_stage=args.agent_ablation_stage,
                max_steps=args.max_steps,
            )
        else:
            report.agent_policy = {"skipped": "exact replay failed"}

    output = args.json_out
    if output is None:
        suffix = f"{selection.episode_id}.json"
        output = DEFAULT_REPORT_DIR / f"{full_game_id}.{suffix}"
    _write_report(output, report, trace_rows)
    _print_report(report, output)

    return 0 if report.exact_replay_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
