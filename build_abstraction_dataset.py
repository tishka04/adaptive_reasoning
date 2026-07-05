"""Phase 1+2: build a stratified, traceable transition dataset.

Each game is explored with several *tagged* episode sources so the learned
models see a clean, ablatable distribution instead of a confused mix:

    random            -> action effects, no-op, danger, simple transitions
    ontology_probe    -> "what each button does" in this game
    heuristic_guided  -> richer states closer to goals (1-step greedy teacher)
    human_replay      -> optional/special, only for games with a human trace
                         an add-on, never a fixed slot

Every row carries ``episode_source`` and the auto-generated heuristic labels.

IMPORTANT: run with the bundled env interpreter:
    ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe build_abstraction_dataset.py \\
        --games public --episodes-per-game 20
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from level7_frontier_recovery import (  # bundled env bootstrap
    Arcade,
    OperationMode,
    ENV_DIR,
    PROJECT_ROOT,
    _action_enum,
    _available_names_from_raw,
    _hash_grid,
    _primary_grid,
    _resolve_full_game_id,
    _state_name,
)

import abstraction_labels as labels
from abstraction_dataset_io import make_history_features
import game_splits
from extract_state_abstractions import (
    FEATURE_SCHEMA,
    LARGEST_COMPONENT_FEATURE_SCHEMA,
    delta_features,
    extract_state_features,
    game_specific_debug,
    largest_component_local_features,
)

ACTIONS = [f"ACTION{i}" for i in range(1, 8)]
SIMPLE_ACTIONS = [a for a in ACTIONS if a != "ACTION6"]
DEFAULT_OUT = PROJECT_ROOT / "training" / "abstraction_dataset.jsonl"
HUMAN_TRACES_DIR = PROJECT_ROOT / "human_traces"

# Stratified ratio for the general (no-human-trace) budget: 8 / 6 / 6 of 20.
SOURCE_RATIOS = {"random": 8, "ontology_probe": 6, "heuristic_guided": 6}

DANGER_HORIZON = 3  # future game-over within this many steps -> danger_probability
LEVELUP_HORIZON = 25  # future level-up within this many steps -> level_up_probability
LEVELUP_HORIZONS = (5, 10, LEVELUP_HORIZON)


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------
def _make_env(arc: Arcade, full_game_id: str) -> Any:
    env = arc.make(full_game_id)
    if env is None:
        raise ValueError(f"Could not make environment for {full_game_id}")
    return env


def _obs(env: Any) -> Any:
    return getattr(env, "observation_space", None)


def _action_data(full_game_id: str, action: str, rng: random.Random, grid_shape: Tuple[int, int]) -> Optional[Dict[str, int]]:
    if action != "ACTION6":
        return None
    h, w = grid_shape
    return {"x": rng.randint(0, max(0, w - 1)), "y": rng.randint(0, max(0, h - 1))}


def _step(env: Any, full_game_id: str, action: str, action_data: Optional[Dict[str, int]]) -> Any:
    data: Dict[str, Any] = {"game_id": full_game_id}
    if action_data:
        data.update(action_data)
    try:
        return env.step(_action_enum(action), data=data)
    except TypeError:
        return env.step(_action_enum(action))


def _changed_cells(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> int:
    left = np.array(a, dtype=np.int32)
    right = np.array(b, dtype=np.int32)
    if left.shape != right.shape:
        return int(max(left.size, right.size))
    return int(np.count_nonzero(left != right))


# ---------------------------------------------------------------------------
# Transition recording
# ---------------------------------------------------------------------------
def _round_feats(feats: Dict[str, float]) -> Dict[str, float]:
    return {k: round(float(v), 4) for k, v in feats.items()}


def _record_transition(
    *,
    game_id: str,
    episode_source: str,
    episode_index: int,
    step_index: int,
    prev_grid: Sequence[Sequence[int]],
    prev_feats: Dict[str, float],
    prev_level: int,
    history_features: Dict[str, Any],
    action: str,
    action_data: Optional[Dict[str, int]],
    next_grid: Sequence[Sequence[int]],
    next_feats: Dict[str, float],
    next_level: int,
    state_name: str,
    with_debug: bool,
) -> Dict[str, Any]:
    delta = delta_features(prev_feats, next_feats)
    total_cells = int(np.array(prev_grid, dtype=np.int32).size)
    changed = _changed_cells(prev_grid, next_grid)
    level_up = bool(next_level > prev_level or state_name == "WIN")
    game_over = bool(state_name == "GAME_OVER" or state_name == "ERROR")
    cursor = None
    if action_data and "x" in action_data and "y" in action_data:
        cursor = (float(action_data["y"]), float(action_data["x"]))

    row: Dict[str, Any] = {
        "game_id": game_id,
        "level": int(prev_level),
        "episode_source": episode_source,
        "episode_index": int(episode_index),
        "step": int(step_index),
        "history_features": history_features,
        "action": action,
        "action_data": action_data,
        "state_features": _round_feats(prev_feats),
        "largest_component_features": _round_feats(
            largest_component_local_features(prev_grid, cursor=cursor)
        ),
        "next_state_features": _round_feats(next_feats),
        "delta_features": {k: round(float(v), 4) for k, v in delta.items()},
        "changed_cells": int(changed),
        "level_up": level_up,
        "game_over": game_over,
        # Continuous heuristic labels (auto_levelup_progress filled in post).
        "break_progress": round(labels.break_progress(delta), 4),
        "fragmentation_progress": round(labels.fragmentation_progress(delta), 4),
        "correspondence_progress": round(labels.correspondence_progress(delta), 4),
        "no_op": round(labels.no_op_score(changed, total_cells), 4),
        "danger": 1.0 if game_over else 0.0,
        "auto_levelup_progress": 0.0,
        "macro_scores": {
            k: round(float(v), 4)
            for k, v in labels.macro_scores(
                delta=delta,
                changed_cells=changed,
                total_cells=total_cells,
                game_over=game_over,
            ).items()
        },
        "macro_label": labels.macro_label(
            delta=delta,
            changed_cells=changed,
            total_cells=total_cells,
            game_over=game_over,
        ),
    }
    if with_debug:
        row["debug"] = game_specific_debug(prev_grid, game_id)
    return row


# ---------------------------------------------------------------------------
# Episode policies
# ---------------------------------------------------------------------------
def _run_episode(
    arc: Arcade,
    full_game_id: str,
    *,
    episode_source: str,
    episode_index: int,
    policy: str,
    max_steps: int,
    rng: random.Random,
    with_debug: bool,
) -> List[Dict[str, Any]]:
    env = _make_env(arc, full_game_id)
    raw = _obs(env)
    if raw is None:
        return []
    grid = _primary_grid(raw)
    feats = extract_state_features(grid)
    level = int(getattr(raw, "levels_completed", 0) or 0)
    visited: set = {_hash_grid(grid)}

    rows: List[Dict[str, Any]] = []
    probe_cycle: List[str] = []
    action_history: List[str] = []
    action_repeat_count = 0
    steps_since_state_change = 0
    for step_index in range(max_steps):
        available = _available_names_from_raw(raw) or ACTIONS
        available = [a for a in available if a in ACTIONS]
        if not available:
            break
        shape = np.array(grid, dtype=np.int32).shape
        shape = (int(shape[0]), int(shape[1])) if len(shape) == 2 else (64, 64)

        if policy == "random":
            action, action_data = _random_policy(available, full_game_id, rng, shape)
        elif policy == "ontology_probe":
            if not probe_cycle:
                probe_cycle = _build_probe_cycle(available)
            action, action_data = _ontology_policy(probe_cycle, full_game_id, rng, shape)
        elif policy == "heuristic_guided":
            action, action_data = _heuristic_policy(
                env, full_game_id, available, grid, feats, visited, rng, shape
            )
        else:
            action, action_data = _random_policy(available, full_game_id, rng, shape)

        next_raw = _step(env, full_game_id, action, action_data)
        next_state = _state_name(getattr(next_raw, "state", "ERROR")) if next_raw is not None else "ERROR"
        next_grid = _primary_grid(next_raw) if next_raw is not None else grid
        next_level = int(getattr(next_raw, "levels_completed", level) or level) if next_raw is not None else level
        next_feats = extract_state_features(next_grid)

        rows.append(
            _record_transition(
                game_id=full_game_id,
                episode_source=episode_source,
                episode_index=episode_index,
                step_index=step_index,
                prev_grid=grid,
                prev_feats=feats,
                prev_level=level,
                history_features=make_history_features(
                    action_history,
                    action_repeat_count,
                    steps_since_state_change,
                ),
                action=action,
                action_data=action_data,
                next_grid=next_grid,
                next_feats=next_feats,
                next_level=next_level,
                state_name=next_state,
                with_debug=with_debug,
            )
        )

        changed = int(rows[-1]["changed_cells"])
        action_repeat_count = action_repeat_count + 1 if action_history and action_history[-1] == action else 1
        action_history.append(action)
        steps_since_state_change = 0 if changed > 0 else steps_since_state_change + 1

        if next_state in {"GAME_OVER", "WIN", "ERROR"}:
            break
        raw, grid, feats, level = next_raw, next_grid, next_feats, next_level
        visited.add(_hash_grid(grid))
    return rows


def _random_policy(available, full_game_id, rng, shape):
    action = rng.choice(available)
    return action, _action_data(full_game_id, action, rng, shape)


def _build_probe_cycle(available: List[str]) -> List[str]:
    # Each available action repeated a few times, to expose its effect.
    cycle: List[str] = []
    for action in available:
        cycle.extend([action] * 3)
    return cycle


def _ontology_policy(probe_cycle, full_game_id, rng, shape):
    action = probe_cycle.pop(0)
    return action, _action_data(full_game_id, action, rng, shape)


def _heuristic_policy(env, full_game_id, available, grid, feats, visited, rng, shape):
    """1-step greedy teacher: deepcopy-lookahead, pick the most promising action.

    Generic objective (no ar25 hardcoding): reward level-up, state change and
    rising top-pair correspondence; penalize game-over and revisiting states.
    """

    best_action = rng.choice(available)
    best_data = _action_data(full_game_id, best_action, rng, shape)
    best_score = -1e18
    for action in available:
        action_data = None
        if action == "ACTION6":
            h, w = shape
            action_data = {"x": w // 2, "y": h // 2}
        probe_env = copy.deepcopy(env)
        raw = _step(probe_env, full_game_id, action, action_data)
        if raw is None:
            continue
        state = _state_name(getattr(raw, "state", "ERROR"))
        nlevel = int(getattr(raw, "levels_completed", 0) or 0)
        ngrid = _primary_grid(raw)
        nfeats = extract_state_features(ngrid)
        changed = _changed_cells(grid, ngrid)
        d_corr = float(nfeats.get("top_pair_0_global_correspondence", 0.0)) - float(
            feats.get("top_pair_0_global_correspondence", 0.0)
        )
        novelty = 0.0 if _hash_grid(ngrid) in visited else 1.0
        score = (
            1000.0 * float(nlevel > 0 and state != "GAME_OVER")
            - 1000.0 * float(state in {"GAME_OVER", "ERROR"})
            + 2.0 * d_corr
            + 0.05 * float(min(changed, 200))
            + 40.0 * novelty
        )
        if score > best_score:
            best_score = score
            best_action = action
            best_data = action_data
    return best_action, best_data


# ---------------------------------------------------------------------------
# Human replay (optional / special, trace-bearing games only)
# ---------------------------------------------------------------------------
def _human_trace_steps(short_id: str) -> List[Dict[str, Any]]:
    if not HUMAN_TRACES_DIR.exists():
        return []
    files = sorted(HUMAN_TRACES_DIR.glob(f"{short_id}-*.steps.jsonl"))
    steps: List[Dict[str, Any]] = []
    if not files:
        return []
    # Use the most recent steps file.
    with files[-1].open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                steps.append(json.loads(line))
    return steps


def _run_human_replay(
    arc: Arcade,
    full_game_id: str,
    short_id: str,
    *,
    episode_index: int,
    max_steps: int,
    with_debug: bool,
) -> List[Dict[str, Any]]:
    steps = _human_trace_steps(short_id)
    if not steps:
        return []
    # Human replay must NOT be throttled by the short search-episode cap: the
    # winning human trace is hundreds of steps long and passes several levels.
    # Truncating it (the old behaviour, capped at the 60-step search budget)
    # dropped every level-transition / fracture state past step 60, starving the
    # break expert of real precursor states. ``max_steps`` here is the dedicated
    # human-replay cap threaded from the caller.
    env = _make_env(arc, full_game_id)
    raw = _obs(env)
    grid = _primary_grid(raw)
    feats = extract_state_features(grid)
    level = int(getattr(raw, "levels_completed", 0) or 0)
    rows: List[Dict[str, Any]] = []
    action_history: List[str] = []
    action_repeat_count = 0
    steps_since_state_change = 0
    trace_episode_indices: Dict[str, int] = {}
    next_episode_index = int(episode_index)
    for step_index, step in enumerate(steps[:max_steps]):
        trace_episode_id = str(step.get("episode_id", "") or "")
        if trace_episode_id not in trace_episode_indices:
            trace_episode_indices[trace_episode_id] = next_episode_index
            next_episode_index += 1
        row_episode_index = trace_episode_indices[trace_episode_id]
        action = str(step.get("action", "")).upper()
        if action == "RESET" or action not in ACTIONS:
            # Replay non-learnable boundary actions without recording.
            data = {"game_id": full_game_id}
            data.update(step.get("action_args") or {})
            raw = _step_with_data(env, action, data)
            if raw is None:
                break
            grid = _primary_grid(raw)
            feats = extract_state_features(grid)
            level = int(getattr(raw, "levels_completed", level) or level)
            action_history = []
            action_repeat_count = 0
            steps_since_state_change = 0
            continue
        action_data = step.get("action_args") or None
        next_raw = _step(env, full_game_id, action, action_data)
        next_state = _state_name(getattr(next_raw, "state", "ERROR")) if next_raw is not None else "ERROR"
        next_grid = _primary_grid(next_raw) if next_raw is not None else grid
        next_level = int(getattr(next_raw, "levels_completed", level) or level) if next_raw is not None else level
        next_feats = extract_state_features(next_grid)
        rows.append(
            _record_transition(
                game_id=full_game_id,
                episode_source="human_replay",
                episode_index=row_episode_index,
                step_index=step_index,
                prev_grid=grid,
                prev_feats=feats,
                prev_level=level,
                history_features=make_history_features(
                    action_history,
                    action_repeat_count,
                    steps_since_state_change,
                ),
                action=action,
                action_data=action_data,
                next_grid=next_grid,
                next_feats=next_feats,
                next_level=next_level,
                state_name=next_state,
                with_debug=with_debug,
            )
        )
        changed = int(rows[-1]["changed_cells"])
        action_repeat_count = action_repeat_count + 1 if action_history and action_history[-1] == action else 1
        action_history.append(action)
        steps_since_state_change = 0 if changed > 0 else steps_since_state_change + 1
        if next_raw is None or next_state == "ERROR":
            break
        # GAME_OVER/WIN ends the current human episode, not the recorded
        # session. Keep the terminal observation so the following recorded
        # RESET can resume the same environment and preserve unlocked levels.
        grid, feats, level = next_grid, next_feats, next_level
    return rows


def _step_with_data(env: Any, action: str, data: Dict[str, Any]) -> Any:
    try:
        return env.step(_action_enum(action), data=data)
    except TypeError:
        return env.step(_action_enum(action))


# ---------------------------------------------------------------------------
# Post-processing: future outcomes + auto_levelup_progress (need context)
# ---------------------------------------------------------------------------
def _vector(feats: Dict[str, float]) -> np.ndarray:
    return np.array([float(feats.get(name, 0.0)) for name in FEATURE_SCHEMA], dtype=np.float32)


def _postprocess_game(rows: List[Dict[str, Any]]) -> None:
    """Add Monte-Carlo value targets and auto_levelup_progress in place."""

    # Group by (episode_source, episode_index) to respect episode boundaries.
    episodes: Dict[Tuple[str, int], List[int]] = {}
    for idx, row in enumerate(rows):
        episodes.setdefault((row["episode_source"], row["episode_index"]), []).append(idx)

    # Reference states: feature vectors just before a level-up (across game).
    references: List[np.ndarray] = []
    for row in rows:
        if row["level_up"]:
            references.append(_vector(row["state_features"]))

    for indices in episodes.values():
        n = len(indices)
        for pos, idx in enumerate(indices):
            row = rows[idx]
            # future level-up within horizon
            steps_to = -1
            for ahead in range(pos, n):
                if rows[indices[ahead]]["level_up"]:
                    steps_to = ahead - pos
                    break
            future_levelup = 0 <= steps_to <= LEVELUP_HORIZON
            # future game-over within short horizon
            danger_soon = False
            for ahead in range(pos, min(n, pos + DANGER_HORIZON + 1)):
                if rows[indices[ahead]]["game_over"]:
                    danger_soon = True
                    break
            row["future_level_up"] = bool(future_levelup)
            for horizon in LEVELUP_HORIZONS:
                row[f"future_level_up_within_{horizon}"] = bool(0 <= steps_to <= horizon)
            row["steps_to_level_up"] = int(steps_to)
            row["future_game_over_soon"] = bool(danger_soon)
            row["discounted_future_progress"] = round(
                float(1.0 / (1.0 + steps_to)) if steps_to >= 0 else 0.0,
                4,
            )
            if row["level_up"]:
                progress = 1.0
            elif row["game_over"]:
                progress = 0.0
            elif future_levelup:
                progress = float(1.0 / (1.0 + steps_to))
            else:
                progress = float(
                    max(0.0, min(0.5, 0.25 + 0.01 * row["correspondence_progress"]))
                )
            row["progress_score"] = round(progress, 4)

    # auto_levelup_progress: did this step move closer to a pre-levelup state?
    if references:
        ref_matrix = np.stack(references)
        for indices in episodes.values():
            for idx in indices:
                row = rows[idx]
                prev_v = _vector(row["state_features"])
                next_v = _vector(row["next_state_features"])
                prev_d = float(np.min(np.linalg.norm(ref_matrix - prev_v, axis=1)))
                next_d = float(np.min(np.linalg.norm(ref_matrix - next_v, axis=1)))
                row["auto_levelup_progress"] = round(prev_d - next_d, 4)
                # Refresh macro label now that auto_levelup_progress is known.
                total_cells = int(
                    float(row["state_features"].get("grid_height", 64.0))
                    * float(row["state_features"].get("grid_width", 64.0))
                )
                row["macro_scores"] = {
                    k: round(float(v), 4)
                    for k, v in labels.macro_scores(
                        delta=row["delta_features"],
                        changed_cells=row["changed_cells"],
                        total_cells=total_cells,
                        game_over=row["game_over"],
                        auto_levelup_progress=row["auto_levelup_progress"],
                    ).items()
                }
                row["macro_label"] = labels.macro_label(
                    delta=row["delta_features"],
                    changed_cells=row["changed_cells"],
                    total_cells=total_cells,
                    game_over=row["game_over"],
                    auto_levelup_progress=row["auto_levelup_progress"],
                )


# ---------------------------------------------------------------------------
# Budget allocation
# ---------------------------------------------------------------------------
def _allocate_episodes(total: int) -> Dict[str, int]:
    base = sum(SOURCE_RATIOS.values())
    counts = {src: int(round(total * ratio / base)) for src, ratio in SOURCE_RATIOS.items()}
    # Fix rounding drift so the counts sum to ``total``.
    drift = total - sum(counts.values())
    order = ["random", "heuristic_guided", "ontology_probe"]
    i = 0
    while drift != 0 and order:
        src = order[i % len(order)]
        counts[src] = max(0, counts[src] + (1 if drift > 0 else -1))
        drift += -1 if drift > 0 else 1
        i += 1
    return counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_dataset(
    *,
    games: List[str],
    episodes_per_game: int,
    max_steps: int,
    seed: int,
    human_replay_extra: int,
    human_replay_max_steps: int,
    with_debug: bool,
    out_path: Path,
    quiet: bool,
) -> Dict[str, Any]:
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ENV_DIR))
    counts = _allocate_episodes(episodes_per_game)
    source_totals: Dict[str, int] = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for game in games:
            short_id = game.split("-", 1)[0]
            full_game_id = _resolve_full_game_id(arc, game)
            game_rows: List[Dict[str, Any]] = []
            episode_index = 0
            rng = random.Random(f"{seed}:{full_game_id}")

            for source, policy in (
                ("random", "random"),
                ("ontology_probe", "ontology_probe"),
                ("heuristic_guided", "heuristic_guided"),
            ):
                for _ in range(counts.get(source, 0)):
                    t0 = time.time()
                    rows = _run_episode(
                        arc,
                        full_game_id,
                        episode_source=source,
                        episode_index=episode_index,
                        policy=policy,
                        max_steps=max_steps,
                        rng=rng,
                        with_debug=with_debug,
                    )
                    game_rows.extend(rows)
                    source_totals[source] = source_totals.get(source, 0) + len(rows)
                    episode_index += 1
                    if not quiet:
                        print(
                            f"[{full_game_id}] {source} ep#{episode_index} "
                            f"rows={len(rows)} ({time.time() - t0:.1f}s)"
                        )

            # Optional special human replay (trace-bearing games only).
            if human_replay_extra > 0:
                for _ in range(human_replay_extra):
                    rows = _run_human_replay(
                        arc,
                        full_game_id,
                        short_id,
                        episode_index=episode_index,
                        max_steps=human_replay_max_steps,
                        with_debug=with_debug,
                    )
                    if rows:
                        game_rows.extend(rows)
                        source_totals["human_replay"] = source_totals.get("human_replay", 0) + len(rows)
                        episode_index = max(
                            episode_index + 1,
                            max(int(row["episode_index"]) for row in rows) + 1,
                        )
                        if not quiet:
                            print(f"[{full_game_id}] human_replay rows={len(rows)}")

            _postprocess_game(game_rows)
            for row in game_rows:
                handle.write(json.dumps(row) + "\n")
            total_rows += len(game_rows)
            if not quiet:
                print(f"[{full_game_id}] total rows={len(game_rows)}")

    schema_path = out_path.with_suffix(".schema.json")
    summary = {
        "out": str(out_path),
        "games": games,
        "episodes_per_game": episodes_per_game,
        "episode_allocation": counts,
        "human_replay_extra": human_replay_extra,
        "human_replay_max_steps": human_replay_max_steps,
        "max_steps": max_steps,
        "seed": seed,
        "total_rows": total_rows,
        "rows_by_source": source_totals,
        "feature_schema": FEATURE_SCHEMA,
        "macro_labels": labels.MACRO_LABELS,
        "macro_score_names": labels.MACRO_SCORE_NAMES,
        "largest_component_feature_schema": LARGEST_COMPONENT_FEATURE_SCHEMA,
        "history_features": [
            "last_action",
            "last_two_actions",
            "action_repeat_count",
            "steps_since_state_change",
        ],
        "levelup_horizons": list(LEVELUP_HORIZONS),
    }
    schema_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not quiet:
        print(f"\nWrote {total_rows} rows -> {out_path}")
        print(f"Schema/summary -> {schema_path}")
        print(f"rows_by_source: {source_totals}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the abstraction transition dataset.")
    parser.add_argument(
        "--games",
        default="public",
        help="Split alias (public|public_seen|public_unseen_split|ar25) or comma list.",
    )
    parser.add_argument("--episodes-per-game", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--human-replay-extra",
        type=int,
        default=0,
        help="Extra human-replay episodes for trace-bearing games (ar25). Special add-on.",
    )
    parser.add_argument(
        "--human-replay-max-steps",
        type=int,
        default=None,
        help=(
            "Dedicated step cap for human replay (defaults to --max-steps). Set this "
            "high (e.g. 900) so the full winning human trace -- and its level-transition "
            "/ fracture states -- is captured instead of being truncated by the short "
            "search-episode budget."
        ),
    )
    parser.add_argument("--with-debug", action="store_true", help="Include game_specific_debug layer.")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    games = game_splits.resolve(args.games, full_ids=True)
    human_replay_max_steps = (
        args.human_replay_max_steps
        if args.human_replay_max_steps is not None
        else args.max_steps
    )
    build_dataset(
        games=games,
        episodes_per_game=args.episodes_per_game,
        max_steps=args.max_steps,
        seed=args.seed,
        human_replay_extra=args.human_replay_extra,
        human_replay_max_steps=human_replay_max_steps,
        with_debug=args.with_debug,
        out_path=Path(args.out),
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
