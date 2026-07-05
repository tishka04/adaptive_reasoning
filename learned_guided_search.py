"""Phase 4: reinjection of the learned score into a guided search.

Action selection blends the (true) handmade heuristic teacher with the learned
score for progressive replacement:

    blended = (1 - w) * handmade_real + w * learned

    w = 0  -> pure handmade teacher (regression guard: reproduces current behavior)
    w = 1  -> pure learned scoring

The handmade component uses a real 1-step deepcopy lookahead (the same generic
teacher used to label the dataset). The learned component uses Model A + Model B
predictions without stepping. This is additive: existing search modes elsewhere
are untouched.

Run with the bundled env interpreter:
    ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe learned_guided_search.py \\
        --games ar25 --learned-weight 0.5 \\
        --action-effect models\\action_effect.joblib --value models\\value.joblib
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from level7_frontier_recovery import (
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

import game_splits
from abstraction_dataset_io import make_history_features
from extract_state_abstractions import extract_state_features, largest_component_local_features
from learned_scoring import LearnedScorer, blended_score

ACTIONS = [f"ACTION{i}" for i in range(1, 8)]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "diagnostics" / "learned_guided_search"


@dataclass
class StepRecord:
    step: int
    action: str
    is_radar: bool
    is_escape: bool
    break_probability: float
    predicted_danger: float
    macro_explore: float
    blended_score: float
    changed_cells: int
    level: int
    component_count: float
    largest_component_size: float


@dataclass
class GameResult:
    game_id: str
    learned_weight: float
    steps: int
    level_reached: int
    won: bool
    died: bool
    actions_to_level_up: int  # -1 if no level-up
    no_op_rate: float
    game_over: bool
    unique_states: int
    radar_steps: int = 0
    radar_overrides: int = 0
    escape_steps: int = 0
    danger_memory_size: int = 0
    trajectory: List[StepRecord] = field(default_factory=list)


def _action_data(action: str, shape, rng) -> Optional[Dict[str, int]]:
    if action != "ACTION6":
        return None
    h, w = shape
    if rng is None:
        return {"x": int(w) // 2, "y": int(h) // 2}
    return {"x": rng.randint(0, max(0, int(w) - 1)), "y": rng.randint(0, max(0, int(h) - 1))}


def _step(env, full_game_id, action, action_data):
    data: Dict[str, Any] = {"game_id": full_game_id}
    if action_data:
        data.update(action_data)
    try:
        return env.step(_action_enum(action), data=data)
    except TypeError:
        return env.step(_action_enum(action))


def _changed(a, b) -> int:
    left = np.array(a, dtype=np.int32)
    right = np.array(b, dtype=np.int32)
    if left.shape != right.shape:
        return int(max(left.size, right.size))
    return int(np.count_nonzero(left != right))


def _local_features_by_action(grid, actions: List[str], shape) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for action in actions:
        cursor = None
        if action == "ACTION6":
            h, w = shape
            cursor = (float(int(h) // 2), float(int(w) // 2))
        out[action] = largest_component_local_features(grid, cursor=cursor)
    return out


def _handmade_real_score(env, full_game_id, action, grid, feats, visited, shape) -> Tuple[float, Optional[Dict[str, int]], str, int]:
    """True 1-step lookahead teacher score (mirrors dataset heuristic).

    Returns ``(score, action_data, next_state, changed_cells)`` so callers can
    reuse the lookahead state/no-op signal as break-radar guards.
    """

    action_data = None
    if action == "ACTION6":
        h, w = shape
        action_data = {"x": int(w) // 2, "y": int(h) // 2}
    probe = copy.deepcopy(env)
    raw = _step(probe, full_game_id, action, action_data)
    if raw is None:
        return -1e18, action_data, "ERROR", 0
    state = _state_name(getattr(raw, "state", "ERROR"))
    nlevel = int(getattr(raw, "levels_completed", 0) or 0)
    ngrid = _primary_grid(raw)
    nfeats = extract_state_features(ngrid)
    changed = _changed(grid, ngrid)
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
    return float(score), action_data, state, changed


def _pick_radar_action(
    action_scores: Dict[str, Any],
    lookahead_state: Dict[str, str],
    lookahead_changed: Dict[str, int],
    use_handmade: bool,
) -> Optional[str]:
    """Break-radar / curiosity head.

    Among guarded break candidates (``break_bonus_applied`` already enforces the
    danger / macro-no-op / ACTION6 guards), pick the one with the highest
    predicted break probability. When a real lookahead is available, additionally
    drop game-over and no-op branches. Returns None when nothing qualifies.
    """

    best_action: Optional[str] = None
    best_prob = -1.0
    for action, score in action_scores.items():
        if not getattr(score, "break_bonus_applied", False):
            continue
        if use_handmade:
            if lookahead_state.get(action, "") in {"GAME_OVER", "ERROR"}:
                continue
            if lookahead_changed.get(action, 1) == 0:
                continue
        prob = float(getattr(score, "predicted_break_probability", 0.0))
        if prob > best_prob:
            best_prob = prob
            best_action = action
    return best_action


def run_game(
    arc: Arcade,
    full_game_id: str,
    *,
    learned_weight: float,
    scorer: Optional[LearnedScorer],
    max_steps: int,
    rng=None,
    break_radar_fraction: float = 0.0,
    danger_memory: Optional[Dict[Tuple[str, str], float]] = None,
) -> GameResult:
    env = arc.make(full_game_id)
    raw = getattr(env, "observation_space", None)
    grid = _primary_grid(raw)
    feats = extract_state_features(grid)
    level = int(getattr(raw, "levels_completed", 0) or 0)
    start_level = level
    visited = {_hash_grid(grid)}

    no_ops = 0
    steps = 0
    actions_to_level_up = -1
    died = False
    won = False
    action_history: List[str] = []
    action_repeat_count = 0
    steps_since_state_change = 0

    use_handmade = learned_weight < 1.0 - 1e-9
    use_learned = learned_weight > 1e-9 and scorer is not None
    use_radar = (
        break_radar_fraction > 1e-9
        and use_learned
        and getattr(scorer, "break_model", None) is not None
    )
    # Beam injection: every step banks break_radar_fraction of a radar slot; a full
    # slot is spent on the next step where a guarded radar candidate exists. Radar
    # branches are injected as soon as they are available, not at a fixed tick.
    radar_budget_accum = 0.0  # fractional accumulator
    radar_steps = 0
    radar_overrides = 0
    trajectory: List[StepRecord] = []

    # Anti-attractor state.
    # no_effect_count[(state_sig, action)]: times this action produced 0 changed
    # cells in this abstract state -> score penalty after NO_EFFECT_BAN_AFTER.
    # action_usage[action]: per-run usage, used to prefer unknown-effect actions
    # during a stagnation escape.
    # danger_memory[(state_sig, action)]: observed (real or lookahead) game-over.
    # Shared across runs when passed in by the caller: this is how the agent
    # "learns the walls". Observation dominates the model: a 1.0 entry applies a
    # -1e6 penalty regardless of predicted_danger.
    NO_EFFECT_BAN_AFTER = 3
    NO_OP_PENALTY = 100.0
    DANGER_PENALTY = 1e6
    # Soft escape (preventive): fire early on action-repetition or low state
    # novelty, not only after a long strict no-op stall.
    STAGNATION_ESCAPE_AFTER = 8
    REPEAT_WINDOW = 8           # last 8 actions with <= 2 distinct -> loop
    NOVELTY_WINDOW = 12         # last 12 states with <= 3 unique -> cycle
    NOVELTY_MIN_UNIQUE = 3
    ESCAPE_BAN_LAST_K = 3
    ESCAPE_COOLDOWN = 5
    no_effect_count: Dict[Tuple[str, str], int] = {}
    action_usage: Dict[str, int] = {}
    if danger_memory is None:
        danger_memory = {}
    escape_steps = 0
    last_escape_step = -10**9
    recent_state_sigs: List[str] = []

    for step_index in range(max_steps):
        available = [a for a in (_available_names_from_raw(raw) or ACTIONS) if a in ACTIONS]
        if not available:
            break
        shape = np.array(grid, dtype=np.int32).shape
        shape = (int(shape[0]), int(shape[1])) if len(shape) == 2 else (64, 64)

        history_features = make_history_features(
            action_history,
            action_repeat_count,
            steps_since_state_change,
        )
        action_scores = (
            {
                s.action: s
                for s in scorer.score_actions(
                    feats,
                    available,
                    history_features,
                    _local_features_by_action(grid, available, shape),
                )
            }
            if use_learned
            else {}
        )

        state_sig = _hash_grid(grid)
        best_action = available[0]
        best_data = None
        best_blend = -1e30
        lookahead_state: Dict[str, str] = {}
        lookahead_changed: Dict[str, int] = {}
        data_by_action: Dict[str, Optional[Dict[str, int]]] = {}
        blend_by_action: Dict[str, float] = {}
        for action in available:
            handmade = 0.0
            action_data = _action_data(action, shape, rng)
            if use_handmade:
                handmade, action_data, la_state, la_changed = _handmade_real_score(
                    env, full_game_id, action, grid, feats, visited, shape
                )
                lookahead_state[action] = la_state
                lookahead_changed[action] = la_changed
            data_by_action[action] = action_data
            # Learn walls from the lookahead too: an observed game-over dominates
            # the model's predicted_danger.
            if use_handmade and lookahead_state.get(action, "") in {"GAME_OVER", "ERROR"}:
                danger_memory[(state_sig, action)] = 1.0
            learned = action_scores[action].learned_score if use_learned else 0.0
            blend = blended_score(handmade, learned, learned_weight)
            # Anti-attractor channel 1: penalise actions repeatedly observed to be
            # no-ops in this exact abstract state.
            if no_effect_count.get((state_sig, action), 0) >= NO_EFFECT_BAN_AFTER:
                blend -= NO_OP_PENALTY
            # Danger memory dominates everything (observation > model).
            blend -= DANGER_PENALTY * danger_memory.get((state_sig, action), 0.0)
            blend_by_action[action] = blend
            if blend > best_blend:
                best_blend = blend
                best_action = action
                best_data = action_data

        # Anti-attractor channel 2: soft preventive escape. Fires on any of:
        #   (a) strict no-op stall (early threshold),
        #   (b) action-repetition loop (last REPEAT_WINDOW actions <= 2 distinct),
        #   (c) low state novelty (last NOVELTY_WINDOW states <= NOVELTY_MIN_UNIQUE
        #       unique) -> catches 2-cycles where cells change but nothing is new.
        # Escape candidates: simple actions (no ACTION6 unless nothing else), not
        # used recently, danger_memory < 1.0, lookahead-safe. Cooldown avoids
        # thrashing.
        is_escape_step = False
        repeat_loop = (
            len(action_history) >= REPEAT_WINDOW
            and len(set(action_history[-REPEAT_WINDOW:])) <= 2
        )
        novelty_stall = (
            len(recent_state_sigs) >= NOVELTY_WINDOW
            and len(set(recent_state_sigs[-NOVELTY_WINDOW:])) <= NOVELTY_MIN_UNIQUE
        )
        should_escape = (
            steps_since_state_change >= STAGNATION_ESCAPE_AFTER
            or repeat_loop
            or novelty_stall
        )
        if should_escape and (step_index - last_escape_step) > ESCAPE_COOLDOWN:
            banned = set(action_history[-ESCAPE_BAN_LAST_K:])
            def _escape_pool(pool: List[str]) -> List[str]:
                return [
                    a for a in pool
                    if a not in banned
                    and danger_memory.get((state_sig, a), 0.0) < 1.0
                    and not (use_handmade and lookahead_state.get(a, "") in {"GAME_OVER", "ERROR"})
                ]
            # ACTION6 only as a last resort (coordinates are not validated).
            candidates = _escape_pool([a for a in available if a != "ACTION6"])
            if not candidates:
                candidates = _escape_pool(list(available))
            if candidates:
                escape_action = min(
                    candidates,
                    key=lambda a: (
                        no_effect_count.get((state_sig, a), 0),
                        action_usage.get(a, 0),
                        -blend_by_action.get(a, -1e30),
                    ),
                )
                if escape_action != best_action:
                    best_action = escape_action
                    best_data = data_by_action.get(escape_action)
                    best_blend = blend_by_action.get(escape_action, best_blend)
                is_escape_step = True
                escape_steps += 1
                last_escape_step = step_index

        # Break-radar beam injection (own channel: candidates come straight from
        # action_scores, NOT from the anti-attractor-filtered beam, so a guarded
        # break branch can never be masked by the no-op penalty).
        is_radar_step = False
        if use_radar and not is_escape_step:
            radar_budget_accum += break_radar_fraction
            if radar_budget_accum >= 1.0:
                radar_steps += 1
                radar_action = _pick_radar_action(
                    action_scores, lookahead_state, lookahead_changed, use_handmade
                )
                if radar_action is not None and radar_action != best_action:
                    best_action = radar_action
                    best_data = data_by_action.get(radar_action)
                    best_blend = blend_by_action.get(radar_action, best_blend)
                    radar_overrides += 1
                    is_radar_step = True
                radar_budget_accum -= 1.0

        # Snapshot signals before the step (pre-step state).
        _sc = action_scores.get(best_action)
        _break_prob = float(getattr(_sc, "predicted_break_probability", 0.0)) if _sc else 0.0
        _danger = float(getattr(_sc, "predicted_danger", 0.0)) if _sc else 0.0
        _macro_explore = float(
            getattr(_sc, "predicted_macro_scores", {}).get("explore", 0.0)
        ) if _sc else 0.0

        next_raw = _step(env, full_game_id, best_action, best_data)
        state = _state_name(getattr(next_raw, "state", "ERROR")) if next_raw is not None else "ERROR"
        next_grid = _primary_grid(next_raw) if next_raw is not None else grid
        next_level = int(getattr(next_raw, "levels_completed", level) or level) if next_raw is not None else level
        steps += 1
        changed = _changed(grid, next_grid)
        if changed == 0:
            no_ops += 1

        # Anti-attractor bookkeeping.
        action_usage[best_action] = action_usage.get(best_action, 0) + 1
        if changed == 0:
            key = (state_sig, best_action)
            no_effect_count[key] = no_effect_count.get(key, 0) + 1
        # Observed death dominates the model: remember the wall.
        if next_raw is None or state in {"GAME_OVER", "ERROR"}:
            danger_memory[(state_sig, best_action)] = 1.0
        recent_state_sigs.append(state_sig)
        if len(recent_state_sigs) > NOVELTY_WINDOW:
            recent_state_sigs.pop(0)

        trajectory.append(StepRecord(
            step=steps,
            action=best_action,
            is_radar=is_radar_step,
            is_escape=is_escape_step,
            break_probability=round(_break_prob, 5),
            predicted_danger=round(_danger, 5),
            macro_explore=round(_macro_explore, 5),
            blended_score=round(float(best_blend), 5),
            changed_cells=changed,
            level=next_level,
            component_count=round(float(feats.get("component_count", 0.0)), 3),
            largest_component_size=round(float(feats.get("largest_component_size", 0.0)), 3),
        ))
        steps_since_state_change = 0 if changed > 0 else steps_since_state_change + 1
        action_repeat_count = (
            action_repeat_count + 1
            if action_history and action_history[-1] == best_action
            else 1
        )
        action_history.append(best_action)
        if next_level > level and actions_to_level_up < 0:
            actions_to_level_up = steps
        level = next_level

        if next_raw is None or state in {"GAME_OVER", "ERROR"}:
            died = True
            break
        if state == "WIN":
            won = True
            break
        raw = next_raw
        grid = next_grid
        feats = extract_state_features(grid)
        visited.add(_hash_grid(grid))

    return GameResult(
        game_id=full_game_id,
        learned_weight=float(learned_weight),
        steps=steps,
        level_reached=int(level),
        won=won,
        died=died,
        actions_to_level_up=actions_to_level_up,
        no_op_rate=round(no_ops / steps, 4) if steps else 0.0,
        game_over=died,
        unique_states=len(visited),
        radar_steps=radar_steps,
        radar_overrides=radar_overrides,
        escape_steps=escape_steps,
        danger_memory_size=len(danger_memory),
        trajectory=trajectory,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Learned guided search (blended handmade + learned).")
    parser.add_argument("--games", default="ar25")
    parser.add_argument("--learned-weight", type=float, default=0.5)
    parser.add_argument("--action-effect", default="models/action_effect.joblib")
    parser.add_argument("--value", default="models/value.joblib")
    parser.add_argument("--macro-scores", default=None)
    parser.add_argument("--break-classifier", default=None)
    parser.add_argument("--break-bonus-weight", type=float, default=2.0)
    parser.add_argument(
        "--break-radar-fraction",
        type=float,
        default=0.0,
        help="Fraction of steps where the guarded break-radar overrides selection (e.g. 0.2).",
    )
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--out", default=str(DEFAULT_REPORT_DIR / "report.json"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ENV_DIR))
    scorer = None
    if args.learned_weight > 1e-9:
        scorer = LearnedScorer(
            args.action_effect,
            args.value,
            macro_scores_path=args.macro_scores,
            break_classifier_path=args.break_classifier,
            break_bonus_weight=args.break_bonus_weight,
        )

    games = game_splits.resolve(args.games, full_ids=True)
    results = []
    for game in games:
        full_game_id = _resolve_full_game_id(arc, game)
        result = run_game(
            arc, full_game_id,
            learned_weight=args.learned_weight, scorer=scorer, max_steps=args.max_steps,
            break_radar_fraction=args.break_radar_fraction,
        )
        results.append(result.__dict__)
        if not args.quiet:
            print(
                f"[{full_game_id}] w={args.learned_weight} level={result.level_reached} "
                f"steps={result.steps} no_op={result.no_op_rate} died={result.died} "
                f"a2lvl={result.actions_to_level_up}"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"learned_weight": args.learned_weight, "results": results}, indent=2), encoding="utf-8")
    if not args.quiet:
        print(f"Report -> {out_path}")


if __name__ == "__main__":
    main()
