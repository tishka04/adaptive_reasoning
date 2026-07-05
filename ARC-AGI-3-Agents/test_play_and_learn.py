"""
Play-and-learn with brain-inspired associative memory.

Usage:
    python test_play_and_learn.py [num_games] [time_per_game] [actions_per_iter]

    num_games       — how many games to test (default: all)
    time_per_game   — seconds to spend per game (default: 30)
    actions_per_iter — actions per attempt (default: 100)

Strategy:
  Phase 1  EXPLORE — play random attempts until first win (time-limited).
           Each episode is consolidated (LTP/LTD) so the brain learns
           which (state, action) pairs lead to progress vs game-over.
  Phase 2  EXPLOIT — once a winning sequence exists, replay it and
           refine with guided exploration for a few more rounds.
"""
import sys, os, time, logging, random
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["OPERATION_MODE"] = "offline"
os.environ["ENVIRONMENTS_DIR"] = str(
    __import__("pathlib").Path(__file__).parent.parent / "environment_files"
)
os.environ["ARC_API_KEY"] = "test"
os.environ["RECORDINGS_DIR"] = "recordings"
logging.disable(logging.CRITICAL)  # silence ALL library logging

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

# -- action helpers ---------------------------------------------------------
_INT_TO_ACTION = {
    1: GameAction.ACTION1, 2: GameAction.ACTION2,
    3: GameAction.ACTION3, 4: GameAction.ACTION4,
    5: GameAction.ACTION5, 6: GameAction.ACTION6,
    7: GameAction.ACTION7,
}


def _pick_click_pos(grid, iteration, step):
    """Pick a click position using multiple strategies, cycling through them.

    Strategies (cycled by iteration):
      0 — non-zero cells (rotate)
      1 — unique-value cell representatives
      2 — edge cells of non-zero regions
      3 — systematic 4×4 grid scan
      4 — center of non-zero bounding box
      5 — random position (for coverage)
    """
    if grid is None:
        return {"x": random.randint(0, 63), "y": random.randint(0, 63)}

    h, w = grid.shape[:2]
    strategy = iteration % 6

    if strategy == 0:
        # Non-zero cells, rotating
        nz_y, nz_x = np.nonzero(grid)
        if len(nz_y) > 0:
            idx = (iteration * 7 + step) % len(nz_y)
            return {"x": int(nz_x[idx]), "y": int(nz_y[idx])}

    elif strategy == 1:
        # Click on representative cells of each unique value
        unique_vals = [v for v in np.unique(grid) if v != 0]
        if unique_vals:
            val = unique_vals[(iteration + step) % len(unique_vals)]
            ys, xs = np.where(grid == val)
            idx = step % len(ys)
            return {"x": int(xs[idx]), "y": int(ys[idx])}

    elif strategy == 2:
        # Edge cells of non-zero regions (borders between 0 and non-0)
        nz_y, nz_x = np.nonzero(grid)
        if len(nz_y) > 0:
            edges = []
            for yi, xi in zip(nz_y, nz_x):
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ny, nx = yi+dy, xi+dx
                    if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] == 0:
                        edges.append((xi, yi))
                        break
            if edges:
                idx = (iteration * 3 + step) % len(edges)
                return {"x": edges[idx][0], "y": edges[idx][1]}

    elif strategy == 3:
        # Systematic grid scan (4x4 = 16 positions per pass)
        scan_idx = (iteration * 13 + step) % 16
        row, col = divmod(scan_idx, 4)
        return {"x": int((col + 0.5) * w / 4), "y": int((row + 0.5) * h / 4)}

    elif strategy == 4:
        # Center of non-zero bounding box
        nz_y, nz_x = np.nonzero(grid)
        if len(nz_y) > 0:
            cy = int((nz_y.min() + nz_y.max()) // 2)
            cx = int((nz_x.min() + nz_x.max()) // 2)
            # Jitter around center
            jx = (step % 5 - 2) * max(1, w // 10)
            jy = (step % 3 - 1) * max(1, h // 10)
            return {"x": max(0, min(w-1, cx + jx)),
                    "y": max(0, min(h-1, cy + jy))}

    else:  # strategy == 5
        # Pure random for maximum coverage
        return {"x": random.randint(0, w-1), "y": random.randint(0, h-1)}

    # Fallback: grid scan
    scan_idx = (iteration * 13 + step) % 256
    row, col = divmod(scan_idx, 16)
    return {"x": int(min(w-1, (col + 0.5) * w / 16)),
            "y": int(min(h-1, (row + 0.5) * h / 16))}


def _parse_grid(frame):
    """Parse a frame into a uint8 numpy grid (fast path)."""
    if frame is None:
        return None
    arr = np.array(frame, dtype=np.uint8)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr


def _do_action(env, brain, act_int, act_data, prev_grid, it, step, level,
               states, recent_actions, episode_actions, f):
    """Execute one action, record results. Returns (f, prev_grid, level, changed)."""
    act = _INT_TO_ACTION.get(act_int, GameAction.ACTION1)
    if act_int == 6:
        if act_data is None:
            act_data = _pick_click_pos(prev_grid, it, step)
        act.set_data(act_data)

    f = env.step(act)
    if f is None or f.frame is None:
        return f, prev_grid, level, False

    cur_grid = _parse_grid(f.frame)
    changed = (cur_grid is not None and prev_grid is not None
               and not np.array_equal(prev_grid, cur_grid))
    lvl_changed = f.levels_completed > level
    game_over = f.state == GameState.GAME_OVER

    brain.record_step(prev_grid, act_int, act_data, changed, lvl_changed, game_over)
    recent_actions.append(act_int)
    episode_actions.append(act_int)
    if len(recent_actions) > 10:
        del recent_actions[:-10]

    if cur_grid is not None:
        states.add(hash(cur_grid.tobytes()))
    if lvl_changed:
        level = f.levels_completed
    return f, (cur_grid if cur_grid is not None else prev_grid), level, changed


# -- single episode runner ---------------------------------------------------
def _run_episode(env, brain, avail, it, actions_per_iter,
                 use_memory=False, temperature=999.0):
    """Play one episode. Returns (level, n_states).

    Exploration strategy (3 phases within each episode):
      1. PROBE   — on early iterations, test each action individually
      2. REPLAY  — if exploiting, replay best known procedure
      3. BUILD   — weighted random from productive actions (avoidance-aware)
    """
    try:
        env.reset()
        f = env.step(GameAction.RESET)
        if f is None or f.frame is None:
            return 0, 0
    except Exception:
        return 0, 0

    prev_grid = _parse_grid(f.frame)
    level = 0
    states = set()
    recent_actions: list = []
    episode_actions: list = []
    step_budget = actions_per_iter
    step_used = 0

    brain.begin_episode()

    # ── Phase 1: PROBE — systematic action testing (first 2 iterations) ──
    if it <= 2 and not use_memory:
        for probe_act in avail:
            if step_used >= step_budget or f is None:
                break
            if f.state in (GameState.GAME_OVER, GameState.NOT_PLAYED):
                brain.record_step(prev_grid, 0, None, False, False, True)
                try:
                    f = env.step(GameAction.RESET)
                    if f is None or f.frame is None:
                        break
                    prev_grid = _parse_grid(f.frame)
                    recent_actions = []
                    episode_actions = []
                except Exception:
                    break
            if f.state == GameState.WIN:
                break
            try:
                f, prev_grid, level, _ = _do_action(
                    env, brain, probe_act, None, prev_grid, it, step_used,
                    level, states, recent_actions, episode_actions, f,
                )
                step_used += 1
            except Exception:
                pass

    # ── Phase 2: REPLAY — best procedure (exploit mode only) ──────────
    procedure = brain.get_best_procedure() if use_memory else None
    if procedure:
        replay_len = min(len(procedure), (step_budget - step_used) // 2)
        for ri in range(replay_len):
            if step_used >= step_budget:
                break
            act_int, act_data = procedure[ri]
            if f is None or f.state in (GameState.GAME_OVER, GameState.WIN):
                break
            try:
                f, prev_grid, level, _ = _do_action(
                    env, brain, act_int, act_data, prev_grid, it, step_used,
                    level, states, recent_actions, episode_actions, f,
                )
                step_used += 1
                if f and f.state in (GameState.GAME_OVER, GameState.WIN):
                    break
            except Exception:
                break

    # ── Phase 3: BUILD — weighted exploration from productive actions ──
    while step_used < step_budget:
        try:
            if f is None or f.state in (GameState.GAME_OVER, GameState.NOT_PLAYED):
                brain.record_step(prev_grid, 0, None, False, False, True)
                f = env.step(GameAction.RESET)
                if f is None or f.frame is None:
                    break
                prev_grid = _parse_grid(f.frame)
                recent_actions = []
                episode_actions = []
                step_used += 1
                continue
            if f.state == GameState.WIN:
                break

            # Action selection
            if use_memory:
                act_int, act_data = brain.retrieve_action(
                    prev_grid, avail, recent_actions, temperature
                )
            else:
                act_int = brain.pick_novel_action(avail, episode_actions)
                act_data = None

            f, prev_grid, level, _ = _do_action(
                env, brain, act_int, act_data, prev_grid, it, step_used,
                level, states, recent_actions, episode_actions, f,
            )
            step_used += 1

        except Exception:
            try:
                f = env.step(GameAction.RESET)
                if f is not None and f.frame is not None:
                    prev_grid = _parse_grid(f.frame)
                else:
                    break
            except Exception:
                break

    # Consolidation
    if f is not None and f.state == GameState.WIN and level == 0:
        level = max(f.levels_completed, 1)
    brain.end_episode()

    return level, len(states)


# -- main play-and-learn loop -----------------------------------------------
def play_and_learn(arc, game_id, time_budget, actions_per_iter,
                   cross_game=None):
    """Play one game with cross-game learning.

    Phase 1 (EXPLORE): Random with avoidance + action productivity bias.
        - NN trains in background but does NOT pick actions.
        - Action priors from previous games seed the productivity weights.
    Phase 2 (EXPLOIT): Replay winning procedure + guided variations.
    """
    from v4_1_reasoning_system.arc_agi.associative_memory import AssociativeMemory

    brain = AssociativeMemory(
        ltp_rate=0.3,
        ltd_rate=0.1,
        decay_rate=0.005,
        max_episodes=5000,
        max_procedures=30,
    )
    # Inherit meta-knowledge from previous games
    if cross_game is not None:
        brain.new_game(cross_game)

    best_level = 0
    best_iter = -1
    total_iters = 0
    total_wins = 0
    avail = None
    start_time = time.time()

    # Discover available actions on first reset
    try:
        env = arc.make(game_id)
        env.reset()
        f = env.step(GameAction.RESET)
        if f is not None and f.available_actions:
            avail = list(f.available_actions)
    except Exception:
        pass
    if avail is None:
        avail = list(range(1, 8))

    phase = "explore"
    pbar = tqdm(desc=f"{game_id[:15]}", leave=False, ncols=80,
                bar_format="{desc}: {elapsed} | iter {n} | {postfix}")

    while True:
        elapsed = time.time() - start_time
        if elapsed >= time_budget:
            break

        total_iters += 1

        try:
            env = arc.make(game_id)
        except Exception:
            continue

        if phase == "explore":
            # Pure random with avoidance — NN trains but doesn't pick actions
            level, n_states = _run_episode(
                env, brain, avail, total_iters, actions_per_iter,
                use_memory=False, temperature=999.0,
            )
        else:
            # Exploit: replay procedure + NN-guided variations
            temperature = max(0.5, 1.5 - (total_iters - best_iter) * 0.03)
            level, n_states = _run_episode(
                env, brain, avail, total_iters, actions_per_iter,
                use_memory=True, temperature=temperature,
            )

        if level > 0:
            total_wins += 1
        if level > best_level:
            best_level = level
            best_iter = total_iters
            if phase == "explore":
                phase = "exploit"

        pbar.update(1)
        pbar.set_postfix_str(
            f"{phase} wins={total_wins} nn={brain._nn_blend:.1f}",
            refresh=True,
        )

        # Once we've consolidated enough wins, move on
        if phase == "exploit" and total_wins >= 10:
            break

    pbar.close()
    elapsed = time.time() - start_time

    # Export learnings back to cross-game memory
    if cross_game is not None:
        brain.export_to_cross_game(cross_game)

    return {
        "game": game_id,
        "best_level": best_level,
        "best_iter": best_iter,
        "total_iters": total_iters,
        "total_wins": total_wins,
        "win_rate": total_wins / max(total_iters, 1),
        "phase": phase,
        "time": round(elapsed, 1),
        "memory_stats": brain.stats(),
    }


# -- main ------------------------------------------------------------------
def main():
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 999
    time_per_game = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    actions_per = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=os.environ["ENVIRONMENTS_DIR"],
    )
    envs = arc.get_environments()
    game_ids = [e.game_id for e in envs][:num_games]

    # Shared cross-game memory — persists across all games
    from v4_1_reasoning_system.arc_agi.associative_memory import CrossGameMemory
    cross_game = CrossGameMemory()

    print(f"\nPlay-and-Learn (cross-game associative memory)")
    print(f"{len(game_ids)} games | {time_per_game}s per game | {actions_per} actions/attempt\n")

    results = []
    game_pbar = tqdm(game_ids, desc="Games", unit="game")
    for gid in game_pbar:
        game_pbar.set_postfix_str(f"{gid[:12]} xg={cross_game.games_played}")
        r = play_and_learn(arc, gid, time_per_game, actions_per, cross_game)
        results.append(r)
    game_pbar.close()

    # ── Summary table ─────────────────────────────────────────────────
    hdr = f"{'Game':<22} {'Iters':>6} {'Wins':>5} {'Rate':>6} {'Best':>5} {'Phase':>8} {'Procs':>5} {'ActEff':>18} {'Time':>7}"
    sep = "-" * len(hdr)
    print(f"\n{sep}")
    print(hdr)
    print(sep)
    total_wins = 0
    total_iters = 0
    for r in results:
        ms = r["memory_stats"]
        marker = "*" if r["total_wins"] > 0 else " "
        # Compact action effects: "A1:23% A3:81% ..."
        ae = ms.get("action_effects", {})
        ae_parts = []
        for act_num, ratio_str in ae.items():
            ch, tot = ratio_str.split("/")
            pct = int(ch) * 100 // max(int(tot), 1) if int(tot) > 0 else 0
            ae_parts.append(f"{act_num}:{pct}%")
        ae_summary = " ".join(ae_parts[:4])  # show top 4
        print(
            f"{marker}{r['game'][:21]:<21} "
            f"{r['total_iters']:>6} "
            f"{r['total_wins']:>5} "
            f"{r['win_rate']:>5.0%} "
            f"{r['best_level']:>5} "
            f"{r['phase']:>8} "
            f"{ms['procedures']:>5} "
            f"{ae_summary:>18} "
            f"{r['time']:>6.1f}s"
        )
        total_wins += r["total_wins"]
        total_iters += r["total_iters"]

    games_won = sum(1 for r in results if r["total_wins"] > 0)
    print(sep)
    print(
        f" {'TOTAL':<21} {total_iters:>6} {total_wins:>5}"
        f"{'':>30}"
        f" {games_won}/{len(results)} solved"
    )
    print(sep)

    # Cross-game memory stats
    xg = cross_game.stats()
    print(f"\nCross-game memory: {xg['games_played']} games, "
          f"{xg['games_won']} won, "
          f"{xg['nn_train_steps']} NN steps")
    if xg["action_priors"]:
        print("Action priors learned:")
        for act, desc in xg["action_priors"].items():
            print(f"  ACTION{act}: {desc}")


if __name__ == "__main__":
    main()
