"""
Full agentic play-and-learn: AdaptiveReasoning agent over training games.

Usage:
    python test_full_agent.py [num_games] [time_budget]

    num_games   — how many games to test (default: all 25)
    time_budget — seconds per game (default: 60)

This uses the FULL reasoning system:
  - StateDescriber, GoalDecomposer, StrategyGenerator
  - JEPA World Model + EBM scoring
  - LLM inference (Qwen2.5-0.5B)
  - Associative memory (LTP/LTD + policy gradient)
  - Goal pursuit: goal bank → goal-conditioned strategy → progress measurement
  - Unified adaptive loop: ε-decayed exploration ↔ goal pursuit
"""
import sys, os, time, logging
from pathlib import Path

# ── Environment setup ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
ENV_DIR = PROJECT_ROOT / "environment_files"

os.environ["OPERATION_MODE"] = "offline"
os.environ["ENVIRONMENTS_DIR"] = str(ENV_DIR)
os.environ.setdefault("ARC_API_KEY", "test")
os.environ.setdefault("RECORDINGS_DIR", "recordings")

# Silence ALL loggers — we control output via print()
logging.disable(logging.CRITICAL)  # nuclear option: suppress everything
os.environ["TQDM_DISABLE"] = "1"   # suppress tqdm progress bars

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, str(PROJECT_ROOT))

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState
from agents import AVAILABLE_AGENTS


# ── Run one game with the full agent ──────────────────────────────
def run_game(arc, game_id, time_budget, cross_game=None):
    """Run the full AdaptiveReasoning agent on one game with cross-game learning."""
    agent_cls = AVAILABLE_AGENTS["adaptivereasoning"]

    # Temporarily override budget
    original_time = agent_cls.TIME_BUDGET
    agent_cls.TIME_BUDGET = float(time_budget)

    env = arc.make(game_id)
    agent = agent_cls(
        card_id=None,
        game_id=game_id,
        agent_name="adaptivereasoning",
        ROOT_URL="",
        record=False,
        arc_env=env,
        cross_game=cross_game,
        arcade=arc,
    )

    start = time.time()
    try:
        agent.main()
    except Exception as e:
        pass  # agent handles errors internally

    elapsed = time.time() - start
    agent_cls.TIME_BUDGET = original_time

    # Gather results
    mem_summary = agent.memory.summary()
    loop_stats = agent.reasoning.get_stats()
    assoc_stats = loop_stats.get("assoc_memory", {})
    vc_stats = loop_stats.get("visual_cortex", {})

    # Check final state — use assoc_memory wins as primary (Phase 1 tracks them)
    assoc_wins = assoc_stats.get("total_wins", 0)
    won = assoc_wins > 0 or mem_summary["max_level"] > 0
    effective = sum(1 for e in agent.memory.action_history if e.anything_changed)
    level = max(mem_summary["max_level"], 1 if assoc_wins > 0 else 0)

    # Goal pursuit stats
    gp_stats = loop_stats.get("goal_pursuit", {})

    return {
        "game": game_id,
        "won": won,
        "level": level,
        "actions": agent.action_counter,
        "effective": effective,
        "states": mem_summary["states_visited"],
        "explore": round(mem_summary["exploration_score"], 2),
        "player": mem_summary["player_identified"],
        "goal": loop_stats.get("overarching_goal", None),
        "subgoals_done": loop_stats["subgoals_completed"],
        "strategies": loop_stats["strategies_tried"],
        "wm_steps": loop_stats["wm_trained_steps"],
        "assoc_episodes": assoc_stats.get("iterations", 0),
        "assoc_procs": assoc_stats.get("procedures", 0),
        "assoc_wins": assoc_stats.get("total_wins", 0),
        "vc_buffer": vc_stats.get("buffer_size", 0),
        "vc_steps": vc_stats.get("trained_steps", 0),
        "time": round(elapsed, 1),
        # Goal pursuit
        "gp_bank_size": gp_stats.get("bank_size", 0),
        "gp_bank_gen": gp_stats.get("bank_generation", 0),
        "gp_active_goal": gp_stats.get("active_goal_desc", None),
        "gp_outcomes": gp_stats.get("total_outcomes", 0),
        "gp_best_progress": gp_stats.get("active_goal_best_progress", 0.0),
        "gp_rejected": gp_stats.get("goals_rejected", 0),
        "gp_confirmed": gp_stats.get("goals_confirmed", 0),
    }


# ── Main ──────────────────────────────────────────────────────────
def main():
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 999
    time_budget = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    # Optional 3rd arg: "competition" mode (default: "development")
    mode = sys.argv[3] if len(sys.argv) > 3 else "development"

    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ENV_DIR),
    )
    envs = arc.get_environments()
    game_ids = [e.game_id for e in envs][:num_games]

    # Shared cross-game memory — persists across runs via disk
    from v4_1_reasoning_system.arc_agi.associative_memory import CrossGameMemory
    MEMORY_PATH = str(Path(__file__).parent.parent / "cross_game_memory.pt")
    cross_game = CrossGameMemory.load(MEMORY_PATH)
    cross_game.mode = mode  # override mode from CLI
    if cross_game.games_played > 0:
        print(f"  Resuming from previous run ({mode} mode): {cross_game.games_played} games, "
              f"{cross_game.games_won} wins, {cross_game.total_train_steps} NN steps, "
              f"trust={cross_game.prior_trust:.2f}")

    W = 100  # table width
    print()
    print("=" * W)
    print("  ADAPTIVE REASONING AGENT  —  Full Play-and-Learn")
    print("=" * W)
    print(f"  Games: {len(game_ids)}  |  Time budget: {time_budget}s per game  |  Stall restart: {time_budget * 0.25:.0f}s")
    print("=" * W)
    print()

    results = []
    solved = 0
    for i, gid in enumerate(game_ids, 1):
        tag = f"[{i}/{len(game_ids)}]"
        print(f"  {tag} {gid} ...", end="", flush=True)
        r = run_game(arc, gid, time_budget, cross_game)
        results.append(r)
        if r["won"]:
            solved += 1
            print(f"  WIN  ({r['assoc_wins']} wins, {r['time']:.0f}s)")
        else:
            print(f"  ---  ({r['time']:.0f}s)")

    # ==================================================================
    # Per-game results table
    # ==================================================================
    print()
    print("=" * W)
    print("  RESULTS")
    print("=" * W)

    # Header
    print(
        f"  {'':>2} {'Game':<16}  {'Time':>4}  "
        f"{'Wins':>4}  {'Eff':>4}  {'VC':>4}  "
        f"{'Bank':>4}  {'Out':>3}  {'BstP':>5}  "
        f"{'R/C':>3}  {'Goal':<28}"
    )
    print("  " + "-" * (W - 4))

    for r in results:
        mark = ">>" if r["won"] else "  "
        vc_info = f"{r['vc_steps']}" if r['vc_steps'] > 0 else "-"
        gp_goal = (r.get("gp_active_goal") or r.get("goal") or "-")[:27]
        gp_bank = f"{r['gp_bank_size']}" if r['gp_bank_size'] > 0 else "-"
        gp_rc = f"{r['gp_rejected']}/{r['gp_confirmed']}"
        gp_bp = f"{r['gp_best_progress']:.2f}" if r['gp_best_progress'] > 0 else "-"

        print(
            f"  {mark} {r['game'][:15]:<15}  "
            f"{r['time']:>3.0f}s  "
            f"{r['assoc_wins']:>4}  "
            f"{r['effective']:>4}  "
            f"{vc_info:>4}  "
            f"{gp_bank:>4}  "
            f"{r['gp_outcomes']:>3}  "
            f"{gp_bp:>5}  "
            f"{gp_rc:>3}  "
            f"{gp_goal:<28}"
        )

    # ==================================================================
    # Summary
    # ==================================================================
    games_won = sum(1 for r in results if r["won"])
    total_actions = sum(r["actions"] for r in results)
    total_effective = sum(r["effective"] for r in results)
    total_time = sum(r["time"] for r in results)
    total_assoc_wins = sum(r.get("assoc_wins", 0) for r in results)
    total_vc_steps = sum(r.get("vc_steps", 0) for r in results)
    avg_explore = sum(r["explore"] for r in results) / max(len(results), 1)
    avg_time = total_time / max(len(results), 1)

    print()
    print("=" * W)
    print("  SUMMARY")
    print("=" * W)
    print(f"  Solved:           {games_won}/{len(results)} games")
    print(f"  Total wins:       {total_assoc_wins} level completions across all games")
    print(f"  Total actions:    {total_actions:,}")
    print(f"  Effective actions: {total_effective:,} ({100*total_effective/max(total_actions,1):.1f}%)")
    print(f"  Avg exploration:  {avg_explore:.2f}")
    print(f"  Visual cortex:    {total_vc_steps} training steps")
    print(f"  Total time:       {total_time:.0f}s ({avg_time:.1f}s avg/game)")

    # Goal pursuit summary
    total_gp_outcomes = sum(r.get("gp_outcomes", 0) for r in results)
    total_gp_banks = sum(r.get("gp_bank_gen", 0) for r in results)
    total_gp_rejected = sum(r.get("gp_rejected", 0) for r in results)
    total_gp_confirmed = sum(r.get("gp_confirmed", 0) for r in results)
    avg_best_progress = sum(r.get("gp_best_progress", 0) for r in results) / max(len(results), 1)
    print()
    print("  Goal pursuit:")
    print(f"    Banks generated: {total_gp_banks}")
    print(f"    Strategy outcomes: {total_gp_outcomes}")
    print(f"    Goals rejected/confirmed: {total_gp_rejected}/{total_gp_confirmed}")
    print(f"    Avg best progress: {avg_best_progress:.3f}")

    # ==================================================================
    # Cross-game memory
    # ==================================================================
    xg = cross_game.stats()
    print()
    print("-" * W)
    print("  CROSS-GAME MEMORY")
    print("-" * W)
    print(f"  Mode: {xg.get('mode', 'development')}  |  Prior trust: {cross_game.prior_trust:.2f}")
    print(f"  Games played: {xg['games_played']}  |  Games won: {xg['games_won']}  |  NN steps: {xg['nn_train_steps']}")
    if xg["action_priors"]:
        print("  Action priors:")
        for act, desc in sorted(xg["action_priors"].items()):
            contra = cross_game.contradicted_priors.get(act, 0)
            suffix = f" [contradicted {contra}x]" if contra else ""
            print(f"    ACTION{act}: {desc}{suffix}")
    if hasattr(cross_game, 'goal_strategy_hints') and cross_game.goal_strategy_hints:
        print(f"  Goal strategy hints: {len(cross_game.goal_strategy_hints)} goal types")
        for gid, hints in list(cross_game.goal_strategy_hints.items())[:5]:
            def _hint_progress(h):
                if isinstance(h, dict):
                    return h.get("progress", 0)
                return getattr(h, "progress_score", 0)
            best_p = max(_hint_progress(h) for h in hints) if hints else 0
            print(f"    {gid}: {len(hints)} hints, best={best_p:.2f}")
    if cross_game.failure_patterns:
        print(f"  Failure patterns: {sum(len(v) for v in cross_game.failure_patterns.values())} records across {len(cross_game.failure_patterns)} goal types")
        for gid, fails in list(cross_game.failure_patterns.items())[:5]:
            print(f"    {gid}: {len(fails)} failures, proxy_progress={fails[-1].get('proxy_progress', 0):.2f}")
    if cross_game.overpredicted_goals:
        print(f"  Over-predicted goals: {dict(cross_game.overpredicted_goals)}")
    print("=" * W)

    # ── Persist cross-game memory to disk for next run ──
    cross_game.save(MEMORY_PATH)
    print(f"\n  Cross-game memory saved to {MEMORY_PATH}")


if __name__ == "__main__":
    main()
