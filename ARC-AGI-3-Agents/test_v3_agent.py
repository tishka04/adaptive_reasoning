"""
Test the V3 Adaptive Reasoning Agent on ARC-AGI-3 training games.

Usage:
    python test_v3_agent.py [num_games] [time_budget]

    num_games   — how many games to test (default: all)
    time_budget — seconds per game (default: 60)
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

logging.disable(logging.CRITICAL)
os.environ["TQDM_DISABLE"] = "1"

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, str(PROJECT_ROOT))

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState


# ── Run one game with V3 agent ────────────────────────────────────
def run_game(arc, game_id, time_budget, cross_game=None):
    """Run the V3 agent on one game."""
    from agents.templates.adaptive_reasoning_v3_agent import AdaptiveReasoningV3

    original_time = AdaptiveReasoningV3.TIME_BUDGET
    AdaptiveReasoningV3.TIME_BUDGET = float(time_budget)

    env = arc.make(game_id)
    agent = AdaptiveReasoningV3(
        card_id=None,
        game_id=game_id,
        agent_name="adaptivev3",
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
        import traceback
        print(f"\n  ERROR in {game_id}: {e}")
        traceback.print_exc()

    elapsed = time.time() - start
    AdaptiveReasoningV3.TIME_BUDGET = original_time

    # Gather results from V3 internals
    v3 = agent.v3
    mem = v3.memory
    won = agent.frames[-1].state == GameState.WIN if agent.frames else False
    levels = agent.frames[-1].levels_completed if agent.frames else 0

    # End game: export cross-game + get summary
    summary = v3.end_game(won=won)

    leverage = summary.get("leverage", {})
    prog = leverage.get("progress", {})
    return {
        "game": game_id,
        "won": won,
        "levels": levels,
        "actions": agent.action_counter,
        "operators": len(v3.inducer.operators),
        "validated": leverage.get("validated_ops", 0),
        "rules": len(v3.rule_engine.rules),
        "macros": leverage.get("macros", 0),
        "states": len(v3._visited_hashes),
        "knowledge": round(mem.knowledge_level(), 2),
        "failures": len(mem.failure_patterns),
        "minds": dict(v3.arbiter._mind_selections),
        "modes": leverage.get("mode_counts", {}),
        "op_pct": leverage.get("operator_driven_pct", 0),
        "fb_pct": leverage.get("fallback_pct", 0),
        "pred_acc": leverage.get("pred_accuracy", 0),
        "ctrl_suc": leverage.get("control_success", 0),
        "compositional": leverage.get("compositional", False),
        "lp": prog.get("local_progress", 0),
        "sp": prog.get("structural_progress", 0),
        "tp": prog.get("terminal_progress", 0),
        "novel_states": prog.get("novel_states", 0),
        "class_removals": prog.get("class_removals", 0),
        "branch_kills": prog.get("branch_id", 0),
        "time": round(elapsed, 1),
    }


# ── Main ──────────────────────────────────────────────────────────
def main():
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 999
    time_budget = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ENV_DIR),
    )
    envs = arc.get_environments()
    game_ids = [e.game_id for e in envs][:num_games]

    # Cross-game memory (V3)
    from v3.memory.cross_game_memory import CrossGameMemoryV3
    MEMORY_PATH = str(PROJECT_ROOT / "cross_game_memory_v3.pkl")
    cross_game = CrossGameMemoryV3.load(MEMORY_PATH)

    if cross_game.games_played > 0:
        print(f"  Resuming: {cross_game.games_played} games, "
              f"{cross_game.games_won} won, trust={cross_game.trust:.2f}")

    W = 100
    print()
    print("=" * W)
    print("  V3 ADAPTIVE REASONING AGENT  —  Operator-Centric Play")
    print("=" * W)
    print(f"  Games: {len(game_ids)}  |  Budget: {time_budget}s/game")
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
            print(f"  WIN  L{r['levels']} ({r['time']:.0f}s)")
        else:
            modes = r.get('modes', {})
            exp = modes.get('experiment', 0)
            srch = modes.get('search', 0)
            comp = "C" if r.get('compositional') else "-"
            print(f"  ---  L{r['levels']} ops={r['operators']}({r['validated']}v) "
                  f"mac={r['macros']} [{comp}] "
                  f"k={r['knowledge']} pred={r['pred_acc']:.0%} "
                  f"LP={r['lp']:.2f} SP={r['sp']:.2f} TP={r['tp']:.2f} "
                  f"({r['time']:.0f}s)")

    # ── Results table ─────────────────────────────────────────────
    print()
    print("=" * W)
    print("  RESULTS")
    print("=" * W)

    print(
        f"  {'':>2} {'Game':<16} {'Time':>4} "
        f"{'Lvl':>3} {'Acts':>5} {'Ops':>3} {'Val':>3} {'Mac':>3} "
        f"{'K':>4} {'PrAc':>5} {'CtSc':>5} "
        f"{'LP':>4} {'SP':>4} {'TP':>4} {'BK':>2} {'C':>1}  "
        f"{'Minds':<25}"
    )
    print("  " + "-" * (W - 4))

    for r in results:
        mark = ">>" if r["won"] else "  "
        minds_str = ", ".join(
            f"{k}:{v}" for k, v in sorted(r["minds"].items(), key=lambda x: -x[1])
        )[:24]
        comp = "C" if r.get("compositional") else "-"
        print(
            f"  {mark} {r['game'][:15]:<15} "
            f"{r['time']:>3.0f}s "
            f"{r['levels']:>3} "
            f"{r['actions']:>5} "
            f"{r['operators']:>3} "
            f"{r['validated']:>3} "
            f"{r['macros']:>3} "
            f"{r['knowledge']:.2f} "
            f"{r['pred_acc']:>5.1%} "
            f"{r['ctrl_suc']:>5.1%} "
            f"{r['lp']:.2f} "
            f"{r['sp']:.2f} "
            f"{r['tp']:.2f} "
            f"{r['branch_kills']:>2} "
            f"{comp:>1}  "
            f"{minds_str:<25}"
        )

    # ── Summary ───────────────────────────────────────────────────
    games_won = sum(1 for r in results if r["won"])
    total_actions = sum(r["actions"] for r in results)
    total_time = sum(r["time"] for r in results)
    avg_ops = sum(r["operators"] for r in results) / max(len(results), 1)
    avg_rules = sum(r["rules"] for r in results) / max(len(results), 1)
    avg_knowledge = sum(r["knowledge"] for r in results) / max(len(results), 1)
    avg_time = total_time / max(len(results), 1)

    # Mind usage across all games
    all_minds: dict[str, int] = {}
    for r in results:
        for k, v in r["minds"].items():
            all_minds[k] = all_minds.get(k, 0) + v

    print()
    print("=" * W)
    print("  SUMMARY")
    print("=" * W)
    avg_validated = sum(r["validated"] for r in results) / max(len(results), 1)
    avg_pred_acc = sum(r["pred_acc"] for r in results) / max(len(results), 1)
    avg_ctrl_suc = sum(r["ctrl_suc"] for r in results) / max(len(results), 1)
    avg_op_pct = sum(r["op_pct"] for r in results) / max(len(results), 1)
    avg_fb_pct = sum(r["fb_pct"] for r in results) / max(len(results), 1)
    total_modes: dict[str, int] = {}
    for r in results:
        for k, v in r.get("modes", {}).items():
            total_modes[k] = total_modes.get(k, 0) + v

    avg_macros = sum(r["macros"] for r in results) / max(len(results), 1)
    compositional_count = sum(1 for r in results if r.get("compositional"))
    avg_lp = sum(r["lp"] for r in results) / max(len(results), 1)
    avg_sp = sum(r["sp"] for r in results) / max(len(results), 1)
    avg_tp = sum(r["tp"] for r in results) / max(len(results), 1)
    avg_novel_states = sum(r["novel_states"] for r in results) / max(len(results), 1)
    total_branch_kills = sum(r["branch_kills"] for r in results)
    total_levels = sum(r["levels"] for r in results)

    print(f"  Solved:             {games_won}/{len(results)} games")
    print(f"  Total levels:       {total_levels}")
    print(f"  Total actions:      {total_actions:,}")
    print(f"  Avg operators:      {avg_ops:.1f} ({avg_validated:.1f} validated)")
    print(f"  Avg rules:          {avg_rules:.1f}")
    print(f"  Avg macros:         {avg_macros:.1f}")
    print(f"  Compositional:      {compositional_count}/{len(results)} games")
    print(f"  Avg knowledge:      {avg_knowledge:.2f}")
    print(f"  Avg pred accuracy:  {avg_pred_acc:.1%}")
    print(f"  Avg ctrl success:   {avg_ctrl_suc:.1%}")
    print(f"  Avg operator-driven: {avg_op_pct:.0f}% (fallback: {avg_fb_pct:.0f}%)")
    print(f"  Avg progress:       LP={avg_lp:.2f}  SP={avg_sp:.2f}  TP={avg_tp:.2f}")
    print(f"  Avg novel states:   {avg_novel_states:.0f}  |  Total branch kills: {total_branch_kills}")
    print(f"  Total time:         {total_time:.0f}s ({avg_time:.1f}s avg/game)")

    if total_modes:
        print(f"\n  Mode totals:")
        for k in ["experiment", "search", "exploit", "fallback"]:
            v = total_modes.get(k, 0)
            pct = 100 * v / max(sum(total_modes.values()), 1)
            print(f"    {k}: {v} ({pct:.0f}%)")

    if all_minds:
        total_selections = sum(all_minds.values())
        print(f"\n  Mind selections ({total_selections} total):")
        for name, count in sorted(all_minds.items(), key=lambda x: -x[1]):
            pct = 100 * count / total_selections
            print(f"    {name}: {count} ({pct:.0f}%)")

    # Per-mind predictive accuracy
    mind_pred: dict[str, list] = {}
    for r in results:
        for k, v in r["minds"].items():
            mind_pred.setdefault(k, []).append(r["pred_acc"])
    if mind_pred:
        print(f"\n  Mind avg pred accuracy:")
        for name, accs in sorted(mind_pred.items()):
            avg = sum(accs) / len(accs)
            print(f"    {name}: {avg:.1%} (across {len(accs)} games)")

    # ── Cross-game memory ─────────────────────────────────────────
    print()
    print("-" * W)
    print("  CROSS-GAME MEMORY (V3)")
    print("-" * W)
    print(f"  Games: {cross_game.games_played}  |  Won: {cross_game.games_won}  |  Trust: {cross_game.trust:.2f}")
    print(f"  Operator priors: {sum(len(v) for v in cross_game.operator_priors.values())} across {len(cross_game.operator_priors)} kinds")
    print(f"  Rule priors: {len(cross_game.rule_priors)}")
    print(f"  Macro schemas: {len(cross_game.macro_schemas)}")
    print(f"  Failure patterns: {len(cross_game.failure_patterns)}")
    if cross_game.mind_reliability:
        print(f"  Mind reliability: {cross_game.mind_reliability}")
    print("=" * W)

    # Save cross-game memory
    cross_game.save(MEMORY_PATH)
    print(f"\n  Cross-game memory saved to {MEMORY_PATH}")


if __name__ == "__main__":
    main()
