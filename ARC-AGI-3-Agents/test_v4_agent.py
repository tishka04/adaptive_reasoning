"""
Test the V4 Adaptive Reasoning Agent on ARC-AGI-3 training games.

Usage:
    python test_v4_agent.py [num_games] [time_budget]
"""

import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
ENV_DIR = PROJECT_ROOT / "environment_files"

os.environ["OPERATION_MODE"] = "offline"
os.environ["ENVIRONMENTS_DIR"] = str(ENV_DIR)
os.environ.setdefault("ARC_API_KEY", "test")
os.environ.setdefault("RECORDINGS_DIR", "recordings")
os.environ["TQDM_DISABLE"] = "1"

logging.disable(logging.CRITICAL)

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, str(PROJECT_ROOT))

from arc_agi import Arcade, OperationMode
from arcengine import GameState


def aggregate_progress_counts(results, bucket_name, channel):
    totals = {}
    for item in results:
        progress_detail = item.get("progress_detail", {})
        counts = progress_detail.get(bucket_name, {}).get(channel, {})
        for key, value in counts.items():
            totals[str(key)] = totals.get(str(key), 0) + int(value)
    return totals


def summarize_selected_counts(counts, keys):
    parts = [f"{key}={counts.get(key, 0)}" for key in keys if counts.get(key, 0) > 0]
    return ", ".join(parts) if parts else "-"


def run_game(arc, game_id, time_budget, cross_game=None, agent_options=None):
    from agents.templates.adaptive_reasoning_v4_agent import AdaptiveReasoningV4

    original_time = AdaptiveReasoningV4.TIME_BUDGET
    AdaptiveReasoningV4.TIME_BUDGET = float(time_budget)

    env = arc.make(game_id)
    agent = AdaptiveReasoningV4(
        card_id=None,
        game_id=game_id,
        agent_name="adaptivev4",
        ROOT_URL="",
        record=False,
        arc_env=env,
        cross_game=cross_game,
        arcade=arc,
        **(agent_options or {}),
    )

    start = time.time()
    try:
        agent.main()
    except Exception as exc:
        import traceback

        print(f"\n  ERROR in {game_id}: {exc}")
        traceback.print_exc()

    elapsed = time.time() - start
    AdaptiveReasoningV4.TIME_BUDGET = original_time

    won = agent.frames[-1].state == GameState.WIN if agent.frames else False
    levels = agent.frames[-1].levels_completed if agent.frames else 0
    summary = agent.end_game()
    progress = summary.get("progress", {})
    ontologies = summary.get("current_ontologies", [])
    learning = summary.get("learning", {})
    diagnostics = summary.get("diagnostics", {})
    top_ontology = ontologies[0][0] if ontologies else "-"

    return {
        "game": game_id,
        "won": won,
        "levels": levels,
        "actions": agent.action_counter,
        "operators": summary.get("operators", 0),
        "rules": summary.get("rules", 0),
        "teleology": summary.get("teleology", 0),
        "motifs": summary.get("motifs", 0),
        "rituals": summary.get("rituals", 0),
        "knowledge": summary.get("knowledge_level", 0.0),
        "pred_acc": summary.get("pred_accuracy", 0.0),
        "ctrl_suc": summary.get("control_success", 0.0),
        "lp": progress.get("lp", 0.0),
        "sp": progress.get("sp", 0.0),
        "tp": progress.get("tp", 0.0),
        "progress_detail": progress,
        "branch": progress.get("branch_id", 0),
        "states": summary.get("states_visited", 0),
        "minds": summary.get("mind_selections", {}),
        "intents": summary.get("intent_counts", {}),
        "phases": summary.get("phase_counts", {}),
        "learning": learning,
        "diagnostics": diagnostics,
        "ontology": top_ontology,
        "time": round(elapsed, 1),
    }


def main():
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 999
    time_budget = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ENV_DIR),
    )
    envs = arc.get_environments()
    game_ids = [env.game_id for env in envs][:num_games]

    from v4.memory.cross_game_memory import CrossGameMemoryV4

    memory_path = str(PROJECT_ROOT / "cross_game_memory_v4.pkl")
    cross_game = CrossGameMemoryV4.load(memory_path)

    if cross_game.games_played > 0:
        print(
            f"  Resuming: {cross_game.games_played} games, "
            f"{cross_game.games_won} won, trust={cross_game.trust:.2f}"
        )

    width = 100
    print()
    print("=" * width)
    print("  V4 ADAPTIVE REASONING AGENT  -  Chambered Ecology")
    print("=" * width)
    print(f"  Games: {len(game_ids)}  |  Budget: {time_budget}s/game")
    print("=" * width)
    print()

    results = []
    for index, game_id in enumerate(game_ids, 1):
        print(f"  [{index}/{len(game_ids)}] {game_id} ...", end="", flush=True)
        result = run_game(arc, game_id, time_budget, cross_game)
        results.append(result)
        if result["won"]:
            print(f"  WIN  L{result['levels']} ({result['time']:.0f}s)")
        else:
            print(
                f"  ---  L{result['levels']} ops={result['operators']} rules={result['rules']} tel={result['teleology']} "
                f"mot={result['motifs']} rit={result['rituals']} "
                f"onto={result['ontology']} k={result['knowledge']:.2f} "
                f"LP={result['lp']:.2f} SP={result['sp']:.2f} TP={result['tp']:.2f} "
                f"({result['time']:.0f}s)"
            )

    print()
    print("=" * width)
    print("  RESULTS")
    print("=" * width)
    print(
        f"  {'Game':<16} {'Time':>4} {'Lvl':>3} {'Acts':>5} {'Ops':>3} "
        f"{'Rul':>3} {'Tel':>3} {'Mot':>3} {'Rit':>3} {'K':>4} "
        f"{'LP':>4} {'SP':>4} {'TP':>4}  {'Ont':<16}"
    )
    print("  " + "-" * (width - 4))
    for result in results:
        print(
            f"  {result['game'][:15]:<15} "
            f"{result['time']:>3.0f}s "
            f"{result['levels']:>3} "
            f"{result['actions']:>5} "
            f"{result['operators']:>3} "
            f"{result['rules']:>3} "
            f"{result['teleology']:>3} "
            f"{result['motifs']:>3} "
            f"{result['rituals']:>3} "
            f"{result['knowledge']:.2f} "
            f"{result['lp']:.2f} "
            f"{result['sp']:.2f} "
            f"{result['tp']:.2f}  "
            f"{result['ontology'][:16]:<16}"
        )

    print()
    print("=" * width)
    print("  SUMMARY")
    print("=" * width)
    solved = sum(1 for item in results if item["won"])
    total_levels = sum(item["levels"] for item in results)
    avg_ops = sum(item["operators"] for item in results) / max(len(results), 1)
    avg_rules = sum(item["rules"] for item in results) / max(len(results), 1)
    avg_tel = sum(item["teleology"] for item in results) / max(len(results), 1)
    avg_motifs = sum(item["motifs"] for item in results) / max(len(results), 1)
    avg_rituals = sum(item["rituals"] for item in results) / max(len(results), 1)
    avg_knowledge = sum(item["knowledge"] for item in results) / max(len(results), 1)
    avg_lp = sum(item["lp"] for item in results) / max(len(results), 1)
    avg_sp = sum(item["sp"] for item in results) / max(len(results), 1)
    avg_tp = sum(item["tp"] for item in results) / max(len(results), 1)
    avg_pred = sum(item["pred_acc"] for item in results) / max(len(results), 1)
    avg_ctrl = sum(item["ctrl_suc"] for item in results) / max(len(results), 1)
    total_actions = sum(item["actions"] for item in results)
    total_time = sum(item["time"] for item in results)
    avg_time = total_time / max(len(results), 1)
    progress_profile = next(
        (item.get("progress_detail", {}).get("profile") for item in results if item.get("progress_detail")),
        "unknown",
    )
    sp_awarded = aggregate_progress_counts(results, "awarded_event_counts", "sp")
    sp_missed = aggregate_progress_counts(results, "missed_event_counts", "sp")

    print(f"  Solved:            {solved}/{len(results)} games")
    print(f"  Total levels:      {total_levels}")
    print(f"  Total actions:     {total_actions:,}")
    print(f"  Avg operators:     {avg_ops:.1f}")
    print(f"  Avg rules:         {avg_rules:.1f}")
    print(f"  Avg teleology:     {avg_tel:.1f}")
    print(f"  Avg motifs:        {avg_motifs:.1f}")
    print(f"  Avg rituals:       {avg_rituals:.1f}")
    print(f"  Avg knowledge:     {avg_knowledge:.2f}")
    print(f"  Avg pred accuracy: {avg_pred:.1%}")
    print(f"  Avg ctrl success:  {avg_ctrl:.1%}")
    print(f"  Progress profile:  {progress_profile}")
    print(f"  Avg progress:      LP={avg_lp:.2f}  SP={avg_sp:.2f}  TP={avg_tp:.2f}")
    print(
        "  SP awarded:       "
        + summarize_selected_counts(
            sp_awarded,
            ["object_change", "structural_change", "novel_state", "region_unlock", "new_rules", "class_depletion"],
        )
    )
    print(
        "  SP missed:        "
        + summarize_selected_counts(
            sp_missed,
            ["object_change", "structural_change", "grid_change_without_sp", "class_depletion"],
        )
    )
    print(f"  Total time:        {total_time:.0f}s ({avg_time:.1f}s avg/game)")

    print()
    print("-" * width)
    print("  CROSS-GAME MEMORY (V4)")
    print("-" * width)
    print(
        f"  Games: {cross_game.games_played}  |  Won: {cross_game.games_won}  |  Trust: {cross_game.trust:.2f}"
    )
    print(f"  Ontology priors: {len(cross_game.ontology_priors)}")
    print(
        f"  Operator templates: {sum(len(items) for items in cross_game.operator_templates.values())} "
        f"across {len(cross_game.operator_templates)} kinds"
    )
    print(f"  Law families: {len(cross_game.law_families)}")
    print(f"  Terminal motifs: {len(cross_game.terminal_motifs)}")
    print(f"  Ritual signatures: {len(cross_game.ritual_signatures)}")
    print(f"  World frame embeddings: {len(getattr(cross_game, 'learned_world_frame_embeddings', {}))}")
    print(f"  World episode embeddings: {len(getattr(cross_game, 'learned_world_episode_embeddings', {}))}")
    print("=" * width)

    cross_game.save(memory_path)
    print(f"\n  Cross-game memory saved to {memory_path}")


if __name__ == "__main__":
    main()
