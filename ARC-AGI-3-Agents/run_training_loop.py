"""
Training loop — runs test_full_agent.py repeatedly to compound cross-game learning.

Usage:
    python run_training_loop.py [iterations] [num_games] [time_budget]

    iterations  — number of full test runs (default: 5)
    num_games   — games per run (default: 25)
    time_budget — seconds per game (default: 60)

Cross-game memory persists automatically between runs via cross_game_memory.pt.
Each iteration's results are captured and a comparative diagnostic is printed.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm

SCRIPT = str(Path(__file__).parent / "test_full_agent.py")
PYTHON = sys.executable
MEMORY_PATH = str(Path(__file__).parent.parent / "cross_game_memory.pt")


def parse_run_output(output: str) -> dict:
    """Extract key metrics from test_full_agent.py stdout."""
    m = {}

    # Solved: N/M games
    match = re.search(r"Solved:\s+(\d+)/(\d+)", output)
    if match:
        m["solved"] = int(match.group(1))
        m["total_games"] = int(match.group(2))

    # Total wins: N level completions
    match = re.search(r"Total wins:\s+(\d+)", output)
    if match:
        m["total_wins"] = int(match.group(1))

    # Total actions: N
    match = re.search(r"Total actions:\s+([\d,]+)", output)
    if match:
        m["total_actions"] = int(match.group(1).replace(",", ""))

    # Effective actions: N (P%)
    match = re.search(r"Effective actions:\s+([\d,]+)\s+\(([\d.]+)%\)", output)
    if match:
        m["effective_actions"] = int(match.group(1).replace(",", ""))
        m["effective_pct"] = float(match.group(2))

    # Avg exploration: F
    match = re.search(r"Avg exploration:\s+([\d.]+)", output)
    if match:
        m["avg_exploration"] = float(match.group(1))

    # Visual cortex: N training steps
    match = re.search(r"Visual cortex:\s+(\d+)", output)
    if match:
        m["vc_steps"] = int(match.group(1))

    # Banks generated: N
    match = re.search(r"Banks generated:\s+(\d+)", output)
    if match:
        m["banks_generated"] = int(match.group(1))

    # Strategy outcomes: N
    match = re.search(r"Strategy outcomes:\s+(\d+)", output)
    if match:
        m["strategy_outcomes"] = int(match.group(1))

    # Avg best progress: F
    match = re.search(r"Avg best progress:\s+([\d.]+)", output)
    if match:
        m["avg_best_progress"] = float(match.group(1))

    # Goals rejected/confirmed: N/N
    match = re.search(r"Goals rejected/confirmed:\s+(\d+)/(\d+)", output)
    if match:
        m["goals_rejected"] = int(match.group(1))
        m["goals_confirmed"] = int(match.group(2))

    # Cross-game: Games played: N | Games won: N | NN steps: N
    match = re.search(r"Games played:\s+(\d+)\s+\|\s+Games won:\s+(\d+)\s+\|\s+NN steps:\s+(\d+)", output)
    if match:
        m["xg_games_played"] = int(match.group(1))
        m["xg_games_won"] = int(match.group(2))
        m["xg_nn_steps"] = int(match.group(3))

    # Goal strategy hints: N goal types
    match = re.search(r"Goal strategy hints:\s+(\d+)", output)
    if match:
        m["xg_goal_types"] = int(match.group(1))

    # Per-game WIN lines (count >> markers)
    m["win_games"] = [g.strip() for g in re.findall(r">>\s+(\S+)", output)]

    return m


def delta_str(cur, prev, key, fmt=".0f", higher_is_better=True):
    """Format a metric with delta from previous run."""
    c = cur.get(key)
    p = prev.get(key) if prev else None
    if c is None:
        return "  —"
    s = f"{c:{fmt}}"
    if p is not None and p != c:
        d = c - p
        arrow = "▲" if (d > 0) == higher_is_better else "▼"
        s += f" ({arrow}{abs(d):{fmt}})"
    return s


def print_diagnostic(history: list):
    """Print a comparative table across all runs."""
    W = 100
    print()
    print("=" * W)
    print("  TRAINING PROGRESS — COMPARATIVE DIAGNOSTIC")
    print("=" * W)

    # Header
    cols = ["Run", "Solved", "Wins", "Actions", "Eff%", "Explore",
            "AvgProg", "Outcomes", "Banks", "VC", "NN", "GoalTypes"]
    header = f"  {'Run':>4}  {'Solved':>7}  {'Wins':>5}  {'Actions':>8}  {'Eff%':>6}  " \
             f"{'Explr':>6}  {'AvgPrg':>7}  {'Outcm':>6}  {'Banks':>6}  " \
             f"{'VC':>5}  {'NN':>5}  {'Goals':>5}"
    print(header)
    print("  " + "-" * (W - 4))

    for i, m in enumerate(history):
        prev = history[i - 1] if i > 0 else None
        solved = f"{m.get('solved', 0)}/{m.get('total_games', '?')}"
        wins = str(m.get("total_wins", 0))
        actions = f"{m.get('total_actions', 0):,}"
        eff = f"{m.get('effective_pct', 0):.1f}%"
        expl = f"{m.get('avg_exploration', 0):.2f}"
        prog = f"{m.get('avg_best_progress', 0):.3f}"
        outcomes = str(m.get("strategy_outcomes", 0))
        banks = str(m.get("banks_generated", 0))
        vc = str(m.get("vc_steps", 0))
        nn = str(m.get("xg_nn_steps", 0))
        gtypes = str(m.get("xg_goal_types", 0))

        # Highlight improvements
        solved_delta = ""
        if prev and m.get("solved", 0) > prev.get("solved", 0):
            solved_delta = " ▲"
        elif prev and m.get("solved", 0) < prev.get("solved", 0):
            solved_delta = " ▼"

        prog_delta = ""
        if prev and m.get("avg_best_progress", 0) > prev.get("avg_best_progress", 0):
            prog_delta = " ▲"
        elif prev and m.get("avg_best_progress", 0) < prev.get("avg_best_progress", 0):
            prog_delta = " ▼"

        print(f"  {i+1:>4}  {solved:>7}{solved_delta:<2} {wins:>5}  {actions:>8}  {eff:>6}  "
              f"{expl:>6}  {prog:>7}{prog_delta:<2} {outcomes:>6}  {banks:>6}  "
              f"{vc:>5}  {nn:>5}  {gtypes:>5}")

    print("  " + "-" * (W - 4))

    # Summary deltas: first vs last
    if len(history) >= 2:
        first, last = history[0], history[-1]
        print()
        print("  IMPROVEMENT (Run 1 → Run {})".format(len(history)))
        print("  " + "-" * 40)

        def show_delta(label, key, fmt=".0f", pct=False):
            f_val = first.get(key, 0) or 0
            l_val = last.get(key, 0) or 0
            diff = l_val - f_val
            if pct and f_val > 0:
                pct_change = (diff / f_val) * 100
                print(f"    {label:<22} {f_val:{fmt}} → {l_val:{fmt}}  ({diff:+{fmt}}, {pct_change:+.0f}%)")
            else:
                print(f"    {label:<22} {f_val:{fmt}} → {l_val:{fmt}}  ({diff:+{fmt}})")

        show_delta("Games solved", "solved")
        show_delta("Total wins", "total_wins")
        show_delta("Avg best progress", "avg_best_progress", ".3f", pct=True)
        show_delta("Effective %", "effective_pct", ".1f")
        show_delta("Strategy outcomes", "strategy_outcomes")
        show_delta("NN train steps", "xg_nn_steps")
        show_delta("Goal types learned", "xg_goal_types")

    # Games won per run
    any_wins = any(m.get("win_games") for m in history)
    if any_wins:
        print()
        print("  GAMES WON PER RUN:")
        for i, m in enumerate(history):
            wins = m.get("win_games", [])
            if wins:
                print(f"    Run {i+1}: {', '.join(wins)}")
            else:
                print(f"    Run {i+1}: (none)")

    print()
    print("=" * W)


def main():
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    num_games = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    time_budget = int(sys.argv[3]) if len(sys.argv) > 3 else 60

    print(f"\n{'='*80}")
    print(f"  TRAINING LOOP — {iterations} iterations × {num_games} games × {time_budget}s")
    if os.path.exists(MEMORY_PATH):
        print(f"  Resuming from existing cross-game memory: {MEMORY_PATH}")
    else:
        print(f"  Starting fresh (no prior memory)")
    print(f"{'='*80}")

    history = []

    pbar = tqdm(range(1, iterations + 1), desc="Training", unit="run",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} runs [{elapsed}<{remaining}]")

    for i in pbar:
        pbar.set_description(f"Run {i}/{iterations}")

        t0 = time.time()
        result = subprocess.run(
            [PYTHON, SCRIPT, str(num_games), str(time_budget)],
            cwd=str(Path(__file__).parent),
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - t0

        # Print the original output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            # Only print errors, not warnings
            for line in result.stderr.splitlines():
                if "error" in line.lower() or "traceback" in line.lower():
                    print(line)

        # Parse metrics
        metrics = parse_run_output(result.stdout or "")
        metrics["run"] = i
        metrics["elapsed"] = round(elapsed, 1)
        metrics["exit_code"] = result.returncode
        history.append(metrics)

        # Update progress bar postfix with key metrics
        pbar.set_postfix({
            "solved": f"{metrics.get('solved', '?')}/{metrics.get('total_games', '?')}",
            "prog": f"{metrics.get('avg_best_progress', 0):.3f}",
            "wins": metrics.get("total_wins", 0),
        })

        # Print comparative diagnostic after each run
        print_diagnostic(history)

    # Final summary
    print(f"\n{'#'*80}")
    print(f"  TRAINING COMPLETE — {iterations} runs finished")
    print(f"  Total time: {sum(m.get('elapsed', 0) for m in history):.0f}s")
    print(f"  Memory saved to: {MEMORY_PATH}")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()
