"""Inspect per-step trajectories from an evaluate_generalization report.

Prints the step table (action / is_radar / break_probability / topology) and
detects four failure patterns:

  P1. break_probability rises but largest_component_size never moves
      -> radar sees precursors the search never converts.
  P2. changed_cells > 0 but component_count stays constant
      -> the agent moves things without deforming the topology.
  P3. radar always injects the same action(s)
      -> curiosity head collapsed onto a single branch.
  P4. no state ever approaches break_probability > 0.3
      -> the beam never reaches break-informative regions (precursor-seeking
         problem, upstream of the classifier).

Usage:
    ARC-AGI-3-Agents\\.venv\\Scripts\\python.exe inspect_trajectory.py ^
        --report diagnostics\\evaluate_generalization\\report.json ^
        --game ar25-e3c63847
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

BP_HIGH = 0.3
BP_RISE = 0.10


def _load_trajectories(report: Dict[str, Any], game: str) -> Dict[str, List[Dict[str, Any]]]:
    """Return {weight: trajectory} for the requested game (prefix match allowed)."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for weight, payload in report.get("per_weight", {}).items():
        for res in payload.get("per_game", []):
            gid = str(res.get("game_id", ""))
            if gid == game or gid.startswith(game) or game.startswith(gid.split("-")[0]):
                traj = res.get("trajectory") or []
                if traj:
                    out[weight] = traj
    return out


def _print_table(traj: List[Dict[str, Any]], max_rows: int) -> None:
    header = (
        f"{'step':>4} {'action':<8} {'radar':<5} {'esc':<3} {'bp':>7} {'danger':>7} "
        f"{'lc_size':>8} {'n_comp':>6} {'chg':>5} {'lvl':>3}"
    )
    print(header)
    print("-" * len(header))
    rows = traj if len(traj) <= max_rows else traj[: max_rows // 2] + [None] + traj[-max_rows // 2 :]
    for s in rows:
        if s is None:
            print(f"{'...':>4}")
            continue
        print(
            f"{s['step']:>4} {s['action']:<8} {('YES' if s.get('is_radar') else '.'):<5} "
            f"{('E' if s.get('is_escape') else '.'):<3} "
            f"{s['break_probability']:>7.4f} {s.get('predicted_danger', 0.0):>7.4f} "
            f"{s['largest_component_size']:>8.1f} {s['component_count']:>6.1f} "
            f"{s['changed_cells']:>5} {s['level']:>3}"
        )


def _detect_patterns(traj: List[Dict[str, Any]]) -> List[str]:
    findings: List[str] = []
    bps = np.array([s["break_probability"] for s in traj], dtype=float)
    lcs = np.array([s["largest_component_size"] for s in traj], dtype=float)
    ncs = np.array([s["component_count"] for s in traj], dtype=float)
    chg = np.array([s["changed_cells"] for s in traj], dtype=float)
    radar = [s for s in traj if s.get("is_radar")]

    # P1: bp rises but lc_size flat
    bp_range = float(bps.max() - bps.min()) if len(bps) else 0.0
    lc_range = float(lcs.max() - lcs.min()) if len(lcs) else 0.0
    if bp_range >= BP_RISE and lc_range < 1.0:
        findings.append(
            f"P1 CONFIRMED: bp varies by {bp_range:.3f} (max {bps.max():.3f}) but "
            f"largest_component_size is flat ({lcs.min():.1f}->{lcs.max():.1f}). "
            "Radar sees precursors the search never converts."
        )

    # P2: cells change but component_count stable
    moving = chg > 0
    if moving.sum() >= 5 and len(set(ncs[moving].tolist())) <= 1:
        findings.append(
            f"P2 CONFIRMED: {int(moving.sum())} steps with changed_cells>0 but "
            f"component_count constant at {ncs[moving][0]:.0f}. "
            "Movement without topological deformation."
        )

    # P3: radar action collapse
    if radar:
        actions = [s["action"] for s in radar]
        counts = {a: actions.count(a) for a in set(actions)}
        top_action, top_n = max(counts.items(), key=lambda kv: kv[1])
        if top_n / len(actions) >= 0.8 and len(actions) >= 3:
            findings.append(
                f"P3 CONFIRMED: radar injected {top_action} on {top_n}/{len(actions)} "
                f"overrides ({100.0 * top_n / len(actions):.0f}%). Curiosity head collapsed."
            )
        else:
            findings.append(f"P3 ok: radar action distribution {counts}")
    else:
        findings.append("P3 n/a: no radar injections in this trajectory.")

    # P5: dead loop — long terminal run of the same action with changed_cells == 0
    run_action: Optional[str] = None
    run_len = 0
    best_run = ("", 0)
    for s in traj:
        if s["changed_cells"] == 0 and s["action"] == run_action:
            run_len += 1
        elif s["changed_cells"] == 0:
            run_action, run_len = s["action"], 1
        else:
            run_action, run_len = None, 0
        if run_len > best_run[1]:
            best_run = (run_action or "", run_len)
    if best_run[1] >= 20:
        findings.append(
            f"P5 CONFIRMED: dead loop — {best_run[0]} repeated {best_run[1]}x with "
            "changed_cells=0. Degenerate attractor; anti-attractor escape should fire here."
        )

    # P4: bp never approaches the informative regime
    max_bp = float(bps.max()) if len(bps) else 0.0
    if max_bp < BP_HIGH:
        findings.append(
            f"P4 CONFIRMED: max break_probability = {max_bp:.4f} < {BP_HIGH}. "
            "The beam never reaches break-informative regions -> precursor-seeking "
            "problem, upstream of the classifier."
        )
    else:
        over = int((bps > BP_HIGH).sum())
        findings.append(f"P4 ok: max bp={max_bp:.4f}, {over} steps above {BP_HIGH}.")

    return findings


def _radar_vs_normal(traj: List[Dict[str, Any]]) -> None:
    radar = [s for s in traj if s.get("is_radar")]
    normal = [s for s in traj if not s.get("is_radar")]
    if not radar or not normal:
        return
    def m(rows, key):
        return float(np.mean([r[key] for r in rows]))
    print("\n--- Radar vs normal steps ---")
    print(f"{'metric':<24} {'radar':>10} {'normal':>10}")
    for key in ("break_probability", "changed_cells", "largest_component_size", "component_count"):
        print(f"{key:<24} {m(radar, key):>10.4f} {m(normal, key):>10.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a guided-search trajectory from a report.")
    parser.add_argument("--report", default="diagnostics/evaluate_generalization/report.json")
    parser.add_argument("--game", required=True, help="Game id or prefix (e.g. ar25-e3c63847 or ar25).")
    parser.add_argument("--weight", default=None, help="Only inspect this learned weight (e.g. 0.1).")
    parser.add_argument("--max-rows", type=int, default=80, help="Max table rows printed (head+tail).")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    trajs = _load_trajectories(report, args.game)
    if not trajs:
        print(f"No trajectory found for game '{args.game}' in {args.report}.")
        print("Available games:", report.get("games"))
        return

    for weight, traj in sorted(trajs.items(), key=lambda kv: float(kv[0])):
        if args.weight is not None and float(weight) != float(args.weight):
            continue
        print(f"\n================ game={args.game} weight={weight} steps={len(traj)} ================")
        _print_table(traj, args.max_rows)
        _radar_vs_normal(traj)
        print("\n--- Pattern detection ---")
        for finding in _detect_patterns(traj):
            print(" *", finding)


if __name__ == "__main__":
    main()
