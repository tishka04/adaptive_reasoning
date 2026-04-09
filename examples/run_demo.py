"""
Standalone demo — runs the full reasoning loop on all three domains
without requiring a GPU or LLM. Just needs: torch, pydantic, ortools.

Usage:
    python examples/run_demo.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from v4_1_reasoning_system.orchestration.controller import ReasoningController


def main():
    print("=" * 70)
    print("  v4.1 Adaptive Reasoning System -- Standalone Demo")
    print("=" * 70)

    controller = ReasoningController(
        use_learned_router=False,
        max_iterations=5,
        max_time_seconds=60.0,
        device="cpu",
        latent_dim=64,
        action_dim=16,
    )

    # ── Problem 1: Planning (Key-Door-Chest) ──────────────────────────
    print("\n" + "-" * 70)
    print("  Problem 1: Planning -- Key-Door-Chest dependency chain")
    print("-" * 70)

    r1 = controller.solve_from_dict(
        problem="Get key, open door, access chest — in dependency order.",
        parsed_json={
            "domain": "planning",
            "description": "Key-door-chest dependency chain",
            "entities": ["key", "door", "chest"],
            "constraints": [
                {"name": "deps", "description": "Respect dependencies", "ctype": "hard", "params": {}},
            ],
            "objective": {"description": "Complete all steps", "sense": "satisfy"},
            "ambiguities": [],
            "structured_data": {
                "subgoals": [
                    {"name": "handle_key", "description": "Get the key", "dependencies": [], "params": {"entity": "key"}},
                    {"name": "handle_door", "description": "Open the door", "dependencies": ["handle_key"], "params": {"entity": "door"}},
                    {"name": "handle_chest", "description": "Access the chest", "dependencies": ["handle_door"], "params": {"entity": "chest"}},
                ]
            },
        },
    )
    _print_result(r1)

    # ── Problem 2: Scheduling (Job-Shop) ──────────────────────────────
    print("\n" + "-" * 70)
    print("  Problem 2: Scheduling -- 5 jobs with precedence constraints")
    print("-" * 70)

    r2 = controller.solve_from_dict(
        problem="Schedule 5 jobs to minimize makespan with dependencies.",
        parsed_json={
            "domain": "scheduling",
            "description": "Minimize makespan for 5 jobs",
            "entities": ["job_0", "job_1", "job_2", "job_3", "job_4"],
            "constraints": [
                {"name": "precedence", "description": "Respect job dependencies", "ctype": "hard", "params": {}},
            ],
            "objective": {"description": "Minimize makespan", "sense": "minimize", "metric": "makespan"},
            "ambiguities": [],
            "structured_data": {
                "type": "scheduling",
                "jobs": [
                    {"name": "job_0", "duration": 3, "dependencies": []},
                    {"name": "job_1", "duration": 5, "dependencies": ["job_0"]},
                    {"name": "job_2", "duration": 2, "dependencies": []},
                    {"name": "job_3", "duration": 4, "dependencies": ["job_1", "job_2"]},
                    {"name": "job_4", "duration": 3, "dependencies": ["job_3"]},
                ],
                "horizon": 30,
                "resources": {},
            },
        },
    )
    _print_result(r2)

    # ── Problem 3: Optimization (Knapsack) ────────────────────────────
    print("\n" + "-" * 70)
    print("  Problem 3: Optimization -- Knapsack with capacity 20")
    print("-" * 70)

    r3 = controller.solve_from_dict(
        problem="Select items to maximize value within weight capacity 20.",
        parsed_json={
            "domain": "optimization",
            "description": "Knapsack: maximize value, capacity 20",
            "entities": ["item_0", "item_1", "item_2", "item_3", "item_4", "item_5"],
            "constraints": [
                {"name": "capacity", "description": "Total weight <= 20", "ctype": "hard",
                 "params": {"type": "sum_leq", "bound": 20}},
            ],
            "objective": {"description": "Maximize total value", "sense": "maximize", "metric": "value"},
            "ambiguities": [],
            "structured_data": {
                "type": "knapsack",
                "items": [
                    {"name": "item_0", "weight": 5, "value": 10},
                    {"name": "item_1", "weight": 8, "value": 15},
                    {"name": "item_2", "weight": 3, "value": 7},
                    {"name": "item_3", "weight": 6, "value": 12},
                    {"name": "item_4", "weight": 7, "value": 14},
                    {"name": "item_5", "weight": 4, "value": 9},
                ],
                "capacity": 20,
            },
        },
    )
    _print_result(r3)

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    stats = controller.replay_buffer.statistics()
    print(f"  Trajectories stored : {stats['count']}")
    print(f"  Success rate        : {stats.get('success_rate', 0):.0%}")
    print(f"  Avg steps           : {stats.get('avg_steps', 0):.1f}")
    print(f"  Mode usage          : {stats.get('mode_counts', {})}")
    print(f"  Domains covered     : {list(stats.get('domains', {}).keys())}")
    print()


def _print_result(result: dict) -> None:
    print(f"  Valid      : {result['valid']}")
    print(f"  Feasible   : {result['feasible']}")
    print(f"  Score      : {result['score']:.3f}")
    print(f"  Iterations : {result['iterations']}")
    print(f"  Elapsed    : {result['elapsed_seconds']:.2f}s")
    print(f"  Domain     : {result['domain']}")
    sol = result["solution"]
    if isinstance(sol, dict):
        print(f"  Solution   : {json.dumps(sol, indent=4, default=str)[:500]}")
    else:
        print(f"  Solution   : {sol}")


if __name__ == "__main__":
    main()
