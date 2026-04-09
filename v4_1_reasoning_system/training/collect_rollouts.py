"""
Rollout collector — runs the reasoning controller on a set of problems
and collects trajectories for training the world model and router.

Supports:
  - Batch problem collection
  - Synthetic problem generation for each domain
  - Trajectory storage in the replay buffer
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.progress import Progress

from ..memory.replay import ReplayBuffer
from ..orchestration.controller import ReasoningController
from ..parser.task_schema import (
    Constraint,
    ConstraintType,
    DomainGuess,
    Objective,
    ObjectiveSense,
    TaskObject,
)

console = Console()


class RolloutCollector:
    """
    Collects reasoning rollouts by running the controller on problems.
    """

    def __init__(self, controller: ReasoningController):
        self.controller = controller

    def collect(
        self,
        problems: List[Dict[str, Any]],
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Run the controller on each problem and collect trajectories.

        Each problem dict should have:
          - "text": natural language problem
          - "parsed": optional pre-parsed JSON (to skip LLM parsing)

        Returns list of result dicts from controller.solve().
        """
        results = []
        with Progress(disable=not verbose) as progress:
            task_bar = progress.add_task("Collecting rollouts", total=len(problems))
            for i, prob in enumerate(problems):
                text = prob.get("text", "")
                parsed = prob.get("parsed")

                if parsed:
                    result = self.controller.solve_from_dict(text, parsed)
                else:
                    result = self.controller.solve(text)

                results.append(result)
                if verbose:
                    console.print(
                        f"  [{i+1}/{len(problems)}] "
                        f"score={result['score']:.3f} "
                        f"valid={result['valid']} "
                        f"iters={result['iterations']} "
                        f"elapsed={result['elapsed_seconds']:.1f}s"
                    )
                progress.advance(task_bar)

        return results

    # ------------------------------------------------------------------
    # Synthetic problem generators for each domain
    # ------------------------------------------------------------------
    @staticmethod
    def generate_planning_problems(n: int = 10) -> List[Dict[str, Any]]:
        """Generate synthetic planning / key-door problems."""
        import random
        problems = []
        for i in range(n):
            num_entities = random.randint(3, 8)
            entities = [f"item_{j}" for j in range(num_entities)]
            # Random dependency chain
            subgoals = []
            for j, entity in enumerate(entities):
                deps = [f"handle_{entities[j-1]}"] if j > 0 and random.random() > 0.3 else []
                subgoals.append({
                    "name": f"handle_{entity}",
                    "description": f"Process {entity}",
                    "dependencies": deps,
                    "params": {"entity": entity},
                })

            parsed = {
                "domain": "planning",
                "description": f"Plan to process {num_entities} items in dependency order",
                "entities": entities,
                "constraints": [
                    {"name": "all_processed", "description": "All items must be processed", "ctype": "hard", "params": {}},
                    {"name": "dep_order", "description": "Dependencies must be respected", "ctype": "hard", "params": {}},
                ],
                "objective": {"description": "Complete all items", "sense": "satisfy", "metric": "completeness", "params": {}},
                "ambiguities": [],
                "structured_data": {"subgoals": subgoals},
            }
            problems.append({
                "text": f"Process items {', '.join(entities)} respecting their dependencies.",
                "parsed": parsed,
            })
        return problems

    @staticmethod
    def generate_scheduling_problems(n: int = 10) -> List[Dict[str, Any]]:
        """Generate synthetic scheduling / job-shop problems."""
        import random
        problems = []
        for i in range(n):
            num_jobs = random.randint(3, 10)
            jobs = []
            for j in range(num_jobs):
                dur = random.randint(1, 10)
                deps = []
                if j > 0 and random.random() > 0.4:
                    dep_idx = random.randint(0, j - 1)
                    deps.append(f"job_{dep_idx}")
                jobs.append({
                    "name": f"job_{j}",
                    "duration": dur,
                    "dependencies": deps,
                })

            horizon = sum(j["duration"] for j in jobs) + 10
            parsed = {
                "domain": "scheduling",
                "description": f"Schedule {num_jobs} jobs to minimize makespan",
                "entities": [j["name"] for j in jobs],
                "constraints": [
                    {"name": "precedence", "description": "Job dependencies must be respected", "ctype": "hard", "params": {}},
                ],
                "objective": {"description": "Minimize makespan", "sense": "minimize", "metric": "makespan", "params": {}},
                "ambiguities": [],
                "structured_data": {
                    "type": "scheduling",
                    "jobs": jobs,
                    "horizon": horizon,
                    "resources": {},
                },
            }
            problems.append({
                "text": f"Schedule {num_jobs} jobs with precedence constraints to minimize makespan.",
                "parsed": parsed,
            })
        return problems

    @staticmethod
    def generate_optimization_problems(n: int = 10) -> List[Dict[str, Any]]:
        """Generate synthetic optimization / resource allocation problems."""
        import random
        problems = []
        for i in range(n):
            num_items = random.randint(5, 15)
            capacity = random.randint(20, 50)
            items = [
                {"name": f"item_{j}", "weight": random.randint(1, 15), "value": random.randint(1, 20)}
                for j in range(num_items)
            ]

            parsed = {
                "domain": "optimization",
                "description": f"Select items to maximize value within capacity {capacity}",
                "entities": [it["name"] for it in items],
                "constraints": [
                    {"name": "capacity", "description": f"Total weight <= {capacity}", "ctype": "hard",
                     "params": {"type": "sum_leq", "bound": capacity}},
                ],
                "objective": {"description": "Maximize total value", "sense": "maximize", "metric": "value", "params": {}},
                "ambiguities": [],
                "structured_data": {
                    "type": "knapsack",
                    "items": items,
                    "capacity": capacity,
                },
            }
            problems.append({
                "text": f"Select from {num_items} items to maximize value with capacity {capacity}.",
                "parsed": parsed,
            })
        return problems

    @staticmethod
    def generate_coding_problems(n: int = 5) -> List[Dict[str, Any]]:
        """Generate simple coding task problems."""
        templates = [
            {
                "text": "Write a Python function that computes the nth Fibonacci number.",
                "parsed": {
                    "domain": "coding",
                    "description": "Compute nth Fibonacci number",
                    "entities": ["fibonacci"],
                    "constraints": [],
                    "objective": {"description": "Correct computation", "sense": "satisfy", "metric": "correctness", "params": {}},
                    "ambiguities": [],
                    "structured_data": {
                        "test_cases": [
                            {"input": "0", "expected_output": "0"},
                            {"input": "1", "expected_output": "1"},
                            {"input": "10", "expected_output": "55"},
                        ],
                    },
                },
            },
            {
                "text": "Write a function to check if a string is a palindrome.",
                "parsed": {
                    "domain": "coding",
                    "description": "Check if string is palindrome",
                    "entities": ["palindrome_checker"],
                    "constraints": [],
                    "objective": {"description": "Correct result", "sense": "satisfy", "metric": "correctness", "params": {}},
                    "ambiguities": [],
                    "structured_data": {
                        "test_cases": [
                            {"input": "racecar", "expected_output": "True"},
                            {"input": "hello", "expected_output": "False"},
                        ],
                    },
                },
            },
            {
                "text": "Write a function to find the maximum subarray sum (Kadane's algorithm).",
                "parsed": {
                    "domain": "coding",
                    "description": "Maximum subarray sum",
                    "entities": ["max_subarray"],
                    "constraints": [],
                    "objective": {"description": "Correct computation", "sense": "satisfy", "metric": "correctness", "params": {}},
                    "ambiguities": [],
                    "structured_data": {
                        "test_cases": [
                            {"input": "[-2,1,-3,4,-1,2,1,-5,4]", "expected_output": "6"},
                        ],
                    },
                },
            },
            {
                "text": "Write code to sort a list of integers using merge sort.",
                "parsed": {
                    "domain": "coding",
                    "description": "Merge sort implementation",
                    "entities": ["merge_sort"],
                    "constraints": [],
                    "objective": {"description": "Correct sorting", "sense": "satisfy", "metric": "correctness", "params": {}},
                    "ambiguities": [],
                    "structured_data": {},
                },
            },
            {
                "text": "Write a function that solves the two-sum problem: given a list and target, find two indices.",
                "parsed": {
                    "domain": "coding",
                    "description": "Two-sum problem",
                    "entities": ["two_sum"],
                    "constraints": [],
                    "objective": {"description": "Return correct indices", "sense": "satisfy", "metric": "correctness", "params": {}},
                    "ambiguities": [],
                    "structured_data": {},
                },
            },
        ]
        return templates[:n]

    @classmethod
    def generate_benchmark_suite(
        cls,
        n_planning: int = 10,
        n_scheduling: int = 10,
        n_optimization: int = 10,
        n_coding: int = 5,
    ) -> List[Dict[str, Any]]:
        """Generate the full benchmark suite across all domains."""
        problems = []
        problems.extend(cls.generate_planning_problems(n_planning))
        problems.extend(cls.generate_scheduling_problems(n_scheduling))
        problems.extend(cls.generate_optimization_problems(n_optimization))
        problems.extend(cls.generate_coding_problems(n_coding))
        return problems
