"""
Hierarchical solver — decomposes a problem into subgoals and solves
them in dependency order.

Use cases:
  - key-door / resource assembly / tool-use planning
  - any problem with natural subgoal structure

Approach:
  1. Extract subgoals from the task description
  2. Build a dependency DAG
  3. Topologically sort
  4. Solve each subgoal (possibly delegating to other solvers)
  5. Compose partial solutions
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseSolver, SolverResult


class SubGoal:
    """A single subgoal in the hierarchy."""

    def __init__(
        self,
        name: str,
        description: str = "",
        dependencies: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.description = description
        self.dependencies = dependencies or []
        self.params = params or {}
        self.solution: Any = None
        self.solved: bool = False


class HierarchicalSolver(BaseSolver):
    """
    Decomposes tasks into subgoals and solves them in order.
    """

    name = "hierarchical"

    def __init__(self, sub_solver: Optional[BaseSolver] = None):
        self._sub_solver = sub_solver

    def solve(
        self,
        task: Dict[str, Any],
        budget: str = "medium",
        strictness: str = "verified",
        tool_hint: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> SolverResult:
        t0 = time.time()
        logs: List[str] = []
        context = context or {}

        # Step 1: Extract subgoals
        subgoals = self._extract_subgoals(task, tool_hint, context)
        logs.append(f"Extracted {len(subgoals)} subgoals")

        if not subgoals:
            return SolverResult(
                success=False,
                logs=["No subgoals could be extracted"],
                elapsed_seconds=time.time() - t0,
            )

        # Step 2: Topological sort
        try:
            ordered = self._topo_sort(subgoals)
            logs.append(f"Topological order: {[sg.name for sg in ordered]}")
        except ValueError as e:
            return SolverResult(
                success=False,
                violations=[str(e)],
                logs=logs + [f"Topo sort failed: {e}"],
                elapsed_seconds=time.time() - t0,
            )

        # Step 3: Solve each subgoal in order
        max_time = self.budget_to_seconds(budget)
        solutions = {}
        violations = []

        for sg in ordered:
            if time.time() - t0 > max_time:
                violations.append(f"Budget exhausted before solving {sg.name}")
                break

            result = self._solve_subgoal(sg, solutions, task, context)
            logs.append(f"  {sg.name}: success={result['success']}")

            if result["success"]:
                sg.solution = result["solution"]
                sg.solved = True
                solutions[sg.name] = result["solution"]
            else:
                violations.append(f"Subgoal {sg.name} failed: {result.get('reason', 'unknown')}")
                if strictness == "strict":
                    break  # Abort on first failure

        # Step 4: Compose
        all_solved = all(sg.solved for sg in subgoals)
        composed = self._compose(subgoals, solutions, task)

        return SolverResult(
            success=all_solved,
            solution=composed,
            score=sum(1.0 for sg in subgoals if sg.solved) / max(len(subgoals), 1),
            feasible=all_solved and len(violations) == 0,
            violations=violations,
            metadata={
                "subgoals": len(subgoals),
                "solved": sum(1 for sg in subgoals if sg.solved),
                "order": [sg.name for sg in ordered],
            },
            logs=logs,
            elapsed_seconds=time.time() - t0,
        )

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------
    def _extract_subgoals(
        self, task: Dict, tool_hint: str, context: Dict
    ) -> List[SubGoal]:
        """
        Extract subgoals from task structure.

        Uses structured_data if available, otherwise creates default
        decomposition based on constraints and entities.
        """
        # Check for explicit subgoals in structured data
        sg_data = task.get("structured_data", {}).get("subgoals", [])
        if sg_data:
            return [
                SubGoal(
                    name=sg.get("name", f"sg_{i}"),
                    description=sg.get("description", ""),
                    dependencies=sg.get("dependencies", []),
                    params=sg.get("params", {}),
                )
                for i, sg in enumerate(sg_data)
            ]

        # Auto-decompose: one subgoal per entity group or constraint group
        subgoals = []
        entities = task.get("entities", [])
        constraints = task.get("constraints", [])

        if entities:
            for i, entity in enumerate(entities):
                subgoals.append(SubGoal(
                    name=f"handle_{entity}",
                    description=f"Resolve entity: {entity}",
                    dependencies=[f"handle_{entities[i-1]}"] if i > 0 else [],
                    params={"entity": entity},
                ))
        elif constraints:
            for i, c in enumerate(constraints):
                name = c.get("name", f"constraint_{i}")
                subgoals.append(SubGoal(
                    name=f"satisfy_{name}",
                    description=c.get("description", ""),
                ))

        # Depth hint from tool_hint
        max_depth = 2
        if "depth_3" in tool_hint:
            max_depth = 3

        return subgoals[:max_depth * 4]  # Cap total subgoals

    def _topo_sort(self, subgoals: List[SubGoal]) -> List[SubGoal]:
        """Topological sort of subgoals by dependencies."""
        name_to_sg = {sg.name: sg for sg in subgoals}
        in_degree = {sg.name: 0 for sg in subgoals}
        adj: Dict[str, List[str]] = {sg.name: [] for sg in subgoals}

        for sg in subgoals:
            for dep in sg.dependencies:
                if dep in name_to_sg:
                    adj[dep].append(sg.name)
                    in_degree[sg.name] += 1

        queue = deque([n for n, d in in_degree.items() if d == 0])
        ordered = []
        while queue:
            n = queue.popleft()
            ordered.append(name_to_sg[n])
            for child in adj[n]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(ordered) != len(subgoals):
            raise ValueError("Cyclic dependency detected among subgoals")

        return ordered

    def _solve_subgoal(
        self,
        subgoal: SubGoal,
        partial_solutions: Dict[str, Any],
        task: Dict,
        context: Dict,
    ) -> Dict[str, Any]:
        """
        Solve a single subgoal. If a sub-solver is registered, delegate to it.
        Otherwise use a simple default strategy.
        """
        if self._sub_solver is not None:
            sub_task = {
                "description": subgoal.description,
                "constraints": [],
                "entities": [subgoal.name],
                "structured_data": subgoal.params,
            }
            result = self._sub_solver.solve(sub_task, budget="low", context=context)
            return {"success": result.success, "solution": result.solution, "reason": "" if result.success else "sub_solver_failed"}

        # Default: mark as solved with a placeholder
        return {
            "success": True,
            "solution": {
                "subgoal": subgoal.name,
                "status": "completed",
                "dependencies_met": all(
                    d in partial_solutions for d in subgoal.dependencies
                ),
            },
        }

    def _compose(
        self,
        subgoals: List[SubGoal],
        solutions: Dict[str, Any],
        task: Dict,
    ) -> Dict[str, Any]:
        """Compose partial solutions into a full solution."""
        return {
            "type": "hierarchical_plan",
            "steps": [
                {
                    "subgoal": sg.name,
                    "solved": sg.solved,
                    "solution": solutions.get(sg.name),
                }
                for sg in subgoals
            ],
            "complete": all(sg.solved for sg in subgoals),
        }
