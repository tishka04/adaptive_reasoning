"""
Planning verifier — checks hierarchical plans for:
  - subgoal completeness
  - dependency satisfaction (all prerequisites met before execution)
  - no cycles in the plan graph
  - all entities / resources accounted for
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional

from .base import BaseVerifier, VerificationResult


class PlanningVerifier(BaseVerifier):
    """Verifies hierarchical plans and dependency-based solutions."""

    name = "planning"

    def verify(
        self,
        task: Dict[str, Any],
        solution: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        logs: List[str] = []
        violations: List[Dict[str, Any]] = []

        if not isinstance(solution, dict):
            return VerificationResult(
                valid=False,
                violations=[{"name": "invalid_format", "description": "Solution is not a dict", "severity": 1.0}],
                logs=["Solution must be a dict with 'steps' key"],
            )

        steps = solution.get("steps", [])
        if not steps:
            return VerificationResult(
                valid=False,
                violations=[{"name": "empty_plan", "description": "No steps in plan", "severity": 1.0}],
                logs=["Plan has no steps"],
            )

        logs.append(f"Verifying plan with {len(steps)} steps")

        # Check 1: All steps solved
        unsolved = [s for s in steps if not s.get("solved", False)]
        if unsolved:
            for s in unsolved:
                violations.append({
                    "name": f"unsolved_{s.get('subgoal', 'unknown')}",
                    "description": f"Subgoal {s.get('subgoal')} not solved",
                    "severity": 1.0,
                })
            logs.append(f"  {len(unsolved)} unsolved subgoals")

        # Check 2: Dependency ordering
        dep_violations = self._check_dependency_order(steps, task)
        violations.extend(dep_violations)
        if dep_violations:
            logs.append(f"  {len(dep_violations)} dependency violations")

        # Check 3: Cycle detection
        has_cycle = self._detect_cycles(steps, task)
        if has_cycle:
            violations.append({
                "name": "cycle_detected",
                "description": "Circular dependency in plan",
                "severity": 1.0,
            })
            logs.append("  Cycle detected in plan graph")

        # Check 4: Entity coverage
        required_entities = set(task.get("entities", []))
        covered_entities = set()
        for step in steps:
            sol = step.get("solution", {})
            if isinstance(sol, dict):
                entity = sol.get("entity") or sol.get("subgoal", "")
                # Strip prefixes
                for prefix in ("handle_", "satisfy_", "resolve_"):
                    if entity.startswith(prefix):
                        entity = entity[len(prefix):]
                covered_entities.add(entity)

        missing = required_entities - covered_entities
        if missing:
            for e in missing:
                violations.append({
                    "name": f"missing_entity_{e}",
                    "description": f"Entity {e} not addressed in plan",
                    "severity": 0.7,
                })
            logs.append(f"  Missing entities: {missing}")

        # Check task constraints
        constraint_violations = self.check_constraints(task, solution)
        violations.extend(constraint_violations)

        # Compute scores
        tests_total = 4  # completeness, deps, cycles, coverage
        tests_passed = tests_total
        if unsolved:
            tests_passed -= 1
        if dep_violations:
            tests_passed -= 1
        if has_cycle:
            tests_passed -= 1
        if missing:
            tests_passed -= 1

        valid = len([v for v in violations if v.get("severity", 0) >= 1.0]) == 0
        score = len([s for s in steps if s.get("solved")]) / max(len(steps), 1)

        return VerificationResult(
            valid=valid,
            feasible=valid and not has_cycle,
            score=score,
            violations=violations,
            tests_passed=tests_passed,
            tests_total=tests_total,
            objective_value=score,
            logs=logs,
        )

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------
    def _check_dependency_order(
        self, steps: List[Dict], task: Dict
    ) -> List[Dict[str, Any]]:
        """Check that dependencies appear before dependents in the plan."""
        violations = []
        subgoal_data = task.get("structured_data", {}).get("subgoals", [])
        dep_map = {}
        for sg in subgoal_data:
            dep_map[sg.get("name", "")] = sg.get("dependencies", [])

        solved_order = []
        for step in steps:
            name = step.get("subgoal", "")
            deps = dep_map.get(name, [])
            for dep in deps:
                if dep not in solved_order:
                    violations.append({
                        "name": f"dep_order_{name}",
                        "description": f"{name} appears before its dependency {dep}",
                        "severity": 0.8,
                    })
            if step.get("solved"):
                solved_order.append(name)

        return violations

    def _detect_cycles(self, steps: List[Dict], task: Dict) -> bool:
        """Detect cycles in the dependency graph."""
        subgoal_data = task.get("structured_data", {}).get("subgoals", [])
        adj: Dict[str, List[str]] = {}
        all_nodes = set()

        for sg in subgoal_data:
            name = sg.get("name", "")
            deps = sg.get("dependencies", [])
            all_nodes.add(name)
            for d in deps:
                all_nodes.add(d)
                adj.setdefault(d, []).append(name)

        # Kahn's algorithm
        in_degree = {n: 0 for n in all_nodes}
        for n in all_nodes:
            for child in adj.get(n, []):
                in_degree[child] = in_degree.get(child, 0) + 1

        queue = deque([n for n, d in in_degree.items() if d == 0])
        visited = 0
        while queue:
            n = queue.popleft()
            visited += 1
            for child in adj.get(n, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return visited < len(all_nodes)
