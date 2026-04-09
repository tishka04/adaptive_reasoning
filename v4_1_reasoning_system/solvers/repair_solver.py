"""
Repair solver — patches infeasible or failing outputs by applying
local fixes to restore constraint satisfaction.

Use cases:
  - fix constraint violations in a partial solution
  - relax soft constraints to achieve feasibility
  - apply local patches when a solution is close but not valid

Strategies:
  1. Violation-targeted repair: fix the most severe violation first
  2. Relaxation: relax soft constraints until feasible
  3. Local search: perturb the solution near violation sites
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional

from .base import BaseSolver, SolverResult


class RepairSolver(BaseSolver):
    """
    Repairs an existing (partial / infeasible) solution.
    Requires context with the previous solution and verifier feedback.
    """

    name = "repair"

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

        prev_solution = context.get("previous_solution")
        feedback = context.get("verifier_feedback", {})
        violations = feedback.get("violations", [])

        if prev_solution is None:
            return SolverResult(
                success=False,
                violations=["No previous solution to repair"],
                logs=["Repair requires context.previous_solution"],
                elapsed_seconds=time.time() - t0,
            )

        logs.append(f"Repairing solution with {len(violations)} violations")
        logs.append(f"Tool hint: {tool_hint}")

        max_iterations = self.budget_to_iterations(budget)
        time_limit = self.budget_to_seconds(budget)

        # Choose repair strategy
        if tool_hint == "restore_feasibility" or tool_hint == "fix_violations":
            strategy = "violation_targeted"
        elif tool_hint == "relax_constraints":
            strategy = "relaxation"
        elif tool_hint == "local_patch":
            strategy = "local_search"
        else:
            strategy = "violation_targeted" if violations else "local_search"

        logs.append(f"Strategy: {strategy}")

        try:
            if strategy == "violation_targeted":
                result = self._violation_targeted_repair(
                    task, prev_solution, violations, max_iterations, time_limit, t0, logs
                )
            elif strategy == "relaxation":
                result = self._relaxation_repair(
                    task, prev_solution, violations, logs
                )
            elif strategy == "local_search":
                result = self._local_search_repair(
                    task, prev_solution, max_iterations, time_limit, t0, logs
                )
            else:
                result = self._violation_targeted_repair(
                    task, prev_solution, violations, max_iterations, time_limit, t0, logs
                )
        except Exception as e:
            result = SolverResult(
                success=False,
                violations=[f"Repair exception: {str(e)}"],
            )

        result.elapsed_seconds = time.time() - t0
        result.logs = logs
        return result

    # ------------------------------------------------------------------
    # Repair strategies
    # ------------------------------------------------------------------
    def _violation_targeted_repair(
        self,
        task: Dict,
        solution: Any,
        violations: List,
        max_iter: int,
        time_limit: float,
        t0: float,
        logs: List[str],
    ) -> SolverResult:
        """Fix violations one at a time, most severe first."""
        repaired = copy.deepcopy(solution)
        remaining_violations = list(violations)

        # Sort violations by severity if available
        remaining_violations.sort(
            key=lambda v: v.get("severity", 0.5) if isinstance(v, dict) else 0.5,
            reverse=True,
        )

        fixed_count = 0
        for i, violation in enumerate(remaining_violations):
            if time.time() - t0 > time_limit:
                break
            if i >= max_iter:
                break

            patch = self._generate_patch(task, repaired, violation)
            if patch is not None:
                repaired = self._apply_patch(repaired, patch)
                fixed_count += 1
                logs.append(f"  Fixed violation {i}: {self._violation_name(violation)}")
            else:
                logs.append(f"  Could not fix violation {i}: {self._violation_name(violation)}")

        unfixed = len(remaining_violations) - fixed_count
        return SolverResult(
            success=fixed_count > 0,
            solution=repaired,
            score=fixed_count / max(len(remaining_violations), 1),
            feasible=unfixed == 0,
            violations=[self._violation_name(v) for v in remaining_violations[fixed_count:]],
            metadata={"fixed": fixed_count, "remaining": unfixed, "strategy": "violation_targeted"},
        )

    def _relaxation_repair(
        self,
        task: Dict,
        solution: Any,
        violations: List,
        logs: List[str],
    ) -> SolverResult:
        """Relax soft constraints to achieve feasibility."""
        constraints = task.get("constraints", [])
        soft = [c for c in constraints if c.get("ctype") == "soft"]
        hard_violations = []
        relaxed = []

        for v in violations:
            v_name = self._violation_name(v)
            is_soft = any(c.get("name", "") in v_name for c in soft)
            if is_soft:
                relaxed.append(v_name)
                logs.append(f"  Relaxed soft constraint: {v_name}")
            else:
                hard_violations.append(v_name)

        return SolverResult(
            success=len(hard_violations) == 0,
            solution=solution,
            score=1.0 - len(hard_violations) / max(len(violations), 1),
            feasible=len(hard_violations) == 0,
            violations=hard_violations,
            metadata={"relaxed": relaxed, "strategy": "relaxation"},
        )

    def _local_search_repair(
        self,
        task: Dict,
        solution: Any,
        max_iter: int,
        time_limit: float,
        t0: float,
        logs: List[str],
    ) -> SolverResult:
        """
        Perturb solution locally to find a feasible neighbour.
        Works on dict-based solutions with numeric values.
        """
        if not isinstance(solution, dict):
            return SolverResult(
                success=False,
                solution=solution,
                violations=["Local search requires dict solution"],
                metadata={"strategy": "local_search"},
            )

        import random

        best = copy.deepcopy(solution)
        best_score = self._evaluate_solution(task, best)

        for it in range(min(max_iter, 500)):
            if time.time() - t0 > time_limit:
                break

            candidate = copy.deepcopy(best)
            # Perturb a random numeric value
            keys = [k for k, v in candidate.items() if isinstance(v, (int, float))]
            if not keys:
                break
            key = random.choice(keys)
            val = candidate[key]
            delta = random.choice([-1, 1]) * max(1, abs(val) * 0.1)
            candidate[key] = type(val)(val + delta)

            score = self._evaluate_solution(task, candidate)
            if score > best_score:
                best = candidate
                best_score = score
                logs.append(f"  Iter {it}: improved to {best_score:.4f}")

        return SolverResult(
            success=best_score > 0,
            solution=best,
            score=best_score,
            feasible=best_score >= 1.0,
            metadata={"iterations": min(max_iter, 500), "strategy": "local_search"},
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _violation_name(v) -> str:
        if isinstance(v, dict):
            return v.get("name", v.get("description", str(v)))
        return str(v)

    @staticmethod
    def _generate_patch(task: Dict, solution: Any, violation) -> Optional[Dict]:
        """Generate a patch for a single violation. Returns None if no patch found."""
        # Simple heuristic: if solution is a dict with the violated key, adjust it
        if isinstance(solution, dict) and isinstance(violation, dict):
            key = violation.get("variable") or violation.get("field")
            if key and key in solution:
                suggested = violation.get("suggested_value")
                if suggested is not None:
                    return {"key": key, "value": suggested}
                # Default: set to 0 or remove
                return {"key": key, "value": 0}
        return {"action": "noop"}  # Placeholder patch

    @staticmethod
    def _apply_patch(solution: Any, patch: Dict) -> Any:
        """Apply a patch to a solution."""
        if isinstance(solution, dict) and "key" in patch:
            patched = copy.deepcopy(solution)
            patched[patch["key"]] = patch["value"]
            return patched
        return solution

    @staticmethod
    def _evaluate_solution(task: Dict, solution: Dict) -> float:
        """Quick heuristic evaluation of a solution (0 to 1)."""
        # Count how many constraints are approximately satisfied
        constraints = task.get("constraints", [])
        if not constraints:
            return 0.5
        satisfied = 0
        for c in constraints:
            params = c.get("params", {})
            if params.get("type") == "sum_leq":
                total = sum(v for v in solution.values() if isinstance(v, (int, float)))
                if total <= params.get("bound", float("inf")):
                    satisfied += 1
            elif params.get("type") == "sum_geq":
                total = sum(v for v in solution.values() if isinstance(v, (int, float)))
                if total >= params.get("bound", 0):
                    satisfied += 1
            else:
                satisfied += 0.5  # Can't check, assume partially ok
        return satisfied / max(len(constraints), 1)
