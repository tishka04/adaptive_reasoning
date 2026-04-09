"""
Dispatch / optimization verifier — checks optimization solutions for:
  - constraint satisfaction (capacity, bounds, budget)
  - objective value computation and comparison
  - feasibility of the solution point
  - variable bound compliance
  - solution completeness
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseVerifier, VerificationResult


class DispatchVerifier(BaseVerifier):
    """
    Verifies optimization / resource-dispatch / allocation solutions.
    Works with dict-based solutions where keys are variable names
    and values are assignments.
    """

    name = "dispatch"

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
                violations=[{"name": "invalid_format", "description": "Solution must be a dict", "severity": 1.0}],
            )

        structured = task.get("structured_data", {})
        constraints = task.get("constraints", [])
        objective = task.get("objective", {})

        logs.append(f"Verifying optimization solution: {len(solution)} variables")

        # Check 1: Variable bounds
        bounds = structured.get("bounds", {})
        for var, val in solution.items():
            if not isinstance(val, (int, float)):
                continue
            if var in bounds:
                lb = bounds[var].get("lb", float("-inf"))
                ub = bounds[var].get("ub", float("inf"))
                if val < lb:
                    violations.append({
                        "name": f"lb_violation_{var}",
                        "description": f"{var}={val} < lower bound {lb}",
                        "severity": 1.0,
                        "variable": var,
                        "suggested_value": lb,
                    })
                if val > ub:
                    violations.append({
                        "name": f"ub_violation_{var}",
                        "description": f"{var}={val} > upper bound {ub}",
                        "severity": 1.0,
                        "variable": var,
                        "suggested_value": ub,
                    })

        # Check 2: Explicit constraints
        for c in constraints:
            params = c.get("params", {})
            ctype = c.get("ctype", "hard")
            severity = 1.0 if ctype == "hard" else 0.5

            violated = False
            desc = ""

            if params.get("type") == "sum_leq":
                bound = params.get("bound", float("inf"))
                total = sum(v for v in solution.values() if isinstance(v, (int, float)))
                if total > bound:
                    violated = True
                    desc = f"Sum {total} > bound {bound}"

            elif params.get("type") == "sum_geq":
                bound = params.get("bound", 0)
                total = sum(v for v in solution.values() if isinstance(v, (int, float)))
                if total < bound:
                    violated = True
                    desc = f"Sum {total} < bound {bound}"

            elif params.get("type") == "capacity":
                resource = params.get("resource", "")
                cap = params.get("capacity", float("inf"))
                usage = sum(
                    v for k, v in solution.items()
                    if isinstance(v, (int, float)) and resource in k
                )
                if usage > cap:
                    violated = True
                    desc = f"Resource {resource} usage {usage} > capacity {cap}"

            elif params.get("type") == "equality":
                var = params.get("variable", "")
                expected = params.get("value", None)
                if var in solution and expected is not None and solution[var] != expected:
                    violated = True
                    desc = f"{var}={solution[var]} != expected {expected}"

            elif params.get("type") == "all_different":
                vals = [v for v in solution.values() if isinstance(v, (int, float))]
                if len(vals) != len(set(vals)):
                    violated = True
                    desc = "Not all values are different"

            if violated:
                violations.append({
                    "name": c.get("name", "unnamed"),
                    "description": desc or c.get("description", ""),
                    "severity": severity,
                })

        # Check 3: Completeness
        required_vars = structured.get("required_variables", [])
        for rv in required_vars:
            if rv not in solution:
                violations.append({
                    "name": f"missing_var_{rv}",
                    "description": f"Required variable {rv} not in solution",
                    "severity": 0.8,
                })

        # Check 4: Objective value
        obj_value = self._compute_objective(solution, objective, structured)
        logs.append(f"  Objective value: {obj_value}")

        # Check 5: Non-negativity (if specified)
        if structured.get("non_negative", False):
            for var, val in solution.items():
                if isinstance(val, (int, float)) and val < 0:
                    violations.append({
                        "name": f"negative_{var}",
                        "description": f"{var}={val} < 0 (non-negativity required)",
                        "severity": 1.0,
                    })

        # Summary
        hard_violations = [v for v in violations if v.get("severity", 0) >= 1.0]
        valid = len(hard_violations) == 0

        tests_total = 5
        tests_passed = tests_total
        if any("lb_violation" in v["name"] or "ub_violation" in v["name"] for v in violations):
            tests_passed -= 1
        if any(v["name"] in [c.get("name", "") for c in constraints] for v in violations):
            tests_passed -= 1
        if any("missing_var" in v["name"] for v in violations):
            tests_passed -= 1
        if any("negative" in v["name"] for v in violations):
            tests_passed -= 1

        return VerificationResult(
            valid=valid,
            feasible=valid,
            score=1.0 - len(hard_violations) / max(len(hard_violations) + 3, 1),
            violations=violations,
            tests_passed=tests_passed,
            tests_total=tests_total,
            objective_value=obj_value,
            metadata={"num_variables": len(solution)},
            logs=logs,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_objective(
        solution: Dict, objective: Optional[Dict], structured: Dict
    ) -> float:
        """Compute objective value from solution."""
        if not objective:
            return 0.0

        coefficients = structured.get("objective_coefficients", {})
        if coefficients:
            val = sum(
                coefficients.get(k, 0) * v
                for k, v in solution.items()
                if isinstance(v, (int, float))
            )
            return val

        # Default: sum of values
        numeric_vals = [v for v in solution.values() if isinstance(v, (int, float))]
        return sum(numeric_vals) if numeric_vals else 0.0
