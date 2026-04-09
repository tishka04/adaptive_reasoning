"""
Global optimization solver — uses OR-Tools CP-SAT for constraint
satisfaction and optimization problems.

Use cases:
  - scheduling / assignment / precedence problems
  - resource allocation / dispatch
  - network flow with constraints
  - any coupled constrained optimization

Wraps Google OR-Tools CP-SAT solver with a uniform interface.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from .base import BaseSolver, SolverResult


class GlobalOptSolver(BaseSolver):
    """
    Constraint-programming / optimization solver backed by OR-Tools CP-SAT.
    """

    name = "global_opt"

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

        time_limit = self.budget_to_seconds(budget)

        try:
            from ortools.sat.python import cp_model
        except ImportError:
            return SolverResult(
                success=False,
                violations=["ortools not installed"],
                logs=["pip install ortools"],
                elapsed_seconds=time.time() - t0,
            )

        # Determine problem type from structured_data
        problem_data = task.get("structured_data", {})
        problem_type = problem_data.get("type", self._infer_type(task, tool_hint))
        logs.append(f"Problem type: {problem_type}")

        try:
            if problem_type == "assignment":
                result = self._solve_assignment(problem_data, time_limit, logs)
            elif problem_type == "scheduling":
                result = self._solve_scheduling(problem_data, time_limit, logs)
            elif problem_type == "knapsack":
                result = self._solve_knapsack(problem_data, time_limit, logs)
            else:
                result = self._solve_generic_csp(task, problem_data, time_limit, logs)
        except Exception as e:
            return SolverResult(
                success=False,
                violations=[f"Solver exception: {str(e)}"],
                logs=logs + [f"Exception: {e}"],
                elapsed_seconds=time.time() - t0,
            )

        result.elapsed_seconds = time.time() - t0
        result.logs = logs
        return result

    # ------------------------------------------------------------------
    # Problem-specific solvers
    # ------------------------------------------------------------------
    def _solve_assignment(
        self, data: Dict, time_limit: float, logs: List[str]
    ) -> SolverResult:
        """Solve an assignment problem (agents → tasks)."""
        from ortools.sat.python import cp_model

        costs = data.get("costs", [[]])
        n_agents = len(costs)
        n_tasks = len(costs[0]) if costs else 0
        logs.append(f"Assignment: {n_agents} agents × {n_tasks} tasks")

        model = cp_model.CpModel()

        # Variables: x[i][j] = 1 if agent i assigned to task j
        x = {}
        for i in range(n_agents):
            for j in range(n_tasks):
                x[i, j] = model.NewBoolVar(f"x_{i}_{j}")

        # Each agent assigned to at most one task
        for i in range(n_agents):
            model.Add(sum(x[i, j] for j in range(n_tasks)) <= 1)

        # Each task assigned to exactly one agent
        for j in range(n_tasks):
            model.Add(sum(x[i, j] for i in range(n_agents)) == 1)

        # Minimize total cost
        objective = sum(
            costs[i][j] * x[i, j]
            for i in range(n_agents)
            for j in range(n_tasks)
        )
        model.Minimize(objective)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            assignment = {}
            for i in range(n_agents):
                for j in range(n_tasks):
                    if solver.Value(x[i, j]):
                        assignment[f"agent_{i}"] = f"task_{j}"
            logs.append(f"Objective: {solver.ObjectiveValue()}")
            return SolverResult(
                success=True,
                solution=assignment,
                score=solver.ObjectiveValue(),
                feasible=status == cp_model.OPTIMAL,
                metadata={"status": "optimal" if status == cp_model.OPTIMAL else "feasible"},
            )
        else:
            return SolverResult(
                success=False,
                violations=["No feasible assignment found"],
                metadata={"status": "infeasible"},
            )

    def _solve_scheduling(
        self, data: Dict, time_limit: float, logs: List[str]
    ) -> SolverResult:
        """Solve a job-shop / precedence scheduling problem."""
        from ortools.sat.python import cp_model

        jobs = data.get("jobs", [])
        horizon = data.get("horizon", 1000)

        if not jobs:
            return SolverResult(success=False, violations=["No jobs in scheduling data"])

        # Sanitize jobs: ensure duration >= 1, name is string
        sanitized_jobs = []
        for j in jobs:
            if not isinstance(j, dict) or "name" not in j:
                continue
            dur = j.get("duration", 1)
            if not isinstance(dur, (int, float)) or dur < 1:
                dur = 1
            sanitized_jobs.append({
                "name": str(j["name"]),
                "duration": int(dur),
                "dependencies": [str(d) for d in j.get("dependencies", []) if d],
            })
        jobs = sanitized_jobs

        # Ensure horizon is large enough
        total_dur = sum(j["duration"] for j in jobs)
        horizon = max(horizon, total_dur + 10)

        logs.append(f"Scheduling: {len(jobs)} jobs, horizon={horizon}")
        for j in jobs:
            logs.append(f"  {j['name']}: dur={j['duration']}, deps={j['dependencies']}")

        model = cp_model.CpModel()

        starts = {}
        ends = {}
        intervals = {}

        for job in jobs:
            name = job["name"]
            dur = job["duration"]
            starts[name] = model.NewIntVar(0, horizon, f"start_{name}")
            ends[name] = model.NewIntVar(0, horizon, f"end_{name}")
            intervals[name] = model.NewIntervalVar(
                starts[name], dur, ends[name], f"interval_{name}"
            )

        # Precedence constraints — only add if dep exists in the model
        dep_count = 0
        for job in jobs:
            for dep in job["dependencies"]:
                if dep in ends:
                    model.Add(starts[job["name"]] >= ends[dep])
                    dep_count += 1
                else:
                    logs.append(f"  WARNING: dependency '{dep}' for {job['name']} not found in jobs")
        logs.append(f"  Added {dep_count} precedence constraints")

        # Resource constraints (optional)
        resources = data.get("resources", {})
        for res_name, res_data in resources.items():
            cap = res_data.get("capacity", 1)
            res_jobs = res_data.get("jobs", [])
            res_intervals = [intervals[j] for j in res_jobs if j in intervals]
            demands = [res_data.get("demands", {}).get(j, 1) for j in res_jobs if j in intervals]
            if res_intervals:
                model.AddCumulative(res_intervals, demands, cap)

        # Minimize makespan
        makespan = model.NewIntVar(0, horizon, "makespan")
        model.AddMaxEquality(makespan, list(ends.values()))
        model.Minimize(makespan)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        status = solver.Solve(model)

        status_names = {
            cp_model.OPTIMAL: "OPTIMAL",
            cp_model.FEASIBLE: "FEASIBLE",
            cp_model.INFEASIBLE: "INFEASIBLE",
            cp_model.MODEL_INVALID: "MODEL_INVALID",
            cp_model.UNKNOWN: "UNKNOWN",
        }
        logs.append(f"  CP-SAT status: {status_names.get(status, status)}")

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            schedule = {
                name: {"start": solver.Value(starts[name]), "end": solver.Value(ends[name])}
                for name in starts
            }
            logs.append(f"Makespan: {solver.Value(makespan)}")
            return SolverResult(
                success=True,
                solution=schedule,
                score=float(solver.Value(makespan)),
                feasible=True,
                metadata={"makespan": solver.Value(makespan)},
            )
        else:
            return SolverResult(
                success=False,
                violations=[f"Scheduling {status_names.get(status, 'failed')}"],
                logs=logs,
            )

    def _solve_knapsack(
        self, data: Dict, time_limit: float, logs: List[str]
    ) -> SolverResult:
        """Solve a bounded knapsack / resource allocation problem."""
        from ortools.sat.python import cp_model

        items = data.get("items", [])
        capacity = data.get("capacity", 100)
        logs.append(f"Knapsack: {len(items)} items, capacity={capacity}")

        model = cp_model.CpModel()
        x = {}
        for i, item in enumerate(items):
            x[i] = model.NewBoolVar(f"item_{i}")

        # Capacity constraint
        model.Add(
            sum(items[i].get("weight", 1) * x[i] for i in range(len(items))) <= capacity
        )

        # Maximize value
        model.Maximize(
            sum(items[i].get("value", 1) * x[i] for i in range(len(items)))
        )

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            selected = [i for i in range(len(items)) if solver.Value(x[i])]
            return SolverResult(
                success=True,
                solution={"selected_items": selected},
                score=solver.ObjectiveValue(),
                feasible=True,
            )
        else:
            return SolverResult(success=False, violations=["Knapsack infeasible"])

    def _solve_generic_csp(
        self, task: Dict, data: Dict, time_limit: float, logs: List[str]
    ) -> SolverResult:
        """
        Generic CSP solver: creates integer variables from entities and
        adds constraints from the task description.
        """
        from ortools.sat.python import cp_model

        entities = task.get("entities", [])
        constraints = task.get("constraints", [])
        n = len(entities) if entities else data.get("num_vars", 5)
        domain_ub = data.get("domain_upper_bound", 100)
        logs.append(f"Generic CSP: {n} variables, {len(constraints)} constraints")

        model = cp_model.CpModel()
        variables = [model.NewIntVar(0, domain_ub, f"v_{i}") for i in range(n)]

        # All-different if specified
        if data.get("all_different", False):
            model.AddAllDifferent(variables)

        # Add parsed constraints
        for c in constraints:
            ctype = c.get("ctype", "hard")
            params = c.get("params", {})
            if params.get("type") == "sum_leq":
                model.Add(sum(variables) <= params.get("bound", domain_ub * n))
            elif params.get("type") == "sum_geq":
                model.Add(sum(variables) >= params.get("bound", 0))

        # Objective
        obj = task.get("objective")
        if obj and obj.get("sense") == "minimize":
            model.Minimize(sum(variables))
        elif obj and obj.get("sense") == "maximize":
            model.Maximize(sum(variables))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            vals = {f"v_{i}": solver.Value(variables[i]) for i in range(n)}
            return SolverResult(
                success=True,
                solution=vals,
                score=solver.ObjectiveValue() if obj else 0.0,
                feasible=True,
            )
        else:
            return SolverResult(success=False, violations=["Generic CSP infeasible"])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _infer_type(task: Dict, tool_hint: str) -> str:
        """Infer problem type from domain and hint."""
        domain = task.get("domain", "unknown")
        if "assignment" in tool_hint or "assignment" in str(task.get("description", "")):
            return "assignment"
        if domain == "scheduling" or "schedul" in tool_hint:
            return "scheduling"
        if "knapsack" in tool_hint or "allocation" in tool_hint:
            return "knapsack"
        return "generic_csp"
