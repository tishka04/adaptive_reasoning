"""
Scheduling verifier — checks scheduling solutions for:
  - precedence constraint satisfaction
  - resource capacity limits
  - no overlapping jobs on exclusive resources
  - makespan / objective quality
  - all jobs scheduled
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseVerifier, VerificationResult


class SchedulingVerifier(BaseVerifier):
    """Verifies scheduling / job-shop / precedence solutions."""

    name = "scheduling"

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
                violations=[{"name": "invalid_format", "description": "Solution must be a dict of job schedules", "severity": 1.0}],
            )

        structured = task.get("structured_data", {})
        jobs = structured.get("jobs", [])
        resources = structured.get("resources", {})
        horizon = structured.get("horizon", 10000)

        # Guard: reject solutions that are clearly not schedules
        sample_vals = list(solution.values())[:3]
        if sample_vals and not any(isinstance(v, dict) and ("start" in v or "end" in v) for v in sample_vals):
            return VerificationResult(
                valid=False,
                violations=[{"name": "wrong_format", "description": "Solution is not a schedule (expected {job: {start, end}})", "severity": 1.0}],
                logs=["Solution format does not match scheduling verifier expectations"],
            )

        logs.append(f"Verifying schedule: {len(solution)} entries, {len(jobs)} jobs defined")

        # Check 1: All jobs scheduled
        job_names = {j["name"] for j in jobs}
        scheduled_names = set(solution.keys())
        missing = job_names - scheduled_names
        if missing:
            for m in missing:
                violations.append({
                    "name": f"unscheduled_{m}",
                    "description": f"Job {m} not scheduled",
                    "severity": 1.0,
                })
            logs.append(f"  Missing jobs: {missing}")

        # Check 2: Valid time windows
        for name, sched in solution.items():
            start = sched.get("start", 0)
            end = sched.get("end", 0)
            if start < 0:
                violations.append({
                    "name": f"negative_start_{name}",
                    "description": f"Job {name} starts at {start} < 0",
                    "severity": 1.0,
                })
            if end > horizon:
                violations.append({
                    "name": f"exceeds_horizon_{name}",
                    "description": f"Job {name} ends at {end} > horizon {horizon}",
                    "severity": 0.7,
                })
            if start >= end:
                violations.append({
                    "name": f"zero_duration_{name}",
                    "description": f"Job {name}: start={start} >= end={end}",
                    "severity": 0.8,
                })

        # Check 3: Duration correctness
        job_map = {j["name"]: j for j in jobs}
        for name, sched in solution.items():
            if name in job_map:
                expected_dur = job_map[name].get("duration", 0)
                actual_dur = sched.get("end", 0) - sched.get("start", 0)
                if expected_dur > 0 and actual_dur < expected_dur:
                    violations.append({
                        "name": f"short_duration_{name}",
                        "description": f"Job {name}: duration {actual_dur} < required {expected_dur}",
                        "severity": 1.0,
                    })

        # Check 4: Precedence constraints
        for job in jobs:
            name = job["name"]
            if name not in solution:
                continue
            for dep in job.get("dependencies", []):
                if dep not in solution:
                    violations.append({
                        "name": f"missing_dep_{name}_{dep}",
                        "description": f"Dependency {dep} of {name} not in schedule",
                        "severity": 1.0,
                    })
                    continue
                dep_end = solution[dep].get("end", 0)
                job_start = solution[name].get("start", 0)
                if job_start < dep_end:
                    violations.append({
                        "name": f"precedence_{name}_{dep}",
                        "description": f"{name} starts at {job_start} before {dep} ends at {dep_end}",
                        "severity": 1.0,
                    })

        # Check 5: Resource capacity
        for res_name, res_data in resources.items():
            capacity = res_data.get("capacity", 1)
            res_jobs = res_data.get("jobs", [])
            demands = res_data.get("demands", {})

            # Build timeline and check capacity at each point
            events = []
            for jn in res_jobs:
                if jn not in solution:
                    continue
                s = solution[jn].get("start", 0)
                e = solution[jn].get("end", 0)
                d = demands.get(jn, 1)
                events.append((s, d))
                events.append((e, -d))

            events.sort()
            current_load = 0
            for time_pt, delta in events:
                current_load += delta
                if current_load > capacity:
                    violations.append({
                        "name": f"capacity_{res_name}_at_{time_pt}",
                        "description": f"Resource {res_name} over capacity ({current_load}/{capacity}) at t={time_pt}",
                        "severity": 1.0,
                    })
                    break  # Report once per resource

        # Compute makespan
        makespan = 0
        if solution:
            makespan = max(s.get("end", 0) for s in solution.values())

        # Check task constraints
        constraint_violations = self.check_constraints(task, solution)
        violations.extend(constraint_violations)

        # Summary
        tests_total = 5
        tests_passed = tests_total
        if missing:
            tests_passed -= 1
        if any("negative_start" in v["name"] or "zero_duration" in v["name"] for v in violations):
            tests_passed -= 1
        if any("precedence" in v["name"] for v in violations):
            tests_passed -= 1
        if any("capacity" in v["name"] for v in violations):
            tests_passed -= 1
        if any("short_duration" in v["name"] for v in violations):
            tests_passed -= 1

        hard_violations = [v for v in violations if v.get("severity", 0) >= 1.0]
        valid = len(hard_violations) == 0

        return VerificationResult(
            valid=valid,
            feasible=valid,
            score=1.0 - len(violations) / max(len(violations) + 5, 1),
            violations=violations,
            tests_passed=tests_passed,
            tests_total=tests_total,
            objective_value=float(makespan),
            metadata={"makespan": makespan, "num_jobs_scheduled": len(solution)},
            logs=logs,
        )
