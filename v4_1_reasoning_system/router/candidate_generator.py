"""
Candidate reasoning generator — produces a small set of reasoning
candidates for the router to score.

Each candidate is: (mode, budget, strictness, tool_hint)

Initially rule-based; can be replaced with a learned generator later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ReasoningCandidate:
    """A single candidate reasoning action."""
    mode: str                          # hierarchical | global_opt | repair | llm_codegen
    budget: str = "medium"             # low | medium | high
    strictness: str = "verified"       # fast | verified | strict
    tool_hint: str = ""                # optional hint for the solver
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "budget": self.budget,
            "strictness": self.strictness,
            "tool_hint": self.tool_hint,
            **self.metadata,
        }


# ------------------------------------------------------------------
# Rule-based templates per domain
# ------------------------------------------------------------------
_PLANNING_CANDIDATES = [
    ReasoningCandidate("hierarchical", "medium", "verified", "depth_2"),
    ReasoningCandidate("hierarchical", "high", "strict", "depth_3"),
    ReasoningCandidate("llm_codegen", "medium", "verified", "make_plan_skeleton"),
    ReasoningCandidate("repair", "low", "verified", "local_patch"),
]

_SCHEDULING_CANDIDATES = [
    ReasoningCandidate("global_opt", "high", "strict", "feasibility_priority"),
    ReasoningCandidate("global_opt", "medium", "verified", "objective_priority"),
    ReasoningCandidate("hierarchical", "medium", "verified", "decompose_phases"),
    ReasoningCandidate("repair", "low", "verified", "fix_violations"),
]

_OPTIMIZATION_CANDIDATES = [
    ReasoningCandidate("global_opt", "high", "strict", "full_solve"),
    ReasoningCandidate("global_opt", "medium", "fast", "greedy_then_refine"),
    ReasoningCandidate("llm_codegen", "medium", "verified", "make_solver_skeleton"),
    ReasoningCandidate("repair", "low", "verified", "relax_constraints"),
]

_CODING_CANDIDATES = [
    ReasoningCandidate("llm_codegen", "medium", "verified", "generate_solution"),
    ReasoningCandidate("llm_codegen", "high", "strict", "generate_and_test"),
    ReasoningCandidate("repair", "low", "verified", "patch_failing_code"),
    ReasoningCandidate("hierarchical", "medium", "verified", "decompose_subtasks"),
]

_DEFAULT_CANDIDATES = [
    ReasoningCandidate("hierarchical", "medium", "verified", ""),
    ReasoningCandidate("global_opt", "medium", "verified", ""),
    ReasoningCandidate("repair", "low", "verified", ""),
    ReasoningCandidate("llm_codegen", "medium", "verified", ""),
]

_DOMAIN_TEMPLATES = {
    "planning": _PLANNING_CANDIDATES,
    "scheduling": _SCHEDULING_CANDIDATES,
    "optimization": _OPTIMIZATION_CANDIDATES,
    "coding": _CODING_CANDIDATES,
    "unknown": _DEFAULT_CANDIDATES,
}


class CandidateGenerator:
    """
    Produces candidate reasoning actions given the current task context.

    Phase 1: rule-based templates per domain.
    Phase 2+: can be augmented with learned generation.
    """

    def __init__(self, max_candidates: int = 8):
        self.max_candidates = max_candidates
        self.templates = dict(_DOMAIN_TEMPLATES)

    def generate(
        self,
        domain: str,
        iteration: int = 0,
        last_mode: Optional[str] = None,
        last_success: bool = True,
        feasible: bool = True,
        score: float = 0.0,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ReasoningCandidate]:
        """
        Generate candidates given reasoning context.

        Args:
            domain: domain guess from parser
            iteration: current reasoning loop iteration
            last_mode: mode used in the previous step
            last_success: whether the last step succeeded
            feasible: whether the current solution is feasible
            score: current objective score (normalised)
            context: optional extra context

        Returns:
            List of ReasoningCandidate
        """
        base = list(self.templates.get(domain, _DEFAULT_CANDIDATES))

        # Adaptive adjustments
        candidates = []
        for c in base:
            candidates.append(c)

        # If last step failed, prioritise repair (but not on first iteration)
        if not last_success and iteration > 0:
            candidates.insert(0, ReasoningCandidate(
                "repair", "medium", "verified", "fix_last_failure",
            ))

        # If infeasible and we have a previous solution, add aggressive repair
        if not feasible and iteration > 0:
            candidates.insert(0, ReasoningCandidate(
                "repair", "high", "strict", "restore_feasibility",
            ))

        # If stuck (same mode used repeatedly), inject diversity
        if last_mode and iteration > 2:
            alt_modes = [m for m in ["hierarchical", "global_opt", "repair", "llm_codegen"] if m != last_mode]
            for m in alt_modes[:2]:
                candidates.append(ReasoningCandidate(m, "medium", "verified", "diversity_inject"))

        # Budget escalation on later iterations
        if iteration >= 3:
            candidates.append(ReasoningCandidate(
                "global_opt", "high", "strict", "escalated_solve",
            ))

        # Deduplicate by (mode, tool_hint)
        seen = set()
        unique = []
        for c in candidates:
            key = (c.mode, c.tool_hint)
            if key not in seen:
                seen.add(key)
                unique.append(c)

        return unique[: self.max_candidates]

    def register_template(self, domain: str, candidates: List[ReasoningCandidate]) -> None:
        """Register custom candidate templates for a new domain."""
        self.templates[domain] = candidates
