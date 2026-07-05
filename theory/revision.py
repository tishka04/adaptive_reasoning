"""Belief revision (step A3) — reuses the v3 audit + rule machinery.

Two reuse points, exactly as called for in CONSOLIDATION_PLAN.md:

1. ``revise_theory`` adapts each ``MechanicHypothesis`` to the operator
   interface that ``v3.memory.belief_debugger.BeliefDebugger`` audits, then
   RE-OPENS (demotes to UNRESOLVED) any confirmed-but-contradicted or risky
   belief so the experiment designer will probe it again.

2. ``verify_relational_rules`` is a thin pass-through to
   ``v3.mechanics.rule_engine.RuleEngine.propose_and_verify`` for the
   relational rule layer (death / completion / blocking / removal), used once
   the loop runs on full ``TransitionRecord`` observations.
"""

from __future__ import annotations

from typing import Any, List

from v3.memory.belief_debugger import BeliefDebugger, BeliefIssue
from v3.mechanics.rule_engine import RuleEngine
from v3.schemas import OperatorKind, Rule, TransitionRecord

from .epistemic_metrics import HypothesisStatus
from .mechanic_hypothesis import GameTheory, MechanicHypothesis


class _OperatorView:
    """Adapt a MechanicHypothesis to the attributes BeliefDebugger reads."""

    def __init__(self, hyp: MechanicHypothesis) -> None:
        self._hyp = hyp
        self.confidence = hyp.confidence
        self.support = hyp.support
        self.contradictions = hyp.contradictions
        self.risk_estimate = 1.0 if hyp.kind == "lethal" else 0.0
        try:
            self.kind = OperatorKind(hyp.kind)
        except ValueError:
            self.kind = OperatorKind.UNKNOWN


class _InducerView:
    """Minimal stand-in exposing ``.operators`` for BeliefDebugger."""

    def __init__(self, theory: GameTheory) -> None:
        self.operators = {h.key: _OperatorView(h) for h in theory.hypotheses()}


def revise_theory(theory: GameTheory, debugger: BeliefDebugger | None = None) -> int:
    """Audit the theory with v3 BeliefDebugger; re-open untrustworthy beliefs.

    Returns the number of hypotheses revised (demoted back to UNRESOLVED).
    """
    debugger = debugger or BeliefDebugger()
    inducer = _InducerView(theory)
    issues: List[BeliefIssue] = debugger._audit_operators(inducer)  # noqa: SLF001

    by_key = {h.key: h for h in theory.hypotheses()}
    revised = 0
    for issue in issues:
        if issue.severity < 0.5:
            continue
        hyp = by_key.get(issue.target_id)
        if hyp is None:
            continue
        if hyp.status == HypothesisStatus.CONFIRMED:
            hyp.status = HypothesisStatus.UNRESOLVED
            revised += 1
    return revised


def verify_relational_rules(
    rule_engine: RuleEngine,
    profiler: Any,
    transitions: List[TransitionRecord],
) -> List[Rule]:
    """Reuse the v3 RuleEngine to propose/verify relational causal rules.

    For the full-observation path (real GameObservations + ActionProfiler).
    The prototype's ObservedEffect stream does not carry grids, so this bridge
    is exercised only once the loop is wired to live perception.
    """
    rules = rule_engine.propose_and_verify(profiler=profiler, transitions=transitions)
    return list(rules.values())
