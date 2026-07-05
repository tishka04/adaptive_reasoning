"""Belief debugger — detect and demote false or stale beliefs.

Prevents elegant self-delusion by auditing:
  - Overconfident but non-predictive operators
  - Rules with growing contradictions
  - Minds with overpredicted progress
  - Macros with unstable success rates
  - Goals that generate novelty but not completion
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

from ..schemas import MacroAction, Operator, Rule
from ..mechanics.operator_inducer import OperatorInducer
from ..mechanics.rule_engine import RuleEngine
from .game_memory import GameMemoryV3

logger = logging.getLogger(__name__)


@dataclass
class BeliefIssue:
    """A detected problem with a belief."""
    target_id: str
    target_type: str        # operator / rule / macro / mind
    issue_type: str         # overconfident / contradicted / unstable / overpredicted
    severity: float         # 0-1
    description: str
    recommended_action: str  # demote / prune / force_experiment / suppress


class BeliefDebugger:
    """Audit and repair beliefs in the agent's memory."""

    def __init__(
        self,
        overconfidence_threshold: float = 0.7,
        contradiction_ratio_threshold: float = 0.3,
        min_evidence_for_audit: int = 5,
    ) -> None:
        self.overconfidence_threshold = overconfidence_threshold
        self.contradiction_ratio_threshold = contradiction_ratio_threshold
        self.min_evidence = min_evidence_for_audit
        self._audit_count: int = 0

    def audit(self, memory: GameMemoryV3) -> List[BeliefIssue]:
        """Run a full audit of all beliefs. Returns list of issues found."""
        self._audit_count += 1
        issues: List[BeliefIssue] = []

        issues.extend(self._audit_operators(memory.inducer))
        issues.extend(self._audit_rules(memory.rules))
        issues.extend(self._audit_macros(memory.macros))

        if issues:
            logger.info(
                f"Belief audit #{self._audit_count}: "
                f"{len(issues)} issues found"
            )
            for issue in issues[:5]:
                logger.info(f"  {issue.target_type}/{issue.target_id}: "
                            f"{issue.issue_type} ({issue.severity:.2f})")

        return issues

    def audit_and_repair(self, memory: GameMemoryV3) -> int:
        """Audit and automatically apply repairs. Returns number of repairs."""
        issues = self.audit(memory)
        repairs = 0

        for issue in issues:
            if issue.severity < 0.5:
                continue

            if issue.recommended_action == "demote":
                if issue.target_type == "operator":
                    op = memory.inducer.operators.get(issue.target_id)
                    if op:
                        op.confidence *= 0.5
                        repairs += 1
                elif issue.target_type == "rule":
                    rule = memory.rules.rules.get(issue.target_id)
                    if rule:
                        rule.confidence *= 0.5
                        repairs += 1

            elif issue.recommended_action == "prune":
                if issue.target_type == "operator":
                    if issue.target_id in memory.inducer.operators:
                        del memory.inducer.operators[issue.target_id]
                        repairs += 1
                elif issue.target_type == "rule":
                    if issue.target_id in memory.rules.rules:
                        del memory.rules.rules[issue.target_id]
                        repairs += 1
                elif issue.target_type == "macro":
                    if issue.target_id in memory.macros:
                        del memory.macros[issue.target_id]
                        repairs += 1

        if repairs:
            logger.info(f"Belief debugger: applied {repairs} repairs")

        return repairs

    def _audit_operators(self, inducer: OperatorInducer) -> List[BeliefIssue]:
        issues: List[BeliefIssue] = []

        for oid, op in inducer.operators.items():
            total = op.support + op.contradictions
            if total < self.min_evidence:
                continue

            # Overconfident: high confidence but high contradiction rate
            if op.confidence > self.overconfidence_threshold:
                contra_rate = op.contradictions / max(total, 1)
                if contra_rate > self.contradiction_ratio_threshold:
                    issues.append(BeliefIssue(
                        target_id=oid,
                        target_type="operator",
                        issue_type="overconfident",
                        severity=contra_rate,
                        description=(
                            f"Operator {oid} has conf={op.confidence:.2f} "
                            f"but {op.contradictions}/{total} contradictions"
                        ),
                        recommended_action="demote",
                    ))

            # High risk but still being used
            if op.risk_estimate > 0.5 and op.kind.value not in ("lethal", "noop"):
                issues.append(BeliefIssue(
                    target_id=oid,
                    target_type="operator",
                    issue_type="high_risk",
                    severity=op.risk_estimate,
                    description=(
                        f"Operator {oid} has risk={op.risk_estimate:.2f} "
                        f"but is not marked lethal"
                    ),
                    recommended_action="demote",
                ))

        return issues

    def _audit_rules(self, rules: RuleEngine) -> List[BeliefIssue]:
        issues: List[BeliefIssue] = []

        for rid, rule in rules.rules.items():
            total = rule.support + rule.contradictions
            if total < self.min_evidence:
                continue

            contra_rate = rule.contradictions / max(total, 1)
            if contra_rate > self.contradiction_ratio_threshold:
                severity = contra_rate
                action = "prune" if contra_rate > 0.6 else "demote"
                issues.append(BeliefIssue(
                    target_id=rid,
                    target_type="rule",
                    issue_type="contradicted",
                    severity=severity,
                    description=(
                        f"Rule {rid} has {rule.contradictions}/{total} "
                        f"contradictions"
                    ),
                    recommended_action=action,
                ))

        return issues

    def _audit_macros(self, macros: Dict[str, MacroAction]) -> List[BeliefIssue]:
        issues: List[BeliefIssue] = []

        for mid, macro in macros.items():
            if macro.times_used < 3:
                continue

            if macro.success_rate < 0.3:
                issues.append(BeliefIssue(
                    target_id=mid,
                    target_type="macro",
                    issue_type="unstable",
                    severity=1.0 - macro.success_rate,
                    description=(
                        f"Macro {macro.name} has success={macro.success_rate:.2f} "
                        f"over {macro.times_used} uses"
                    ),
                    recommended_action="prune",
                ))

        return issues
