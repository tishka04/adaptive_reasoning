"""Ranking for operators, constraints, and teleological laws."""

from __future__ import annotations

from ..schemas import Operator, Rule


class LawCompetition:
    """Maintain ranked views of the current law ecology."""

    def __init__(self) -> None:
        self.ranked_operators: list[Operator] = []
        self.ranked_rules: list[Rule] = []
        self.ranked_teleology: list[Rule] = []

    def update(
        self,
        operators: list[Operator],
        rules: list[Rule],
        teleology: list[Rule],
    ) -> None:
        self.ranked_operators = sorted(
            operators,
            key=lambda op: (op.confidence, op.support, -op.risk_estimate),
            reverse=True,
        )
        self.ranked_rules = sorted(
            rules,
            key=lambda rule: (rule.confidence, rule.support, -rule.contradictions),
            reverse=True,
        )
        self.ranked_teleology = sorted(
            teleology,
            key=lambda rule: (rule.confidence, rule.support),
            reverse=True,
        )
