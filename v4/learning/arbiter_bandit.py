"""Contextual bandit for proposal arbitration."""

from __future__ import annotations

import math

from .common import ContextTable, bucket_float, clamp


class ArbiterBandit:
    """Learn which mind proposal is worth trusting in context."""

    def __init__(self) -> None:
        self.table = ContextTable()

    def score(self, proposal, memory, heuristic_score: float) -> tuple[float, tuple[object, ...], float]:
        signatures = self._signatures(proposal, memory)
        prior = clamp(0.50 + 0.35 * heuristic_score)
        learned = self.table.estimate(signatures, prior=prior)
        support = self.table.support(signatures)
        exploration = 0.12 / math.sqrt(1.0 + support)
        adjusted = heuristic_score + 0.30 * (learned - 0.50) + exploration
        return adjusted, signatures[0][0], learned

    def update(self, signature: tuple[object, ...], reward: float) -> None:
        self.table.update(signature, clamp(reward))

    def update_by_metadata(
        self,
        mind_name: str,
        project_kind: str,
        ontology_kind: str,
        phase: str,
        lp: float,
        sp: float,
        tp: float,
        reward: float,
    ) -> None:
        signature = (
            "mind",
            mind_name,
            project_kind,
            ontology_kind,
            phase,
            bucket_float(lp),
            bucket_float(sp),
            bucket_float(tp),
        )
        self.table.update(signature, clamp(reward))

    def _signatures(self, proposal, memory) -> list[tuple[tuple[object, ...], float]]:
        lp, sp, tp = memory.game.progress.scores()
        ontology_kind = memory.game.current_ontologies[0].kind if memory.game.current_ontologies else "unknown"
        phase = memory.fast.current_phase
        return [
            (
                (
                    "mind",
                    proposal.mind_name,
                    proposal.project.kind,
                    ontology_kind,
                    phase,
                    bucket_float(lp),
                    bucket_float(sp),
                    bucket_float(tp),
                ),
                0.60,
            ),
            (("mind_kind", proposal.mind_name, proposal.project.kind), 0.35),
            (("mind_phase", proposal.mind_name, phase), 0.25),
        ]

