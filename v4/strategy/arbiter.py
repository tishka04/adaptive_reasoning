"""Proposal arbitration for V4."""

from __future__ import annotations

from ..schemas import MindProposal


class Arbiter:
    """Select the best current proposal from the committee."""

    def __init__(self) -> None:
        self.selections: dict[str, int] = {}

    def select(self, proposals: list[MindProposal], memory) -> MindProposal | None:
        if not proposals:
            return None
        scored = []
        for proposal in proposals:
            heuristic = self._score(proposal, memory)
            adjusted = heuristic
            if getattr(memory.game, "learning", None) is not None:
                adjusted, signature, learned = memory.game.learning.arbiter_bandit.score(
                    proposal,
                    memory,
                    heuristic_score=heuristic,
                )
                proposal.metadata["bandit_signature"] = signature
                proposal.metadata["bandit_value"] = round(learned, 3)
            proposal.metadata["arbiter_score"] = round(adjusted, 3)
            scored.append((adjusted, proposal))
        scored.sort(key=lambda item: item[0], reverse=True)
        choice = scored[0][1]
        self.selections[choice.mind_name] = self.selections.get(choice.mind_name, 0) + 1
        memory.game.record_mind_selection(choice.mind_name)
        return choice

    def _score(self, proposal: MindProposal, memory) -> float:
        return (
            0.20 * proposal.expected_lp
            + 0.25 * proposal.expected_sp
            + 0.30 * proposal.expected_tp
            + 0.15 * memory.game.mind_reliability(proposal.mind_name)
            + 0.10 * proposal.confidence
            - 0.10 * proposal.cost
            - 0.10 * proposal.risk
        )
