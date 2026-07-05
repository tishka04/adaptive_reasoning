"""Forgetting and decay for V4."""

from __future__ import annotations


class ForgettingManager:
    """Decay stale beliefs so premature coherence does not calcify."""

    def decay(self, memory) -> None:
        drop_motifs = []
        for motif_id, motif in memory.game.motifs.items():
            motif.utility *= 0.99
            motif.terminal_association *= 0.995
            motif.structural_association *= 0.995
            if motif.support <= 1 and motif.utility < 0.04:
                drop_motifs.append(motif_id)
        for motif_id in drop_motifs:
            memory.game.motifs.pop(motif_id, None)

        if memory.game.project_market is not None:
            for project in memory.game.project_market.projects.values():
                project.dignity *= 0.995

        stale_rules = [
            rule_id
            for rule_id, rule in memory.game.constraints.rules.items()
            if rule.confidence < 0.1 and rule.support < 2
        ]
        for rule_id in stale_rules[:8]:
            memory.game.constraints.rules.pop(rule_id, None)
