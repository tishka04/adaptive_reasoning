"""Learn which teleology hypotheses deserve promotion."""

from __future__ import annotations

from .common import ContextTable, bucket_float, clamp


class TeleologyValidator:
    """Convert speculative teleology into validated guidance cautiously."""

    def __init__(self) -> None:
        self.table = ContextTable()

    def refine(self, rule, memory):
        learned = self.estimate(rule, memory)
        rule.confidence = clamp(0.75 * rule.confidence + 0.25 * learned)
        if rule.stage == "speculative" and rule.support >= 2 and learned >= 0.62:
            rule.stage = "validated"
        elif rule.stage == "validated" and rule.support < 5 and learned <= 0.25:
            rule.stage = "speculative"
        stage_bonus = 0.10 if rule.stage == "validated" else 0.0
        base = 0.50 * rule.confidence + 0.20 * min(rule.support / 4.0, 1.0) + stage_bonus
        rule.survival_score = clamp(0.60 * rule.survival_score + 0.40 * base)
        return rule

    def estimate(self, rule, memory) -> float:
        prior = clamp(0.65 * rule.confidence + 0.20 * min(rule.support / 4.0, 1.0))
        return self.table.estimate(self._signatures(rule, memory), prior=prior)

    def update(self, rule, memory, reward: float) -> None:
        for signature, _ in self._signatures(rule, memory):
            self.table.update(signature, clamp(reward))

    def _signatures(self, rule, memory) -> list[tuple[tuple[object, ...], float]]:
        phase = memory.fast.current_phase
        effect_kind = str(rule.effect.args.get("kind", rule.effect.kind))
        ontology_tag = rule.ontology_tags[0] if rule.ontology_tags else "unknown"
        lp, sp, tp = memory.game.progress.scores()
        return [
            (("teleology_kind", effect_kind), 0.45),
            (("teleology_context", effect_kind, ontology_tag, phase), 0.55),
            (
                (
                    "teleology_profile",
                    effect_kind,
                    phase,
                    bucket_float(lp),
                    bucket_float(sp),
                    bucket_float(tp),
                ),
                0.35,
            ),
        ]

