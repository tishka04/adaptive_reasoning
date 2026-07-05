"""Learn operator usefulness beyond predictive existence."""

from __future__ import annotations

from .common import ContextTable, bucket_float, clamp


class OperatorUtilityModel:
    """Predict whether an operator is strategically valuable."""

    def __init__(self) -> None:
        self.table = ContextTable()

    def estimate(self, operator, memory) -> float:
        prior = clamp(0.50 * operator.confidence + 0.20 * min(operator.support / 4.0, 1.0))
        return self.table.estimate(self._signatures(operator, memory), prior=prior)

    def adjust(self, base_score: float, operator, memory) -> float:
        learned = self.estimate(operator, memory)
        return base_score + 0.35 * (learned - 0.50)

    def update(self, operator, memory, reward: float) -> None:
        for signature, _ in self._signatures(operator, memory):
            self.table.update(signature, clamp(reward))

    def _signatures(self, operator, memory) -> list[tuple[tuple[object, ...], float]]:
        lp, sp, tp = memory.game.progress.scores()
        phase = memory.fast.current_phase
        ontology_kind = memory.game.current_ontologies[0].kind if memory.game.current_ontologies else "unknown"
        project_kind = "none"
        project_id = memory.fast.current_project_id
        if project_id and memory.game.project_market is not None:
            project = memory.game.project_market.projects.get(project_id)
            if project is not None:
                project_kind = project.kind
        return [
            (("operator_kind", operator.kind), 0.45),
            (("operator_context", operator.kind, ontology_kind, project_kind), 0.45),
            (
                (
                    "operator_profile",
                    operator.kind,
                    phase,
                    bucket_float(lp),
                    bucket_float(sp),
                    bucket_float(tp),
                ),
                0.40,
            ),
            (("operator_action", operator.primitive_action, operator.kind), 0.25),
        ]

