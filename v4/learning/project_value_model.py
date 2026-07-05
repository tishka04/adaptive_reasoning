"""Learn which project types pay off in which contexts."""

from __future__ import annotations

from .common import ContextTable, bucket_float, clamp


class ProjectValueModel:
    """Estimate future project payoff from coarse context."""

    def __init__(self) -> None:
        self.table = ContextTable()
        self.kind_priors: dict[str, float] = {}

    def estimate(self, project, obs, memory) -> float:
        prior = clamp(
            0.25 * project.expected_info_gain
            + 0.30 * project.expected_structural_gain
            + 0.35 * project.expected_terminal_gain
            + 0.15 * project.dignity
            - 0.12 * project.fragility
            - 0.12 * project.estimated_cost
        )
        if project.kind in self.kind_priors:
            prior = clamp(0.75 * prior + 0.25 * self.kind_priors[project.kind])
        return self.table.estimate(self._signatures(project, memory), prior=prior)

    def update(self, project, memory, reward: float) -> None:
        value = clamp(reward)
        for signature, _ in self._signatures(project, memory):
            self.table.update(signature, value)

    def update_by_metadata(
        self,
        project_kind: str,
        ontology_kind: str,
        phase: str,
        lp: float,
        sp: float,
        tp: float,
        reward: float,
    ) -> None:
        signatures = [
            ("project_kind", project_kind),
            ("project_kind_onto", project_kind, ontology_kind),
            ("project_kind_phase", project_kind, phase),
            (
                "project_profile",
                project_kind,
                ontology_kind,
                phase,
                bucket_float(lp),
                bucket_float(sp),
                bucket_float(tp),
            ),
        ]
        for signature in signatures:
            self.table.update(signature, clamp(reward))

    def seed_priors(self, priors: dict[str, float]) -> None:
        self.kind_priors.update({str(key): clamp(float(value)) for key, value in priors.items()})

    def export_priors(self) -> dict[str, float]:
        exported: dict[str, float] = {}
        for signature, stat in self.table.stats.items():
            if len(signature) == 2 and signature[0] == "project_kind" and stat.count >= 3:
                exported[str(signature[1])] = round(stat.mean, 3)
        return exported

    def _signatures(self, project, memory) -> list[tuple[tuple[object, ...], float]]:
        lp, sp, tp = memory.game.progress.scores()
        ontology_kind = self._ontology_kind(project, memory)
        phase = memory.fast.current_phase
        return [
            (("project_kind", project.kind), 0.45),
            (("project_kind_onto", project.kind, ontology_kind), 0.40),
            (("project_kind_phase", project.kind, phase), 0.35),
            (
                (
                    "project_profile",
                    project.kind,
                    ontology_kind,
                    phase,
                    bucket_float(lp),
                    bucket_float(sp),
                    bucket_float(tp),
                ),
                0.55,
            ),
        ]

    def _ontology_kind(self, project, memory) -> str:
        for ontology in getattr(memory.game, "current_ontologies", []):
            if ontology.ontology_id == project.ontology_id:
                return ontology.kind
        return str(project.ontology_id or "unknown")

