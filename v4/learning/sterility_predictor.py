"""Predict when a branch is likely to turn into a zombie trajectory."""

from __future__ import annotations

from .common import ContextTable, bucket_float, clamp, entropy


class SterilityPredictor:
    """Forecast branch sterility before hand-tuned reset rules fully trigger."""

    def __init__(self) -> None:
        self.table = ContextTable()

    def predict(self, memory) -> float:
        lp, sp, tp = memory.game.progress.scores()
        repeat_ratio = self._repeat_ratio(memory)
        project_concentration = self._project_concentration(memory)
        ontology_entropy = entropy(item.confidence for item in memory.game.current_ontologies[:3])
        phase = memory.fast.current_phase
        prior = clamp(
            0.12
            + 0.40 * repeat_ratio
            + 0.20 * max(0.0, 0.12 - sp) * 3.0
            + 0.20 * max(0.0, 0.08 - tp) * 4.0
            + 0.08 * project_concentration
            - 0.05 * min(1.0, ontology_entropy / 1.1)
        )
        signatures = [
            (("sterility_phase", phase), 0.30),
            (
                (
                    "sterility_profile",
                    phase,
                    bucket_float(lp),
                    bucket_float(sp),
                    bucket_float(tp),
                    bucket_float(repeat_ratio),
                    bucket_float(project_concentration),
                ),
                0.60,
            ),
        ]
        return self.table.estimate(signatures, prior=prior)

    def update(self, memory, sterile: bool, reward: float) -> None:
        lp, sp, tp = memory.game.progress.scores()
        repeat_ratio = self._repeat_ratio(memory)
        project_concentration = self._project_concentration(memory)
        phase = memory.fast.current_phase
        value = 1.0 if sterile else clamp(0.35 - 0.25 * reward)
        signatures = [
            ("sterility_phase", phase),
            (
                "sterility_profile",
                phase,
                bucket_float(lp),
                bucket_float(sp),
                bucket_float(tp),
                bucket_float(repeat_ratio),
                bucket_float(project_concentration),
            ),
        ]
        for signature in signatures:
            self.table.update(signature, value)

    def _repeat_ratio(self, memory) -> float:
        hashes = list(memory.fast.recent_hashes)
        if not hashes:
            return 0.0
        return max(hashes.count(item) for item in set(hashes)) / max(len(hashes), 1)

    def _project_concentration(self, memory) -> float:
        projects = memory.game.selected_projects[-12:]
        if not projects:
            return 0.0
        return max(projects.count(item) for item in set(projects)) / max(len(projects), 1)

