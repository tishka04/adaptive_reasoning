"""Estimate how trustworthy the current internal world-model is."""

from __future__ import annotations

from .common import ContextTable, bucket_float, clamp, entropy


class WorldReliabilityModel:
    """Predict whether the current belief state is likely to pay off soon."""

    def __init__(self) -> None:
        self.table = ContextTable()
        self.ontology_priors: dict[str, float] = {}

    def estimate(self, memory) -> float:
        lp, sp, tp = memory.game.progress.scores()
        repeat_ratio = self._repeat_ratio(memory)
        ontologies = getattr(memory.game, "current_ontologies", [])
        top_kind = ontologies[0].kind if ontologies else "unknown"
        ontology_entropy = entropy(item.confidence for item in ontologies[:3])
        phase = memory.fast.current_phase

        prior = clamp(
            0.12
            + 0.20 * lp
            + 0.28 * sp
            + 0.28 * tp
            - 0.30 * repeat_ratio
            - 0.05 * min(1.0, ontology_entropy / 1.1)
        )
        if top_kind in self.ontology_priors:
            prior = clamp(0.80 * prior + 0.20 * self.ontology_priors[top_kind])

        signatures = [
            (("world", top_kind), 0.45),
            (("world_phase", top_kind, phase), 0.40),
            (
                (
                    "world_profile",
                    top_kind,
                    phase,
                    bucket_float(lp),
                    bucket_float(sp),
                    bucket_float(tp),
                    bucket_float(repeat_ratio),
                ),
                0.55,
            ),
        ]
        return self.table.estimate(signatures, prior=prior)

    def update(self, memory, reward: float, sterile: bool = False, solved: bool = False) -> None:
        ontologies = getattr(memory.game, "current_ontologies", [])
        top_kind = ontologies[0].kind if ontologies else "unknown"
        lp, sp, tp = memory.game.progress.scores()
        repeat_ratio = self._repeat_ratio(memory)
        phase = memory.fast.current_phase
        value = clamp(0.80 * reward + (0.15 if solved else 0.0) - (0.20 if sterile else 0.0))
        signatures = [
            ("world", top_kind),
            ("world_phase", top_kind, phase),
            (
                "world_profile",
                top_kind,
                phase,
                bucket_float(lp),
                bucket_float(sp),
                bucket_float(tp),
                bucket_float(repeat_ratio),
            ),
        ]
        for signature in signatures:
            self.table.update(signature, value)

    def seed_priors(self, priors: dict[str, float]) -> None:
        self.ontology_priors.update({str(key): clamp(float(value)) for key, value in priors.items()})

    def export_priors(self) -> dict[str, float]:
        exported: dict[str, float] = {}
        for signature, stat in self.table.stats.items():
            if len(signature) == 2 and signature[0] == "world" and stat.count >= 3:
                exported[str(signature[1])] = round(stat.mean, 3)
        return exported

    def _repeat_ratio(self, memory) -> float:
        hashes = list(memory.fast.recent_hashes)
        if not hashes:
            return 0.0
        return max(hashes.count(item) for item in set(hashes)) / max(len(hashes), 1)

