"""Learn which motifs and rituals are worth preserving."""

from __future__ import annotations

from .common import ContextTable, bucket_float, clamp


class CompressionValueModel:
    """Estimate reuse value for motifs and rituals."""

    def __init__(self) -> None:
        self.motif_table = ContextTable()
        self.ritual_table = ContextTable()
        self.kind_priors: dict[str, float] = {}

    def estimate_motif(self, motif, memory) -> float:
        prior = clamp(
            0.35 * motif.utility
            + 0.25 * motif.structural_association
            + 0.25 * motif.terminal_association
            + 0.15 * min(motif.support / 4.0, 1.0)
        )
        if motif.kind in self.kind_priors:
            prior = clamp(0.80 * prior + 0.20 * self.kind_priors[motif.kind])
        signatures = [
            (("motif_kind", motif.kind), 0.50),
            (("motif_support", motif.kind, bucket_float(min(motif.support / 5.0, 1.0))), 0.35),
        ]
        return self.motif_table.estimate(signatures, prior=prior)

    def estimate_ritual(self, ritual, memory) -> float:
        prior = clamp(
            0.60 * ritual.success_rate
            + 0.20 * (1.0 / max(len(ritual.prefix), 1))
            + 0.20 * min(1.0, ritual.terminal_signature.get("levels_completed", 0) / 3.0)
        )
        key = f"ritual:{ritual.ontology_kind}"
        if key in self.kind_priors:
            prior = clamp(0.80 * prior + 0.20 * self.kind_priors[key])
        signatures = [
            (("ritual_ontology", ritual.ontology_kind), 0.55),
            (("ritual_length", ritual.ontology_kind, bucket_float(1.0 / max(len(ritual.prefix), 1), step=0.1, maximum=9)), 0.25),
        ]
        return self.ritual_table.estimate(signatures, prior=prior)

    def update_motif(self, motif, reward: float) -> None:
        signatures = [
            ("motif_kind", motif.kind),
            ("motif_support", motif.kind, bucket_float(min(motif.support / 5.0, 1.0))),
        ]
        for signature in signatures:
            self.motif_table.update(signature, clamp(reward))

    def update_ritual(self, ritual, reward: float) -> None:
        signatures = [
            ("ritual_ontology", ritual.ontology_kind),
            ("ritual_length", ritual.ontology_kind, bucket_float(1.0 / max(len(ritual.prefix), 1), step=0.1, maximum=9)),
        ]
        for signature in signatures:
            self.ritual_table.update(signature, clamp(reward))

    def seed_priors(self, priors: dict[str, float]) -> None:
        self.kind_priors.update({str(key): clamp(float(value)) for key, value in priors.items()})

    def export_priors(self) -> dict[str, float]:
        exported: dict[str, float] = {}
        for signature, stat in self.motif_table.stats.items():
            if len(signature) == 2 and signature[0] == "motif_kind" and stat.count >= 3:
                exported[str(signature[1])] = round(stat.mean, 3)
        for signature, stat in self.ritual_table.stats.items():
            if len(signature) == 2 and signature[0] == "ritual_ontology" and stat.count >= 2:
                exported[f"ritual:{signature[1]}"] = round(stat.mean, 3)
        return exported

