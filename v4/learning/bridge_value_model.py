"""Learn which frame crossings are useful for bissociation."""

from __future__ import annotations

from .common import ContextTable, bucket_float, clamp


class BridgeValueModel:
    """Estimate whether a candidate frame crossing is productive."""

    def __init__(self) -> None:
        self.table = ContextTable()
        self.kind_priors: dict[str, float] = {}

    def estimate(self, bridge, primary_frame, secondary_frame, memory) -> float:
        pair = f"{primary_frame.ontology_kind}->{secondary_frame.ontology_kind}"
        prior = clamp(
            0.45 * bridge.utility_score
            + 0.20 * primary_frame.reliability
            + 0.20 * secondary_frame.reliability
            + 0.15 * bridge.novelty_score
        )
        if pair in self.kind_priors:
            prior = clamp(0.75 * prior + 0.25 * self.kind_priors[pair])
        phase = memory.fast.current_phase
        lp, sp, tp = memory.game.progress.scores()
        signatures = [
            (("bridge_pair", pair), 0.50),
            (("bridge_tension", str(bridge.hybrid_hypothesis_template.get("tension_type", "hybrid"))), 0.25),
            (
                (
                    "bridge_profile",
                    pair,
                    phase,
                    bucket_float(sp),
                    bucket_float(tp),
                ),
                0.45,
            ),
        ]
        return self.table.estimate(signatures, prior=prior)

    def update(self, bridge_key: str, phase: str, sp: float, tp: float, reward: float) -> None:
        signatures = [
            ("bridge_pair", bridge_key),
            ("bridge_profile", bridge_key, phase, bucket_float(sp), bucket_float(tp)),
        ]
        for signature in signatures:
            self.table.update(signature, clamp(reward))

    def seed_priors(self, priors: dict[str, float]) -> None:
        self.kind_priors.update({str(key): clamp(float(value)) for key, value in priors.items()})

    def export_priors(self) -> dict[str, float]:
        exported: dict[str, float] = {}
        for signature, stat in self.table.stats.items():
            if len(signature) == 2 and signature[0] == "bridge_pair" and stat.count >= 2:
                exported[str(signature[1])] = round(stat.mean, 3)
        return exported

