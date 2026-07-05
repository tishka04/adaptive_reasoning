"""Learn how to calibrate ontology confidence from outcomes."""

from __future__ import annotations

from .common import ContextTable, bucket_float, clamp


class OntologyCalibrator:
    """Adjust symbolic ontology confidence with learned usefulness."""

    def __init__(self) -> None:
        self.table = ContextTable()
        self.kind_priors: dict[str, float] = {}

    def calibrate(self, ranked, obs, memory):
        phase = memory.fast.current_phase
        click_change = bool(
            memory.fast.last_transition is not None
            and memory.fast.last_transition.action.x is not None
            and memory.fast.last_transition.diff.num_changed > 0
        )
        big_change = bool(obs.frame_diff is not None and obs.frame_diff.num_changed >= 8)
        player_present = bool(obs.best_player is not None)
        mix = 0.10 if memory.game.total_actions < 15 else 0.18 if memory.game.total_actions < 40 else 0.28

        calibrated = []
        for ontology in ranked:
            prior = clamp(ontology.confidence)
            if ontology.kind in self.kind_priors:
                prior = clamp(0.80 * prior + 0.20 * self.kind_priors[ontology.kind])
            signatures = [
                (("ontology", ontology.kind), 0.45),
                (("ontology_phase", ontology.kind, phase), 0.35),
                (
                    (
                        "ontology_profile",
                        ontology.kind,
                        phase,
                        int(player_present),
                        int(click_change),
                        int(big_change),
                    ),
                    0.55,
                ),
            ]
            learned = self.table.estimate(signatures, prior=prior)
            ontology.confidence = clamp((1.0 - mix) * ontology.confidence + mix * learned)
            calibrated.append(ontology)

        calibrated.sort(key=lambda item: (item.confidence, item.evidence_for), reverse=True)
        return calibrated

    def update(self, ontology_kind: str, memory, reward: float) -> None:
        obs = memory.fast.current_obs
        click_change = bool(
            memory.fast.last_transition is not None
            and memory.fast.last_transition.action.x is not None
            and memory.fast.last_transition.diff.num_changed > 0
        )
        big_change = bool(obs is not None and obs.frame_diff is not None and obs.frame_diff.num_changed >= 8)
        player_present = bool(obs is not None and obs.best_player is not None)
        phase = memory.fast.current_phase
        signatures = [
            ("ontology", ontology_kind),
            ("ontology_phase", ontology_kind, phase),
            (
                "ontology_profile",
                ontology_kind,
                phase,
                int(player_present),
                int(click_change),
                int(big_change),
            ),
        ]
        for signature in signatures:
            self.table.update(signature, clamp(reward))

    def seed_priors(self, priors: dict[str, float]) -> None:
        self.kind_priors.update({str(key): clamp(float(value)) for key, value in priors.items()})

    def export_priors(self) -> dict[str, float]:
        exported: dict[str, float] = {}
        for signature, stat in self.table.stats.items():
            if len(signature) == 2 and signature[0] == "ontology" and stat.count >= 3:
                exported[str(signature[1])] = round(stat.mean, 3)
        return exported

