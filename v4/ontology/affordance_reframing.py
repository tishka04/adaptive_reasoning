"""Ontology-conditioned affordance reweighting."""

from __future__ import annotations

from ..schemas import ObservationV4, OntologyHypothesis


class AffordanceReframer:
    """Reweight the same scene differently under different world-models."""

    def reweight(
        self,
        obs: ObservationV4,
        ontology: OntologyHypothesis,
    ) -> dict[str, float]:
        weights = {
            "movement": 1.0,
            "click": 1.0,
            "counts": 1.0,
            "transform": 1.0,
            "reachability": 1.0,
            "small_objects": 1.0,
            "coverage": 1.0,
            "novelty": 1.0,
            "sequence": 1.0,
        }
        weights.update(ontology.active_affordance_biases)

        if obs.best_player is None:
            weights["movement"] *= 0.75
            weights["reachability"] *= 0.80
        if not any(action == "ACTION6" for action in obs.available_actions):
            weights["click"] *= 0.50
        if obs.frame_diff is not None and obs.frame_diff.num_changed >= 8:
            weights["transform"] *= 1.10
        if len(obs.objects) <= 4:
            weights["small_objects"] *= 0.90
            weights["counts"] *= 0.85
        return weights
