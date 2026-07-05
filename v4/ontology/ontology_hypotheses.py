"""Ontology hypothesis factory for V4."""

from __future__ import annotations

from ..schemas import ObservationV4, OntologyHypothesis

ONTOLOGY_KINDS = (
    "avatar_world",
    "click_world",
    "token_world",
    "field_world",
    "phase_world",
    "transform_world",
)

_ONTOLOGY_TERMINALS = {
    "avatar_world": ["reach_zone", "unlock_path", "avoid_hazard"],
    "click_world": ["clear_targets", "click_ordering", "toggle_completion"],
    "token_world": ["class_exhaustion", "object_removal", "count_reduction"],
    "field_world": ["paint_grid", "global_pattern", "coverage"],
    "phase_world": ["regime_shift", "unlock_after_sequence", "timing"],
    "transform_world": ["transform_then_finish", "state_toggle", "global_alignment"],
}

_ONTOLOGY_BIASES = {
    "avatar_world": {"movement": 1.25, "reachability": 1.10, "click": 0.80},
    "click_world": {"click": 1.30, "movement": 0.80, "small_objects": 1.15},
    "token_world": {"counts": 1.25, "class_targets": 1.20, "movement": 0.90},
    "field_world": {"coverage": 1.20, "transform": 1.05, "objects": 0.85},
    "phase_world": {"novelty": 1.20, "sequence": 1.15, "stability": 0.85},
    "transform_world": {"transform": 1.35, "click": 1.05, "movement": 0.90},
}


class OntologyFactory:
    """Create the initial ecology of candidate world-interpretations."""

    def initial_hypotheses(self, obs: ObservationV4) -> list[OntologyHypothesis]:
        hypotheses: list[OntologyHypothesis] = []
        salient_ids = [obj.object_id for obj in obs.objects[:12]]
        for kind in ONTOLOGY_KINDS:
            hypotheses.append(
                OntologyHypothesis(
                    ontology_id=kind,
                    kind=kind,
                    confidence=0.5,
                    evidence_for=0.0,
                    evidence_against=0.0,
                    salient_object_ids=salient_ids[:],
                    active_affordance_biases=_ONTOLOGY_BIASES[kind].copy(),
                    terminal_hypotheses=_ONTOLOGY_TERMINALS[kind][:],
                )
            )
        return hypotheses
