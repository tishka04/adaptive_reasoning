"""Competition among candidate ontologies."""

from __future__ import annotations

import math
from typing import Any

from ..schemas import ObservationV4, OntologyHypothesis
from .ontology_hypotheses import ONTOLOGY_KINDS, OntologyFactory


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-8.0, min(8.0, x))))


class OntologyCompetition:
    """Maintain and update multiple candidate interpretations of the game."""

    def __init__(self) -> None:
        self._factory = OntologyFactory()
        self._hypotheses: dict[str, OntologyHypothesis] = {}
        self._last_ranked: list[OntologyHypothesis] = []

    def ensure_initialized(self, obs: ObservationV4, memory: Any) -> None:
        if self._hypotheses:
            return
        hypotheses = self._factory.initial_hypotheses(obs)
        priors = getattr(memory.game, "ontology_priors", {})
        for hypothesis in hypotheses:
            prior = float(priors.get(hypothesis.kind, 0.0))
            hypothesis.evidence_for += prior
            hypothesis.confidence = _sigmoid(prior)
            self._hypotheses[hypothesis.ontology_id] = hypothesis
        self._last_ranked = list(self._hypotheses.values())

    def update(self, obs: ObservationV4, memory: Any) -> None:
        self.ensure_initialized(obs, memory)
        transition = memory.fast.last_transition
        best_player = obs.best_player
        object_count = len(obs.objects)
        small_object_count = sum(1 for obj in obs.objects if obj.area <= 12 and obj.value != 0)
        diff = obs.frame_diff
        big_change = diff is not None and diff.num_changed >= 8
        click_change = (
            transition is not None
            and transition.action.x is not None
            and transition.diff.num_changed > 0
        )
        removed_values = transition.metadata.get("removed_values", {}) if transition else {}
        topology_unlock = bool(obs.topology.unlocked_regions)
        move_ops = memory.game.inducer.get_by_kind("move")
        click_ops = memory.game.inducer.get_by_kind("click")
        transform_ops = memory.game.inducer.get_by_kind("global_transform")
        teleology = memory.game.teleology.hypotheses()

        for kind in ONTOLOGY_KINDS:
            ontology = self._hypotheses[kind]
            ontology.evidence_for *= 0.97
            ontology.evidence_against *= 0.97

            if kind == "avatar_world":
                if best_player is not None and best_player.confidence > 0.35:
                    ontology.evidence_for += 0.9 + 0.4 * best_player.confidence
                else:
                    ontology.evidence_against += 0.3
                if diff is not None and diff.player_displacement is not None:
                    ontology.evidence_for += 0.8
                if move_ops:
                    ontology.evidence_for += 0.3
            elif kind == "click_world":
                if click_change:
                    ontology.evidence_for += 1.0
                elif transition is not None and transition.action.x is not None:
                    ontology.evidence_against += 0.2
                if click_ops:
                    ontology.evidence_for += 0.35
                if small_object_count > 0:
                    ontology.evidence_for += 0.15
            elif kind == "token_world":
                if object_count >= 4:
                    ontology.evidence_for += 0.3
                if removed_values:
                    ontology.evidence_for += 0.6
                if any(rule.effect.args.get("kind") == "class_exhaustion" for rule in teleology):
                    ontology.evidence_for += 0.35
            elif kind == "field_world":
                if big_change:
                    ontology.evidence_for += 0.45
                if topology_unlock:
                    ontology.evidence_for += 0.35
                if object_count <= 2:
                    ontology.evidence_for += 0.25
            elif kind == "phase_world":
                if transition is not None and transition.metadata.get("phase_shift"):
                    ontology.evidence_for += 0.75
                if obs.surprise.causal_surprise > 0.35:
                    ontology.evidence_for += 0.25
                if memory.game.progress.state.branch_id > 0:
                    ontology.evidence_for += 0.10
            elif kind == "transform_world":
                if big_change:
                    ontology.evidence_for += 0.85
                if transform_ops:
                    ontology.evidence_for += 0.45
                if click_change and big_change:
                    ontology.evidence_for += 0.20

            if transition is not None and transition.game_over:
                ontology.evidence_against += 0.08
            if transition is not None and transition.diff.is_noop:
                ontology.evidence_against += 0.04

            net = ontology.evidence_for - ontology.evidence_against
            ontology.confidence = _sigmoid(net)
            ontology.salient_object_ids = [
                obj.object_id
                for obj in sorted(obs.objects, key=lambda obj: (obj.area, -obj.value))[:12]
            ]

        self._last_ranked = sorted(
            self._hypotheses.values(),
            key=lambda item: (item.confidence, item.evidence_for),
            reverse=True,
        )
        if getattr(memory.game, "learning", None) is not None:
            self._last_ranked = memory.game.learning.ontology_calibrator.calibrate(
                self._last_ranked,
                obs,
                memory,
            )
        memory.game.record_ontology_ranking(self._last_ranked)

    def ranked(self) -> list[OntologyHypothesis]:
        return list(self._last_ranked)

    def top(self) -> OntologyHypothesis:
        return self._last_ranked[0]

    def downweight(self, ontology_id: str, amount: float = 0.4) -> None:
        ontology = self._hypotheses.get(ontology_id)
        if ontology is None:
            return
        ontology.evidence_against += amount
        ontology.confidence = _sigmoid(ontology.evidence_for - ontology.evidence_against)
        self._last_ranked = sorted(
            self._hypotheses.values(),
            key=lambda item: (item.confidence, item.evidence_for),
            reverse=True,
        )
