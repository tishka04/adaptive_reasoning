"""Simplified ontology competition for V5.

Operates directly on V3 schemas (GameObservation / FrameDiff / Operator).
Four seeded kinds:

  - navigator : avatar-centric, movement operators matter most
  - click     : object-selection matters, clicks change state
  - token     : objects represent counts/tokens, class-exhaustion drives progress
  - transform : interactions trigger big grid changes / toggles

The competition updates per action using cheap evidence signals, with a
soft decay so old evidence fades. It is intentionally small and
auditable: no learning sub-models, no hidden state, no LLMs.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from ..schemas import FrameDiff, GameObservation, OperatorKind
from ..schemas_ext import OntologyHypothesis


ONTOLOGY_KINDS: List[str] = ["navigator", "click", "token", "transform"]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-8.0, min(8.0, x))))


class OntologyCompetition:
    """Maintain competing interpretations of what kind of game this is."""

    def __init__(self, priors: Optional[Dict[str, float]] = None) -> None:
        self._hypotheses: Dict[str, OntologyHypothesis] = {}
        self._last_ranked: List[OntologyHypothesis] = []
        self._clamped_kind: Optional[str] = None
        priors = priors or {}
        for kind in ONTOLOGY_KINDS:
            prior = float(priors.get(kind, 0.0))
            hyp = OntologyHypothesis(
                ontology_id=kind,
                kind=kind,
                evidence_for=prior,
                confidence=_sigmoid(prior),
            )
            self._hypotheses[kind] = hyp
        self._last_ranked = list(self._hypotheses.values())

    # -----------------------------------------------------------------
    def update(
        self,
        obs: GameObservation,
        diff: Optional[FrameDiff],
        last_action_x: Optional[int],
        operator_kinds_seen: Dict[str, int],
    ) -> None:
        """Update evidence for each ontology from the latest transition.

        Args:
            obs: current observation
            diff: FrameDiff produced by the last action (None on first frame)
            last_action_x: x-coord of the last action (None for non-click actions)
            operator_kinds_seen: dict {kind_name -> induced_count} from inducer
        """
        best_player = obs.best_player
        object_count = len(obs.objects)
        small_object_count = sum(
            1 for obj in obs.objects if obj.area <= 12 and obj.value != 0
        )
        big_change = bool(diff is not None and diff.num_changed >= 8)
        click_change = bool(
            diff is not None
            and last_action_x is not None
            and diff.num_changed > 0
        )

        move_ops = int(operator_kinds_seen.get(OperatorKind.MOVE.value, 0))
        click_ops = int(operator_kinds_seen.get(OperatorKind.CLICK.value, 0))
        transform_ops = int(operator_kinds_seen.get(
            OperatorKind.GLOBAL_TRANSFORM.value, 0
        ))

        for kind, ontology in self._hypotheses.items():
            # decay
            ontology.evidence_for *= 0.97
            ontology.evidence_against *= 0.97

            if kind == "navigator":
                if best_player is not None and best_player.confidence > 0.35:
                    ontology.evidence_for += 0.9 + 0.4 * best_player.confidence
                else:
                    ontology.evidence_against += 0.3
                if diff is not None and diff.player_displacement is not None:
                    ontology.evidence_for += 0.8
                if move_ops:
                    ontology.evidence_for += 0.3
            elif kind == "click":
                if click_change:
                    ontology.evidence_for += 1.0
                elif last_action_x is not None:
                    ontology.evidence_against += 0.2
                if click_ops:
                    ontology.evidence_for += 0.35
                if small_object_count > 0:
                    ontology.evidence_for += 0.15
            elif kind == "token":
                if object_count >= 4:
                    ontology.evidence_for += 0.3
                if diff is not None and diff.removed_objects:
                    ontology.evidence_for += 0.6
                if object_count <= 2:
                    ontology.evidence_against += 0.1
            elif kind == "transform":
                if big_change:
                    ontology.evidence_for += 0.85
                if transform_ops:
                    ontology.evidence_for += 0.45
                if click_change and big_change:
                    ontology.evidence_for += 0.20

            # penalties for any ontology when action is sterile
            if diff is not None and diff.game_over:
                ontology.evidence_against += 0.08
            if diff is not None and diff.is_noop:
                ontology.evidence_against += 0.04

            net = ontology.evidence_for - ontology.evidence_against
            ontology.confidence = _sigmoid(net)

        self._last_ranked = sorted(
            self._hypotheses.values(),
            key=lambda item: (item.confidence, item.evidence_for),
            reverse=True,
        )

    # -----------------------------------------------------------------
    def ranked(self) -> List[OntologyHypothesis]:
        return list(self._last_ranked)

    def top(self) -> OntologyHypothesis:
        """Top-ranked ontology — honours the clamp if set."""
        if self._clamped_kind is not None:
            clamped = self._hypotheses.get(self._clamped_kind)
            if clamped is not None:
                return clamped
        return self._last_ranked[0] if self._last_ranked else self._hypotheses[ONTOLOGY_KINDS[0]]

    # -----------------------------------------------------------------
    # Clamp API (used by DissentController.force_ontology)
    # -----------------------------------------------------------------
    def clamp_top(self, kind: str) -> None:
        """Force `kind` to be returned as `top()`.

        This does NOT rewrite the evidence ecology — the underlying
        confidences still evolve — but queries to `top()` and
        `summary()` will report the clamped kind first.
        """
        if kind in self._hypotheses:
            self._clamped_kind = kind

    def clear_clamp(self) -> None:
        self._clamped_kind = None

    @property
    def clamped_kind(self) -> Optional[str]:
        return self._clamped_kind

    def downweight(self, ontology_id: str, amount: float = 0.4) -> None:
        """Dissent controller can call this to penalise a dominant ontology."""
        ontology = self._hypotheses.get(ontology_id)
        if ontology is None:
            return
        ontology.evidence_against += amount
        ontology.confidence = _sigmoid(
            ontology.evidence_for - ontology.evidence_against
        )
        self._last_ranked = sorted(
            self._hypotheses.values(),
            key=lambda item: (item.confidence, item.evidence_for),
            reverse=True,
        )

    def summary(self) -> List[tuple[str, float]]:
        return [(o.kind, round(o.confidence, 3)) for o in self._last_ranked]

    def export_priors(self) -> Dict[str, float]:
        """Export evidence_for values as priors for cross-game seeding."""
        return {
            kind: round(float(o.evidence_for), 3)
            for kind, o in self._hypotheses.items()
        }
