"""Lightweight bissociation probe for V5.

Not a full subsystem — just a cheap hybrid-proposal generator that
forces cross-ontology exploration when the current frame stagnates.

Policy:
  - Gate: at most one probe every `min_gap` actions (default 25).
  - Guard: only probe when SP has been flat for `stagnation_steps` and
    the current top ontology has monopolised recent steps.
  - Source: pick one distant digest from cross-game memory whose
    ontology kind differs from ours and whose `confirmed_operator_ids`
    or `useful_primitives` is non-empty.
  - Effect: emit a short hybrid probe (1–2 primitives) that pairs one
    action from the distant digest with one click/transform natural to
    the current ontology.

The probe never replaces the main control loop; it's a once-per-gap
injection that may surface a useful cross-frame action.
"""

from __future__ import annotations

import random
from typing import Any, List, Optional

from ..schemas import PrimitiveAction


class BissociationLite:
    def __init__(
        self,
        *,
        min_gap: int = 25,
        stagnation_steps: int = 40,
        max_probe_len: int = 2,
        rng_seed: int = 0,
    ) -> None:
        self._min_gap = int(min_gap)
        self._stagnation_steps = int(stagnation_steps)
        self._max_probe_len = int(max_probe_len)
        self._last_probe_action: int = -999
        self._last_sp: float = 0.0
        self._last_sp_improve_action: int = 0
        self._rng = random.Random(rng_seed)

    # -----------------------------------------------------------------
    def observe(self, *, action_counter: int, sp: float) -> None:
        """Call every step to track SP stagnation."""
        if sp > self._last_sp + 0.02:
            self._last_sp = sp
            self._last_sp_improve_action = action_counter

    # -----------------------------------------------------------------
    def should_probe(
        self,
        *,
        action_counter: int,
        top_ontology_kind: str,
        ontology_monoculture: bool,
    ) -> bool:
        if action_counter - self._last_probe_action < self._min_gap:
            return False
        stagnant_for = action_counter - self._last_sp_improve_action
        if stagnant_for < self._stagnation_steps:
            return False
        if not ontology_monoculture:
            return False
        return True

    # -----------------------------------------------------------------
    def build_probe(
        self,
        *,
        action_counter: int,
        top_ontology_kind: str,
        cross_game_digests: dict[str, dict[str, Any]],
        available_actions: List[str],
        objects: List[Any],
    ) -> Optional[List[PrimitiveAction]]:
        """Return a short probe list or None if nothing useful to try.

        The probe is a list of 1–2 PrimitiveActions.
        """
        distant_prim = self._pick_distant_primitive(
            top_ontology_kind, cross_game_digests, available_actions,
        )
        current_prim = self._pick_current_frame_primitive(
            top_ontology_kind, available_actions, objects,
        )
        probe: List[PrimitiveAction] = []
        if distant_prim is not None:
            probe.append(distant_prim)
        if current_prim is not None and (
            not probe or current_prim.name != probe[0].name
        ):
            probe.append(current_prim)
        if not probe:
            return None
        probe = probe[: self._max_probe_len]
        self._last_probe_action = action_counter
        return probe

    # -----------------------------------------------------------------
    def _pick_distant_primitive(
        self,
        current_kind: str,
        digests: dict[str, dict[str, Any]],
        available: List[str],
    ) -> Optional[PrimitiveAction]:
        """Pick a primitive from a distant-ontology digest."""
        if not digests:
            return None
        candidates = [
            d for d in digests.values()
            if str(d.get("ontology", "unknown")) != current_kind
            and (d.get("confirmed_operator_ids") or d.get("useful_primitives"))
        ]
        if not candidates:
            return None
        # Prefer won digests, then higher-TP
        candidates.sort(
            key=lambda d: (
                bool(d.get("won", False)),
                float(d.get("tp", 0.0) or 0.0),
                float(d.get("sp", 0.0) or 0.0),
            ),
            reverse=True,
        )
        for digest in candidates[:3]:
            prims = [
                p for p in (digest.get("useful_primitives") or [])
                if p in available and p != "RESET"
            ]
            if prims:
                return PrimitiveAction(self._rng.choice(prims))
        return None

    def _pick_current_frame_primitive(
        self,
        kind: str,
        available: List[str],
        objects: List[Any],
    ) -> Optional[PrimitiveAction]:
        """Pick a primitive natural to the current ontology."""
        non_reset = [a for a in available if a != "RESET"]
        if not non_reset:
            return None
        if kind == "click" and "ACTION6" in non_reset and objects:
            target = self._rng.choice(objects[: min(len(objects), 4)])
            # Objects from V3 have .center = (row, col)
            y = int(round(target.center[0]))
            x = int(round(target.center[1]))
            return PrimitiveAction("ACTION6", x=x, y=y)
        if kind == "transform" and len(non_reset) >= 1:
            return PrimitiveAction(self._rng.choice(non_reset))
        if kind == "navigator":
            # Pick a non-click movement action
            moves = [a for a in non_reset if a != "ACTION6"]
            if moves:
                return PrimitiveAction(self._rng.choice(moves))
        # Default: random available
        return PrimitiveAction(self._rng.choice(non_reset))
