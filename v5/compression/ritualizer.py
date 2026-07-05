"""Ritualizer — compiles a successful level prefix into a Ritual.

V4's ritualizer was coupled to V4's memory/observation shapes. This V5
version takes explicit arguments and returns a ritual (no hidden state).
"""

from __future__ import annotations

from typing import Iterable, List

from ..schemas import GameObservation, PrimitiveAction
from ..schemas_ext import Ritual


def compile_ritual(
    *,
    ritual_id: str,
    ontology_kind: str,
    successful_prefix: List[PrimitiveAction],
    current_obs: GameObservation,
    levels_completed: int,
) -> Ritual:
    """Compile a successful prefix into a Ritual."""
    small_remaining = sum(
        1 for obj in current_obs.objects
        if obj.value != 0 and obj.area <= 12
    )
    signature = {
        "remaining_small_objects": int(small_remaining),
        "levels_completed": int(levels_completed),
        "n_objects": len(current_obs.objects),
    }
    return Ritual(
        ritual_id=ritual_id,
        ontology_kind=ontology_kind,
        prefix=list(successful_prefix),
        terminal_signature=signature,
        success_rate=1.0,
    )


def matches_terminal(
    ritual: Ritual,
    current_obs: GameObservation,
    tolerance: int = 1,
) -> bool:
    """Does the current observation look close to this ritual's terminal signature?"""
    sig = ritual.terminal_signature
    small_now = sum(
        1 for obj in current_obs.objects if obj.value != 0 and obj.area <= 12
    )
    expected = int(sig.get("remaining_small_objects", 0))
    return abs(small_now - expected) <= tolerance
