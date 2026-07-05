"""Ritual compilation for V4."""

from __future__ import annotations

from ..schemas import Ritual


class Ritualizer:
    """Compile short, ontology-tagged successful prefixes."""

    def compile(self, memory, successful_prefix) -> None:
        if not successful_prefix:
            return
        top = memory.game.current_ontologies[0] if memory.game.current_ontologies else None
        ontology_kind = top.kind if top is not None else "token_world"
        signature = {
            "remaining_small_objects": sum(
                1
                for obj in (memory.fast.current_obs.objects if memory.fast.current_obs else [])
                if obj.value != 0 and obj.area <= 12
            ),
            "levels_completed": memory.game.total_levels_completed,
        }
        ritual = Ritual(
            ritual_id=f"ritual_{ontology_kind}_{len(memory.game.rituals)}",
            ontology_kind=ontology_kind,
            prefix=successful_prefix[:],
            terminal_signature=signature,
            success_rate=1.0,
        )
        memory.game.add_ritual(ritual)
