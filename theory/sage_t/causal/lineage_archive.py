"""Lineage-preserving variant of the symbolic Go-Explore archive.

T12.3c isolates one change: transitions observed inside an excursion are attached
to the prefix that was actually executed.  The legacy archive instead rebases a
transition onto the shortest representative already stored for the same visible
state hash, which can manufacture a prefix that was never executed end-to-end.
"""

from __future__ import annotations

from collections.abc import Sequence

from .archive import (
    ArchiveEdge,
    GoExploreArchive,
    _action_payload,
    _digest,
)
from .contracts import GroundedAction
from theory.sage_t.contracts import AbstractState


class LineagePreservingArchive(GoExploreArchive):
    """Go-Explore archive that never rebases an observed transition."""

    def __init__(self, *, maximum_cells: int = 50_000, seed: int = 0) -> None:
        super().__init__(maximum_cells=maximum_cells, seed=seed)
        self.lineage_attached_transitions = 0
        self.shortest_prefix_rebases_avoided = 0

    def add_lineage_transition(
        self,
        *,
        source_cell_id: str,
        source_exact_hash: str,
        source_prefix_id: str,
        source_path_edge_ids: Sequence[str],
        action: GroundedAction,
        target_state: AbstractState,
        target_exact_hash: str,
        target_level: int,
        target_legal_actions: Sequence[GroundedAction],
        terminal: bool,
        success: bool,
        changed: bool,
    ) -> ArchiveEdge:
        source = self.cells[source_cell_id]
        representative = source.variants.get(source_exact_hash)
        if representative is None:
            raise KeyError(f"unknown exact source variant: {source_exact_hash}")
        actual_path = tuple(str(value) for value in source_path_edge_ids)
        actual_prefix = str(source_prefix_id)
        if self.prefixes.depth(actual_prefix) != len(actual_path):
            raise ValueError("executed prefix depth/path lineage mismatch")
        if actual_path:
            previous = self.edges.get(actual_path[-1])
            if previous is None or previous.target_exact_hash != source_exact_hash:
                raise ValueError("executed lineage does not end at the source hash")
        if representative.prefix_id != actual_prefix:
            self.shortest_prefix_rebases_avoided += 1

        prefix_id = self.prefixes.extend(actual_prefix, action)
        target_cell_id = self.cell_key(
            target_state,
            level=target_level,
            legal_actions=target_legal_actions,
        )
        novel = target_cell_id not in self.cells
        ordinal = self._edge_observations
        edge_id = "edge_" + _digest(
            {
                "source": source_cell_id,
                "source_exact": source_exact_hash,
                "source_prefix": actual_prefix,
                "action": _action_payload(action),
                "target_exact": target_exact_hash,
                "ordinal": ordinal,
            }
        )
        self._edge_observations += 1
        edge = ArchiveEdge(
            edge_id=edge_id,
            ordinal=ordinal,
            source_cell_id=source_cell_id,
            source_exact_hash=source_exact_hash,
            action=action,
            target_cell_id=target_cell_id,
            target_exact_hash=str(target_exact_hash),
            level_delta=max(0, int(target_level) - int(source.level)),
            terminal=bool(terminal),
            success=bool(success),
            changed=bool(changed),
            novel=bool(novel),
            prefix_id=prefix_id,
        )
        self.edges[edge.edge_id] = edge
        source.expansions += 1
        source.action_attempts[action.key] = source.action_attempts.get(action.key, 0) + 1
        self.observe_state(
            state=target_state,
            exact_hash=target_exact_hash,
            level=target_level,
            legal_actions=target_legal_actions,
            prefix_id=prefix_id,
            path_edge_ids=actual_path + (edge.edge_id,),
            terminal=terminal,
        )
        self.lineage_attached_transitions += 1
        return edge

    def metrics(self) -> dict[str, object]:
        return {
            **super().metrics(),
            "lineage_attached_transitions": self.lineage_attached_transitions,
            "shortest_prefix_rebases_avoided": self.shortest_prefix_rebases_avoided,
            "lineage_rebased_transitions": 0,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **super().to_dict(),
            "lineage_metrics": {
                "lineage_attached_transitions": self.lineage_attached_transitions,
                "shortest_prefix_rebases_avoided": self.shortest_prefix_rebases_avoided,
                "lineage_rebased_transitions": 0,
            },
        }


__all__ = ["LineagePreservingArchive"]
