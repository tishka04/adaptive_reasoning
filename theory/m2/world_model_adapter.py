"""World-model adapter contract for M2."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Protocol, Tuple


@dataclass(frozen=True)
class FrontierState:
    frontier_context_id: str
    game_id: str
    replay_actions: Tuple[str, ...] = ()
    live_state_signature: str = ""
    local_patch: Tuple[Tuple[int, ...], ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frontier_context_id": self.frontier_context_id,
            "game_id": self.game_id,
            "replay_actions": list(self.replay_actions),
            "live_state_signature": self.live_state_signature,
            "local_patch": (
                [list(row) for row in self.local_patch]
                if self.local_patch is not None
                else None
            ),
        }


@dataclass(frozen=True)
class WorldModelActionPrediction:
    candidate_action: str
    predicted_change_probability: float
    predicted_local_signal_probability: float
    predicted_topology_change_probability: float
    predicted_object_count_change_probability: float
    predicted_contact_graph_change_probability: float
    predicted_observables: Mapping[str, Any] = field(default_factory=dict)
    uncertainty: float = 0.0
    epistemic_value: float = 0.0
    recommended_metric: str = "changed_pixels"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["predicted_observables"] = dict(self.predicted_observables)
        return data


class WorldModelHypothesisGenerator(Protocol):
    def score_candidate_actions(
        self,
        frontier_state: FrontierState,
        allowed_actions: list[str],
    ) -> list[WorldModelActionPrediction]:
        ...


def frontier_state_from_request(frontier_request: Mapping[str, Any]) -> FrontierState:
    return FrontierState(
        frontier_context_id=str(frontier_request.get("frontier_context_id", "")),
        game_id=str(frontier_request.get("game_id", "")),
        replay_actions=tuple(
            str(item) for item in frontier_request.get("context_signature", []) or []
        ),
        live_state_signature=str(frontier_request.get("live_state_signature", "")),
    )
