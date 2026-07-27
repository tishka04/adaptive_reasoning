"""Versioned audit records for future SAGE12 semantic-trajectory datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple


DATASET_FORMAT_VERSION = "sage12-semantic-trajectory-v1"


@dataclass(frozen=True)
class SemanticTrajectoryRecord:
    """One proposal/ranking/execution cycle with observed outcomes separated."""

    source_game_id: str
    branch_index: int
    step_index: int
    scene_signature: str
    subgoal: str
    proposal_ids: Tuple[str, ...]
    rejected_proposals: Tuple[str, ...]
    trajectory_option_ids: Tuple[Tuple[str, ...], ...]
    trajectory_energies: Tuple[float, ...]
    counterfactual_option_id: str
    selected_option_id: str
    selected_action_name: str
    selected_action_data: Mapping[str, Any] = field(default_factory=dict)
    applied: bool = False
    observed_effects: Tuple[str, ...] = ()
    productive: bool | None = None
    unsafe: bool | None = None
    format_version: str = DATASET_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version != DATASET_FORMAT_VERSION:
            raise ValueError("unsupported SAGE12 trajectory record version")
        if len(self.trajectory_option_ids) != len(self.trajectory_energies):
            raise ValueError("trajectory ids and energies must have equal length")

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["selected_action_data"] = _json_safe(
            self.selected_action_data
        )
        return payload

    def with_outcome(
        self,
        *,
        observed_effects: Sequence[str],
        productive: bool,
        unsafe: bool,
    ) -> "SemanticTrajectoryRecord":
        payload = self.to_dict()
        payload.update(
            {
                "observed_effects": tuple(sorted(observed_effects)),
                "productive": bool(productive),
                "unsafe": bool(unsafe),
            }
        )
        for key in (
            "proposal_ids",
            "rejected_proposals",
            "trajectory_option_ids",
        ):
            payload[key] = tuple(payload[key])
        payload["trajectory_energies"] = tuple(payload["trajectory_energies"])
        return SemanticTrajectoryRecord(**payload)


class SemanticTraceWriter:
    """Append-only JSONL writer for audited trajectory collection."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append(self, record: SemanticTrajectoryRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "DATASET_FORMAT_VERSION",
    "SemanticTraceWriter",
    "SemanticTrajectoryRecord",
]
