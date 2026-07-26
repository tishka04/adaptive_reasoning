"""Shared versioned SAGE.11 streaming features for training and live use."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    Mapping,
    Sequence,
    Tuple,
)

import numpy as np


STREAMING_FEATURE_FORMAT_VERSION = "sage11-streaming-features-v2"
ACTION_NAMES: Tuple[str, ...] = tuple(
    f"ACTION{index}"
    for index in range(1, 7)
)
CHANGED_BUCKETS: Tuple[str, ...] = (
    "zero",
    "one",
    "few",
    "some",
    "many",
)
BOOLEAN_VALUES: Tuple[str, ...] = ("False", "True")
CORE_FACTOR_HEADS: Tuple[str, ...] = (
    "changed_cells",
    "player_moved",
)
AUDIT_FACTOR_HEADS: Tuple[str, ...] = (
    "level_complete",
    "game_over",
)
FACTOR_VALUE_NAMES: Mapping[str, Tuple[str, ...]] = {
    "changed_cells": CHANGED_BUCKETS,
    "player_moved": BOOLEAN_VALUES,
    "level_complete": BOOLEAN_VALUES,
    "game_over": BOOLEAN_VALUES,
}

ACTION_FEATURE_NAMES: Tuple[str, ...] = (
    *(f"current_action:{name}" for name in ACTION_NAMES),
    "current_argument:has_xy",
    "current_argument:on_boundary",
    "current_argument:on_corner",
    "current_argument:on_diagonal",
)
ARGUMENT_FEATURE_NAMES: Tuple[str, ...] = ACTION_FEATURE_NAMES[-4:]
CONTEXT_FEATURE_NAMES: Tuple[str, ...] = (
    "context:exact_continuity",
    "reset_step:zero",
    "reset_step:one_to_three",
    "reset_step:four_to_fifteen",
    "reset_step:sixteen_plus",
    "state_visit:first",
    "state_visit:second",
    "state_visit:third_or_fourth",
    "state_visit:fifth_plus",
    "state_recency:new",
    "state_recency:one",
    "state_recency:two_to_four",
    "state_recency:five_to_sixteen",
    "state_recency:seventeen_plus",
    *(f"previous_action:{name}" for name in ACTION_NAMES),
    "previous_action:same_as_current",
    *(f"previous_changed:{bucket}" for bucket in CHANGED_BUCKETS),
    "previous_effect:player_moved",
    "previous_effect:level_complete",
    "previous_effect:game_over",
    "relative_target:has_xy",
    "relative_target:same_target",
    "relative_target:same_row",
    "relative_target:same_column",
    "relative_target:dx_negative",
    "relative_target:dx_zero",
    "relative_target:dx_positive",
    "relative_target:dy_negative",
    "relative_target:dy_zero",
    "relative_target:dy_positive",
    "relative_target:distance_zero",
    "relative_target:distance_one_to_four",
    "relative_target:distance_five_to_sixteen",
    "relative_target:distance_seventeen_plus",
    *(f"atom_delta:{bucket}" for bucket in CHANGED_BUCKETS),
)
CURRENT_ACTION_DEPENDENT_CONTEXT_NAMES: Tuple[str, ...] = (
    "previous_action:same_as_current",
    "relative_target:has_xy",
    "relative_target:same_target",
    "relative_target:same_row",
    "relative_target:same_column",
    "relative_target:dx_negative",
    "relative_target:dx_zero",
    "relative_target:dx_positive",
    "relative_target:dy_negative",
    "relative_target:dy_zero",
    "relative_target:dy_positive",
    "relative_target:distance_zero",
    "relative_target:distance_one_to_four",
    "relative_target:distance_five_to_sixteen",
    "relative_target:distance_seventeen_plus",
)


def factor_labels(effect_atoms: Sequence[str]) -> Dict[str, int]:
    """Parse the four independent effect factors from shared effect atoms."""
    values = {
        "changed_cells": _effect_value(
            effect_atoms,
            kind="effect",
            predicate="changed_cells",
        ),
        "player_moved": _effect_value(
            effect_atoms,
            kind="effect",
            predicate="player_moved",
        ),
        "level_complete": _effect_value(
            effect_atoms,
            kind="progress",
            predicate="level_complete",
        ),
        "game_over": _effect_value(
            effect_atoms,
            kind="risk",
            predicate="game_over",
        ),
    }
    return {
        head: FACTOR_VALUE_NAMES[head].index(value)
        for head, value in values.items()
    }


@dataclass(frozen=True)
class StreamingFeatureSchema:
    """Frozen feature ordering and train-only atom vocabulary."""

    atom_vocabulary: Tuple[str, ...]
    format_version: str = STREAMING_FEATURE_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version != STREAMING_FEATURE_FORMAT_VERSION:
            raise ValueError("unsupported SAGE.11 streaming feature version")
        canonical = tuple(sorted(set(self.atom_vocabulary)))
        if self.atom_vocabulary != canonical:
            raise ValueError("streaming atom vocabulary must be unique and sorted")

    @classmethod
    def fit(
        cls,
        training_atom_rows: Iterable[Sequence[str]],
    ) -> "StreamingFeatureSchema":
        atoms = {
            str(atom)
            for row in training_atom_rows
            for atom in row
        }
        return cls(atom_vocabulary=tuple(sorted(atoms)))

    @property
    def atom_feature_names(self) -> Tuple[str, ...]:
        return tuple(
            f"current_atom:{atom}"
            for atom in self.atom_vocabulary
        )

    @property
    def feature_names(self) -> Tuple[str, ...]:
        return (
            ACTION_FEATURE_NAMES
            + self.atom_feature_names
            + CONTEXT_FEATURE_NAMES
        )

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    @property
    def feature_to_index(self) -> Mapping[str, int]:
        return {
            name: index
            for index, name in enumerate(self.feature_names)
        }

    def indices_for(
        self,
        names: Sequence[str],
    ) -> Tuple[int, ...]:
        mapping = self.feature_to_index
        return tuple(mapping[name] for name in names)

    @property
    def action_feature_indices(self) -> Tuple[int, ...]:
        return self.indices_for(ACTION_FEATURE_NAMES)

    @property
    def argument_feature_indices(self) -> Tuple[int, ...]:
        return self.indices_for(ARGUMENT_FEATURE_NAMES)

    @property
    def action_dependent_feature_indices(self) -> Tuple[int, ...]:
        return self.indices_for(
            ACTION_FEATURE_NAMES
            + CURRENT_ACTION_DEPENDENT_CONTEXT_NAMES
        )

    @property
    def state_only_feature_indices(self) -> Tuple[int, ...]:
        action_dependent = set(self.action_dependent_feature_indices)
        return tuple(
            index
            for index in range(self.feature_count)
            if index not in action_dependent
        )

    @property
    def game_signature_feature_indices(self) -> Tuple[int, ...]:
        names = tuple(
            name
            for name in self.atom_feature_names
            if name.startswith("current_atom:action:available(")
            or name.startswith("current_atom:object:role_present(")
        )
        return self.indices_for(names)

    @property
    def checksum(self) -> str:
        payload = {
            "format_version": self.format_version,
            "atom_vocabulary": list(self.atom_vocabulary),
            "feature_names": list(self.feature_names),
        }
        return hashlib.sha256(json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": self.format_version,
            "atom_vocabulary": list(self.atom_vocabulary),
            "feature_names": list(self.feature_names),
            "feature_count": self.feature_count,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "StreamingFeatureSchema":
        schema = cls(
            atom_vocabulary=tuple(
                str(atom)
                for atom in payload["atom_vocabulary"]
            ),
            format_version=str(payload["format_version"]),
        )
        expected = str(payload.get("checksum", ""))
        if expected and schema.checksum != expected:
            raise ValueError("streaming feature schema checksum mismatch")
        return schema


@dataclass(frozen=True)
class _ObservedStreamingTransition:
    step_index: int
    action_name: str
    action_data: Mapping[str, Any]
    atoms_after: Tuple[str, ...]
    state_digest_after: str
    effect_atoms: Tuple[str, ...]


@dataclass(frozen=True)
class StreamingStepContext:
    """One pre-action state shared across every candidate action."""

    sequence_key: Hashable
    step_index: int
    atoms_before: Tuple[str, ...]
    state_digest_before: str
    previous_visits: int
    recency: int | None
    previous: _ObservedStreamingTransition | None
    exact_continuity: bool


class StreamingFeatureTracker:
    """Stateful encoder used identically by row loaders and live inference."""

    def __init__(self, schema: StreamingFeatureSchema) -> None:
        self.schema = schema
        self._previous: Dict[
            Hashable,
            _ObservedStreamingTransition,
        ] = {}
        self._state_visits: Dict[
            Hashable,
            Counter[str],
        ] = defaultdict(Counter)
        self._state_last_steps: Dict[
            Hashable,
            Dict[str, int],
        ] = defaultdict(dict)

    def begin_step(
        self,
        *,
        sequence_key: Hashable,
        step_index: int,
        atoms_before: Sequence[str],
        state_digest_before: str,
    ) -> StreamingStepContext:
        """Create one immutable context before candidate actions are encoded."""
        atoms = tuple(str(atom) for atom in atoms_before)
        digest = str(state_digest_before)
        step = int(step_index)
        previous = self._previous.get(sequence_key)
        exact_continuity = bool(
            previous
            and step == previous.step_index + 1
            and digest == previous.state_digest_after
        )
        previous_visits = self._state_visits[sequence_key][digest]
        last_step = self._state_last_steps[sequence_key].get(digest)
        recency = None if last_step is None else max(0, step - last_step)
        return StreamingStepContext(
            sequence_key=sequence_key,
            step_index=step,
            atoms_before=atoms,
            state_digest_before=digest,
            previous_visits=previous_visits,
            recency=recency,
            previous=previous,
            exact_continuity=exact_continuity,
        )

    def encode_action(
        self,
        context: StreamingStepContext,
        *,
        action_name: str,
        action_data: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        """Encode one counterfactual action against a shared pre-action state."""
        action = str(action_name)
        if action not in ACTION_NAMES:
            raise ValueError(f"unsupported action {action}")
        mapping = self.schema.feature_to_index
        vector = np.zeros(self.schema.feature_count, dtype=np.float32)

        vector[mapping[f"current_action:{action}"]] = 1.0
        coordinates = coordinate_pair(action_data)
        if coordinates is not None:
            x, y = coordinates
            vector[mapping["current_argument:has_xy"]] = 1.0
            on_x_boundary = x in {0, 63}
            on_y_boundary = y in {0, 63}
            vector[mapping["current_argument:on_boundary"]] = float(
                on_x_boundary or on_y_boundary
            )
            vector[mapping["current_argument:on_corner"]] = float(
                on_x_boundary and on_y_boundary
            )
            vector[mapping["current_argument:on_diagonal"]] = float(
                x == y or x + y == 63
            )

        for atom in context.atoms_before:
            atom_index = mapping.get(f"current_atom:{atom}")
            if atom_index is not None:
                vector[atom_index] = 1.0

        vector[
            mapping[f"reset_step:{_reset_step_bucket(context.step_index)}"]
        ] = 1.0
        vector[
            mapping[f"state_visit:{_visit_bucket(context.previous_visits)}"]
        ] = 1.0
        vector[
            mapping[f"state_recency:{_recency_bucket(context.recency)}"]
        ] = 1.0

        previous = context.previous
        if context.exact_continuity and previous is not None:
            vector[mapping["context:exact_continuity"]] = 1.0
            vector[
                mapping[f"previous_action:{previous.action_name}"]
            ] = 1.0
            vector[mapping["previous_action:same_as_current"]] = float(
                previous.action_name == action
            )

            previous_factors = factor_labels(previous.effect_atoms)
            previous_changed = CHANGED_BUCKETS[
                previous_factors["changed_cells"]
            ]
            vector[
                mapping[f"previous_changed:{previous_changed}"]
            ] = 1.0
            vector[mapping["previous_effect:player_moved"]] = float(
                previous_factors["player_moved"]
            )
            vector[mapping["previous_effect:level_complete"]] = float(
                previous_factors["level_complete"]
            )
            vector[mapping["previous_effect:game_over"]] = float(
                previous_factors["game_over"]
            )

            previous_coordinates = coordinate_pair(previous.action_data)
            if coordinates is not None and previous_coordinates is not None:
                x, y = coordinates
                previous_x, previous_y = previous_coordinates
                dx = x - previous_x
                dy = y - previous_y
                vector[mapping["relative_target:has_xy"]] = 1.0
                vector[mapping["relative_target:same_target"]] = float(
                    dx == 0 and dy == 0
                )
                vector[mapping["relative_target:same_row"]] = float(
                    dy == 0
                )
                vector[mapping["relative_target:same_column"]] = float(
                    dx == 0
                )
                vector[
                    mapping[f"relative_target:dx_{_direction(dx)}"]
                ] = 1.0
                vector[
                    mapping[f"relative_target:dy_{_direction(dy)}"]
                ] = 1.0
                distance = abs(dx) + abs(dy)
                vector[
                    mapping[
                        "relative_target:distance_"
                        f"{_distance_bucket(distance)}"
                    ]
                ] = 1.0

            atom_delta = _count_bucket(len(
                set(context.atoms_before).symmetric_difference(
                    previous.atoms_after
                )
            ))
            vector[mapping[f"atom_delta:{atom_delta}"]] = 1.0
        return vector

    def observe_transition(
        self,
        context: StreamingStepContext,
        *,
        action_name: str,
        action_data: Mapping[str, Any] | None,
        atoms_after: Sequence[str],
        state_digest_after: str,
        effect_atoms: Sequence[str],
    ) -> None:
        """Commit the observed outcome only after pre-action encoding."""
        key = context.sequence_key
        self._state_visits[key][context.state_digest_before] += 1
        self._state_last_steps[key][
            context.state_digest_before
        ] = context.step_index
        self._previous[key] = _ObservedStreamingTransition(
            step_index=context.step_index,
            action_name=str(action_name),
            action_data=dict(action_data or {}),
            atoms_after=tuple(str(atom) for atom in atoms_after),
            state_digest_after=str(state_digest_after),
            effect_atoms=tuple(str(atom) for atom in effect_atoms),
        )

    def reset(self, sequence_key: Hashable | None = None) -> None:
        """Clear one live/reset sequence or every tracked sequence."""
        if sequence_key is None:
            self._previous.clear()
            self._state_visits.clear()
            self._state_last_steps.clear()
            return
        self._previous.pop(sequence_key, None)
        self._state_visits.pop(sequence_key, None)
        self._state_last_steps.pop(sequence_key, None)


@dataclass(frozen=True)
class EncodedStreamingDataset:
    """Shared matrix consumed by audits and model data loading."""

    features: np.ndarray
    labels: Mapping[str, np.ndarray]
    train_mask: np.ndarray
    actions: np.ndarray
    games: np.ndarray
    schema: StreamingFeatureSchema
    exact_continuity: np.ndarray
    revisited_state: np.ndarray
    has_xy: np.ndarray
    manifest_checksum: str


def encode_transition_rows(
    row_factory: Callable[[], Iterable[Mapping[str, Any]]],
    *,
    total_rows: int,
    manifest_checksum: str,
    training_split: str = "source_train",
) -> EncodedStreamingDataset:
    """Fit train-only atoms, then encode rows through the shared tracker."""
    schema = StreamingFeatureSchema.fit(
        tuple(str(atom) for atom in row["atoms_before"])
        for row in row_factory()
        if str(row["source_split"]) == training_split
    )
    features = np.zeros(
        (int(total_rows), schema.feature_count),
        dtype=np.float32,
    )
    labels = {
        head: np.empty(int(total_rows), dtype=np.int64)
        for head in CORE_FACTOR_HEADS + AUDIT_FACTOR_HEADS
    }
    train_mask = np.empty(int(total_rows), dtype=bool)
    actions = np.empty(int(total_rows), dtype="<U16")
    games = np.empty(int(total_rows), dtype="<U16")
    exact_continuity = np.zeros(int(total_rows), dtype=bool)
    revisited_state = np.zeros(int(total_rows), dtype=bool)
    has_xy = np.zeros(int(total_rows), dtype=bool)
    tracker = StreamingFeatureTracker(schema)

    encoded_rows = 0
    for index, row in enumerate(row_factory()):
        if index >= int(total_rows):
            raise ValueError("streaming row count exceeds declaration")
        game = str(row["game_id"])
        sequence_key = (
            game,
            int(row["seed"]),
            int(row["reset_index"]),
        )
        context = tracker.begin_step(
            sequence_key=sequence_key,
            step_index=int(row["step_index"]),
            atoms_before=tuple(
                str(atom)
                for atom in row["atoms_before"]
            ),
            state_digest_before=str(row["state_digest_before"]),
        )
        action = str(row["action_name"])
        action_data = dict(row.get("action_data", {}) or {})
        features[index] = tracker.encode_action(
            context,
            action_name=action,
            action_data=action_data,
        )
        factors = factor_labels(tuple(
            str(atom)
            for atom in row["effect_atoms"]
        ))
        for head in labels:
            labels[head][index] = factors[head]
        train_mask[index] = str(row["source_split"]) == training_split
        actions[index] = action
        games[index] = game
        exact_continuity[index] = context.exact_continuity
        revisited_state[index] = context.previous_visits > 0
        has_xy[index] = coordinate_pair(action_data) is not None
        tracker.observe_transition(
            context,
            action_name=action,
            action_data=action_data,
            atoms_after=tuple(
                str(atom)
                for atom in row["atoms_after"]
            ),
            state_digest_after=str(row["state_digest_after"]),
            effect_atoms=tuple(
                str(atom)
                for atom in row["effect_atoms"]
            ),
        )
        encoded_rows = index + 1
    if encoded_rows != int(total_rows):
        raise ValueError(
            f"streaming row count {encoded_rows} != {total_rows}"
        )
    return EncodedStreamingDataset(
        features=features,
        labels=labels,
        train_mask=train_mask,
        actions=actions,
        games=games,
        schema=schema,
        exact_continuity=exact_continuity,
        revisited_state=revisited_state,
        has_xy=has_xy,
        manifest_checksum=str(manifest_checksum),
    )


def coordinate_pair(
    action_data: Mapping[str, Any] | None,
) -> Tuple[int, int] | None:
    data = dict(action_data or {})
    if "x" not in data or "y" not in data:
        return None
    try:
        return int(data["x"]), int(data["y"])
    except (TypeError, ValueError):
        return None


def _effect_value(
    atoms: Sequence[str],
    *,
    kind: str,
    predicate: str,
) -> str:
    prefix = f"{kind}:{predicate}("
    for atom in atoms:
        if atom.startswith(prefix) and atom.endswith(")"):
            return atom[len(prefix):-1]
    raise ValueError(f"missing effect atom {kind}:{predicate}")


def _count_bucket(value: int) -> str:
    count = max(0, int(value))
    if count == 0:
        return "zero"
    if count == 1:
        return "one"
    if count <= 4:
        return "few"
    if count <= 15:
        return "some"
    return "many"


def _reset_step_bucket(step_index: int) -> str:
    step = max(0, int(step_index))
    if step == 0:
        return "zero"
    if step <= 3:
        return "one_to_three"
    if step <= 15:
        return "four_to_fifteen"
    return "sixteen_plus"


def _visit_bucket(previous_visits: int) -> str:
    if previous_visits <= 0:
        return "first"
    if previous_visits == 1:
        return "second"
    if previous_visits <= 3:
        return "third_or_fourth"
    return "fifth_plus"


def _recency_bucket(distance: int | None) -> str:
    if distance is None:
        return "new"
    if distance <= 1:
        return "one"
    if distance <= 4:
        return "two_to_four"
    if distance <= 16:
        return "five_to_sixteen"
    return "seventeen_plus"


def _distance_bucket(distance: int) -> str:
    if distance <= 0:
        return "zero"
    if distance <= 4:
        return "one_to_four"
    if distance <= 16:
        return "five_to_sixteen"
    return "seventeen_plus"


def _direction(value: int) -> str:
    if value < 0:
        return "negative"
    if value > 0:
        return "positive"
    return "zero"


__all__ = [
    "ACTION_FEATURE_NAMES",
    "ACTION_NAMES",
    "ARGUMENT_FEATURE_NAMES",
    "AUDIT_FACTOR_HEADS",
    "CHANGED_BUCKETS",
    "CONTEXT_FEATURE_NAMES",
    "CORE_FACTOR_HEADS",
    "EncodedStreamingDataset",
    "STREAMING_FEATURE_FORMAT_VERSION",
    "StreamingFeatureSchema",
    "StreamingFeatureTracker",
    "StreamingStepContext",
    "coordinate_pair",
    "encode_transition_rows",
    "factor_labels",
]
