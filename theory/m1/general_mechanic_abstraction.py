"""M1.G0 general game-mechanic abstraction candidates.

This module emits candidate-only entity and role hypotheses from grid histories.
It deliberately avoids bp35-specific mechanics: no gravity, no orange box, no
agent truth. Every output remains unresolved material for later M3 tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from theory.m1.polymorphic_a25_adapter import _step_env_action
from theory.m2.m3_execution_smoke import _make_env, _reset_env
from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir, _valid_actions
from theory.p1.bp35_sage_candidate_policy_probe import DEFAULT_GAME_ID
from theory.real_env_option_adapter import snapshot_frame


DEFAULT_GENERAL_MECHANIC_CANDIDATES_OUTPUT_PATH = (
    Path("diagnostics") / "m1" / "general_mechanic_candidates.json"
)
DEFAULT_ACTION_SWEEP_SEQUENCE = ("ACTION3", "ACTION4", "ACTION1", "ACTION2", "ACTION6")
SCHEMA_VERSION = "m1.general_mechanic_candidates.v1"
TRUTH_STATUS = "NOT_EVALUATED_BY_M1"
REVISION_STATUS = "CANDIDATE_ONLY"
STATUS = "UNRESOLVED"


@dataclass(frozen=True)
class ComponentObservation:
    frame_index: int
    component_id: int
    color: int
    size: int
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    shape_signature: str
    edge_touching: bool
    edge_locations: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        row = asdict(self)
        row["bbox"] = [int(value) for value in self.bbox]
        row["centroid"] = [round(float(value), 4) for value in self.centroid]
        row["edge_locations"] = list(self.edge_locations)
        return row


@dataclass(frozen=True)
class EntityTrack:
    entity_id: str
    observations: Tuple[ComponentObservation, ...]
    total_frames: int
    action_history: Tuple[str, ...] = ()

    @property
    def color(self) -> int:
        colors = [obs.color for obs in self.observations]
        return int(Counter(colors).most_common(1)[0][0]) if colors else -1

    @property
    def first_frame(self) -> int:
        return min((obs.frame_index for obs in self.observations), default=0)

    @property
    def last_frame(self) -> int:
        return max((obs.frame_index for obs in self.observations), default=0)

    def summary(self) -> Dict[str, Any]:
        observations = sorted(self.observations, key=lambda obs: obs.frame_index)
        centroids = [obs.centroid for obs in observations]
        sizes = [obs.size for obs in observations]
        shapes = [obs.shape_signature for obs in observations]
        color_values = [obs.color for obs in observations]
        edge_values = [obs.edge_touching for obs in observations]
        frames_seen = len(observations)
        movement_distances = [
            _distance(centroids[index - 1], centroids[index])
            for index in range(1, len(centroids))
        ]
        changed_frames = _changed_observation_frames(observations)
        action_counts = Counter(
            self.action_history[frame - 1]
            for frame in changed_frames
            if 0 < frame <= len(self.action_history)
        )
        dominant_action = action_counts.most_common(1)[0][0] if action_counts else None
        action_correlation_score = (
            action_counts.most_common(1)[0][1] / max(1, sum(action_counts.values()))
            if action_counts
            else 0.0
        )
        size_monotonicity = _monotonicity_score(sizes)
        appears_after_action = (
            self.action_history[self.first_frame - 1]
            if 0 < self.first_frame <= len(self.action_history)
            else None
        )
        disappears_after_action = (
            self.action_history[self.last_frame]
            if self.last_frame < self.total_frames - 1
            and self.last_frame < len(self.action_history)
            else None
        )
        return {
            "entity_id": self.entity_id,
            "color": self.color,
            "frames_seen": int(frames_seen),
            "total_frames": int(self.total_frames),
            "first_frame": int(self.first_frame),
            "last_frame": int(self.last_frame),
            "persistence_score": round(frames_seen / max(1, self.total_frames), 6),
            "shape_stability_score": round(_mode_rate(shapes), 6),
            "color_stability_score": round(_mode_rate(color_values), 6),
            "motion_score": round(min(1.0, _mean(movement_distances)), 6),
            "mean_centroid_step_distance": round(_mean(movement_distances), 6),
            "edge_contact_score": round(
                sum(1 for value in edge_values if value) / max(1, frames_seen),
                6,
            ),
            "size_monotonicity_score": round(size_monotonicity, 6),
            "size_sequence": [int(value) for value in sizes],
            "bbox_sequence": [
                [int(value) for value in obs.bbox] for obs in observations
            ],
            "centroid_sequence": [
                [round(float(value), 4) for value in obs.centroid]
                for obs in observations
            ],
            "shape_signature_sequence": list(shapes),
            "appears_after_action": appears_after_action,
            "disappears_after_action": disappears_after_action,
            "action_correlation_score": round(float(action_correlation_score), 6),
            "dominant_change_action": dominant_action,
            "changed_frames": [int(frame) for frame in changed_frames],
            "observation_counted_as_confirmation": False,
            "support": 0,
            "revision_status": REVISION_STATUS,
            "truth_status": TRUTH_STATUS,
        }


@dataclass(frozen=True)
class RoleHypothesis:
    entity_id: str
    role: str
    score: float
    evidence: Tuple[str, ...] = ()
    support: int = 0
    revision_status: str = REVISION_STATUS
    truth_status: str = TRUTH_STATUS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "role": self.role,
            "score": round(float(self.score), 6),
            "evidence": list(self.evidence),
            "support": int(self.support),
            "revision_status": self.revision_status,
            "truth_status": self.truth_status,
            "role_counted_as_confirmation": False,
        }


@dataclass(frozen=True)
class ActionEffectAbstraction:
    action: str
    transitions_observed: int
    affected_entities: Tuple[str, ...] = ()
    effect_families: Tuple[str, ...] = ()
    effect_family_counts: Tuple[Tuple[str, int], ...] = ()
    entity_delta_summary: Tuple[Dict[str, Any], ...] = ()
    relation_delta_summary: Tuple[Dict[str, Any], ...] = ()
    hud_or_latent_delta_summary: Tuple[Dict[str, Any], ...] = ()
    global_delta_summary: Dict[str, Any] = field(default_factory=dict)
    candidate_preconditions: Tuple[str, ...] = ()
    support: int = 0
    revision_status: str = REVISION_STATUS
    truth_status: str = TRUTH_STATUS
    effect_score_counted_as_support: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "transitions_observed": int(self.transitions_observed),
            "affected_entities": list(self.affected_entities),
            "effect_families": list(self.effect_families),
            "effect_family_counts": [
                {"family": family, "count": int(count)}
                for family, count in self.effect_family_counts
            ],
            "entity_delta_summary": [_json_ready(dict(row)) for row in self.entity_delta_summary],
            "relation_delta_summary": [_json_ready(dict(row)) for row in self.relation_delta_summary],
            "hud_or_latent_delta_summary": [
                _json_ready(dict(row)) for row in self.hud_or_latent_delta_summary
            ],
            "global_delta_summary": _json_ready(dict(self.global_delta_summary)),
            "candidate_preconditions": list(self.candidate_preconditions),
            "support": int(self.support),
            "revision_status": self.revision_status,
            "truth_status": self.truth_status,
            "effect_score_counted_as_support": self.effect_score_counted_as_support,
            "effect_counted_as_confirmation": False,
        }


@dataclass(frozen=True)
class RelationEdge:
    frame_index: int
    source_entity: str
    target_entity: str
    relation: str
    value: float | str | bool
    source_roles: Tuple[str, ...] = ()
    target_roles: Tuple[str, ...] = ()

    def key(self) -> Tuple[str, str, str]:
        return (self.source_entity, self.target_entity, self.relation)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": int(self.frame_index),
            "source_entity": self.source_entity,
            "target_entity": self.target_entity,
            "relation": self.relation,
            "value": _json_ready(self.value),
            "source_roles": list(self.source_roles),
            "target_roles": list(self.target_roles),
            "support": 0,
            "revision_status": REVISION_STATUS,
            "truth_status": TRUTH_STATUS,
        }


@dataclass(frozen=True)
class RelationDelta:
    action: str
    frame_index: int
    source_entity: str
    target_entity: str
    relation: str
    relation_before: str | None
    relation_after: str | None
    relation_delta_type: str
    source_roles: Tuple[str, ...] = ()
    target_roles: Tuple[str, ...] = ()
    numeric_delta: float | None = None
    support: int = 0
    revision_status: str = REVISION_STATUS
    truth_status: str = TRUTH_STATUS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "frame_index": int(self.frame_index),
            "source_entity": self.source_entity,
            "target_entity": self.target_entity,
            "relation": self.relation,
            "relation_before": self.relation_before,
            "relation_after": self.relation_after,
            "relation_delta_type": self.relation_delta_type,
            "source_roles": list(self.source_roles),
            "target_roles": list(self.target_roles),
            "numeric_delta": self.numeric_delta,
            "support": int(self.support),
            "revision_status": self.revision_status,
            "truth_status": self.truth_status,
            "relation_delta_counted_as_confirmation": False,
        }


@dataclass(frozen=True)
class DynamicInvariantCandidate:
    invariant_id: str
    invariant_family: str
    source: str
    entity_id: str | None = None
    affected_entities: Tuple[str, ...] = ()
    affected_colors: Tuple[int, ...] = ()
    direction_candidate: str | None = None
    monotonicity_score: float = 0.0
    action_correlation_score: float = 0.0
    irreversibility_score: float = 0.0
    periodicity_score: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    policy_relevance: str = ""
    support: int = 0
    revision_status: str = REVISION_STATUS
    truth_status: str = TRUTH_STATUS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "invariant_id": self.invariant_id,
            "invariant_family": self.invariant_family,
            "source": self.source,
            "entity_id": self.entity_id,
            "affected_entities": list(self.affected_entities),
            "affected_colors": [int(value) for value in self.affected_colors],
            "direction_candidate": self.direction_candidate,
            "monotonicity_score": round(float(self.monotonicity_score), 6),
            "action_correlation_score": round(float(self.action_correlation_score), 6),
            "irreversibility_score": round(float(self.irreversibility_score), 6),
            "periodicity_score": round(float(self.periodicity_score), 6),
            "evidence": _json_ready(dict(self.evidence)),
            "policy_relevance": self.policy_relevance,
            "support": int(self.support),
            "revision_status": self.revision_status,
            "truth_status": self.truth_status,
            "invariant_score_counted_as_support": False,
            "invariant_counted_as_confirmation": False,
        }


@dataclass(frozen=True)
class MechanicHypothesis:
    mechanic_hypothesis_id: str
    mechanic_family: str
    entities: Tuple[str, ...] = ()
    actions: Tuple[str, ...] = ()
    relations: Tuple[str, ...] = ()
    latent_variables: Tuple[str, ...] = ()
    preconditions: Tuple[str, ...] = ()
    predicted_effects: Tuple[str, ...] = ()
    test_suggestions: Tuple[str, ...] = ()
    confidence: float = 0.0
    status: str = STATUS
    support: int = 0
    revision_status: str = REVISION_STATUS
    truth_status: str = TRUTH_STATUS
    controlled_test_required: bool = True
    revision_performed: bool = False
    wrong_confirmations: int = 0
    trace_support_counted_as_proof: bool = False
    prior_counted_as_proof: bool = False
    confidence_counted_as_support: bool = False

    def to_dict(self) -> Dict[str, Any]:
        row = asdict(self)
        for key in (
            "entities",
            "actions",
            "relations",
            "latent_variables",
            "preconditions",
            "predicted_effects",
            "test_suggestions",
        ):
            row[key] = list(row[key])
        row["confidence"] = round(float(self.confidence), 6)
        row["support"] = int(self.support)
        return row


def extract_components(
    grid: Any,
    *,
    background: int | None = None,
    frame_index: int = 0,
) -> Tuple[ComponentObservation, ...]:
    array = _as_grid(grid)
    if array is None:
        return ()
    height, width = array.shape
    seen = np.zeros((height, width), dtype=bool)
    components: List[ComponentObservation] = []
    next_id = 0
    for y in range(height):
        for x in range(width):
            color = int(array[y, x])
            if seen[y, x] or (background is not None and color == background):
                continue
            stack = [(y, x)]
            points: List[Tuple[int, int]] = []
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                points.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and not seen[ny, nx]
                        and int(array[ny, nx]) == color
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            ys = [point[0] for point in points]
            xs = [point[1] for point in points]
            bbox = (min(ys), min(xs), max(ys), max(xs))
            centroid = (
                sum(ys) / max(1, len(points)),
                sum(xs) / max(1, len(points)),
            )
            components.append(
                ComponentObservation(
                    frame_index=int(frame_index),
                    component_id=next_id,
                    color=color,
                    size=len(points),
                    bbox=bbox,
                    centroid=centroid,
                    shape_signature=_shape_signature(points),
                    edge_touching=bool(
                        bbox[0] == 0
                        or bbox[1] == 0
                        or bbox[2] == height - 1
                        or bbox[3] == width - 1
                    ),
                    edge_locations=_edge_locations(bbox, array.shape),
                )
            )
            next_id += 1
    return tuple(components)


def track_entities(
    grid_history: Sequence[Any],
    action_history: Sequence[str] = (),
    *,
    background: int | None = None,
    max_match_score: float = 4.5,
) -> Tuple[EntityTrack, ...]:
    arrays = [grid for grid in (_as_grid(grid) for grid in grid_history) if grid is not None]
    if not arrays:
        return ()
    background = infer_background(arrays) if background is None else int(background)
    components_by_frame = [
        extract_components(grid, background=background, frame_index=index)
        for index, grid in enumerate(arrays)
    ]
    tracks: Dict[str, List[ComponentObservation]] = {}
    last_track_component: Dict[str, ComponentObservation] = {}
    next_entity_id = 1
    for frame_components in components_by_frame:
        unmatched_tracks = set(last_track_component)
        matched_tracks: set[str] = set()
        for component in sorted(
            frame_components,
            key=lambda item: (item.color, item.bbox, item.size),
        ):
            best_track = None
            best_score = float("inf")
            for track_id in list(unmatched_tracks):
                if track_id in matched_tracks:
                    continue
                previous = last_track_component[track_id]
                if previous.frame_index != component.frame_index - 1:
                    continue
                score = _component_match_score(previous, component)
                if score < best_score:
                    best_score = score
                    best_track = track_id
            if best_track is not None and best_score <= max_match_score:
                tracks[best_track].append(component)
                last_track_component[best_track] = component
                matched_tracks.add(best_track)
                unmatched_tracks.discard(best_track)
            else:
                track_id = f"E{next_entity_id:03d}"
                next_entity_id += 1
                tracks[track_id] = [component]
                last_track_component[track_id] = component
                matched_tracks.add(track_id)
    return tuple(
        EntityTrack(
            entity_id=track_id,
            observations=tuple(observations),
            total_frames=len(arrays),
            action_history=tuple(str(action) for action in action_history),
        )
        for track_id, observations in sorted(tracks.items())
    )


def generate_role_hypotheses(track: EntityTrack) -> Tuple[RoleHypothesis, ...]:
    summary = track.summary()
    persistence = float(summary["persistence_score"])
    motion = float(summary["motion_score"])
    shape_stability = float(summary["shape_stability_score"])
    edge_score = float(summary["edge_contact_score"])
    monotone_size = float(summary["size_monotonicity_score"])
    action_corr = float(summary["action_correlation_score"])
    roles: List[RoleHypothesis] = []

    if edge_score >= 0.8 and persistence >= 0.5 and monotone_size >= 0.8:
        roles.append(
            RoleHypothesis(
                entity_id=track.entity_id,
                role="timer_or_hud",
                score=min(1.0, 0.35 * edge_score + 0.3 * monotone_size + 0.2 * persistence + 0.15),
                evidence=(
                    "edge_aligned_entity",
                    "monotone_size_sequence",
                    "persistent_entity",
                ),
            )
        )

    if persistence >= 0.4 and motion > 0.0 and edge_score < 0.85:
        evidence = ["centroid_changes", "persistent_entity"]
        if action_corr > 0:
            evidence.append("action_correlated_change")
        roles.append(
            RoleHypothesis(
                entity_id=track.entity_id,
                role="controllable_actor",
                score=min(
                    1.0,
                    0.3 * persistence + 0.35 * motion + 0.25 * action_corr + 0.1 * (1.0 - edge_score),
                ),
                evidence=tuple(evidence),
            )
        )
        roles.append(
            RoleHypothesis(
                entity_id=track.entity_id,
                role="moving_object",
                score=min(1.0, 0.45 * motion + 0.25 * persistence + 0.15 * shape_stability),
                evidence=("centroid_changes",),
            )
        )

    if persistence >= 0.5 and motion <= 0.05 and edge_score >= 0.5:
        roles.append(
            RoleHypothesis(
                entity_id=track.entity_id,
                role="boundary",
                score=min(1.0, 0.45 * edge_score + 0.35 * persistence + 0.2 * shape_stability),
                evidence=("edge_contact", "low_motion", "persistent_entity"),
            )
        )

    if persistence >= 0.5 and motion <= 0.1 and edge_score < 0.8:
        roles.append(
            RoleHypothesis(
                entity_id=track.entity_id,
                role="passive_object",
                score=min(1.0, 0.4 * persistence + 0.35 * shape_stability + 0.15 * (1.0 - motion)),
                evidence=("persistent_entity", "low_motion"),
            )
        )

    if shape_stability < 0.9 or _has_size_variation(summary.get("size_sequence", [])):
        roles.append(
            RoleHypothesis(
                entity_id=track.entity_id,
                role="transformed_object",
                score=min(1.0, 0.35 * (1.0 - shape_stability) + 0.35 * monotone_size + 0.15),
                evidence=("shape_or_size_changes",),
            )
        )

    if summary.get("appears_after_action") is not None:
        roles.append(
            RoleHypothesis(
                entity_id=track.entity_id,
                role="created_object",
                score=min(1.0, 0.45 + 0.25 * persistence),
                evidence=("appears_after_action",),
            )
        )
    if summary.get("disappears_after_action") is not None:
        roles.append(
            RoleHypothesis(
                entity_id=track.entity_id,
                role="target_candidate",
                score=min(1.0, 0.35 + 0.2 * persistence),
                evidence=("disappears_after_action",),
            )
        )

    if not roles:
        roles.append(
            RoleHypothesis(
                entity_id=track.entity_id,
                role="unknown",
                score=0.1,
                evidence=("insufficient_role_signal",),
            )
        )
    return tuple(sorted(roles, key=lambda role: (-role.score, role.role)))


def emit_entity_role_mechanic_hypotheses(
    role_hypotheses: Sequence[RoleHypothesis],
) -> Tuple[MechanicHypothesis, ...]:
    hypotheses: List[MechanicHypothesis] = []
    for role in role_hypotheses:
        if role.role == "unknown":
            continue
        hypotheses.append(
            MechanicHypothesis(
                mechanic_hypothesis_id=(
                    f"m1g0::entity_role::{role.entity_id}::{role.role}"
                ),
                mechanic_family="entity_role",
                entities=(role.entity_id,),
                predicted_effects=(
                    f"{role.entity_id} may play role {role.role} in transitions",
                ),
                test_suggestions=(
                    _role_test_suggestion(role.role, role.entity_id),
                ),
                confidence=float(role.score),
            )
        )
    return tuple(hypotheses)


def infer_action_effect_abstractions(
    *,
    grid_history: Sequence[Any],
    action_history: Sequence[str],
    tracks: Sequence[EntityTrack],
    role_hypotheses: Sequence[RoleHypothesis] = (),
) -> Tuple[ActionEffectAbstraction, ...]:
    arrays = [grid for grid in (_as_grid(grid) for grid in grid_history) if grid is not None]
    if len(arrays) < 2 or not action_history:
        return ()
    roles_by_entity: Dict[str, set[str]] = {}
    for role in role_hypotheses:
        roles_by_entity.setdefault(role.entity_id, set()).add(role.role)
    observations_by_entity_frame: Dict[Tuple[str, int], ComponentObservation] = {}
    for track in tracks:
        for obs in track.observations:
            observations_by_entity_frame[(track.entity_id, obs.frame_index)] = obs

    transitions_by_action: Dict[str, List[Dict[str, Any]]] = {}
    for frame_index, action in enumerate(action_history[: len(arrays) - 1]):
        before = arrays[frame_index]
        after = arrays[frame_index + 1]
        transition = _action_transition_delta(
            action=str(action),
            frame_index=frame_index,
            before=before,
            after=after,
            tracks=tracks,
            observations_by_entity_frame=observations_by_entity_frame,
            roles_by_entity=roles_by_entity,
        )
        transitions_by_action.setdefault(str(action), []).append(transition)

    return tuple(
        _aggregate_action_effect(action, rows)
        for action, rows in sorted(transitions_by_action.items())
    )


def build_relation_graphs_by_frame(
    tracks: Sequence[EntityTrack],
    role_hypotheses: Sequence[RoleHypothesis] = (),
    *,
    max_entities_per_frame: int = 16,
    near_distance_threshold: float = 3.0,
) -> Tuple[Dict[str, Any], ...]:
    roles_by_entity = _roles_by_entity(role_hypotheses)
    frames = sorted(
        {
            obs.frame_index
            for track in tracks
            for obs in track.observations
        }
    )
    graphs: List[Dict[str, Any]] = []
    for frame_index in frames:
        entities = [
            (track.entity_id, obs)
            for track in tracks
            for obs in track.observations
            if obs.frame_index == frame_index
        ]
        entities = sorted(
            entities,
            key=lambda item: (
                _entity_priority(item[0], item[1], roles_by_entity),
                item[0],
            ),
        )[: max(1, int(max_entities_per_frame))]
        edges: List[RelationEdge] = []
        for index, (source_id, source_obs) in enumerate(entities):
            edges.extend(
                _unary_relation_edges(
                    frame_index=frame_index,
                    entity_id=source_id,
                    obs=source_obs,
                    roles=roles_by_entity.get(source_id, set()),
                )
            )
            for target_id, target_obs in entities[index + 1 :]:
                edges.extend(
                    _pair_relation_edges(
                        frame_index=frame_index,
                        left_id=source_id,
                        left=source_obs,
                        right_id=target_id,
                        right=target_obs,
                        roles_by_entity=roles_by_entity,
                        near_distance_threshold=near_distance_threshold,
                    )
                )
        graphs.append(
            {
                "frame_index": int(frame_index),
                "entities_considered": [entity_id for entity_id, _ in entities],
                "relations": [edge.to_dict() for edge in edges],
                "support": 0,
                "revision_status": REVISION_STATUS,
                "truth_status": TRUTH_STATUS,
            }
        )
    return tuple(graphs)


def infer_relation_deltas(
    relation_graphs_by_frame: Sequence[Mapping[str, Any]],
    action_history: Sequence[str],
) -> Tuple[RelationDelta, ...]:
    edges_by_frame: Dict[int, Dict[Tuple[str, str, str], Mapping[str, Any]]] = {}
    for graph in relation_graphs_by_frame:
        frame = int(graph.get("frame_index", 0) or 0)
        edges_by_frame[frame] = {
            (
                str(edge.get("source_entity", "")),
                str(edge.get("target_entity", "")),
                str(edge.get("relation", "")),
            ): edge
            for edge in graph.get("relations", []) or []
            if isinstance(edge, Mapping)
        }
    deltas: List[RelationDelta] = []
    for frame_index, action in enumerate(action_history):
        before = edges_by_frame.get(frame_index, {})
        after = edges_by_frame.get(frame_index + 1, {})
        keys = sorted(set(before) | set(after))
        for key in keys:
            before_edge = before.get(key)
            after_edge = after.get(key)
            delta = _relation_delta_from_edges(
                action=str(action),
                frame_index=frame_index,
                key=key,
                before_edge=before_edge,
                after_edge=after_edge,
            )
            if delta is not None:
                deltas.append(delta)
    return tuple(deltas)


def emit_relation_change_mechanic_hypotheses(
    relation_deltas: Sequence[RelationDelta],
    *,
    max_hypotheses: int = 512,
) -> Tuple[MechanicHypothesis, ...]:
    grouped: Dict[Tuple[str, str, str, str, str], List[RelationDelta]] = {}
    for delta in relation_deltas:
        if not _relation_delta_is_hypothesis_worthy(delta):
            continue
        key = (
            delta.action,
            delta.source_entity,
            delta.target_entity,
            delta.relation,
            delta.relation_delta_type,
        )
        grouped.setdefault(key, []).append(delta)
    hypotheses: List[MechanicHypothesis] = []
    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (
            -_relation_delta_priority(item[1][0]),
            -len(item[1]),
            item[0],
        ),
    )
    for (action, source, target, relation, delta_type), rows in ordered_groups[: max(1, int(max_hypotheses))]:
        source_roles = tuple(sorted({role for row in rows for role in row.source_roles}))
        target_roles = tuple(sorted({role for row in rows for role in row.target_roles}))
        confidence = min(1.0, len(rows) / max(1, len(relation_deltas)))
        hypotheses.append(
            MechanicHypothesis(
                mechanic_hypothesis_id=(
                    f"m1g0::relation_change::{action}::{source}::{target}::{relation}::{delta_type}"
                ),
                mechanic_family="relation_change",
                entities=(source, target),
                actions=(action,),
                relations=(relation,),
                predicted_effects=(
                    f"{action} may produce {delta_type} for {relation} between {source} and {target}",
                ),
                test_suggestions=(
                    f"replay same context and measure {relation} delta between {source} and {target} after {action} versus controls",
                ),
                confidence=confidence,
                preconditions=tuple(
                    role for role in (*source_roles, *target_roles) if role
                ),
            )
        )
    return tuple(hypotheses)


def detect_dynamic_invariants(
    *,
    grid_history: Sequence[Any],
    action_history: Sequence[str],
    tracks: Sequence[EntityTrack],
    role_hypotheses: Sequence[RoleHypothesis] = (),
) -> Tuple[DynamicInvariantCandidate, ...]:
    candidates: List[DynamicInvariantCandidate] = []
    roles_by_entity = _roles_by_entity(role_hypotheses)
    for track in tracks:
        summary = track.summary()
        roles = roles_by_entity.get(track.entity_id, set())
        sizes = [int(value) for value in summary.get("size_sequence", [])]
        centroids = [
            tuple(float(value) for value in centroid)
            for centroid in summary.get("centroid_sequence", [])
            if isinstance(centroid, Sequence) and len(centroid) == 2
        ]
        if "timer_or_hud" in roles and _series_changes(sizes):
            monotone = _monotonicity_score(sizes)
            if monotone >= 0.9:
                direction = "increasing" if sizes[-1] > sizes[0] else "decreasing"
                candidates.append(
                    DynamicInvariantCandidate(
                        invariant_id=f"m1g0::invariant::{track.entity_id}::monotone_counter",
                        invariant_family="monotone_counter",
                        source="entity_size_sequence",
                        entity_id=track.entity_id,
                        affected_entities=(track.entity_id,),
                        direction_candidate=direction,
                        monotonicity_score=monotone,
                        action_correlation_score=_action_step_correlation(sizes, action_history),
                        evidence={
                            "size_sequence": sizes[:32],
                            "role_basis": sorted(roles),
                            "remaining_semantics_unknown": True,
                        },
                        policy_relevance="terminal_or_resource_horizon_candidate",
                    )
                )
        if centroids and _directional_drift_candidate(centroids) is not None:
            direction, consistency = _directional_drift_candidate(centroids) or ("", 0.0)
            if consistency >= 0.7 and "timer_or_hud" not in roles:
                candidates.append(
                    DynamicInvariantCandidate(
                        invariant_id=f"m1g0::invariant::{track.entity_id}::exogenous_motion",
                        invariant_family="exogenous_motion",
                        source="entity_centroid_sequence",
                        entity_id=track.entity_id,
                        affected_entities=(track.entity_id,),
                        direction_candidate=direction,
                        action_correlation_score=float(summary.get("action_correlation_score", 0.0) or 0.0),
                        evidence={
                            "centroid_sequence": summary.get("centroid_sequence", [])[:32],
                            "direction_consistency": round(float(consistency), 6),
                            "role_basis": sorted(roles),
                        },
                        policy_relevance="forced_motion_or_drift_candidate",
                    )
                )
        if _irreversible_entity_change(summary):
            candidates.append(
                DynamicInvariantCandidate(
                    invariant_id=f"m1g0::invariant::{track.entity_id}::irreversible_change",
                    invariant_family="irreversible_change",
                    source="entity_lifecycle_or_size_sequence",
                    entity_id=track.entity_id,
                    affected_entities=(track.entity_id,),
                    irreversibility_score=1.0,
                    evidence={
                        "first_frame": summary.get("first_frame"),
                        "last_frame": summary.get("last_frame"),
                        "size_sequence": sizes[:32],
                        "appears_after_action": summary.get("appears_after_action"),
                        "disappears_after_action": summary.get("disappears_after_action"),
                    },
                    policy_relevance="consumption_or_one_way_transition_candidate",
                )
            )

    candidates.extend(_global_color_count_invariants(grid_history, action_history))
    return tuple(_dedupe_invariants(candidates))


def emit_dynamic_invariant_mechanic_hypotheses(
    invariants: Sequence[DynamicInvariantCandidate],
) -> Tuple[MechanicHypothesis, ...]:
    hypotheses: List[MechanicHypothesis] = []
    for invariant in invariants:
        confidence = max(
            float(invariant.monotonicity_score),
            float(invariant.action_correlation_score),
            float(invariant.irreversibility_score),
            float(invariant.periodicity_score),
        )
        hypotheses.append(
            MechanicHypothesis(
                mechanic_hypothesis_id=(
                    f"m1g0::dynamic_invariant::{invariant.invariant_id.split('::')[-2]}::{invariant.invariant_family}"
                ),
                mechanic_family="dynamic_invariant",
                entities=tuple(invariant.affected_entities),
                latent_variables=(invariant.invariant_id,),
                predicted_effects=(
                    f"{invariant.invariant_family} may constrain trajectory via {invariant.source}",
                ),
                test_suggestions=(
                    _dynamic_invariant_test_suggestion(invariant),
                ),
                confidence=confidence,
            )
        )
    return tuple(hypotheses)


def emit_action_effect_mechanic_hypotheses(
    action_effects: Sequence[ActionEffectAbstraction],
) -> Tuple[MechanicHypothesis, ...]:
    hypotheses: List[MechanicHypothesis] = []
    for effect in action_effects:
        if not effect.effect_families:
            continue
        denominator = max(1, int(effect.transitions_observed))
        for family, count in effect.effect_family_counts:
            if family in {"unknown"}:
                continue
            confidence = min(1.0, float(count) / float(denominator))
            latent_variables = (
                ("terminal_horizon_candidate",)
                if family == "tick_latent"
                else ()
            )
            hypotheses.append(
                MechanicHypothesis(
                    mechanic_hypothesis_id=(
                        f"m1g0::action_effect::{effect.action}::{family}"
                    ),
                    mechanic_family="action_effect",
                    entities=tuple(effect.affected_entities),
                    actions=(effect.action,),
                    latent_variables=latent_variables,
                    predicted_effects=(
                        f"{effect.action} may produce {family}",
                    ),
                    test_suggestions=(
                        _action_effect_test_suggestion(effect.action, family),
                    ),
                    confidence=confidence,
                )
            )
    return tuple(hypotheses)


def build_general_mechanic_candidate_ledger(
    *,
    grid_history: Sequence[Any],
    action_history: Sequence[str],
    game_id: str,
    source_label: str,
    policy_label: str = "",
) -> Dict[str, Any]:
    tracks = track_entities(grid_history, action_history)
    entity_summaries = [track.summary() for track in tracks]
    role_rows: List[Dict[str, Any]] = []
    role_hypotheses: List[RoleHypothesis] = []
    for track in tracks:
        rows = generate_role_hypotheses(track)
        role_hypotheses.extend(rows)
        role_rows.append(
            {
                "entity_id": track.entity_id,
                "role_hypotheses": [role.to_dict() for role in rows],
                "support": 0,
                "revision_status": REVISION_STATUS,
                "truth_status": TRUTH_STATUS,
            }
        )
    action_effects = infer_action_effect_abstractions(
        grid_history=grid_history,
        action_history=action_history,
        tracks=tracks,
        role_hypotheses=role_hypotheses,
    )
    relation_graphs_by_frame = build_relation_graphs_by_frame(
        tracks,
        role_hypotheses,
    )
    relation_deltas = infer_relation_deltas(
        relation_graphs_by_frame,
        action_history,
    )
    dynamic_invariants = detect_dynamic_invariants(
        grid_history=grid_history,
        action_history=action_history,
        tracks=tracks,
        role_hypotheses=role_hypotheses,
    )
    entity_role_hypotheses = emit_entity_role_mechanic_hypotheses(role_hypotheses)
    action_effect_hypotheses = emit_action_effect_mechanic_hypotheses(action_effects)
    relation_change_hypotheses = emit_relation_change_mechanic_hypotheses(
        relation_deltas
    )
    dynamic_invariant_hypotheses = emit_dynamic_invariant_mechanic_hypotheses(
        dynamic_invariants
    )
    mechanic_hypotheses = (
        entity_role_hypotheses
        + action_effect_hypotheses
        + relation_change_hypotheses
        + dynamic_invariant_hypotheses
    )
    role_counter = Counter(role.role for role in role_hypotheses)
    effect_counter: Counter[str] = Counter()
    for effect in action_effects:
        effect_counter.update(
            {
                family: count
                for family, count in effect.effect_family_counts
            }
        )
    relation_counter = Counter(delta.relation_delta_type for delta in relation_deltas)
    invariant_counter = Counter(
        invariant.invariant_family for invariant in dynamic_invariants
    )
    summary = {
        "game_id": game_id,
        "source_label": source_label,
        "policy_label": policy_label,
        "frames_consumed": len(grid_history),
        "actions_consumed": len(action_history),
        "entities_tracked": len(entity_summaries),
        "role_hypothesis_rows": len(role_rows),
        "actions_analyzed": len(action_effects),
        "action_effect_rows": len(action_effects),
        "entity_role_hypotheses_generated": len(entity_role_hypotheses),
        "action_effect_hypotheses_generated": len(action_effect_hypotheses),
        "relation_graph_frames": len(relation_graphs_by_frame),
        "relation_delta_rows": len(relation_deltas),
        "relation_change_hypotheses_generated": len(relation_change_hypotheses),
        "dynamic_invariant_candidates": len(dynamic_invariants),
        "dynamic_invariant_hypotheses_generated": len(dynamic_invariant_hypotheses),
        "mechanic_hypotheses_generated": len(mechanic_hypotheses),
        "controllable_actor_candidates": int(role_counter.get("controllable_actor", 0)),
        "timer_or_hud_candidates": int(role_counter.get("timer_or_hud", 0)),
        "effect_family_counts": dict(sorted(effect_counter.items())),
        "relation_delta_type_counts": dict(sorted(relation_counter.items())),
        "dynamic_invariant_family_counts": dict(sorted(invariant_counter.items())),
        "unknown_role_candidates": int(role_counter.get("unknown", 0)),
        "ready_for_m3_g0": bool(mechanic_hypotheses),
        "status": STATUS,
        "support": 0,
        "revision_status": REVISION_STATUS,
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
    }
    return {
        "config": {
            "schema_version": SCHEMA_VERSION,
            "game_id": game_id,
            "source_label": source_label,
            "policy_label": policy_label,
            "inputs_read": [source_label],
            "artifacts_not_read": ["A33", "LLM", "world_model"],
            "artifacts_not_modified": ["M2", "M3", "A32", "A33"],
            "milestones_covered": ["M1.G0.1", "M1.G0.2", "M1.G0.3", "M1.G0.4", "M1.G0.5"],
        },
        "entity_tracks": entity_summaries,
        "role_hypothesis_ledger": role_rows,
        "action_effect_abstractions": [
            effect.to_dict() for effect in action_effects
        ],
        "relation_graphs_by_frame": [dict(graph) for graph in relation_graphs_by_frame],
        "relation_delta_rows": [delta.to_dict() for delta in relation_deltas],
        "dynamic_invariant_candidates": [
            invariant.to_dict() for invariant in dynamic_invariants
        ],
        "mechanic_hypotheses": [
            hypothesis.to_dict() for hypothesis in mechanic_hypotheses
        ],
        "summary": summary,
        "status": STATUS,
        "support": 0,
        "revision_status": REVISION_STATUS,
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
    }


def run_bp35_general_mechanic_candidate_extractor(
    *,
    game_id: str = DEFAULT_GAME_ID,
    max_steps: int = 32,
    tie_break_seed: int = 0,
    environments_dir: str | Path | None = None,
    action_sequence: Sequence[str] = DEFAULT_ACTION_SWEEP_SEQUENCE,
) -> Dict[str, Any]:
    capture = capture_bp35_action_sweep_history(
        game_id=game_id,
        max_steps=max_steps,
        tie_break_seed=tie_break_seed,
        environments_dir=environments_dir,
        action_sequence=action_sequence,
    )
    return build_general_mechanic_candidate_ledger(
        grid_history=capture["grid_history"],
        action_history=capture["action_history"],
        game_id=game_id,
        source_label="real_bp35_action_sweep_visual_history",
        policy_label=str(capture.get("policy", "")),
    )


def capture_bp35_action_sweep_history(
    *,
    game_id: str = DEFAULT_GAME_ID,
    max_steps: int = 32,
    tie_break_seed: int = 0,
    environments_dir: str | Path | None = None,
    action_sequence: Sequence[str] = DEFAULT_ACTION_SWEEP_SEQUENCE,
) -> Dict[str, Any]:
    """Capture a generic exploratory visual history without policy verdicts."""
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    _configure_offline_env(env_dir)
    env = _make_env(game_id, env_dir)
    current_frame = _reset_env(env)
    initial = snapshot_frame(current_frame)
    grid_history = [initial.grid]
    action_history: list[str] = []
    action_rows: list[Dict[str, Any]] = []
    sequence = [str(action) for action in action_sequence if str(action)]
    if int(tie_break_seed) and sequence:
        offset = int(tie_break_seed) % len(sequence)
        sequence = sequence[offset:] + sequence[:offset]
    for step_index in range(max(0, int(max_steps))):
        before = snapshot_frame(current_frame)
        if str(before.game_state).upper() == "GAME_OVER":
            break
        action_name = sequence[step_index % len(sequence)] if sequence else ""
        selected = _first_valid_action_by_name(_valid_actions(env), action_name)
        if selected is None:
            action_rows.append(
                {
                    "step": step_index,
                    "requested_action": action_name,
                    "executed": False,
                    "blocked_reason": "action_not_available",
                    "support": 0,
                    "revision_status": REVISION_STATUS,
                    "truth_status": TRUTH_STATUS,
                }
            )
            continue
        after_frame = _step_env_action(env, selected)
        if after_frame is None:
            action_rows.append(
                {
                    "step": step_index,
                    "requested_action": action_name,
                    "executed": False,
                    "blocked_reason": "env_step_returned_none",
                    "support": 0,
                    "revision_status": REVISION_STATUS,
                    "truth_status": TRUTH_STATUS,
                }
            )
            break
        current_frame = after_frame
        after = snapshot_frame(after_frame, fallback_available_actions=before.available_actions)
        action_history.append(str(getattr(selected, "name", action_name)))
        grid_history.append(after.grid)
        action_rows.append(
            {
                "step": step_index,
                "requested_action": action_name,
                "executed_action": str(getattr(selected, "name", action_name)),
                "action_args": dict(getattr(selected, "action_args", {}) or {}),
                "executed": True,
                "game_state_after": after.game_state,
                "levels_completed_after": int(after.levels_completed),
                "support": 0,
                "revision_status": REVISION_STATUS,
                "truth_status": TRUTH_STATUS,
            }
        )
    return {
        "game_id": game_id,
        "policy": "generic_action_sweep_observation",
        "action_sequence": list(action_sequence),
        "grid_history": grid_history,
        "action_history": action_history,
        "action_rows": action_rows,
        "support": 0,
        "revision_status": REVISION_STATUS,
        "truth_status": TRUTH_STATUS,
    }


def write_general_mechanic_candidate_ledger(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_GENERAL_MECHANIC_CANDIDATES_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def infer_background(grid_history: Sequence[Any]) -> int | None:
    counts: Counter[int] = Counter()
    for grid in grid_history:
        array = _as_grid(grid)
        if array is None:
            continue
        values, value_counts = np.unique(array, return_counts=True)
        counts.update({int(value): int(count) for value, count in zip(values, value_counts)})
    return int(counts.most_common(1)[0][0]) if counts else None


def _component_match_score(left: ComponentObservation, right: ComponentObservation) -> float:
    if left.color != right.color:
        return 99.0
    centroid_distance = _distance(left.centroid, right.centroid)
    shape_penalty = 0.0 if left.shape_signature == right.shape_signature else 2.0
    size_penalty = abs(left.size - right.size) / max(1, max(left.size, right.size))
    return float(centroid_distance + shape_penalty + size_penalty)


def _first_valid_action_by_name(valid_actions: Iterable[Any], action_name: str) -> Any | None:
    matches = [
        action
        for action in valid_actions
        if str(getattr(action, "name", "")) == str(action_name)
    ]
    if not matches:
        return None
    return sorted(
        matches,
        key=lambda action: json.dumps(
            dict(getattr(action, "action_args", {}) or {}),
            sort_keys=True,
        ),
    )[0]


def _action_transition_delta(
    *,
    action: str,
    frame_index: int,
    before: np.ndarray,
    after: np.ndarray,
    tracks: Sequence[EntityTrack],
    observations_by_entity_frame: Mapping[Tuple[str, int], ComponentObservation],
    roles_by_entity: Mapping[str, set[str]],
) -> Dict[str, Any]:
    changed_pixels = (
        int(np.count_nonzero(before != after))
        if before.shape == after.shape
        else int(max(before.size, after.size))
    )
    effect_families: set[str] = set()
    affected_entities: set[str] = set()
    entity_deltas: List[Dict[str, Any]] = []
    latent_deltas: List[Dict[str, Any]] = []
    relation_deltas: List[Dict[str, Any]] = []
    for track in tracks:
        before_obs = observations_by_entity_frame.get((track.entity_id, frame_index))
        after_obs = observations_by_entity_frame.get((track.entity_id, frame_index + 1))
        delta = _entity_transition_delta(track.entity_id, before_obs, after_obs)
        if not delta:
            continue
        affected_entities.add(track.entity_id)
        entity_deltas.append(delta)
        effect_families.update(delta.get("effect_families", []))
        if "timer_or_hud" in roles_by_entity.get(track.entity_id, set()):
            effect_families.add("tick_latent")
            latent_deltas.append(
                {
                    "entity_id": track.entity_id,
                    "latent_family": "monotone_counter_candidate",
                    "delta_reason": "timer_or_hud_entity_changed",
                    "support": 0,
                    "revision_status": REVISION_STATUS,
                    "truth_status": TRUTH_STATUS,
                }
            )

    non_hud_entities = [
        entity_id
        for entity_id in affected_entities
        if "timer_or_hud" not in roles_by_entity.get(entity_id, set())
    ]
    if len(non_hud_entities) >= 2:
        effect_families.add("change_relation")
        relation_deltas.append(
            {
                "relation_family": "multi_entity_delta_candidate",
                "affected_entities": sorted(non_hud_entities)[:12],
                "description": "multiple non-HUD entities changed in the same transition",
                "support": 0,
                "revision_status": REVISION_STATUS,
                "truth_status": TRUTH_STATUS,
            }
        )
    if changed_pixels > 0 and not entity_deltas:
        effect_families.add("global_transition")
    if changed_pixels == 0 and not entity_deltas:
        effect_families.add("no_effect")
    if not effect_families:
        effect_families.add("unknown")
    return {
        "frame_index": int(frame_index),
        "action": action,
        "affected_entities": sorted(affected_entities),
        "effect_families": sorted(effect_families),
        "entity_deltas": entity_deltas,
        "relation_deltas": relation_deltas,
        "hud_or_latent_deltas": latent_deltas,
        "changed_pixels": int(changed_pixels),
        "grid_shape_before": list(before.shape),
        "grid_shape_after": list(after.shape),
        "support": 0,
        "revision_status": REVISION_STATUS,
        "truth_status": TRUTH_STATUS,
    }


def _entity_transition_delta(
    entity_id: str,
    before: ComponentObservation | None,
    after: ComponentObservation | None,
) -> Dict[str, Any] | None:
    if before is None and after is None:
        return None
    families: set[str] = set()
    row: Dict[str, Any] = {
        "entity_id": entity_id,
        "support": 0,
        "revision_status": REVISION_STATUS,
        "truth_status": TRUTH_STATUS,
    }
    if before is None and after is not None:
        families.add("create_entity")
        row.update(
            {
                "before_present": False,
                "after_present": True,
                "after_bbox": [int(value) for value in after.bbox],
                "after_centroid": [round(float(value), 4) for value in after.centroid],
                "after_size": int(after.size),
            }
        )
    elif before is not None and after is None:
        families.add("delete_entity")
        row.update(
            {
                "before_present": True,
                "after_present": False,
                "before_bbox": [int(value) for value in before.bbox],
                "before_centroid": [round(float(value), 4) for value in before.centroid],
                "before_size": int(before.size),
            }
        )
    elif before is not None and after is not None:
        dy = round(float(after.centroid[0] - before.centroid[0]), 4)
        dx = round(float(after.centroid[1] - before.centroid[1]), 4)
        distance = round(math.hypot(dy, dx), 6)
        size_delta = int(after.size - before.size)
        shape_changed = before.shape_signature != after.shape_signature
        bbox_delta = tuple(int(after.bbox[index] - before.bbox[index]) for index in range(4))
        if distance > 0:
            families.add("move_entity")
        if size_delta != 0 or shape_changed:
            families.add("transform_entity")
        if not families:
            return None
        row.update(
            {
                "before_present": True,
                "after_present": True,
                "centroid_delta": [dy, dx],
                "centroid_distance": distance,
                "bbox_delta": [int(value) for value in bbox_delta],
                "size_delta": int(size_delta),
                "shape_changed": bool(shape_changed),
                "before_bbox": [int(value) for value in before.bbox],
                "after_bbox": [int(value) for value in after.bbox],
            }
        )
    row["effect_families"] = sorted(families)
    return row


def _aggregate_action_effect(action: str, rows: Sequence[Mapping[str, Any]]) -> ActionEffectAbstraction:
    family_counts: Counter[str] = Counter()
    affected_entities: set[str] = set()
    entity_deltas: List[Dict[str, Any]] = []
    relation_deltas: List[Dict[str, Any]] = []
    latent_deltas: List[Dict[str, Any]] = []
    changed_pixels_values: List[int] = []
    changed_transition_count = 0
    for row in rows:
        families = [str(family) for family in row.get("effect_families", []) or []]
        family_counts.update(families)
        affected_entities.update(str(entity_id) for entity_id in row.get("affected_entities", []) or [])
        changed_pixels = int(row.get("changed_pixels", 0) or 0)
        changed_pixels_values.append(changed_pixels)
        if changed_pixels > 0:
            changed_transition_count += 1
        for delta in row.get("entity_deltas", []) or []:
            if isinstance(delta, Mapping):
                item = dict(delta)
                item["transition_frame_index"] = int(row.get("frame_index", 0) or 0)
                entity_deltas.append(item)
        for delta in row.get("relation_deltas", []) or []:
            if isinstance(delta, Mapping):
                item = dict(delta)
                item["transition_frame_index"] = int(row.get("frame_index", 0) or 0)
                relation_deltas.append(item)
        for delta in row.get("hud_or_latent_deltas", []) or []:
            if isinstance(delta, Mapping):
                item = dict(delta)
                item["transition_frame_index"] = int(row.get("frame_index", 0) or 0)
                latent_deltas.append(item)
    transitions = len(rows)
    global_delta = {
        "transitions_observed": int(transitions),
        "changed_transition_count": int(changed_transition_count),
        "mean_changed_pixels": round(_mean([float(value) for value in changed_pixels_values]), 6),
        "max_changed_pixels": int(max(changed_pixels_values or [0])),
        "support": 0,
        "revision_status": REVISION_STATUS,
        "truth_status": TRUTH_STATUS,
    }
    return ActionEffectAbstraction(
        action=str(action),
        transitions_observed=transitions,
        affected_entities=tuple(sorted(affected_entities)),
        effect_families=tuple(sorted(family_counts)),
        effect_family_counts=tuple(
            sorted(family_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        entity_delta_summary=tuple(entity_deltas[:24]),
        relation_delta_summary=tuple(relation_deltas[:12]),
        hud_or_latent_delta_summary=tuple(latent_deltas[:12]),
        global_delta_summary=global_delta,
        candidate_preconditions=(),
    )


def _changed_observation_frames(
    observations: Sequence[ComponentObservation],
) -> Tuple[int, ...]:
    frames = []
    previous = None
    for obs in observations:
        if previous is not None and (
            obs.centroid != previous.centroid
            or obs.size != previous.size
            or obs.shape_signature != previous.shape_signature
            or obs.bbox != previous.bbox
        ):
            frames.append(obs.frame_index)
        previous = obs
    return tuple(frames)


def _role_test_suggestion(role: str, entity_id: str) -> str:
    if role == "controllable_actor":
        return f"compare same-state actions and measure centroid/relation delta for {entity_id}"
    if role == "timer_or_hud":
        return f"verify monotone edge variable for {entity_id} across actions and terminality"
    if role == "boundary":
        return f"test whether moving entities are blocked by contact with {entity_id}"
    if role == "target_candidate":
        return f"test whether contact or transformation involving {entity_id} changes objective state"
    if role == "transformed_object":
        return f"test which action changes shape or size of {entity_id}"
    return f"run controlled action/entity probe for {entity_id} role {role}"


def _action_effect_test_suggestion(action: str, family: str) -> str:
    if family == "move_entity":
        return f"replay same state and compare {action} against controls on entity centroid delta"
    if family == "transform_entity":
        return f"replay same state and compare {action} against controls on entity size/shape delta"
    if family == "create_entity":
        return f"test whether {action} creates a matching entity from the same context"
    if family == "delete_entity":
        return f"test whether {action} removes a matching entity from the same context"
    if family == "change_relation":
        return f"measure relation graph before/after {action} versus controls"
    if family == "tick_latent":
        return f"measure monotone latent/HUD delta after {action} and controls"
    if family == "global_transition":
        return f"measure global grid and terminal/objective deltas after {action}"
    if family == "no_effect":
        return f"verify whether {action} remains no-effect across comparable contexts"
    return f"run controlled action-effect probe for {action} family {family}"


def _roles_by_entity(role_hypotheses: Sequence[RoleHypothesis]) -> Dict[str, set[str]]:
    roles: Dict[str, set[str]] = {}
    for role in role_hypotheses:
        roles.setdefault(role.entity_id, set()).add(role.role)
    return roles


def _entity_priority(
    entity_id: str,
    obs: ComponentObservation,
    roles_by_entity: Mapping[str, set[str]],
) -> Tuple[int, int, int, str]:
    roles = roles_by_entity.get(entity_id, set())
    if "controllable_actor" in roles:
        role_rank = 0
    elif "timer_or_hud" in roles:
        role_rank = 3
    elif roles & {"target_candidate", "moving_object", "transformed_object"}:
        role_rank = 1
    else:
        role_rank = 2
    return (role_rank, -int(obs.size), int(obs.color), entity_id)


def _unary_relation_edges(
    *,
    frame_index: int,
    entity_id: str,
    obs: ComponentObservation,
    roles: set[str],
) -> Tuple[RelationEdge, ...]:
    edges: List[RelationEdge] = []
    if obs.edge_touching:
        for location in obs.edge_locations:
            edges.append(
                RelationEdge(
                    frame_index=frame_index,
                    source_entity=entity_id,
                    target_entity=f"EDGE_{location.upper()}",
                    relation="near_edge",
                    value=location,
                    source_roles=tuple(sorted(roles)),
                    target_roles=("boundary",),
                )
            )
    return tuple(edges)


def _pair_relation_edges(
    *,
    frame_index: int,
    left_id: str,
    left: ComponentObservation,
    right_id: str,
    right: ComponentObservation,
    roles_by_entity: Mapping[str, set[str]],
    near_distance_threshold: float,
) -> Tuple[RelationEdge, ...]:
    if left_id == right_id:
        return ()
    source_roles = tuple(sorted(roles_by_entity.get(left_id, set())))
    target_roles = tuple(sorted(roles_by_entity.get(right_id, set())))
    reverse_source_roles = tuple(sorted(roles_by_entity.get(right_id, set())))
    reverse_target_roles = tuple(sorted(roles_by_entity.get(left_id, set())))
    distance = _bbox_distance(left.bbox, right.bbox)
    centroid_distance = _distance(left.centroid, right.centroid)
    edges: List[RelationEdge] = [
        RelationEdge(
            frame_index=frame_index,
            source_entity=left_id,
            target_entity=right_id,
            relation="distance",
            value=round(float(centroid_distance), 6),
            source_roles=source_roles,
            target_roles=target_roles,
        )
    ]
    if left.color == right.color:
        edges.append(
            RelationEdge(
                frame_index=frame_index,
                source_entity=left_id,
                target_entity=right_id,
                relation="same_color",
                value=True,
                source_roles=source_roles,
                target_roles=target_roles,
            )
        )
    if left.shape_signature == right.shape_signature:
        edges.append(
            RelationEdge(
                frame_index=frame_index,
                source_entity=left_id,
                target_entity=right_id,
                relation="same_shape_signature",
                value=True,
                source_roles=source_roles,
                target_roles=target_roles,
            )
        )
    if _bbox_overlaps(left.bbox, right.bbox):
        edges.append(
            RelationEdge(
                frame_index=frame_index,
                source_entity=left_id,
                target_entity=right_id,
                relation="overlaps",
                value=True,
                source_roles=source_roles,
                target_roles=target_roles,
            )
        )
    if distance == 0:
        edges.append(
            RelationEdge(
                frame_index=frame_index,
                source_entity=left_id,
                target_entity=right_id,
                relation="touches",
                value=True,
                source_roles=source_roles,
                target_roles=target_roles,
            )
        )
        edges.append(
            RelationEdge(
                frame_index=frame_index,
                source_entity=left_id,
                target_entity=right_id,
                relation="adjacent_to",
                value=True,
                source_roles=source_roles,
                target_roles=target_roles,
            )
        )
    elif distance <= near_distance_threshold:
        edges.append(
            RelationEdge(
                frame_index=frame_index,
                source_entity=left_id,
                target_entity=right_id,
                relation="near",
                value=round(float(distance), 6),
                source_roles=source_roles,
                target_roles=target_roles,
            )
        )
    vertical, horizontal = _relative_position(left, right)
    if vertical:
        edges.append(
            RelationEdge(
                frame_index=frame_index,
                source_entity=left_id,
                target_entity=right_id,
                relation=vertical,
                value=True,
                source_roles=source_roles,
                target_roles=target_roles,
            )
        )
    if horizontal:
        edges.append(
            RelationEdge(
                frame_index=frame_index,
                source_entity=left_id,
                target_entity=right_id,
                relation=horizontal,
                value=True,
                source_roles=source_roles,
                target_roles=target_roles,
            )
        )
    if _bbox_contains(left.bbox, right.bbox):
        edges.append(
            RelationEdge(
                frame_index=frame_index,
                source_entity=left_id,
                target_entity=right_id,
                relation="contains",
                value=True,
                source_roles=source_roles,
                target_roles=target_roles,
            )
        )
        edges.append(
            RelationEdge(
                frame_index=frame_index,
                source_entity=right_id,
                target_entity=left_id,
                relation="inside",
                value=True,
                source_roles=reverse_source_roles,
                target_roles=reverse_target_roles,
            )
        )
    elif _bbox_contains(right.bbox, left.bbox):
        edges.append(
            RelationEdge(
                frame_index=frame_index,
                source_entity=right_id,
                target_entity=left_id,
                relation="contains",
                value=True,
                source_roles=reverse_source_roles,
                target_roles=reverse_target_roles,
            )
        )
        edges.append(
            RelationEdge(
                frame_index=frame_index,
                source_entity=left_id,
                target_entity=right_id,
                relation="inside",
                value=True,
                source_roles=source_roles,
                target_roles=target_roles,
            )
        )
    return tuple(edges)


def _relation_delta_from_edges(
    *,
    action: str,
    frame_index: int,
    key: Tuple[str, str, str],
    before_edge: Mapping[str, Any] | None,
    after_edge: Mapping[str, Any] | None,
) -> RelationDelta | None:
    source, target, relation = key
    before_value = None if before_edge is None else before_edge.get("value")
    after_value = None if after_edge is None else after_edge.get("value")
    if before_edge is None and after_edge is None:
        return None
    if before_edge is not None and after_edge is None:
        delta_type = _relation_removed_delta_type(relation)
    elif before_edge is None and after_edge is not None:
        delta_type = _relation_created_delta_type(relation)
    elif relation == "distance":
        try:
            numeric_delta = float(after_value) - float(before_value)
        except (TypeError, ValueError):
            return None
        if abs(numeric_delta) < 1e-9:
            return None
        delta_type = "distance_increases" if numeric_delta > 0 else "distance_decreases"
        return RelationDelta(
            action=action,
            frame_index=frame_index,
            source_entity=source,
            target_entity=target,
            relation=relation,
            relation_before=str(before_value),
            relation_after=str(after_value),
            relation_delta_type=delta_type,
            source_roles=tuple(before_edge.get("source_roles", []) or after_edge.get("source_roles", [])),
            target_roles=tuple(before_edge.get("target_roles", []) or after_edge.get("target_roles", [])),
            numeric_delta=round(float(numeric_delta), 6),
        )
    elif before_value != after_value:
        delta_type = "relation_value_changed"
    else:
        return None
    source_roles = tuple(
        (after_edge or before_edge or {}).get("source_roles", [])
    )
    target_roles = tuple(
        (after_edge or before_edge or {}).get("target_roles", [])
    )
    return RelationDelta(
        action=action,
        frame_index=frame_index,
        source_entity=source,
        target_entity=target,
        relation=relation,
        relation_before=None if before_value is None else str(before_value),
        relation_after=None if after_value is None else str(after_value),
        relation_delta_type=delta_type,
        source_roles=source_roles,
        target_roles=target_roles,
    )


def _relation_delta_is_hypothesis_worthy(delta: RelationDelta) -> bool:
    if delta.relation_delta_type == "unchanged":
        return False
    if delta.relation_delta_type in {"relation_created", "relation_removed"}:
        return bool(
            {"controllable_actor", "target_candidate", "moving_object", "timer_or_hud"}
            & set(delta.source_roles)
            or {"controllable_actor", "target_candidate", "moving_object", "timer_or_hud"}
            & set(delta.target_roles)
        )
    return True


def _relation_delta_priority(delta: RelationDelta) -> int:
    roles = set(delta.source_roles) | set(delta.target_roles)
    score = 0
    if "controllable_actor" in roles:
        score += 5
    if "target_candidate" in roles:
        score += 3
    if "moving_object" in roles:
        score += 2
    if delta.relation_delta_type in {"contact_created", "contact_removed"}:
        score += 4
    if delta.relation_delta_type in {"distance_decreases", "distance_increases"}:
        score += 3
    if delta.relation_delta_type in {"near_relation_created", "near_relation_removed"}:
        score += 2
    if delta.relation_delta_type in {"relation_created", "relation_removed"}:
        score -= 2
    return score


def _series_changes(values: Sequence[Any]) -> bool:
    return len(set(values)) > 1 if values else False


def _action_step_correlation(values: Sequence[int], action_history: Sequence[str]) -> float:
    if len(values) < 2 or not action_history:
        return 0.0
    deltas = [
        values[index + 1] - values[index]
        for index in range(min(len(values) - 1, len(action_history)))
    ]
    changed = sum(1 for delta in deltas if delta != 0)
    return changed / max(1, len(deltas))


def _directional_drift_candidate(
    centroids: Sequence[Tuple[float, float]],
) -> Tuple[str, float] | None:
    if len(centroids) < 3:
        return None
    deltas: List[Tuple[float, float]] = []
    for index in range(len(centroids) - 1):
        dy = round(float(centroids[index + 1][0] - centroids[index][0]), 6)
        dx = round(float(centroids[index + 1][1] - centroids[index][1]), 6)
        if abs(dy) > 1e-9 or abs(dx) > 1e-9:
            deltas.append((dy, dx))
    if len(deltas) < 2:
        return None
    directions = [_direction_label(dy, dx) for dy, dx in deltas]
    label, count = Counter(directions).most_common(1)[0]
    return label, count / max(1, len(directions))


def _direction_label(dy: float, dx: float) -> str:
    if abs(dy) >= abs(dx):
        return "down" if dy > 0 else "up"
    return "right" if dx > 0 else "left"


def _irreversible_entity_change(summary: Mapping[str, Any]) -> bool:
    appears = summary.get("appears_after_action") is not None
    disappears = summary.get("disappears_after_action") is not None
    sizes = [int(value) for value in summary.get("size_sequence", []) if value is not None]
    monotone = bool(_monotonicity_score(sizes) >= 0.9 and _series_changes(sizes))
    return bool(appears or disappears or monotone)


def _global_color_count_invariants(
    grid_history: Sequence[Any],
    action_history: Sequence[str],
) -> Tuple[DynamicInvariantCandidate, ...]:
    arrays = [array for array in (_as_grid(grid) for grid in grid_history) if array is not None]
    if len(arrays) < 2:
        return ()
    colors = sorted({int(value) for array in arrays for value in np.unique(array)})
    candidates: List[DynamicInvariantCandidate] = []
    for color in colors:
        counts = [int(np.count_nonzero(array == color)) for array in arrays]
        if not _series_changes(counts):
            continue
        monotone = _monotonicity_score(counts)
        if monotone >= 0.9:
            direction = "increasing" if counts[-1] > counts[0] else "decreasing"
            candidates.append(
                DynamicInvariantCandidate(
                    invariant_id=f"m1g0::invariant::color_{color}::monotone_counter",
                    invariant_family="monotone_counter",
                    source="global_color_count_sequence",
                    affected_colors=(int(color),),
                    direction_candidate=direction,
                    monotonicity_score=monotone,
                    action_correlation_score=_action_step_correlation(counts, action_history),
                    evidence={
                        "color": int(color),
                        "count_sequence": counts[:32],
                    },
                    policy_relevance="resource_or_terminal_counter_candidate",
                )
            )
        if _irreversible_count_change(counts):
            candidates.append(
                DynamicInvariantCandidate(
                    invariant_id=f"m1g0::invariant::color_{color}::irreversible_change",
                    invariant_family="irreversible_change",
                    source="global_color_count_sequence",
                    affected_colors=(int(color),),
                    irreversibility_score=1.0,
                    evidence={
                        "color": int(color),
                        "count_sequence": counts[:32],
                    },
                    policy_relevance="one_way_color_inventory_change_candidate",
                )
            )
    phase = _phase_indicator_candidate(arrays)
    return tuple(candidates + list(phase))


def _irreversible_count_change(values: Sequence[int]) -> bool:
    if len(values) < 2 or not _series_changes(values):
        return False
    nondecreasing = all(values[index] <= values[index + 1] for index in range(len(values) - 1))
    nonincreasing = all(values[index] >= values[index + 1] for index in range(len(values) - 1))
    return bool(nondecreasing or nonincreasing)


def _phase_indicator_candidate(
    arrays: Sequence[np.ndarray],
) -> Tuple[DynamicInvariantCandidate, ...]:
    unique_signatures = [_grid_digest(array) for array in arrays]
    repeated = len(set(unique_signatures)) < len(unique_signatures)
    if not repeated:
        return ()
    return (
        DynamicInvariantCandidate(
            invariant_id="m1g0::invariant::global_state::phase_indicator",
            invariant_family="phase_indicator",
            source="global_grid_signature_sequence",
            periodicity_score=0.5,
            evidence={
                "unique_state_signatures": len(set(unique_signatures)),
                "states_observed": len(unique_signatures),
                "repeated_state_observed": True,
            },
            policy_relevance="cycle_or_phase_candidate",
        ),
    )


def _grid_digest(array: np.ndarray) -> str:
    return hashlib.sha1(array.tobytes()).hexdigest()[:16]


def _dedupe_invariants(
    candidates: Sequence[DynamicInvariantCandidate],
) -> Tuple[DynamicInvariantCandidate, ...]:
    by_id: Dict[str, DynamicInvariantCandidate] = {}
    for candidate in candidates:
        by_id[candidate.invariant_id] = candidate
    return tuple(by_id[key] for key in sorted(by_id))


def _dynamic_invariant_test_suggestion(
    invariant: DynamicInvariantCandidate,
) -> str:
    if invariant.invariant_family == "monotone_counter":
        return "compare repeated actions and controls on monotone latent/counter progression"
    if invariant.invariant_family == "exogenous_motion":
        return "test whether entity drift persists under different actions and is blocked by contacts"
    if invariant.invariant_family == "irreversible_change":
        return "test whether the observed entity/color change can be reversed by alternative actions"
    if invariant.invariant_family == "phase_indicator":
        return "test whether repeated state signatures predict cycles or phase transitions"
    return f"run controlled invariant probe for {invariant.invariant_family}"


def _relation_created_delta_type(relation: str) -> str:
    if relation in {"touches", "adjacent_to"}:
        return "contact_created"
    if relation == "near":
        return "near_relation_created"
    if relation in {"contains", "inside"}:
        return "containment_created"
    if relation == "near_edge":
        return "edge_relation_created"
    return "relation_created"


def _relation_removed_delta_type(relation: str) -> str:
    if relation in {"touches", "adjacent_to"}:
        return "contact_removed"
    if relation == "near":
        return "near_relation_removed"
    if relation in {"contains", "inside"}:
        return "containment_removed"
    if relation == "near_edge":
        return "edge_relation_removed"
    return "relation_removed"


def _bbox_distance(
    left: Tuple[int, int, int, int],
    right: Tuple[int, int, int, int],
) -> float:
    left_y0, left_x0, left_y1, left_x1 = left
    right_y0, right_x0, right_y1, right_x1 = right
    dy = max(right_y0 - left_y1 - 1, left_y0 - right_y1 - 1, 0)
    dx = max(right_x0 - left_x1 - 1, left_x0 - right_x1 - 1, 0)
    return math.hypot(float(dy), float(dx))


def _bbox_overlaps(
    left: Tuple[int, int, int, int],
    right: Tuple[int, int, int, int],
) -> bool:
    return not (
        left[2] < right[0]
        or right[2] < left[0]
        or left[3] < right[1]
        or right[3] < left[1]
    )


def _bbox_contains(
    outer: Tuple[int, int, int, int],
    inner: Tuple[int, int, int, int],
) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def _relative_position(
    left: ComponentObservation,
    right: ComponentObservation,
) -> Tuple[str | None, str | None]:
    ly, lx = left.centroid
    ry, rx = right.centroid
    vertical = "above" if ly < ry else "below" if ly > ry else None
    horizontal = "left_of" if lx < rx else "right_of" if lx > rx else None
    return vertical, horizontal


def _as_grid(value: Any) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.int32)
    except (TypeError, ValueError):
        return None
    if array.ndim != 2:
        return None
    return array


def _shape_signature(points: Sequence[Tuple[int, int]]) -> str:
    if not points:
        return ""
    min_y = min(point[0] for point in points)
    min_x = min(point[1] for point in points)
    digest = hashlib.sha1()
    for y, x in sorted((y - min_y, x - min_x) for y, x in points):
        digest.update(f"{y}:{x};".encode("ascii"))
    return digest.hexdigest()[:16]


def _edge_locations(
    bbox: Tuple[int, int, int, int],
    shape: Tuple[int, int],
) -> Tuple[str, ...]:
    height, width = shape
    locations = []
    if bbox[0] == 0:
        locations.append("top")
    if bbox[2] == height - 1:
        locations.append("bottom")
    if bbox[1] == 0:
        locations.append("left")
    if bbox[3] == width - 1:
        locations.append("right")
    return tuple(locations)


def _distance(left: Tuple[float, float], right: Tuple[float, float]) -> float:
    return math.hypot(float(left[0] - right[0]), float(left[1] - right[1]))


def _mean(values: Sequence[float]) -> float:
    return sum(float(value) for value in values) / max(1, len(values))


def _mode_rate(values: Sequence[Any]) -> float:
    if not values:
        return 0.0
    return Counter(values).most_common(1)[0][1] / max(1, len(values))


def _monotonicity_score(values: Sequence[int]) -> float:
    if len(values) < 2:
        return 0.0
    nondecreasing = all(values[index] <= values[index + 1] for index in range(len(values) - 1))
    nonincreasing = all(values[index] >= values[index + 1] for index in range(len(values) - 1))
    changed = any(values[index] != values[index + 1] for index in range(len(values) - 1))
    return 1.0 if changed and (nondecreasing or nonincreasing) else 0.0


def _has_size_variation(values: Sequence[Any]) -> bool:
    cleaned = [int(value) for value in values if value is not None]
    return len(set(cleaned)) > 1


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(child) for child in value]
    if isinstance(value, list):
        return [_json_ready(child) for child in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run M1.G0 general mechanic candidate extraction.",
    )
    parser.add_argument("--game-id", default=DEFAULT_GAME_ID)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--tie-break-seed", type=int, default=0)
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument(
        "--action-sequence",
        nargs="*",
        default=list(DEFAULT_ACTION_SWEEP_SEQUENCE),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_GENERAL_MECHANIC_CANDIDATES_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_bp35_general_mechanic_candidate_extractor(
        game_id=args.game_id,
        max_steps=args.max_steps,
        tie_break_seed=args.tie_break_seed,
        environments_dir=args.environments_dir,
        action_sequence=tuple(args.action_sequence or DEFAULT_ACTION_SWEEP_SEQUENCE),
    )
    write_general_mechanic_candidate_ledger(payload, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "summary": payload["summary"],
                "status": STATUS,
                "support": 0,
                "revision_status": REVISION_STATUS,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
