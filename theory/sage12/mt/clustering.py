"""Stable transformation clusters and observed-evidence prototype memory."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .model import TransformationEmbedding
from .transition import MTTransitionRecord

CLUSTER_FORMAT_VERSION = "sage12-mt-clusters-v4.16"


def _normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-8, None)


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _candidate_alias(
    signatures: Sequence[str],
    events: Sequence[str],
) -> str:
    counts = Counter(
        item.split(":", 1)[0]
        for item in signatures
    )
    event_names = Counter(
        event.split("#", 1)[0]
        for event in events
        if not event.startswith("invariant:")
    )
    for name, alias in (
        ("merge", "fusion"),
        ("split", "separation"),
        ("growth", "croissance"),
        ("contraction", "contraction"),
        ("birth", "apparition"),
        ("death", "disparition"),
        ("relative_motion", "deplacement"),
    ):
        if event_names[name] or counts[name]:
            return alias
    if any("free_regions:decreased" in event for event in events):
        return "connexion_regions"
    if any("free_regions:increased" in event for event in events):
        return "separation_regions"
    return "transformation_mixte"


@dataclass(frozen=True)
class TransformationPrototype:
    prototype_id: str
    centroid: tuple[float, ...]
    medoid_transition_id: str
    assignment_threshold: float
    dispersion: float
    support: int
    games: tuple[str, ...]
    action_families: tuple[str, ...]
    dominant_delta_signatures: tuple[str, ...]
    enriched_events: tuple[str, ...]
    candidate_alias: str
    productive_observations: int = 0
    risk_observations: int = 0

    @property
    def productive_probability(self) -> float:
        return (self.productive_observations + 1.0) / (self.support + 2.0)

    @property
    def risk_probability(self) -> float:
        return (self.risk_observations + 1.0) / (self.support + 2.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TransformationPrototype:
        values = dict(payload)
        for key in (
            "centroid",
            "games",
            "action_families",
            "dominant_delta_signatures",
            "enriched_events",
        ):
            values[key] = tuple(values.get(key, ()))
        return cls(**values)


@dataclass(frozen=True)
class TransformationMatch:
    prototype_id: str
    similarity: float
    productive_probability: float
    risk_probability: float
    support: int
    game_diversity: int
    candidate_alias: str


@dataclass(frozen=True)
class ClusterRegistry:
    prototypes: tuple[TransformationPrototype, ...]
    labels_by_transition: Mapping[str, str]
    noise_transition_ids: tuple[str, ...]
    selected_parameters: Mapping[str, int]
    stability_ari: float
    eligible_coverage: float
    format_version: str = CLUSTER_FORMAT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "prototypes": [item.to_dict() for item in self.prototypes],
            "labels_by_transition": dict(self.labels_by_transition),
            "noise_transition_ids": list(self.noise_transition_ids),
            "selected_parameters": dict(self.selected_parameters),
            "stability_ari": self.stability_ari,
            "eligible_coverage": self.eligible_coverage,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ClusterRegistry:
        if payload.get("format_version") != CLUSTER_FORMAT_VERSION:
            raise ValueError("unsupported SAGE-MT cluster registry")
        return cls(
            prototypes=tuple(
                TransformationPrototype.from_dict(item)
                for item in payload.get("prototypes", ())
            ),
            labels_by_transition=dict(payload.get("labels_by_transition", {})),
            noise_transition_ids=tuple(payload.get("noise_transition_ids", ())),
            selected_parameters=dict(payload.get("selected_parameters", {})),
            stability_ari=float(payload.get("stability_ari", 0.0)),
            eligible_coverage=float(payload.get("eligible_coverage", 0.0)),
        )


def _fit_hdbscan(
    matrix: np.ndarray,
    *,
    min_cluster_size: int,
    min_samples: int,
) -> np.ndarray:
    from sklearn.cluster import HDBSCAN

    if len(matrix) < max(2, min_cluster_size):
        return np.full(len(matrix), -1, dtype=np.int64)
    return HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        allow_single_cluster=False,
        copy=True,
    ).fit_predict(matrix)


def _eligible_labels(
    labels: np.ndarray,
    games: Sequence[str],
    *,
    minimum_support: int,
    minimum_games: int,
) -> set[int]:
    output = set()
    for label in sorted(set(labels.tolist()) - {-1}):
        indices = np.flatnonzero(labels == label)
        if len(indices) < minimum_support:
            continue
        if len({games[int(index)] for index in indices}) < minimum_games:
            continue
        output.add(int(label))
    return output


def _bootstrap_stability(
    matrix: np.ndarray,
    full_labels: np.ndarray,
    *,
    min_cluster_size: int,
    min_samples: int,
    samples: int,
    seed: int,
) -> float:
    from sklearn.metrics import adjusted_rand_score

    if len(matrix) < 4:
        return 0.0
    rng = np.random.default_rng(seed)
    values = []
    subset_size = max(2, round(0.80 * len(matrix)))
    for _ in range(max(1, int(samples))):
        indices = np.sort(
            rng.choice(len(matrix), size=subset_size, replace=False)
        )
        labels = _fit_hdbscan(
            matrix[indices],
            min_cluster_size=min(min_cluster_size, max(2, subset_size // 2)),
            min_samples=min(min_samples, max(1, subset_size // 3)),
        )
        values.append(
            float(adjusted_rand_score(full_labels[indices], labels))
        )
    return float(np.mean(values))


def fit_cluster_registry(
    embeddings: Sequence[TransformationEmbedding],
    records: Sequence[MTTransitionRecord],
    *,
    minimum_support: int = 20,
    minimum_games: int = 3,
    bootstrap_samples: int = 20,
    parameter_grid: Sequence[tuple[int, int]] = (
        (16, 5),
        (16, 10),
        (32, 5),
        (32, 10),
        (64, 5),
        (64, 10),
    ),
    seed: int = 5_160,
) -> ClusterRegistry:
    """Fit train-only HDBSCAN and compile content-addressed prototypes."""

    if len(embeddings) != len(records) or not embeddings:
        raise ValueError("SAGE-MT clustering requires aligned non-empty inputs")
    by_transition = {record.transition_id: record for record in records}
    ordered_records = [by_transition[item.transition_id] for item in embeddings]
    matrix = _normalize(
        np.asarray([item.vector for item in embeddings], dtype=np.float32)
    )
    games = [item.source_game_id for item in embeddings]
    candidates = []
    for min_cluster_size, min_samples in parameter_grid:
        labels = _fit_hdbscan(
            matrix,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )
        eligible = _eligible_labels(
            labels,
            games,
            minimum_support=minimum_support,
            minimum_games=minimum_games,
        )
        coverage = float(
            np.mean([int(int(label) in eligible) for label in labels])
        )
        stability = _bootstrap_stability(
            matrix,
            labels,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            samples=bootstrap_samples,
            seed=seed + min_cluster_size * 10 + min_samples,
        )
        candidates.append(
            (
                stability * coverage,
                min_cluster_size,
                min_samples,
                stability,
                coverage,
                labels,
                eligible,
            )
        )
    # Larger minimum sizes win exact ties to keep the registry parsimonious.
    selected = max(candidates, key=lambda row: (row[0], row[1], row[2]))
    (
        _,
        min_cluster_size,
        min_samples,
        stability,
        coverage,
        labels,
        eligible,
    ) = selected

    prototypes: list[TransformationPrototype] = []
    labels_by_transition: dict[str, str] = {}
    noise: list[str] = []
    for label in sorted(eligible):
        indices = np.flatnonzero(labels == label)
        vectors = matrix[indices]
        centroid = vectors.mean(axis=0)
        centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-8)
        similarities = vectors @ centroid
        medoid_local = int(np.argmax(similarities))
        medoid_index = int(indices[medoid_local])
        cluster_records = [ordered_records[int(index)] for index in indices]
        signatures = [record.delta_signature for record in cluster_records]
        events = [
            event
            for record in cluster_records
            for event in record.events
        ]
        dominant_signatures = tuple(
            key
            for key, _ in Counter(signatures).most_common(5)
        )
        enriched_events = tuple(
            key
            for key, _ in Counter(events).most_common(8)
        )
        identity_payload = {
            "medoid": embeddings[medoid_index].transition_id,
            "signatures": dominant_signatures,
            "centroid": [round(float(value), 5) for value in centroid],
        }
        prototype_id = "mt::" + hashlib.sha256(
            _canonical(identity_payload).encode("utf-8")
        ).hexdigest()[:16]
        productive = sum(record.productive is True for record in cluster_records)
        risk = sum(record.risk is True for record in cluster_records)
        prototype = TransformationPrototype(
            prototype_id=prototype_id,
            centroid=tuple(float(value) for value in centroid),
            medoid_transition_id=embeddings[medoid_index].transition_id,
            assignment_threshold=float(np.quantile(similarities, 0.05)),
            dispersion=float(np.mean(1.0 - similarities)),
            support=len(indices),
            games=tuple(sorted({record.source_game_id for record in cluster_records})),
            action_families=tuple(
                sorted({record.graph_before.action_family for record in cluster_records})
            ),
            dominant_delta_signatures=dominant_signatures,
            enriched_events=enriched_events,
            candidate_alias=_candidate_alias(signatures, events),
            productive_observations=int(productive),
            risk_observations=int(risk),
        )
        prototypes.append(prototype)
        for index in indices:
            labels_by_transition[embeddings[int(index)].transition_id] = prototype_id
    for index, embedding in enumerate(embeddings):
        if int(labels[index]) not in eligible:
            noise.append(embedding.transition_id)
    prototypes.sort(key=lambda item: item.prototype_id)
    return ClusterRegistry(
        prototypes=tuple(prototypes),
        labels_by_transition=labels_by_transition,
        noise_transition_ids=tuple(sorted(noise)),
        selected_parameters={
            "min_cluster_size": int(min_cluster_size),
            "min_samples": int(min_samples),
        },
        stability_ari=float(stability),
        eligible_coverage=float(coverage),
    )


@dataclass
class TransformationPrototypeMemory:
    """Retrieve transformation analogies and update only observed evidence."""

    registry: ClusterRegistry
    _observations: dict[str, dict[str, int]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._observations = {
            item.prototype_id: {
                "support": int(item.support),
                "productive": int(item.productive_observations),
                "risk": int(item.risk_observations),
            }
            for item in self.registry.prototypes
        }

    def retrieve(
        self,
        vector: Sequence[float],
        *,
        action_family: str = "",
        uncertainty: float = 0.0,
        maximum_matches: int = 8,
    ) -> tuple[TransformationMatch, ...]:
        if uncertainty > 1.0:
            return ()
        query = np.asarray(vector, dtype=np.float32)
        query = query / max(float(np.linalg.norm(query)), 1e-8)
        output = []
        for prototype in self.registry.prototypes:
            if (
                action_family
                and prototype.action_families
                and action_family not in prototype.action_families
            ):
                continue
            similarity = float(
                query @ np.asarray(prototype.centroid, dtype=np.float32)
            )
            if similarity < prototype.assignment_threshold:
                continue
            evidence = self._observations[prototype.prototype_id]
            support = evidence["support"]
            output.append(
                TransformationMatch(
                    prototype_id=prototype.prototype_id,
                    similarity=similarity,
                    productive_probability=(
                        evidence["productive"] + 1.0
                    )
                    / (support + 2.0),
                    risk_probability=(evidence["risk"] + 1.0) / (support + 2.0),
                    support=support,
                    game_diversity=len(prototype.games),
                    candidate_alias=prototype.candidate_alias,
                )
            )
        output.sort(
            key=lambda item: (
                item.similarity,
                item.support,
                item.game_diversity,
                item.prototype_id,
            ),
            reverse=True,
        )
        return tuple(output[: max(1, int(maximum_matches))])

    def assign(
        self,
        vector: Sequence[float],
        *,
        action_family: str = "",
        uncertainty: float = 0.0,
    ) -> str:
        matches = self.retrieve(
            vector,
            action_family=action_family,
            uncertainty=uncertainty,
            maximum_matches=1,
        )
        return matches[0].prototype_id if matches else "unknown"

    def observe(
        self,
        prototype_id: str,
        *,
        productive: bool,
        risk: bool,
    ) -> None:
        if prototype_id == "unknown":
            return
        if prototype_id not in self._observations:
            raise ValueError("cannot update an unknown SAGE-MT prototype")
        evidence = self._observations[prototype_id]
        evidence["support"] += 1
        evidence["productive"] += int(bool(productive))
        evidence["risk"] += int(bool(risk))

    def snapshot(self) -> Mapping[str, Any]:
        return {
            "prototype_count": len(self.registry.prototypes),
            "evidence": {
                key: dict(value)
                for key, value in sorted(self._observations.items())
            },
        }


__all__ = [
    "CLUSTER_FORMAT_VERSION",
    "ClusterRegistry",
    "TransformationMatch",
    "TransformationPrototype",
    "TransformationPrototypeMemory",
    "fit_cluster_registry",
]
