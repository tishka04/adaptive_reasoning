"""Reliability-gated target-local hierarchy for SAGE.T12.6.1c."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .future_viability_hierarchy import (
    HierarchicalFutureViabilityModel,
    HierarchicalViabilityObservation,
)

RELIABILITY_HIERARCHY_MODEL_FORMAT = (
    "sage-t12.6.1c-reliability-gated-hierarchy-model-v1"
)

RELIABILITY_CANDIDATES: Mapping[str, tuple[int, float]] = {
    "exact_span2_range0": (2, 0.0),
    "exact_span2_range1": (2, 1.0),
    "exact_span2_range2": (2, 2.0),
}


def _canonical(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    )


def _checksum(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReliabilitySupport:
    key: str
    observations: int
    mean_value: float
    search_seeds: tuple[int, ...]
    lineage_seeds: tuple[int, ...]
    minimum_value: float
    maximum_value: float
    value_range: float
    population_variance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "lineage_seeds": list(self.lineage_seeds),
            "maximum_value": self.maximum_value,
            "mean_value": self.mean_value,
            "minimum_value": self.minimum_value,
            "observations": self.observations,
            "population_variance": self.population_variance,
            "search_seeds": list(self.search_seeds),
            "value_range": self.value_range,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReliabilitySupport:
        return cls(
            key=str(payload["key"]),
            observations=int(payload["observations"]),
            mean_value=float(payload["mean_value"]),
            search_seeds=tuple(int(value) for value in payload["search_seeds"]),
            lineage_seeds=tuple(int(value) for value in payload["lineage_seeds"]),
            minimum_value=float(payload["minimum_value"]),
            maximum_value=float(payload["maximum_value"]),
            value_range=float(payload["value_range"]),
            population_variance=float(payload["population_variance"]),
        )


class ReliabilityGatedFutureViabilityModel:
    """Use exact support only when it is cross-seed and label-stable."""

    def __init__(
        self,
        *,
        target_field: str,
        radius: int,
        minimum_signature_support: int,
        minimum_exact_seed_span: int,
        maximum_exact_label_range: float,
        exact_support: Mapping[str, ReliabilitySupport],
        composition_support: Mapping[str, ReliabilitySupport],
        backoff_support: Mapping[str, ReliabilitySupport],
        global_mean: float,
    ) -> None:
        if target_field not in {"productive_reach", "immediate_score"}:
            raise ValueError("unsupported T12.6.1c hierarchy target")
        if minimum_exact_seed_span < 2:
            raise ValueError("T12.6.1c exact support must span at least two seeds")
        self.target_field = str(target_field)
        self.radius = int(radius)
        self.minimum_signature_support = int(minimum_signature_support)
        self.minimum_exact_seed_span = int(minimum_exact_seed_span)
        self.maximum_exact_label_range = float(maximum_exact_label_range)
        self.exact_support = dict(exact_support)
        self.composition_support = dict(composition_support)
        self.backoff_support = dict(backoff_support)
        self.global_mean = float(global_mean)

    @staticmethod
    def _support(
        grouped: Mapping[str, Sequence[HierarchicalViabilityObservation]],
        *,
        target_field: str,
        minimum: int,
    ) -> dict[str, ReliabilitySupport]:
        output = {}
        for key, rows in grouped.items():
            if len(rows) < int(minimum):
                continue
            values = [float(getattr(row.base, target_field)) for row in rows]
            mean = sum(values) / len(values)
            output[key] = ReliabilitySupport(
                key=key,
                observations=len(rows),
                mean_value=mean,
                search_seeds=tuple(sorted({row.base.search_seed for row in rows})),
                lineage_seeds=tuple(sorted({row.base.lineage_seed for row in rows})),
                minimum_value=min(values),
                maximum_value=max(values),
                value_range=max(values) - min(values),
                population_variance=(
                    sum((value - mean) ** 2 for value in values) / len(values)
                ),
            )
        return output

    @classmethod
    def fit(
        cls,
        observations: Sequence[HierarchicalViabilityObservation],
        *,
        target_field: str,
        radius: int,
        minimum_signature_support: int,
        minimum_exact_seed_span: int,
        maximum_exact_label_range: float,
    ) -> ReliabilityGatedFutureViabilityModel:
        if not observations:
            raise ValueError("T12.6.1c cannot fit an empty hierarchy")
        exact: dict[str, list[HierarchicalViabilityObservation]] = defaultdict(list)
        composition: dict[str, list[HierarchicalViabilityObservation]] = defaultdict(
            list
        )
        backoff: dict[str, list[HierarchicalViabilityObservation]] = defaultdict(list)
        for item in observations:
            exact[item.base.local_signature].append(item)
            composition[item.composition_signature].append(item)
            backoff[item.base.backoff_key].append(item)
        values = [float(getattr(item.base, target_field)) for item in observations]
        return cls(
            target_field=target_field,
            radius=radius,
            minimum_signature_support=minimum_signature_support,
            minimum_exact_seed_span=minimum_exact_seed_span,
            maximum_exact_label_range=maximum_exact_label_range,
            exact_support=cls._support(
                exact,
                target_field=target_field,
                minimum=minimum_signature_support,
            ),
            composition_support=cls._support(
                composition,
                target_field=target_field,
                minimum=minimum_signature_support,
            ),
            backoff_support=cls._support(
                backoff,
                target_field=target_field,
                minimum=1,
            ),
            global_mean=sum(values) / len(values),
        )

    def score_with_audit(
        self, item: HierarchicalViabilityObservation
    ) -> tuple[float, str, Mapping[str, Any]]:
        exact = self.exact_support.get(item.base.local_signature)
        rejection_reasons = []
        if exact is not None:
            if len(exact.search_seeds) < self.minimum_exact_seed_span:
                rejection_reasons.append("insufficient_search_seed_span")
            if exact.value_range > self.maximum_exact_label_range:
                rejection_reasons.append("excessive_label_range")
            if not rejection_reasons:
                return (
                    exact.mean_value,
                    "reliable_exact_local_signature",
                    {
                        "exact_candidate_present": True,
                        "exact_candidate_reliable": True,
                        "exact_rejection_reasons": [],
                        "support": exact.to_dict(),
                    },
                )
        composition = self.composition_support.get(item.composition_signature)
        if composition is not None:
            return (
                composition.mean_value,
                "local_composition_signature",
                {
                    "exact_candidate_present": exact is not None,
                    "exact_candidate_reliable": False,
                    "exact_rejection_reasons": rejection_reasons,
                    "support": composition.to_dict(),
                },
            )
        backoff = self.backoff_support.get(item.base.backoff_key)
        if backoff is not None:
            return (
                backoff.mean_value,
                "action_family_backoff",
                {
                    "exact_candidate_present": exact is not None,
                    "exact_candidate_reliable": False,
                    "exact_rejection_reasons": rejection_reasons,
                    "support": backoff.to_dict(),
                },
            )
        return (
            self.global_mean,
            "global_backoff",
            {
                "exact_candidate_present": exact is not None,
                "exact_candidate_reliable": False,
                "exact_rejection_reasons": rejection_reasons,
                "support": None,
            },
        )

    def score(self, item: HierarchicalViabilityObservation) -> tuple[float, str]:
        value, tier, _ = self.score_with_audit(item)
        return value, tier

    def to_dict(self) -> dict[str, Any]:
        def rows(values: Mapping[str, ReliabilitySupport]) -> list[dict[str, Any]]:
            return [values[key].to_dict() for key in sorted(values)]

        payload = {
            "backoff_support": rows(self.backoff_support),
            "composition_support": rows(self.composition_support),
            "exact_support": rows(self.exact_support),
            "format_version": RELIABILITY_HIERARCHY_MODEL_FORMAT,
            "global_mean": self.global_mean,
            "maximum_exact_label_range": self.maximum_exact_label_range,
            "minimum_exact_seed_span": self.minimum_exact_seed_span,
            "minimum_signature_support": self.minimum_signature_support,
            "radius": self.radius,
            "target_field": self.target_field,
        }
        return {**payload, "model_checksum": _checksum(payload)}

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> ReliabilityGatedFutureViabilityModel:
        if payload.get("format_version") != RELIABILITY_HIERARCHY_MODEL_FORMAT:
            raise ValueError("unsupported T12.6.1c reliability model")
        unsigned = dict(payload)
        checksum = str(unsigned.pop("model_checksum"))
        if _checksum(unsigned) != checksum:
            raise ValueError("T12.6.1c reliability model checksum mismatch")

        def load(rows: Sequence[Mapping[str, Any]]) -> dict[str, ReliabilitySupport]:
            return {
                str(row["key"]): ReliabilitySupport.from_dict(row) for row in rows
            }

        return cls(
            target_field=str(payload["target_field"]),
            radius=int(payload["radius"]),
            minimum_signature_support=int(payload["minimum_signature_support"]),
            minimum_exact_seed_span=int(payload["minimum_exact_seed_span"]),
            maximum_exact_label_range=float(payload["maximum_exact_label_range"]),
            exact_support=load(payload.get("exact_support", ())),
            composition_support=load(payload.get("composition_support", ())),
            backoff_support=load(payload.get("backoff_support", ())),
            global_mean=float(payload["global_mean"]),
        )


def _selected_index(
    rows: Sequence[HierarchicalViabilityObservation], scores: Sequence[float]
) -> int:
    return max(
        range(len(rows)),
        key=lambda index: (float(scores[index]), rows[index].base.action_key),
    )


def summarize_reliability_cells(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(cells)
    future = sum(bool(row["future_binding_hit"]) for row in cells)
    immediate = sum(bool(row["immediate_binding_hit"]) for row in cells)
    incumbent = sum(bool(row["incumbent_binding_hit"]) for row in cells)
    swap = sum(bool(row["binding_swap_hit"]) for row in cells)
    supported = sum(bool(row["hierarchy_supported"]) for row in cells)
    rejected = sum(bool(row["exact_rejection_exercised"]) for row in cells)
    unique = sum(bool(row["unique_top_score"]) for row in cells)
    recommendations = sum(bool(row["recommendation_issued"]) for row in cells)
    recommendation_hits = sum(bool(row["recommendation_hit"]) for row in cells)
    tiers = (
        "reliable_exact_local_signature",
        "local_composition_signature",
        "action_family_backoff",
        "global_backoff",
    )
    return {
        "binding_swap_hits": swap,
        "binding_swap_top1_accuracy": swap / max(1, count),
        "eligible_groups": count,
        "exact_rejection_exercised_groups": rejected,
        "exact_rejection_exercised_rate": rejected / max(1, count),
        "future_binding_hits": future,
        "future_binding_top1_accuracy": future / max(1, count),
        "future_gain_over_binding_swap": (future - swap) / max(1, count),
        "future_gain_over_immediate": (future - immediate) / max(1, count),
        "future_gain_over_incumbent": (future - incumbent) / max(1, count),
        "hierarchy_coverage": supported / max(1, count),
        "immediate_binding_hits": immediate,
        "immediate_binding_top1_accuracy": immediate / max(1, count),
        "incumbent_binding_hits": incumbent,
        "incumbent_binding_top1_accuracy": incumbent / max(1, count),
        "recommendation_accuracy": recommendation_hits / max(1, recommendations),
        "recommendation_coverage": recommendations / max(1, count),
        "recommendation_hits": recommendation_hits,
        "recommendations": recommendations,
        "selected_support_tier_counts": {
            tier: sum(row["selected_support_tier"] == tier for row in cells)
            for tier in tiers
        },
        "unique_top_rate": unique / max(1, count),
    }


def evaluate_reliability_gated_ranking(
    observations: Sequence[HierarchicalViabilityObservation],
    *,
    future_model: ReliabilityGatedFutureViabilityModel,
    immediate_model: ReliabilityGatedFutureViabilityModel,
    incumbent_model: HierarchicalFutureViabilityModel,
    binding_shift: int,
) -> dict[str, Any]:
    grouped: dict[str, list[HierarchicalViabilityObservation]] = defaultdict(list)
    for item in observations:
        grouped[item.base.group_id].append(item)
    cells = []
    for group_id, raw in sorted(grouped.items()):
        rows = sorted(raw, key=lambda item: item.base.action_key)
        if len(rows) < 2 or len({row.base.productive_reach for row in rows}) < 2:
            continue
        future = [future_model.score_with_audit(row) for row in rows]
        immediate = [immediate_model.score(row) for row in rows]
        incumbent = [incumbent_model.score(row) for row in rows]
        future_scores = [float(value[0]) for value in future]
        shift = int(binding_shift) % len(rows)
        swapped_scores = future_scores[shift:] + future_scores[:shift]
        future_index = _selected_index(rows, future_scores)
        immediate_index = _selected_index(rows, [value[0] for value in immediate])
        incumbent_index = _selected_index(rows, [value[0] for value in incumbent])
        swapped_index = _selected_index(rows, swapped_scores)
        best = max(row.base.productive_reach for row in rows)
        future_hit = rows[future_index].base.productive_reach == best
        unique_top = sum(score == max(future_scores) for score in future_scores) == 1
        selected_tier = future[future_index][1]
        supported = selected_tier in {
            "reliable_exact_local_signature",
            "local_composition_signature",
        }
        recommendation = bool(unique_top and supported)
        cells.append(
            {
                "arm": rows[0].base.arm,
                "binding_swap_hit": rows[swapped_index].base.productive_reach == best,
                "exact_rejection_exercised": any(
                    bool(value[2]["exact_candidate_present"])
                    and not bool(value[2]["exact_candidate_reliable"])
                    for value in future
                ),
                "future_binding_hit": future_hit,
                "group_id": group_id,
                "hierarchy_supported": supported,
                "immediate_binding_hit": (
                    rows[immediate_index].base.productive_reach == best
                ),
                "incumbent_binding_hit": (
                    rows[incumbent_index].base.productive_reach == best
                ),
                "incumbent_selected_support_tier": incumbent[incumbent_index][1],
                "lineage_seed": rows[0].base.lineage_seed,
                "recommendation_hit": bool(recommendation and future_hit),
                "recommendation_issued": recommendation,
                "search_seed": rows[0].base.search_seed,
                "selected_exact_audit": dict(future[future_index][2]),
                "selected_support_tier": selected_tier,
                "unique_top_score": unique_top,
            }
        )
    return {"cells": cells, "metrics": summarize_reliability_cells(cells)}


def crossfit_reliability_gated_viability(
    observations: Sequence[HierarchicalViabilityObservation],
    *,
    search_seeds: Sequence[int],
    radius: int,
    minimum_signature_support: int,
    minimum_exact_seed_span: int,
    maximum_exact_label_range: float,
    binding_shift: int,
) -> dict[str, Any]:
    folds = []
    all_cells = []
    for holdout in search_seeds:
        training = tuple(
            row for row in observations if row.base.search_seed != int(holdout)
        )
        evaluation = tuple(
            row for row in observations if row.base.search_seed == int(holdout)
        )
        common = {
            "radius": radius,
            "minimum_signature_support": minimum_signature_support,
            "minimum_exact_seed_span": minimum_exact_seed_span,
            "maximum_exact_label_range": maximum_exact_label_range,
        }
        future = ReliabilityGatedFutureViabilityModel.fit(
            training, target_field="productive_reach", **common
        )
        immediate = ReliabilityGatedFutureViabilityModel.fit(
            training, target_field="immediate_score", **common
        )
        incumbent = HierarchicalFutureViabilityModel.fit(
            training,
            target_field="productive_reach",
            radius=radius,
            minimum_signature_support=minimum_signature_support,
        )
        audit = evaluate_reliability_gated_ranking(
            evaluation,
            future_model=future,
            immediate_model=immediate,
            incumbent_model=incumbent,
            binding_shift=binding_shift,
        )
        metrics = dict(audit["metrics"])
        metrics["per_lineage"] = {
            str(lineage): summarize_reliability_cells(
                [
                    cell
                    for cell in audit["cells"]
                    if int(cell["lineage_seed"]) == int(lineage)
                ]
            )
            for lineage in sorted(
                {int(cell["lineage_seed"]) for cell in audit["cells"]}
            )
        }
        folds.append(
            {
                "holdout_search_seed": int(holdout),
                "metrics": metrics,
                "training_observations": len(training),
            }
        )
        all_cells.extend(audit["cells"])
    return {
        "folds": folds,
        "micro_metrics": summarize_reliability_cells(all_cells),
    }


def evaluate_reliability_candidates(
    observations: Sequence[HierarchicalViabilityObservation],
    *,
    search_seeds: Sequence[int],
    radius: int,
    minimum_signature_support: int,
    candidate_names: Sequence[str],
    binding_shift: int,
) -> dict[str, Any]:
    results = {}
    for name in candidate_names:
        if name not in RELIABILITY_CANDIDATES:
            raise ValueError(f"unknown T12.6.1c reliability candidate: {name}")
        seed_span, label_range = RELIABILITY_CANDIDATES[name]
        results[name] = crossfit_reliability_gated_viability(
            observations,
            search_seeds=search_seeds,
            radius=radius,
            minimum_signature_support=minimum_signature_support,
            minimum_exact_seed_span=seed_span,
            maximum_exact_label_range=label_range,
            binding_shift=binding_shift,
        )
    ordered = tuple(candidate_names)

    def selection_key(name: str) -> tuple[float, float, float, float, int]:
        result = results[name]
        metrics = result["micro_metrics"]
        worst = min(
            float(fold["metrics"]["future_binding_top1_accuracy"])
            for fold in result["folds"]
        )
        return (
            worst,
            float(metrics["future_binding_top1_accuracy"]),
            float(metrics["future_gain_over_incumbent"]),
            float(metrics["hierarchy_coverage"]),
            -ordered.index(name),
        )

    selected = max(ordered, key=selection_key)
    summary = {}
    for name, result in results.items():
        metrics = result["micro_metrics"]
        summary[name] = {
            "future_binding_top1_accuracy": metrics[
                "future_binding_top1_accuracy"
            ],
            "future_gain_over_incumbent": metrics["future_gain_over_incumbent"],
            "hierarchy_coverage": metrics["hierarchy_coverage"],
            "exact_rejection_exercised_rate": metrics[
                "exact_rejection_exercised_rate"
            ],
            "worst_fold_top1_accuracy": min(
                float(fold["metrics"]["future_binding_top1_accuracy"])
                for fold in result["folds"]
            ),
        }
    return {
        "candidate_results": results,
        "candidate_summary": summary,
        "selected_candidate": selected,
        "selection_criterion": (
            "lexicographic(worst_fold_accuracy,micro_accuracy,"
            "gain_over_exact_first,coverage,frozen_candidate_precedence)"
        ),
    }


__all__ = [
    "RELIABILITY_CANDIDATES",
    "RELIABILITY_HIERARCHY_MODEL_FORMAT",
    "ReliabilityGatedFutureViabilityModel",
    "ReliabilitySupport",
    "crossfit_reliability_gated_viability",
    "evaluate_reliability_candidates",
    "evaluate_reliability_gated_ranking",
    "summarize_reliability_cells",
]
