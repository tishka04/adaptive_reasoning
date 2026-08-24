"""Hierarchical target-local future viability for SAGE.T12.6.1."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .archive import _action_from_payload, abstract_state_from_payload
from .future_viability import (
    ExtractionResult,
    FutureViabilityModel,
    FutureViabilityObservation,
    ViabilitySupport,
    extract_future_viability_observations,
)
from .hazard_diversity_model import local_hazard_descriptor

HIERARCHICAL_VIABILITY_MODEL_FORMAT = (
    "sage-t12.6.1-hierarchical-future-viability-model-v1"
)


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


def local_composition_signature(
    state: Any,
    action: Any,
    *,
    radius: int,
) -> str:
    """Coarsen an exact local geometry to a target-centred typed multiset."""

    descriptor = local_hazard_descriptor(state, action, radius=radius)
    entity_types = []
    for entity in descriptor["local_entities"]:
        attributes = dict(entity["attributes"])
        entity_types.append(
            {
                "area": str(attributes.get("area", "unknown")),
                "aspect": str(attributes.get("aspect", "unknown")),
                "roles": list(entity["roles"]),
            }
        )
    entity_types.sort(key=_canonical)
    return "composition_" + _checksum(
        {
            "action_name": descriptor["action_name"],
            "coordinate_grounded": descriptor["coordinate_grounded"],
            "entity_types": entity_types,
            "radius": int(radius),
        }
    )[:24]


@dataclass(frozen=True)
class HierarchicalViabilityObservation:
    base: FutureViabilityObservation
    composition_signature: str

    @property
    def value_key(self) -> tuple[int, int, str, str, str]:
        return (
            self.base.search_seed,
            self.base.lineage_seed,
            self.base.arm,
            self.base.source_exact_hash,
            self.base.action_key,
        )


@dataclass(frozen=True)
class HierarchicalExtractionResult:
    observations: tuple[HierarchicalViabilityObservation, ...]
    metrics: Mapping[str, Any]


class HierarchicalFutureViabilityModel:
    """Exact geometry, typed local composition, then action-family backoff."""

    def __init__(
        self,
        *,
        target_field: str,
        radius: int,
        minimum_signature_support: int,
        exact_support: Mapping[str, ViabilitySupport],
        composition_support: Mapping[str, ViabilitySupport],
        backoff_support: Mapping[str, ViabilitySupport],
        global_mean: float,
    ) -> None:
        if target_field not in {"productive_reach", "immediate_score"}:
            raise ValueError("unsupported T12.6.1 hierarchy target")
        self.target_field = str(target_field)
        self.radius = int(radius)
        self.minimum_signature_support = int(minimum_signature_support)
        self.exact_support = dict(exact_support)
        self.composition_support = dict(composition_support)
        self.backoff_support = dict(backoff_support)
        self.global_mean = float(global_mean)

    @classmethod
    def fit(
        cls,
        observations: Sequence[HierarchicalViabilityObservation],
        *,
        target_field: str,
        radius: int,
        minimum_signature_support: int,
    ) -> HierarchicalFutureViabilityModel:
        if not observations:
            raise ValueError("T12.6.1 cannot fit an empty hierarchy")
        exact: dict[str, list[HierarchicalViabilityObservation]] = defaultdict(list)
        composition: dict[str, list[HierarchicalViabilityObservation]] = defaultdict(
            list
        )
        backoff: dict[str, list[HierarchicalViabilityObservation]] = defaultdict(list)
        for item in observations:
            exact[item.base.local_signature].append(item)
            composition[item.composition_signature].append(item)
            backoff[item.base.backoff_key].append(item)

        def support(
            grouped: Mapping[str, Sequence[HierarchicalViabilityObservation]],
            *,
            minimum: int,
        ) -> dict[str, ViabilitySupport]:
            output = {}
            for key, rows in grouped.items():
                if len(rows) < int(minimum):
                    continue
                output[key] = ViabilitySupport(
                    key=key,
                    observations=len(rows),
                    mean_value=sum(
                        float(getattr(row.base, target_field)) for row in rows
                    )
                    / len(rows),
                    search_seeds=tuple(
                        sorted({row.base.search_seed for row in rows})
                    ),
                    lineage_seeds=tuple(
                        sorted({row.base.lineage_seed for row in rows})
                    ),
                )
            return output

        values = [float(getattr(item.base, target_field)) for item in observations]
        return cls(
            target_field=target_field,
            radius=radius,
            minimum_signature_support=minimum_signature_support,
            exact_support=support(exact, minimum=minimum_signature_support),
            composition_support=support(
                composition, minimum=minimum_signature_support
            ),
            backoff_support=support(backoff, minimum=1),
            global_mean=sum(values) / len(values),
        )

    def score(
        self, item: HierarchicalViabilityObservation
    ) -> tuple[float, str]:
        exact = self.exact_support.get(item.base.local_signature)
        if exact is not None:
            return exact.mean_value, "exact_local_signature"
        composition = self.composition_support.get(item.composition_signature)
        if composition is not None:
            return composition.mean_value, "local_composition_signature"
        backoff = self.backoff_support.get(item.base.backoff_key)
        if backoff is not None:
            return backoff.mean_value, "action_family_backoff"
        return self.global_mean, "global_backoff"

    def to_dict(self) -> dict[str, Any]:
        def rows(values: Mapping[str, ViabilitySupport]) -> list[dict[str, Any]]:
            return [values[key].to_dict() for key in sorted(values)]

        payload = {
            "backoff_support": rows(self.backoff_support),
            "composition_support": rows(self.composition_support),
            "exact_support": rows(self.exact_support),
            "format_version": HIERARCHICAL_VIABILITY_MODEL_FORMAT,
            "global_mean": self.global_mean,
            "minimum_signature_support": self.minimum_signature_support,
            "radius": self.radius,
            "target_field": self.target_field,
        }
        return {**payload, "model_checksum": _checksum(payload)}

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> HierarchicalFutureViabilityModel:
        if payload.get("format_version") != HIERARCHICAL_VIABILITY_MODEL_FORMAT:
            raise ValueError("unsupported T12.6.1 hierarchy model")
        unsigned = dict(payload)
        checksum = str(unsigned.pop("model_checksum"))
        if _checksum(unsigned) != checksum:
            raise ValueError("T12.6.1 hierarchy model checksum mismatch")

        def load(rows: Sequence[Mapping[str, Any]]) -> dict[str, ViabilitySupport]:
            return {
                str(row["key"]): ViabilitySupport(
                    key=str(row["key"]),
                    observations=int(row["observations"]),
                    mean_value=float(row["mean_value"]),
                    search_seeds=tuple(int(value) for value in row["search_seeds"]),
                    lineage_seeds=tuple(int(value) for value in row["lineage_seeds"]),
                )
                for row in rows
            }

        return cls(
            target_field=str(payload["target_field"]),
            radius=int(payload["radius"]),
            minimum_signature_support=int(payload["minimum_signature_support"]),
            exact_support=load(payload.get("exact_support", ())),
            composition_support=load(payload.get("composition_support", ())),
            backoff_support=load(payload.get("backoff_support", ())),
            global_mean=float(payload["global_mean"]),
        )


def extract_hierarchical_viability_observations(
    *,
    archive_metas: Sequence[Mapping[str, Any]],
    root: Path,
    corpus: str,
    expected_search_seeds: Sequence[int],
    expected_lineages: Sequence[int],
    expected_arms: Sequence[str],
    future_horizon: int,
    local_radius: int,
) -> HierarchicalExtractionResult:
    base: ExtractionResult = extract_future_viability_observations(
        archive_metas=archive_metas,
        root=root,
        corpus=corpus,
        expected_search_seeds=expected_search_seeds,
        expected_lineages=expected_lineages,
        expected_arms=expected_arms,
        future_horizon=future_horizon,
        local_radius=local_radius,
    )
    signatures: dict[tuple[int, int, str, str, str], str] = {}
    for meta in archive_metas:
        path = Path(str(meta["path"]))
        if not path.is_absolute():
            path = root / path
        payload = json.loads(path.read_text(encoding="utf-8"))
        states = {
            str(row["cell_id"]): abstract_state_from_payload(dict(row["state"]))
            for row in payload.get("cells", ())
        }
        for edge in payload.get("edges", ()):
            action = _action_from_payload(dict(edge["action"]))
            source_id = str(edge["source_cell_id"])
            state = states.get(source_id)
            if state is None:
                raise ValueError("T12.6.1 archive edge has no source state")
            key = (
                int(meta["search_seed"]),
                int(meta["lineage_seed"]),
                str(meta["arm"]),
                str(edge["source_exact_hash"]),
                action.key,
            )
            signature = local_composition_signature(
                state, action, radius=int(local_radius)
            )
            previous = signatures.get(key)
            if previous is not None and previous != signature:
                raise ValueError("T12.6.1 composition signature conflict")
            signatures[key] = signature
    observations = []
    for item in base.observations:
        key = (
            item.search_seed,
            item.lineage_seed,
            item.arm,
            item.source_exact_hash,
            item.action_key,
        )
        if key not in signatures:
            raise ValueError("T12.6.1 composition signature is missing")
        observations.append(
            HierarchicalViabilityObservation(
                base=item,
                composition_signature=signatures[key],
            )
        )
    return HierarchicalExtractionResult(tuple(observations), base.metrics)


def _selected_index(
    rows: Sequence[HierarchicalViabilityObservation], scores: Sequence[float]
) -> int:
    return max(
        range(len(rows)),
        key=lambda index: (float(scores[index]), rows[index].base.action_key),
    )


def evaluate_hierarchical_viability_ranking(
    observations: Sequence[HierarchicalViabilityObservation],
    *,
    future_model: HierarchicalFutureViabilityModel,
    immediate_model: HierarchicalFutureViabilityModel,
    incumbent_model: FutureViabilityModel,
    binding_shift: int,
) -> dict[str, Any]:
    grouped: dict[str, list[HierarchicalViabilityObservation]] = defaultdict(list)
    for item in observations:
        grouped[item.base.group_id].append(item)
    cells = []
    for group_id, raw in sorted(grouped.items()):
        rows = sorted(raw, key=lambda item: item.base.action_key)
        if (
            len(rows) < 2
            or len({item.base.productive_reach for item in rows}) < 2
        ):
            continue
        future = [future_model.score(item) for item in rows]
        immediate = [immediate_model.score(item) for item in rows]
        incumbent = [incumbent_model.score(item.base) for item in rows]
        future_scores = [float(value[0]) for value in future]
        shift = int(binding_shift) % len(rows)
        swapped_scores = future_scores[shift:] + future_scores[:shift]
        future_index = _selected_index(rows, future_scores)
        immediate_index = _selected_index(rows, [value[0] for value in immediate])
        incumbent_index = _selected_index(rows, [value[0] for value in incumbent])
        swapped_index = _selected_index(rows, swapped_scores)
        best = max(item.base.productive_reach for item in rows)
        unique_top = sum(score == max(future_scores) for score in future_scores) == 1
        selected_tier = future[future_index][1]
        incumbent_supported = (
            incumbent[incumbent_index][1] == "target_local_signature"
        )
        hierarchy_supported = selected_tier in {
            "exact_local_signature",
            "local_composition_signature",
        }
        recommendation_issued = bool(unique_top and hierarchy_supported)
        future_hit = rows[future_index].base.productive_reach == best
        cells.append(
            {
                "arm": rows[0].base.arm,
                "binding_swap_hit": rows[swapped_index].base.productive_reach
                == best,
                "future_binding_hit": future_hit,
                "group_id": group_id,
                "hierarchy_supported": hierarchy_supported,
                "immediate_binding_hit": rows[
                    immediate_index
                ].base.productive_reach
                == best,
                "incumbent_binding_hit": rows[
                    incumbent_index
                ].base.productive_reach
                == best,
                "incumbent_supported": incumbent_supported,
                "lineage_seed": rows[0].base.lineage_seed,
                "maximum_productive_reach": best,
                "recommendation_hit": bool(recommendation_issued and future_hit),
                "recommendation_issued": recommendation_issued,
                "search_seed": rows[0].base.search_seed,
                "selected_support_tier": selected_tier,
                "unique_top_score": unique_top,
            }
        )
    return {"cells": cells, "metrics": summarize_hierarchical_cells(cells)}


def summarize_hierarchical_cells(
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    count = len(cells)
    future = sum(bool(item["future_binding_hit"]) for item in cells)
    immediate = sum(bool(item["immediate_binding_hit"]) for item in cells)
    incumbent = sum(bool(item["incumbent_binding_hit"]) for item in cells)
    incumbent_supported = sum(bool(item["incumbent_supported"]) for item in cells)
    swap = sum(bool(item["binding_swap_hit"]) for item in cells)
    supported = sum(bool(item["hierarchy_supported"]) for item in cells)
    unique = sum(bool(item["unique_top_score"]) for item in cells)
    recommendations = sum(bool(item["recommendation_issued"]) for item in cells)
    recommendation_hits = sum(bool(item["recommendation_hit"]) for item in cells)
    tier_counts = {
        tier: sum(item["selected_support_tier"] == tier for item in cells)
        for tier in (
            "exact_local_signature",
            "local_composition_signature",
            "action_family_backoff",
            "global_backoff",
        )
    }
    return {
        "binding_swap_hits": swap,
        "binding_swap_top1_accuracy": swap / max(1, count),
        "eligible_groups": count,
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
        "incumbent_signature_coverage": incumbent_supported / max(1, count),
        "recommendation_accuracy": recommendation_hits / max(1, recommendations),
        "recommendation_coverage": recommendations / max(1, count),
        "recommendation_hits": recommendation_hits,
        "recommendations": recommendations,
        "selected_support_tier_counts": tier_counts,
        "unique_top_rate": unique / max(1, count),
    }


def crossfit_hierarchical_viability(
    observations: Sequence[HierarchicalViabilityObservation],
    *,
    search_seeds: Sequence[int],
    radius: int,
    minimum_signature_support: int,
    binding_shift: int,
) -> dict[str, Any]:
    folds = []
    all_cells = []
    for holdout in search_seeds:
        training = tuple(
            item for item in observations if item.base.search_seed != holdout
        )
        evaluation = tuple(
            item for item in observations if item.base.search_seed == holdout
        )
        future = HierarchicalFutureViabilityModel.fit(
            training,
            target_field="productive_reach",
            radius=radius,
            minimum_signature_support=minimum_signature_support,
        )
        immediate = HierarchicalFutureViabilityModel.fit(
            training,
            target_field="immediate_score",
            radius=radius,
            minimum_signature_support=minimum_signature_support,
        )
        incumbent = FutureViabilityModel.fit(
            tuple(item.base for item in training),
            target_field="productive_reach",
            radius=radius,
            minimum_signature_support=minimum_signature_support,
        )
        audit = evaluate_hierarchical_viability_ranking(
            evaluation,
            future_model=future,
            immediate_model=immediate,
            incumbent_model=incumbent,
            binding_shift=binding_shift,
        )
        metrics = dict(audit["metrics"])
        metrics["per_lineage"] = {
            str(lineage): summarize_hierarchical_cells(
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
        "micro_metrics": summarize_hierarchical_cells(all_cells),
    }


__all__ = [
    "HIERARCHICAL_VIABILITY_MODEL_FORMAT",
    "HierarchicalExtractionResult",
    "HierarchicalFutureViabilityModel",
    "HierarchicalViabilityObservation",
    "crossfit_hierarchical_viability",
    "evaluate_hierarchical_viability_ranking",
    "extract_hierarchical_viability_observations",
    "local_composition_signature",
    "summarize_hierarchical_cells",
]
