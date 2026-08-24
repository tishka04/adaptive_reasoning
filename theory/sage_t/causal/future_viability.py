"""Target-local future-viability grounding for the offline SAGE.T12.6 audit."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .archive import _action_from_payload, abstract_state_from_payload
from .hazard_diversity_model import local_hazard_signature

FUTURE_VIABILITY_MODEL_FORMAT = "sage-t12.6-future-viability-model-v1"


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
class FutureViabilityObservation:
    """One pre-action binding with an archive-derived future reach label."""

    group_id: str
    corpus: str
    search_seed: int
    lineage_seed: int
    arm: str
    source_exact_hash: str
    action_key: str
    action_name: str
    coordinate_grounded: bool
    local_signature: str
    productive_reach: int
    immediate_score: int
    terminal: bool
    changed: bool
    novel: bool

    @property
    def backoff_key(self) -> str:
        return f"{self.action_name}|coordinate={int(self.coordinate_grounded)}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExtractionResult:
    observations: tuple[FutureViabilityObservation, ...]
    metrics: Mapping[str, Any]


@dataclass(frozen=True)
class ViabilitySupport:
    key: str
    observations: int
    mean_value: float
    search_seeds: tuple[int, ...]
    lineage_seeds: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FutureViabilityModel:
    """Factorized table with a target-local signature and action-family backoff."""

    def __init__(
        self,
        *,
        target_field: str,
        radius: int,
        minimum_signature_support: int,
        signature_support: Mapping[str, ViabilitySupport],
        backoff_support: Mapping[str, ViabilitySupport],
        global_mean: float,
    ) -> None:
        if target_field not in {"productive_reach", "immediate_score"}:
            raise ValueError("unsupported T12.6 model target")
        self.target_field = str(target_field)
        self.radius = int(radius)
        self.minimum_signature_support = int(minimum_signature_support)
        self.signature_support = dict(signature_support)
        self.backoff_support = dict(backoff_support)
        self.global_mean = float(global_mean)

    @classmethod
    def fit(
        cls,
        observations: Sequence[FutureViabilityObservation],
        *,
        target_field: str,
        radius: int,
        minimum_signature_support: int,
    ) -> FutureViabilityModel:
        if not observations:
            raise ValueError("T12.6 cannot fit an empty viability model")
        exact: dict[str, list[FutureViabilityObservation]] = defaultdict(list)
        backoff: dict[str, list[FutureViabilityObservation]] = defaultdict(list)
        values: list[float] = []
        for item in observations:
            value = float(getattr(item, target_field))
            exact[item.local_signature].append(item)
            backoff[item.backoff_key].append(item)
            values.append(value)

        def support(
            grouped: Mapping[str, Sequence[FutureViabilityObservation]],
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
                    mean_value=sum(float(getattr(row, target_field)) for row in rows)
                    / len(rows),
                    search_seeds=tuple(sorted({row.search_seed for row in rows})),
                    lineage_seeds=tuple(sorted({row.lineage_seed for row in rows})),
                )
            return output

        return cls(
            target_field=target_field,
            radius=radius,
            minimum_signature_support=minimum_signature_support,
            signature_support=support(
                exact,
                minimum=minimum_signature_support,
            ),
            backoff_support=support(backoff, minimum=1),
            global_mean=sum(values) / len(values),
        )

    def score(self, item: FutureViabilityObservation) -> tuple[float, str]:
        exact = self.signature_support.get(item.local_signature)
        if exact is not None:
            return exact.mean_value, "target_local_signature"
        backoff = self.backoff_support.get(item.backoff_key)
        if backoff is not None:
            return backoff.mean_value, "action_family_backoff"
        return self.global_mean, "global_backoff"

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "format_version": FUTURE_VIABILITY_MODEL_FORMAT,
            "backoff_support": [
                self.backoff_support[key].to_dict()
                for key in sorted(self.backoff_support)
            ],
            "global_mean": self.global_mean,
            "minimum_signature_support": self.minimum_signature_support,
            "radius": self.radius,
            "signature_support": [
                self.signature_support[key].to_dict()
                for key in sorted(self.signature_support)
            ],
            "target_field": self.target_field,
        }
        return {**payload, "model_checksum": _checksum(payload)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FutureViabilityModel:
        if payload.get("format_version") != FUTURE_VIABILITY_MODEL_FORMAT:
            raise ValueError("unsupported T12.6 future-viability model")
        unsigned = dict(payload)
        checksum = str(unsigned.pop("model_checksum"))
        if _checksum(unsigned) != checksum:
            raise ValueError("T12.6 future-viability model checksum mismatch")

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
            signature_support=load(payload.get("signature_support", ())),
            backoff_support=load(payload.get("backoff_support", ())),
            global_mean=float(payload["global_mean"]),
        )


def _productive_reach(
    cell_id: str,
    *,
    outgoing: Mapping[str, Sequence[Mapping[str, Any]]],
    remaining_horizon: int,
    visited: frozenset[str],
) -> int:
    if remaining_horizon <= 0 or cell_id in visited:
        return 0
    next_visited = visited | {cell_id}
    return max(
        [0]
        + [
            1
            + _productive_reach(
                str(edge["target_cell_id"]),
                outgoing=outgoing,
                remaining_horizon=remaining_horizon - 1,
                visited=next_visited,
            )
            for edge in outgoing.get(cell_id, ())
            if not bool(edge.get("terminal")) and bool(edge.get("changed"))
        ]
    )


def extract_future_viability_observations(
    *,
    archive_metas: Sequence[Mapping[str, Any]],
    root: Path,
    corpus: str,
    expected_search_seeds: Sequence[int],
    expected_lineages: Sequence[int],
    expected_arms: Sequence[str],
    future_horizon: int,
    local_radius: int,
) -> ExtractionResult:
    """Extract deterministic decision groups from signed symbolic archives."""

    expected_seed_set = {int(value) for value in expected_search_seeds}
    expected_lineage_set = {int(value) for value in expected_lineages}
    expected_arm_set = {str(value) for value in expected_arms}
    observations: list[FutureViabilityObservation] = []
    archive_keys: set[tuple[int, int, str]] = set()
    conflicts = 0
    total_edges = 0
    multi_action_groups = 0
    label_variable_groups = 0
    for meta in archive_metas:
        search_seed = int(meta["search_seed"])
        lineage_seed = int(meta["lineage_seed"])
        arm = str(meta["arm"])
        if search_seed not in expected_seed_set:
            raise ValueError("T12.6 archive has an unregistered search seed")
        if lineage_seed not in expected_lineage_set:
            raise ValueError("T12.6 archive has an unregistered lineage")
        if arm not in expected_arm_set:
            raise ValueError("T12.6 archive has an unregistered arm")
        archive_key = (search_seed, lineage_seed, arm)
        if archive_key in archive_keys:
            raise ValueError("T12.6 archive condition is duplicated")
        archive_keys.add(archive_key)
        path = Path(str(meta["path"]))
        if not path.is_absolute():
            path = root / path
        payload = json.loads(path.read_text(encoding="utf-8"))
        cells = {
            str(row["cell_id"]): abstract_state_from_payload(dict(row["state"]))
            for row in payload.get("cells", ())
        }
        outgoing: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for edge in payload.get("edges", ()):
            outgoing[str(edge["source_cell_id"])].append(edge)
            total_edges += 1
        for source_cell_id, edges in outgoing.items():
            by_action: dict[str, Mapping[str, Any]] = {}
            outcomes: dict[str, tuple[Any, ...]] = {}
            for edge in edges:
                action = _action_from_payload(dict(edge["action"]))
                outcome = (
                    bool(edge.get("terminal")),
                    bool(edge.get("changed")),
                    bool(edge.get("novel")),
                    str(edge.get("target_cell_id")),
                )
                if action.key in outcomes and outcomes[action.key] != outcome:
                    conflicts += 1
                    continue
                outcomes[action.key] = outcome
                by_action[action.key] = edge
            if len(by_action) < 2:
                continue
            multi_action_groups += 1
            source_state = cells.get(source_cell_id)
            if source_state is None:
                raise ValueError("T12.6 archive edge has no source state")
            group_id = _checksum(
                {
                    "arm": arm,
                    "corpus": corpus,
                    "lineage_seed": lineage_seed,
                    "search_seed": search_seed,
                    "source_cell_id": source_cell_id,
                }
            )
            group_rows: list[FutureViabilityObservation] = []
            for action_key, edge in sorted(by_action.items()):
                action = _action_from_payload(dict(edge["action"]))
                terminal = bool(edge.get("terminal"))
                reach = (
                    0
                    if terminal
                    else _productive_reach(
                        str(edge["target_cell_id"]),
                        outgoing=outgoing,
                        remaining_horizon=int(future_horizon),
                        visited=frozenset({source_cell_id}),
                    )
                )
                group_rows.append(
                    FutureViabilityObservation(
                        group_id=group_id,
                        corpus=str(corpus),
                        search_seed=search_seed,
                        lineage_seed=lineage_seed,
                        arm=arm,
                        source_exact_hash=str(edge["source_exact_hash"]),
                        action_key=action_key,
                        action_name=action.action_name,
                        coordinate_grounded=bool(action.action_data),
                        local_signature=local_hazard_signature(
                            source_state,
                            action,
                            radius=int(local_radius),
                        ),
                        productive_reach=reach,
                        immediate_score=(
                            4 * int(not terminal)
                            + 2 * int(bool(edge.get("changed")))
                            + int(bool(edge.get("novel")))
                        ),
                        terminal=terminal,
                        changed=bool(edge.get("changed")),
                        novel=bool(edge.get("novel")),
                    )
                )
            if len({item.productive_reach for item in group_rows}) > 1:
                label_variable_groups += 1
            observations.extend(group_rows)

    expected_archive_keys = {
        (seed, lineage, arm)
        for seed in expected_seed_set
        for lineage in expected_lineage_set
        for arm in expected_arm_set
    }
    metrics = {
        "all_archive_conditions_present": archive_keys == expected_archive_keys,
        "archive_condition_count": len(archive_keys),
        "duplicate_action_conflicts": conflicts,
        "expected_archive_condition_count": len(expected_archive_keys),
        "label_variable_group_count": label_variable_groups,
        "multi_action_group_count": multi_action_groups,
        "observation_count": len(observations),
        "search_seeds": sorted({item.search_seed for item in observations}),
        "source_lineages": sorted({item.lineage_seed for item in observations}),
        "total_archive_edges": total_edges,
    }
    return ExtractionResult(tuple(observations), metrics)


def evaluate_future_viability_ranking(
    observations: Sequence[FutureViabilityObservation],
    *,
    future_model: FutureViabilityModel,
    immediate_model: FutureViabilityModel,
    binding_shift: int,
) -> dict[str, Any]:
    """Compare correct future binding with immediate and score-swap controls."""

    grouped: dict[str, list[FutureViabilityObservation]] = defaultdict(list)
    for item in observations:
        grouped[item.group_id].append(item)
    cells: list[dict[str, Any]] = []
    for group_id, raw in sorted(grouped.items()):
        rows = sorted(raw, key=lambda item: item.action_key)
        if len(rows) < 2 or len({item.productive_reach for item in rows}) < 2:
            continue
        future = [future_model.score(item) for item in rows]
        immediate = [immediate_model.score(item) for item in rows]
        shift = int(binding_shift) % len(rows)
        swapped_scores = [value[0] for value in future[shift:] + future[:shift]]

        def selected_index(scores: Sequence[float]) -> int:
            return max(
                range(len(rows)),
                key=lambda index: (float(scores[index]), rows[index].action_key),
            )

        future_index = selected_index([item[0] for item in future])
        immediate_index = selected_index([item[0] for item in immediate])
        swapped_index = selected_index(swapped_scores)
        best = max(item.productive_reach for item in rows)
        cells.append(
            {
                "arm": rows[0].arm,
                "binding_swap_hit": rows[swapped_index].productive_reach == best,
                "future_binding_hit": rows[future_index].productive_reach == best,
                "future_selected_support_tier": future[future_index][1],
                "group_id": group_id,
                "immediate_binding_hit": (
                    rows[immediate_index].productive_reach == best
                ),
                "lineage_seed": rows[0].lineage_seed,
                "maximum_productive_reach": best,
                "search_seed": rows[0].search_seed,
            }
        )

    def summary(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        count = len(selected)
        future_hits = sum(bool(item["future_binding_hit"]) for item in selected)
        immediate_hits = sum(bool(item["immediate_binding_hit"]) for item in selected)
        swap_hits = sum(bool(item["binding_swap_hit"]) for item in selected)
        exact_support = sum(
            item["future_selected_support_tier"] == "target_local_signature"
            for item in selected
        )
        return {
            "binding_swap_hits": swap_hits,
            "binding_swap_top1_accuracy": swap_hits / max(1, count),
            "eligible_groups": count,
            "future_binding_hits": future_hits,
            "future_binding_top1_accuracy": future_hits / max(1, count),
            "future_gain_over_binding_swap": (future_hits - swap_hits)
            / max(1, count),
            "future_gain_over_immediate": (future_hits - immediate_hits)
            / max(1, count),
            "immediate_binding_hits": immediate_hits,
            "immediate_binding_top1_accuracy": immediate_hits / max(1, count),
            "target_local_signature_coverage": exact_support / max(1, count),
        }

    overall = summary(cells)
    per_seed = {
        str(seed): summary([item for item in cells if item["search_seed"] == seed])
        for seed in sorted({int(item["search_seed"]) for item in cells})
    }
    per_lineage = {
        str(lineage): summary(
            [item for item in cells if item["lineage_seed"] == lineage]
        )
        for lineage in sorted({int(item["lineage_seed"]) for item in cells})
    }
    return {
        "cells": cells,
        "metrics": {
            **overall,
            "per_lineage": per_lineage,
            "per_search_seed": per_seed,
        },
    }


def crossfit_future_viability(
    observations: Sequence[FutureViabilityObservation],
    *,
    search_seeds: Sequence[int],
    radius: int,
    minimum_signature_support: int,
    binding_shift: int,
) -> dict[str, Any]:
    folds = []
    all_cells: list[dict[str, Any]] = []
    for holdout in search_seeds:
        training = tuple(item for item in observations if item.search_seed != holdout)
        evaluation = tuple(item for item in observations if item.search_seed == holdout)
        future_model = FutureViabilityModel.fit(
            training,
            target_field="productive_reach",
            radius=radius,
            minimum_signature_support=minimum_signature_support,
        )
        immediate_model = FutureViabilityModel.fit(
            training,
            target_field="immediate_score",
            radius=radius,
            minimum_signature_support=minimum_signature_support,
        )
        audit = evaluate_future_viability_ranking(
            evaluation,
            future_model=future_model,
            immediate_model=immediate_model,
            binding_shift=binding_shift,
        )
        folds.append(
            {
                "holdout_search_seed": int(holdout),
                "metrics": audit["metrics"],
                "training_observations": len(training),
            }
        )
        all_cells.extend(audit["cells"])

    def count_metrics(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        count = len(cells)
        future = sum(bool(item["future_binding_hit"]) for item in cells)
        immediate = sum(bool(item["immediate_binding_hit"]) for item in cells)
        swap = sum(bool(item["binding_swap_hit"]) for item in cells)
        exact = sum(
            item["future_selected_support_tier"] == "target_local_signature"
            for item in cells
        )
        return {
            "binding_swap_top1_accuracy": swap / max(1, count),
            "eligible_groups": count,
            "future_binding_top1_accuracy": future / max(1, count),
            "future_gain_over_binding_swap": (future - swap) / max(1, count),
            "future_gain_over_immediate": (future - immediate) / max(1, count),
            "immediate_binding_top1_accuracy": immediate / max(1, count),
            "target_local_signature_coverage": exact / max(1, count),
        }

    return {
        "folds": folds,
        "micro_metrics": count_metrics(all_cells),
    }


__all__ = [
    "FUTURE_VIABILITY_MODEL_FORMAT",
    "ExtractionResult",
    "FutureViabilityModel",
    "FutureViabilityObservation",
    "ViabilitySupport",
    "crossfit_future_viability",
    "evaluate_future_viability_ranking",
    "extract_future_viability_observations",
]
