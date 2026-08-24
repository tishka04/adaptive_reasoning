"""Post-hoc seed-shift attribution for the SAGE.T12.6.1b audit."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from statistics import mean, median, pvariance
from typing import Any

from .future_viability_hierarchy import (
    HierarchicalFutureViabilityModel,
    HierarchicalViabilityObservation,
)

SEED_SHIFT_DIAGNOSTIC_FORMAT = (
    "sage-t12.6.1b-future-viability-seed-shift-diagnostic-v1"
)
SEED_SHIFT_DIAGNOSTIC_AXES = (
    "support_tier_attribution",
    "training_support_heterogeneity",
    "within_group_pairwise_concordance",
    "reference_seed_contrast",
    "leave_one_training_seed_stability",
    "lineage_and_archive_arm_stratification",
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


def _selected_index(
    rows: Sequence[HierarchicalViabilityObservation], scores: Sequence[float]
) -> int:
    return max(
        range(len(rows)),
        key=lambda index: (float(scores[index]), rows[index].base.action_key),
    )


def _eligible_groups(
    observations: Sequence[HierarchicalViabilityObservation],
) -> tuple[tuple[HierarchicalViabilityObservation, ...], ...]:
    grouped: dict[str, list[HierarchicalViabilityObservation]] = defaultdict(list)
    for item in observations:
        grouped[item.base.group_id].append(item)
    return tuple(
        tuple(sorted(rows, key=lambda item: item.base.action_key))
        for _, rows in sorted(grouped.items())
        if len(rows) >= 2
        and len({item.base.productive_reach for item in rows}) >= 2
    )


def _support_indices(
    observations: Sequence[HierarchicalViabilityObservation],
) -> tuple[
    Mapping[str, tuple[HierarchicalViabilityObservation, ...]],
    Mapping[str, tuple[HierarchicalViabilityObservation, ...]],
    Mapping[str, tuple[HierarchicalViabilityObservation, ...]],
]:
    exact: dict[str, list[HierarchicalViabilityObservation]] = defaultdict(list)
    composition: dict[str, list[HierarchicalViabilityObservation]] = defaultdict(list)
    family: dict[str, list[HierarchicalViabilityObservation]] = defaultdict(list)
    for item in observations:
        exact[item.base.local_signature].append(item)
        composition[item.composition_signature].append(item)
        family[item.base.backoff_key].append(item)
    return (
        {key: tuple(rows) for key, rows in exact.items()},
        {key: tuple(rows) for key, rows in composition.items()},
        {key: tuple(rows) for key, rows in family.items()},
    )


def _support_detail(
    item: HierarchicalViabilityObservation,
    *,
    model: HierarchicalFutureViabilityModel,
    exact: Mapping[str, Sequence[HierarchicalViabilityObservation]],
    composition: Mapping[str, Sequence[HierarchicalViabilityObservation]],
    family: Mapping[str, Sequence[HierarchicalViabilityObservation]],
) -> dict[str, Any]:
    score, tier = model.score(item)
    if tier == "exact_local_signature":
        key = item.base.local_signature
        rows = tuple(exact.get(key, ()))
    elif tier == "local_composition_signature":
        key = item.composition_signature
        rows = tuple(composition.get(key, ()))
    elif tier == "action_family_backoff":
        key = item.base.backoff_key
        rows = tuple(family.get(key, ()))
    else:
        key = "global"
        rows = tuple(row for values in family.values() for row in values)
    values = [int(row.base.productive_reach) for row in rows]
    histogram = Counter(values)
    return {
        "action_family": item.base.backoff_key,
        "arm_counts": dict(sorted(Counter(row.base.arm for row in rows).items())),
        "distinct_labels": len(histogram),
        "label_histogram": {
            str(value): histogram[value] for value in sorted(histogram)
        },
        "label_variance": pvariance(values) if len(values) >= 2 else 0.0,
        "lineage_counts": {
            str(value): count
            for value, count in sorted(
                Counter(row.base.lineage_seed for row in rows).items()
            )
        },
        "mean_label": mean(values) if values else None,
        "observations": len(rows),
        "score": float(score),
        "search_seed_counts": {
            str(value): count
            for value, count in sorted(
                Counter(row.base.search_seed for row in rows).items()
            )
        },
        "search_seed_span": len({row.base.search_seed for row in rows}),
        "support_key_checksum": _checksum(key),
        "tier": tier,
    }


def _error_mechanism(
    *,
    selected_support: Mapping[str, Any],
    oracle_support: Mapping[str, Any],
    selected_score: float,
    oracle_score: float,
) -> str:
    if selected_score == oracle_score:
        return "score_tie"
    if selected_support["tier"] != oracle_support["tier"]:
        return "cross_tier_misranking"
    tier = str(selected_support["tier"])
    heterogeneous = (
        int(selected_support["distinct_labels"]) > 1
        or int(oracle_support["distinct_labels"]) > 1
    )
    if tier == "exact_local_signature":
        return (
            "heterogeneous_exact_signature_misranking"
            if heterogeneous
            else "stable_exact_signature_reversal"
        )
    if tier == "local_composition_signature":
        return (
            "heterogeneous_composition_misranking"
            if heterogeneous
            else "stable_composition_reversal"
        )
    if tier == "action_family_backoff":
        return "action_family_backoff_misranking"
    return "global_backoff_misranking"


def _diagnostic_rows(
    groups: Sequence[Sequence[HierarchicalViabilityObservation]],
    *,
    model: HierarchicalFutureViabilityModel,
    exact: Mapping[str, Sequence[HierarchicalViabilityObservation]],
    composition: Mapping[str, Sequence[HierarchicalViabilityObservation]],
    family: Mapping[str, Sequence[HierarchicalViabilityObservation]],
) -> tuple[dict[str, Any], ...]:
    output = []
    for rows in groups:
        scores = [float(model.score(item)[0]) for item in rows]
        selected_index = _selected_index(rows, scores)
        best = max(item.base.productive_reach for item in rows)
        oracle_indices = [
            index
            for index, item in enumerate(rows)
            if item.base.productive_reach == best
        ]
        oracle_index = max(
            oracle_indices,
            key=lambda index: (scores[index], rows[index].base.action_key),
        )
        selected = rows[selected_index]
        oracle = rows[oracle_index]
        selected_support = _support_detail(
            selected,
            model=model,
            exact=exact,
            composition=composition,
            family=family,
        )
        oracle_support = _support_detail(
            oracle,
            model=model,
            exact=exact,
            composition=composition,
            family=family,
        )
        hit = selected_index in oracle_indices
        pairwise_concordant = 0
        pairwise_discordant = 0
        pairwise_tied = 0
        for left in range(len(rows)):
            for right in range(left + 1, len(rows)):
                label_delta = (
                    rows[left].base.productive_reach
                    - rows[right].base.productive_reach
                )
                if label_delta == 0:
                    continue
                score_delta = scores[left] - scores[right]
                if score_delta == 0:
                    pairwise_tied += 1
                elif (score_delta > 0) == (label_delta > 0):
                    pairwise_concordant += 1
                else:
                    pairwise_discordant += 1
        candidate_errors = [
            abs(float(score) - float(item.base.productive_reach))
            for item, score in zip(rows, scores, strict=True)
        ]
        candidate_biases = [
            float(score) - float(item.base.productive_reach)
            for item, score in zip(rows, scores, strict=True)
        ]
        output.append(
            {
                "arm": selected.base.arm,
                "candidate_count": len(rows),
                "candidate_mean_absolute_error": mean(candidate_errors),
                "candidate_mean_prediction_bias": mean(candidate_biases),
                "error_mechanism": (
                    "correct"
                    if hit
                    else _error_mechanism(
                        selected_support=selected_support,
                        oracle_support=oracle_support,
                        selected_score=scores[selected_index],
                        oracle_score=scores[oracle_index],
                    )
                ),
                "future_binding_hit": hit,
                "group_id": selected.base.group_id,
                "label_regret": best - selected.base.productive_reach,
                "lineage_seed": selected.base.lineage_seed,
                "maximum_productive_reach": best,
                "oracle_action_key": oracle.base.action_key,
                "oracle_action_name": oracle.base.action_name,
                "oracle_productive_reach": oracle.base.productive_reach,
                "oracle_support": oracle_support,
                "pairwise_concordant": pairwise_concordant,
                "pairwise_discordant": pairwise_discordant,
                "pairwise_tied": pairwise_tied,
                "score_gap_selected_over_oracle": (
                    scores[selected_index] - scores[oracle_index]
                ),
                "search_seed": selected.base.search_seed,
                "selected_action_key": selected.base.action_key,
                "selected_action_name": selected.base.action_name,
                "selected_productive_reach": selected.base.productive_reach,
                "selected_support": selected_support,
                "top_score_tie_count": sum(
                    score == max(scores) for score in scores
                ),
            }
        )
    return tuple(output)


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    hits = sum(bool(row["future_binding_hit"]) for row in rows)
    errors = [row for row in rows if not row["future_binding_hit"]]
    concordant = sum(int(row["pairwise_concordant"]) for row in rows)
    discordant = sum(int(row["pairwise_discordant"]) for row in rows)
    tied = sum(int(row["pairwise_tied"]) for row in rows)
    pair_count = concordant + discordant + tied
    tier_counts = Counter(str(row["selected_support"]["tier"]) for row in rows)
    tier_hits = Counter(
        str(row["selected_support"]["tier"])
        for row in rows
        if bool(row["future_binding_hit"])
    )
    spans = [int(row["selected_support"]["search_seed_span"]) for row in rows]
    supports = [int(row["selected_support"]["observations"]) for row in rows]
    return {
        "accuracy": hits / max(1, count),
        "eligible_groups": count,
        "error_mechanism_counts": dict(
            sorted(Counter(str(row["error_mechanism"]) for row in errors).items())
        ),
        "errors": len(errors),
        "hits": hits,
        "mean_candidate_absolute_error": (
            mean(float(row["candidate_mean_absolute_error"]) for row in rows)
            if rows
            else 0.0
        ),
        "mean_candidate_prediction_bias": (
            mean(float(row["candidate_mean_prediction_bias"]) for row in rows)
            if rows
            else 0.0
        ),
        "mean_error_label_regret": (
            mean(float(row["label_regret"]) for row in errors)
            if errors
            else 0.0
        ),
        "pairwise_concordance": concordant / max(1, pair_count),
        "pairwise_concordant": concordant,
        "pairwise_discordant": discordant,
        "pairwise_tied": tied,
        "selected_support_observations_median": median(supports) if supports else 0,
        "selected_support_seed_span_counts": {
            str(value): count for value, count in sorted(Counter(spans).items())
        },
        "selected_tier_accuracy": {
            tier: tier_hits[tier] / max(1, tier_counts[tier])
            for tier in sorted(tier_counts)
        },
        "selected_tier_counts": dict(sorted(tier_counts.items())),
        "selection_to_oracle_action_counts": dict(
            sorted(
                Counter(
                    f"{row['selected_action_name']}->{row['oracle_action_name']}"
                    for row in errors
                ).items()
            )
        ),
        "top_score_tie_groups": sum(
            int(row["top_score_tie_count"]) > 1 for row in rows
        ),
    }


def _model_selection_audit(
    groups: Sequence[Sequence[HierarchicalViabilityObservation]],
    *,
    model: HierarchicalFutureViabilityModel,
) -> tuple[int, dict[str, str]]:
    hits = 0
    selections = {}
    for rows in groups:
        scores = [float(model.score(item)[0]) for item in rows]
        selected_index = _selected_index(rows, scores)
        best = max(item.base.productive_reach for item in rows)
        hits += int(rows[selected_index].base.productive_reach == best)
        selections[rows[0].base.group_id] = rows[selected_index].base.action_key
    return hits, selections


def diagnose_future_viability_seed_shift(
    training_observations: Sequence[HierarchicalViabilityObservation],
    evaluation_observations: Sequence[HierarchicalViabilityObservation],
    *,
    future_model: HierarchicalFutureViabilityModel,
    focal_search_seed: int,
    reference_search_seeds: Sequence[int],
    training_search_seeds: Sequence[int],
    radius: int,
    minimum_signature_support: int,
) -> dict[str, Any]:
    """Attribute the focal seed miss without changing representation or gates."""

    if not training_observations or not evaluation_observations:
        raise ValueError("T12.6.1b requires non-empty frozen observations")
    exact, composition, family = _support_indices(training_observations)
    groups = _eligible_groups(evaluation_observations)
    diagnostic_rows = _diagnostic_rows(
        groups,
        model=future_model,
        exact=exact,
        composition=composition,
        family=family,
    )
    per_seed = {
        str(seed): _summary(
            [row for row in diagnostic_rows if int(row["search_seed"]) == seed]
        )
        for seed in sorted(
            {int(row["search_seed"]) for row in diagnostic_rows}
        )
    }
    focal_rows = [
        row
        for row in diagnostic_rows
        if int(row["search_seed"]) == int(focal_search_seed)
    ]
    reference_rows = [
        row
        for row in diagnostic_rows
        if int(row["search_seed"])
        in {int(value) for value in reference_search_seeds}
    ]
    focal_summary = _summary(focal_rows)
    reference_summary = _summary(reference_rows)
    focal_groups = tuple(
        rows
        for rows in groups
        if rows[0].base.search_seed == int(focal_search_seed)
    )
    frozen_hits, frozen_selections = _model_selection_audit(
        focal_groups, model=future_model
    )
    leave_one_out = {}
    for omitted in training_search_seeds:
        selected_training = tuple(
            item
            for item in training_observations
            if item.base.search_seed != int(omitted)
        )
        model = HierarchicalFutureViabilityModel.fit(
            selected_training,
            target_field="productive_reach",
            radius=radius,
            minimum_signature_support=minimum_signature_support,
        )
        hits, selections = _model_selection_audit(focal_groups, model=model)
        corrected = 0
        worsened = 0
        changed = 0
        for rows in focal_groups:
            group_id = rows[0].base.group_id
            reach = {
                item.base.action_key: item.base.productive_reach for item in rows
            }
            best = max(reach.values())
            frozen_key = frozen_selections[group_id]
            selected_key = selections[group_id]
            changed += int(frozen_key != selected_key)
            corrected += int(
                reach[frozen_key] < best and reach[selected_key] == best
            )
            worsened += int(
                reach[frozen_key] == best and reach[selected_key] < best
            )
        leave_one_out[str(omitted)] = {
            "accuracy": hits / max(1, len(focal_groups)),
            "changed_selections": changed,
            "hits": hits,
            "pooled_hits_corrected": corrected,
            "pooled_hits_worsened": worsened,
            "training_observations": len(selected_training),
        }

    focal_errors = [row for row in focal_rows if not row["future_binding_hit"]]
    error_counts = Counter(str(row["error_mechanism"]) for row in focal_errors)
    dominant = (
        None
        if not error_counts
        else sorted(error_counts, key=lambda key: (-error_counts[key], key))[0]
    )
    return {
        "classification": (
            "NO_FOCAL_TRANSFER_ERRORS"
            if dominant is None
            else f"POSTHOC_{int(focal_search_seed)}_{dominant.upper()}_DOMINANT"
        ),
        "diagnostic_axes": list(SEED_SHIFT_DIAGNOSTIC_AXES),
        "focal_search_seed": int(focal_search_seed),
        "focal_summary": focal_summary,
        "format_version": SEED_SHIFT_DIAGNOSTIC_FORMAT,
        "leave_one_training_seed_out": leave_one_out,
        "per_arm": {
            arm: _summary([row for row in focal_rows if row["arm"] == arm])
            for arm in sorted({str(row["arm"]) for row in focal_rows})
        },
        "per_lineage": {
            str(lineage): _summary(
                [
                    row
                    for row in focal_rows
                    if int(row["lineage_seed"]) == lineage
                ]
            )
            for lineage in sorted(
                {int(row["lineage_seed"]) for row in focal_rows}
            )
        },
        "per_search_seed": per_seed,
        "reference_contrast": {
            "accuracy_delta_focal_minus_reference": (
                float(focal_summary["accuracy"])
                - float(reference_summary["accuracy"])
            ),
            "pairwise_concordance_delta_focal_minus_reference": (
                float(focal_summary["pairwise_concordance"])
                - float(reference_summary["pairwise_concordance"])
            ),
            "prediction_mae_delta_focal_minus_reference": (
                float(focal_summary["mean_candidate_absolute_error"])
                - float(reference_summary["mean_candidate_absolute_error"])
            ),
            "reference_search_seeds": [
                int(value) for value in reference_search_seeds
            ],
            "reference_summary": reference_summary,
        },
        "rows": list(diagnostic_rows),
        "training_observations": len(training_observations),
        "frozen_focal_hits": frozen_hits,
    }


__all__ = [
    "SEED_SHIFT_DIAGNOSTIC_AXES",
    "SEED_SHIFT_DIAGNOSTIC_FORMAT",
    "diagnose_future_viability_seed_shift",
]
