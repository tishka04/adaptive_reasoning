"""Fail-closed post-hoc diagnostics for the sealed SAGE.T12.6 compile miss."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from statistics import mean, pvariance
from typing import Any

from .future_viability import FutureViabilityModel, FutureViabilityObservation

FUTURE_VIABILITY_DIAGNOSTIC_FORMAT = (
    "sage-t12.6a-future-viability-diagnostic-v1"
)


def _checksum(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _selected_index(
    rows: Sequence[FutureViabilityObservation], scores: Sequence[float]
) -> int:
    return max(
        range(len(rows)),
        key=lambda index: (float(scores[index]), rows[index].action_key),
    )


def _eligible_groups(
    observations: Sequence[FutureViabilityObservation],
) -> tuple[tuple[FutureViabilityObservation, ...], ...]:
    grouped: dict[str, list[FutureViabilityObservation]] = defaultdict(list)
    for item in observations:
        grouped[item.group_id].append(item)
    return tuple(
        tuple(sorted(rows, key=lambda item: item.action_key))
        for _, rows in sorted(grouped.items())
        if len(rows) >= 2 and len({item.productive_reach for item in rows}) >= 2
    )


def _support_diagnostic(
    item: FutureViabilityObservation,
    *,
    model: FutureViabilityModel,
    training: Sequence[FutureViabilityObservation],
    focal_lineage_seed: int,
) -> dict[str, Any]:
    score, tier = model.score(item)
    if tier == "target_local_signature":
        support = [row for row in training if row.local_signature == item.local_signature]
        support_key = _checksum(item.local_signature)
    elif tier == "action_family_backoff":
        support = [row for row in training if row.backoff_key == item.backoff_key]
        support_key = item.backoff_key
    else:
        support = list(training)
        support_key = "global"
    values = [int(row.productive_reach) for row in support]
    counts = Counter(values)
    return {
        "arm_counts": dict(sorted(Counter(row.arm for row in support).items())),
        "distinct_productive_reach_values": len(counts),
        "label_histogram": {str(key): counts[key] for key in sorted(counts)},
        "label_variance": pvariance(values) if len(values) >= 2 else 0.0,
        "lineage_counts": {
            str(key): value
            for key, value in sorted(
                Counter(row.lineage_seed for row in support).items()
            )
        },
        "other_lineage_observations": sum(
            row.lineage_seed != int(focal_lineage_seed) for row in support
        ),
        "same_lineage_observations": sum(
            row.lineage_seed == int(focal_lineage_seed) for row in support
        ),
        "score": float(score),
        "search_seeds": sorted({int(row.search_seed) for row in support}),
        "support_key": support_key,
        "support_observations": len(support),
        "support_tier": tier,
        "training_maximum": max(values) if values else None,
        "training_mean": mean(values) if values else None,
        "training_minimum": min(values) if values else None,
    }


def _model_accuracy(
    groups: Sequence[Sequence[FutureViabilityObservation]],
    *,
    scorer: Callable[[FutureViabilityObservation], float],
) -> dict[str, Any]:
    hits = 0
    selections: dict[str, str] = {}
    for rows in groups:
        scores = [float(scorer(item)) for item in rows]
        selected = _selected_index(rows, scores)
        best = max(item.productive_reach for item in rows)
        hits += int(rows[selected].productive_reach == best)
        selections[rows[0].group_id] = rows[selected].action_key
    return {
        "accuracy": hits / max(1, len(groups)),
        "eligible_groups": len(groups),
        "hits": hits,
        "selections": selections,
    }


def diagnose_future_viability_fold(
    observations: Sequence[FutureViabilityObservation],
    *,
    holdout_search_seed: int,
    focal_lineage_seed: int,
    reference_lineage_seed: int,
    radius: int,
    minimum_signature_support: int,
    binding_shift: int,
) -> dict[str, Any]:
    """Explain one registered T12.6 cross-fit miss without changing its gate."""

    training = tuple(
        item for item in observations if item.search_seed != int(holdout_search_seed)
    )
    focal = tuple(
        item
        for item in observations
        if item.search_seed == int(holdout_search_seed)
        and item.lineage_seed == int(focal_lineage_seed)
    )
    if not training or not focal:
        raise ValueError("T12.6a requires non-empty training and focal observations")
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
    groups = _eligible_groups(focal)
    diagnostic_rows: list[dict[str, Any]] = []
    cause_counts: Counter[str] = Counter()
    arm_errors: Counter[str] = Counter()
    selected_actions: Counter[str] = Counter()
    oracle_actions: Counter[str] = Counter()

    for rows in groups:
        future = [future_model.score(item) for item in rows]
        immediate = [immediate_model.score(item) for item in rows]
        future_scores = [float(value[0]) for value in future]
        immediate_scores = [float(value[0]) for value in immediate]
        shift = int(binding_shift) % len(rows)
        swapped_scores = future_scores[shift:] + future_scores[:shift]
        future_index = _selected_index(rows, future_scores)
        immediate_index = _selected_index(rows, immediate_scores)
        swapped_index = _selected_index(rows, swapped_scores)
        maximum_reach = max(item.productive_reach for item in rows)
        oracle_indices = [
            index
            for index, item in enumerate(rows)
            if item.productive_reach == maximum_reach
        ]
        oracle_index = max(
            oracle_indices,
            key=lambda index: (future_scores[index], rows[index].action_key),
        )
        top_score = max(future_scores)
        top_indices = [
            index for index, score in enumerate(future_scores) if score == top_score
        ]
        hit = future_index in oracle_indices
        oracle_in_top_score = any(index in top_indices for index in oracle_indices)
        selected_support = _support_diagnostic(
            rows[future_index],
            model=future_model,
            training=training,
            focal_lineage_seed=focal_lineage_seed,
        )
        oracle_support = _support_diagnostic(
            rows[oracle_index],
            model=future_model,
            training=training,
            focal_lineage_seed=focal_lineage_seed,
        )
        if hit:
            cause = "correct"
        elif oracle_in_top_score:
            same_signature = any(
                rows[index].local_signature == rows[future_index].local_signature
                for index in oracle_indices
                if index in top_indices
            )
            cause = (
                "within_signature_score_tie"
                if same_signature
                else "cross_signature_score_tie"
            )
        elif (
            selected_support["support_tier"] != "target_local_signature"
            or oracle_support["support_tier"] != "target_local_signature"
        ):
            cause = "backoff_misranking"
        elif (
            int(selected_support["distinct_productive_reach_values"]) > 1
            or int(oracle_support["distinct_productive_reach_values"]) > 1
        ):
            cause = "heterogeneous_exact_signature_misranking"
        else:
            cause = "stable_exact_signature_misranking"

        cause_counts[cause] += 1
        selected_actions[rows[future_index].action_name] += 1
        for index in oracle_indices:
            oracle_actions[rows[index].action_name] += 1
        if not hit:
            arm_errors[rows[0].arm] += 1
        diagnostic_rows.append(
            {
                "arm": rows[0].arm,
                "binding_swap_hit": (
                    rows[swapped_index].productive_reach == maximum_reach
                ),
                "candidate_count": len(rows),
                "error_mechanism": cause,
                "future_binding_hit": hit,
                "group_id": rows[0].group_id,
                "immediate_binding_hit": (
                    rows[immediate_index].productive_reach == maximum_reach
                ),
                "label_regret": (
                    maximum_reach - rows[future_index].productive_reach
                ),
                "lineage_seed": rows[0].lineage_seed,
                "maximum_productive_reach": maximum_reach,
                "oracle_action_keys": [rows[index].action_key for index in oracle_indices],
                "oracle_action_names": [
                    rows[index].action_name for index in oracle_indices
                ],
                "oracle_in_top_score": oracle_in_top_score,
                "oracle_support": oracle_support,
                "predicted_score_gap_over_best_oracle": (
                    future_scores[future_index]
                    - max(future_scores[index] for index in oracle_indices)
                ),
                "score_tie_count": len(top_indices),
                "search_seed": rows[0].search_seed,
                "selected_action_key": rows[future_index].action_key,
                "selected_action_name": rows[future_index].action_name,
                "selected_productive_reach": rows[future_index].productive_reach,
                "selected_support": selected_support,
                "top_score_oracle_upper_bound_hit": hit or oracle_in_top_score,
            }
        )

    pooled = _model_accuracy(
        groups,
        scorer=lambda item: future_model.score(item)[0],
    )
    same_lineage_training = tuple(
        item for item in training if item.lineage_seed == int(focal_lineage_seed)
    )
    other_lineage_training = tuple(
        item for item in training if item.lineage_seed == int(reference_lineage_seed)
    )
    same_lineage_model = FutureViabilityModel.fit(
        same_lineage_training,
        target_field="productive_reach",
        radius=radius,
        minimum_signature_support=minimum_signature_support,
    )
    other_lineage_model = FutureViabilityModel.fit(
        other_lineage_training,
        target_field="productive_reach",
        radius=radius,
        minimum_signature_support=minimum_signature_support,
    )
    same_lineage = _model_accuracy(
        groups,
        scorer=lambda item: same_lineage_model.score(item)[0],
    )
    other_lineage = _model_accuracy(
        groups,
        scorer=lambda item: other_lineage_model.score(item)[0],
    )
    arm_models = {
        arm: FutureViabilityModel.fit(
            tuple(item for item in training if item.arm == arm),
            target_field="productive_reach",
            radius=radius,
            minimum_signature_support=minimum_signature_support,
        )
        for arm in sorted({item.arm for item in focal})
    }
    arm_conditioned = _model_accuracy(
        groups,
        scorer=lambda item: arm_models[item.arm].score(item)[0],
    )

    def comparison(candidate: Mapping[str, Any]) -> dict[str, int]:
        pooled_selections = pooled["selections"]
        candidate_selections = candidate["selections"]
        corrected = 0
        worsened = 0
        changed = 0
        for rows in groups:
            group_id = rows[0].group_id
            best = max(item.productive_reach for item in rows)
            reach_by_action = {item.action_key: item.productive_reach for item in rows}
            pooled_key = pooled_selections[group_id]
            candidate_key = candidate_selections[group_id]
            changed += int(pooled_key != candidate_key)
            corrected += int(
                reach_by_action[pooled_key] < best
                and reach_by_action[candidate_key] == best
            )
            worsened += int(
                reach_by_action[pooled_key] == best
                and reach_by_action[candidate_key] < best
            )
        return {
            "changed_selections": changed,
            "pooled_misses_corrected": corrected,
            "pooled_hits_worsened": worsened,
        }

    errors = [row for row in diagnostic_rows if not row["future_binding_hit"]]
    error_causes = Counter(row["error_mechanism"] for row in errors)
    dominant = (
        None
        if not error_causes
        else sorted(error_causes, key=lambda key: (-error_causes[key], key))[0]
    )
    return {
        "classification": (
            "NO_FOCAL_ERRORS"
            if dominant is None
            else f"POSTHOC_{dominant.upper()}_DOMINANT"
        ),
        "counterfactual_sensitivities": {
            "arm_conditioned": {
                **{key: value for key, value in arm_conditioned.items() if key != "selections"},
                **comparison(arm_conditioned),
            },
            "other_lineage_only": {
                **{key: value for key, value in other_lineage.items() if key != "selections"},
                **comparison(other_lineage),
            },
            "same_lineage_only": {
                **{key: value for key, value in same_lineage.items() if key != "selections"},
                **comparison(same_lineage),
            },
            "top_score_oracle_upper_bound": {
                "accuracy": sum(
                    bool(row["top_score_oracle_upper_bound_hit"])
                    for row in diagnostic_rows
                )
                / max(1, len(diagnostic_rows)),
                "hits": sum(
                    bool(row["top_score_oracle_upper_bound_hit"])
                    for row in diagnostic_rows
                ),
            },
        },
        "diagnostic_axes": [
            "support_tier",
            "score_ties",
            "local_signature_label_heterogeneity",
            "lineage_conditioning",
            "archive_arm_conditioning",
        ],
        "error_summary": {
            "arm_error_counts": dict(sorted(arm_errors.items())),
            "dominant_error_mechanism": dominant,
            "error_mechanism_counts": dict(sorted(error_causes.items())),
            "errors": len(errors),
            "mean_error_label_regret": (
                mean(float(row["label_regret"]) for row in errors)
                if errors
                else 0.0
            ),
            "oracle_action_name_counts": dict(sorted(oracle_actions.items())),
            "selected_action_name_counts": dict(sorted(selected_actions.items())),
        },
        "focal_metrics": {
            "accuracy": pooled["accuracy"],
            "eligible_groups": pooled["eligible_groups"],
            "hits": pooled["hits"],
            "target_local_signature_selected": sum(
                row["selected_support"]["support_tier"]
                == "target_local_signature"
                for row in diagnostic_rows
            ),
        },
        "format_version": FUTURE_VIABILITY_DIAGNOSTIC_FORMAT,
        "focal_lineage_seed": int(focal_lineage_seed),
        "holdout_search_seed": int(holdout_search_seed),
        "reference_lineage_seed": int(reference_lineage_seed),
        "rows": diagnostic_rows,
        "training_observations": len(training),
    }


__all__ = [
    "FUTURE_VIABILITY_DIAGNOSTIC_FORMAT",
    "diagnose_future_viability_fold",
]
