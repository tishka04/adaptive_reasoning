"""Metric registry for M2 testability checks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from .schema import FalsificationCriterion


ALLOWED_METRICS: Tuple[str, ...] = (
    "local_patch_before_after",
    "object_counts_before_after",
    "object_positions_before_after",
    "contact_graph_before_after",
    "object_shape_zone_before_after",
    "topology_before_after",
    "changed_pixels",
    "final_game_state",
    "levels_completed_after_rollout",
    "terminal_state_after_rollout",
    "objective_progress_proxy",
    # M2.13a typed extension (already used in the P2.G5 -> M3.G5/G6 branch).
    "available_actions_before_after",
    "objective_completion_signal",
    "terminal_reentry_rate",
)


# Typed metric profiles aligned with M3.G5/G6. The profile records the metric
# family, the default falsification semantics, the success direction and the
# executor support tier. A metric can be a known vocabulary item even if the
# current controlled executor cannot measure it yet.
METRIC_PROFILES: Dict[str, Dict[str, Any]] = {
    "available_actions_before_after": {
        "family": "affordance_or_unlock",
        "default_falsification": "available_actions_set_unchanged_after_candidate_context",
        "success_direction": "changed_or_target_action_unlocked",
        "executor_support": "requires_env_frame_metadata",
        "diagnostic_only_if_unmeasurable": True,
    },
    "objective_completion_signal": {
        "family": "objective_completion",
        "default_falsification": "no_completed_level_delta_vs_matched_controls",
        "success_direction": "increase",
        "executor_support": "m3_g5_or_later",
        "proxy_metric": False,
    },
    "terminal_reentry_rate": {
        "family": "safety",
        "default_falsification": "terminal_reentry_rate_exceeds_control",
        "success_direction": "decrease_or_bounded",
        "executor_support": "m3_g5_or_later",
        "proxy_metric": False,
    },
}

# Executor support tiers that the current controlled executor can measure.
SUPPORTED_EXECUTOR_TIERS: frozenset[str] = frozenset({"current", "m3_g5_or_later"})

# Metrics whose use is only legitimate inside the unlock/availability family.
UNLOCK_ONLY_METRICS: frozenset[str] = frozenset({"available_actions_before_after"})
UNLOCK_HYPOTHESIS_FAMILY = "action_availability_or_unlocking_precondition"

METRIC_KNOWN_BUT_UNMEASURABLE_REASON = (
    "metric_known_but_unmeasurable_in_current_executor"
)


def available_metrics() -> Tuple[str, ...]:
    return ALLOWED_METRICS


def is_metric_measurable(metric: str) -> bool:
    return str(metric) in set(ALLOWED_METRICS)


def metric_profile(metric: str) -> Dict[str, Any] | None:
    return METRIC_PROFILES.get(str(metric))


def is_metric_executor_supported(metric: str) -> bool:
    """True when the current controlled executor can actually measure the metric.

    Metrics without an explicit profile are assumed supported (the existing
    extractor-backed metrics). Profiled metrics are supported only when their
    ``executor_support`` tier is runnable now.
    """
    profile = METRIC_PROFILES.get(str(metric))
    if profile is None:
        return is_metric_measurable(metric)
    return str(profile.get("executor_support", "")) in SUPPORTED_EXECUTOR_TIERS


def metric_executor_block_reason(metric: str) -> str | None:
    """Reason a known metric is not yet measurable, else ``None``."""
    if not is_metric_measurable(metric):
        return None
    if is_metric_executor_supported(metric):
        return None
    return METRIC_KNOWN_BUT_UNMEASURABLE_REASON


def is_unlock_only_metric(metric: str) -> bool:
    return str(metric) in UNLOCK_ONLY_METRICS


def _support_condition_for_profile(profile: Mapping[str, Any]) -> str:
    direction = str(profile.get("success_direction", ""))
    if direction == "increase":
        return "target_action_signal > best_control_signal"
    if direction == "decrease_or_bounded":
        return "target_action_signal < best_control_signal"
    if direction == "changed_or_target_action_unlocked":
        return (
            "available_actions_set_changes_or_target_action_unlocked_"
            "after_candidate_context"
        )
    return "target_action_signal > best_control_signal"


def default_falsification_for_metric(metric: str) -> FalsificationCriterion:
    metric_name = str(metric)
    profile = METRIC_PROFILES.get(metric_name)
    if profile is not None:
        return FalsificationCriterion(
            metric=metric_name,
            support_condition=_support_condition_for_profile(profile),
            failure_condition=str(profile.get("default_falsification", "")),
            minimum_effect_size=1,
        )
    return FalsificationCriterion(
        metric=metric_name,
        support_condition="target_action_signal > best_control_signal",
        failure_condition="target_action_signal <= best_control_signal",
        minimum_effect_size=1,
    )


def metric_registry_payload() -> Dict[str, Any]:
    return {
        "metrics": list(ALLOWED_METRICS),
        "profiles": {metric: dict(profile) for metric, profile in METRIC_PROFILES.items()},
        "executor_supported": {
            metric: is_metric_executor_supported(metric) for metric in ALLOWED_METRICS
        },
        "default_falsification": {
            metric: default_falsification_for_metric(metric).to_dict()
            for metric in ALLOWED_METRICS
        },
    }
