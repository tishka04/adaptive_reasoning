from theory.m2.metric_registry import (
    METRIC_PROFILES,
    available_metrics,
    default_falsification_for_metric,
    is_metric_executor_supported,
    is_metric_measurable,
    is_unlock_only_metric,
    metric_executor_block_reason,
)


def test_metric_registry_contains_initial_allowed_metrics():
    metrics = set(available_metrics())

    assert "local_patch_before_after" in metrics
    assert "contact_graph_before_after" in metrics
    assert "changed_pixels" in metrics
    assert is_metric_measurable("local_patch_before_after")
    assert not is_metric_measurable("unknown_metric")


def test_default_falsification_is_hard_failure_condition():
    criterion = default_falsification_for_metric("local_patch_before_after")

    assert criterion.support_condition == "target_action_signal > best_control_signal"
    assert criterion.failure_condition == "target_action_signal <= best_control_signal"
    assert criterion.minimum_effect_size == 1


def test_m2_13a_typed_metrics_are_registered():
    metrics = set(available_metrics())

    for metric in (
        "available_actions_before_after",
        "objective_completion_signal",
        "terminal_reentry_rate",
    ):
        assert metric in metrics
        assert is_metric_measurable(metric)
        assert metric in METRIC_PROFILES


def test_executor_support_tiers_match_m3_g5_g6():
    # objective_completion_signal and terminal_reentry_rate are measured by M3.G5/G6.
    assert is_metric_executor_supported("objective_completion_signal")
    assert is_metric_executor_supported("terminal_reentry_rate")
    # available_actions_before_after needs an env frame metadata extractor.
    assert not is_metric_executor_supported("available_actions_before_after")
    assert (
        metric_executor_block_reason("available_actions_before_after")
        == "metric_known_but_unmeasurable_in_current_executor"
    )
    assert metric_executor_block_reason("objective_completion_signal") is None
    assert metric_executor_block_reason("unknown_metric") is None


def test_unlock_only_metric_classification():
    assert is_unlock_only_metric("available_actions_before_after")
    assert not is_unlock_only_metric("objective_completion_signal")


def test_profiled_metric_default_falsification_uses_profile():
    criterion = default_falsification_for_metric("objective_completion_signal")

    assert criterion.metric == "objective_completion_signal"
    assert criterion.failure_condition == "no_completed_level_delta_vs_matched_controls"
    assert criterion.support_condition
