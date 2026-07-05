from theory.sage.live_prefix_counterfactual_collector import LivePrefixAction
from theory.sage.policy_loop_guard import (
    LOOP_GUARD_SWITCH_REASON,
    apply_policy_loop_guard,
    max_consecutive_same_action_arg_repeats,
)


class FakeAction:
    def __init__(self, name, action_args=None):
        self.name = name
        self.action_args = dict(action_args or {})


def test_policy_loop_guard_switches_after_repeated_target_fallback():
    valid_actions = [
        FakeAction("ACTION3"),
        FakeAction("ACTION4"),
        FakeAction("ACTION6", {"x": 18, "y": 0}),
    ]
    prefix = [
        LivePrefixAction("ACTION6", {"x": 18, "y": 0}),
        LivePrefixAction("ACTION6", {"x": 18, "y": 0}),
    ]

    decision = apply_policy_loop_guard(
        proposed_action=valid_actions[2],
        valid_actions=valid_actions,
        prefix=prefix,
        decision_reason="candidate_policy_live_target_fallback",
        target_action="ACTION6",
        max_same_action_arg_repeats=2,
        switch_preference=("ACTION3",),
    )

    assert decision.loop_guard_triggered is True
    assert decision.repeated_same_action_args_detected is True
    assert decision.fallback_loop_interrupted is True
    assert decision.switch_action_selected_after_exhaustion is True
    assert decision.selected_action_raw is valid_actions[0]
    assert decision.selected_switch_action == "ACTION3"
    assert decision.blocked_action == "ACTION6"
    assert decision.blocked_action_args == {"x": 18, "y": 0}
    assert decision.consecutive_repeats_before == 2
    assert decision.max_same_action_arg_repeats == 2
    assert decision.loop_guard_reason == LOOP_GUARD_SWITCH_REASON


def test_policy_loop_guard_does_not_interrupt_non_fallback_decision():
    valid_actions = [
        FakeAction("ACTION3"),
        FakeAction("ACTION6", {"x": 18, "y": 0}),
    ]
    prefix = [
        LivePrefixAction("ACTION6", {"x": 18, "y": 0}),
        LivePrefixAction("ACTION6", {"x": 18, "y": 0}),
    ]

    decision = apply_policy_loop_guard(
        proposed_action=valid_actions[1],
        valid_actions=valid_actions,
        prefix=prefix,
        decision_reason="candidate_policy_live_success_like_target",
        target_action="ACTION6",
        max_same_action_arg_repeats=2,
        switch_preference=("ACTION3",),
    )

    assert decision.loop_guard_triggered is False
    assert decision.repeated_same_action_args_detected is True
    assert decision.fallback_loop_interrupted is False
    assert decision.selected_action_raw is valid_actions[1]


def test_max_consecutive_same_action_arg_repeats_counts_runs_not_totals():
    rows = [
        {"selected_action": "ACTION6", "selected_action_args": {"x": 18, "y": 0}},
        {"selected_action": "ACTION6", "selected_action_args": {"x": 18, "y": 0}},
        {"selected_action": "ACTION3", "selected_action_args": {}},
        {"selected_action": "ACTION6", "selected_action_args": {"x": 18, "y": 0}},
        {"selected_action": "ACTION6", "selected_action_args": {"x": 18, "y": 0}},
    ]

    assert max_consecutive_same_action_arg_repeats(rows) == 2
