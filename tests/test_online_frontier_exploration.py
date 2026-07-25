"""SAGE.9v frontier-oriented exploration tests."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from theory.online_frontier_exploration import OnlineFrontierExplorer


@dataclass(frozen=True)
class _Action:
    name: str
    action_args: dict = field(default_factory=dict)


def _grid(value: int = 2) -> np.ndarray:
    grid = np.zeros((7, 9), dtype=np.int32)
    grid[2:4, 2:4] = value
    grid[5, 7] = value + 1
    return grid


def _stalled(*, actions: int = 8) -> dict:
    return {
        "branch_actions": actions,
        "actions_since_terminal_improvement": actions,
        "max_hash_repeat": 3,
        "unique_states_in_window": 2,
    }


def _candidates() -> tuple[_Action, ...]:
    return (
        _Action("ACTION1"),
        _Action("ACTION6", {"x": 2, "y": 2}),
        _Action("ACTION6", {"x": 7, "y": 5}),
    )


def test_frontier_exploration_waits_for_observed_stagnation():
    explorer = OnlineFrontierExplorer(minimum_stagnant_steps=6)

    decision = explorer.select(
        current_grid=_grid(),
        available_actions=("ACTION1", "ACTION6"),
        available_action_candidates=_candidates(),
        branch_diagnostics=_stalled(actions=5),
    )

    assert decision is None
    assert explorer.summary()["experiments"] == 0


def test_frontier_exploration_selects_and_credits_untested_actuator():
    explorer = OnlineFrontierExplorer(minimum_stagnant_steps=2)
    before = _grid()

    decision = explorer.select(
        current_grid=before,
        available_actions=("ACTION1", "ACTION6"),
        available_action_candidates=_candidates(),
        branch_diagnostics=_stalled(),
    )

    assert decision is not None
    assert decision.state_action_untested is True
    assert decision.actuator_untested is True
    assert decision.sequence_step == 1

    after = before.copy()
    after[decision.action_data.get("y", 0), decision.action_data.get("x", 0)] = 8
    outcome = explorer.observe_transition(
        grid_before=before,
        grid_after=after,
        action_name=decision.action_name,
        action_data=decision.action_data,
        no_effect=False,
        game_over=False,
        terminal_success=False,
    )

    assert outcome["productive"] is True
    assert outcome["novel_effect"] is True
    assert outcome["novel_state"] is True
    summary = explorer.summary()
    assert summary["productive_experiments"] == 1
    assert summary["information_gain"] == 2.0
    assert summary["multi_step_sequences"] == 1


def test_productive_frontier_experiment_opens_bounded_continuation():
    explorer = OnlineFrontierExplorer(
        minimum_stagnant_steps=2,
        max_sequence_actions=2,
    )
    before = _grid()
    first = explorer.select(
        current_grid=before,
        available_actions=("ACTION1", "ACTION6"),
        available_action_candidates=_candidates(),
        branch_diagnostics=_stalled(),
    )
    assert first is not None
    after = before.copy()
    after[0, 0] = 4
    explorer.observe_transition(
        grid_before=before,
        grid_after=after,
        action_name=first.action_name,
        action_data=first.action_data,
        no_effect=False,
        game_over=False,
        terminal_success=False,
    )

    continuation = explorer.select(
        current_grid=after,
        available_actions=("ACTION1", "ACTION6"),
        available_action_candidates=_candidates(),
        branch_diagnostics={
            "branch_actions": 1,
            "actions_since_terminal_improvement": 0,
            "max_hash_repeat": 1,
            "unique_states_in_window": 2,
        },
    )

    assert continuation is not None
    assert continuation.sequence_id == first.sequence_id
    assert continuation.sequence_step == 2
    explorer.observe_transition(
        grid_before=after,
        grid_after=after,
        action_name=continuation.action_name,
        action_data=continuation.action_data,
        no_effect=True,
        game_over=False,
        terminal_success=False,
    )
    assert explorer.summary()["active_sequence_id"] == ""


def test_unsafe_actuator_is_deprioritized_after_observation():
    explorer = OnlineFrontierExplorer(
        minimum_stagnant_steps=1,
        max_experiments_per_state=10,
    )
    grid = _grid()
    first = explorer.select(
        current_grid=grid,
        available_actions=("ACTION1",),
        available_action_candidates=(_Action("ACTION1"),),
        branch_diagnostics=_stalled(),
    )
    assert first is not None
    explorer.observe_transition(
        grid_before=grid,
        grid_after=grid,
        action_name=first.action_name,
        action_data=first.action_data,
        no_effect=True,
        game_over=True,
        terminal_success=False,
    )
    explorer.start_branch()

    second = explorer.select(
        current_grid=grid,
        available_actions=("ACTION1", "ACTION2"),
        available_action_candidates=(
            _Action("ACTION1"),
            _Action("ACTION2"),
        ),
        branch_diagnostics=_stalled(),
    )

    assert second is not None
    assert second.action_name == "ACTION2"


def test_object_roles_are_palette_invariant():
    first = OnlineFrontierExplorer(minimum_stagnant_steps=1)
    second = OnlineFrontierExplorer(minimum_stagnant_steps=1)
    action = (_Action("ACTION6", {"x": 2, "y": 2}),)

    first_decision = first.select(
        current_grid=_grid(2),
        available_actions=("ACTION6",),
        available_action_candidates=action,
        branch_diagnostics=_stalled(),
    )
    second_decision = second.select(
        current_grid=_grid(8),
        available_actions=("ACTION6",),
        available_action_candidates=action,
        branch_diagnostics=_stalled(),
    )

    assert first_decision is not None
    assert second_decision is not None
    assert (
        first_decision.target_role_signature
        == second_decision.target_role_signature
    )
    assert (
        first_decision.actuator_signature
        == second_decision.actuator_signature
    )


def test_known_deterministic_actuator_is_not_retested_forever():
    explorer = OnlineFrontierExplorer(
        minimum_stagnant_steps=1,
        max_trials_per_actuator=2,
    )
    grid = _grid()
    candidate = (_Action("ACTION1"),)
    for _ in range(2):
        selected = explorer.select(
            current_grid=grid,
            available_actions=("ACTION1",),
            available_action_candidates=candidate,
            branch_diagnostics=_stalled(),
        )
        assert selected is not None
        explorer.observe_transition(
            grid_before=grid,
            grid_after=grid,
            action_name=selected.action_name,
            action_data=selected.action_data,
            no_effect=True,
            game_over=False,
            terminal_success=False,
        )

    assert explorer.select(
        current_grid=grid,
        available_actions=("ACTION1",),
        available_action_candidates=candidate,
        branch_diagnostics=_stalled(),
    ) is None
    assert explorer.summary()["experiments"] == 2


def test_frontier_exploration_waits_for_failed_branches_and_yields_to_progress():
    explorer = OnlineFrontierExplorer(
        minimum_stagnant_steps=1,
        minimum_failed_branches=2,
    )
    grid = _grid()
    kwargs = {
        "current_grid": grid,
        "available_actions": ("ACTION1",),
        "available_action_candidates": (_Action("ACTION1"),),
        "branch_diagnostics": _stalled(),
    }
    explorer.start_branch()
    assert explorer.select(**kwargs) is None
    explorer.start_branch()
    assert explorer.select(**kwargs) is None
    explorer.start_branch()

    selected = explorer.select(**kwargs)

    assert selected is not None
    explorer.observe_transition(
        grid_before=grid,
        grid_after=grid,
        action_name=selected.action_name,
        action_data=selected.action_data,
        no_effect=True,
        game_over=False,
        terminal_success=False,
    )
    explorer.note_transition(terminal_success=True)
    assert explorer.select(**kwargs) is None
    summary = explorer.summary()
    assert summary["failed_branches"] == 2
    assert summary["terminal_progress_observed"] is True


def test_productive_frontier_effect_receives_delayed_terminal_credit():
    explorer = OnlineFrontierExplorer(
        minimum_stagnant_steps=1,
        delayed_terminal_credit_window=4,
    )
    before = _grid()
    selected = explorer.select(
        current_grid=before,
        available_actions=("ACTION6",),
        available_action_candidates=(_Action(
            "ACTION6",
            {"x": 2, "y": 2},
        ),),
        branch_diagnostics=_stalled(),
    )
    assert selected is not None
    after = before.copy()
    after[2:4, 2:4] = 0
    outcome = explorer.observe_transition(
        grid_before=before,
        grid_after=after,
        action_name=selected.action_name,
        action_data=selected.action_data,
        no_effect=False,
        game_over=False,
        terminal_success=False,
    )
    eligibility_id = outcome["delayed_credit_eligibility_id"]
    assert eligibility_id

    explorer.note_transition(terminal_success=False)
    explorer.note_transition(terminal_success=False)
    update = explorer.note_transition(terminal_success=True)

    assert len(update.credited) == 1
    assert update.credited[0].eligibility_id == eligibility_id
    assert update.credited[0].delay_actions == 2
    summary = explorer.summary()
    assert summary["delayed_eligibilities_registered"] == 1
    assert summary["delayed_terminal_events"] == 1
    assert summary["delayed_terminal_credits"] == 1
    assert summary["delayed_credit_max_delay"] == 2
    assert summary["delayed_eligibilities_pending"] == 0


def test_delayed_frontier_credit_expires_and_never_crosses_reset():
    explorer = OnlineFrontierExplorer(
        minimum_stagnant_steps=1,
        delayed_terminal_credit_window=1,
    )
    before = _grid()
    selected = explorer.select(
        current_grid=before,
        available_actions=("ACTION6",),
        available_action_candidates=(_Action(
            "ACTION6",
            {"x": 2, "y": 2},
        ),),
        branch_diagnostics=_stalled(),
    )
    assert selected is not None
    after = before.copy()
    after[2:4, 2:4] = 0
    outcome = explorer.observe_transition(
        grid_before=before,
        grid_after=after,
        action_name=selected.action_name,
        action_data=selected.action_data,
        no_effect=False,
        game_over=False,
        terminal_success=False,
    )
    eligibility_id = outcome["delayed_credit_eligibility_id"]
    explorer.note_transition(terminal_success=False)
    explorer.note_transition(terminal_success=False)
    expired = explorer.note_transition(terminal_success=False)

    assert expired.expired_eligibility_ids == (eligibility_id,)
    assert explorer.note_transition(
        terminal_success=True
    ).credited == ()

    explorer = OnlineFrontierExplorer(minimum_stagnant_steps=1)
    selected = explorer.select(
        current_grid=before,
        available_actions=("ACTION6",),
        available_action_candidates=(_Action(
            "ACTION6",
            {"x": 2, "y": 2},
        ),),
        branch_diagnostics=_stalled(),
    )
    assert selected is not None
    outcome = explorer.observe_transition(
        grid_before=before,
        grid_after=after,
        action_name=selected.action_name,
        action_data=selected.action_data,
        no_effect=False,
        game_over=False,
        terminal_success=False,
    )
    discarded = explorer.start_branch()

    assert discarded == (outcome["delayed_credit_eligibility_id"],)
    assert explorer.note_transition(
        terminal_success=True
    ).credited == ()
    summary = explorer.summary()
    assert summary["censored_delayed_eligibilities"] == 1


def test_delayed_frontier_credit_is_disabled_by_ablation():
    explorer = OnlineFrontierExplorer(
        minimum_stagnant_steps=1,
        enable_delayed_terminal_credit=False,
    )
    before = _grid()
    selected = explorer.select(
        current_grid=before,
        available_actions=("ACTION6",),
        available_action_candidates=(_Action(
            "ACTION6",
            {"x": 2, "y": 2},
        ),),
        branch_diagnostics=_stalled(),
    )
    assert selected is not None
    after = before.copy()
    after[2:4, 2:4] = 0
    outcome = explorer.observe_transition(
        grid_before=before,
        grid_after=after,
        action_name=selected.action_name,
        action_data=selected.action_data,
        no_effect=False,
        game_over=False,
        terminal_success=False,
    )
    explorer.note_transition(terminal_success=False)
    update = explorer.note_transition(terminal_success=True)

    assert outcome["delayed_credit_eligibility_id"] == ""
    assert update.credited == ()
    assert explorer.summary()["delayed_terminal_credits"] == 0


def test_delayed_credit_selects_at_most_one_action_per_sequence():
    explorer = OnlineFrontierExplorer(
        minimum_stagnant_steps=1,
        max_sequence_actions=2,
        delayed_terminal_credit_window=4,
    )
    before = _grid()
    first = explorer.select(
        current_grid=before,
        available_actions=("ACTION6",),
        available_action_candidates=(_Action(
            "ACTION6",
            {"x": 2, "y": 2},
        ),),
        branch_diagnostics=_stalled(),
    )
    assert first is not None
    middle = before.copy()
    middle[2, 2] = 8
    first_outcome = explorer.observe_transition(
        grid_before=before,
        grid_after=middle,
        action_name=first.action_name,
        action_data=first.action_data,
        no_effect=False,
        game_over=False,
        terminal_success=False,
    )
    explorer.note_transition(terminal_success=False)
    second = explorer.select(
        current_grid=middle,
        available_actions=("ACTION6",),
        available_action_candidates=(_Action(
            "ACTION6",
            {"x": 2, "y": 2},
        ),),
        branch_diagnostics=_stalled(actions=1),
    )
    assert second is not None
    assert second.sequence_id == first.sequence_id
    after = middle.copy()
    after[2, 3] = 7
    second_outcome = explorer.observe_transition(
        grid_before=middle,
        grid_after=after,
        action_name=second.action_name,
        action_data=second.action_data,
        no_effect=False,
        game_over=False,
        terminal_success=False,
    )
    explorer.note_transition(terminal_success=False)

    update = explorer.note_transition(terminal_success=True)

    assert len(update.credited) == 1
    assert len(update.discarded_eligibility_ids) == 1
    all_ids = {
        first_outcome["delayed_credit_eligibility_id"],
        second_outcome["delayed_credit_eligibility_id"],
    }
    assert {
        update.credited[0].eligibility_id,
        update.discarded_eligibility_ids[0],
    } == all_ids


def test_disabled_frontier_explorer_is_inert():
    explorer = OnlineFrontierExplorer(
        enabled=False,
        minimum_stagnant_steps=1,
    )

    assert explorer.select(
        current_grid=_grid(),
        available_actions=("ACTION1",),
        available_action_candidates=(_Action("ACTION1"),),
        branch_diagnostics=_stalled(),
    ) is None
    assert explorer.summary()["experiments"] == 0
