from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from theory.online_terminal_relational_stencil import (
    OnlineTerminalRelationalStencilLearner,
)


BACKGROUND = 5
SPACING = 8
TILE = 6
STENCIL = (18, 18)
MARKERS = (
    (0, 2, 2),
    (0, 8, 0),
    (0, 2, 2),
)


def _actions():
    return tuple(
        SimpleNamespace(
            name="ACTION6",
            action_args={"x": x, "y": y},
        )
        for y in (10, 18, 26)
        for x in (10, 18, 26)
        if (x, y) != STENCIL
    )


def _grid(center, colors):
    grid = np.full((40, 40), BACKGROUND, dtype=np.int64)
    for (x, y), color in colors.items():
        grid[y : y + TILE, x : x + TILE] = 1
        grid[y + 3, x + 3] = color
    sx, sy = STENCIL
    for row, values in enumerate(MARKERS):
        for column, value in enumerate(values):
            grid[sy + 2 * row : sy + 2 * row + 2,
                 sx + 2 * column : sx + 2 * column + 2] = value
    grid[sy + 3, sx + 3] = center
    return grid


def _terminal_predecessor():
    colors = {}
    for y in (10, 18, 26):
        for x in (10, 18, 26):
            if (x, y) == STENCIL:
                continue
            marker = MARKERS[(y - 10) // SPACING][
                (x - 10) // SPACING
            ]
            colors[(x, y)] = 8 if marker == 0 else 9
    colors[(18, 26)] = 8
    return _grid(8, colors)


def _trained_learner(*, permuted: bool = False):
    learner = OnlineTerminalRelationalStencilLearner(
        permute_confirmed_relation=permuted,
    )
    predecessor = _terminal_predecessor()
    after_click = predecessor.copy()
    after_click[29, 21] = 9
    learner.observe_transition(
        grid_before=predecessor,
        grid_after=after_click,
        action_name="ACTION6",
        action_data={"x": 18, "y": 26},
        available_action_candidates=_actions(),
        terminal_success_confirmed=False,
    )
    learner.observe_transition(
        grid_before=predecessor,
        grid_after=np.zeros((8, 8), dtype=np.int64),
        action_name="ACTION6",
        action_data={"x": 18, "y": 26},
        available_action_candidates=_actions(),
        terminal_success_confirmed=True,
    )
    return learner


def test_learns_relation_only_from_confirmed_terminal_example():
    learner = OnlineTerminalRelationalStencilLearner()
    actions = _actions()
    before = _terminal_predecessor()
    after_click = before.copy()
    after_click[29, 21] = 9

    learner.observe_transition(
        grid_before=before,
        grid_after=after_click,
        action_name="ACTION6",
        action_data={"x": 18, "y": 26},
        available_action_candidates=actions,
        terminal_success_confirmed=False,
    )
    assert learner.summary()["confirmed_marker_rules"] == {}

    learner.observe_transition(
        grid_before=before,
        grid_after=np.zeros((8, 8), dtype=np.int64),
        action_name="ACTION6",
        action_data={"x": 18, "y": 26},
        available_action_candidates=actions,
        terminal_success_confirmed=True,
    )
    assert learner.summary()["confirmed_marker_rules"] == {
        "filled": "different_from_center",
        "void": "equal_to_center",
    }


def test_applies_confirmed_relation_to_new_colors_and_arrangement():
    learner = OnlineTerminalRelationalStencilLearner()
    actions = _actions()
    predecessor = _terminal_predecessor()
    after_click = predecessor.copy()
    after_click[29, 21] = 9
    learner.observe_transition(
        grid_before=predecessor,
        grid_after=after_click,
        action_name="ACTION6",
        action_data={"x": 18, "y": 26},
        available_action_candidates=actions,
        terminal_success_confirmed=False,
    )
    learner.observe_transition(
        grid_before=predecessor,
        grid_after=np.zeros((8, 8), dtype=np.int64),
        action_name="ACTION6",
        action_data={"x": 18, "y": 26},
        available_action_candidates=actions,
        terminal_success_confirmed=True,
    )

    current = _grid(
        12,
        {
            (x, y): 9
            for y in (10, 18, 26)
            for x in (10, 18, 26)
            if (x, y) != STENCIL
        },
    )
    selection = learner.select(
        current_grid=current,
        available_action_candidates=actions,
    )
    assert selection is not None
    assert selection.action_name == "ACTION6"
    assert selection.violations_before == 1
    assert selection.expected_violations_after == 0
    x = selection.action_data["x"]
    y = selection.action_data["y"]
    assert MARKERS[(y - 10) // SPACING][(x - 10) // SPACING] == 0


def test_ablation_never_learns_or_selects():
    learner = OnlineTerminalRelationalStencilLearner(enabled=False)
    actions = _actions()
    predecessor = _terminal_predecessor()
    learner.observe_transition(
        grid_before=predecessor,
        grid_after=np.zeros((8, 8), dtype=np.int64),
        action_name="ACTION6",
        action_data={"x": 18, "y": 26},
        available_action_candidates=actions,
        terminal_success_confirmed=True,
    )
    assert learner.select(
        current_grid=predecessor,
        available_action_candidates=actions,
    ) is None
    assert learner.summary()["terminal_examples"] == 0


def test_structural_signature_ignores_palette_but_preserves_topology():
    learner = _trained_learner()
    colors = {
        (x, y): 9
        for y in (10, 18, 26)
        for x in (10, 18, 26)
        if (x, y) != STENCIL
    }
    first = learner.assess(
        current_grid=_grid(12, colors),
        available_action_candidates=_actions(),
    )
    second = learner.assess(
        current_grid=_grid(17, {
            coordinate: 14 for coordinate in colors
        }),
        available_action_candidates=_actions(),
    )
    fewer_actions = tuple(
        action
        for action in _actions()
        if action.action_args != {"x": 10, "y": 10}
    )
    changed = learner.assess(
        current_grid=_grid(12, colors),
        available_action_candidates=fewer_actions,
    )

    assert first.structural_signature == second.structural_signature
    assert first.structural_signature != changed.structural_signature
    assert (
        first.structural_family_signature
        == second.structural_family_signature
    )


def test_structural_family_ignores_global_layout_extent():
    learner = _trained_learner()
    colors = {
        (x, y): 9
        for y in (10, 18, 26)
        for x in (10, 18, 26)
        if (x, y) != STENCIL
    }
    base = learner.assess(
        current_grid=_grid(12, colors),
        available_action_candidates=_actions(),
    )
    extended_actions = _actions() + (
        SimpleNamespace(
            name="ACTION6",
            action_args={"x": 34, "y": 34},
        ),
    )
    extended = learner.assess(
        current_grid=_grid(12, colors),
        available_action_candidates=extended_actions,
    )

    assert base.structural_signature != extended.structural_signature
    assert (
        base.structural_family_signature
        == extended.structural_family_signature
    )


def test_active_experiment_maximizes_disagreement_between_rule_candidates():
    learner = _trained_learner()
    current = _grid(
        12,
        {
            (x, y): 9
            for y in (10, 18, 26)
            for x in (10, 18, 26)
            if (x, y) != STENCIL
        },
    )
    rules = learner.confirmed_rules()
    inverted = {
        marker: not desired
        for marker, desired in rules.items()
    }

    experiment = learner.select_discriminating_experiment(
        current_grid=current,
        available_action_candidates=_actions(),
        hypothesis_rules={
            "normal": rules,
            "inverted": inverted,
        },
        hypothesis_priority=("normal", "inverted"),
    )

    assert experiment is not None
    assert experiment.hypothesis_id in {"normal", "inverted"}
    assert experiment.compared_hypothesis_ids == ("inverted", "normal")
    assert experiment.disagreement_score > 0
    assert len(set(dict(experiment.predicted_reductions).values())) > 1


def test_permuted_relation_is_a_real_policy_control():
    normal = _trained_learner()
    permuted = _trained_learner(permuted=True)
    current = _grid(
        12,
        {
            (x, y): 9
            for y in (10, 18, 26)
            for x in (10, 18, 26)
            if (x, y) != STENCIL
        },
    )

    normal_selection = normal.select(
        current_grid=current,
        available_action_candidates=_actions(),
    )
    permuted_selection = permuted.select(
        current_grid=current,
        available_action_candidates=_actions(),
    )

    assert normal_selection is not None
    assert permuted_selection is not None
    normal_coordinate = (
        normal_selection.action_data["x"],
        normal_selection.action_data["y"],
    )
    permuted_coordinate = (
        permuted_selection.action_data["x"],
        permuted_selection.action_data["y"],
    )
    assert normal_coordinate != permuted_coordinate
