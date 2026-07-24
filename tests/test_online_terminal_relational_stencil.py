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
