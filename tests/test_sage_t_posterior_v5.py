from __future__ import annotations

import pytest

from theory.sage_t.posterior_v3 import ChannelPosteriorUpdatePolicy
from theory.sage_t.posterior_v5 import (
    T8_6D_POLICIES,
    terminal_temperature_policy,
)


def test_terminal_sweep_is_exactly_the_preregistered_grid() -> None:
    assert tuple(T8_6D_POLICIES) == (
        "legacy",
        "terminal_tempered_25",
        "terminal_tempered_20",
        "terminal_tempered_15",
        "terminal_tempered_10",
    )
    assert [
        policy.channel_multiplier("terminal", 9)
        for name, policy in T8_6D_POLICIES.items()
        if name != "legacy"
    ] == [0.25, 0.20, 0.15, 0.10]


def test_terminal_sweep_never_tempers_dynamics_or_goal_channels() -> None:
    for policy in T8_6D_POLICIES.values():
        for channel in ("objects", "relations", "topology", "progress", "goal"):
            assert policy.channel_multiplier(channel, 25) == 1.0


def test_temperature_25_is_equivalent_to_t8_6b_terminal_policy() -> None:
    expected = ChannelPosteriorUpdatePolicy.terminal_tempered()
    candidate = T8_6D_POLICIES["terminal_tempered_25"]

    assert candidate.channel_temperatures == expected.channel_temperatures
    assert candidate.repeated_context_channels == expected.repeated_context_channels


def test_terminal_temperature_rejects_unregistered_domain_values() -> None:
    with pytest.raises(ValueError):
        terminal_temperature_policy(0.0)
    with pytest.raises(ValueError):
        terminal_temperature_policy(1.1)
