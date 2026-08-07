"""T8.6d pre-registered terminal-only temperature sweep."""

from __future__ import annotations

from collections.abc import Mapping

from .posterior_v3 import CHANNELS, ChannelPosteriorUpdatePolicy


def terminal_temperature_policy(
    temperature: float,
    *,
    name: str | None = None,
) -> ChannelPosteriorUpdatePolicy:
    value = float(temperature)
    if not 0.0 < value <= 1.0:
        raise ValueError("terminal temperature must be in (0, 1]")
    label = name or f"terminal_tempered_{round(100 * value):02d}"
    return ChannelPosteriorUpdatePolicy(
        name=label,
        channel_temperatures=tuple(
            (channel, value if channel == "terminal" else 1.0)
            for channel in CHANNELS
        ),
    )


T8_6D_POLICIES: Mapping[str, ChannelPosteriorUpdatePolicy] = {
    policy.name: policy
    for policy in (
        ChannelPosteriorUpdatePolicy.legacy(),
        terminal_temperature_policy(0.25),
        terminal_temperature_policy(0.20),
        terminal_temperature_policy(0.15),
        terminal_temperature_policy(0.10),
    )
}


__all__ = [
    "T8_6D_POLICIES",
    "terminal_temperature_policy",
]
