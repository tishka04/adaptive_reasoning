"""Common utilities for lightweight online learning in V4."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import log
from typing import Iterable


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def bucket_float(value: float, step: float = 0.2, maximum: int = 5) -> int:
    if step <= 0:
        return 0
    return max(0, min(maximum, int(value / step)))


def bucket_int(value: int, step: int = 2, maximum: int = 6) -> int:
    if step <= 0:
        return 0
    return max(0, min(maximum, int(value / step)))


def entropy(values: Iterable[float]) -> float:
    probs = [max(0.0, float(item)) for item in values]
    total = sum(probs)
    if total <= 0.0:
        return 0.0
    result = 0.0
    for value in probs:
        if value <= 0.0:
            continue
        p = value / total
        result -= p * log(p + 1e-12)
    return result


def outcome_reward(
    lp_gain: float = 0.0,
    sp_gain: float = 0.0,
    tp_gain: float = 0.0,
    solved: bool = False,
    sterile: bool = False,
    loop: bool = False,
) -> float:
    reward = (
        0.20 * max(0.0, lp_gain)
        + 0.35 * max(0.0, sp_gain)
        + 0.45 * max(0.0, tp_gain)
        + (0.45 if solved else 0.0)
        - (0.30 if sterile else 0.0)
        - (0.15 if loop else 0.0)
    )
    return clamp(reward)


@dataclass
class RunningAverage:
    count: float = 0.0
    mean: float = 0.5

    def update(self, value: float, weight: float = 1.0) -> None:
        weight = max(0.0, float(weight))
        if weight <= 0.0:
            return
        value = clamp(float(value))
        total = self.count + weight
        self.mean = (self.mean * self.count + value * weight) / max(total, 1e-9)
        self.count = total

    @property
    def confidence(self) -> float:
        return clamp(self.count / 8.0)


class ContextTable:
    """Track coarse contextual values with simple online averaging."""

    def __init__(self) -> None:
        self.stats: dict[tuple[object, ...], RunningAverage] = defaultdict(RunningAverage)

    def estimate(
        self,
        signatures: list[tuple[tuple[object, ...], float]],
        prior: float = 0.5,
    ) -> float:
        weighted = 0.0
        total = 0.0
        for signature, weight in signatures:
            stat = self.stats.get(signature)
            if stat is None or stat.count <= 0.0:
                continue
            support = max(0.0, float(weight)) * stat.confidence
            weighted += stat.mean * support
            total += support
        if total <= 0.0:
            return clamp(prior)
        learned = weighted / total
        blend = clamp(total / (total + 1.25), 0.0, 0.70)
        return clamp((1.0 - blend) * prior + blend * learned)

    def support(self, signatures: list[tuple[tuple[object, ...], float]]) -> float:
        total = 0.0
        for signature, weight in signatures:
            stat = self.stats.get(signature)
            if stat is None:
                continue
            total += stat.count * max(0.0, float(weight))
        return total

    def update(self, signature: tuple[object, ...], value: float, weight: float = 1.0) -> None:
        self.stats[signature].update(value, weight=weight)

