"""Terminal horizon estimation helpers for P3 policy probes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TerminalHorizonEstimate:
    observed: bool
    estimated_moves_remaining: int | None
    estimated_total_budget: int | None
    moves_used: int
    confidence: float
    source: str
    terminal_fraction_remaining: float | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed": self.observed,
            "estimated_moves_remaining": (
                None
                if self.estimated_moves_remaining is None
                else int(self.estimated_moves_remaining)
            ),
            "estimated_total_budget": (
                None if self.estimated_total_budget is None else int(self.estimated_total_budget)
            ),
            "moves_used": int(self.moves_used),
            "terminal_fraction_remaining": self.terminal_fraction_remaining,
            "confidence": float(self.confidence),
            "source": self.source,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class BudgetBarCandidate:
    location: str
    orientation: str
    bbox: tuple[int, int, int, int]
    fill_value: int
    empty_value: int | None
    length: int
    initial_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "location": self.location,
            "orientation": self.orientation,
            "bar_bbox": list(self.bbox),
            "fill_value": int(self.fill_value),
            "empty_value": None if self.empty_value is None else int(self.empty_value),
            "length": int(self.length),
            "initial_score": float(self.initial_score),
        }


def terminal_fraction(
    estimated_moves_remaining: int | None,
    estimated_total_budget: int | None,
) -> float | None:
    if estimated_moves_remaining is None or not estimated_total_budget:
        return None
    total = max(1, int(estimated_total_budget))
    return max(0.0, min(1.0, float(estimated_moves_remaining) / float(total)))


def moves_used_from_policy_state(policy_state: Any) -> int:
    """Count all consumed environment actions, not only ACTION6 decisions."""
    if policy_state is None:
        return 0
    if isinstance(policy_state, Mapping):
        for key in ("env_actions_executed", "moves_used", "policy_steps"):
            if key in policy_state:
                return max(0, int(policy_state.get(key) or 0))
    if isinstance(policy_state, Sequence) and not isinstance(policy_state, (str, bytes)):
        return sum(int(getattr(step, "env_actions", 1) or 0) for step in policy_state)
    for attr in ("env_actions_executed", "moves_used", "policy_steps"):
        if hasattr(policy_state, attr):
            return max(0, int(getattr(policy_state, attr) or 0))
    return 0


def estimate_moves_remaining_fallback(
    policy_state: Any,
    *,
    terminal_budget_estimate: int = 64,
) -> TerminalHorizonEstimate:
    moves_used = moves_used_from_policy_state(policy_state)
    total = max(0, int(terminal_budget_estimate))
    return TerminalHorizonEstimate(
        observed=False,
        estimated_moves_remaining=max(0, total - moves_used),
        estimated_total_budget=total,
        moves_used=moves_used,
        confidence=0.5,
        source="empirical_fallback",
        terminal_fraction_remaining=terminal_fraction(max(0, total - moves_used), total),
        evidence={
            "terminal_budget_estimate": total,
            "estimation_rule": "max(0, terminal_budget_estimate - moves_used)",
            "moves_used_variable": "env_actions_executed",
        },
    )


def estimate_from_environment_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    policy_state: Any = None,
) -> TerminalHorizonEstimate | None:
    if not metadata:
        return None
    moves_used = moves_used_from_policy_state(policy_state)
    if metadata.get("estimated_moves_remaining") is not None:
        remaining = max(0, int(metadata.get("estimated_moves_remaining") or 0))
        total = metadata.get("estimated_total_budget")
        return TerminalHorizonEstimate(
            observed=True,
            estimated_moves_remaining=remaining,
            estimated_total_budget=None if total is None else int(total),
            moves_used=moves_used,
            confidence=0.9,
            source="environment_metadata",
            terminal_fraction_remaining=terminal_fraction(
                remaining,
                None if total is None else int(total),
            ),
            evidence={"metadata_field": "estimated_moves_remaining"},
        )
    if metadata.get("moves_remaining") is not None:
        remaining = max(0, int(metadata.get("moves_remaining") or 0))
        total = metadata.get("total_budget")
        return TerminalHorizonEstimate(
            observed=True,
            estimated_moves_remaining=remaining,
            estimated_total_budget=None if total is None else int(total),
            moves_used=moves_used,
            confidence=0.9,
            source="environment_metadata",
            terminal_fraction_remaining=terminal_fraction(
                remaining,
                None if total is None else int(total),
            ),
            evidence={"metadata_field": "moves_remaining"},
        )
    if metadata.get("total_budget") is not None:
        total = max(0, int(metadata.get("total_budget") or 0))
        return TerminalHorizonEstimate(
            observed=True,
            estimated_moves_remaining=max(0, total - moves_used),
            estimated_total_budget=total,
            moves_used=moves_used,
            confidence=0.85,
            source="environment_metadata",
            terminal_fraction_remaining=terminal_fraction(max(0, total - moves_used), total),
            evidence={"metadata_field": "total_budget"},
        )
    return None


def _grid_from_observation(observation: Any) -> np.ndarray | None:
    if observation is None:
        return None
    if isinstance(observation, Mapping):
        for key in ("grid", "current_grid", "frame", "observation"):
            if key in observation:
                return _grid_from_observation(observation.get(key))
        return None
    try:
        grid = np.asarray(observation, dtype=np.int32)
    except (TypeError, ValueError):
        return None
    if grid.ndim != 2 or grid.size == 0:
        return None
    return grid


def _line_candidate(
    *,
    line: np.ndarray,
    location: str,
    orientation: str,
    bbox: tuple[int, int, int, int],
) -> BudgetBarCandidate | None:
    if line.size < 4:
        return None
    values, counts = np.unique(line, return_counts=True)
    if values.size > max(4, int(line.size * 0.35)):
        return None
    nonzero = [(int(value), int(count)) for value, count in zip(values, counts) if int(value) != 0]
    if nonzero:
        fill_value, fill_count = max(nonzero, key=lambda row: row[1])
    else:
        fill_value = int(values[np.argmax(counts)])
        fill_count = int(max(counts))
    empty_values = [int(value) for value in values if int(value) != int(fill_value)]
    empty_value = empty_values[0] if empty_values else None
    fill_ratio = float(fill_count) / float(max(1, line.size))
    low_entropy_bonus = 1.0 - min(1.0, float(values.size - 1) / 4.0)
    if fill_ratio < 0.05:
        return None
    return BudgetBarCandidate(
        location=location,
        orientation=orientation,
        bbox=bbox,
        fill_value=int(fill_value),
        empty_value=empty_value,
        length=int(line.size),
        initial_score=round(0.4 + 0.4 * fill_ratio + 0.2 * low_entropy_bonus, 4),
    )


def detect_budget_bar_candidates(grid: Any) -> list[BudgetBarCandidate]:
    array = _grid_from_observation(grid)
    if array is None:
        return []
    height, width = array.shape
    edge = min(3, height, width)
    candidates: list[BudgetBarCandidate] = []
    for offset in range(edge):
        top_r = offset
        bottom_r = height - 1 - offset
        top = _line_candidate(
            line=array[top_r, :],
            location="top",
            orientation="horizontal_top",
            bbox=(top_r, 0, top_r, width - 1),
        )
        bottom = _line_candidate(
            line=array[bottom_r, :],
            location="bottom",
            orientation="horizontal_bottom",
            bbox=(bottom_r, 0, bottom_r, width - 1),
        )
        if top is not None:
            candidates.append(top)
        if bottom is not None and bottom.bbox != (top_r, 0, top_r, width - 1):
            candidates.append(bottom)
        left_c = offset
        right_c = width - 1 - offset
        left = _line_candidate(
            line=array[:, left_c],
            location="left",
            orientation="vertical_left",
            bbox=(0, left_c, height - 1, left_c),
        )
        right = _line_candidate(
            line=array[:, right_c],
            location="right",
            orientation="vertical_right",
            bbox=(0, right_c, height - 1, right_c),
        )
        if left is not None:
            candidates.append(left)
        if right is not None and right.bbox != (0, left_c, height - 1, left_c):
            candidates.append(right)
    return candidates


def _candidate_line(candidate: BudgetBarCandidate, grid: Any) -> np.ndarray | None:
    array = _grid_from_observation(grid)
    if array is None:
        return None
    r0, c0, r1, c1 = candidate.bbox
    if r0 == r1 and 0 <= r0 < array.shape[0]:
        return array[r0, max(0, c0) : min(array.shape[1], c1 + 1)]
    if c0 == c1 and 0 <= c0 < array.shape[1]:
        return array[max(0, r0) : min(array.shape[0], r1 + 1), c0]
    return None


def measure_filled_length(candidate: BudgetBarCandidate, grid: Any) -> int:
    line = _candidate_line(candidate, grid)
    if line is None or line.size == 0:
        return 0
    return int(np.sum(line == int(candidate.fill_value)))


def _history_grids(
    *,
    observation: Any = None,
    history: Any = None,
) -> list[Any]:
    grids: list[Any] = []
    if isinstance(history, Mapping):
        source = history.get("grids")
        if source is None:
            source = history.get("observations")
        if source is not None:
            grids.extend(list(source))
    elif isinstance(history, Sequence) and not isinstance(history, (str, bytes)):
        grids.extend(list(history))
    if observation is not None:
        obs_grid = _grid_from_observation(observation)
        if obs_grid is not None:
            grids.append(obs_grid)
    normalized = []
    for item in grids:
        grid = _grid_from_observation(item)
        if grid is not None:
            normalized.append(grid)
    return normalized


def _monotonic_direction(values: Sequence[int]) -> str | None:
    if len(values) < 2:
        return None
    non_increasing = all(values[index] >= values[index + 1] for index in range(len(values) - 1))
    non_decreasing = all(values[index] <= values[index + 1] for index in range(len(values) - 1))
    if non_increasing and len(set(values)) > 1:
        return "non_increasing"
    if non_decreasing and len(set(values)) > 1:
        return "non_decreasing"
    return None


def score_budget_bar_candidate(
    candidate: BudgetBarCandidate,
    *,
    observation: Any = None,
    history: Any = None,
) -> dict[str, Any]:
    grids = _history_grids(observation=observation, history=history)
    if not grids:
        return {
            **candidate.to_dict(),
            "score": 0.0,
            "filled_values": [],
            "monotonic_delta_observed": False,
            "ticks_lost_per_action": None,
            "stable_location": True,
        }
    values = [measure_filled_length(candidate, grid) for grid in grids]
    direction = _monotonic_direction(values)
    deltas = [values[index] - values[index + 1] for index in range(len(values) - 1)]
    changed_steps = len([delta for delta in deltas if delta != 0])
    abs_deltas = [abs(delta) for delta in deltas if delta != 0]
    ticks = None if not abs_deltas else float(np.median(abs_deltas))
    action_correlation = changed_steps / max(1, len(deltas))
    monotonic = direction is not None
    tick_bonus = 1.0 if ticks == 1.0 else 0.6 if ticks and ticks <= 3.0 else 0.0
    score = (
        0.25 * candidate.initial_score
        + 0.35 * float(monotonic)
        + 0.25 * action_correlation
        + 0.15 * tick_bonus
    )
    return {
        **candidate.to_dict(),
        "score": round(float(score), 4),
        "filled_values": [int(value) for value in values],
        "monotonic_delta_observed": monotonic,
        "monotonic_direction": direction,
        "ticks_lost_per_action": ticks,
        "ticks_changed_per_action": ticks,
        "stable_location": True,
        "action_correlation": round(float(action_correlation), 4),
    }


def estimate_from_hud_bar(
    hud_observation: Mapping[str, Any] | None,
    *,
    observation: Any = None,
    history: Any = None,
    policy_state: Any = None,
    confidence_threshold: float = 0.7,
) -> TerminalHorizonEstimate | None:
    """Estimate from a pre-extracted HUD bar or a simple monotone edge detector."""
    if not hud_observation or not bool(hud_observation.get("action_budget_bar_detected")):
        grids = _history_grids(observation=observation, history=history)
        if not grids:
            return None
        scored: list[dict[str, Any]] = []
        for candidate in detect_budget_bar_candidates(grids[-1]):
            scored.append(
                score_budget_bar_candidate(
                    candidate,
                    observation=observation,
                    history=history,
                )
            )
        if not scored:
            return None
        best = max(scored, key=lambda row: float(row.get("score", 0.0) or 0.0))
        if float(best.get("score", 0.0) or 0.0) < float(confidence_threshold):
            return None
        filled_values = [int(value) for value in best.get("filled_values", []) or []]
        filled = int(filled_values[-1]) if filled_values else 0
        total = int(best.get("length", 0) or 0)
        direction = str(best.get("monotonic_direction", "") or "")
        if direction == "non_decreasing":
            remaining = max(0, total - filled)
            semantics = "elapsed_ticks_increasing"
        else:
            remaining = filled
            semantics = "remaining_ticks_decreasing"
        best = {
            **best,
            "bar_semantics": semantics,
            "estimated_remaining_rule": (
                "length - filled_length"
                if semantics == "elapsed_ticks_increasing"
                else "filled_length"
            ),
        }
        moves_used = moves_used_from_policy_state(policy_state)
        return TerminalHorizonEstimate(
            observed=True,
            estimated_moves_remaining=remaining,
            estimated_total_budget=total,
            moves_used=moves_used,
            confidence=float(best.get("score", 0.0) or 0.0),
            source="hud_bar",
            terminal_fraction_remaining=terminal_fraction(remaining, total),
            evidence=best,
        )

    if hud_observation.get("estimated_moves_remaining") is None:
        return None
    moves_used = moves_used_from_policy_state(policy_state)
    remaining = max(0, int(hud_observation.get("estimated_moves_remaining") or 0))
    filled = hud_observation.get("filled_cells")
    empty = hud_observation.get("empty_cells")
    total = None
    if filled is not None and empty is not None:
        total = int(filled or 0) + int(empty or 0)
    confidence = float(hud_observation.get("confidence", 0.75) or 0.75)
    return TerminalHorizonEstimate(
        observed=True,
        estimated_moves_remaining=remaining,
        estimated_total_budget=total,
        moves_used=moves_used,
        confidence=confidence,
        source="hud_bar",
        terminal_fraction_remaining=terminal_fraction(remaining, total),
        evidence={
            "bar_orientation": hud_observation.get("orientation"),
            "bar_bbox": hud_observation.get("bar_bbox"),
            "filled_cells": filled,
            "empty_cells": empty,
            "monotonic_delta_observed": hud_observation.get("monotonic_delta_observed"),
            "ticks_lost_per_action": hud_observation.get("ticks_lost_per_action"),
        },
    )


@dataclass(frozen=True)
class TerminalHorizonObserver:
    hud_confidence_threshold: float = 0.7
    empirical_confidence_threshold: float = 0.4
    terminal_budget_estimate: int | None = 64

    def estimate(
        self,
        *,
        observation: Any = None,
        history: Any = None,
        env_info: Mapping[str, Any] | None = None,
        policy_state: Any = None,
    ) -> TerminalHorizonEstimate:
        metadata = estimate_from_environment_metadata(env_info, policy_state=policy_state)
        if metadata is not None:
            return metadata

        hud_estimate = estimate_from_hud_bar(
            (
                None
                if observation is None
                else dict(observation.get("hud_bar", observation))
                if isinstance(observation, Mapping)
                else None
            ),
            observation=observation,
            history=history,
            policy_state=policy_state,
            confidence_threshold=self.hud_confidence_threshold,
        )
        if hud_estimate is not None and hud_estimate.confidence >= self.hud_confidence_threshold:
            return hud_estimate

        if self.terminal_budget_estimate is not None:
            empirical = estimate_moves_remaining_fallback(
                policy_state,
                terminal_budget_estimate=int(self.terminal_budget_estimate),
            )
            if empirical.confidence >= self.empirical_confidence_threshold:
                return empirical

        return TerminalHorizonEstimate(
            observed=False,
            estimated_moves_remaining=None,
            estimated_total_budget=None,
            moves_used=moves_used_from_policy_state(policy_state),
            confidence=0.0,
            source="unknown",
            evidence={"reason": "no_terminal_horizon_source_available"},
        )


def estimate_terminal_horizon(
    *,
    observation: Any = None,
    history: Any = None,
    policy_state: Any = None,
    environment_metadata: Mapping[str, Any] | None = None,
    terminal_budget_estimate: int | None = 64,
) -> TerminalHorizonEstimate:
    return TerminalHorizonObserver(
        terminal_budget_estimate=terminal_budget_estimate,
    ).estimate(
        observation=observation,
        history=history,
        env_info=environment_metadata,
        policy_state=policy_state,
    )
