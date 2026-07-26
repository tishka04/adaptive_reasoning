"""Pre-registered SAGE.11 shadow, bounded, and holdout promotion gates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .splits import (
    ArtifactPurpose,
    NEURO_HOLDOUT_V1,
    SAGE11_SPLITS,
    short_game_id,
)


@dataclass(frozen=True)
class ShadowGateMetrics:
    action_identity_mismatches: int
    would_be_successful_route_preemptions: int
    neural_top1_productivity: float
    symbolic_top1_productivity: float
    neural_top3_productivity: float
    symbolic_top3_productivity: float
    risk_ece: float
    noop_ece: float
    inference_peak_ms: float
    inference_budget_ms: float


def shadow_gate_report(
    metrics: ShadowGateMetrics,
) -> Dict[str, Any]:
    gates = {
        "byte_identical_executed_actions": (
            metrics.action_identity_mismatches == 0
        ),
        "zero_would_be_successful_route_preemptions": (
            metrics.would_be_successful_route_preemptions == 0
        ),
        "retrospective_top1_productivity": (
            metrics.neural_top1_productivity
            > metrics.symbolic_top1_productivity
        ),
        "retrospective_top3_productivity": (
            metrics.neural_top3_productivity
            > metrics.symbolic_top3_productivity
        ),
        "risk_calibration": metrics.risk_ece <= 0.10,
        "noop_calibration": metrics.noop_ece <= 0.10,
        "inference_cost_budget": (
            metrics.inference_peak_ms <= metrics.inference_budget_ms
        ),
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "metrics": asdict(metrics),
    }


@dataclass(frozen=True)
class PairedRunResult:
    game_id: str
    seed: int
    active_score: float
    off_score: float
    active_levels: int
    off_levels: int
    active_wins: int
    off_wins: int
    unsafe_outcomes: int = 0
    controller_errors: int = 0
    protected_route_preemptions: int = 0
    active_digest: str = ""
    off_digest: str = ""

    @property
    def score_delta(self) -> float:
        return self.active_score - self.off_score

    @property
    def novel_success(self) -> bool:
        return bool(
            self.active_levels > self.off_levels
            or self.active_wins > self.off_wins
        )


def paired_bootstrap_lower_bound(
    deltas: Sequence[float],
    *,
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = 11,
) -> float:
    values = np.asarray(deltas, dtype=float)
    if values.size == 0:
        raise ValueError("paired bootstrap requires at least one delta")
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(
        0,
        len(values),
        size=(max(100, int(samples)), len(values)),
    )
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 1.0 - float(confidence)))


def holdout_promotion_report(
    runs: Sequence[PairedRunResult],
) -> Dict[str, Any]:
    games = {short_game_id(run.game_id) for run in runs}
    SAGE11_SPLITS.assert_authorized(
        games,
        purpose=ArtifactPurpose.HOLDOUT_CONFIRMATION,
    )
    expected_pairs = {
        (game, seed)
        for game in NEURO_HOLDOUT_V1
        for seed in range(5)
    }
    observed_pairs = {
        (short_game_id(run.game_id), int(run.seed))
        for run in runs
    }
    lower_bound = paired_bootstrap_lower_bound([
        run.score_delta for run in runs
    ])
    gates = {
        "complete_5x5_protocol": observed_pairs == expected_pairs,
        "bootstrap_95pct_lower_score_gain_above_zero": lower_bound > 0.0,
        "no_win_lost": all(
            run.active_wins >= run.off_wins for run in runs
        ),
        "at_least_one_new_level_or_win": any(
            run.novel_success for run in runs
        ),
        "zero_unsafe": all(run.unsafe_outcomes == 0 for run in runs),
        "zero_controller_errors": all(
            run.controller_errors == 0 for run in runs
        ),
        "zero_preemptions": all(
            run.protected_route_preemptions == 0 for run in runs
        ),
        "paired_digests_present": all(
            run.active_digest and run.off_digest for run in runs
        ),
    }
    return {
        "passed": all(gates.values()),
        "otherwise": "remain_in_shadow_and_document_negative_result",
        "gates": gates,
        "bootstrap_95pct_lower_bound": lower_bound,
        "runs": [asdict(run) for run in runs],
    }


class CheckpointedRunLog:
    """Atomic per-run JSON checkpoints for multi-day evaluation matrices."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._runs: Dict[str, Mapping[str, Any]] = {}
        if self.path.is_file():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            self._runs.update(dict(payload.get("runs", {}) or {}))

    @staticmethod
    def run_key(
        *,
        game_id: str,
        seed: int,
        arm: str,
        budget: int,
        resets: int,
    ) -> str:
        return "|".join((
            short_game_id(game_id),
            str(int(seed)),
            str(arm),
            str(int(budget)),
            str(int(resets)),
        ))

    def completed(self, run_key: str) -> bool:
        return str(run_key) in self._runs

    def record(self, run_key: str, payload: Mapping[str, Any]) -> None:
        self._runs[str(run_key)] = dict(payload)
        document = {
            "format_version": "sage11-run-log-v1",
            "split_registry_checksum": SAGE11_SPLITS.checksum,
            "runs": dict(sorted(self._runs.items())),
        }
        document["checksum"] = _checksum({
            key: value
            for key, value in document.items()
            if key != "checksum"
        })
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)


def _checksum(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


__all__ = [
    "CheckpointedRunLog",
    "PairedRunResult",
    "ShadowGateMetrics",
    "holdout_promotion_report",
    "paired_bootstrap_lower_bound",
    "shadow_gate_report",
]
