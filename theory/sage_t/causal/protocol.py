"""Preregistered SAGE.T11/T12 split firewall and promotion gates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from theory.sage11.splits import SAGE11_SPLITS, short_game_id

FT09_BASELINE: Mapping[str, int] = {
    "actions": 1822,
    "levels": 43,
    "max_level": 6,
    "wins": 3,
    "protected_route_preemptions": 0,
    "frontier_experiments": 9,
    "multiform_selections": 20,
}


class CausalProtocolStage(str, Enum):
    SOURCE_TRAIN = "source_train"
    SOURCE_VALIDATION = "source_validation"
    HISTORICAL = "historical"
    HOLDOUT_CONFIRMATION = "holdout_confirmation"
    REGRESSION = "regression"


@dataclass(frozen=True)
class CausalProtocol:
    format_version: str = "sage-t11-causal-protocol-v1"
    posterior_cap: int = 64
    decision_particle_cap: int = 16
    ordinary_horizon: int = 3
    maximum_terminal_probe_risk: float = 0.05
    maximum_interventions_per_reset: int = 5
    authority_default: str = "shadow"
    split_checksum: str = SAGE11_SPLITS.checksum
    ft09_baseline: Mapping[str, int] = field(default_factory=lambda: dict(FT09_BASELINE))

    @property
    def checksum(self) -> str:
        payload = asdict(self)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class GateReceipt:
    stage: CausalProtocolStage
    passed: bool
    protocol_checksum: str
    metrics: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""

    @property
    def checksum(self) -> str:
        payload = {
            "stage": self.stage.value,
            "passed": self.passed,
            "protocol_checksum": self.protocol_checksum,
            "metrics": dict(self.metrics),
            "reason": self.reason,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class CausalEvaluationFirewall:
    def __init__(self, *, protocol: CausalProtocol | None = None) -> None:
        self.protocol = protocol or CausalProtocol()
        self.source_validation_receipt: GateReceipt | None = None
        self.holdout_consumed = False

    def assert_authorized(
        self,
        game_ids: Sequence[str],
        *,
        stage: CausalProtocolStage | str,
    ) -> None:
        stage = stage if isinstance(stage, CausalProtocolStage) else CausalProtocolStage(str(stage))
        groups = {short_game_id(game): SAGE11_SPLITS.split_for(game) for game in game_ids}
        allowed = {
            CausalProtocolStage.SOURCE_TRAIN: {"source_train"},
            CausalProtocolStage.SOURCE_VALIDATION: {"source_validation"},
            CausalProtocolStage.HISTORICAL: {"historical_benchmark"},
            CausalProtocolStage.HOLDOUT_CONFIRMATION: {"neuro_holdout_v1"},
            CausalProtocolStage.REGRESSION: {"historical_benchmark", "ar25_regression_only"},
        }[stage]
        violations = {game: group for game, group in groups.items() if group not in allowed}
        if violations:
            raise ValueError(f"causal split firewall rejected {stage.value}: {violations}")
        if stage is CausalProtocolStage.HOLDOUT_CONFIRMATION:
            if self.source_validation_receipt is None or not self.source_validation_receipt.passed:
                raise ValueError("holdout remains closed without a passing source-validation receipt")
            if self.source_validation_receipt.protocol_checksum != self.protocol.checksum:
                raise ValueError("source-validation receipt belongs to another protocol")
            if self.holdout_consumed:
                raise ValueError("the neural holdout is single-confirmation only")

    def register_source_validation(self, receipt: GateReceipt) -> None:
        if receipt.stage is not CausalProtocolStage.SOURCE_VALIDATION:
            raise ValueError("expected a source-validation gate receipt")
        if receipt.protocol_checksum != self.protocol.checksum:
            raise ValueError("gate receipt protocol checksum mismatch")
        if receipt.passed and (
            int(receipt.metrics.get("games_with_progress", 0)) < 2
            or int(receipt.metrics.get("safety_regressions", 1)) != 0
            or receipt.metrics.get("posterior_ablation_advantage") is not True
        ):
            raise ValueError(
                "source-validation receipt does not satisfy the frozen promotion gate"
            )
        self.source_validation_receipt = receipt

    def consume_holdout(self, game_ids: Sequence[str]) -> None:
        self.assert_authorized(
            game_ids,
            stage=CausalProtocolStage.HOLDOUT_CONFIRMATION,
        )
        self.holdout_consumed = True


def ft09_non_regression(metrics: Mapping[str, Any]) -> bool:
    return bool(
        int(metrics.get("max_level", -1)) >= FT09_BASELINE["max_level"]
        and int(metrics.get("levels", -1)) >= FT09_BASELINE["levels"]
        and int(metrics.get("wins", -1)) >= FT09_BASELINE["wins"]
        and int(metrics.get("protected_route_preemptions", 1)) == 0
    )


def ft09_efficiency_gain(metrics: Mapping[str, Any]) -> bool:
    return bool(
        int(metrics.get("actions", FT09_BASELINE["actions"]))
        < FT09_BASELINE["actions"]
        or int(
            metrics.get(
                "frontier_experiments", FT09_BASELINE["frontier_experiments"]
            )
        )
        < FT09_BASELINE["frontier_experiments"]
        or int(
            metrics.get(
                "multiform_selections", FT09_BASELINE["multiform_selections"]
            )
        )
        < FT09_BASELINE["multiform_selections"]
    )


__all__ = [
    "FT09_BASELINE",
    "CausalEvaluationFirewall",
    "CausalProtocol",
    "CausalProtocolStage",
    "GateReceipt",
    "ft09_efficiency_gain",
    "ft09_non_regression",
]
