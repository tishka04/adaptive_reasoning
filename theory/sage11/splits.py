"""Immutable SAGE.11 experimental splits and leakage firewall."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Tuple


SOURCE_TRAIN: Tuple[str, ...] = (
    "bp35",
    "cd82",
    "dc22",
    "g50t",
    "ka59",
    "lf52",
    "lp85",
    "sp80",
    "su15",
    "tr87",
    "tu93",
)
SOURCE_VALIDATION: Tuple[str, ...] = ("re86", "ls20", "sc25")
NEURO_HOLDOUT_V1: Tuple[str, ...] = (
    "s5i5",
    "vc33",
    "m0r0",
    "sk48",
    "r11l",
)
HISTORICAL_BENCHMARK: Tuple[str, ...] = (
    "wa30",
    "tn36",
    "ft09",
    "cn04",
    "sb26",
)
AR25_REGRESSION_ONLY: Tuple[str, ...] = ("ar25",)


class ArtifactPurpose(str, Enum):
    """Authorized reason for reading or producing a split-bound artifact."""

    TRAIN = "train"
    VALIDATE_SOURCE = "validate_source"
    HISTORICAL_REPORT = "historical_report"
    HOLDOUT_CONFIRMATION = "holdout_confirmation"
    REGRESSION = "regression"


@dataclass(frozen=True)
class Sage11SplitRegistry:
    """Content-addressed registry used by collection, training, and CI."""

    source_train: Tuple[str, ...] = SOURCE_TRAIN
    source_validation: Tuple[str, ...] = SOURCE_VALIDATION
    neuro_holdout_v1: Tuple[str, ...] = NEURO_HOLDOUT_V1
    historical_benchmark: Tuple[str, ...] = HISTORICAL_BENCHMARK
    ar25_regression_only: Tuple[str, ...] = AR25_REGRESSION_ONLY
    format_version: str = "sage11-splits-v1"

    def __post_init__(self) -> None:
        groups = self.groups()
        seen: Dict[str, str] = {}
        for group_name, games in groups.items():
            if not games:
                raise ValueError(f"SAGE.11 split {group_name} is empty")
            for game in games:
                short = short_game_id(game)
                previous = seen.get(short)
                if previous is not None:
                    raise ValueError(
                        f"SAGE.11 game {short} occurs in {previous} and "
                        f"{group_name}"
                    )
                seen[short] = group_name
        if len(seen) != 25:
            raise ValueError(
                f"SAGE.11 registry must cover 25 games, found {len(seen)}"
            )

    def groups(self) -> Dict[str, Tuple[str, ...]]:
        return {
            "source_train": tuple(self.source_train),
            "source_validation": tuple(self.source_validation),
            "neuro_holdout_v1": tuple(self.neuro_holdout_v1),
            "historical_benchmark": tuple(self.historical_benchmark),
            "ar25_regression_only": tuple(self.ar25_regression_only),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": self.format_version,
            "groups": {
                name: list(games)
                for name, games in self.groups().items()
            },
            "policy": {
                "tuning_splits": [
                    "source_train",
                    "source_validation",
                ],
                "historical_is_report_only": True,
                "holdout_is_single_confirmation_only": True,
                "ar25_is_regression_only": True,
            },
        }

    @property
    def checksum(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def split_for(self, game_id: str) -> str:
        short = short_game_id(game_id)
        for name, games in self.groups().items():
            if short in games:
                return name
        raise ValueError(f"game {short!r} is outside SAGE.11 registry")

    def assert_authorized(
        self,
        game_ids: Iterable[str],
        *,
        purpose: ArtifactPurpose | str,
    ) -> None:
        normalized = (
            purpose
            if isinstance(purpose, ArtifactPurpose)
            else ArtifactPurpose(str(purpose))
        )
        permitted = {
            ArtifactPurpose.TRAIN: {"source_train"},
            ArtifactPurpose.VALIDATE_SOURCE: {
                "source_train",
                "source_validation",
            },
            ArtifactPurpose.HISTORICAL_REPORT: {
                "historical_benchmark",
            },
            ArtifactPurpose.HOLDOUT_CONFIRMATION: {
                "neuro_holdout_v1",
            },
            ArtifactPurpose.REGRESSION: {
                "ar25_regression_only",
                "historical_benchmark",
            },
        }[normalized]
        violations = {
            short_game_id(game): self.split_for(game)
            for game in game_ids
            if self.split_for(game) not in permitted
        }
        if violations:
            detail = ", ".join(
                f"{game}:{split}"
                for game, split in sorted(violations.items())
            )
            raise ValueError(
                f"SAGE.11 leakage firewall rejected {normalized.value}: "
                f"{detail}"
            )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "Sage11SplitRegistry":
        if payload.get("format_version") != "sage11-splits-v1":
            raise ValueError("unsupported SAGE.11 split registry")
        groups = dict(payload.get("groups", {}) or {})
        registry = cls(
            source_train=tuple(groups.get("source_train", ())),
            source_validation=tuple(groups.get("source_validation", ())),
            neuro_holdout_v1=tuple(groups.get("neuro_holdout_v1", ())),
            historical_benchmark=tuple(
                groups.get("historical_benchmark", ())
            ),
            ar25_regression_only=tuple(
                groups.get("ar25_regression_only", ())
            ),
        )
        expected_checksum = payload.get("checksum")
        if (
            expected_checksum is not None
            and str(expected_checksum) != registry.checksum
        ):
            raise ValueError("SAGE.11 split-registry checksum mismatch")
        return registry


def short_game_id(game_id: str) -> str:
    """Normalize a short or hash-qualified ARC game id."""
    return str(game_id).split("-", 1)[0]


SAGE11_SPLITS = Sage11SplitRegistry()


__all__ = [
    "AR25_REGRESSION_ONLY",
    "ArtifactPurpose",
    "HISTORICAL_BENCHMARK",
    "NEURO_HOLDOUT_V1",
    "SAGE11_SPLITS",
    "SOURCE_TRAIN",
    "SOURCE_VALIDATION",
    "Sage11SplitRegistry",
    "short_game_id",
]
