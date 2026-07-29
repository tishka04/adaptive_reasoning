"""Fail-closed shadow advisor backed by transformation prototypes."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .clustering import TransformationMatch, TransformationPrototypeMemory
from .graph import MorphoTopologicalGraph, build_mt_graph
from .model import MTModelConfig, encode_transitions, predict_graphs
from .transition import MTTransitionRecord, compile_mt_transition

SHADOW_FORMAT_VERSION = "sage12-mt-shadow-v4.16"


class SageMTMode(str, Enum):
    OFF = "off"
    SHADOW = "shadow"


@dataclass(frozen=True)
class SageMTConfig:
    mode: SageMTMode = SageMTMode.OFF
    maximum_matches: int = 8
    maximum_uncertainty: float = 1.0
    similarity_temperature: float = 0.10
    support_coefficient: float = 0.05
    uncertainty_coefficient: float = 0.10


@dataclass(frozen=True)
class MTCandidateAdvisory:
    action_name: str
    action_data: Mapping[str, Any]
    score: float
    uncertainty: float
    predicted_vector: tuple[float, ...]
    matches: tuple[TransformationMatch, ...]

    @property
    def key(self) -> str:
        return _action_key(self.action_name, self.action_data)


@dataclass(frozen=True)
class MTAdvisory:
    advisory_id: str
    suggested_action_name: str
    suggested_action_data: Mapping[str, Any]
    score: float
    candidates: tuple[MTCandidateAdvisory, ...]
    applied: bool = False
    mode: str = SageMTMode.SHADOW.value

    @classmethod
    def empty(cls, *, mode: str = SageMTMode.OFF.value) -> MTAdvisory:
        return cls(
            advisory_id="",
            suggested_action_name="",
            suggested_action_data={},
            score=0.0,
            candidates=(),
            applied=False,
            mode=mode,
        )


@dataclass(frozen=True)
class SageMTShadowRecord:
    advisory: MTAdvisory
    executed_action_name: str
    executed_action_data: Mapping[str, Any]
    observed_transition_id: str = ""
    observed_prototype_id: str = ""
    productive: bool | None = None
    risk: bool | None = None
    format_version: str = SHADOW_FORMAT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "advisory": {
                "advisory_id": self.advisory.advisory_id,
                "suggested_action_name": self.advisory.suggested_action_name,
                "suggested_action_data": dict(
                    self.advisory.suggested_action_data
                ),
                "score": self.advisory.score,
                "applied": False,
                "mode": self.advisory.mode,
                "candidates": [
                    {
                        "action_name": candidate.action_name,
                        "action_data": dict(candidate.action_data),
                        "score": candidate.score,
                        "uncertainty": candidate.uncertainty,
                        "matches": [asdict(match) for match in candidate.matches],
                    }
                    for candidate in self.advisory.candidates
                ],
            },
            "executed": {
                "action_name": self.executed_action_name,
                "action_data": dict(self.executed_action_data),
            },
            "observed": {
                "transition_id": self.observed_transition_id,
                "prototype_id": self.observed_prototype_id,
                "productive": self.productive,
                "risk": self.risk,
            },
        }


class SageMTShadowWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: SageMTShadowRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(record.to_dict(), sort_keys=True) + "\n"
            )


@dataclass(frozen=True)
class _PendingAdvice:
    record: SageMTShadowRecord


def _action_key(action_name: str, action_data: Mapping[str, Any]) -> str:
    return (
        str(action_name).strip().upper()
        + ":"
        + json.dumps(
            {
                str(key): value
                for key, value in dict(action_data or {}).items()
                if value is not None
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    )


def _normalize_candidates(candidates: Sequence[Any]) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    output: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for candidate in candidates:
        if isinstance(candidate, str):
            name, data = candidate, {}
        else:
            name = getattr(candidate, "action_name", "")
            data = dict(getattr(candidate, "action_data", {}) or {})
        if not name:
            continue
        key = _action_key(str(name), data)
        output[key] = (str(name).strip().upper(), data)
    return tuple(output[key] for key in sorted(output))


class MorphoTopologicalAnalogyAdvisor:
    """Rank legal actions by predicted transformation analogies.

    The advisor never returns an executable authority flag.  ``applied`` is
    pinned to ``False`` in both the in-memory advisory and the audit record.
    """

    def __init__(
        self,
        *,
        model: Any,
        model_config: MTModelConfig,
        memory: TransformationPrototypeMemory,
        config: SageMTConfig | None = None,
        device: str = "cpu",
        writer: SageMTShadowWriter | None = None,
    ) -> None:
        self.model = model
        self.model_config = model_config
        self.memory = memory
        self.config = config or SageMTConfig()
        self.device = str(device)
        self.writer = writer
        self._pending: _PendingAdvice | None = None
        self._evaluations = 0
        self._failures = 0
        self._unknown_candidates = 0
        self._observations = 0
        self._agreements = 0
        self._history_tokens: list[str] = []
        self._last_advisory = MTAdvisory.empty()

    @property
    def mode(self) -> SageMTMode:
        value = self.config.mode
        return value if isinstance(value, SageMTMode) else SageMTMode(str(value))

    def _predict(
        self,
        graphs: Sequence[MorphoTopologicalGraph],
    ) -> tuple[tuple[tuple[float, ...], float], ...]:
        if hasattr(self.model, "predict_graphs"):
            predictor = self.model.predict_graphs
            if "histories" in inspect.signature(predictor).parameters:
                return tuple(
                    predictor(
                        graphs,
                        histories=tuple(
                            tuple(self._history_tokens) for _ in graphs
                        ),
                    )
                )
            return tuple(predictor(graphs))
        return predict_graphs(
            self.model,
            graphs,
            config=self.model_config,
            device=self.device,
            histories=tuple(
                tuple(self._history_tokens) for _ in graphs
            ),
        )

    def _encode_observed(
        self,
        record: MTTransitionRecord,
    ) -> tuple[float, ...]:
        if hasattr(self.model, "encode_transition"):
            return tuple(self.model.encode_transition(record))
        encoded = encode_transitions(
            self.model,
            (record,),
            config=self.model_config,
            device=self.device,
        )
        return encoded[0].vector

    def advise(
        self,
        *,
        observation: Any,
        candidates: Sequence[Any],
        executed_action_name: str,
        executed_action_data: Mapping[str, Any] | None = None,
    ) -> MTAdvisory:
        if self.mode == SageMTMode.OFF:
            self._pending = None
            self._last_advisory = MTAdvisory.empty(mode=self.mode.value)
            return self._last_advisory
        normalized = _normalize_candidates(candidates)
        if not normalized:
            self._pending = None
            self._last_advisory = MTAdvisory.empty(mode=self.mode.value)
            return self._last_advisory
        try:
            player = getattr(observation, "best_player", None)
            player_position = (
                tuple(player.position) if player is not None else None
            )
            graphs = [
                build_mt_graph(
                    observation.raw_grid,
                    action_name=name,
                    action_data=data,
                    player_position=player_position,
                )
                for name, data in normalized
            ]
            predictions = self._predict(graphs)
            advisories: list[MTCandidateAdvisory] = []
            for (name, data), graph, (vector, uncertainty) in zip(
                normalized,
                graphs,
                predictions,
            ):
                matches = (
                    self.memory.retrieve(
                        vector,
                        action_family=graph.action_family,
                        uncertainty=uncertainty,
                        maximum_matches=self.config.maximum_matches,
                    )
                    if uncertainty <= self.config.maximum_uncertainty
                    else ()
                )
                if not matches:
                    self._unknown_candidates += 1
                    score = -float(uncertainty)
                else:
                    similarities = np.asarray(
                        [match.similarity for match in matches],
                        dtype=np.float64,
                    )
                    logits = similarities / max(
                        float(self.config.similarity_temperature),
                        1e-4,
                    )
                    weights = np.exp(logits - logits.max())
                    weights /= weights.sum()
                    expected_value = sum(
                        float(weight)
                        * (
                            match.productive_probability
                            - match.risk_probability
                        )
                        for weight, match in zip(weights, matches)
                    )
                    support = sum(
                        float(weight) * math.log1p(match.support)
                        for weight, match in zip(weights, matches)
                    )
                    score = (
                        expected_value
                        + self.config.support_coefficient * support
                        - self.config.uncertainty_coefficient
                        * float(uncertainty)
                    )
                advisories.append(
                    MTCandidateAdvisory(
                        action_name=name,
                        action_data=dict(data),
                        score=float(score),
                        uncertainty=float(uncertainty),
                        predicted_vector=tuple(vector),
                        matches=tuple(matches),
                    )
                )
            advisories.sort(
                key=lambda item: (item.score, item.key),
                reverse=True,
            )
            selected = advisories[0]
            digest_payload = {
                "scene": getattr(observation, "grid_hash", ""),
                "candidates": [
                    {
                        "key": item.key,
                        "score": round(item.score, 8),
                        "prototypes": [
                            match.prototype_id for match in item.matches
                        ],
                    }
                    for item in advisories
                ],
            }
            advisory = MTAdvisory(
                advisory_id=hashlib.sha256(
                    json.dumps(
                        digest_payload,
                        sort_keys=True,
                        default=str,
                    ).encode("utf-8")
                ).hexdigest(),
                suggested_action_name=selected.action_name,
                suggested_action_data=dict(selected.action_data),
                score=selected.score,
                candidates=tuple(advisories),
                applied=False,
                mode=self.mode.value,
            )
            self._evaluations += 1
            pending_record = SageMTShadowRecord(
                advisory=advisory,
                executed_action_name=str(executed_action_name).strip().upper(),
                executed_action_data=dict(executed_action_data or {}),
            )
            self._pending = _PendingAdvice(pending_record)
            self._last_advisory = advisory
            return advisory
        except Exception:  # noqa: BLE001 - advisory failures must fail closed
            self._failures += 1
            self._pending = None
            self._last_advisory = MTAdvisory.empty(mode=self.mode.value)
            return self._last_advisory

    def observe_transition(self, record: Any) -> None:
        pending = self._pending
        self._pending = None
        if pending is None:
            return
        try:
            action = getattr(record.action, "name", str(record.action))
            action_data = {
                key: value
                for key, value in {
                    "x": getattr(record.action, "x", None),
                    "y": getattr(record.action, "y", None),
                }.items()
                if value is not None
            }
            transition = compile_mt_transition(
                record.obs_before.raw_grid,
                action,
                record.obs_after.raw_grid,
                action_data=action_data,
                productive=bool(
                    not record.diff.is_noop or record.diff.level_complete
                ),
                risk=bool(record.diff.game_over),
            )
            vector = self._encode_observed(transition)
            prototype_id = self.memory.assign(
                vector,
                action_family=transition.graph_before.action_family,
            )
            self.memory.observe(
                prototype_id,
                productive=bool(transition.productive),
                risk=bool(transition.risk),
            )
            completed = SageMTShadowRecord(
                advisory=pending.record.advisory,
                executed_action_name=str(action).strip().upper(),
                executed_action_data=action_data,
                observed_transition_id=transition.transition_id,
                observed_prototype_id=prototype_id,
                productive=transition.productive,
                risk=transition.risk,
            )
            self._observations += 1
            self._history_tokens.append(transition.delta_signature)
            self._history_tokens = self._history_tokens[
                -self.model_config.history_width :
            ]
            self._agreements += int(
                _action_key(
                    pending.record.advisory.suggested_action_name,
                    pending.record.advisory.suggested_action_data,
                )
                == _action_key(action, action_data)
            )
            if self.writer is not None:
                self.writer.append(completed)
        except Exception:  # noqa: BLE001 - observed core updates retain priority
            self._failures += 1

    def start_branch(self) -> None:
        self._pending = None
        self._history_tokens.clear()

    def summary(self) -> Mapping[str, Any]:
        return {
            "mode": self.mode.value,
            "evaluations": self._evaluations,
            "failures": self._failures,
            "unknown_candidates": self._unknown_candidates,
            "observations": self._observations,
            "agreements": self._agreements,
            "history_length": len(self._history_tokens),
            "last_advisory_id": self._last_advisory.advisory_id,
            "last_suggested_action": self._last_advisory.suggested_action_name,
            "memory": self.memory.snapshot(),
        }


__all__ = [
    "SHADOW_FORMAT_VERSION",
    "MTAdvisory",
    "MTCandidateAdvisory",
    "MorphoTopologicalAnalogyAdvisor",
    "SageMTConfig",
    "SageMTMode",
    "SageMTShadowRecord",
    "SageMTShadowWriter",
]
