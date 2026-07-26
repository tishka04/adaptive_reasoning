"""Staged off/shadow/bounded/active neural authority with symbolic supremacy."""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Mapping, Protocol, Sequence, Tuple

from .atoms import TypedAtom


class NeuralAuthorityMode(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    BOUNDED = "bounded"
    ACTIVE = "active"


@dataclass(frozen=True)
class NeuralAuthorityConfig:
    mode: NeuralAuthorityMode = NeuralAuthorityMode.OFF
    bounded_gate_passed: bool = False
    active_gate_passed: bool = False
    maximum_advisory_risk: float = 0.10
    minimum_information_gain: float = 0.0
    nonproductive_demotion_threshold: int = 2
    max_inference_ms: float = 10.0
    top_k_productivity: int = 3


@dataclass(frozen=True)
class NeuralActionCandidate:
    action_name: str
    action_data: Mapping[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return _action_key(self.action_name, self.action_data)


@dataclass(frozen=True)
class NeuralActionPrediction:
    """Calibrated counterfactual with candidate-only symbolic atoms."""

    action_name: str
    action_data: Mapping[str, Any] = field(default_factory=dict)
    predicted_progress: float = 0.0
    predicted_effect: float = 0.0
    predicted_information_gain: float = 0.0
    predicted_risk: float = 0.0
    predicted_noop: float = 1.0
    epistemic_variance: float = 0.0
    hypotheses: Tuple[TypedAtom, ...] = ()

    def __post_init__(self) -> None:
        for hypothesis in self.hypotheses:
            if hypothesis.support != 0:
                raise ValueError(
                    "neural hypotheses must enter the bridge with support=0"
                )
        for name in (
            "predicted_progress",
            "predicted_effect",
            "predicted_information_gain",
            "predicted_risk",
            "predicted_noop",
            "epistemic_variance",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} may not be negative")

    @property
    def key(self) -> str:
        return _action_key(self.action_name, self.action_data)

    @property
    def score(self) -> float:
        return (
            2.0 * float(self.predicted_progress)
            + float(self.predicted_effect)
            + float(self.predicted_information_gain)
            + 0.5 * float(self.epistemic_variance)
            - 3.0 * float(self.predicted_risk)
            - float(self.predicted_noop)
        )


class NeuralPredictor(Protocol):
    def __call__(
        self,
        observation: Any,
        candidates: Sequence[NeuralActionCandidate],
    ) -> Sequence[NeuralActionPrediction]:
        ...


@dataclass(frozen=True)
class NeuralArbitration:
    action_name: str
    action_data: Mapping[str, Any]
    source: str
    reason: str
    confidence: float
    applied: bool
    configured_mode: str
    effective_mode: str
    counterfactual_top_key: str = ""


@dataclass(frozen=True)
class _PendingRanking:
    symbolic_key: str
    ranked_keys: Tuple[str, ...]
    applied_key: str
    context_signature: str
    protected_competence: bool
    inference_ms: float


class NeuroSymbolicRanker:
    """Counterfactual ranker whose authority is earned by staged gates."""

    def __init__(
        self,
        predictor: NeuralPredictor | None = None,
        *,
        config: NeuralAuthorityConfig | None = None,
    ) -> None:
        self.predictor = predictor or _zero_predictor
        self.config = config or NeuralAuthorityConfig()
        self._branch_index = 0
        self._probed_contexts: set[str] = set()
        self._context_nonproductive: Counter[str] = Counter()
        self._demoted_contexts: set[str] = set()
        self._pending: _PendingRanking | None = None
        self._evaluations = 0
        self._shadow_evaluations = 0
        self._bounded_probes = 0
        self._active_selections = 0
        self._symbolic_danger_vetoes = 0
        self._protected_competence_blocks = 0
        self._advisory_risk_blocks = 0
        self._information_gain_blocks = 0
        self._demotions = 0
        self._rearms = 0
        self._inference_budget_exceeded = 0
        self._action_identity_checks = 0
        self._action_identity_mismatches = 0
        self._productive_executions = 0
        self._productive_top1 = 0
        self._productive_top3 = 0
        self._would_be_successful_route_preemptions = 0
        self._unsafe_neural_outcomes = 0
        self._inference_total_ms = 0.0
        self._inference_peak_ms = 0.0
        self._logs: list[Dict[str, Any]] = []

    @property
    def configured_mode(self) -> NeuralAuthorityMode:
        mode = self.config.mode
        if isinstance(mode, NeuralAuthorityMode):
            return mode
        return NeuralAuthorityMode(str(mode))

    @property
    def effective_mode(self) -> NeuralAuthorityMode:
        mode = self.configured_mode
        if mode == NeuralAuthorityMode.ACTIVE:
            if self.config.active_gate_passed:
                return mode
            if self.config.bounded_gate_passed:
                return NeuralAuthorityMode.BOUNDED
            return NeuralAuthorityMode.SHADOW
        if (
            mode == NeuralAuthorityMode.BOUNDED
            and not self.config.bounded_gate_passed
        ):
            return NeuralAuthorityMode.SHADOW
        return mode

    def arbitrate(
        self,
        *,
        symbolic_action_name: str,
        symbolic_action_data: Mapping[str, Any] | None,
        symbolic_source: str,
        observation: Any,
        candidates: Sequence[NeuralActionCandidate],
        protected_competence_available: bool,
        context_signature: str,
        danger_veto: Callable[[str, Mapping[str, Any]], bool],
    ) -> NeuralArbitration:
        """Rank alternatives and optionally grant one bounded neural probe."""
        configured = self.configured_mode
        effective = self.effective_mode
        symbolic_data = dict(symbolic_action_data or {})
        unchanged = NeuralArbitration(
            action_name=str(symbolic_action_name),
            action_data=symbolic_data,
            source=str(symbolic_source),
            reason="neural authority disabled",
            confidence=0.0,
            applied=False,
            configured_mode=configured.value,
            effective_mode=effective.value,
        )
        if configured == NeuralAuthorityMode.OFF:
            return unchanged
        candidate_list = _deduplicate_candidates(
            candidates,
            include=NeuralActionCandidate(
                action_name=str(symbolic_action_name),
                action_data=symbolic_data,
            ),
        )
        started = time.perf_counter()
        predictions = tuple(self.predictor(observation, candidate_list))
        inference_ms = (time.perf_counter() - started) * 1000.0
        self._evaluations += 1
        self._inference_total_ms += inference_ms
        self._inference_peak_ms = max(self._inference_peak_ms, inference_ms)
        if inference_ms > self.config.max_inference_ms:
            self._inference_budget_exceeded += 1
        prediction_by_key = {prediction.key: prediction for prediction in predictions}
        admissible = []
        for candidate in candidate_list:
            prediction = prediction_by_key.get(candidate.key)
            if prediction is None:
                continue
            if danger_veto(candidate.action_name, candidate.action_data):
                self._symbolic_danger_vetoes += 1
                continue
            admissible.append(prediction)
        admissible.sort(
            key=lambda item: (item.score, item.key),
            reverse=True,
        )
        symbolic_key = _action_key(symbolic_action_name, symbolic_data)
        ranked_keys = tuple(item.key for item in admissible)
        top = admissible[0] if admissible else None
        self._shadow_evaluations += int(
            effective == NeuralAuthorityMode.SHADOW
        )
        selected = None
        block_reason = ""
        if top is None:
            block_reason = "no symbolically admissible neural candidate"
        elif protected_competence_available:
            self._protected_competence_blocks += 1
            block_reason = "protected symbolic competence retains authority"
        elif context_signature in self._demoted_contexts:
            block_reason = "neural context demoted after non-productivity"
        elif (
            top.predicted_information_gain
            <= self.config.minimum_information_gain
        ):
            self._information_gain_blocks += 1
            block_reason = "predicted information gain is not positive"
        elif top.predicted_risk > self.config.maximum_advisory_risk:
            self._advisory_risk_blocks += 1
            block_reason = "advisory neural risk exceeds bounded threshold"
        elif inference_ms > self.config.max_inference_ms:
            block_reason = "neural inference cost exceeds budget"
        elif effective == NeuralAuthorityMode.SHADOW:
            block_reason = "shadow mode records ranking without authority"
        elif context_signature in self._probed_contexts:
            block_reason = "one neural probe already spent in branch/context"
        elif top.key == symbolic_key:
            block_reason = "neural and symbolic rankings already agree"
        else:
            selected = top
            self._probed_contexts.add(context_signature)
            if effective == NeuralAuthorityMode.BOUNDED:
                self._bounded_probes += 1
            else:
                self._active_selections += 1
        applied_key = selected.key if selected is not None else symbolic_key
        self._pending = _PendingRanking(
            symbolic_key=symbolic_key,
            ranked_keys=ranked_keys,
            applied_key=applied_key,
            context_signature=str(context_signature),
            protected_competence=bool(protected_competence_available),
            inference_ms=inference_ms,
        )
        self._action_identity_checks += int(
            effective == NeuralAuthorityMode.SHADOW
        )
        if effective == NeuralAuthorityMode.SHADOW and applied_key != symbolic_key:
            self._action_identity_mismatches += 1
        self._logs.append({
            "branch": self._branch_index,
            "configured_mode": configured.value,
            "effective_mode": effective.value,
            "symbolic_source": str(symbolic_source),
            "symbolic_key": symbolic_key,
            "ranked_keys": list(ranked_keys),
            "applied_key": applied_key,
            "protected_competence": bool(protected_competence_available),
            "inference_ms": inference_ms,
            "block_reason": block_reason,
        })
        if selected is None:
            return NeuralArbitration(
                **{
                    **asdict(unchanged),
                    "reason": block_reason,
                    "counterfactual_top_key": (
                        "" if top is None else top.key
                    ),
                }
            )
        return NeuralArbitration(
            action_name=selected.action_name,
            action_data=dict(selected.action_data),
            source=(
                "neural_bounded_probe"
                if effective == NeuralAuthorityMode.BOUNDED
                else "neural_active_probe"
            ),
            reason=(
                "symbolically admissible neural probe with positive "
                "information gain; support remains zero until observed"
            ),
            confidence=max(0.0, min(1.0, selected.score)),
            applied=True,
            configured_mode=configured.value,
            effective_mode=effective.value,
            counterfactual_top_key=selected.key,
        )

    def observe_outcome(
        self,
        *,
        productive: bool,
        unsafe: bool,
        successful_route: bool,
    ) -> None:
        """Attach actual productivity only after the environment transition."""
        pending = self._pending
        self._pending = None
        if pending is None:
            return
        if productive:
            self._productive_executions += 1
            if pending.symbolic_key in pending.ranked_keys[:1]:
                self._productive_top1 += 1
            if pending.symbolic_key in pending.ranked_keys[
                : self.config.top_k_productivity
            ]:
                self._productive_top3 += 1
        if (
            successful_route
            and pending.ranked_keys
            and pending.ranked_keys[0] != pending.symbolic_key
        ):
            self._would_be_successful_route_preemptions += 1
        if pending.applied_key != pending.symbolic_key:
            if unsafe:
                self._unsafe_neural_outcomes += 1
            if productive and not unsafe:
                self._context_nonproductive[pending.context_signature] = 0
            else:
                self._context_nonproductive[pending.context_signature] += 1
                if (
                    self._context_nonproductive[pending.context_signature]
                    >= self.config.nonproductive_demotion_threshold
                    and pending.context_signature not in self._demoted_contexts
                ):
                    self._demoted_contexts.add(pending.context_signature)
                    self._demotions += 1

    def start_branch(self) -> None:
        self._branch_index += 1
        self._probed_contexts.clear()
        self._pending = None

    def rearm(self, *, reason: str) -> int:
        if str(reason) not in {
            "context_change",
            "new_effect",
            "route_refutation",
            "level_change",
        }:
            raise ValueError("invalid SAGE.11 neural re-arm reason")
        count = len(self._demoted_contexts)
        self._demoted_contexts.clear()
        self._context_nonproductive.clear()
        self._rearms += count
        return count

    def summary(self) -> Dict[str, Any]:
        productive = max(1, self._productive_executions)
        return {
            "configured_mode": self.configured_mode.value,
            "effective_mode": self.effective_mode.value,
            "evaluations": self._evaluations,
            "shadow_evaluations": self._shadow_evaluations,
            "bounded_probes": self._bounded_probes,
            "active_selections": self._active_selections,
            "symbolic_danger_vetoes": self._symbolic_danger_vetoes,
            "symbolic_danger_memory_is_hard_veto": True,
            "protected_competence_blocks": (
                self._protected_competence_blocks
            ),
            "advisory_risk_blocks": self._advisory_risk_blocks,
            "information_gain_blocks": self._information_gain_blocks,
            "demotions": self._demotions,
            "rearms": self._rearms,
            "action_identity_checks": self._action_identity_checks,
            "action_identity_mismatches": self._action_identity_mismatches,
            "productive_executions": self._productive_executions,
            "productive_top1_rate": self._productive_top1 / productive,
            "productive_top3_rate": self._productive_top3 / productive,
            "would_be_successful_route_preemptions": (
                self._would_be_successful_route_preemptions
            ),
            "unsafe_neural_outcomes": self._unsafe_neural_outcomes,
            "inference_mean_ms": (
                self._inference_total_ms / max(1, self._evaluations)
            ),
            "inference_peak_ms": self._inference_peak_ms,
            "inference_budget_exceeded": self._inference_budget_exceeded,
            "candidate_hypothesis_initial_support": 0,
            "log_records": len(self._logs),
        }

    def logs(self) -> Tuple[Mapping[str, Any], ...]:
        return tuple(dict(item) for item in self._logs)


def _zero_predictor(
    _observation: Any,
    candidates: Sequence[NeuralActionCandidate],
) -> Sequence[NeuralActionPrediction]:
    return tuple(
        NeuralActionPrediction(
            action_name=candidate.action_name,
            action_data=dict(candidate.action_data),
        )
        for candidate in candidates
    )


def _deduplicate_candidates(
    candidates: Sequence[NeuralActionCandidate],
    *,
    include: NeuralActionCandidate,
) -> Tuple[NeuralActionCandidate, ...]:
    result: Dict[str, NeuralActionCandidate] = {include.key: include}
    for candidate in candidates:
        result.setdefault(candidate.key, candidate)
    return tuple(result.values())


def _action_key(
    action_name: str,
    action_data: Mapping[str, Any] | None,
) -> str:
    return json.dumps(
        [str(action_name), dict(action_data or {})],
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


__all__ = [
    "NeuralActionCandidate",
    "NeuralActionPrediction",
    "NeuralArbitration",
    "NeuralAuthorityConfig",
    "NeuralAuthorityMode",
    "NeuroSymbolicRanker",
]
