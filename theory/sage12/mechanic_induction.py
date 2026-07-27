"""Sequence-conditioned mechanic induction for the SAGE12 V4 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier

from theory.sage11.splits import SOURCE_TRAIN, SOURCE_VALIDATION

from .action_target_data import (
    EFFECT_LABELS,
    ActionTargetTrace,
    build_observation,
    conservative_match_objects,
    grid_sha256,
)
from .hypotheses import (
    EntityRef,
    SemanticEffect,
    SemanticHypothesis,
    SemanticPredicate,
)
from .llm import TransformersJSONModel, TransformersModelConfig


FORMAT_VERSION = "sage12-mechanic-window-v4"
PREFLIGHT_FORMAT_VERSION = "sage12-mechanic-preflight-v4"
RESULT_FORMAT_VERSION = "sage12-mechanic-pilot-result-v4"
DEFAULT_OUTPUT_DIR = Path("training") / "sage12" / "mechanic_induction_v4"
DEFAULT_FROZEN_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "frozen_manifest.json"
V3_OUTPUT_DIR = Path("training") / "sage12" / "action_target_pilot_v3"

ANCHOR_CONDITIONS = (
    "any",
    "open",
    "occupied_actor",
    "occupied_object",
    "empty",
    "targetless",
    "unknown",
)
ACTION_SCOPE_KINDS = ("exact", "family")
_MOVE_VECTORS = {
    "ACTION1": (-1, 0),
    "ACTION2": (1, 0),
    "ACTION3": (0, -1),
    "ACTION4": (0, 1),
}


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def _checksum(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SemanticTransitionEvent:
    """One model-facing observed transition with no raw identity fields."""

    action_name: str
    action_family: str
    anchor_condition: str
    effects: Mapping[str, bool]
    applicable: Mapping[str, bool]
    actor_role_known: bool

    def __post_init__(self) -> None:
        if self.anchor_condition not in ANCHOR_CONDITIONS:
            raise ValueError("unsupported V4 anchor condition")
        if set(self.effects) != set(EFFECT_LABELS):
            raise ValueError("V4 event requires the complete effect vocabulary")
        if set(self.applicable) != set(EFFECT_LABELS):
            raise ValueError("V4 event requires complete applicability masks")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_name": self.action_name,
            "action_family": self.action_family,
            "anchor_condition": self.anchor_condition,
            "effects": {
                label: bool(self.effects[label]) for label in EFFECT_LABELS
            },
            "applicable": {
                label: bool(self.applicable[label]) for label in EFFECT_LABELS
            },
            "actor_role_known": bool(self.actor_role_known),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SemanticTransitionEvent":
        effects = dict(payload.get("effects", {}))
        applicable = dict(payload.get("applicable", {}))
        return cls(
            action_name=str(payload["action_name"]),
            action_family=str(payload["action_family"]),
            anchor_condition=str(payload["anchor_condition"]),
            effects={label: bool(effects.get(label, False)) for label in EFFECT_LABELS},
            applicable={
                label: bool(applicable.get(label, False)) for label in EFFECT_LABELS
            },
            actor_role_known=bool(payload.get("actor_role_known", False)),
        )


@dataclass(frozen=True)
class MechanicQuery:
    action_name: str
    action_family: str
    anchor_condition: str

    def __post_init__(self) -> None:
        if self.anchor_condition not in ANCHOR_CONDITIONS:
            raise ValueError("unsupported V4 query anchor condition")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MechanicQuery":
        return cls(
            action_name=str(payload["action_name"]),
            action_family=str(payload["action_family"]),
            anchor_condition=str(payload["anchor_condition"]),
        )


@dataclass(frozen=True)
class MechanicRule:
    """A bounded proposal. Observed evidence is deliberately stored elsewhere."""

    rule_id: str
    action_scope_kind: str
    action_scope_value: str
    anchor_condition: str
    effect: str
    support: int = 0
    source: str = "structured"

    def __post_init__(self) -> None:
        if self.action_scope_kind not in ACTION_SCOPE_KINDS:
            raise ValueError("unsupported mechanic action scope")
        if self.anchor_condition not in ANCHOR_CONDITIONS:
            raise ValueError("unsupported mechanic anchor condition")
        if self.effect not in EFFECT_LABELS:
            raise ValueError("unsupported mechanic effect")
        if self.support != 0:
            raise ValueError("mechanic proposals must enter with support=0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MechanicRule":
        return cls(
            rule_id=str(payload["rule_id"]),
            action_scope_kind=str(payload["action_scope_kind"]),
            action_scope_value=str(payload["action_scope_value"]),
            anchor_condition=str(payload["anchor_condition"]),
            effect=str(payload["effect"]),
            support=int(payload.get("support", 0)),
            source=str(payload.get("source", "local_llm"))[:32],
        )

    def matches_event(self, event: SemanticTransitionEvent) -> bool:
        action = (
            event.action_name
            if self.action_scope_kind == "exact"
            else event.action_family
        )
        return action == self.action_scope_value and (
            self.anchor_condition == "any"
            or event.anchor_condition == self.anchor_condition
        )

    def matches_query(self, query: MechanicQuery) -> bool:
        action = (
            query.action_name
            if self.action_scope_kind == "exact"
            else query.action_family
        )
        return action == self.action_scope_value and (
            self.anchor_condition == "any"
            or query.anchor_condition == self.anchor_condition
        )


@dataclass(frozen=True)
class MechanicEvidence:
    rule: MechanicRule
    observed_support: int
    observed_refutations: int
    prior_probability: float
    posterior_probability: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule.to_dict(),
            "observed_support": self.observed_support,
            "observed_refutations": self.observed_refutations,
            "prior_probability": self.prior_probability,
            "posterior_probability": self.posterior_probability,
        }


@dataclass(frozen=True)
class MechanicWindowRecord:
    """Eight observed events and one outcome-blind query."""

    game_id: str
    source_split: str
    policy_seed: int
    reset_index: int
    query_step_index: int
    context: tuple[SemanticTransitionEvent, ...]
    query: MechanicQuery
    labels: Mapping[str, bool]
    applicable: Mapping[str, bool]
    actor_role_known: bool
    window_digest: str = ""
    format_version: str = FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise ValueError("unsupported SAGE12 V4 window format")
        if set(self.labels) != set(EFFECT_LABELS):
            raise ValueError("V4 window requires complete labels")
        if set(self.applicable) != set(EFFECT_LABELS):
            raise ValueError("V4 window requires complete masks")
        if not self.window_digest:
            payload = {
                "context": [item.to_dict() for item in self.context],
                "query": self.query.to_dict(),
            }
            object.__setattr__(self, "window_digest", _checksum(payload))

    @property
    def run_key(self) -> str:
        return f"{self.game_id}:{self.policy_seed}:{self.reset_index}"

    def model_view(self) -> dict[str, Any]:
        return {
            "context": [item.to_dict() for item in self.context],
            "query": self.query.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "game_id": self.game_id,
            "source_split": self.source_split,
            "policy_seed": self.policy_seed,
            "reset_index": self.reset_index,
            "query_step_index": self.query_step_index,
            "context": [item.to_dict() for item in self.context],
            "query": self.query.to_dict(),
            "labels": {label: bool(self.labels[label]) for label in EFFECT_LABELS},
            "applicable": {
                label: bool(self.applicable[label]) for label in EFFECT_LABELS
            },
            "actor_role_known": bool(self.actor_role_known),
            "window_digest": self.window_digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MechanicWindowRecord":
        labels = dict(payload.get("labels", {}))
        applicable = dict(payload.get("applicable", {}))
        return cls(
            format_version=str(payload.get("format_version", FORMAT_VERSION)),
            game_id=str(payload["game_id"]),
            source_split=str(payload["source_split"]),
            policy_seed=int(payload["policy_seed"]),
            reset_index=int(payload["reset_index"]),
            query_step_index=int(payload["query_step_index"]),
            context=tuple(
                SemanticTransitionEvent.from_dict(item)
                for item in payload.get("context", ())
            ),
            query=MechanicQuery.from_dict(dict(payload["query"])),
            labels={label: bool(labels.get(label, False)) for label in EFFECT_LABELS},
            applicable={
                label: bool(applicable.get(label, False)) for label in EFFECT_LABELS
            },
            actor_role_known=bool(payload.get("actor_role_known", False)),
            window_digest=str(payload.get("window_digest", "")),
        )


class PersistentRoleTracker:
    """Conservative reset-local actor evidence over consecutive raw frames."""

    def __init__(self) -> None:
        self._actor_signature: tuple[int, str, str] | None = None

    def reset(self) -> None:
        self._actor_signature = None

    def observe(self, trace: ActionTargetTrace) -> SemanticTransitionEvent:
        effects = {
            label: bool(trace.effects.labels[label]) for label in EFFECT_LABELS
        }
        applicable = {
            label: bool(trace.effects.applicable[label]) for label in EFFECT_LABELS
        }
        inferred = self._infer_actor(trace)
        if inferred is not None:
            applicable["actor_displaced"] = True
            effects["actor_displaced"] = inferred
        actor_known = bool(applicable["actor_displaced"])
        return SemanticTransitionEvent(
            action_name=trace.selected_action_name,
            action_family=trace.anchor.action_family,
            anchor_condition=self._anchor_condition(trace),
            effects=effects,
            applicable=applicable,
            actor_role_known=actor_known,
        )

    def _infer_actor(self, trace: ActionTargetTrace) -> bool | None:
        before = build_observation(
            trace.frame_before,
            available_actions=trace.available_action_names,
            game_state=trace.game_state_before,
            levels_completed=trace.levels_completed_before,
            infer_players=True,
        )
        after = build_observation(
            trace.frame_after,
            available_actions=trace.available_action_names,
            game_state=trace.game_state_after,
            levels_completed=trace.levels_completed_after,
            infer_players=True,
        )
        matched = conservative_match_objects(before.objects, after.objects)
        before_by_id = {item.object_id: item for item in before.objects}
        after_by_id = {item.object_id: item for item in after.objects}
        candidates = []
        requested = _MOVE_VECTORS.get(trace.selected_action_name)
        for left_id, right_id in matched.matched.items():
            left = before_by_id[left_id]
            right = after_by_id[right_id]
            signature = _object_signature(left)
            delta = (
                int(round(right.center[0] - left.center[0])),
                int(round(right.center[1] - left.center[1])),
            )
            if self._actor_signature is not None and signature == self._actor_signature:
                candidates.append((signature, delta))
            elif requested is not None and delta == requested:
                candidates.append((signature, delta))
        if len(candidates) != 1:
            return None
        self._actor_signature = candidates[0][0]
        return candidates[0][1] != (0, 0)

    def _anchor_condition(self, trace: ActionTargetTrace) -> str:
        anchor = trace.anchor
        if anchor.kind == "targetless":
            return "targetless"
        if anchor.kind == "clicked_empty":
            return "empty"
        if anchor.path_status == "open":
            return "open"
        if not anchor.occupied:
            return "empty"
        if anchor.target_object_id is None:
            return "unknown"
        before = build_observation(
            trace.frame_before,
            available_actions=trace.available_action_names,
            game_state=trace.game_state_before,
            levels_completed=trace.levels_completed_before,
            infer_players=True,
        )
        target = next(
            (
                item
                for item in before.objects
                if item.object_id == anchor.target_object_id
            ),
            None,
        )
        if (
            target is not None
            and self._actor_signature is not None
            and _object_signature(target) == self._actor_signature
        ):
            return "occupied_actor"
        return "occupied_object"


def _object_signature(item: Any) -> tuple[int, str, str]:
    r0, c0, r1, c1 = item.bbox
    height = max(1, r1 - r0 + 1)
    width = max(1, c1 - c0 + 1)
    if item.area <= 1:
        area = "one"
    elif item.area <= 4:
        area = "small"
    elif item.area <= 16:
        area = "medium"
    else:
        area = "large"
    ratio = width / height
    aspect = "wide" if ratio >= 1.5 else "tall" if ratio <= 2 / 3 else "square"
    return int(item.value), area, aspect


def build_mechanic_windows(
    traces: Sequence[ActionTargetTrace],
    *,
    context_length: int = 8,
) -> list[MechanicWindowRecord]:
    """Build unique sliding windows without crossing a reset or frame gap."""
    grouped: dict[tuple[str, int, int], list[ActionTargetTrace]] = defaultdict(list)
    for trace in traces:
        grouped[(trace.game_id, trace.policy_seed, trace.reset_index)].append(trace)
    windows = []
    seen: set[str] = set()
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda item: item.step_index)
        runs: list[list[ActionTargetTrace]] = []
        current: list[ActionTargetTrace] = []
        for row in rows:
            contiguous = bool(current) and (
                row.step_index == current[-1].step_index + 1
                and grid_sha256(current[-1].frame_after)
                == grid_sha256(row.frame_before)
            )
            if current and not contiguous:
                runs.append(current)
                current = []
            current.append(row)
        if current:
            runs.append(current)
        for run in runs:
            tracker = PersistentRoleTracker()
            events = [tracker.observe(row) for row in run]
            for query_index in range(int(context_length), len(run)):
                row = run[query_index]
                event = events[query_index]
                window = MechanicWindowRecord(
                    game_id=row.game_id,
                    source_split=row.source_split,
                    policy_seed=row.policy_seed,
                    reset_index=row.reset_index,
                    query_step_index=row.step_index,
                    context=tuple(events[query_index - context_length : query_index]),
                    query=MechanicQuery(
                        action_name=event.action_name,
                        action_family=event.action_family,
                        anchor_condition=event.anchor_condition,
                    ),
                    labels=event.effects,
                    applicable=event.applicable,
                    actor_role_known=event.actor_role_known,
                    window_digest=_checksum(
                        {
                            "trace_digests": [
                                item.trace_digest
                                for item in run[
                                    query_index - context_length : query_index + 1
                                ]
                            ]
                        }
                    ),
                )
                if window.window_digest not in seen:
                    seen.add(window.window_digest)
                    windows.append(window)
    return windows


def validate_model_view(window: MechanicWindowRecord) -> None:
    rendered = _canonical(window.model_view()).lower()
    forbidden = (
        window.game_id.lower(),
        "game_id",
        "policy_seed",
        "reset_index",
        "query_step_index",
        "frame_before",
        "frame_after",
        "grid",
        "trace_digest",
        "\"x\"",
        "\"y\"",
        "\"row\"",
        "\"col\"",
        "\"value\"",
        "\"color\"",
    )
    if any(token and token in rendered for token in forbidden):
        raise ValueError("V4 model view contains forbidden provenance")


def _rule_key(
    kind: str,
    value: str,
    anchor: str,
    effect: str,
) -> str:
    return "|".join((kind, value, anchor, effect))


def _rule_id(kind: str, value: str, anchor: str, effect: str) -> str:
    digest = hashlib.sha256(
        _rule_key(kind, value, anchor, effect).encode("utf-8")
    ).hexdigest()[:12]
    return f"mechanic_{digest}"


def _rules_for_query(query: MechanicQuery, effect: str) -> tuple[MechanicRule, ...]:
    specs = (
        ("exact", query.action_name, query.anchor_condition),
        ("family", query.action_family, query.anchor_condition),
        ("exact", query.action_name, "any"),
        ("family", query.action_family, "any"),
    )
    return tuple(
        MechanicRule(
            rule_id=_rule_id(kind, value, anchor, effect),
            action_scope_kind=kind,
            action_scope_value=value,
            anchor_condition=anchor,
            effect=effect,
        )
        for kind, value, anchor in specs
    )


def fit_source_priors(
    windows: Sequence[MechanicWindowRecord],
) -> dict[str, dict[str, int]]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for window in windows:
        for effect in EFFECT_LABELS:
            if not window.applicable[effect]:
                continue
            for rule in _rules_for_query(window.query, effect):
                pair = counts[
                    _rule_key(
                        rule.action_scope_kind,
                        rule.action_scope_value,
                        rule.anchor_condition,
                        effect,
                    )
                ]
                pair[1] += 1
                pair[0] += int(window.labels[effect])
    return {
        key: {"positive": value[0], "applicable": value[1]}
        for key, value in sorted(counts.items())
    }


def _prior_probability(
    priors: Mapping[str, Mapping[str, int]],
    rule: MechanicRule,
) -> float:
    item = priors.get(
        _rule_key(
            rule.action_scope_kind,
            rule.action_scope_value,
            rule.anchor_condition,
            rule.effect,
        ),
        {},
    )
    positive = int(item.get("positive", 0))
    applicable = int(item.get("applicable", 0))
    return (positive + 1.0) / (applicable + 2.0)


def score_rule(
    rule: MechanicRule,
    context: Sequence[SemanticTransitionEvent],
    priors: Mapping[str, Mapping[str, int]],
    *,
    prior_strength: float = 2.0,
) -> MechanicEvidence:
    eligible = [
        event
        for event in context
        if rule.matches_event(event) and event.applicable[rule.effect]
    ]
    support = sum(int(event.effects[rule.effect]) for event in eligible)
    refutations = len(eligible) - support
    prior = _prior_probability(priors, rule)
    posterior = (prior_strength * prior + support) / (
        prior_strength + len(eligible)
    )
    return MechanicEvidence(
        rule=rule,
        observed_support=support,
        observed_refutations=refutations,
        prior_probability=prior,
        posterior_probability=posterior,
    )


def predict_mechanic_effects(
    context: Sequence[SemanticTransitionEvent],
    query: MechanicQuery,
    priors: Mapping[str, Mapping[str, int]],
) -> tuple[dict[str, float], tuple[MechanicEvidence, ...]]:
    probabilities = {}
    selected = []
    for effect in EFFECT_LABELS:
        candidates = [
            score_rule(rule, context, priors)
            for rule in _rules_for_query(query, effect)
        ]
        evidence = next(
            (
                item
                for item in candidates
                if item.observed_support + item.observed_refutations >= 2
            ),
            candidates[-1],
        )
        probabilities[effect] = evidence.posterior_probability
        selected.append(evidence)
    selected.sort(
        key=lambda item: (
            item.observed_support + item.observed_refutations,
            abs(item.posterior_probability - 0.5),
            item.rule.rule_id,
        ),
        reverse=True,
    )
    return probabilities, tuple(selected[:8])


def rule_to_semantic_hypothesis(
    rule: MechanicRule,
    query: MechanicQuery,
    *,
    confidence: float,
) -> SemanticHypothesis:
    subject = (
        EntityRef("actor")
        if rule.effect == "actor_displaced"
        else EntityRef("target")
    )
    if rule.effect in {"actor_displaced", "target_moved"}:
        predicate = SemanticPredicate(name="moved", subject=subject)
        operation = "assert"
    elif rule.effect == "target_created":
        predicate = SemanticPredicate(name="exists")
        operation = "assert"
    else:
        predicate = SemanticPredicate(name="exists", subject=subject)
        operation = "retract"
    return SemanticHypothesis(
        hypothesis_id=rule.rule_id,
        action_name=query.action_name,
        effects=(SemanticEffect(predicate=predicate, operation=operation),),
        confidence=float(min(1.0, max(0.0, confidence))),
        source=rule.source,
        support=0,
    )


def _action_only_probability(
    window: MechanicWindowRecord,
    effect: str,
    priors: Mapping[str, Mapping[str, int]],
    *,
    use_context: bool,
) -> float:
    rule = MechanicRule(
        rule_id=_rule_id("exact", window.query.action_name, "any", effect),
        action_scope_kind="exact",
        action_scope_value=window.query.action_name,
        anchor_condition="any",
        effect=effect,
        source="action_only",
    )
    context = window.context if use_context else ()
    return score_rule(rule, context, priors).posterior_probability


def _template_probabilities(window: MechanicWindowRecord) -> dict[str, float]:
    return {
        "actor_displaced": float(
            window.query.action_family == "move"
            and window.query.anchor_condition == "open"
        ),
        "target_created": float(window.query.anchor_condition == "empty"),
        "target_removed": float(
            window.query.action_family == "click"
            and window.query.anchor_condition == "occupied_object"
        ),
        "target_moved": float(
            window.query.action_family == "move"
            and window.query.anchor_condition == "occupied_object"
        ),
    }


def _arrays(
    windows: Sequence[MechanicWindowRecord],
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(
        [[int(window.labels[label]) for label in EFFECT_LABELS] for window in windows],
        dtype=np.int8,
    )
    masks = np.asarray(
        [
            [int(window.applicable[label]) for label in EFFECT_LABELS]
            for window in windows
        ],
        dtype=np.int8,
    )
    return targets, masks


def _ece(targets: np.ndarray, probabilities: np.ndarray) -> float:
    if not len(targets):
        return 0.0
    values = []
    bins = np.linspace(0.0, 1.0, 11)
    for lower, upper in zip(bins[:-1], bins[1:]):
        selected = (probabilities >= lower) & (
            probabilities <= upper if upper == 1.0 else probabilities < upper
        )
        if np.any(selected):
            values.append(
                np.mean(selected)
                * abs(
                    float(np.mean(targets[selected]))
                    - float(np.mean(probabilities[selected]))
                )
            )
    return float(sum(values))


def multilabel_metrics(
    targets: np.ndarray,
    masks: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    per_label = {}
    for index, label in enumerate(EFFECT_LABELS):
        selected = masks[:, index].astype(bool)
        y_true = targets[selected, index]
        y_prob = probabilities[selected, index]
        y_pred = y_prob >= 0.5
        per_label[label] = {
            "applicable": int(len(y_true)),
            "positives": int(np.sum(y_true)),
            "negatives": int(len(y_true) - np.sum(y_true)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "brier": float(np.mean((y_prob - y_true) ** 2)) if len(y_true) else 0.0,
            "ece": _ece(y_true, y_prob),
        }
    return {
        "macro_f1": float(np.mean([item["f1"] for item in per_label.values()])),
        "macro_brier": float(
            np.mean([item["brier"] for item in per_label.values()])
        ),
        "macro_ece": float(np.mean([item["ece"] for item in per_label.values()])),
        "per_label": per_label,
    }


def _probability_matrix(
    windows: Sequence[MechanicWindowRecord],
    priors: Mapping[str, Mapping[str, int]],
    *,
    mode: str,
) -> tuple[np.ndarray, list[tuple[MechanicEvidence, ...]]]:
    rows = []
    evidence_rows = []
    for window in windows:
        if mode == "structured":
            probabilities, evidence = predict_mechanic_effects(
                window.context, window.query, priors
            )
        elif mode == "context_ablation":
            probabilities, evidence = predict_mechanic_effects(
                (), window.query, priors
            )
        elif mode == "local_action":
            probabilities = {
                effect: _action_only_probability(
                    window, effect, priors, use_context=True
                )
                for effect in EFFECT_LABELS
            }
            evidence = ()
        elif mode == "global_action":
            probabilities = {
                effect: _action_only_probability(
                    window, effect, priors, use_context=False
                )
                for effect in EFFECT_LABELS
            }
            evidence = ()
        elif mode == "template":
            probabilities = _template_probabilities(window)
            evidence = ()
        else:
            raise ValueError(f"unknown V4 probability mode: {mode}")
        rows.append([probabilities[label] for label in EFFECT_LABELS])
        evidence_rows.append(tuple(evidence))
    return np.asarray(rows, dtype=np.float64), evidence_rows


def _shuffle_context(
    windows: Sequence[MechanicWindowRecord],
    *,
    binding: bool,
) -> list[MechanicWindowRecord]:
    shuffled = []
    for window in windows:
        events = list(window.context)
        if len(events) > 1:
            initial = 1 + int(window.window_digest[:8], 16) % (len(events) - 1)
            offsets = [
                ((initial - 1 + step) % (len(events) - 1)) + 1
                for step in range(len(events) - 1)
            ]
            if binding:
                signature = [event.anchor_condition for event in events]
                offset = next(
                    (
                        candidate
                        for candidate in offsets
                        if signature[candidate:] + signature[:candidate]
                        != signature
                    ),
                    initial,
                )
            else:
                signature = [
                    (tuple(event.effects.items()), tuple(event.applicable.items()))
                    for event in events
                ]
                offset = next(
                    (
                        candidate
                        for candidate in offsets
                        if signature[candidate:] + signature[:candidate]
                        != signature
                    ),
                    initial,
                )
            rotated = events[offset:] + events[:offset]
            if binding:
                events = [
                    replace(event, anchor_condition=source.anchor_condition)
                    for event, source in zip(events, rotated)
                ]
            else:
                events = [
                    replace(
                        event,
                        effects=source.effects,
                        applicable=source.applicable,
                        actor_role_known=source.actor_role_known,
                    )
                    for event, source in zip(events, rotated)
                ]
        shuffled.append(replace(window, context=tuple(events), window_digest=""))
    return shuffled


def _brier_skill(model: Mapping[str, Any], baseline: Mapping[str, Any]) -> float:
    denominator = float(baseline["macro_brier"])
    if denominator <= 0.0:
        return 0.0
    return 1.0 - float(model["macro_brier"]) / denominator


def _bootstrap_skill(
    windows: Sequence[MechanicWindowRecord],
    targets: np.ndarray,
    masks: np.ndarray,
    model: np.ndarray,
    baseline: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    groups: dict[str, np.ndarray] = {}
    keys = np.asarray([window.run_key for window in windows])
    for key in sorted(set(keys)):
        groups[key] = np.flatnonzero(keys == key)
    by_game: dict[str, list[np.ndarray]] = defaultdict(list)
    for key, indices in groups.items():
        by_game[key.split(":", 1)[0]].append(indices)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(samples)):
        sampled_parts = []
        for game in sorted(by_game):
            game_groups = by_game[game]
            choices = rng.integers(0, len(game_groups), size=len(game_groups))
            sampled_parts.extend(game_groups[index] for index in choices)
        sampled = np.concatenate(sampled_parts)
        left = multilabel_metrics(
            targets[sampled], masks[sampled], model[sampled]
        )
        right = multilabel_metrics(
            targets[sampled], masks[sampled], baseline[sampled]
        )
        values.append(_brier_skill(left, right))
    return {
        "samples": int(samples),
        "mean": float(np.mean(values)),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def load_frozen_manifest(path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH) -> dict:
    payload = _read_json(Path(path))
    expected = str(payload.get("manifest_checksum", ""))
    check = dict(payload)
    check.pop("manifest_checksum", None)
    actual = _checksum(check)
    if expected != actual:
        raise ValueError(f"V4 frozen-manifest checksum mismatch: {actual} != {expected}")
    if payload.get("format_version") != "sage12-mechanic-induction-v4":
        raise ValueError("unsupported SAGE12 V4 manifest")
    return payload


def run_source_train_preflight(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    frozen_manifest_path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH,
) -> dict[str, Any]:
    destination = Path(output_dir)
    frozen = load_frozen_manifest(frozen_manifest_path)
    traces = _load_traces(V3_OUTPUT_DIR / "shards", SOURCE_TRAIN)
    windows = build_mechanic_windows(
        traces, context_length=int(frozen["window"]["context_length"])
    )
    for window in windows:
        validate_model_view(window)
    priors = fit_source_priors(windows)
    _write_jsonl_atomic(destination / "source_train_windows.jsonl", windows)
    priors_payload: dict[str, Any] = {
        "format_version": "sage12-mechanic-priors-v4",
        "frozen_manifest_checksum": frozen["manifest_checksum"],
        "prior_strength": float(frozen["model"]["prior_strength"]),
        "counts": priors,
    }
    priors_payload["priors_checksum"] = _checksum(priors_payload)
    _write_json_atomic(destination / "source_priors.json", priors_payload)
    quality = _window_quality(windows)
    static_rows = [
        {
            f"action:{window.query.action_name}": 1,
            f"family:{window.query.action_family}": 1,
            f"anchor:{window.query.anchor_condition}": 1,
        }
        for window in windows
    ]
    probe = _identity_probe(static_rows, [window.game_id for window in windows])
    action_rows = [
        {f"action:{window.query.action_name}": 1} for window in windows
    ]
    action_probe = _identity_probe(
        action_rows, [window.game_id for window in windows]
    )
    identity_gain = probe["accuracy"] - action_probe["accuracy"]
    gates = {
        "minimum_source_train_windows": len(windows)
        >= int(frozen["gates"]["minimum_source_train_windows"]),
        "source_train_label_capacity": _label_capacity(
            quality["per_label"],
            int(frozen["gates"]["minimum_source_train_positives_per_label"]),
            int(frozen["gates"]["minimum_source_train_negatives_per_label"]),
        ),
        "minimum_global_actor_known_rate": quality["actor_role_known_rate"]
        >= float(frozen["gates"]["minimum_global_actor_known_rate"]),
        "minimum_per_game_actor_known_rate": all(
            item["actor_role_known_rate"]
            >= float(frozen["gates"]["minimum_per_game_actor_known_rate"])
            for item in quality["per_game"].values()
        ),
        "static_identity_leakage": identity_gain
        <= float(frozen["gates"]["maximum_static_identity_gain_over_action"]),
        "model_view_firewall": True,
    }
    payload: dict[str, Any] = {
        "format_version": PREFLIGHT_FORMAT_VERSION,
        "status": (
            "PASS_SOURCE_TRAIN_PREFLIGHT"
            if all(gates.values())
            else "FAIL_SOURCE_TRAIN_PREFLIGHT"
        ),
        "frozen_manifest_checksum": frozen["manifest_checksum"],
        "rows": len(traces),
        "windows": len(windows),
        "quality": quality,
        "identity_probe": {
            "static": probe,
            "action_only": action_probe,
            "gain": identity_gain,
        },
        "gates": gates,
        "priors_checksum": priors_payload["priors_checksum"],
        "source_validation_opened": False,
        "world_model_fit_authorized": False,
    }
    payload["preflight_checksum"] = _checksum(payload)
    _write_json_atomic(destination / "source_train_preflight.json", payload)
    return payload


def run_evaluation(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    frozen_manifest_path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH,
    run_qwen: bool = True,
) -> dict[str, Any]:
    destination = Path(output_dir)
    frozen = load_frozen_manifest(frozen_manifest_path)
    preflight = _read_json(destination / "source_train_preflight.json")
    priors_payload = _read_json(destination / "source_priors.json")
    priors = dict(priors_payload["counts"])
    traces = _load_traces(destination / "shards", SOURCE_VALIDATION)
    windows = build_mechanic_windows(
        traces, context_length=int(frozen["window"]["context_length"])
    )
    for window in windows:
        validate_model_view(window)
    _write_jsonl_atomic(destination / "validation_windows.jsonl", windows)
    targets, masks = _arrays(windows)
    matrices = {}
    evidence_rows = None
    for mode in (
        "structured",
        "context_ablation",
        "local_action",
        "global_action",
        "template",
    ):
        matrices[mode], evidence = _probability_matrix(windows, priors, mode=mode)
        if mode == "structured":
            evidence_rows = evidence
    metrics = {
        mode: multilabel_metrics(targets, masks, values)
        for mode, values in matrices.items()
    }
    stronger_name = min(
        ("local_action", "global_action", "template"),
        key=lambda name: metrics[name]["macro_brier"],
    )
    stronger = metrics[stronger_name]
    primary_skill = _brier_skill(metrics["structured"], stronger)
    f1_gain = metrics["structured"]["macro_f1"] - stronger["macro_f1"]
    shuffled_effect_windows = _shuffle_context(windows, binding=False)
    shuffled_effect, _ = _probability_matrix(
        shuffled_effect_windows, priors, mode="structured"
    )
    shuffled_effect_metrics = multilabel_metrics(targets, masks, shuffled_effect)
    shuffled_binding_windows = _shuffle_context(windows, binding=True)
    shuffled_binding, _ = _probability_matrix(
        shuffled_binding_windows, priors, mode="structured"
    )
    shuffled_binding_metrics = multilabel_metrics(
        targets, masks, shuffled_binding
    )
    source_windows = _load_windows(destination / "source_train_windows.jsonl")
    permuted_windows = _permute_source_labels(source_windows)
    permuted_priors = fit_source_priors(permuted_windows)
    label_permutation, _ = _probability_matrix(
        windows, permuted_priors, mode="structured"
    )
    label_permutation_metrics = multilabel_metrics(
        targets, masks, label_permutation
    )
    outcome_shuffle_drop = primary_skill - _brier_skill(
        shuffled_effect_metrics, stronger
    )
    context_gain = _brier_skill(
        metrics["structured"], metrics["context_ablation"]
    )
    bootstrap = _bootstrap_skill(
        windows,
        targets,
        masks,
        matrices["structured"],
        matrices[stronger_name],
        samples=int(frozen["evaluation"]["bootstrap_samples"]),
        seed=int(frozen["evaluation"]["random_seed"]),
    )
    per_game = {}
    for game in SOURCE_VALIDATION:
        selected = np.asarray([window.game_id == game for window in windows])
        model_metric = multilabel_metrics(
            targets[selected], masks[selected], matrices["structured"][selected]
        )
        candidates = {
            name: multilabel_metrics(
                targets[selected], masks[selected], matrices[name][selected]
            )
            for name in ("local_action", "global_action", "template")
        }
        baseline_name = min(
            candidates, key=lambda name: candidates[name]["macro_brier"]
        )
        per_game[game] = {
            "windows": int(np.sum(selected)),
            "structured": model_metric,
            "stronger_baseline": baseline_name,
            "baseline": candidates[baseline_name],
            "brier_skill": _brier_skill(
                model_metric, candidates[baseline_name]
            ),
        }
    output_contract = _output_contract(windows, evidence_rows or [])
    qwen = (
        _evaluate_qwen(
            windows,
            priors,
            frozen,
            targets,
            masks,
            output_dir=destination,
        )
        if run_qwen
        else {"status": "SKIPPED", "promotion_gate": False}
    )
    quality = _window_quality(windows)
    gates_cfg = frozen["gates"]
    gates = {
        "minimum_prospective_windows": len(windows)
        >= int(gates_cfg["minimum_prospective_windows"]),
        "prospective_label_capacity": _label_capacity(
            quality["per_label"],
            int(gates_cfg["minimum_validation_positives_per_label"]),
            int(gates_cfg["minimum_validation_negatives_per_label"]),
        ),
        "minimum_global_actor_known_rate": quality["actor_role_known_rate"]
        >= float(gates_cfg["minimum_global_actor_known_rate"]),
        "minimum_per_game_actor_known_rate": all(
            item["actor_role_known_rate"]
            >= float(gates_cfg["minimum_per_game_actor_known_rate"])
            for item in quality["per_game"].values()
        ),
        "strict_json_validity": output_contract["strict_json_validity"] == 1.0,
        "support_zero_rate": output_contract["support_zero_rate"] == 1.0,
        "grounded_hypothesis_rate": output_contract["grounded_hypothesis_rate"]
        == 1.0,
        "minimum_brier_skill": primary_skill
        >= float(gates_cfg["minimum_macro_brier_skill"]),
        "bootstrap_lower_bound_positive": bootstrap["lower_95"] > 0.0,
        "minimum_macro_f1_gain": f1_gain
        >= float(gates_cfg["minimum_macro_f1_gain"]),
        "minimum_outcome_shuffle_drop": outcome_shuffle_drop
        >= float(gates_cfg["minimum_outcome_shuffle_skill_drop"]),
        "minimum_context_gain": context_gain
        >= float(gates_cfg["minimum_context_brier_skill_gain"]),
        "every_game_nonnegative": all(
            item["brier_skill"] >= 0.0 for item in per_game.values()
        ),
        "maximum_macro_ece": metrics["structured"]["macro_ece"]
        <= float(gates_cfg["maximum_macro_ece"]),
        "source_preflight_passed": all(preflight["gates"].values()),
    }
    passed = all(gates.values())
    predictions = []
    for index, window in enumerate(windows):
        predictions.append(
            {
                "window_digest": window.window_digest,
                "game_id": window.game_id,
                "run_key": window.run_key,
                "query": window.query.to_dict(),
                "labels": dict(window.labels),
                "applicable": dict(window.applicable),
                "structured_probabilities": {
                    label: float(matrices["structured"][index, label_index])
                    for label_index, label in enumerate(EFFECT_LABELS)
                },
                "evidence": [
                    item.to_dict() for item in (evidence_rows or [])[index]
                ],
            }
        )
    _write_jsonl_dicts_atomic(destination / "predictions.jsonl", predictions)
    payload: dict[str, Any] = {
        "format_version": RESULT_FORMAT_VERSION,
        "status": "PASS" if passed else "FAIL_CLOSED",
        "all_gates_passed": passed,
        "frozen_manifest_checksum": frozen["manifest_checksum"],
        "preflight_checksum": preflight["preflight_checksum"],
        "rows": {
            "prospective_transitions": len(traces),
            "prospective_windows": len(windows),
        },
        "quality": quality,
        "metrics": metrics,
        "stronger_baseline": stronger_name,
        "primary_macro_brier_skill": primary_skill,
        "primary_macro_f1_gain": f1_gain,
        "bootstrap_skill": bootstrap,
        "outcome_shuffle": {
            "metrics": shuffled_effect_metrics,
            "skill_drop": outcome_shuffle_drop,
        },
        "binding_shuffle": {
            "metrics": shuffled_binding_metrics,
            "skill": _brier_skill(shuffled_binding_metrics, stronger),
        },
        "label_permutation": {
            "metrics": label_permutation_metrics,
            "skill": _brier_skill(label_permutation_metrics, stronger),
        },
        "context_brier_skill_gain": context_gain,
        "per_game": per_game,
        "output_contract": output_contract,
        "qwen": qwen,
        "gates": gates,
        "firewall": {
            "source_only": True,
            "holdout_opened": False,
            "historical_opened": False,
            "ar25_opened": False,
            "controller_executed": False,
        },
        "world_model_fit_authorized": passed,
        "ebm_fit_authorized": False,
    }
    payload["result_checksum"] = _checksum(payload)
    _write_json_atomic(destination / "pilot_result.json", payload)
    return payload


def _output_contract(
    windows: Sequence[MechanicWindowRecord],
    evidence_rows: Sequence[Sequence[MechanicEvidence]],
) -> dict[str, Any]:
    emitted = 0
    valid = 0
    support_zero = 0
    grounded = 0
    for window, row in zip(windows, evidence_rows):
        rules = [item.rule.to_dict() for item in row]
        raw = _canonical({"hypotheses": rules})
        parsed = json.loads(raw)["hypotheses"]
        for item, evidence in zip(parsed, row):
            emitted += 1
            rule = MechanicRule.from_dict(item)
            valid += 1
            support_zero += int(rule.support == 0)
            hypothesis = rule_to_semantic_hypothesis(
                rule,
                window.query,
                confidence=evidence.posterior_probability,
            )
            grounded += int(
                hypothesis.action_name == window.query.action_name
                and rule.matches_query(window.query)
            )
    denominator = max(1, emitted)
    return {
        "emitted_hypotheses": emitted,
        "strict_json_validity": valid / denominator,
        "support_zero_rate": support_zero / denominator,
        "grounded_hypothesis_rate": grounded / denominator,
    }


def _evaluate_qwen(
    windows: Sequence[MechanicWindowRecord],
    priors: Mapping[str, Mapping[str, int]],
    frozen: Mapping[str, Any],
    targets: np.ndarray,
    masks: np.ndarray,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    count = min(int(frozen["qwen"]["contexts"]), len(windows))
    selected_indices = []
    per_game = max(1, count // len(SOURCE_VALIDATION))
    for game in SOURCE_VALIDATION:
        candidates = [
            (index, window)
            for index, window in enumerate(windows)
            if window.game_id == game
        ]
        candidates.sort(
            key=lambda item: hashlib.sha256(
                (
                    item[1].query.action_name
                    + ":"
                    + item[1].query.anchor_condition
                    + ":"
                    + item[1].window_digest
                ).encode()
            ).hexdigest()
        )
        selected_indices.extend(index for index, _ in candidates[:per_game])
    if len(selected_indices) < count:
        remaining = [
            index
            for index in range(len(windows))
            if index not in set(selected_indices)
        ]
        selected_indices.extend(remaining[: count - len(selected_indices)])
    config = TransformersModelConfig(
        model_path=str(frozen["qwen"]["path"]),
        device=str(frozen["qwen"]["device"]),
        temperature=0.0,
        maximum_input_tokens=int(frozen["qwen"]["maximum_input_tokens"]),
    )
    model = TransformersJSONModel(config)
    probabilities = []
    outputs = []
    valid = 0
    grounded = 0
    started = time.perf_counter()
    for index in selected_indices:
        window = windows[index]
        raw = ""
        rules: tuple[MechanicRule, ...] = ()
        error = ""
        try:
            raw = model.generate_json(
                prompt=_qwen_prompt(window),
                schema=_qwen_schema(),
                maximum_tokens=int(frozen["qwen"]["maximum_output_tokens"]),
            )
            payload = json.loads(raw)
            items = payload.get("hypotheses", [])
            rules = tuple(
                MechanicRule.from_dict({**item, "source": "local_llm"})
                for item in items[:8]
            )
            valid += 1
        except Exception as exc:  # environment/model output is audited
            error = f"{type(exc).__name__}: {exc}"
        row = {
            effect: _action_only_probability(
                window, effect, priors, use_context=False
            )
            for effect in EFFECT_LABELS
        }
        for rule in rules:
            if rule.matches_query(window.query):
                evidence = score_rule(rule, window.context, priors)
                row[rule.effect] = evidence.posterior_probability
                grounded += 1
        probabilities.append([row[label] for label in EFFECT_LABELS])
        outputs.append(
            {
                "window_digest": window.window_digest,
                "raw_response": raw[:4096],
                "parse_error": error,
                "hypotheses": [rule.to_dict() for rule in rules],
            }
        )
    matrix = np.asarray(probabilities, dtype=np.float64)
    subset = np.asarray(selected_indices, dtype=int)
    metrics = multilabel_metrics(targets[subset], masks[subset], matrix)
    _write_jsonl_dicts_atomic(output_dir / "qwen_outputs.jsonl", outputs)
    emitted = sum(len(item["hypotheses"]) for item in outputs)
    return {
        "status": "COMPLETE",
        "promotion_gate": False,
        "contexts": len(selected_indices),
        "strict_json_validity": valid / max(1, len(selected_indices)),
        "grounded_hypothesis_rate": grounded / max(1, emitted),
        "emitted_hypotheses": emitted,
        "metrics": metrics,
        "inference_seconds": time.perf_counter() - started,
        "device": str(frozen["qwen"]["device"]),
        "model": str(frozen["qwen"]["name"]),
    }


def _qwen_prompt(window: MechanicWindowRecord) -> str:
    return (
        "Infer at most eight causal rules from the observed semantic history. "
        "Use only exact/family action scopes, the supplied anchor conditions, "
        "and the four allowed effects. Every support field must be 0. "
        "Return JSON only.\n"
        + _canonical(window.model_view())
    )


def _qwen_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["hypotheses"],
        "properties": {
            "hypotheses": {
                "type": "array",
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "required": [
                        "rule_id",
                        "action_scope_kind",
                        "action_scope_value",
                        "anchor_condition",
                        "effect",
                        "support",
                    ],
                    "properties": {
                        "rule_id": {"type": "string"},
                        "action_scope_kind": {
                            "type": "string",
                            "enum": list(ACTION_SCOPE_KINDS),
                        },
                        "action_scope_value": {"type": "string"},
                        "anchor_condition": {
                            "type": "string",
                            "enum": list(ANCHOR_CONDITIONS),
                        },
                        "effect": {
                            "type": "string",
                            "enum": list(EFFECT_LABELS),
                        },
                        "support": {"const": 0},
                    },
                    "additionalProperties": False,
                },
            }
        },
        "additionalProperties": False,
    }


def _window_quality(windows: Sequence[MechanicWindowRecord]) -> dict[str, Any]:
    per_label = {}
    for label in EFFECT_LABELS:
        eligible = [window for window in windows if window.applicable[label]]
        positive = sum(int(window.labels[label]) for window in eligible)
        per_label[label] = {
            "applicable": len(eligible),
            "positives": positive,
            "negatives": len(eligible) - positive,
        }
    per_game = {}
    for game in sorted({window.game_id for window in windows}):
        selected = [window for window in windows if window.game_id == game]
        per_game[game] = {
            "windows": len(selected),
            "actor_role_known_rate": sum(
                int(window.actor_role_known) for window in selected
            )
            / max(1, len(selected)),
        }
    return {
        "windows": len(windows),
        "unique_window_digests": len({window.window_digest for window in windows}),
        "actor_role_known_rate": sum(
            int(window.actor_role_known) for window in windows
        )
        / max(1, len(windows)),
        "per_label": per_label,
        "per_game": per_game,
    }


def _label_capacity(
    per_label: Mapping[str, Mapping[str, int]],
    minimum_positive: int,
    minimum_negative: int,
) -> bool:
    return all(
        int(item["positives"]) >= minimum_positive
        and int(item["negatives"]) >= minimum_negative
        for item in per_label.values()
    )


def _identity_probe(
    rows: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
) -> dict[str, float]:
    vectorizer = DictVectorizer(sparse=True)
    matrix = vectorizer.fit_transform(rows)
    label_array = np.asarray(labels)
    counts = Counter(label_array)
    splits = min(5, min(counts.values()))
    estimator = OneVsRestClassifier(
        LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=131,
        )
    )
    predicted = cross_val_predict(
        estimator,
        matrix,
        label_array,
        cv=StratifiedKFold(n_splits=splits, shuffle=True, random_state=131),
    )
    majority = max(counts.values()) / len(label_array)
    return {
        "accuracy": float(np.mean(predicted == label_array)),
        "majority_accuracy": float(majority),
    }


def _load_traces(
    shard_dir: Path, games: Sequence[str]
) -> list[ActionTargetTrace]:
    rows = []
    for game in games:
        path = shard_dir / f"{game}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(ActionTargetTrace.from_dict(json.loads(line)))
    return rows


def _load_windows(path: Path) -> list[MechanicWindowRecord]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(MechanicWindowRecord.from_dict(json.loads(line)))
    return rows


def _permute_source_labels(
    windows: Sequence[MechanicWindowRecord],
) -> list[MechanicWindowRecord]:
    if len(windows) < 2:
        return list(windows)
    offset = 1 + int(
        hashlib.sha256(b"sage12-v4-label-permutation").hexdigest()[:8], 16
    ) % (len(windows) - 1)
    rotated = list(windows[offset:]) + list(windows[:offset])
    return [
        replace(
            window,
            labels=source.labels,
            applicable=source.applicable,
        )
        for window, source in zip(windows, rotated)
    ]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl_atomic(
    path: Path, windows: Sequence[MechanicWindowRecord]
) -> None:
    _write_jsonl_dicts_atomic(path, [window.to_dict() for window in windows])


def _write_jsonl_dicts_atomic(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical(row) + "\n")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "evaluate"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--frozen-manifest", default=str(DEFAULT_FROZEN_MANIFEST_PATH)
    )
    parser.add_argument("--skip-qwen", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = run_source_train_preflight(
            output_dir=args.output_dir,
            frozen_manifest_path=args.frozen_manifest,
        )
    else:
        result = run_evaluation(
            output_dir=args.output_dir,
            frozen_manifest_path=args.frozen_manifest,
            run_qwen=not args.skip_qwen,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANCHOR_CONDITIONS",
    "FORMAT_VERSION",
    "MechanicEvidence",
    "MechanicQuery",
    "MechanicRule",
    "MechanicWindowRecord",
    "PersistentRoleTracker",
    "SemanticTransitionEvent",
    "build_mechanic_windows",
    "fit_source_priors",
    "multilabel_metrics",
    "predict_mechanic_effects",
    "rule_to_semantic_hypothesis",
    "run_evaluation",
    "run_source_train_preflight",
    "score_rule",
    "validate_model_view",
]
