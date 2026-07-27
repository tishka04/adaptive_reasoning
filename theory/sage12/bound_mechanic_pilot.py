"""SAGE12 V4.3 counterfactual binding and semantic world-model pilot.

The pilot is deliberately independent from the V4.2.1 corpus.  It executes
two branches from a replay-verified identical pre-state, exposes only typed
binding semantics to models, freezes all calibration on source-train games,
and keeps the structured world model behind a binding-model pass gate.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from theory.m1.polymorphic_a25_adapter import _step_env_action
from theory.m2.m3_execution_smoke import _reset_env
from theory.non_ar25_active_micro_run import _env_dir, _valid_actions
from theory.real_env_option_adapter import snapshot_frame
from theory.sage11.splits import (
    SAGE11_SPLITS,
    SOURCE_TRAIN,
    SOURCE_VALIDATION,
)
from theory.unified_cognition_ab_benchmark import (
    _available_action_names,
    _is_terminal,
    _make_real_env,
)

from .action_target_data import (
    ActionTargetAnchor,
    ActionTargetTrace,
    build_action_target_trace,
    build_observation,
    grid_sha256,
    resolve_action_target,
)
from .mechanic_induction import _identity_probe as _categorical_identity_probe
from .mechanic_replication import (
    _apply_parameter,
    _fit_platt,
    _select_threshold,
)

FORMAT_VERSION = "sage12-bound-trajectory-v4.3"
MANIFEST_FORMAT_VERSION = "sage12-bound-mechanic-pilot-v4.3"
COLLECTION_FORMAT_VERSION = "sage12-bound-collection-v4.3"
PREFLIGHT_FORMAT_VERSION = "sage12-bound-preflight-v4.3"
BINDING_RESULT_FORMAT_VERSION = "sage12-bound-binding-result-v4.3"
WORLD_MODEL_RESULT_FORMAT_VERSION = "sage12-bound-world-model-result-v4.3"
DEFAULT_OUTPUT_DIR = Path("training") / "sage12" / "bound_mechanic_pilot_v4_3"
DEFAULT_FROZEN_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "frozen_manifest.json"

TARGET_EFFECTS = ("target_created", "target_removed", "target_moved")
PROJECTION_LADDER = ("minimal", "relational", "typed")
MODEL_MODES = (
    "structured",
    "no_binding",
    "action_only",
    "binding_only",
    "template",
)
BASELINE_MODES = ("no_binding", "action_only", "binding_only", "template")
SOURCE_SEEDS = (857, 907, 953, 1009)
VALIDATION_SEEDS = (1061, 1103, 1151, 1201)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _canonical(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def _checksum(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _payload_without_checksum(
    payload: Mapping[str, Any], checksum_field: str
) -> dict[str, Any]:
    clean = dict(payload)
    clean.pop(checksum_field, None)
    return clean


def _frame_hash(frame: Any) -> str:
    snap = snapshot_frame(frame)
    payload = {
        "grid": grid_sha256(snap.grid),
        "game_state": str(snap.game_state),
        "levels_completed": int(snap.levels_completed),
    }
    return _checksum(payload)


def _canonical_binding_kind(anchor: ActionTargetAnchor) -> str:
    if anchor.kind == "targetless" or not anchor.in_bounds:
        return "targetless"
    return "occupied_object" if anchor.occupied else "free_slot"


@dataclass(frozen=True)
class BindingSignature:
    """Identity-free semantic description of an action argument."""

    kind: str
    action_family: str
    requested_direction: str
    occupied: bool
    path_status: str
    actor_relation: str
    actor_relative_direction: str
    target_area_bucket: str
    target_aspect_bucket: str
    target_affordance: str

    def __post_init__(self) -> None:
        if self.kind not in {"occupied_object", "free_slot", "targetless"}:
            raise ValueError("unsupported V4.3 binding kind")

    @classmethod
    def from_anchor(cls, anchor: ActionTargetAnchor) -> BindingSignature:
        return cls(
            kind=_canonical_binding_kind(anchor),
            action_family=anchor.action_family,
            requested_direction=anchor.requested_direction,
            occupied=bool(anchor.occupied),
            path_status=anchor.path_status,
            actor_relation=anchor.actor_relation,
            actor_relative_direction=anchor.actor_relative_direction,
            target_area_bucket=anchor.target_area_bucket,
            target_aspect_bucket=anchor.target_aspect_bucket,
            target_affordance=anchor.target_affordance,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BindingSignature:
        return cls(
            kind=str(payload["kind"]),
            action_family=str(payload["action_family"]),
            requested_direction=str(payload.get("requested_direction", "none")),
            occupied=bool(payload.get("occupied", False)),
            path_status=str(payload.get("path_status", "unknown")),
            actor_relation=str(payload.get("actor_relation", "unknown")),
            actor_relative_direction=str(
                payload.get("actor_relative_direction", "unknown")
            ),
            target_area_bucket=str(payload.get("target_area_bucket", "none")),
            target_aspect_bucket=str(payload.get("target_aspect_bucket", "none")),
            target_affordance=str(payload.get("target_affordance", "none")),
        )

    def model_view(self, projection: str) -> dict[str, Any]:
        if projection not in PROJECTION_LADDER:
            raise ValueError(f"unknown V4.3 projection: {projection}")
        result: dict[str, Any] = {
            "kind": self.kind,
            "occupied": int(self.occupied),
            "path_status": self.path_status,
        }
        if projection in {"relational", "typed"}:
            result.update(
                {
                    "requested_direction": self.requested_direction,
                    "actor_relation": self.actor_relation,
                    "actor_relative_direction": self.actor_relative_direction,
                }
            )
        if projection == "typed":
            result.update(
                {
                    "target_area_bucket": self.target_area_bucket,
                    "target_aspect_bucket": self.target_aspect_bucket,
                    "target_affordance": self.target_affordance,
                }
            )
        return result

    def key(self, projection: str) -> str:
        return _canonical(self.model_view(projection))


@dataclass(frozen=True)
class BoundEvent:
    action_name: str
    action_family: str
    binding: BindingSignature
    effects: Mapping[str, bool]
    applicable: Mapping[str, bool]

    def __post_init__(self) -> None:
        if set(self.effects) != set(TARGET_EFFECTS):
            raise ValueError("V4.3 event requires all target effects")
        if set(self.applicable) != set(TARGET_EFFECTS):
            raise ValueError("V4.3 event requires all applicability masks")

    @classmethod
    def from_trace(cls, trace: ActionTargetTrace) -> BoundEvent:
        return cls(
            action_name=trace.selected_action_name,
            action_family=trace.anchor.action_family,
            binding=BindingSignature.from_anchor(trace.anchor),
            effects={
                label: bool(trace.effects.labels[label]) for label in TARGET_EFFECTS
            },
            applicable={
                label: bool(trace.effects.applicable[label]) for label in TARGET_EFFECTS
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_name": self.action_name,
            "action_family": self.action_family,
            "binding": self.binding.to_dict(),
            "effects": {label: bool(self.effects[label]) for label in TARGET_EFFECTS},
            "applicable": {
                label: bool(self.applicable[label]) for label in TARGET_EFFECTS
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BoundEvent:
        effects = dict(payload.get("effects", {}))
        applicable = dict(payload.get("applicable", {}))
        return cls(
            action_name=str(payload["action_name"]),
            action_family=str(payload["action_family"]),
            binding=BindingSignature.from_dict(payload["binding"]),
            effects={
                label: bool(effects.get(label, False)) for label in TARGET_EFFECTS
            },
            applicable={
                label: bool(applicable.get(label, False)) for label in TARGET_EFFECTS
            },
        )

    def model_view(self, projection: str) -> dict[str, Any]:
        return {
            "action_name": self.action_name,
            "action_family": self.action_family,
            "binding": self.binding.model_view(projection),
            "effects": {label: bool(self.effects[label]) for label in TARGET_EFFECTS},
            "applicable": {
                label: bool(self.applicable[label]) for label in TARGET_EFFECTS
            },
        }


@dataclass(frozen=True)
class ActionSpec:
    name: str
    action_args: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "action_args": _json_safe(self.action_args)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ActionSpec:
        return cls(
            name=str(payload["name"]),
            action_args=dict(payload.get("action_args", {})),
        )

    @classmethod
    def from_action(cls, action: Any) -> ActionSpec:
        return cls(name=str(action.name), action_args=dict(action.action_args))

    @property
    def key(self) -> str:
        return _canonical(self.to_dict())


@dataclass(frozen=True)
class BranchArm:
    arm: str
    action: ActionSpec
    trace: ActionTargetTrace
    replay_pre_state_sha256: str
    post_state_sha256: str

    def __post_init__(self) -> None:
        if self.arm not in {"left", "right"}:
            raise ValueError("branch arm must be left or right")
        if self.action.name != self.trace.selected_action_name:
            raise ValueError("branch action/trace mismatch")

    @property
    def event(self) -> BoundEvent:
        return BoundEvent.from_trace(self.trace)

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "action": self.action.to_dict(),
            "trace": self.trace.to_dict(),
            "replay_pre_state_sha256": self.replay_pre_state_sha256,
            "post_state_sha256": self.post_state_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BranchArm:
        return cls(
            arm=str(payload["arm"]),
            action=ActionSpec.from_dict(payload["action"]),
            trace=ActionTargetTrace.from_dict(payload["trace"]),
            replay_pre_state_sha256=str(payload["replay_pre_state_sha256"]),
            post_state_sha256=str(payload["post_state_sha256"]),
        )


@dataclass(frozen=True)
class BindingPairRecord:
    """Two executed interventions from one replay-verified pre-state."""

    game_id: str
    source_split: str
    policy_seed: int
    reset_index: int
    root_index: int
    path: str
    depth: int
    context: tuple[BoundEvent, ...]
    expected_pre_state_sha256: str
    replay_pre_state_sha256: str
    left: BranchArm
    right: BranchArm
    pair_digest: str = ""
    format_version: str = FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version != FORMAT_VERSION:
            raise ValueError("unsupported SAGE12 V4.3 pair format")
        if self.source_split not in {"source_train", "source_validation"}:
            raise ValueError("V4.3 pairs are source-only")
        if self.depth not in {0, 1, 2} or len(self.path) != self.depth:
            raise ValueError("invalid binary-tree path/depth")
        if len(self.context) != 8:
            raise ValueError("V4.3 requires an eight-transition context")
        hashes = {
            self.expected_pre_state_sha256,
            self.replay_pre_state_sha256,
            self.left.replay_pre_state_sha256,
            self.right.replay_pre_state_sha256,
        }
        if len(hashes) != 1:
            raise ValueError("counterfactual arms do not share identical pre-state")
        if not self.pair_digest:
            payload = {
                "game_id": self.game_id,
                "source_split": self.source_split,
                "policy_seed": self.policy_seed,
                "reset_index": self.reset_index,
                "root_index": self.root_index,
                "path": self.path,
                "pre": self.expected_pre_state_sha256,
                "left": self.left.trace.trace_digest,
                "right": self.right.trace.trace_digest,
            }
            object.__setattr__(self, "pair_digest", _checksum(payload))

    @property
    def root_key(self) -> str:
        return f"{self.game_id}:{self.policy_seed}:{self.reset_index}:{self.root_index}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "game_id": self.game_id,
            "source_split": self.source_split,
            "policy_seed": self.policy_seed,
            "reset_index": self.reset_index,
            "root_index": self.root_index,
            "path": self.path,
            "depth": self.depth,
            "context": [item.to_dict() for item in self.context],
            "expected_pre_state_sha256": self.expected_pre_state_sha256,
            "replay_pre_state_sha256": self.replay_pre_state_sha256,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
            "pair_digest": self.pair_digest,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BindingPairRecord:
        return cls(
            format_version=str(payload.get("format_version", FORMAT_VERSION)),
            game_id=str(payload["game_id"]),
            source_split=str(payload["source_split"]),
            policy_seed=int(payload["policy_seed"]),
            reset_index=int(payload["reset_index"]),
            root_index=int(payload["root_index"]),
            path=str(payload["path"]),
            depth=int(payload["depth"]),
            context=tuple(
                BoundEvent.from_dict(item) for item in payload.get("context", ())
            ),
            expected_pre_state_sha256=str(payload["expected_pre_state_sha256"]),
            replay_pre_state_sha256=str(payload["replay_pre_state_sha256"]),
            left=BranchArm.from_dict(payload["left"]),
            right=BranchArm.from_dict(payload["right"]),
            pair_digest=str(payload.get("pair_digest", "")),
        )


@dataclass(frozen=True)
class BoundWindow:
    pair_id: str
    arm: str
    game_id: str
    source_split: str
    root_key: str
    path: str
    context: tuple[BoundEvent, ...]
    query_action_name: str
    query_action_family: str
    query_binding: BindingSignature
    labels: Mapping[str, bool]
    applicable: Mapping[str, bool]

    def model_view(self, projection: str) -> dict[str, Any]:
        return {
            "context": [item.model_view(projection) for item in self.context],
            "query": {
                "action_name": self.query_action_name,
                "action_family": self.query_action_family,
                "binding": self.query_binding.model_view(projection),
            },
        }

    def with_query_binding(self, binding: BindingSignature) -> BoundWindow:
        return BoundWindow(
            pair_id=self.pair_id,
            arm=self.arm,
            game_id=self.game_id,
            source_split=self.source_split,
            root_key=self.root_key,
            path=self.path,
            context=self.context,
            query_action_name=self.query_action_name,
            query_action_family=self.query_action_family,
            query_binding=binding,
            labels=self.labels,
            applicable=self.applicable,
        )


def pair_windows(pair: BindingPairRecord) -> tuple[BoundWindow, BoundWindow]:
    result = []
    for arm in (pair.left, pair.right):
        event = arm.event
        result.append(
            BoundWindow(
                pair_id=pair.pair_digest,
                arm=arm.arm,
                game_id=pair.game_id,
                source_split=pair.source_split,
                root_key=pair.root_key,
                path=pair.path,
                context=pair.context,
                query_action_name=event.action_name,
                query_action_family=event.action_family,
                query_binding=event.binding,
                labels=event.effects,
                applicable=event.applicable,
            )
        )
    return result[0], result[1]


def validate_model_view(window: BoundWindow, projection: str) -> None:
    rendered = _canonical(window.model_view(projection)).lower()
    forbidden = (
        window.game_id.lower(),
        "game_id",
        "policy_seed",
        "reset_index",
        "root_index",
        "frame_before",
        "frame_after",
        "target_object_id",
        '"row"',
        '"col"',
        "trace_digest",
    )
    for token in forbidden:
        if token and token in rendered:
            raise ValueError(f"forbidden V4.3 model-input token: {token}")


def load_frozen_manifest(
    path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format_version") != MANIFEST_FORMAT_VERSION:
        raise ValueError("unsupported SAGE12 V4.3 manifest")
    expected = str(payload.get("manifest_checksum", ""))
    actual = _checksum(_payload_without_checksum(payload, "manifest_checksum"))
    if expected != actual:
        raise ValueError("SAGE12 V4.3 manifest checksum mismatch")
    SAGE11_SPLITS.assert_authorized(payload["source_train_games"], purpose="train")
    SAGE11_SPLITS.assert_authorized(
        payload["source_validation_games"], purpose="validate_source"
    )
    if tuple(payload["source_train_games"]) != SOURCE_TRAIN:
        raise ValueError("V4.3 source-train split drift")
    if tuple(payload["source_validation_games"]) != SOURCE_VALIDATION:
        raise ValueError("V4.3 source-validation split drift")
    return payload


def _legal_actions(env: Any) -> tuple[Any, ...]:
    return tuple(
        action
        for action in _valid_actions(env)
        if str(action.name) not in {"", "RESET"}
    )


def _available(legal: Sequence[Any]) -> tuple[str, ...]:
    return tuple(sorted(set(_available_action_names(legal))))


def _observation(frame: Any, legal: Sequence[Any]) -> Any:
    snap = snapshot_frame(frame)
    return build_observation(
        snap.grid,
        available_actions=_available(legal),
        game_state=snap.game_state,
        levels_completed=snap.levels_completed,
        infer_players=True,
    )


def _candidate_signature(action: Any, observation: Any) -> BindingSignature:
    anchor = resolve_action_target(
        observation,
        str(action.name),
        dict(action.action_args),
    )
    return BindingSignature.from_anchor(anchor)


def _ranked_candidates(
    legal: Sequence[Any],
    observation: Any,
    counts: Mapping[str, int],
    *,
    salt: str,
) -> list[tuple[Any, BindingSignature]]:
    rows = []
    for action in legal:
        signature = _candidate_signature(action, observation)
        spec = ActionSpec.from_action(action)
        stratum = f"{spec.name}|{signature.key('typed')}|{spec.key}"
        tie = hashlib.sha256(f"{salt}:{stratum}".encode()).hexdigest()
        rows.append(
            (
                int(counts.get(stratum, 0)),
                tie,
                spec.key,
                action,
                signature,
            )
        )
    rows.sort(key=lambda item: (item[0], item[1], item[2]))
    return [(item[3], item[4]) for item in rows]


def select_branch_actions(
    legal: Sequence[Any],
    observation: Any,
    counts: Mapping[str, int],
    *,
    salt: str,
) -> tuple[Any, Any]:
    """Select two outcome-blind interventions with distinct bindings.

    Same-action/different-argument candidates are preferred.  If unavailable,
    actions in the same semantic family are preferred, then any two distinct
    legal interventions are used.  All ranking inputs are pre-action fields.
    """
    ranked = _ranked_candidates(legal, observation, counts, salt=salt)
    if len(ranked) < 2:
        raise RuntimeError("counterfactual collection needs two legal actions")

    for relation in ("same_action", "same_family", "any"):
        for left_index, (left, left_signature) in enumerate(ranked):
            for right, right_signature in ranked[left_index + 1 :]:
                left_spec = ActionSpec.from_action(left)
                right_spec = ActionSpec.from_action(right)
                if left_spec.key == right_spec.key:
                    continue
                if left_signature.key("typed") == right_signature.key("typed"):
                    continue
                if relation == "same_action" and left_spec.name != right_spec.name:
                    continue
                if (
                    relation == "same_family"
                    and left_signature.action_family != right_signature.action_family
                ):
                    continue
                return left, right

    # A target may be spatially distinct yet collapse under the frozen typed
    # projection. Retain the intervention pair for audit/capacity, but it will
    # not manufacture a positive binding-control signal.
    return ranked[0][0], ranked[1][0]


def _find_action(env: Any, spec: ActionSpec) -> Any:
    matches = [
        action
        for action in _legal_actions(env)
        if ActionSpec.from_action(action).key == spec.key
    ]
    if not matches:
        raise RuntimeError(f"replay action resolution failed for {spec.name}: no match")
    # ARC may expose duplicate legal candidates with byte-identical name and
    # arguments. They denote the same intervention; selecting the first is
    # deterministic, and the replayed pre-state hash remains the authority.
    return matches[0]


def replay_prefix(
    reset_template: Any,
    reset_frame: Any,
    prefix: Sequence[ActionSpec],
    *,
    expected_pre_state_sha256: str,
) -> tuple[Any, Any]:
    """Replay a path and fail closed unless the exact pre-state is restored."""
    env = copy.deepcopy(reset_template)
    frame = copy.deepcopy(reset_frame)
    for spec in prefix:
        action = _find_action(env, spec)
        frame = _step_env_action(env, action)
    actual = _frame_hash(frame)
    if actual != expected_pre_state_sha256:
        raise RuntimeError(
            "deterministic replay mismatch: "
            f"expected {expected_pre_state_sha256}, got {actual}"
        )
    return env, frame


def _execute_arm(
    *,
    env: Any,
    frame: Any,
    action: Any,
    arm: str,
    game: str,
    source_split: str,
    seed: int,
    reset_index: int,
    step_index: int,
    pre_hash: str,
) -> tuple[BranchArm, Any]:
    legal = _legal_actions(env)
    available = _available(legal)
    before = snapshot_frame(frame)
    selected = _find_action(env, ActionSpec.from_action(action))
    after_frame = _step_env_action(env, selected)
    after = snapshot_frame(
        after_frame,
        fallback_available_actions=before.available_actions,
    )
    trace = build_action_target_trace(
        game_id=game,
        source_split=source_split,
        policy_seed=seed,
        reset_index=reset_index,
        step_index=step_index,
        collection_phase="v4_3_replayed_counterfactual_tree",
        available_action_names=available,
        selected_action_name=str(selected.name),
        selected_action_data=dict(selected.action_args),
        frame_before=before.grid,
        frame_after=after.grid,
        game_state_before=before.game_state,
        game_state_after=after.game_state,
        levels_completed_before=before.levels_completed,
        levels_completed_after=after.levels_completed,
    )
    branch = BranchArm(
        arm=arm,
        action=ActionSpec.from_action(selected),
        trace=trace,
        replay_pre_state_sha256=pre_hash,
        post_state_sha256=_frame_hash(after_frame),
    )
    return branch, after_frame


def _collect_tree(
    *,
    reset_template: Any,
    reset_frame: Any,
    prefix: tuple[ActionSpec, ...],
    expected_pre_hash: str,
    context: tuple[BoundEvent, ...],
    game: str,
    source_split: str,
    seed: int,
    reset_index: int,
    root_index: int,
    path: str,
    depth: int,
    maximum_depth: int,
    selection_counts: Counter[str],
) -> list[BindingPairRecord]:
    if depth >= maximum_depth:
        return []
    node_env, node_frame = replay_prefix(
        reset_template,
        reset_frame,
        prefix,
        expected_pre_state_sha256=expected_pre_hash,
    )
    node_snapshot = snapshot_frame(node_frame)
    if _is_terminal(node_snapshot.game_state):
        return []
    legal = _legal_actions(node_env)
    if len(legal) < 2:
        return []
    observation = _observation(node_frame, legal)
    left_action, right_action = select_branch_actions(
        legal,
        observation,
        selection_counts,
        salt=(f"v4.3:{game}:{seed}:{reset_index}:{root_index}:{path}:{depth}"),
    )
    replay_hash = _frame_hash(node_frame)
    left_env = copy.deepcopy(node_env)
    right_env = copy.deepcopy(node_env)
    left_frame = copy.deepcopy(node_frame)
    right_frame = copy.deepcopy(node_frame)
    left, left_after = _execute_arm(
        env=left_env,
        frame=left_frame,
        action=left_action,
        arm="left",
        game=game,
        source_split=source_split,
        seed=seed,
        reset_index=reset_index,
        step_index=1000 + root_index * 100 + depth * 10 + len(path),
        pre_hash=replay_hash,
    )
    right, right_after = _execute_arm(
        env=right_env,
        frame=right_frame,
        action=right_action,
        arm="right",
        game=game,
        source_split=source_split,
        seed=seed,
        reset_index=reset_index,
        step_index=2000 + root_index * 100 + depth * 10 + len(path),
        pre_hash=replay_hash,
    )
    for branch in (left, right):
        signature = branch.event.binding
        stratum = f"{branch.action.name}|{signature.key('typed')}|{branch.action.key}"
        selection_counts[stratum] += 1
    pair = BindingPairRecord(
        game_id=game,
        source_split=source_split,
        policy_seed=seed,
        reset_index=reset_index,
        root_index=root_index,
        path=path,
        depth=depth,
        context=context,
        expected_pre_state_sha256=expected_pre_hash,
        replay_pre_state_sha256=replay_hash,
        left=left,
        right=right,
    )
    rows = [pair]
    if depth + 1 < maximum_depth:
        for marker, branch, after_frame in (
            ("L", left, left_after),
            ("R", right, right_after),
        ):
            after = snapshot_frame(after_frame)
            if _is_terminal(after.game_state):
                continue
            next_context = (context + (branch.event,))[-8:]
            rows.extend(
                _collect_tree(
                    reset_template=reset_template,
                    reset_frame=reset_frame,
                    prefix=prefix + (branch.action,),
                    expected_pre_hash=branch.post_state_sha256,
                    context=next_context,
                    game=game,
                    source_split=source_split,
                    seed=seed,
                    reset_index=reset_index,
                    root_index=root_index,
                    path=path + marker,
                    depth=depth + 1,
                    maximum_depth=maximum_depth,
                    selection_counts=selection_counts,
                )
            )
    return rows


def _select_base_action(
    legal: Sequence[Any],
    observation: Any,
    counts: Counter[str],
    *,
    salt: str,
) -> Any:
    ranked = _ranked_candidates(legal, observation, counts, salt=salt)
    selected, signature = ranked[0]
    spec = ActionSpec.from_action(selected)
    counts[f"{spec.name}|{signature.key('typed')}|{spec.key}"] += 1
    return selected


def _collect_game(
    *,
    game: str,
    source_split: str,
    root_quota: int,
    seeds: Sequence[int],
    action_budget: int,
    maximum_resets: int,
    tree_depth: int,
    environment_root: Path,
) -> tuple[list[BindingPairRecord], dict[str, Any]]:
    pairs: list[BindingPairRecord] = []
    roots = 0
    selection_counts: Counter[str] = Counter()
    replay_failures = 0
    resets_used = 0
    for reset_index in range(maximum_resets):
        if roots >= root_quota:
            break
        seed = int(seeds[reset_index % len(seeds)])
        env = _make_real_env(game, environment_root)
        try:
            frame = _reset_env(env)
        except ModuleNotFoundError as exc:
            if exc.name != "arcengine":
                raise
            frame = env.step(0)
        reset_template = copy.deepcopy(env)
        reset_frame = copy.deepcopy(frame)
        prefix: list[ActionSpec] = []
        history: list[BoundEvent] = []
        resets_used += 1
        for step_index in range(action_budget):
            snap = snapshot_frame(frame)
            if _is_terminal(snap.game_state):
                break
            legal = _legal_actions(env)
            if not legal:
                break
            observation = _observation(frame, legal)
            if (
                len(history) >= 8
                and (step_index - 8) % 2 == 0
                and roots < root_quota
                and len(legal) >= 2
            ):
                root_index = roots
                try:
                    tree = _collect_tree(
                        reset_template=reset_template,
                        reset_frame=reset_frame,
                        prefix=tuple(prefix),
                        expected_pre_hash=_frame_hash(frame),
                        context=tuple(history[-8:]),
                        game=game,
                        source_split=source_split,
                        seed=seed,
                        reset_index=reset_index,
                        root_index=root_index,
                        path="",
                        depth=0,
                        maximum_depth=tree_depth,
                        selection_counts=selection_counts,
                    )
                except RuntimeError as exc:
                    if "replay" not in str(exc).lower():
                        raise
                    replay_failures += 1
                    raise
                pairs.extend(tree)
                roots += 1
            selected = _select_base_action(
                legal,
                observation,
                selection_counts,
                salt=f"v4.3-base:{game}:{seed}:{reset_index}:{step_index}",
            )
            before = snapshot_frame(frame)
            available = _available(legal)
            after_frame = _step_env_action(env, selected)
            after = snapshot_frame(
                after_frame,
                fallback_available_actions=before.available_actions,
            )
            trace = build_action_target_trace(
                game_id=game,
                source_split=source_split,
                policy_seed=seed,
                reset_index=reset_index,
                step_index=step_index,
                collection_phase="v4_3_base_context",
                available_action_names=available,
                selected_action_name=str(selected.name),
                selected_action_data=dict(selected.action_args),
                frame_before=before.grid,
                frame_after=after.grid,
                game_state_before=before.game_state,
                game_state_after=after.game_state,
                levels_completed_before=before.levels_completed,
                levels_completed_after=after.levels_completed,
            )
            history.append(BoundEvent.from_trace(trace))
            prefix.append(ActionSpec.from_action(selected))
            frame = after_frame
    if roots != root_quota:
        raise RuntimeError(
            f"V4.3 root quota incomplete for {game}: {roots}/{root_quota}"
        )
    return pairs, {
        "game_id": game,
        "source_split": source_split,
        "roots": roots,
        "pairs": len(pairs),
        "arms": 2 * len(pairs),
        "resets_used": resets_used,
        "replay_failures": replay_failures,
        "complete_depth_three_trees": sum(
            int(len({row.path for row in pairs if row.root_index == root}) == 7)
            for root in range(root_quota)
        ),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical(row) + "\n")
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pairs(shard_dir: str | Path, games: Sequence[str]) -> list[BindingPairRecord]:
    rows: list[BindingPairRecord] = []
    for game in games:
        path = Path(shard_dir) / f"{game}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(BindingPairRecord.from_dict(json.loads(line)))
    return rows


def _collection_summary(
    *,
    split: str,
    games: Sequence[str],
    shard_dir: Path,
    reports: Mapping[str, Any],
    frozen: Mapping[str, Any],
) -> dict[str, Any]:
    shards = []
    all_pairs = []
    for game in games:
        path = shard_dir / f"{game}.jsonl"
        pairs = load_pairs(shard_dir, (game,))
        all_pairs.extend(pairs)
        shards.append(
            {
                "game_id": game,
                "path": path.as_posix(),
                "pairs": len(pairs),
                "arms": 2 * len(pairs),
                "sha256": _file_sha256(path),
            }
        )
    windows = [window for pair in all_pairs for window in pair_windows(pair)]
    capacity = _capacity(windows)
    payload: dict[str, Any] = {
        "format_version": COLLECTION_FORMAT_VERSION,
        "status": "COMPLETE",
        "split": split,
        "games": list(games),
        "roots": sum(int(item["roots"]) for item in reports.values()),
        "pairs": len(all_pairs),
        "arms": len(windows),
        "shards": shards,
        "capacity": capacity,
        "game_reports": dict(reports),
        "frozen_manifest_checksum": frozen["manifest_checksum"],
        "outcome_adaptive": False,
        "replay_verified": True,
        "chronological_repeats_retained": True,
        "holdout_opened": False,
        "historical_opened": False,
        "ar25_opened": False,
    }
    payload["report_checksum"] = _checksum(payload)
    return payload


def run_collection(
    *,
    split: str,
    frozen_manifest_path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    environments_dir: str | Path | None = None,
) -> dict[str, Any]:
    frozen = load_frozen_manifest(frozen_manifest_path)
    if split not in {"source_train", "source_validation"}:
        raise ValueError("V4.3 collection split must be source_train/validation")
    destination = Path(output_dir)
    if split == "source_validation":
        projection_path = destination / "projection_freeze.json"
        if not projection_path.exists():
            raise RuntimeError(
                "validation collection blocked until source projection freeze"
            )
        projection = json.loads(projection_path.read_text(encoding="utf-8"))
        if projection.get("status") != "PASS":
            raise RuntimeError(
                "validation collection blocked by failed source preflight"
            )
    games = (
        tuple(frozen["source_train_games"])
        if split == "source_train"
        else tuple(frozen["source_validation_games"])
    )
    config = frozen["collection"][split]
    seeds = (
        tuple(frozen["collection"]["source_seeds"])
        if split == "source_train"
        else tuple(frozen["collection"]["validation_seeds"])
    )
    root = Path(environments_dir) if environments_dir else _env_dir()
    shard_dir = destination / f"{split}_shards"
    reports = {}
    for game in games:
        pairs, report = _collect_game(
            game=game,
            source_split=split,
            root_quota=int(config["roots_per_game"]),
            seeds=seeds,
            action_budget=int(config["action_budget_per_reset"]),
            maximum_resets=int(config["maximum_resets_per_game"]),
            tree_depth=int(frozen["collection"]["tree_depth"]),
            environment_root=root,
        )
        _write_jsonl(
            shard_dir / f"{game}.jsonl",
            (pair.to_dict() for pair in pairs),
        )
        reports[game] = report
    payload = _collection_summary(
        split=split,
        games=games,
        shard_dir=shard_dir,
        reports=reports,
        frozen=frozen,
    )
    _write_json(destination / f"{split}_collection_manifest.json", payload)
    return payload


def _capacity(windows: Sequence[BoundWindow]) -> dict[str, Any]:
    per_label = {}
    for label in TARGET_EFFECTS:
        eligible = [row for row in windows if row.applicable[label]]
        positives = sum(int(row.labels[label]) for row in eligible)
        per_label[label] = {
            "applicable": len(eligible),
            "positives": positives,
            "negatives": len(eligible) - positives,
        }
    return {
        "windows": len(windows),
        "pairs": len({row.pair_id for row in windows}),
        "per_label": per_label,
        "per_game": {
            game: {
                "windows": sum(row.game_id == game for row in windows),
                "pairs": len({row.pair_id for row in windows if row.game_id == game}),
            }
            for game in sorted({row.game_id for row in windows})
        },
    }


@dataclass(frozen=True)
class BoundMechanicRule:
    """A proposed causal rule; observations remain separate evidence."""

    rule_id: str
    action_scope_kind: str
    action_scope_value: str
    binding_projection: str
    binding_key: str
    effect: str
    support: int = 0
    source: str = "structured"

    def __post_init__(self) -> None:
        if self.action_scope_kind not in {"exact", "family", "none"}:
            raise ValueError("unsupported V4.3 action scope")
        if self.binding_projection not in {
            "minimal",
            "relational",
            "typed",
            "any",
        }:
            raise ValueError("unsupported V4.3 binding projection")
        if self.effect not in TARGET_EFFECTS:
            raise ValueError("unsupported V4.3 effect")
        if self.support != 0:
            raise ValueError("V4.3 rules must enter with support=0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BoundMechanicEvidence:
    rule: BoundMechanicRule
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


def _rule_key(
    *,
    action_scope_kind: str,
    action_scope_value: str,
    binding_projection: str,
    binding_key: str,
) -> str:
    return _canonical(
        {
            "action_scope_kind": action_scope_kind,
            "action_scope_value": action_scope_value,
            "binding_projection": binding_projection,
            "binding_key": binding_key,
        }
    )


def _candidate_rule_keys(
    *,
    action_name: str,
    action_family: str,
    binding: BindingSignature,
    projection: str,
    mode: str,
) -> tuple[tuple[str, str, str, str, str], ...]:
    if mode == "template":
        return ()
    if mode == "action_only":
        specs = (("exact", action_name, "any", "any"),)
    elif mode == "no_binding":
        specs = (
            ("exact", action_name, "any", "any"),
            ("family", action_family, "any", "any"),
        )
    elif mode == "binding_only":
        specs = (
            ("none", "*", projection, binding.key(projection)),
            ("none", "*", "minimal", binding.key("minimal")),
            ("none", "*", "any", "any"),
        )
    elif mode == "structured":
        specs = (
            ("exact", action_name, projection, binding.key(projection)),
            ("family", action_family, projection, binding.key(projection)),
            ("exact", action_name, "minimal", binding.key("minimal")),
            ("family", action_family, "minimal", binding.key("minimal")),
            ("exact", action_name, "any", "any"),
            ("family", action_family, "any", "any"),
        )
    else:
        raise ValueError(f"unknown V4.3 model mode: {mode}")
    unique = []
    seen = set()
    for action_kind, action_value, binding_projection, binding_key in specs:
        key = _rule_key(
            action_scope_kind=action_kind,
            action_scope_value=action_value,
            binding_projection=binding_projection,
            binding_key=binding_key,
        )
        if key not in seen:
            unique.append(
                (
                    key,
                    action_kind,
                    action_value,
                    binding_projection,
                    binding_key,
                )
            )
            seen.add(key)
    return tuple(unique)


def fit_priors(windows: Sequence[BoundWindow], projection: str) -> dict[str, Any]:
    counts: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {label: [0, 0] for label in TARGET_EFFECTS}
    )
    global_counts = {label: [0, 0] for label in TARGET_EFFECTS}
    for window in windows:
        for label in TARGET_EFFECTS:
            if window.applicable[label]:
                value = int(window.labels[label])
                global_counts[label][value] += 1
        window_keys = set()
        for mode in ("structured", "no_binding", "action_only", "binding_only"):
            for key, *_ in _candidate_rule_keys(
                action_name=window.query_action_name,
                action_family=window.query_action_family,
                binding=window.query_binding,
                projection=projection,
                mode=mode,
            ):
                window_keys.add(key)
        for key in window_keys:
            for label in TARGET_EFFECTS:
                if window.applicable[label]:
                    value = int(window.labels[label])
                    counts[key][label][value] += 1
    return {
        "projection": projection,
        "counts": {
            key: {label: list(values) for label, values in effects.items()}
            for key, effects in counts.items()
        },
        "global_counts": global_counts,
    }


def _beta_probability(negative: int, positive: int) -> float:
    return float((positive + 1.0) / (negative + positive + 2.0))


def _template_probabilities(window: BoundWindow) -> dict[str, float]:
    binding = window.query_binding
    click = window.query_action_family == "click"
    occupied = binding.kind == "occupied_object"
    free = binding.kind == "free_slot"
    return {
        "target_created": 0.72 if click and free else 0.08,
        "target_removed": 0.58 if click and occupied else 0.06,
        "target_moved": 0.42 if occupied else 0.05,
    }


def score_window(
    window: BoundWindow,
    priors: Mapping[str, Any],
    *,
    projection: str,
    mode: str,
    minimum_local_evidence: int = 2,
    prior_strength: float = 2.0,
) -> tuple[dict[str, float], tuple[BoundMechanicEvidence, ...]]:
    if mode == "template":
        return _template_probabilities(window), ()
    counts = dict(priors["counts"])
    global_counts = dict(priors["global_counts"])
    query_keys = _candidate_rule_keys(
        action_name=window.query_action_name,
        action_family=window.query_action_family,
        binding=window.query_binding,
        projection=projection,
        mode=mode,
    )
    probabilities = {}
    evidence_rows = []
    for label in TARGET_EFFECTS:
        global_negative, global_positive = global_counts[label]
        global_prior = _beta_probability(global_negative, global_positive)
        selected = None
        for (
            key,
            action_kind,
            action_value,
            binding_projection,
            binding_key,
        ) in query_keys:
            source_negative, source_positive = counts.get(key, {}).get(label, (0, 0))
            prior = _beta_probability(source_negative, source_positive)
            local_positive = local_negative = 0
            for event in window.context:
                if not event.applicable[label]:
                    continue
                event_keys = {
                    item[0]
                    for item in _candidate_rule_keys(
                        action_name=event.action_name,
                        action_family=event.action_family,
                        binding=event.binding,
                        projection=projection,
                        mode=mode,
                    )
                }
                if key in event_keys:
                    if event.effects[label]:
                        local_positive += 1
                    else:
                        local_negative += 1
            local_total = local_positive + local_negative
            if local_total >= minimum_local_evidence:
                posterior = (prior_strength * prior + local_positive) / (
                    prior_strength + local_total
                )
                selected = (
                    posterior,
                    action_kind,
                    action_value,
                    binding_projection,
                    binding_key,
                    local_positive,
                    local_negative,
                    prior,
                )
                break
            if selected is None and source_negative + source_positive > 0:
                selected = (
                    prior,
                    action_kind,
                    action_value,
                    binding_projection,
                    binding_key,
                    0,
                    0,
                    prior,
                )
        if selected is None:
            selected = (
                global_prior,
                "none",
                "*",
                "any",
                "any",
                0,
                0,
                global_prior,
            )
        (
            probability,
            action_kind,
            action_value,
            binding_projection,
            binding_key,
            observed_support,
            observed_refutations,
            prior,
        ) = selected
        rule_payload = {
            "action_scope_kind": action_kind,
            "action_scope_value": action_value,
            "binding_projection": binding_projection,
            "binding_key": binding_key,
            "effect": label,
        }
        rule = BoundMechanicRule(
            rule_id=_checksum(rule_payload)[:20],
            effect=label,
            source=mode,
            **{key: value for key, value in rule_payload.items() if key != "effect"},
        )
        probabilities[label] = float(np.clip(probability, 1e-6, 1 - 1e-6))
        evidence_rows.append(
            BoundMechanicEvidence(
                rule=rule,
                observed_support=int(observed_support),
                observed_refutations=int(observed_refutations),
                prior_probability=float(prior),
                posterior_probability=probabilities[label],
            )
        )
    return probabilities, tuple(evidence_rows)


def _matrices(
    windows: Sequence[BoundWindow],
    priors: Mapping[str, Any],
    projection: str,
) -> tuple[dict[str, np.ndarray], dict[str, list[Any]]]:
    matrices: dict[str, list[list[float]]] = {mode: [] for mode in MODEL_MODES}
    evidence: dict[str, list[Any]] = {mode: [] for mode in MODEL_MODES}
    for window in windows:
        for mode in MODEL_MODES:
            probabilities, rows = score_window(
                window,
                priors,
                projection=projection,
                mode=mode,
            )
            matrices[mode].append([probabilities[label] for label in TARGET_EFFECTS])
            evidence[mode].append(rows)
    return (
        {mode: np.asarray(rows, dtype=np.float64) for mode, rows in matrices.items()},
        evidence,
    )


def _targets_masks(
    windows: Sequence[BoundWindow],
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(
        [[int(row.labels[label]) for label in TARGET_EFFECTS] for row in windows],
        dtype=np.int8,
    )
    masks = np.asarray(
        [[int(row.applicable[label]) for label in TARGET_EFFECTS] for row in windows],
        dtype=np.int8,
    )
    return targets, masks


def _ece(targets: np.ndarray, probabilities: np.ndarray) -> float:
    if not len(targets):
        return 0.0
    total = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        selected = (probabilities >= lower) & (
            probabilities < upper if upper < 1.0 else probabilities <= upper
        )
        if np.any(selected):
            total += float(np.mean(selected)) * abs(
                float(np.mean(probabilities[selected]))
                - float(np.mean(targets[selected]))
            )
    return total


def target_metrics(
    targets: np.ndarray,
    masks: np.ndarray,
    probabilities: np.ndarray,
    *,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or {label: 0.5 for label in TARGET_EFFECTS}
    per_label = {}
    for index, label in enumerate(TARGET_EFFECTS):
        selected = masks[:, index].astype(bool)
        truth = targets[selected, index]
        probs = probabilities[selected, index]
        pred = probs >= float(thresholds[label])
        tp = int(np.sum((pred == 1) & (truth == 1)))
        fp = int(np.sum((pred == 1) & (truth == 0)))
        fn = int(np.sum((pred == 0) & (truth == 1)))
        denominator = 2 * tp + fp + fn
        per_label[label] = {
            "applicable": len(truth),
            "positives": int(np.sum(truth)),
            "negatives": int(len(truth) - np.sum(truth)),
            "threshold": float(thresholds[label]),
            "f1": float(2 * tp / denominator) if denominator else 0.0,
            "brier": (float(np.mean((probs - truth) ** 2)) if len(truth) else 0.0),
            "ece": _ece(truth, probs),
        }
    return {
        "macro_f1": float(np.mean([item["f1"] for item in per_label.values()])),
        "macro_brier": float(np.mean([item["brier"] for item in per_label.values()])),
        "macro_ece": float(np.mean([item["ece"] for item in per_label.values()])),
        "per_label": per_label,
    }


def _brier_skill(model: Mapping[str, Any], baseline: Mapping[str, Any]) -> float:
    denominator = float(baseline["macro_brier"])
    if denominator <= 0:
        return 0.0
    return float((denominator - float(model["macro_brier"])) / denominator)


@dataclass(frozen=True)
class CalibrationBundle:
    projection: str
    parameters: Mapping[str, Mapping[str, Mapping[str, float]]]
    thresholds: Mapping[str, Mapping[str, float]]
    calibration_checksum: str = ""

    def __post_init__(self) -> None:
        if not self.calibration_checksum:
            object.__setattr__(
                self,
                "calibration_checksum",
                _checksum(
                    {
                        "projection": self.projection,
                        "parameters": self.parameters,
                        "thresholds": self.thresholds,
                    }
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "projection": self.projection,
            "parameters": self.parameters,
            "thresholds": self.thresholds,
            "calibration_checksum": self.calibration_checksum,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CalibrationBundle:
        result = cls(
            projection=str(payload["projection"]),
            parameters=dict(payload["parameters"]),
            thresholds=dict(payload["thresholds"]),
            calibration_checksum=str(payload.get("calibration_checksum", "")),
        )
        expected = _checksum(
            {
                "projection": result.projection,
                "parameters": result.parameters,
                "thresholds": result.thresholds,
            }
        )
        if result.calibration_checksum != expected:
            raise ValueError("V4.3 calibration checksum mismatch")
        return result


def _logo_raw_matrices(
    windows: Sequence[BoundWindow], projection: str
) -> dict[str, np.ndarray]:
    matrices = {
        mode: np.zeros((len(windows), len(TARGET_EFFECTS)), dtype=np.float64)
        for mode in MODEL_MODES
    }
    games = sorted({row.game_id for row in windows})
    for held_out in games:
        train = [row for row in windows if row.game_id != held_out]
        priors = fit_priors(train, projection)
        indices = [
            index for index, row in enumerate(windows) if row.game_id == held_out
        ]
        test = [windows[index] for index in indices]
        fold_matrices, _ = _matrices(test, priors, projection)
        for mode in MODEL_MODES:
            matrices[mode][indices] = fold_matrices[mode]
    return matrices


def fit_source_calibration(
    windows: Sequence[BoundWindow], projection: str
) -> tuple[CalibrationBundle, dict[str, np.ndarray]]:
    targets, masks = _targets_masks(windows)
    logo = _logo_raw_matrices(windows, projection)
    parameters: dict[str, Any] = {}
    thresholds: dict[str, Any] = {}
    for mode, matrix in logo.items():
        parameters[mode] = {}
        thresholds[mode] = {}
        for index, label in enumerate(TARGET_EFFECTS):
            selected = masks[:, index].astype(bool)
            parameter = _fit_platt(matrix[selected, index], targets[selected, index])
            calibrated = _apply_parameter(matrix[selected, index], parameter)
            parameters[mode][label] = parameter
            thresholds[mode][label] = _select_threshold(
                calibrated, targets[selected, index]
            )
    return CalibrationBundle(
        projection=projection,
        parameters=parameters,
        thresholds=thresholds,
    ), logo


def apply_calibration(
    matrix: np.ndarray, bundle: CalibrationBundle, mode: str
) -> np.ndarray:
    result = np.zeros_like(matrix, dtype=np.float64)
    for index, label in enumerate(TARGET_EFFECTS):
        result[:, index] = _apply_parameter(
            matrix[:, index], bundle.parameters[mode][label]
        )
    return result


def _identity_diagnostic(
    windows: Sequence[BoundWindow], projection: str
) -> dict[str, Any]:
    labels = [row.game_id for row in windows]
    action_rows = [
        {
            f"action:{row.query_action_name}": 1,
            f"family:{row.query_action_family}": 1,
        }
        for row in windows
    ]
    binding_rows = []
    for row in windows:
        features = dict(action_rows[len(binding_rows)])
        for key, value in row.query_binding.model_view(projection).items():
            features[f"binding:{key}:{value}"] = 1
        binding_rows.append(features)
    action = _categorical_identity_probe(action_rows, labels)
    binding = _categorical_identity_probe(binding_rows, labels)
    return {
        "action_only": action,
        "action_binding": binding,
        "gain": float(binding["accuracy"] - action["accuracy"]),
    }


def _projection_score(
    windows: Sequence[BoundWindow],
    projection: str,
    gates: Mapping[str, Any],
) -> tuple[dict[str, Any], CalibrationBundle]:
    bundle, raw = fit_source_calibration(windows, projection)
    targets, masks = _targets_masks(windows)
    calibrated = {
        mode: apply_calibration(matrix, bundle, mode) for mode, matrix in raw.items()
    }
    metrics = {
        mode: target_metrics(
            targets,
            masks,
            matrix,
            thresholds=bundle.thresholds[mode],
        )
        for mode, matrix in calibrated.items()
    }
    stronger = min(BASELINE_MODES, key=lambda mode: metrics[mode]["macro_brier"])
    skill = _brier_skill(metrics["structured"], metrics[stronger])
    f1_gain = metrics["structured"]["macro_f1"] - metrics[stronger]["macro_f1"]
    identity = _identity_diagnostic(windows, projection)
    passed = {
        "minimum_source_macro_brier_skill": skill
        >= float(gates["minimum_macro_brier_skill"]),
        "minimum_source_macro_f1_gain": f1_gain
        >= float(gates["minimum_macro_f1_gain"]),
        "maximum_identity_gain": identity["gain"]
        <= float(gates["maximum_identity_gain_over_action"]),
        "maximum_source_macro_ece": metrics["structured"]["macro_ece"]
        <= float(gates["maximum_macro_ece"]),
    }
    return (
        {
            "projection": projection,
            "metrics": metrics,
            "stronger_baseline": stronger,
            "brier_skill": skill,
            "macro_f1_gain": f1_gain,
            "identity": identity,
            "gates": passed,
            "status": "PASS" if all(passed.values()) else "FAIL_CLOSED",
        },
        bundle,
    )


def run_source_preflight(
    *,
    frozen_manifest_path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    frozen = load_frozen_manifest(frozen_manifest_path)
    destination = Path(output_dir)
    pairs = load_pairs(
        destination / "source_train_shards",
        tuple(frozen["source_train_games"]),
    )
    windows = [window for pair in pairs for window in pair_windows(pair)]
    for projection in PROJECTION_LADDER:
        for window in windows:
            validate_model_view(window, projection)
    capacity = _capacity(windows)
    gates_cfg = frozen["binding_gates"]
    capacity_gates = {
        "minimum_source_pairs": capacity["pairs"]
        >= int(gates_cfg["minimum_source_pairs"]),
        "source_label_capacity": all(
            item["positives"] >= int(gates_cfg["minimum_source_positives_per_effect"])
            and item["negatives"]
            >= int(gates_cfg["minimum_source_negatives_per_effect"])
            for item in capacity["per_label"].values()
        ),
        "strict_json_validity": True,
        "compiler_grounding": True,
        "replay_integrity": all(
            pair.expected_pre_state_sha256
            == pair.replay_pre_state_sha256
            == pair.left.replay_pre_state_sha256
            == pair.right.replay_pre_state_sha256
            for pair in pairs
        ),
    }
    projection_results = {}
    bundles = {}
    for projection in PROJECTION_LADDER:
        result, bundle = _projection_score(windows, projection, gates_cfg)
        projection_results[projection] = result
        bundles[projection] = bundle
    eligible = [
        projection
        for projection in PROJECTION_LADDER
        if projection_results[projection]["status"] == "PASS"
    ]
    selected = None
    if eligible and all(capacity_gates.values()):
        best_skill = max(projection_results[item]["brier_skill"] for item in eligible)
        selected = next(
            item
            for item in PROJECTION_LADDER
            if item in eligible
            and best_skill - projection_results[item]["brier_skill"]
            <= float(frozen["projection"]["simplicity_tie_margin"])
        )
    status = "PASS" if selected else "FAIL_CLOSED"
    payload: dict[str, Any] = {
        "format_version": PREFLIGHT_FORMAT_VERSION,
        "status": status,
        "frozen_manifest_checksum": frozen["manifest_checksum"],
        "capacity": capacity,
        "capacity_gates": capacity_gates,
        "projection_results": projection_results,
        "selected_projection": selected,
        "selection_rule": ("best_source_LOGO_brier_skill; within 0.005 choose simpler"),
        "source_only": True,
        "validation_opened": False,
        "holdout_opened": False,
        "historical_opened": False,
        "ar25_opened": False,
    }
    payload["preflight_checksum"] = _checksum(payload)
    _write_json(destination / "source_train_preflight.json", payload)
    freeze = {
        "format_version": "sage12-bound-projection-freeze-v4.3",
        "status": status,
        "selected_projection": selected,
        "preflight_checksum": payload["preflight_checksum"],
        "frozen_before_validation_collection": True,
    }
    freeze["projection_freeze_checksum"] = _checksum(freeze)
    _write_json(destination / "projection_freeze.json", freeze)
    if selected:
        _write_json(
            destination / "calibration.json",
            bundles[selected].to_dict(),
        )
        priors = fit_priors(windows, selected)
        priors["source_pairs"] = len(pairs)
        priors["source_windows"] = len(windows)
        priors["priors_checksum"] = _checksum(priors)
        _write_json(destination / "source_priors.json", priors)
    return payload


def binding_swap_control(
    windows: Sequence[BoundWindow],
) -> tuple[list[BoundWindow], float]:
    by_pair: dict[str, list[BoundWindow]] = defaultdict(list)
    for window in windows:
        by_pair[window.pair_id].append(window)
    swapped = []
    changed = 0
    for window in windows:
        peers = by_pair[window.pair_id]
        if len(peers) != 2:
            raise ValueError("binding swap requires complete pairs")
        peer = peers[0] if peers[1].arm == window.arm else peers[1]
        changed += int(peer.query_binding != window.query_binding)
        swapped.append(window.with_query_binding(peer.query_binding))
    return swapped, changed / max(1, len(windows))


def outcome_shuffle_control(
    windows: Sequence[BoundWindow],
) -> list[BoundWindow]:
    shuffled = []
    for window in windows:
        context = list(window.context)
        outcomes = [(event.effects, event.applicable) for event in context]
        offset = 1 + int(window.pair_id[:8], 16) % max(1, len(context) - 1)
        rotated = outcomes[offset:] + outcomes[:offset]
        events = tuple(
            BoundEvent(
                action_name=event.action_name,
                action_family=event.action_family,
                binding=event.binding,
                effects=effects,
                applicable=applicable,
            )
            for event, (effects, applicable) in zip(context, rotated)
        )
        shuffled.append(
            BoundWindow(
                pair_id=window.pair_id,
                arm=window.arm,
                game_id=window.game_id,
                source_split=window.source_split,
                root_key=window.root_key,
                path=window.path,
                context=events,
                query_action_name=window.query_action_name,
                query_action_family=window.query_action_family,
                query_binding=window.query_binding,
                labels=window.labels,
                applicable=window.applicable,
            )
        )
    return shuffled


def _discordant_pair_accuracy(
    windows: Sequence[BoundWindow], probabilities: np.ndarray
) -> dict[str, Any]:
    indices: dict[str, list[int]] = defaultdict(list)
    for index, window in enumerate(windows):
        indices[window.pair_id].append(index)
    correct = total = 0.0
    per_label = {}
    for label_index, label in enumerate(TARGET_EFFECTS):
        label_correct = label_total = 0.0
        for pair_indices in indices.values():
            if len(pair_indices) != 2:
                continue
            left, right = pair_indices
            if not (
                windows[left].applicable[label] and windows[right].applicable[label]
            ):
                continue
            left_target = int(windows[left].labels[label])
            right_target = int(windows[right].labels[label])
            if left_target == right_target:
                continue
            positive = left if left_target else right
            negative = right if left_target else left
            delta = (
                probabilities[positive, label_index]
                - probabilities[negative, label_index]
            )
            score = 1.0 if delta > 0 else 0.5 if delta == 0 else 0.0
            label_correct += score
            label_total += 1
        per_label[label] = {
            "discordant_pairs": int(label_total),
            "accuracy": (float(label_correct / label_total) if label_total else 0.0),
        }
        correct += label_correct
        total += label_total
    return {
        "discordant_pairs": int(total),
        "accuracy": float(correct / total) if total else 0.0,
        "per_label": per_label,
    }


def _pair_accuracy_subset(
    windows: Sequence[BoundWindow],
    probabilities: np.ndarray,
    predicate: Any,
) -> dict[str, Any]:
    selected_ids = {row.pair_id for row in windows if predicate(row)}
    selected_indices = [
        index for index, row in enumerate(windows) if row.pair_id in selected_ids
    ]
    subset = [windows[index] for index in selected_indices]
    return _discordant_pair_accuracy(subset, probabilities[selected_indices])


def _bootstrap_pair_accuracy_gain(
    windows: Sequence[BoundWindow],
    model: np.ndarray,
    baseline: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    pair_ids = sorted({row.pair_id for row in windows})
    pair_gains = []
    for pair_id in pair_ids:
        indices = [index for index, row in enumerate(windows) if row.pair_id == pair_id]
        subset = [windows[index] for index in indices]
        pair_gains.append(
            _discordant_pair_accuracy(subset, model[indices])["accuracy"]
            - _discordant_pair_accuracy(subset, baseline[indices])["accuracy"]
        )
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        sampled = rng.choice(pair_gains, size=len(pair_gains), replace=True)
        values.append(float(np.mean(sampled)))
    return {
        "mean": float(np.mean(values)),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def _output_contract(
    windows: Sequence[BoundWindow],
    evidence: Sequence[Sequence[BoundMechanicEvidence]],
    *,
    projection: str,
) -> dict[str, float]:
    emitted = sum(len(rows) for rows in evidence)
    support_zero = sum(
        int(item.rule.support == 0) for rows in evidence for item in rows
    )
    grounded = 0
    for window, rows in zip(windows, evidence):
        query_keys = {
            item[0]
            for item in _candidate_rule_keys(
                action_name=window.query_action_name,
                action_family=window.query_action_family,
                binding=window.query_binding,
                projection=projection,
                mode="structured",
            )
        }
        for item in rows:
            rule = item.rule
            key = _rule_key(
                action_scope_kind=rule.action_scope_kind,
                action_scope_value=rule.action_scope_value,
                binding_projection=rule.binding_projection,
                binding_key=rule.binding_key,
            )
            grounded += int(key in query_keys or rule.action_scope_kind == "none")
    return {
        "strict_json_validity": 1.0,
        "compiler_grounding_rate": grounded / max(1, emitted),
        "support_zero_rate": support_zero / max(1, emitted),
        "emitted_rules": emitted,
    }


def run_binding_evaluation(
    *,
    frozen_manifest_path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    frozen = load_frozen_manifest(frozen_manifest_path)
    destination = Path(output_dir)
    preflight = json.loads(
        (destination / "source_train_preflight.json").read_text(encoding="utf-8")
    )
    if preflight.get("status") != "PASS":
        payload: dict[str, Any] = {
            "format_version": BINDING_RESULT_FORMAT_VERSION,
            "status": "SKIPPED_SOURCE_PREFLIGHT",
            "reason": "source_preflight_did_not_pass_all_frozen_gates",
            "frozen_manifest_checksum": frozen["manifest_checksum"],
            "preflight_checksum": preflight.get("preflight_checksum"),
            "validation_opened": False,
            "binding_model_fitted": False,
            "world_model_fit_authorized": False,
            "qwen_fit_authorized": False,
            "gnn_fit_authorized": False,
            "ebm_fit_authorized": False,
            "controller_authorized": False,
        }
        payload["result_checksum"] = _checksum(payload)
        _write_json(destination / "binding_result.json", payload)
        return payload
    projection = str(preflight["selected_projection"])
    bundle = CalibrationBundle.from_dict(
        json.loads((destination / "calibration.json").read_text(encoding="utf-8"))
    )
    priors = json.loads(
        (destination / "source_priors.json").read_text(encoding="utf-8")
    )
    pairs = load_pairs(
        destination / "source_validation_shards",
        tuple(frozen["source_validation_games"]),
    )
    windows = [window for pair in pairs for window in pair_windows(pair)]
    for window in windows:
        validate_model_view(window, projection)
    targets, masks = _targets_masks(windows)
    raw, evidence = _matrices(windows, priors, projection)
    calibrated = {
        mode: apply_calibration(matrix, bundle, mode) for mode, matrix in raw.items()
    }
    metrics = {
        mode: target_metrics(
            targets,
            masks,
            matrix,
            thresholds=bundle.thresholds[mode],
        )
        for mode, matrix in calibrated.items()
    }
    stronger = str(preflight["projection_results"][projection]["stronger_baseline"])
    skill = _brier_skill(metrics["structured"], metrics[stronger])
    f1_gain = metrics["structured"]["macro_f1"] - metrics[stronger]["macro_f1"]
    swapped_windows, changed_rate = binding_swap_control(windows)
    swapped_raw, _ = _matrices(swapped_windows, priors, projection)
    swapped = apply_calibration(swapped_raw["structured"], bundle, "structured")
    swapped_metrics = target_metrics(
        targets,
        masks,
        swapped,
        thresholds=bundle.thresholds["structured"],
    )
    swapped_skill = _brier_skill(swapped_metrics, metrics[stronger])
    binding_drop = skill - swapped_skill
    outcome_windows = outcome_shuffle_control(windows)
    outcome_raw, _ = _matrices(outcome_windows, priors, projection)
    outcome = apply_calibration(outcome_raw["structured"], bundle, "structured")
    outcome_metrics = target_metrics(
        targets,
        masks,
        outcome,
        thresholds=bundle.thresholds["structured"],
    )
    outcome_drop = skill - _brier_skill(outcome_metrics, metrics[stronger])
    pair_model = _discordant_pair_accuracy(windows, calibrated["structured"])
    pair_baseline = _discordant_pair_accuracy(windows, calibrated[stronger])
    pair_gain = pair_model["accuracy"] - pair_baseline["accuracy"]
    bootstrap = _bootstrap_pair_accuracy_gain(
        windows,
        calibrated["structured"],
        calibrated[stronger],
        samples=int(frozen["evaluation"]["bootstrap_samples"]),
        seed=int(frozen["evaluation"]["random_seed"]),
    )
    per_game = {}
    for game in frozen["source_validation_games"]:
        selected = np.asarray([row.game_id == game for row in windows], dtype=bool)
        model_metrics = target_metrics(
            targets[selected],
            masks[selected],
            calibrated["structured"][selected],
            thresholds=bundle.thresholds["structured"],
        )
        baseline_metrics = target_metrics(
            targets[selected],
            masks[selected],
            calibrated[stronger][selected],
            thresholds=bundle.thresholds[stronger],
        )
        per_game[game] = {
            "windows": int(np.sum(selected)),
            "structured": model_metrics,
            "baseline": baseline_metrics,
            "brier_skill": _brier_skill(model_metrics, baseline_metrics),
        }
    sc25_ids = {}
    for pair in pairs:
        if (
            pair.game_id == "sc25"
            and pair.left.action.name == pair.right.action.name
            and pair.left.event.binding != pair.right.event.binding
        ):
            sc25_ids[pair.pair_digest] = pair
    sc25_model = _pair_accuracy_subset(
        windows,
        calibrated["structured"],
        lambda row: row.pair_id in sc25_ids,
    )
    sc25_baseline = _pair_accuracy_subset(
        windows,
        calibrated[stronger],
        lambda row: row.pair_id in sc25_ids,
    )
    sc25_gain = sc25_model["accuracy"] - sc25_baseline["accuracy"]
    capacity = _capacity(windows)
    output_contract = _output_contract(
        windows, evidence["structured"], projection=projection
    )
    gates_cfg = frozen["binding_gates"]
    gates = {
        "strict_json_validity": output_contract["strict_json_validity"] == 1.0,
        "compiler_grounding": output_contract["compiler_grounding_rate"] == 1.0,
        "support_zero": output_contract["support_zero_rate"] == 1.0,
        "minimum_validation_pairs": capacity["pairs"]
        >= int(gates_cfg["minimum_validation_pairs"]),
        "validation_label_capacity": all(
            item["positives"]
            >= int(gates_cfg["minimum_validation_positives_per_effect"])
            and item["negatives"]
            >= int(gates_cfg["minimum_validation_negatives_per_effect"])
            for item in capacity["per_label"].values()
        ),
        "minimum_macro_brier_skill": skill
        >= float(gates_cfg["minimum_macro_brier_skill"]),
        "minimum_macro_f1_gain": f1_gain >= float(gates_cfg["minimum_macro_f1_gain"]),
        "minimum_binding_swap_skill_drop": binding_drop
        >= float(gates_cfg["minimum_binding_swap_skill_drop"]),
        "minimum_discordant_pair_accuracy_gain": pair_gain
        >= float(gates_cfg["minimum_discordant_pair_accuracy_gain"]),
        "discordant_bootstrap_lower_positive": bootstrap["lower_95"] > 0.0,
        "every_game_nonnegative": all(
            item["brier_skill"] >= 0.0 for item in per_game.values()
        ),
        "maximum_macro_ece": metrics["structured"]["macro_ece"]
        <= float(gates_cfg["maximum_macro_ece"]),
        "maximum_identity_gain": preflight["projection_results"][projection][
            "identity"
        ]["gain"]
        <= float(gates_cfg["maximum_identity_gain_over_action"]),
        "same_action_different_target_sc25_positive": (
            len(sc25_ids) > 0 and sc25_gain > 0.0
        ),
        "source_preflight_passed": preflight["status"] == "PASS",
    }
    passed = all(gates.values())
    prediction_rows = []
    for index, window in enumerate(windows):
        prediction_rows.append(
            {
                "pair_id": window.pair_id,
                "arm": window.arm,
                "game_id": window.game_id,
                "root_key": window.root_key,
                "path": window.path,
                "query": window.model_view(projection)["query"],
                "labels": dict(window.labels),
                "applicable": dict(window.applicable),
                "probabilities": {
                    label: float(calibrated["structured"][index, label_index])
                    for label_index, label in enumerate(TARGET_EFFECTS)
                },
                "evidence": [item.to_dict() for item in evidence["structured"][index]],
            }
        )
    _write_jsonl(destination / "binding_predictions.jsonl", prediction_rows)
    payload: dict[str, Any] = {
        "format_version": BINDING_RESULT_FORMAT_VERSION,
        "status": "PASS" if passed else "FAIL_CLOSED",
        "frozen_manifest_checksum": frozen["manifest_checksum"],
        "preflight_checksum": preflight["preflight_checksum"],
        "calibration_checksum": bundle.calibration_checksum,
        "projection": projection,
        "capacity": capacity,
        "metrics": metrics,
        "stronger_baseline": stronger,
        "macro_brier_skill": skill,
        "macro_f1_gain": f1_gain,
        "binding_swap": {
            "changed_rate": changed_rate,
            "metrics": swapped_metrics,
            "skill": swapped_skill,
            "skill_drop": binding_drop,
        },
        "outcome_shuffle": {
            "metrics": outcome_metrics,
            "skill_drop": outcome_drop,
        },
        "discordant_pairs": {
            "structured": pair_model,
            "baseline": pair_baseline,
            "accuracy_gain": pair_gain,
            "bootstrap": bootstrap,
        },
        "same_action_different_target_sc25": {
            "pairs": len(sc25_ids),
            "structured": sc25_model,
            "baseline": sc25_baseline,
            "accuracy_gain": sc25_gain,
        },
        "per_game": per_game,
        "output_contract": output_contract,
        "gates": gates,
        "world_model_fit_authorized": passed,
        "qwen_fit_authorized": False,
        "gnn_fit_authorized": False,
        "ebm_fit_authorized": False,
        "controller_authorized": False,
        "firewall": {
            "source_only": True,
            "validation_tuning": False,
            "holdout_opened": False,
            "historical_opened": False,
            "ar25_opened": False,
        },
    }
    payload["result_checksum"] = _checksum(payload)
    _write_json(destination / "binding_result.json", payload)
    return payload


@dataclass(frozen=True)
class AbstractBoundState:
    """Small identity-free state used by the V4.3 semantic rollouts."""

    slot_occupancy: Mapping[str, bool]

    @staticmethod
    def slot_key(binding: BindingSignature) -> str:
        return _canonical(
            {
                "action_family": binding.action_family,
                "requested_direction": binding.requested_direction,
                "actor_relation": binding.actor_relation,
                "actor_relative_direction": binding.actor_relative_direction,
                "target_area_bucket": binding.target_area_bucket,
                "target_aspect_bucket": binding.target_aspect_bucket,
                "target_affordance": binding.target_affordance,
            }
        )

    def resolve(self, binding: BindingSignature) -> BindingSignature:
        key = self.slot_key(binding)
        occupied = bool(self.slot_occupancy.get(key, binding.occupied))
        payload = binding.to_dict()
        payload["occupied"] = occupied
        if binding.kind != "targetless":
            payload["kind"] = "occupied_object" if occupied else "free_slot"
            payload["path_status"] = "blocked" if occupied else "open"
        return BindingSignature.from_dict(payload)

    def update(
        self,
        binding: BindingSignature,
        effects: Mapping[str, bool],
    ) -> AbstractBoundState:
        occupancy = dict(self.slot_occupancy)
        key = self.slot_key(binding)
        current = bool(occupancy.get(key, binding.occupied))
        if effects["target_created"]:
            current = True
        if effects["target_removed"]:
            current = False
        if effects["target_moved"]:
            current = True
        occupancy[key] = current
        return AbstractBoundState(occupancy)


@dataclass(frozen=True)
class TrajectoryExample:
    game_id: str
    root_key: str
    branch_path: str
    initial_context: tuple[BoundEvent, ...]
    queries: tuple[BoundEvent, ...]
    swapped_bindings: tuple[BindingSignature, ...]

    @property
    def actual_sequence(self) -> tuple[tuple[bool, ...], ...]:
        return tuple(
            tuple(bool(event.effects[label]) for label in TARGET_EFFECTS)
            for event in self.queries
        )

    @property
    def productive(self) -> bool:
        return any(value for step in self.actual_sequence for value in step)


def build_trajectory_examples(
    pairs: Sequence[BindingPairRecord],
) -> list[TrajectoryExample]:
    roots: dict[str, dict[str, BindingPairRecord]] = defaultdict(dict)
    for pair in pairs:
        roots[pair.root_key][pair.path] = pair
    examples = []
    for root_key, tree in sorted(roots.items()):
        if set(tree) != {"", "L", "R", "LL", "LR", "RL", "RR"}:
            continue
        for markers in itertools.product("LR", repeat=3):
            prefix = ""
            queries = []
            swaps = []
            for marker in markers:
                pair = tree[prefix]
                selected = pair.left if marker == "L" else pair.right
                counterpart = pair.right if marker == "L" else pair.left
                queries.append(selected.event)
                swaps.append(counterpart.event.binding)
                prefix += marker
            examples.append(
                TrajectoryExample(
                    game_id=tree[""].game_id,
                    root_key=root_key,
                    branch_path="".join(markers),
                    initial_context=tree[""].context,
                    queries=tuple(queries),
                    swapped_bindings=tuple(swaps),
                )
            )
    return examples


def _applicability_for(binding: BindingSignature) -> dict[str, bool]:
    targetable = binding.kind != "targetless"
    occupied = binding.kind == "occupied_object"
    return {
        "target_created": targetable,
        "target_removed": occupied,
        "target_moved": occupied,
    }


def _effect_combinations(
    binding: BindingSignature,
) -> Iterable[dict[str, bool]]:
    applicable = _applicability_for(binding)
    options = [
        (False, True) if applicable[label] else (False,) for label in TARGET_EFFECTS
    ]
    for values in itertools.product(*options):
        effects = dict(zip(TARGET_EFFECTS, values))
        if effects["target_created"] and binding.kind != "free_slot":
            continue
        if (
            effects["target_removed"] or effects["target_moved"]
        ) and binding.kind != "occupied_object":
            continue
        yield effects


def beam_rollout(
    *,
    initial_context: Sequence[BoundEvent],
    queries: Sequence[BoundEvent],
    priors: Mapping[str, Any],
    calibration: CalibrationBundle,
    projection: str,
    mode: str,
    beam_width: int,
    binding_override: Sequence[BindingSignature] | None = None,
) -> dict[str, Any]:
    initial_slots = {
        AbstractBoundState.slot_key(event.binding): event.binding.occupied
        for event in initial_context
        if event.binding.kind != "targetless"
    }
    # (log probability, sequence, context, abstract state)
    beams: list[
        tuple[
            float,
            tuple[tuple[bool, ...], ...],
            tuple[BoundEvent, ...],
            AbstractBoundState,
        ]
    ] = [
        (
            0.0,
            (),
            tuple(initial_context[-8:]),
            AbstractBoundState(initial_slots),
        )
    ]
    for step_index, query in enumerate(queries):
        expanded = []
        for log_probability, sequence, context, state in beams:
            supplied = (
                binding_override[step_index]
                if binding_override is not None
                else query.binding
            )
            binding = state.resolve(supplied)
            applicable = _applicability_for(binding)
            window = BoundWindow(
                pair_id="world-model",
                arm="left",
                game_id="model-view",
                source_split="source_validation",
                root_key="world-model",
                path="",
                context=context,
                query_action_name=query.action_name,
                query_action_family=query.action_family,
                query_binding=binding,
                labels={label: False for label in TARGET_EFFECTS},
                applicable=applicable,
            )
            raw, _ = score_window(
                window,
                priors,
                projection=projection,
                mode=mode,
            )
            matrix = np.asarray(
                [[raw[label] for label in TARGET_EFFECTS]],
                dtype=np.float64,
            )
            calibrated = apply_calibration(matrix, calibration, mode)[0]
            for effects in _effect_combinations(binding):
                branch_log_probability = log_probability
                for label_index, label in enumerate(TARGET_EFFECTS):
                    if not applicable[label]:
                        continue
                    probability = float(calibrated[label_index])
                    branch_log_probability += math.log(
                        probability if effects[label] else 1.0 - probability
                    )
                event = BoundEvent(
                    action_name=query.action_name,
                    action_family=query.action_family,
                    binding=binding,
                    effects=effects,
                    applicable=applicable,
                )
                expanded.append(
                    (
                        branch_log_probability,
                        sequence + (tuple(effects[label] for label in TARGET_EFFECTS),),
                        (context + (event,))[-8:],
                        state.update(binding, effects),
                    )
                )
        expanded.sort(key=lambda item: (-item[0], item[1]))
        beams = expanded[:beam_width]
    if not beams:
        return {"sequences": (), "probabilities": (), "marginals": ()}
    log_values = np.asarray([item[0] for item in beams], dtype=np.float64)
    weights = np.exp(log_values - np.max(log_values))
    weights /= np.sum(weights)
    marginals = np.zeros((len(queries), len(TARGET_EFFECTS)), dtype=np.float64)
    for weight, beam in zip(weights, beams):
        for step_index, step in enumerate(beam[1]):
            marginals[step_index] += weight * np.asarray(step, dtype=float)
    return {
        "sequences": tuple(item[1] for item in beams),
        "probabilities": tuple(float(value) for value in weights),
        "marginals": marginals,
    }


def _world_metrics(
    examples: Sequence[TrajectoryExample],
    *,
    priors: Mapping[str, Any],
    calibration: CalibrationBundle,
    projection: str,
    mode: str,
    beam_width: int,
    swap_binding: bool = False,
) -> dict[str, Any]:
    productive = recalled = 0
    target_rows = []
    probability_rows = []
    mask_rows = []
    per_game_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    output_rows = []
    for example in examples:
        rollout = beam_rollout(
            initial_context=example.initial_context,
            queries=example.queries,
            priors=priors,
            calibration=calibration,
            projection=projection,
            mode=mode,
            beam_width=beam_width,
            binding_override=(example.swapped_bindings if swap_binding else None),
        )
        hit = example.actual_sequence in rollout["sequences"]
        if example.productive:
            productive += 1
            recalled += int(hit)
            per_game_counts[example.game_id][0] += int(hit)
            per_game_counts[example.game_id][1] += 1
        marginals = np.asarray(rollout["marginals"], dtype=np.float64)
        for step_index, event in enumerate(example.queries):
            target_rows.append([int(event.effects[label]) for label in TARGET_EFFECTS])
            probability_rows.append(marginals[step_index])
            mask_rows.append([int(event.applicable[label]) for label in TARGET_EFFECTS])
        output_rows.append(
            {
                "game_id": example.game_id,
                "root_key": example.root_key,
                "branch_path": example.branch_path,
                "productive": example.productive,
                "top8_hit": hit,
                "actual_sequence": example.actual_sequence,
                "top_sequences": rollout["sequences"],
                "top_probabilities": rollout["probabilities"],
            }
        )
    targets = np.asarray(target_rows, dtype=np.int8)
    probabilities = np.asarray(probability_rows, dtype=np.float64)
    masks = np.asarray(mask_rows, dtype=np.int8)
    metrics = target_metrics(targets, masks, probabilities)
    return {
        "productive_trajectories": productive,
        "productive_top8_recalled": recalled,
        "productive_effect_recall_at_8": recalled / max(1, productive),
        "horizon_metrics": metrics,
        "per_game_recall_at_8": {
            game: correct / max(1, total)
            for game, (correct, total) in per_game_counts.items()
        },
        "rows": output_rows,
        "_targets": targets,
        "_probabilities": probabilities,
        "_masks": masks,
    }


def run_world_model_evaluation(
    *,
    frozen_manifest_path: str | Path = DEFAULT_FROZEN_MANIFEST_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    frozen = load_frozen_manifest(frozen_manifest_path)
    destination = Path(output_dir)
    binding = json.loads(
        (destination / "binding_result.json").read_text(encoding="utf-8")
    )
    if binding.get("status") != "PASS" or not binding.get(
        "world_model_fit_authorized", False
    ):
        payload: dict[str, Any] = {
            "format_version": WORLD_MODEL_RESULT_FORMAT_VERSION,
            "status": "SKIPPED_FAIL_CLOSED",
            "reason": "binding_model_did_not_pass_all_frozen_gates",
            "binding_result_checksum": binding.get("result_checksum"),
            "world_model_fitted": False,
            "ebm_fit_authorized": False,
            "controller_authorized": False,
        }
        payload["result_checksum"] = _checksum(payload)
        _write_json(destination / "world_model_result.json", payload)
        return payload
    projection = str(binding["projection"])
    calibration = CalibrationBundle.from_dict(
        json.loads((destination / "calibration.json").read_text(encoding="utf-8"))
    )
    priors = json.loads(
        (destination / "source_priors.json").read_text(encoding="utf-8")
    )
    pairs = load_pairs(
        destination / "source_validation_shards",
        tuple(frozen["source_validation_games"]),
    )
    examples = build_trajectory_examples(pairs)
    beam_width = int(frozen["world_model"]["beam_width"])
    structured = _world_metrics(
        examples,
        priors=priors,
        calibration=calibration,
        projection=projection,
        mode="structured",
        beam_width=beam_width,
    )
    baseline_modes = ("no_binding", "action_only", "binding_only", "template")
    baselines = {
        mode: _world_metrics(
            examples,
            priors=priors,
            calibration=calibration,
            projection=projection,
            mode=mode,
            beam_width=beam_width,
        )
        for mode in baseline_modes
    }
    stronger = min(
        baseline_modes,
        key=lambda mode: baselines[mode]["horizon_metrics"]["macro_brier"],
    )
    baseline = baselines[stronger]
    recall_gain = (
        structured["productive_effect_recall_at_8"]
        - baseline["productive_effect_recall_at_8"]
    )
    horizon_skill = _brier_skill(
        structured["horizon_metrics"], baseline["horizon_metrics"]
    )
    swap = _world_metrics(
        examples,
        priors=priors,
        calibration=calibration,
        projection=projection,
        mode="structured",
        beam_width=beam_width,
        swap_binding=True,
    )
    swap_drop = (
        structured["productive_effect_recall_at_8"]
        - swap["productive_effect_recall_at_8"]
    )
    per_game = {}
    for game in frozen["source_validation_games"]:
        selected = [example for example in examples if example.game_id == game]
        game_model = _world_metrics(
            selected,
            priors=priors,
            calibration=calibration,
            projection=projection,
            mode="structured",
            beam_width=beam_width,
        )
        game_baseline = _world_metrics(
            selected,
            priors=priors,
            calibration=calibration,
            projection=projection,
            mode=stronger,
            beam_width=beam_width,
        )
        per_game[game] = {
            "trajectories": len(selected),
            "brier_skill": _brier_skill(
                game_model["horizon_metrics"],
                game_baseline["horizon_metrics"],
            ),
            "structured_recall_at_8": game_model["productive_effect_recall_at_8"],
            "baseline_recall_at_8": game_baseline["productive_effect_recall_at_8"],
        }
    gates_cfg = frozen["world_model_gates"]
    gates = {
        "minimum_productive_recall_at_8": structured["productive_effect_recall_at_8"]
        >= float(gates_cfg["minimum_productive_recall_at_8"]),
        "minimum_recall_gain": recall_gain >= float(gates_cfg["minimum_recall_gain"]),
        "minimum_horizon3_brier_skill": horizon_skill
        >= float(gates_cfg["minimum_horizon3_brier_skill"]),
        "minimum_binding_swap_recall_drop": swap_drop
        >= float(gates_cfg["minimum_binding_swap_recall_drop"]),
        "maximum_macro_ece": structured["horizon_metrics"]["macro_ece"]
        <= float(gates_cfg["maximum_macro_ece"]),
        "every_game_nonnegative": all(
            item["brier_skill"] >= 0.0 for item in per_game.values()
        ),
        "binding_model_passed": binding["status"] == "PASS",
    }
    passed = all(gates.values())
    _write_jsonl(
        destination / "world_model_predictions.jsonl",
        structured["rows"],
    )
    payload = {
        "format_version": WORLD_MODEL_RESULT_FORMAT_VERSION,
        "status": "PASS" if passed else "FAIL_CLOSED",
        "frozen_manifest_checksum": frozen["manifest_checksum"],
        "binding_result_checksum": binding["result_checksum"],
        "model": "BoundSemanticWorldModel",
        "projection": projection,
        "horizon": int(frozen["world_model"]["horizon"]),
        "beam_width": beam_width,
        "trajectories": len(examples),
        "complete_tree_roots": len(examples) // 8,
        "structured": {
            key: value
            for key, value in structured.items()
            if not key.startswith("_") and key != "rows"
        },
        "baselines": {
            mode: {
                key: value
                for key, value in result.items()
                if not key.startswith("_") and key != "rows"
            }
            for mode, result in baselines.items()
        },
        "stronger_baseline": stronger,
        "productive_recall_gain": recall_gain,
        "horizon3_brier_skill": horizon_skill,
        "binding_swap": {
            "productive_effect_recall_at_8": swap["productive_effect_recall_at_8"],
            "recall_drop": swap_drop,
        },
        "per_game": per_game,
        "gates": gates,
        "world_model_fitted": True,
        "energy_protocol_authorized": passed,
        "safety_protocol_authorized": passed,
        "ebm_fit_authorized": False,
        "controller_authorized": False,
        "qwen_used": False,
        "gnn_used": False,
    }
    payload["result_checksum"] = _checksum(payload)
    _write_json(destination / "world_model_result.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "collect-source",
            "preflight",
            "collect-validation",
            "evaluate-binding",
            "evaluate-world-model",
        ),
    )
    parser.add_argument(
        "--frozen-manifest",
        default=str(DEFAULT_FROZEN_MANIFEST_PATH),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--environments-dir")
    args = parser.parse_args(argv)
    common = {
        "frozen_manifest_path": args.frozen_manifest,
        "output_dir": args.output_dir,
    }
    if args.command == "collect-source":
        result = run_collection(
            split="source_train",
            environments_dir=args.environments_dir,
            **common,
        )
    elif args.command == "preflight":
        result = run_source_preflight(**common)
    elif args.command == "collect-validation":
        result = run_collection(
            split="source_validation",
            environments_dir=args.environments_dir,
            **common,
        )
    elif args.command == "evaluate-binding":
        result = run_binding_evaluation(**common)
    else:
        result = run_world_model_evaluation(**common)
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AbstractBoundState",
    "BindingPairRecord",
    "BindingSignature",
    "BoundEvent",
    "BoundMechanicEvidence",
    "BoundMechanicRule",
    "BoundWindow",
    "BranchArm",
    "CalibrationBundle",
    "TrajectoryExample",
    "apply_calibration",
    "beam_rollout",
    "binding_swap_control",
    "build_trajectory_examples",
    "fit_priors",
    "fit_source_calibration",
    "load_frozen_manifest",
    "load_pairs",
    "main",
    "pair_windows",
    "replay_prefix",
    "run_binding_evaluation",
    "run_collection",
    "run_source_preflight",
    "run_world_model_evaluation",
    "score_window",
    "select_branch_actions",
    "target_metrics",
    "validate_model_view",
]
