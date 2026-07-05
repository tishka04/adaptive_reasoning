"""Live prefix replay collector for SAGE counterfactual alternatives."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np

from theory.m1.polymorphic_a25_adapter import _step_env_action
from theory.m2.m3_execution_smoke import _make_env, _reset_env
from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir, _valid_actions
from theory.real_env_option_adapter import snapshot_frame


SAGE1_TRUTH_STATUS = "NOT_EVALUATED_BY_SAGE_1"
REPLAY_EXACT = "LIVE_PREFIX_REPLAY_EXACT"
REPLAY_DIVERGED = "LIVE_PREFIX_REPLAY_DIVERGED"
ACTION_UNAVAILABLE = "ALTERNATIVE_ACTION_NOT_AVAILABLE_AFTER_REPLAY"
SETUP_FAILED = "LIVE_PREFIX_REPLAY_SETUP_FAILED"


@dataclass(frozen=True)
class LivePrefixAction:
    """One action in a live prefix or alternative probe."""

    name: str
    action_args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "action_args": dict(self.action_args),
        }


@dataclass(frozen=True)
class LivePrefixCounterfactualResult:
    """Result of replaying a prefix and trying an alternative action."""

    game_id: str
    prefix_actions: tuple[LivePrefixAction, ...]
    target_state_signature: str
    replay_state_signature: str = ""
    alternative_action: str = ""
    alternative_action_args: Dict[str, Any] = field(default_factory=dict)
    available_actions_source: str = "real_env_live_api"
    synthetic_available_actions_used: bool = False
    real_env_available_actions_used: bool = True
    live_prefix_replay_exact: bool = False
    active_counterfactual_collection_attempted: bool = False
    selected_action_legal: bool = False
    invalid_action_selected: bool = False
    status: str = REPLAY_DIVERGED
    reason: str = ""
    before_alternative_signature: str = ""
    after_alternative_signature: str = ""
    state_changed: bool = False
    env_actions: int = 0
    support: int = 0
    truth_status: str = SAGE1_TRUTH_STATUS
    revision_status: str = "CANDIDATE_ONLY"
    revision_performed: bool = False
    wrong_confirmations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_id": self.game_id,
            "prefix_actions": [action.to_dict() for action in self.prefix_actions],
            "target_state_signature": self.target_state_signature,
            "replay_state_signature": self.replay_state_signature,
            "alternative_action": self.alternative_action,
            "alternative_action_args": dict(self.alternative_action_args),
            "available_actions_source": self.available_actions_source,
            "synthetic_available_actions_used": self.synthetic_available_actions_used,
            "real_env_available_actions_used": self.real_env_available_actions_used,
            "live_prefix_replay_exact": self.live_prefix_replay_exact,
            "active_counterfactual_collection_attempted": (
                self.active_counterfactual_collection_attempted
            ),
            "selected_action_legal": self.selected_action_legal,
            "invalid_action_selected": self.invalid_action_selected,
            "status": self.status,
            "reason": self.reason,
            "before_alternative_signature": self.before_alternative_signature,
            "after_alternative_signature": self.after_alternative_signature,
            "state_changed": self.state_changed,
            "env_actions": int(self.env_actions),
            "support": int(self.support),
            "truth_status": self.truth_status,
            "revision_status": self.revision_status,
            "revision_performed": self.revision_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
        }


EnvFactory = Callable[[str], Any]


def collect_live_prefix_counterfactual(
    *,
    game_id: str,
    prefix_actions: Sequence[LivePrefixAction | Mapping[str, Any] | str],
    target_state_signature: str,
    alternative_action: str,
    alternative_action_args: Mapping[str, Any] | None = None,
    environments_dir: str | Path | None = None,
    env_factory: EnvFactory | None = None,
) -> LivePrefixCounterfactualResult:
    """Replay a live prefix from RESET and try one alternative action.

    The collector requires exact state-signature replay before it executes the
    alternative action. If replay diverges, no alternative action is sent.
    """
    normalized_prefix = tuple(normalize_prefix_action(action) for action in prefix_actions)
    env_actions = 0
    try:
        env = (
            env_factory(game_id)
            if env_factory is not None
            else _make_real_env(game_id, environments_dir)
        )
        frame = _reset_env(env)
    except Exception as exc:  # pragma: no cover - integration failure path
        return _collector_result(
            game_id=game_id,
            prefix_actions=normalized_prefix,
            target_state_signature=target_state_signature,
            alternative_action=alternative_action,
            alternative_action_args=alternative_action_args,
            status=SETUP_FAILED,
            reason=f"setup_failed:{exc}",
            env_actions=env_actions,
        )

    for action in normalized_prefix:
        selected = select_live_action(
            env,
            action.name,
            action_args=action.action_args,
        )
        if selected is None:
            return _collector_result(
                game_id=game_id,
                prefix_actions=normalized_prefix,
                target_state_signature=target_state_signature,
                alternative_action=alternative_action,
                alternative_action_args=alternative_action_args,
                replay_state_signature=state_signature_from_frame(frame),
                status=ACTION_UNAVAILABLE,
                reason=f"prefix_action_unavailable:{action.name}",
                env_actions=env_actions,
                invalid_action_selected=True,
            )
        frame = _step_env_action(env, selected)
        env_actions += 1

    replay_signature = state_signature_from_frame(frame)
    if replay_signature != str(target_state_signature):
        return _collector_result(
            game_id=game_id,
            prefix_actions=normalized_prefix,
            target_state_signature=target_state_signature,
            alternative_action=alternative_action,
            alternative_action_args=alternative_action_args,
            replay_state_signature=replay_signature,
            live_prefix_replay_exact=False,
            status=REPLAY_DIVERGED,
            reason="target_state_signature_mismatch",
            env_actions=env_actions,
        )

    selected_alternative = select_live_action(
        env,
        alternative_action,
        action_args=dict(alternative_action_args or {}),
    )
    if selected_alternative is None:
        return _collector_result(
            game_id=game_id,
            prefix_actions=normalized_prefix,
            target_state_signature=target_state_signature,
            alternative_action=alternative_action,
            alternative_action_args=alternative_action_args,
            replay_state_signature=replay_signature,
            live_prefix_replay_exact=True,
            active_counterfactual_collection_attempted=True,
            selected_action_legal=False,
            invalid_action_selected=True,
            status=ACTION_UNAVAILABLE,
            reason=f"alternative_action_unavailable:{alternative_action}",
            env_actions=env_actions,
        )

    before_alt = frame
    after_alt = _step_env_action(env, selected_alternative)
    env_actions += 1
    before_signature = state_signature_from_frame(before_alt)
    after_signature = state_signature_from_frame(after_alt)
    return _collector_result(
        game_id=game_id,
        prefix_actions=normalized_prefix,
        target_state_signature=target_state_signature,
        alternative_action=alternative_action,
        alternative_action_args=alternative_action_args,
        replay_state_signature=replay_signature,
        live_prefix_replay_exact=True,
        active_counterfactual_collection_attempted=True,
        selected_action_legal=True,
        invalid_action_selected=False,
        status=REPLAY_EXACT,
        reason="alternative_action_collected_from_replayed_live_prefix",
        before_alternative_signature=before_signature,
        after_alternative_signature=after_signature,
        state_changed=before_signature != after_signature,
        env_actions=env_actions,
    )


def select_live_action(
    env: Any,
    action_name: str,
    *,
    action_args: Mapping[str, Any] | None = None,
) -> Any | None:
    wanted = str(action_name)
    wanted_args = {str(key): str(value) for key, value in dict(action_args or {}).items()}
    fallback = None
    for action in _valid_actions(env):
        name = str(getattr(action, "name", ""))
        if name != wanted:
            continue
        actual_args = {
            str(key): str(value)
            for key, value in dict(getattr(action, "action_args", {}) or {}).items()
        }
        if wanted_args and actual_args == wanted_args:
            return action
        if not wanted_args and not actual_args:
            return action
        if fallback is None:
            fallback = action
    return fallback if not wanted_args else None


def normalize_prefix_action(
    action: LivePrefixAction | Mapping[str, Any] | str,
) -> LivePrefixAction:
    if isinstance(action, LivePrefixAction):
        return action
    if isinstance(action, Mapping):
        return LivePrefixAction(
            name=str(action.get("name", action.get("action", ""))),
            action_args=dict(action.get("action_args", {}) or {}),
        )
    return LivePrefixAction(name=str(action), action_args={})


def state_signature_from_frame(frame: Any) -> str:
    snapshot = snapshot_frame(frame)
    array = np.asarray(snapshot.grid, dtype=np.int32)
    payload = {
        "shape": tuple(int(v) for v in array.shape),
        "digest": hashlib.sha1(array.tobytes()).hexdigest()[:16],
        "levels_completed": int(snapshot.levels_completed),
        "game_state": str(snapshot.game_state),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _make_real_env(game_id: str, environments_dir: str | Path | None) -> Any:
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    _configure_offline_env(env_dir)
    return _make_env(game_id, env_dir)


def _collector_result(
    *,
    game_id: str,
    prefix_actions: tuple[LivePrefixAction, ...],
    target_state_signature: str,
    alternative_action: str,
    alternative_action_args: Mapping[str, Any] | None,
    replay_state_signature: str = "",
    live_prefix_replay_exact: bool = False,
    active_counterfactual_collection_attempted: bool = False,
    selected_action_legal: bool = False,
    invalid_action_selected: bool = False,
    status: str,
    reason: str,
    before_alternative_signature: str = "",
    after_alternative_signature: str = "",
    state_changed: bool = False,
    env_actions: int = 0,
) -> LivePrefixCounterfactualResult:
    return LivePrefixCounterfactualResult(
        game_id=game_id,
        prefix_actions=prefix_actions,
        target_state_signature=target_state_signature,
        replay_state_signature=replay_state_signature,
        alternative_action=str(alternative_action),
        alternative_action_args=dict(alternative_action_args or {}),
        live_prefix_replay_exact=live_prefix_replay_exact,
        active_counterfactual_collection_attempted=active_counterfactual_collection_attempted,
        selected_action_legal=selected_action_legal,
        invalid_action_selected=invalid_action_selected,
        status=status,
        reason=reason,
        before_alternative_signature=before_alternative_signature,
        after_alternative_signature=after_alternative_signature,
        state_changed=state_changed,
        env_actions=env_actions,
    )
