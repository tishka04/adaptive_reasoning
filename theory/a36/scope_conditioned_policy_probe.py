"""A36 scope-conditioned action policy probe.

A36 is the first minimal motor policy that uses confirmed mechanics plus their
A35 usage scope to choose an action. It does not read M3/A32/A34 artifacts and
does not re-evaluate truth.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from theory.a33.confirmed_mechanics_registry import (
    DEFAULT_A33_CONFIRMED_MECHANICS_REGISTRY_OUTPUT_PATH,
)
from theory.a35.confirmed_mechanic_scope_map import (
    DEFAULT_A35_SCOPE_MAP_OUTPUT_PATH,
    context_id,
    execute_contextual_action_measurement,
    score_or_level_unchanged_or_improved,
)
from theory.non_ar25_active_micro_run import _env_dir


DEFAULT_A36_POLICY_PROBE_OUTPUT_PATH = (
    Path("diagnostics") / "a36" / "scope_conditioned_policy_probe.json"
)
DEFAULT_BASELINE_ORDER = ("ACTION3", "ACTION4", "ACTION1", "ACTION2")
TRUTH_STATUS = "NOT_REEVALUATED_BY_A36"
ELIGIBLE_SCOPE = "CONTEXTUALLY_STABLE"


@dataclass(frozen=True)
class ScopeConditionedPolicyProbe:
    """One live-context policy decision and its measured effect."""

    key: str
    game_id: str
    live_context_id: str
    live_context_sequence: Tuple[str, ...]
    policy_selected_action: str
    fallback_action: str
    predicted_metric: str
    selected_from_confirmed_mechanic: bool
    scope_used: str
    context_match: bool
    context_match_reason: str
    decision_reason: str
    selected_measurement: Dict[str, Any] = field(default_factory=dict)
    fallback_measurement: Dict[str, Any] = field(default_factory=dict)
    selected_signal: float = 0.0
    fallback_signal: float = 0.0
    functional_progress: bool = False
    useful_new_state: bool = False
    usage_contradiction: bool = False
    score_or_level_unchanged_or_improved: bool = True
    env_actions: int = 0
    error: str = ""
    truth_status: str = TRUTH_STATUS
    revision_performed: bool = False
    wrong_confirmations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "game_id": self.game_id,
            "live_context_id": self.live_context_id,
            "live_context_sequence": list(self.live_context_sequence),
            "policy_selected_action": self.policy_selected_action,
            "fallback_action": self.fallback_action,
            "predicted_metric": self.predicted_metric,
            "selected_from_confirmed_mechanic": self.selected_from_confirmed_mechanic,
            "scope_used": self.scope_used,
            "context_match": self.context_match,
            "context_match_reason": self.context_match_reason,
            "decision_reason": self.decision_reason,
            "selected_measurement": dict(self.selected_measurement),
            "fallback_measurement": dict(self.fallback_measurement),
            "selected_signal": self.selected_signal,
            "fallback_signal": self.fallback_signal,
            "functional_progress": self.functional_progress,
            "useful_new_state": self.useful_new_state,
            "usage_contradiction": self.usage_contradiction,
            "score_or_level_unchanged_or_improved": (
                self.score_or_level_unchanged_or_improved
            ),
            "env_actions": int(self.env_actions),
            "error": self.error,
            "truth_status": self.truth_status,
            "revision_performed": self.revision_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
        }


def run_scope_conditioned_policy_probe(
    *,
    registry_path: str | Path = DEFAULT_A33_CONFIRMED_MECHANICS_REGISTRY_OUTPUT_PATH,
    scope_map_path: str | Path = DEFAULT_A35_SCOPE_MAP_OUTPUT_PATH,
    live_context_sequence: Sequence[str] = (),
    environments_dir: str | Path | None = None,
    baseline_order: Sequence[str] = DEFAULT_BASELINE_ORDER,
) -> Dict[str, Any]:
    """Run one A36 policy decision from A33 registry and A35 scope only."""
    registry_payload = _load_json(registry_path)
    scope_payload = _load_json(scope_map_path)
    registry_entries = [
        dict(entry)
        for entry in registry_payload.get("confirmed_mechanics", []) or []
        if isinstance(entry, Mapping)
    ]
    scopes_by_key = _scope_maps_by_key(scope_payload)
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    probe = build_scope_conditioned_policy_probe(
        registry_entries,
        scopes_by_key=scopes_by_key,
        live_context_sequence=tuple(str(action) for action in live_context_sequence),
        environments_dir=env_dir,
        baseline_order=baseline_order,
    )
    return {
        "config": {
            "registry_path": str(registry_path),
            "scope_map_path": str(scope_map_path),
            "environments_dir": str(env_dir),
            "live_context_sequence": list(live_context_sequence),
            "baseline_order": list(baseline_order),
            "inputs_read": ["A33", "A35"],
            "artifacts_not_read": ["M3", "A32", "A34"],
        },
        "summary": summarize_policy_probe(probe),
        "policy_probe": probe.to_dict(),
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def build_scope_conditioned_policy_probe(
    registry_entries: Sequence[Mapping[str, Any]],
    *,
    scopes_by_key: Mapping[str, Mapping[str, Any]],
    live_context_sequence: Sequence[str],
    environments_dir: str | Path,
    baseline_order: Sequence[str] = DEFAULT_BASELINE_ORDER,
) -> ScopeConditionedPolicyProbe:
    """Choose and evaluate one action using confirmed mechanics and scope."""
    live_context = tuple(str(action) for action in live_context_sequence)
    selected_entry, selected_scope, match_reason = select_scoped_mechanic(
        registry_entries,
        scopes_by_key=scopes_by_key,
        live_context_sequence=live_context,
    )
    entry = selected_entry or first_registry_entry(registry_entries)
    if entry is None:
        return ScopeConditionedPolicyProbe(
            key="",
            game_id="",
            live_context_id=context_id(live_context),
            live_context_sequence=live_context,
            policy_selected_action="",
            fallback_action="",
            predicted_metric="",
            selected_from_confirmed_mechanic=False,
            scope_used="",
            context_match=False,
            context_match_reason="no_confirmed_mechanic_available",
            decision_reason="fallback_unavailable_no_registry_entry",
            error="no_confirmed_mechanic_available",
        )

    key = str(entry.get("key", ""))
    game_id = str(entry.get("game_id", ""))
    predicted_metric = str(entry.get("predicted_metric", ""))
    fallback_action = fallback_action_for_context(
        selected_scope or scopes_by_key.get(key, {}),
        live_context_sequence=live_context,
        treatment_action=str(entry.get("action", "")),
        baseline_order=baseline_order,
    )
    selected_from_mechanic = selected_entry is not None
    selected_action = (
        str(entry.get("action", "")) if selected_from_mechanic else fallback_action
    )
    decision_reason = (
        "stable_scope_context_match_prioritize_confirmed_mechanic"
        if selected_from_mechanic
        else "fallback_baseline_scope_not_eligible_or_not_matched"
    )
    return execute_policy_probe(
        key=key,
        game_id=game_id,
        live_context_sequence=live_context,
        selected_action=selected_action,
        fallback_action=fallback_action,
        predicted_metric=predicted_metric,
        selected_from_confirmed_mechanic=selected_from_mechanic,
        scope_used=str((selected_scope or scopes_by_key.get(key, {})).get("scope_assessment", "")),
        context_match=selected_from_mechanic,
        context_match_reason=match_reason,
        decision_reason=decision_reason,
        environments_dir=environments_dir,
    )


def select_scoped_mechanic(
    registry_entries: Sequence[Mapping[str, Any]],
    *,
    scopes_by_key: Mapping[str, Mapping[str, Any]],
    live_context_sequence: Sequence[str],
) -> Tuple[Mapping[str, Any] | None, Mapping[str, Any] | None, str]:
    """Return the first stable scoped mechanic matching the live context."""
    for entry in registry_entries:
        key = str(entry.get("key", ""))
        scope_map = scopes_by_key.get(key)
        if not scope_map:
            continue
        if str(scope_map.get("scope_assessment", "")) != ELIGIBLE_SCOPE:
            continue
        matched, reason = context_matches_scope(
            scope_map,
            live_context_sequence=live_context_sequence,
        )
        if matched:
            return dict(entry), dict(scope_map), reason
    return None, None, "no_stable_scope_context_match"


def context_matches_scope(
    scope_map: Mapping[str, Any],
    *,
    live_context_sequence: Sequence[str],
) -> Tuple[bool, str]:
    """Match exact covered contexts, or short neighbors for stable scopes."""
    live_context = tuple(str(action) for action in live_context_sequence)
    covered = [
        tuple(str(action) for action in probe.get("context_sequence", []) or [])
        for probe in scope_map.get("context_probes", []) or []
        if isinstance(probe, Mapping) and not probe.get("error")
    ]
    if live_context in covered:
        return True, "covered_context_exact"
    if str(scope_map.get("scope_assessment", "")) != ELIGIBLE_SCOPE:
        return False, "scope_not_stable"
    if not covered:
        return False, "no_covered_contexts"
    covered_actions = {action for context in covered for action in context}
    max_len = max(len(context) for context in covered)
    if len(live_context) <= max_len + 1 and set(live_context).issubset(covered_actions):
        return True, "neighbor_of_contextually_stable_scope"
    return False, "context_not_covered_by_scope"


def execute_policy_probe(
    *,
    key: str,
    game_id: str,
    live_context_sequence: Sequence[str],
    selected_action: str,
    fallback_action: str,
    predicted_metric: str,
    selected_from_confirmed_mechanic: bool,
    scope_used: str,
    context_match: bool,
    context_match_reason: str,
    decision_reason: str,
    environments_dir: str | Path,
) -> ScopeConditionedPolicyProbe:
    """Execute selected policy action and neutral fallback from same context."""
    live_context = tuple(str(action) for action in live_context_sequence)
    try:
        selected = execute_contextual_action_measurement(
            game_id,
            live_context,
            selected_action,
            predicted_metric,
            environments_dir=environments_dir,
        )
        if selected.get("error"):
            return _error_probe(
                key,
                game_id,
                live_context,
                selected_action,
                fallback_action,
                predicted_metric,
                selected_from_confirmed_mechanic,
                scope_used,
                context_match,
                context_match_reason,
                decision_reason,
                error=str(selected.get("error", "")),
            )
        fallback = execute_contextual_action_measurement(
            game_id,
            live_context,
            fallback_action,
            predicted_metric,
            environments_dir=environments_dir,
            metric_action_args=dict(selected.get("metric_action_args", {}) or {}),
        )
        if fallback.get("error"):
            return _error_probe(
                key,
                game_id,
                live_context,
                selected_action,
                fallback_action,
                predicted_metric,
                selected_from_confirmed_mechanic,
                scope_used,
                context_match,
                context_match_reason,
                decision_reason,
                error=str(fallback.get("error", "")),
            )
    except Exception as exc:  # pragma: no cover - integration failure path
        return _error_probe(
            key,
            game_id,
            live_context,
            selected_action,
            fallback_action,
            predicted_metric,
            selected_from_confirmed_mechanic,
            scope_used,
            context_match,
            context_match_reason,
            decision_reason,
            error=f"policy_probe_failed:{exc}",
        )

    selected_measurement = dict(selected.get("measurement", {}) or {})
    fallback_measurement = dict(fallback.get("measurement", {}) or {})
    selected_signal = metric_signal(selected_measurement, predicted_metric)
    fallback_signal = metric_signal(fallback_measurement, predicted_metric)
    no_regression = score_or_level_unchanged_or_improved(selected)
    contradiction = selected_signal < fallback_signal
    useful = bool(
        selected_from_confirmed_mechanic
        and selected_measurement.get("changed", False)
        and selected_signal > fallback_signal
        and no_regression
    )
    return ScopeConditionedPolicyProbe(
        key=key,
        game_id=game_id,
        live_context_id=context_id(live_context),
        live_context_sequence=live_context,
        policy_selected_action=selected_action,
        fallback_action=fallback_action,
        predicted_metric=predicted_metric,
        selected_from_confirmed_mechanic=selected_from_confirmed_mechanic,
        scope_used=scope_used,
        context_match=context_match,
        context_match_reason=context_match_reason,
        decision_reason=decision_reason,
        selected_measurement=selected_measurement,
        fallback_measurement=fallback_measurement,
        selected_signal=selected_signal,
        fallback_signal=fallback_signal,
        functional_progress=useful,
        useful_new_state=useful,
        usage_contradiction=contradiction,
        score_or_level_unchanged_or_improved=no_regression,
        env_actions=int(selected.get("env_actions", 0) or 0)
        + int(fallback.get("env_actions", 0) or 0),
    )


def fallback_action_for_context(
    scope_map: Mapping[str, Any],
    *,
    live_context_sequence: Sequence[str],
    treatment_action: str,
    baseline_order: Sequence[str] = DEFAULT_BASELINE_ORDER,
) -> str:
    live_context = tuple(str(action) for action in live_context_sequence)
    for probe in scope_map.get("context_probes", []) or []:
        if not isinstance(probe, Mapping):
            continue
        probe_context = tuple(str(action) for action in probe.get("context_sequence", []) or [])
        baseline = str(probe.get("baseline_action", ""))
        if probe_context == live_context and baseline and baseline != treatment_action:
            return baseline
    for action in baseline_order:
        if str(action) != str(treatment_action):
            return str(action)
    return ""


def metric_signal(measurement: Mapping[str, Any], predicted_metric: str) -> float:
    if predicted_metric == "local_patch_before_after":
        return float(measurement.get("local_changed_pixels", 0) or 0)
    if predicted_metric == "object_counts_before_after":
        return abs(float(measurement.get("object_count_delta", 0) or 0))
    if predicted_metric == "contact_graph_before_after":
        return float(len(measurement.get("contact_pairs_added", []) or [])) + float(
            len(measurement.get("contact_pairs_removed", []) or [])
        )
    if predicted_metric == "object_positions_before_after":
        return float(measurement.get("moved_component_count", 0) or 0)
    if predicted_metric == "object_shape_zone_before_after":
        return float(len(measurement.get("zone_delta", {}) or {}))
    return float(measurement.get("changed_pixels", 0) or 0)


def summarize_policy_probe(probe: ScopeConditionedPolicyProbe) -> Dict[str, Any]:
    return {
        "policy_selected_action": probe.policy_selected_action,
        "selected_from_confirmed_mechanic": probe.selected_from_confirmed_mechanic,
        "scope_used": probe.scope_used,
        "context_match": probe.context_match,
        "functional_progress": probe.functional_progress,
        "useful_new_state": probe.useful_new_state,
        "usage_contradiction": probe.usage_contradiction,
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def write_scope_conditioned_policy_probe(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_A36_POLICY_PROBE_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def first_registry_entry(
    registry_entries: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    return dict(registry_entries[0]) if registry_entries else None


def _scope_maps_by_key(payload: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(row.get("key", "")): dict(row)
        for row in payload.get("scope_maps", []) or []
        if isinstance(row, Mapping) and str(row.get("key", ""))
    }


def _error_probe(
    key: str,
    game_id: str,
    live_context: Tuple[str, ...],
    selected_action: str,
    fallback_action: str,
    predicted_metric: str,
    selected_from_confirmed_mechanic: bool,
    scope_used: str,
    context_match: bool,
    context_match_reason: str,
    decision_reason: str,
    *,
    error: str,
) -> ScopeConditionedPolicyProbe:
    return ScopeConditionedPolicyProbe(
        key=key,
        game_id=game_id,
        live_context_id=context_id(live_context),
        live_context_sequence=live_context,
        policy_selected_action=selected_action,
        fallback_action=fallback_action,
        predicted_metric=predicted_metric,
        selected_from_confirmed_mechanic=selected_from_confirmed_mechanic,
        scope_used=scope_used,
        context_match=context_match,
        context_match_reason=context_match_reason,
        decision_reason=decision_reason,
        error=error,
    )


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run A36 scope-conditioned action policy probe.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_A33_CONFIRMED_MECHANICS_REGISTRY_OUTPUT_PATH,
    )
    parser.add_argument(
        "--scope-map",
        type=Path,
        default=DEFAULT_A35_SCOPE_MAP_OUTPUT_PATH,
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument(
        "--context-sequence",
        nargs="*",
        default=(),
        help="Short live context replayed from RESET before policy action.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_A36_POLICY_PROBE_OUTPUT_PATH)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_scope_conditioned_policy_probe(
        registry_path=args.registry,
        scope_map_path=args.scope_map,
        live_context_sequence=tuple(args.context_sequence),
        environments_dir=args.environments_dir,
    )
    write_scope_conditioned_policy_probe(payload, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "summary": payload["summary"],
                "truth_status": TRUTH_STATUS,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
