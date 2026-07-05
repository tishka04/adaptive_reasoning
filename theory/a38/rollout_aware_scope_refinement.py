"""A38 rollout-aware scope refinement.

A38 consumes live rollout usage evidence and refines applicability scope. It
does not change mechanic truth and does not modify A33.
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
)
from theory.a37.scope_conditioned_policy_rollout import (
    DEFAULT_A37_POLICY_ROLLOUT_OUTPUT_PATH,
)


DEFAULT_A38_SCOPE_REFINEMENT_OUTPUT_PATH = (
    Path("diagnostics") / "a38" / "rollout_aware_scope_refinement.json"
)
TRUTH_STATUS = "NOT_REEVALUATED_BY_A38"
REFINED_WITH_PRECONDITIONS = "CONTEXTUALLY_STABLE_WITH_PRECONDITIONS"


@dataclass(frozen=True)
class RolloutAwareScopeRefinement:
    """Refined usage scope for one known mechanic."""

    key: str
    game_id: str
    action: str
    mechanic_family: str
    predicted_metric: str
    source_scope_assessment: str
    refined_scope_assessment: str
    usage_preconditions: Tuple[str, ...] = field(default_factory=tuple)
    positive_usage_contexts: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    negative_usage_contexts: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    blocked_contexts: Tuple[str, ...] = field(default_factory=tuple)
    blocked_context_details: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    refinement_notes: Tuple[str, ...] = field(default_factory=tuple)
    truth_status: str = TRUTH_STATUS
    revision_performed: bool = False
    wrong_confirmations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "game_id": self.game_id,
            "mechanic": {
                "action": self.action,
                "mechanic_family": self.mechanic_family,
                "predicted_metric": self.predicted_metric,
            },
            "source_scope_assessment": self.source_scope_assessment,
            "refined_scope_assessment": self.refined_scope_assessment,
            "usage_preconditions": list(self.usage_preconditions),
            "positive_usage_contexts": [dict(row) for row in self.positive_usage_contexts],
            "negative_usage_contexts": [dict(row) for row in self.negative_usage_contexts],
            "blocked_contexts": list(self.blocked_contexts),
            "blocked_context_details": [
                dict(row) for row in self.blocked_context_details
            ],
            "refinement_notes": list(self.refinement_notes),
            "truth_status": self.truth_status,
            "revision_performed": self.revision_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
        }


def run_rollout_aware_scope_refinement(
    *,
    registry_path: str | Path = DEFAULT_A33_CONFIRMED_MECHANICS_REGISTRY_OUTPUT_PATH,
    scope_map_path: str | Path = DEFAULT_A35_SCOPE_MAP_OUTPUT_PATH,
    rollout_path: str | Path = DEFAULT_A37_POLICY_ROLLOUT_OUTPUT_PATH,
) -> Dict[str, Any]:
    """Build rollout-aware scope refinements from A37 usage evidence."""
    registry_payload = _load_json(registry_path)
    scope_payload = _load_json(scope_map_path)
    rollout_payload = _load_json(rollout_path)
    scopes_by_key = _scope_maps_by_key(scope_payload)
    rollout_steps_by_key = _rollout_steps_by_key(rollout_payload)
    refinements = [
        build_rollout_aware_scope_refinement(
            entry,
            scope_map=scopes_by_key.get(str(entry.get("key", "")), {}),
            rollout_steps=rollout_steps_by_key.get(str(entry.get("key", "")), ()),
        )
        for entry in registry_payload.get("confirmed_mechanics", []) or []
        if isinstance(entry, Mapping)
    ]
    return {
        "config": {
            "registry_path": str(registry_path),
            "scope_map_path": str(scope_map_path),
            "rollout_path": str(rollout_path),
            "inputs_read": ["A33", "A35", "A37"],
            "artifacts_not_modified": ["A33"],
            "artifacts_not_read_for_truth": ["M3", "A32", "A34", "A36"],
        },
        "summary": summarize_refinements(refinements),
        "scope_refinements": [refinement.to_dict() for refinement in refinements],
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def build_rollout_aware_scope_refinement(
    entry: Mapping[str, Any],
    *,
    scope_map: Mapping[str, Any],
    rollout_steps: Sequence[Mapping[str, Any]],
) -> RolloutAwareScopeRefinement:
    """Refine one A35 scope with A37 positive/negative usage observations."""
    positive_steps = [dict(step) for step in rollout_steps if is_positive_usage(step)]
    negative_steps = [dict(step) for step in rollout_steps if is_negative_usage(step)]
    positive_after_patches = [
        dict(step.get("measurement", {}) or {}).get("local_patch_after")
        for step in positive_steps
        if dict(step.get("measurement", {}) or {}).get("local_patch_after") is not None
    ]
    history_before_by_step = action_history_before_steps(rollout_steps)
    positive_contexts = tuple(
        positive_usage_context(step) for step in positive_steps
    )
    negative_contexts = tuple(
        negative_usage_context(
            step,
            history_before=history_before_by_step.get(int(step.get("step", 0) or 0), ()),
            positive_after_patches=positive_after_patches,
        )
        for step in negative_steps
    )
    blocked_details = tuple(
        blocked_context_detail(row)
        for row in negative_contexts
        if str(row.get("blocked_context_id", ""))
    )
    blocked = tuple(str(row["blocked_context_id"]) for row in blocked_details)
    refined_scope = refined_scope_assessment(
        source_scope=str(scope_map.get("scope_assessment", "")),
        positive_contexts=positive_contexts,
        negative_contexts=negative_contexts,
    )
    return RolloutAwareScopeRefinement(
        key=str(entry.get("key", "")),
        game_id=str(entry.get("game_id", "")),
        action=str(entry.get("action", "")),
        mechanic_family=str(entry.get("mechanic_family", "")),
        predicted_metric=str(entry.get("predicted_metric", "")),
        source_scope_assessment=str(scope_map.get("scope_assessment", "")),
        refined_scope_assessment=refined_scope,
        usage_preconditions=usage_preconditions_from_evidence(
            positive_contexts=positive_contexts,
            negative_contexts=negative_contexts,
        ),
        positive_usage_contexts=positive_contexts,
        negative_usage_contexts=negative_contexts,
        blocked_contexts=blocked,
        blocked_context_details=blocked_details,
        refinement_notes=refinement_notes_from_evidence(
            positive_contexts=positive_contexts,
            negative_contexts=negative_contexts,
        ),
    )


def is_positive_usage(step: Mapping[str, Any]) -> bool:
    return bool(
        step.get("selected_from_confirmed_mechanic")
        and step.get("functional_progress")
        and step.get("useful_new_state")
        and not step.get("usage_contradiction")
        and float(step.get("selected_signal", 0.0) or 0.0) > 0
    )


def is_negative_usage(step: Mapping[str, Any]) -> bool:
    if not step.get("selected_from_confirmed_mechanic"):
        return False
    return bool(
        step.get("usage_contradiction")
        or float(step.get("selected_signal", 0.0) or 0.0) <= 0
        or not step.get("useful_new_state")
    )


def positive_usage_context(step: Mapping[str, Any]) -> Dict[str, Any]:
    measurement = dict(step.get("measurement", {}) or {})
    return {
        "step": int(step.get("step", 0) or 0),
        "context_id": str(step.get("context_id", "")),
        "context_signature": list(step.get("context_signature", []) or []),
        "policy_selected_action": str(step.get("policy_selected_action", "")),
        "selected_signal": float(step.get("selected_signal", 0.0) or 0.0),
        "local_patch_available": bool(measurement.get("local_patch_available", False)),
        "local_changed_pixels": int(measurement.get("local_changed_pixels", 0) or 0),
        "target_patch_not_already_saturated": target_patch_not_already_saturated(
            measurement
        ),
        "functional_progress": bool(step.get("functional_progress", False)),
        "useful_new_state": bool(step.get("useful_new_state", False)),
        "usage_contradiction": bool(step.get("usage_contradiction", False)),
        "patch_bbox": list(measurement.get("patch_bbox", []) or []),
        "local_patch_before": measurement.get("local_patch_before"),
        "local_patch_after": measurement.get("local_patch_after"),
    }


def negative_usage_context(
    step: Mapping[str, Any],
    *,
    history_before: Sequence[str],
    positive_after_patches: Sequence[Any],
) -> Dict[str, Any]:
    measurement = dict(step.get("measurement", {}) or {})
    already_saturated = target_patch_already_saturated(
        measurement,
        positive_after_patches=positive_after_patches,
    )
    previous_confirmed = previous_confirmed_action(history_before, str(step.get("policy_selected_action", "")))
    blocked_id = blocked_context_id(
        str(step.get("context_id", "")),
        previous_confirmed_action=previous_confirmed,
    )
    return {
        "step": int(step.get("step", 0) or 0),
        "context_id": str(step.get("context_id", "")),
        "blocked_context_id": blocked_id,
        "context_signature": list(step.get("context_signature", []) or []),
        "live_action_history_before": list(history_before),
        "policy_selected_action": str(step.get("policy_selected_action", "")),
        "selected_signal": float(step.get("selected_signal", 0.0) or 0.0),
        "local_patch_available": bool(measurement.get("local_patch_available", False)),
        "local_changed_pixels": int(measurement.get("local_changed_pixels", 0) or 0),
        "local_patch_before_equals_after": local_patch_before_equals_after(measurement),
        "target_patch_already_saturated": already_saturated,
        "probable_blocker": (
            "target_patch_already_saturated"
            if already_saturated
            else "predicted_metric_signal_unavailable"
        ),
        "functional_progress": bool(step.get("functional_progress", False)),
        "useful_new_state": bool(step.get("useful_new_state", False)),
        "usage_contradiction": bool(step.get("usage_contradiction", False)),
        "patch_bbox": list(measurement.get("patch_bbox", []) or []),
        "local_patch_before": measurement.get("local_patch_before"),
        "local_patch_after": measurement.get("local_patch_after"),
    }


def usage_preconditions_from_evidence(
    *,
    positive_contexts: Sequence[Mapping[str, Any]],
    negative_contexts: Sequence[Mapping[str, Any]],
) -> Tuple[str, ...]:
    if not positive_contexts and not negative_contexts:
        return tuple()
    preconditions = [
        "local_patch_available=true",
        "predicted_metric_signal_available=true",
        "selected_signal_expected=1",
    ]
    if any(row.get("target_patch_already_saturated") for row in negative_contexts):
        preconditions.append("target_patch_not_already_saturated=true")
    return tuple(preconditions)


def refined_scope_assessment(
    *,
    source_scope: str,
    positive_contexts: Sequence[Mapping[str, Any]],
    negative_contexts: Sequence[Mapping[str, Any]],
) -> str:
    if positive_contexts and negative_contexts:
        return REFINED_WITH_PRECONDITIONS
    if source_scope:
        return source_scope
    if positive_contexts:
        return "ROLL_OUT_USEFUL_NO_REFINEMENT"
    return "UNSTABLE_OR_NOT_USEFUL"


def refinement_notes_from_evidence(
    *,
    positive_contexts: Sequence[Mapping[str, Any]],
    negative_contexts: Sequence[Mapping[str, Any]],
) -> Tuple[str, ...]:
    notes = [
        "truth_not_reevaluated",
        "a33_registry_not_modified",
        "rollout_usage_updates_applicability_scope_only",
    ]
    if any(row.get("target_patch_already_saturated") for row in negative_contexts):
        notes.append("effect_saturation_detected")
        notes.append("avoid_reusing_action_when_target_patch_already_transformed")
    if positive_contexts:
        notes.append("positive_rollout_usage_retained_as_policy_evidence")
    return tuple(notes)


def target_patch_not_already_saturated(measurement: Mapping[str, Any]) -> bool:
    return bool(
        measurement.get("local_patch_available")
        and int(measurement.get("local_changed_pixels", 0) or 0) > 0
        and not local_patch_before_equals_after(measurement)
    )


def target_patch_already_saturated(
    measurement: Mapping[str, Any],
    *,
    positive_after_patches: Sequence[Any],
) -> bool:
    before = measurement.get("local_patch_before")
    return bool(
        measurement.get("local_patch_available")
        and int(measurement.get("local_changed_pixels", 0) or 0) == 0
        and local_patch_before_equals_after(measurement)
        and (not positive_after_patches or any(before == patch for patch in positive_after_patches))
    )


def local_patch_before_equals_after(measurement: Mapping[str, Any]) -> bool:
    before = measurement.get("local_patch_before")
    after = measurement.get("local_patch_after")
    return before is not None and after is not None and before == after


def action_history_before_steps(
    rollout_steps: Sequence[Mapping[str, Any]],
) -> Dict[int, Tuple[str, ...]]:
    history: list[str] = []
    by_step: Dict[int, Tuple[str, ...]] = {}
    ordered = sorted(
        [dict(step) for step in rollout_steps],
        key=lambda step: int(step.get("step", 0) or 0),
    )
    for step in ordered:
        step_index = int(step.get("step", 0) or 0)
        by_step[step_index] = tuple(history)
        action = str(step.get("policy_selected_action", ""))
        if action:
            history.append(action)
    return by_step


def previous_confirmed_action(
    history_before: Sequence[str],
    current_action: str,
) -> str:
    current = str(current_action)
    for action in reversed(tuple(str(action) for action in history_before)):
        if action == current:
            return action
    return ""


def blocked_context_id(
    context_id: str,
    *,
    previous_confirmed_action: str,
) -> str:
    if previous_confirmed_action:
        return f"{context_id}_live_after_{previous_confirmed_action}"
    return f"{context_id}_live_after_prior_policy"


def blocked_context_detail(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "blocked_context_id": str(row.get("blocked_context_id", "")),
        "context_id": str(row.get("context_id", "")),
        "reason": str(row.get("probable_blocker", "")),
        "step": int(row.get("step", 0) or 0),
        "selected_signal": float(row.get("selected_signal", 0.0) or 0.0),
        "local_patch_before_equals_after": bool(
            row.get("local_patch_before_equals_after", False)
        ),
    }


def summarize_refinements(
    refinements: Sequence[RolloutAwareScopeRefinement],
) -> Dict[str, Any]:
    return {
        "mechanics_refined": len(refinements),
        "refinements_with_preconditions": len(
            [
                refinement
                for refinement in refinements
                if refinement.refined_scope_assessment == REFINED_WITH_PRECONDITIONS
            ]
        ),
        "positive_usage_contexts": sum(
            len(refinement.positive_usage_contexts) for refinement in refinements
        ),
        "negative_usage_contexts": sum(
            len(refinement.negative_usage_contexts) for refinement in refinements
        ),
        "blocked_contexts": sum(
            len(refinement.blocked_contexts) for refinement in refinements
        ),
        "effect_saturation_detected": any(
            "effect_saturation_detected" in refinement.refinement_notes
            for refinement in refinements
        ),
        "truth_status": TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def write_rollout_aware_scope_refinement(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_A38_SCOPE_REFINEMENT_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _scope_maps_by_key(payload: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(row.get("key", "")): dict(row)
        for row in payload.get("scope_maps", []) or []
        if isinstance(row, Mapping) and str(row.get("key", ""))
    }


def _rollout_steps_by_key(payload: Mapping[str, Any]) -> Dict[str, Tuple[Dict[str, Any], ...]]:
    by_key: Dict[str, list[Dict[str, Any]]] = {}
    for step in payload.get("rollout_steps", []) or []:
        if not isinstance(step, Mapping):
            continue
        key = str(step.get("key", ""))
        if not key:
            continue
        by_key.setdefault(key, []).append(dict(step))
    return {
        key: tuple(sorted(rows, key=lambda row: int(row.get("step", 0) or 0)))
        for key, rows in by_key.items()
    }


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run A38 rollout-aware scope refinement.",
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
    parser.add_argument(
        "--rollout",
        type=Path,
        default=DEFAULT_A37_POLICY_ROLLOUT_OUTPUT_PATH,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_A38_SCOPE_REFINEMENT_OUTPUT_PATH)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_rollout_aware_scope_refinement(
        registry_path=args.registry,
        scope_map_path=args.scope_map,
        rollout_path=args.rollout,
    )
    write_rollout_aware_scope_refinement(payload, args.out)
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
