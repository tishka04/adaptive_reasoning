"""M3.22 planner for follow-ups requested by A32 patch-similarity review."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir

from .dynamic_retarget_patch_similarity_expansion_planner import (
    PATCH_SIMILARITY_EXPANSION_POLICY,
    score_patch_similarity_candidate,
)
from .dynamic_retarget_selection_followup_executor import (
    available_args_for_request,
    grid_after_replay,
)
from .m2_observation_refinement import M3_REFINEMENT_TRUTH_STATUS


DEFAULT_A32_PATCH_SIMILARITY_REVISION_DECISIONS_INPUT_PATH = (
    Path("diagnostics") / "a32" / "patch_similarity_revision_decisions.json"
)
DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_FOLLOWUP_REQUESTS_OUTPUT_PATH = (
    Path("diagnostics")
    / "m3"
    / "a32_requested_patch_similarity_followup_requests.json"
)

READY_FOR_M3_A32_PATCH_FOLLOWUP = "READY_FOR_M3_A32_PATCH_FOLLOWUP"
BLOCKED_A32_PATCH_FOLLOWUP = "BLOCKED_A32_PATCH_FOLLOWUP"
DEFAULT_MAX_DYNAMIC_ARGS = 2


@dataclass(frozen=True)
class A32RequestedPatchSimilarityFollowupRequest:
    """Executable M3 request derived from A32.3 requested follow-up tests."""

    request_id: str
    source_a32_queue_item_id: str
    source_a32_decision: str
    source_a32_recommended_next_step: str
    source_a32_decision_reasons: Tuple[str, ...]
    a32_decision_counted_as_confirmation: bool
    game_id: str
    candidate_rule_family: str
    followup_family: str
    purpose: str
    context_replay: Tuple[str, ...]
    context_replay_args: Tuple[Dict[str, Any], ...] | None
    target_action: str
    target_action_arg_policy: str
    resolved_target_action_args: Tuple[Dict[str, Any], ...]
    candidate_resolution_args: Tuple[Dict[str, Any], ...]
    excluded_args: Tuple[Dict[str, Any], ...]
    seed_successful_args: Tuple[Dict[str, Any], ...]
    seed_failed_args: Tuple[Dict[str, Any], ...]
    max_dynamic_args: int
    suggested_control_actions: Tuple[str, ...]
    control_policy: str
    metrics: Tuple[str, ...]
    success_metrics: Tuple[str, ...]
    diagnostic_metrics: Tuple[str, ...]
    expected_signal: str
    falsification_criterion: str
    planning_rationale: str
    resolution_basis: str
    status: str = READY_FOR_M3_A32_PATCH_FOLLOWUP
    revision_status: str = "CANDIDATE_ONLY"
    support: int = 0
    controlled_test_required: bool = True
    truth_status: str = M3_REFINEMENT_TRUTH_STATUS
    revision_performed: bool = False
    wrong_confirmations: int = 0
    trace_support_counted_as_proof: bool = False
    prior_counted_as_proof: bool = False
    observation_counted_as_confirmation: bool = False
    rule_counted_as_confirmation: bool = False
    execution_performed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "source_a32_queue_item_id": self.source_a32_queue_item_id,
            "source_a32_decision": self.source_a32_decision,
            "source_a32_recommended_next_step": (
                self.source_a32_recommended_next_step
            ),
            "source_a32_decision_reasons": list(self.source_a32_decision_reasons),
            "a32_decision_counted_as_confirmation": (
                self.a32_decision_counted_as_confirmation
            ),
            "game_id": self.game_id,
            "candidate_rule_family": self.candidate_rule_family,
            "followup_family": self.followup_family,
            "purpose": self.purpose,
            "context_replay": list(self.context_replay),
            "context_replay_args": (
                [dict(item) for item in self.context_replay_args]
                if self.context_replay_args is not None
                else None
            ),
            "target_action": self.target_action,
            "target_action_arg_policy": self.target_action_arg_policy,
            "resolved_target_action_args": [
                dict(item) for item in self.resolved_target_action_args
            ],
            "candidate_resolution_args": [
                dict(item) for item in self.candidate_resolution_args
            ],
            "excluded_args": [dict(item) for item in self.excluded_args],
            "seed_successful_args": [
                dict(item) for item in self.seed_successful_args
            ],
            "seed_failed_args": [dict(item) for item in self.seed_failed_args],
            "max_dynamic_args": int(self.max_dynamic_args),
            "suggested_control_actions": list(self.suggested_control_actions),
            "control_policy": self.control_policy,
            "metrics": list(self.metrics),
            "success_metrics": list(self.success_metrics),
            "diagnostic_metrics": list(self.diagnostic_metrics),
            "expected_signal": self.expected_signal,
            "falsification_criterion": self.falsification_criterion,
            "planning_rationale": self.planning_rationale,
            "resolution_basis": self.resolution_basis,
            "status": self.status,
            "revision_status": self.revision_status,
            "support": int(self.support),
            "controlled_test_required": self.controlled_test_required,
            "truth_status": self.truth_status,
            "revision_performed": self.revision_performed,
            "wrong_confirmations": int(self.wrong_confirmations),
            "trace_support_counted_as_proof": self.trace_support_counted_as_proof,
            "prior_counted_as_proof": self.prior_counted_as_proof,
            "observation_counted_as_confirmation": (
                self.observation_counted_as_confirmation
            ),
            "rule_counted_as_confirmation": self.rule_counted_as_confirmation,
            "execution_performed": self.execution_performed,
        }


def run_a32_requested_patch_similarity_followup_planning(
    *,
    a32_decisions_path: str | Path = DEFAULT_A32_PATCH_SIMILARITY_REVISION_DECISIONS_INPUT_PATH,
    environments_dir: str | Path | None = None,
    max_dynamic_args: int = DEFAULT_MAX_DYNAMIC_ARGS,
) -> Dict[str, Any]:
    payload = _load_json(a32_decisions_path)
    decisions = [
        dict(row) for row in payload.get("revision_decisions", []) or []
    ]
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    _configure_offline_env(env_dir)

    requests: list[A32RequestedPatchSimilarityFollowupRequest] = []
    blocked: list[Dict[str, Any]] = []
    candidate_groups: list[Dict[str, Any]] = []
    for decision in decisions:
        if str(decision.get("recommended_next_step", "")) != (
            "REQUEST_MORE_TESTS_WITH_SCOPE_LIMITS"
        ):
            continue
        for followup in decision.get("requested_followup_tests", []) or []:
            if not isinstance(followup, Mapping):
                continue
            group = build_followup_candidate_group(
                decision=decision,
                followup=followup,
                environments_dir=env_dir,
                max_dynamic_args=max_dynamic_args,
            )
            candidate_groups.append(group)
            selected = [
                dict(item) for item in group.get("selected_candidates", []) or []
            ]
            if not selected:
                blocked.append(blocked_followup(decision, followup, group))
                continue
            requests.append(
                build_followup_request(
                    decision=decision,
                    followup=followup,
                    group=group,
                    selected_candidates=selected,
                    max_dynamic_args=max_dynamic_args,
                )
            )

    return {
        "config": {
            "a32_decisions_path": str(a32_decisions_path),
            "environments_dir": str(env_dir),
            "schema_version": "m3.a32_requested_patch_similarity_followup_requests.v1",
            "inputs_read": ["A32.3"],
            "artifacts_not_modified": ["A32", "A33"],
            "execution_performed": False,
            "max_dynamic_args": int(max_dynamic_args),
        },
        "summary": summarize_followup_planning(
            decisions=decisions,
            candidate_groups=candidate_groups,
            requests=requests,
            blocked=blocked,
        ),
        "candidate_arg_groups": [dict(group) for group in candidate_groups],
        "a32_requested_followup_requests": [
            request.to_dict() for request in requests
        ],
        "blocked_followup_requests": [dict(row) for row in blocked],
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "trace_support_counted_as_proof": False,
        "prior_counted_as_proof": False,
        "observation_counted_as_confirmation": False,
        "rule_counted_as_confirmation": False,
        "a32_decision_counted_as_confirmation": False,
        "execution_performed": False,
    }


def build_followup_candidate_group(
    *,
    decision: Mapping[str, Any],
    followup: Mapping[str, Any],
    environments_dir: str | Path,
    max_dynamic_args: int,
) -> Dict[str, Any]:
    family = str(followup.get("followup_family", ""))
    request_view = request_view_for_followup(decision, followup)
    if family == "alternate_repositioning_context_probe":
        request_view = alternate_context_request_view(
            decision=decision,
            followup=followup,
            environments_dir=environments_dir,
        )
    try:
        live_args = available_args_for_request(
            request_view,
            environments_dir=environments_dir,
        )
        replay_grid = grid_after_replay(request_view, environments_dir=environments_dir)
        exact_replay_available = True
        blocked_reason = None
    except Exception as exc:  # pragma: no cover - integration failure path
        live_args = ()
        replay_grid = None
        exact_replay_available = False
        blocked_reason = f"exact_replay_failed:{exc}"

    excluded = dedupe_args(followup.get("exclude_known_args", []) or [])
    excluded_keys = {_args_key(args) for args in excluded}
    available_after_exclusion = [
        dict(args) for args in live_args if _args_key(args) not in excluded_keys
    ]
    seed_successes = dedupe_args(
        decision.get("scope_limits", {}).get("successful_args_total", []) or []
    )
    seed_failures = dedupe_args(
        decision.get("scope_limits", {}).get("failed_args", []) or []
    )
    if replay_grid is None:
        candidates: Tuple[Dict[str, Any], ...] = ()
    else:
        candidates = tuple(
            score_followup_candidate(
                args,
                replay_grid=replay_grid,
                success_seeds=seed_successes,
                failure_seeds=seed_failures,
                followup_family=family,
            )
            for args in available_after_exclusion
        )
    filtered = filter_candidates_for_family(candidates, followup_family=family)
    selected = select_followup_candidates(
        filtered,
        followup_family=family,
        max_dynamic_args=max_dynamic_args,
    )
    if not selected and blocked_reason is None:
        blocked_reason = "no_live_candidate_args_after_a32_constraints"
    return {
        "source_a32_queue_item_id": str(decision.get("queue_item_id", "")),
        "source_a32_decision": str(decision.get("decision", "")),
        "source_a32_recommended_next_step": str(
            decision.get("recommended_next_step", "")
        ),
        "followup_family": family,
        "game_id": str(decision.get("game_id", "")),
        "context_replay": list(request_view.get("context_replay", []) or []),
        "context_replay_args": _context_args_list(
            request_view.get("context_replay_args")
        ),
        "target_action": str(request_view.get("target_action", "")),
        "live_available_args": [dict(item) for item in live_args],
        "excluded_args": [dict(item) for item in excluded],
        "available_args_after_exclusion": [
            dict(item) for item in available_after_exclusion
        ],
        "seed_successful_args": [dict(item) for item in seed_successes],
        "seed_failed_args": [dict(item) for item in seed_failures],
        "candidate_resolution_args": [dict(item) for item in candidates],
        "selected_candidates": [dict(item) for item in selected],
        "exact_replay_available": exact_replay_available,
        "blocked_reason": blocked_reason,
        "max_dynamic_args": int(max_dynamic_args),
        "status": (
            "CANDIDATE_ARGS_GENERATED"
            if selected
            else "NO_EXECUTABLE_A32_FOLLOWUP_ARGS"
        ),
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "wrong_confirmations": 0,
    }


def request_view_for_followup(
    decision: Mapping[str, Any],
    followup: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "game_id": str(decision.get("game_id", "")),
        "context_replay": list(followup.get("context_replay", []) or []),
        "context_replay_args": _context_args_list(
            followup.get("context_replay_args")
        ),
        "target_action": str(followup.get("target_action", "")),
    }


def alternate_context_request_view(
    *,
    decision: Mapping[str, Any],
    followup: Mapping[str, Any],
    environments_dir: str | Path,
) -> Dict[str, Any]:
    scope = dict(decision.get("scope_limits", {}) or {})
    base_replay = list(scope.get("context_replay", []) or [])
    base_args = _context_args_list(scope.get("context_replay_args")) or []
    variants = alternate_context_variants(base_replay, base_args)
    for replay, args in variants:
        view = {
            "game_id": str(decision.get("game_id", "")),
            "context_replay": replay,
            "context_replay_args": args,
            "target_action": str(followup.get("target_action", "")),
        }
        try:
            if available_args_for_request(view, environments_dir=environments_dir):
                return view
        except Exception:
            continue
    return request_view_for_followup(decision, followup)


def alternate_context_variants(
    base_replay: Sequence[str],
    base_args: Sequence[Mapping[str, Any]],
) -> Tuple[Tuple[list[str], list[Dict[str, Any]]], ...]:
    if len(base_replay) >= 3 and base_replay[:3] == ["ACTION6", "ACTION3", "ACTION4"]:
        first_args = dict(base_args[0]) if base_args else {}
        return (
            (["ACTION6", "ACTION4"], [first_args, {}]),
            (["ACTION6", "ACTION4", "ACTION3"], [first_args, {}, {}]),
            (
                ["ACTION6", "ACTION3", "ACTION4", "ACTION3"],
                [first_args, {}, {}, {}],
            ),
            (
                ["ACTION6", "ACTION3", "ACTION4", "ACTION4"],
                [first_args, {}, {}, {}],
            ),
        )
    return ((list(base_replay), [dict(args) for args in base_args]),)


def score_followup_candidate(
    action_args: Mapping[str, Any],
    *,
    replay_grid: Any,
    success_seeds: Sequence[Mapping[str, Any]],
    failure_seeds: Sequence[Mapping[str, Any]],
    followup_family: str,
) -> Dict[str, Any]:
    scored = score_patch_similarity_candidate(
        action_args,
        replay_grid=replay_grid,
        success_seeds=success_seeds,
        failure_seeds=failure_seeds,
    )
    scored["outside_known_y12_region"] = int(dict(action_args).get("y", -1)) != 12
    scored["followup_family"] = followup_family
    return scored


def filter_candidates_for_family(
    candidates: Sequence[Mapping[str, Any]],
    *,
    followup_family: str,
) -> Tuple[Dict[str, Any], ...]:
    if followup_family == "outside_known_y12_region_probe":
        return tuple(
            dict(row)
            for row in candidates
            if bool(row.get("outside_known_y12_region"))
        )
    return tuple(dict(row) for row in candidates)


def select_followup_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    followup_family: str,
    max_dynamic_args: int,
) -> Tuple[Dict[str, Any], ...]:
    def key(row: Mapping[str, Any]) -> Tuple[float, float, str]:
        if followup_family == "outside_known_y12_region_probe":
            return (
                float(row.get("success_patch_distance", 0.0) or 0.0),
                float(row.get("failure_patch_distance", 0.0) or 0.0),
                _args_key(row.get("action_args", {}) or {}),
            )
        return (
            float(row.get("success_patch_distance", 0.0) or 0.0),
            abs(
                float(row.get("success_patch_distance", 0.0) or 0.0)
                - float(row.get("failure_patch_distance", 0.0) or 0.0)
            ),
            _args_key(row.get("action_args", {}) or {}),
        )

    selected = []
    for rank, row in enumerate(
        sorted(candidates, key=key)[: max(0, int(max_dynamic_args))],
        start=1,
    ):
        value = dict(row)
        value["candidate_arg_rank"] = rank
        value["resolution_basis"] = resolution_basis_for_family(followup_family)
        selected.append(value)
    return tuple(selected)


def build_followup_request(
    *,
    decision: Mapping[str, Any],
    followup: Mapping[str, Any],
    group: Mapping[str, Any],
    selected_candidates: Sequence[Mapping[str, Any]],
    max_dynamic_args: int,
) -> A32RequestedPatchSimilarityFollowupRequest:
    selected_args = tuple(
        dict(row.get("action_args", {}) or {}) for row in selected_candidates
    )
    success_metrics = tuple(str(item) for item in followup.get("success_metrics", []) or [])
    diagnostic_metrics = tuple(
        str(item) for item in followup.get("diagnostic_metrics", []) or []
    )
    metrics = ordered_unique((*success_metrics, *diagnostic_metrics))
    family = str(followup.get("followup_family", ""))
    return A32RequestedPatchSimilarityFollowupRequest(
        request_id=followup_request_id(decision, family, selected_args),
        source_a32_queue_item_id=str(decision.get("queue_item_id", "")),
        source_a32_decision=str(decision.get("decision", "")),
        source_a32_recommended_next_step=str(decision.get("recommended_next_step", "")),
        source_a32_decision_reasons=tuple(
            str(reason) for reason in decision.get("reasons", []) or []
        ),
        a32_decision_counted_as_confirmation=False,
        game_id=str(decision.get("game_id", "")),
        candidate_rule_family=str(
            decision.get("scope_limits", {}).get("candidate_rule_family", "")
        ),
        followup_family=family,
        purpose=str(followup.get("purpose", "")),
        context_replay=tuple(str(item) for item in group.get("context_replay", []) or []),
        context_replay_args=_context_args_tuple(group.get("context_replay_args")),
        target_action=str(group.get("target_action", "")),
        target_action_arg_policy=PATCH_SIMILARITY_EXPANSION_POLICY,
        resolved_target_action_args=selected_args,
        candidate_resolution_args=tuple(
            dict(row) for row in group.get("candidate_resolution_args", []) or []
        ),
        excluded_args=tuple(dict(row) for row in group.get("excluded_args", []) or []),
        seed_successful_args=tuple(
            dict(row) for row in group.get("seed_successful_args", []) or []
        ),
        seed_failed_args=tuple(
            dict(row) for row in group.get("seed_failed_args", []) or []
        ),
        max_dynamic_args=max_dynamic_args,
        suggested_control_actions=("ACTION3", "ACTION4"),
        control_policy="m3_dynamic_available_controls",
        metrics=metrics,
        success_metrics=success_metrics,
        diagnostic_metrics=diagnostic_metrics,
        expected_signal=expected_signal_for_family(family),
        falsification_criterion=falsification_for_family(family),
        planning_rationale=planning_rationale_for_family(family),
        resolution_basis=resolution_basis_for_family(family),
    )


def blocked_followup(
    decision: Mapping[str, Any],
    followup: Mapping[str, Any],
    group: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "source_a32_queue_item_id": str(decision.get("queue_item_id", "")),
        "source_a32_decision": str(decision.get("decision", "")),
        "source_a32_recommended_next_step": str(
            decision.get("recommended_next_step", "")
        ),
        "followup_family": str(followup.get("followup_family", "")),
        "target_action": str(followup.get("target_action", "")),
        "context_replay": list(group.get("context_replay", []) or []),
        "blocked_reason": str(group.get("blocked_reason", "")),
        "a32_decision_counted_as_confirmation": False,
        "status": BLOCKED_A32_PATCH_FOLLOWUP,
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def summarize_followup_planning(
    *,
    decisions: Sequence[Mapping[str, Any]],
    candidate_groups: Sequence[Mapping[str, Any]],
    requests: Sequence[A32RequestedPatchSimilarityFollowupRequest],
    blocked: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    selected_args = [
        dict(args)
        for request in requests
        for args in request.resolved_target_action_args
    ]
    return {
        "a32_revision_decisions_consumed": len(decisions),
        "a32_requested_followup_tests_seen": sum(
            len(decision.get("requested_followup_tests", []) or [])
            for decision in decisions
        ),
        "candidate_arg_groups": len(candidate_groups),
        "planned_followup_requests": len(requests),
        "blocked_followup_requests": len(blocked),
        "outside_known_y12_region_requests": len(
            [
                request
                for request in requests
                if request.followup_family == "outside_known_y12_region_probe"
            ]
        ),
        "alternate_repositioning_context_requests": len(
            [
                request
                for request in requests
                if request.followup_family == "alternate_repositioning_context_probe"
            ]
        ),
        "resolved_request_arg_pairs": len(selected_args),
        "unique_resolved_target_arg_sets": len({_args_key(args) for args in selected_args}),
        "a32_decision_counted_as_confirmation": False,
        "execution_performed": False,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "a32_remains_only_verdict_location": True,
    }


def expected_signal_for_family(family: str) -> str:
    if family == "outside_known_y12_region_probe":
        return "outside_region_targets_test_patch_similarity_scope_limit"
    return "alternate_repositioning_context_tests_patch_similarity_scope_limit"


def falsification_for_family(family: str) -> str:
    if family == "outside_known_y12_region_probe":
        return (
            "Outside-region ACTION6 args fail on local_patch_before_after and "
            "object_positions_before_after."
        )
    return (
        "Patch-similar ACTION6 args in the alternate repositioning context fail "
        "on grounded success metrics."
    )


def planning_rationale_for_family(family: str) -> str:
    if family == "outside_known_y12_region_probe":
        return (
            "A32.3 judged the current evidence mono-context and asked whether "
            "the rule holds outside the known success line/region."
        )
    return (
        "A32.3 asked whether the patch-similarity rule survives a different "
        "repositioning replay context."
    )


def resolution_basis_for_family(family: str) -> str:
    if family == "outside_known_y12_region_probe":
        return "nearest_success_patch_outside_known_y12_region"
    return "nearest_success_patch_in_alternate_repositioning_context"


def followup_request_id(
    decision: Mapping[str, Any],
    family: str,
    selected_args: Sequence[Mapping[str, Any]],
) -> str:
    queue = str(decision.get("queue_item_id", "")).replace("::", "_") or "unknown"
    if selected_args:
        args_token = "__".join(
            "_".join(f"{key}{args[key]}" for key in sorted(args))
            for args in selected_args
        )
    else:
        args_token = "dynamic"
    return f"m3_22::{queue}::{family}::{args_token}"


def dedupe_args(values: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], ...]:
    by_key = {_args_key(dict(value)): dict(value) for value in values if value}
    return tuple(by_key[key] for key in sorted(by_key))


def ordered_unique(values: Sequence[str]) -> Tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return tuple(result)


def _context_args_list(raw: Any) -> list[Dict[str, Any]] | None:
    if raw is None:
        return None
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return None
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def _context_args_tuple(raw: Any) -> Tuple[Dict[str, Any], ...] | None:
    items = _context_args_list(raw)
    if items is None:
        return None
    return tuple(items)


def _args_key(args: Mapping[str, Any]) -> str:
    return json.dumps({str(key): args[key] for key in sorted(args)}, sort_keys=True)


def write_a32_requested_patch_similarity_followup_requests(
    payload: Mapping[str, Any],
    output_path: str | Path = (
        DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_FOLLOWUP_REQUESTS_OUTPUT_PATH
    ),
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan M3.22 follow-ups requested by A32 patch review.",
    )
    parser.add_argument(
        "--a32-decisions",
        type=Path,
        default=DEFAULT_A32_PATCH_SIMILARITY_REVISION_DECISIONS_INPUT_PATH,
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--max-dynamic-args", type=int, default=DEFAULT_MAX_DYNAMIC_ARGS)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_A32_REQUESTED_PATCH_SIMILARITY_FOLLOWUP_REQUESTS_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_a32_requested_patch_similarity_followup_planning(
        a32_decisions_path=args.a32_decisions,
        environments_dir=args.environments_dir,
        max_dynamic_args=args.max_dynamic_args,
    )
    write_a32_requested_patch_similarity_followup_requests(payload, args.out)
    print(
        json.dumps(
            {
                "output_path": str(args.out),
                "summary": payload["summary"],
                "status": "UNRESOLVED",
                "revision_status": "CANDIDATE_ONLY",
                "support": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
