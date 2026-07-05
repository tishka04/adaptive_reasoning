"""M3.18 planner for expanding patch-similarity retarget affordances."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir

from .dynamic_retarget_selection_rule_consolidation import (
    DEFAULT_DYNAMIC_RETARGET_SELECTION_RULE_CONSOLIDATION_OUTPUT_PATH,
)
from .dynamic_retarget_selection_followup_executor import (
    available_args_for_request,
    grid_after_replay,
    local_patch_signature,
    patch_distance,
)
from .m2_observation_refinement import M3_REFINEMENT_TRUTH_STATUS


DEFAULT_DYNAMIC_RETARGET_PATCH_SIMILARITY_EXPANSION_REQUESTS_OUTPUT_PATH = (
    Path("diagnostics")
    / "m3"
    / "dynamic_retarget_patch_similarity_expansion_requests.json"
)
PATCH_SIMILARITY_EXPANSION_POLICY = "patch_similarity_excluding_known_args"
READY_FOR_M3_PATCH_EXPANSION = "READY_FOR_M3_PATCH_EXPANSION"
DEFAULT_MAX_DYNAMIC_ARGS = 3
EXPANSION_PROBE_FAMILIES = (
    "success_patch_similarity_expansion",
    "failure_patch_negative_control_expansion",
    "mixed_patch_boundary_probe",
)


@dataclass(frozen=True)
class PatchSimilarityExpansionRequest:
    """Candidate-only request for testing whether patch similarity generalizes."""

    request_id: str
    source_selection_rule_consolidation_id: str
    source_selection_rule_candidate_id: str
    source_mechanism_candidate_id: str
    game_id: str
    rule_family: str
    probe_family: str
    hypothesis_tested: str
    context_replay: Tuple[str, ...]
    context_replay_args: Tuple[Dict[str, Any], ...] | None
    target_action: str
    target_action_args: Dict[str, Any] | None
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
    status: str = READY_FOR_M3_PATCH_EXPANSION
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
    expansion_request_counted_as_support: bool = False
    execution_performed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "source_selection_rule_consolidation_id": (
                self.source_selection_rule_consolidation_id
            ),
            "source_selection_rule_candidate_id": self.source_selection_rule_candidate_id,
            "source_mechanism_candidate_id": self.source_mechanism_candidate_id,
            "game_id": self.game_id,
            "rule_family": self.rule_family,
            "probe_family": self.probe_family,
            "hypothesis_tested": self.hypothesis_tested,
            "context_replay": list(self.context_replay),
            "context_replay_args": (
                [dict(item) for item in self.context_replay_args]
                if self.context_replay_args is not None
                else None
            ),
            "target_action": self.target_action,
            "target_action_args": (
                dict(self.target_action_args)
                if self.target_action_args is not None
                else None
            ),
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
            "expansion_request_counted_as_support": (
                self.expansion_request_counted_as_support
            ),
            "execution_performed": self.execution_performed,
        }


def run_dynamic_retarget_patch_similarity_expansion_planning(
    *,
    selection_rule_consolidation_path: str | Path = (
        DEFAULT_DYNAMIC_RETARGET_SELECTION_RULE_CONSOLIDATION_OUTPUT_PATH
    ),
    environments_dir: str | Path | None = None,
    max_dynamic_args: int = DEFAULT_MAX_DYNAMIC_ARGS,
) -> Dict[str, Any]:
    payload = _load_json(selection_rule_consolidation_path)
    consolidations = [
        dict(row) for row in payload.get("selection_rule_consolidations", []) or []
    ]
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    _configure_offline_env(env_dir)

    requests: list[PatchSimilarityExpansionRequest] = []
    candidate_groups: list[Dict[str, Any]] = []
    blocked: list[Dict[str, Any]] = []
    for consolidation in consolidations:
        group = build_patch_similarity_candidate_group(
            consolidation,
            environments_dir=env_dir,
            max_dynamic_args=max_dynamic_args,
        )
        candidate_groups.append(group)
        for probe_family in EXPANSION_PROBE_FAMILIES:
            selected = tuple(
                dict(item)
                for item in group.get("selected_by_probe_family", {}).get(
                    probe_family, []
                )
            )
            if not selected:
                blocked.append(blocked_expansion_request(consolidation, probe_family))
                continue
            requests.append(
                build_patch_similarity_expansion_request(
                    consolidation=consolidation,
                    probe_family=probe_family,
                    selected_candidates=selected,
                    all_candidates=tuple(
                        dict(item) for item in group.get("candidate_resolution_args", [])
                    ),
                    max_dynamic_args=max_dynamic_args,
                )
            )

    return {
        "config": {
            "selection_rule_consolidation_path": str(selection_rule_consolidation_path),
            "environments_dir": str(env_dir),
            "schema_version": "m3.dynamic_retarget_patch_similarity_expansion_requests.v1",
            "inputs_read": ["M3.17"],
            "artifacts_not_modified": [
                "M2",
                "M3.8",
                "M3.9",
                "M3.10",
                "M3.11",
                "M3.12",
                "M3.13",
                "M3.14",
                "M3.15",
                "M3.16",
                "M3.17",
                "A32",
                "A33",
            ],
            "execution_performed": False,
            "max_dynamic_args": int(max_dynamic_args),
        },
        "summary": summarize_patch_similarity_expansion(
            consolidations=consolidations,
            candidate_groups=candidate_groups,
            requests=requests,
            blocked=blocked,
        ),
        "candidate_arg_groups": [dict(item) for item in candidate_groups],
        "expansion_experiment_requests": [request.to_dict() for request in requests],
        "blocked_expansion_requests": [dict(item) for item in blocked],
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
        "expansion_request_counted_as_support": False,
        "a32_remains_only_verdict_location": True,
    }


def build_patch_similarity_candidate_group(
    consolidation: Mapping[str, Any],
    *,
    environments_dir: str | Path,
    max_dynamic_args: int,
) -> Dict[str, Any]:
    live_args = available_args_for_request(consolidation, environments_dir=environments_dir)
    excluded = dedupe_args(consolidation.get("excluded_args_for_next_expansion", []) or [])
    excluded_keys = {_args_key(args) for args in excluded}
    available_after_exclusion = [
        dict(args) for args in live_args if _args_key(args) not in excluded_keys
    ]
    replay_grid = grid_after_replay(consolidation, environments_dir=environments_dir)
    success_seeds = dedupe_args(
        (
            *[dict(item) for item in consolidation.get("known_successful_retargets", []) or []],
            *[dict(item) for item in consolidation.get("unique_new_successful_args", []) or []],
        )
    )
    failure_seeds = dedupe_args(
        [dict(item) for item in consolidation.get("known_failed_retargets", []) or []]
    )
    candidates = tuple(
        score_patch_similarity_candidate(
            args,
            replay_grid=replay_grid,
            success_seeds=success_seeds,
            failure_seeds=failure_seeds,
        )
        for args in available_after_exclusion
    )
    selected = {
        "success_patch_similarity_expansion": select_candidates(
            candidates,
            mode="success",
            max_dynamic_args=max_dynamic_args,
        ),
        "failure_patch_negative_control_expansion": select_candidates(
            candidates,
            mode="failure",
            max_dynamic_args=max_dynamic_args,
        ),
        "mixed_patch_boundary_probe": select_candidates(
            candidates,
            mode="mixed",
            max_dynamic_args=max_dynamic_args,
        ),
    }
    return {
        "source_selection_rule_consolidation_id": str(
            consolidation.get("selection_rule_consolidation_id", "")
        ),
        "game_id": str(consolidation.get("game_id", "")),
        "target_action": str(consolidation.get("target_action", "")),
        "context_replay": list(consolidation.get("context_replay", []) or []),
        "context_replay_args": _context_args_list(
            consolidation.get("context_replay_args")
        ),
        "live_available_args": [dict(item) for item in live_args],
        "excluded_args": [dict(item) for item in excluded],
        "available_args_after_exclusion": [
            dict(item) for item in available_after_exclusion
        ],
        "seed_successful_args": [dict(item) for item in success_seeds],
        "seed_failed_args": [dict(item) for item in failure_seeds],
        "candidate_resolution_args": [dict(item) for item in candidates],
        "selected_by_probe_family": {
            family: [dict(item) for item in rows] for family, rows in selected.items()
        },
        "max_dynamic_args": int(max_dynamic_args),
        "status": (
            "CANDIDATE_ARGS_GENERATED"
            if available_after_exclusion
            else "NO_NEW_LIVE_ARGS_AFTER_EXCLUSION"
        ),
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "wrong_confirmations": 0,
    }


def score_patch_similarity_candidate(
    action_args: Mapping[str, Any],
    *,
    replay_grid: Any,
    success_seeds: Sequence[Mapping[str, Any]],
    failure_seeds: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    signature = local_patch_signature(replay_grid, action_args)
    success_distance = nearest_patch_distance(signature, replay_grid, success_seeds)
    failure_distance = nearest_patch_distance(signature, replay_grid, failure_seeds)
    boundary_score = abs(success_distance - failure_distance)
    return {
        "action_args": dict(action_args),
        "success_patch_distance": float(success_distance),
        "failure_patch_distance": float(failure_distance),
        "mixed_patch_boundary_score": float(boundary_score),
        "similarity_interpretation": similarity_interpretation(
            success_distance=success_distance,
            failure_distance=failure_distance,
        ),
        "patch_signature": [list(row) for row in signature],
        "excluded_known_args_respected": True,
        "status": "UNRESOLVED",
        "support": 0,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "wrong_confirmations": 0,
    }


def nearest_patch_distance(
    signature: Sequence[Sequence[int]],
    replay_grid: Any,
    seeds: Sequence[Mapping[str, Any]],
) -> float:
    if not seeds:
        return 1_000_000.0
    return float(
        min(
            patch_distance(signature, local_patch_signature(replay_grid, seed))
            for seed in seeds
        )
    )


def similarity_interpretation(*, success_distance: float, failure_distance: float) -> str:
    if success_distance < failure_distance:
        return "success_like"
    if failure_distance < success_distance:
        return "failure_like"
    return "ambiguous_between_success_and_failure"


def select_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    max_dynamic_args: int,
) -> Tuple[Dict[str, Any], ...]:
    if mode == "success":
        key = lambda row: (
            float(row.get("success_patch_distance", 0.0) or 0.0),
            float(row.get("failure_patch_distance", 0.0) or 0.0),
            _args_key(row.get("action_args", {}) or {}),
        )
        basis = "nearest_patch_signature_to_success_seeds_excluding_known_args"
    elif mode == "failure":
        key = lambda row: (
            float(row.get("failure_patch_distance", 0.0) or 0.0),
            float(row.get("success_patch_distance", 0.0) or 0.0),
            _args_key(row.get("action_args", {}) or {}),
        )
        basis = "nearest_patch_signature_to_failure_seed_excluding_known_args"
    else:
        key = lambda row: (
            float(row.get("mixed_patch_boundary_score", 0.0) or 0.0),
            min(
                float(row.get("success_patch_distance", 0.0) or 0.0),
                float(row.get("failure_patch_distance", 0.0) or 0.0),
            ),
            _args_key(row.get("action_args", {}) or {}),
        )
        basis = "nearest_ambiguous_patch_boundary_excluding_known_args"
    selected = []
    for rank, row in enumerate(sorted(candidates, key=key)[: max(0, int(max_dynamic_args))], start=1):
        value = dict(row)
        value["candidate_arg_rank"] = rank
        value["resolution_basis"] = basis
        selected.append(value)
    return tuple(selected)


def build_patch_similarity_expansion_request(
    *,
    consolidation: Mapping[str, Any],
    probe_family: str,
    selected_candidates: Sequence[Mapping[str, Any]],
    all_candidates: Sequence[Mapping[str, Any]],
    max_dynamic_args: int,
) -> PatchSimilarityExpansionRequest:
    selected_args = tuple(
        dict(row.get("action_args", {}) or {}) for row in selected_candidates
    )
    target_args = dict(selected_args[0]) if len(selected_args) == 1 else None
    success_metrics = tuple(
        str(metric) for metric in consolidation.get("success_metrics", []) or []
    )
    diagnostic_metrics = tuple(
        str(metric) for metric in consolidation.get("diagnostic_metrics", []) or []
    )
    metrics = ordered_unique((*success_metrics, *diagnostic_metrics))
    return PatchSimilarityExpansionRequest(
        request_id=patch_similarity_expansion_request_id(
            consolidation_id=str(
                consolidation.get("selection_rule_consolidation_id", "")
            ),
            probe_family=probe_family,
            selected_args=selected_args,
        ),
        source_selection_rule_consolidation_id=str(
            consolidation.get("selection_rule_consolidation_id", "")
        ),
        source_selection_rule_candidate_id=str(
            consolidation.get("source_selection_rule_candidate_id", "")
        ),
        source_mechanism_candidate_id=str(
            consolidation.get("source_mechanism_candidate_id", "")
        ),
        game_id=str(consolidation.get("game_id", "")),
        rule_family="local_patch_transformability",
        probe_family=probe_family,
        hypothesis_tested=hypothesis_for_probe(probe_family),
        context_replay=tuple(
            str(item) for item in consolidation.get("context_replay", []) or []
        ),
        context_replay_args=_context_args_tuple(consolidation.get("context_replay_args")),
        target_action=str(consolidation.get("target_action", "")),
        target_action_args=target_args,
        target_action_arg_policy=PATCH_SIMILARITY_EXPANSION_POLICY,
        resolved_target_action_args=selected_args,
        candidate_resolution_args=tuple(dict(item) for item in all_candidates),
        excluded_args=tuple(
            dict(item)
            for item in consolidation.get("excluded_args_for_next_expansion", []) or []
        ),
        seed_successful_args=tuple(
            dict(item)
            for item in (
                list(consolidation.get("known_successful_retargets", []) or [])
                + list(consolidation.get("unique_new_successful_args", []) or [])
            )
        ),
        seed_failed_args=tuple(
            dict(item) for item in consolidation.get("known_failed_retargets", []) or []
        ),
        max_dynamic_args=int(max_dynamic_args),
        suggested_control_actions=("ACTION3", "ACTION4"),
        control_policy="m3_dynamic_available_controls",
        metrics=metrics,
        success_metrics=success_metrics,
        diagnostic_metrics=diagnostic_metrics,
        expected_signal="patch_similarity_expansion_target_matches_grounded_success_metrics",
        falsification_criterion=falsification_for_probe(probe_family),
        planning_rationale=planning_rationale_for_probe(probe_family),
        resolution_basis=resolution_basis_for_probe(probe_family),
    )


def summarize_patch_similarity_expansion(
    *,
    consolidations: Sequence[Mapping[str, Any]],
    candidate_groups: Sequence[Mapping[str, Any]],
    requests: Sequence[PatchSimilarityExpansionRequest],
    blocked: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    resolved_occurrences = [
        dict(args)
        for request in requests
        for args in request.resolved_target_action_args
    ]
    unique_resolved = {_args_key(args) for args in resolved_occurrences}
    probe_families = {
        family: len([request for request in requests if request.probe_family == family])
        for family in EXPANSION_PROBE_FAMILIES
    }
    return {
        "selection_rule_consolidations_consumed": len(consolidations),
        "candidate_arg_groups": len(candidate_groups),
        "live_available_args_seen": sum(
            len(group.get("live_available_args", []) or []) for group in candidate_groups
        ),
        "excluded_args_count": sum(
            len(group.get("excluded_args", []) or []) for group in candidate_groups
        ),
        "available_args_after_exclusion": sum(
            len(group.get("available_args_after_exclusion", []) or [])
            for group in candidate_groups
        ),
        "candidate_resolution_args": sum(
            len(group.get("candidate_resolution_args", []) or [])
            for group in candidate_groups
        ),
        "expansion_requests_generated": len(requests),
        "blocked_expansion_requests": len(blocked),
        "probe_families": probe_families,
        "resolved_request_arg_pairs": len(resolved_occurrences),
        "unique_resolved_target_arg_sets": len(unique_resolved),
        "duplicate_resolved_target_arg_sets": max(
            0,
            len(resolved_occurrences) - len(unique_resolved),
        ),
        "duplicate_resolved_target_arg_sets_counted_as_independent": False,
        "execution_performed": False,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "expansion_request_counted_as_support": False,
        "a32_remains_only_verdict_location": True,
    }


def blocked_expansion_request(
    consolidation: Mapping[str, Any],
    probe_family: str,
) -> Dict[str, Any]:
    return {
        "source_selection_rule_consolidation_id": str(
            consolidation.get("selection_rule_consolidation_id", "")
        ),
        "probe_family": probe_family,
        "target_action": str(consolidation.get("target_action", "")),
        "blocked_reason": "no_new_patch_similarity_args_after_exclusion",
        "status": "BLOCKED_NOT_PLANNED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def patch_similarity_expansion_request_id(
    *,
    consolidation_id: str,
    probe_family: str,
    selected_args: Sequence[Mapping[str, Any]],
) -> str:
    source = consolidation_id.replace("::", "_") or "unknown_consolidation"
    if selected_args:
        args_token = "__".join(
            "_".join(f"{key}{args[key]}" for key in sorted(args))
            for args in selected_args
        )
    else:
        args_token = "dynamic"
    return f"m3_18::{source}::{probe_family}::{args_token}"


def hypothesis_for_probe(probe_family: str) -> str:
    if probe_family == "success_patch_similarity_expansion":
        return "patch-similar live ACTION6 targets generalize beyond the first attractor"
    if probe_family == "failure_patch_negative_control_expansion":
        return "failure-patch-similar live ACTION6 targets act as negative controls"
    return "mixed patch candidates reveal the boundary of ACTION6 retarget validity"


def falsification_for_probe(probe_family: str) -> str:
    if probe_family == "success_patch_similarity_expansion":
        return (
            "Patch-similar targets selected outside known args fail on "
            "local_patch_before_after and object_positions_before_after."
        )
    if probe_family == "failure_patch_negative_control_expansion":
        return (
            "Failure-patch-similar targets unexpectedly succeed on grounded "
            "retarget success metrics."
        )
    return (
        "Boundary candidates do not separate success-like and failure-like patch "
        "features under grounded retarget metrics."
    )


def planning_rationale_for_probe(probe_family: str) -> str:
    if probe_family == "success_patch_similarity_expansion":
        return (
            "M3.17 found one patch-similar affordance; this asks whether the "
            "success-patch selector finds additional live affordances when known "
            "args are excluded."
        )
    if probe_family == "failure_patch_negative_control_expansion":
        return (
            "A failure-similar live target is useful as a negative control for "
            "the local-patch transformability rule."
        )
    return (
        "Candidates close to both success and failure patches test whether the "
        "rule needs a richer boundary than pure patch similarity."
    )


def resolution_basis_for_probe(probe_family: str) -> str:
    if probe_family == "success_patch_similarity_expansion":
        return "nearest_patch_signature_to_success_seeds_excluding_known_args"
    if probe_family == "failure_patch_negative_control_expansion":
        return "nearest_patch_signature_to_failure_seed_excluding_known_args"
    return "nearest_ambiguous_patch_boundary_excluding_known_args"


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


def write_dynamic_retarget_patch_similarity_expansion_requests(
    payload: Mapping[str, Any],
    output_path: str | Path = (
        DEFAULT_DYNAMIC_RETARGET_PATCH_SIMILARITY_EXPANSION_REQUESTS_OUTPUT_PATH
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
        description="Plan M3.18 patch-similarity expansion requests.",
    )
    parser.add_argument(
        "--selection-rule-consolidation",
        type=Path,
        default=DEFAULT_DYNAMIC_RETARGET_SELECTION_RULE_CONSOLIDATION_OUTPUT_PATH,
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument(
        "--max-dynamic-args",
        type=int,
        default=DEFAULT_MAX_DYNAMIC_ARGS,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_DYNAMIC_RETARGET_PATCH_SIMILARITY_EXPANSION_REQUESTS_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_dynamic_retarget_patch_similarity_expansion_planning(
        selection_rule_consolidation_path=args.selection_rule_consolidation,
        environments_dir=args.environments_dir,
        max_dynamic_args=args.max_dynamic_args,
    )
    write_dynamic_retarget_patch_similarity_expansion_requests(payload, args.out)
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
