"""M3.11 planner for dynamic retargeting after repositioning."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from math import inf
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from theory.m2.m3_execution_smoke import (
    _execute_named_action,
    _make_env,
    _matching_actions,
    _reset_env,
)
from theory.non_ar25_active_micro_run import _configure_offline_env, _env_dir

from .m2_observation_refinement import M3_REFINEMENT_TRUTH_STATUS
from .refined_followup_executor import (
    DEFAULT_REFINED_FOLLOWUP_RESULTS_OUTPUT_PATH,
)
from .refined_followup_planner import (
    DEFAULT_REACTIVATION_METRICS,
    READY_FOR_M3_FOLLOWUP,
    falsification_criterion,
)


DEFAULT_DYNAMIC_RETARGET_REQUESTS_OUTPUT_PATH = (
    Path("diagnostics") / "m3" / "dynamic_retarget_followup_requests.json"
)
DYNAMIC_RETARGET_POLICY = "dynamic_retarget_after_repositioning"
DEFAULT_MAX_CANDIDATE_ARGS = 5


@dataclass(frozen=True)
class DynamicRetargetCandidate:
    """One candidate target-action argument set after repositioning."""

    action_args: Dict[str, Any]
    rank: int
    score: float
    generation_sources: Tuple[str, ...]
    nearest_motion_hint: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_args": dict(self.action_args),
            "rank": int(self.rank),
            "score": float(self.score),
            "generation_sources": list(self.generation_sources),
            "nearest_motion_hint": (
                dict(self.nearest_motion_hint)
                if self.nearest_motion_hint is not None
                else None
            ),
        }


@dataclass(frozen=True)
class DynamicRetargetFollowupRequest:
    """A candidate-only request for testing a new target location."""

    request_id: str
    source_refined_hypothesis_id: str
    source_followup_request_id: str
    source_hypothesis_ids: Tuple[str, ...]
    game_id: str
    hypothesis_tested: str
    context_replay: Tuple[str, ...]
    context_replay_args: Tuple[Dict[str, Any], ...] | None
    target_action: str
    target_action_args: Dict[str, Any]
    target_action_arg_policy: str
    excluded_args: Tuple[Dict[str, Any], ...]
    suggested_control_actions: Tuple[str, ...]
    metrics: Tuple[str, ...]
    expected_signal: str
    falsification_criteria: Tuple[Dict[str, Any], ...]
    candidate_arg_rank: int
    candidate_arg_score: float
    candidate_arg_generation_sources: Tuple[str, ...]
    planning_rationale: str
    control_policy: str = "m3_dynamic_available_controls"
    status: str = READY_FOR_M3_FOLLOWUP
    revision_status: str = "CANDIDATE_ONLY"
    support: int = 0
    controlled_test_required: bool = True
    truth_status: str = M3_REFINEMENT_TRUTH_STATUS
    revision_performed: bool = False
    wrong_confirmations: int = 0
    trace_support_counted_as_proof: bool = False
    prior_counted_as_proof: bool = False
    observation_counted_as_confirmation: bool = False
    followup_request_counted_as_support: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "source_refined_hypothesis_id": self.source_refined_hypothesis_id,
            "source_followup_request_id": self.source_followup_request_id,
            "source_hypothesis_ids": list(self.source_hypothesis_ids),
            "game_id": self.game_id,
            "hypothesis_tested": self.hypothesis_tested,
            "context_replay": list(self.context_replay),
            "context_replay_args": (
                [dict(item) for item in self.context_replay_args]
                if self.context_replay_args is not None
                else None
            ),
            "target_action": self.target_action,
            "target_action_args": dict(self.target_action_args),
            "target_action_arg_policy": self.target_action_arg_policy,
            "excluded_args": [dict(item) for item in self.excluded_args],
            "suggested_control_actions": list(self.suggested_control_actions),
            "control_policy": self.control_policy,
            "metrics": list(self.metrics),
            "expected_signal": self.expected_signal,
            "falsification_criteria": [
                dict(item) for item in self.falsification_criteria
            ],
            "candidate_arg_rank": int(self.candidate_arg_rank),
            "candidate_arg_score": float(self.candidate_arg_score),
            "candidate_arg_generation_sources": list(
                self.candidate_arg_generation_sources
            ),
            "planning_rationale": self.planning_rationale,
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
            "followup_request_counted_as_support": (
                self.followup_request_counted_as_support
            ),
        }


def run_dynamic_retarget_followup_planning(
    *,
    followup_results_path: str | Path = DEFAULT_REFINED_FOLLOWUP_RESULTS_OUTPUT_PATH,
    environments_dir: str | Path | None = None,
    max_candidate_args: int = DEFAULT_MAX_CANDIDATE_ARGS,
) -> Dict[str, Any]:
    payload = _load_json(followup_results_path)
    experiments = [
        dict(row) for row in payload.get("controlled_experiments", []) or []
    ]
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    _configure_offline_env(env_dir)

    requests: list[DynamicRetargetFollowupRequest] = []
    candidate_groups: list[Dict[str, Any]] = []
    skipped: list[Dict[str, Any]] = []
    for group in grouped_reactivation_experiments(experiments):
        candidates = generate_dynamic_retarget_candidates(
            group,
            environments_dir=env_dir,
            max_candidate_args=max_candidate_args,
        )
        candidate_groups.append(
            {
                "source_refined_hypothesis_id": str(
                    group.get("source_refined_hypothesis_id", "")
                ),
                "source_followup_request_id": str(group.get("request_id", "")),
                "excluded_args": [
                    dict(item) for item in group.get("excluded_args", []) or []
                ],
                "candidate_args": [candidate.to_dict() for candidate in candidates],
                "status": "CANDIDATE_ARGS_GENERATED" if candidates else "NO_CANDIDATES",
            }
        )
        if not candidates:
            skipped.append(skipped_group(group, reason="no_new_target_args_available"))
            continue
        for candidate in candidates:
            requests.append(build_dynamic_retarget_request(group, candidate))

    return {
        "config": {
            "followup_results_path": str(followup_results_path),
            "environments_dir": str(env_dir),
            "schema_version": "m3.dynamic_retarget_followup_requests.v1",
            "inputs_read": ["M3.10"],
            "artifacts_not_modified": ["M2", "M3.8", "M3.9", "M3.10", "A32", "A33"],
            "execution_performed": False,
            "max_candidate_args": int(max_candidate_args),
        },
        "summary": summarize_dynamic_retarget_planning(
            experiments=experiments,
            requests=requests,
            candidate_groups=candidate_groups,
            skipped=skipped,
            max_candidate_args=max_candidate_args,
        ),
        "candidate_arg_groups": [dict(item) for item in candidate_groups],
        "followup_experiment_requests": [request.to_dict() for request in requests],
        "skipped_retarget_groups": [dict(item) for item in skipped],
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
        "followup_request_counted_as_support": False,
        "a32_remains_only_verdict_location": True,
    }


def grouped_reactivation_experiments(
    experiments: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
    by_key: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in experiments:
        key = "::".join(
            [
                str(row.get("source_refined_hypothesis_id", "")),
                str(row.get("request_id", "")),
                str(row.get("game_id", "")),
                _stable_json(row.get("context_replay", []) or []),
                str(row.get("target_action", "")),
            ]
        )
        if not key.strip(":"):
            continue
        by_key[key].append(dict(row))

    groups: list[Dict[str, Any]] = []
    for rows in by_key.values():
        first = rows[0]
        target_action_args = [
            dict(row.get("target_action_args", {}) or {})
            for row in rows
            if isinstance(row.get("target_action_args"), Mapping)
        ]
        excluded = _dedupe_args(target_action_args)
        groups.append(
            {
                "source_refined_hypothesis_id": str(
                    first.get("source_refined_hypothesis_id", "")
                ),
                "request_id": str(first.get("request_id", "")),
                "source_hypothesis_ids": list(
                    first.get("source_hypothesis_ids", []) or []
                ),
                "game_id": str(first.get("game_id", "")),
                "context_replay": list(first.get("context_replay", []) or []),
                "context_replay_args": _context_args_list(
                    first.get("context_replay_args")
                ),
                "target_action": str(first.get("target_action", "")),
                "excluded_args": excluded,
                "controls_seen": tuple(
                    sorted(
                        {
                            str(row.get("control_action", ""))
                            for row in rows
                            if row.get("control_action")
                        }
                    )
                ),
                "experiments": rows,
            }
        )
    return tuple(groups)


def generate_dynamic_retarget_candidates(
    group: Mapping[str, Any],
    *,
    environments_dir: str | Path,
    max_candidate_args: int,
) -> Tuple[DynamicRetargetCandidate, ...]:
    target_action = str(group.get("target_action", ""))
    live_args = available_target_args_after_replay(
        game_id=str(group.get("game_id", "")),
        context_replay=group.get("context_replay", []) or [],
        context_replay_args=group.get("context_replay_args"),
        target_action=target_action,
        environments_dir=environments_dir,
    )
    excluded = [dict(item) for item in group.get("excluded_args", []) or []]
    excluded_keys = {_args_key(args) for args in excluded}
    motion_hints = motion_offset_hints(group.get("experiments", []) or [], excluded)
    candidates: list[DynamicRetargetCandidate] = []
    for args in live_args:
        if _args_key(args) in excluded_keys:
            continue
        score, nearest = score_candidate_args(args, motion_hints)
        sources = ["live_available_target_after_replay"]
        if nearest is not None and nearest.get("distance", inf) <= 6:
            sources.append("object_motion_offset_hint")
        if any_changed_pixel_effect(group.get("experiments", []) or []):
            sources.append("changed_pixels_effect_present")
        candidates.append(
            DynamicRetargetCandidate(
                action_args=dict(args),
                rank=0,
                score=score,
                generation_sources=tuple(sources),
                nearest_motion_hint=nearest,
            )
        )
    candidates = sorted(candidates, key=lambda item: (item.score, _args_key(item.action_args)))
    ranked = [
        DynamicRetargetCandidate(
            action_args=candidate.action_args,
            rank=index,
            score=candidate.score,
            generation_sources=candidate.generation_sources,
            nearest_motion_hint=candidate.nearest_motion_hint,
        )
        for index, candidate in enumerate(candidates[: max(0, int(max_candidate_args))], start=1)
    ]
    return tuple(ranked)


def available_target_args_after_replay(
    *,
    game_id: str,
    context_replay: Sequence[str],
    context_replay_args: Any,
    target_action: str,
    environments_dir: str | Path,
) -> Tuple[Dict[str, Any], ...]:
    env = _make_env(game_id, environments_dir)
    _reset_env(env)
    replay_args = list(context_replay_args or [])
    for index, action_name in enumerate(context_replay):
        action_args = replay_args[index] if index < len(replay_args) else None
        _execute_named_action(
            env,
            str(action_name),
            required_observation="",
            action_args=action_args if isinstance(action_args, Mapping) else None,
        )
    return tuple(
        _dedupe_args(
            [
                dict(getattr(action, "action_args", {}) or {})
                for action in _matching_actions(env, target_action)
            ]
        )
    )


def motion_offset_hints(
    experiments: Sequence[Mapping[str, Any]],
    excluded_args: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
    anchors = [
        dict(args)
        for args in excluded_args
        if _safe_number(args.get("x")) is not None and _safe_number(args.get("y")) is not None
    ]
    if not anchors:
        return ()
    hints: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in experiments:
        if str(row.get("metric", "")) != "object_positions_before_after":
            continue
        for side in ("observed_baseline", "observed_perturbation"):
            observation = dict(row.get(side, {}) or {})
            for vector in observation.get("motion_vectors", []) or []:
                dx = _safe_number(dict(vector).get("dx"))
                dy = _safe_number(dict(vector).get("dy"))
                if dx is None or dy is None:
                    continue
                if abs(dx) < 1 and abs(dy) < 1:
                    continue
                for anchor in anchors:
                    ax = float(anchor["x"])
                    ay = float(anchor["y"])
                    for sign in (1, -1):
                        predicted = {
                            "x": int(round(ax + sign * dx)),
                            "y": int(round(ay + sign * dy)),
                        }
                        key = _stable_json(predicted)
                        if key in seen:
                            continue
                        seen.add(key)
                        hints.append(
                            {
                                "predicted_args": predicted,
                                "dx": dx,
                                "dy": dy,
                                "sign": sign,
                                "source_metric": "object_positions_before_after",
                                "source_side": side,
                                "source_control_action": str(
                                    row.get("control_action", "")
                                ),
                                "source_color": dict(vector).get("color"),
                                "source_size": dict(vector).get("size"),
                            }
                        )
    return tuple(hints)


def score_candidate_args(
    args: Mapping[str, Any],
    motion_hints: Sequence[Mapping[str, Any]],
) -> Tuple[float, Dict[str, Any] | None]:
    x = _safe_number(args.get("x"))
    y = _safe_number(args.get("y"))
    if x is None or y is None or not motion_hints:
        return (1000.0, None)
    best_distance = inf
    best_hint: Dict[str, Any] | None = None
    for hint in motion_hints:
        predicted = dict(hint.get("predicted_args", {}) or {})
        hx = _safe_number(predicted.get("x"))
        hy = _safe_number(predicted.get("y"))
        if hx is None or hy is None:
            continue
        distance = abs(float(x) - hx) + abs(float(y) - hy)
        if distance < best_distance:
            best_distance = distance
            best_hint = {**dict(hint), "distance": float(distance)}
    return (float(best_distance if best_hint is not None else 1000.0), best_hint)


def any_changed_pixel_effect(experiments: Sequence[Mapping[str, Any]]) -> bool:
    for row in experiments:
        if str(row.get("metric", "")) != "changed_pixels":
            continue
        baseline = float(row.get("baseline_signal", 0.0) or 0.0)
        perturbation = float(row.get("perturbation_signal", 0.0) or 0.0)
        if baseline > 0 or perturbation > 0:
            return True
    return False


def build_dynamic_retarget_request(
    group: Mapping[str, Any],
    candidate: DynamicRetargetCandidate,
) -> DynamicRetargetFollowupRequest:
    source_refined_id = str(group.get("source_refined_hypothesis_id", ""))
    target_action = str(group.get("target_action", ""))
    target_args = dict(candidate.action_args)
    return DynamicRetargetFollowupRequest(
        request_id=dynamic_retarget_request_id(
            source_refined_id=source_refined_id,
            target_action=target_action,
            rank=candidate.rank,
            target_args=target_args,
        ),
        source_refined_hypothesis_id=source_refined_id,
        source_followup_request_id=str(group.get("request_id", "")),
        source_hypothesis_ids=tuple(
            str(item) for item in group.get("source_hypothesis_ids", []) or []
        ),
        game_id=str(group.get("game_id", "")),
        hypothesis_tested=(
            f"ACTION4 may create a new {target_action} target after repositioning"
        ),
        context_replay=tuple(str(item) for item in group.get("context_replay", []) or []),
        context_replay_args=_context_args_tuple(group.get("context_replay_args")),
        target_action=target_action,
        target_action_args=target_args,
        target_action_arg_policy=DYNAMIC_RETARGET_POLICY,
        excluded_args=tuple(dict(item) for item in group.get("excluded_args", []) or []),
        suggested_control_actions=tuple(str(item) for item in group.get("controls_seen", []) or []),
        metrics=DEFAULT_REACTIVATION_METRICS,
        expected_signal="dynamic_retargeted_action_signal_exceeds_controls",
        falsification_criteria=tuple(
            falsification_criterion(metric) for metric in DEFAULT_REACTIVATION_METRICS
        ),
        candidate_arg_rank=candidate.rank,
        candidate_arg_score=candidate.score,
        candidate_arg_generation_sources=candidate.generation_sources,
        planning_rationale=(
            "M3.10 did not support reusing the consumed ACTION6 args; M3.11 "
            "therefore proposes bounded new ACTION6 args available after the "
            "repositioning context."
        ),
    )


def dynamic_retarget_request_id(
    *,
    source_refined_id: str,
    target_action: str,
    rank: int,
    target_args: Mapping[str, Any],
) -> str:
    source = source_refined_id.replace("::", "_") or "unknown_refined"
    args_token = "_".join(f"{key}{target_args[key]}" for key in sorted(target_args))
    return f"m3_11::{source}::retarget_{target_action}_{rank:02d}_{args_token}"


def summarize_dynamic_retarget_planning(
    *,
    experiments: Sequence[Mapping[str, Any]],
    requests: Sequence[DynamicRetargetFollowupRequest],
    candidate_groups: Sequence[Mapping[str, Any]],
    skipped: Sequence[Mapping[str, Any]],
    max_candidate_args: int,
) -> Dict[str, Any]:
    return {
        "followup_results_consumed": 1 if experiments else 0,
        "source_experiments_consumed": len(experiments),
        "retarget_groups": len(candidate_groups),
        "candidate_args_generated": sum(
            len(group.get("candidate_args", []) or []) for group in candidate_groups
        ),
        "followup_requests_generated": len(requests),
        "max_candidate_args": int(max_candidate_args),
        "excluded_args_count": sum(
            len(group.get("excluded_args", []) or []) for group in candidate_groups
        ),
        "skipped_retarget_groups": len(skipped),
        "execution_performed": False,
        "status": "UNRESOLVED",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "controlled_test_required": True,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
        "followup_request_counted_as_support": False,
        "a32_remains_only_verdict_location": True,
    }


def skipped_group(group: Mapping[str, Any], *, reason: str) -> Dict[str, Any]:
    return {
        "source_refined_hypothesis_id": str(group.get("source_refined_hypothesis_id", "")),
        "source_followup_request_id": str(group.get("request_id", "")),
        "target_action": str(group.get("target_action", "")),
        "reason": reason,
        "status": "BLOCKED_NO_DYNAMIC_RETARGET",
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "truth_status": M3_REFINEMENT_TRUTH_STATUS,
        "revision_performed": False,
        "wrong_confirmations": 0,
    }


def write_dynamic_retarget_followup_requests(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_DYNAMIC_RETARGET_REQUESTS_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _dedupe_args(values: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    result: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        args = dict(value)
        key = _args_key(args)
        if key in seen:
            continue
        seen.add(key)
        result.append(args)
    return result


def _args_key(args: Mapping[str, Any]) -> str:
    return _stable_json({str(key): args[key] for key in sorted(args)})


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


def _safe_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build M3.11 dynamic retarget follow-up requests.",
    )
    parser.add_argument(
        "--followup-results",
        type=Path,
        default=DEFAULT_REFINED_FOLLOWUP_RESULTS_OUTPUT_PATH,
    )
    parser.add_argument("--environments-dir", type=Path, default=None)
    parser.add_argument("--max-candidate-args", type=int, default=DEFAULT_MAX_CANDIDATE_ARGS)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_DYNAMIC_RETARGET_REQUESTS_OUTPUT_PATH,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_dynamic_retarget_followup_planning(
        followup_results_path=args.followup_results,
        environments_dir=args.environments_dir,
        max_candidate_args=args.max_candidate_args,
    )
    write_dynamic_retarget_followup_requests(payload, args.out)
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
