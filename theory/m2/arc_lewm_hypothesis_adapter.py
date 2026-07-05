"""M2.14e-mini adapter from ARC-LeWM signals to M2/M3 candidates."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .arc_lewm_signal_extractor import (
    DEFAULT_ARC_LEWM_SIGNAL_REPORT_OUTPUT_PATH,
    candidate_only_contract,
)
from .metric_registry import default_falsification_for_metric
from .normalizer import default_controls_for_action
from .schema import (
    ContextSnapshot,
    FalsificationCriterion,
    FrontierConditionedHypothesis,
    HypothesisTestability,
    M2_BLOCKED_FOR_M3_STATUS,
    M2_DYNAMIC_CONTROL_POLICY,
    M2_HYPOTHESIS_STATUS,
    M2_READY_FOR_M3_STATUS,
    M2_SCHEMA_VERSION,
    M2_TRUTH_STATUS,
    M3CandidateExperimentRequest,
    SourceGenerationAudit,
)
from .testability_compiler import request_id_for_hypothesis, write_m3_requests_payload
from .validators import validate_hypothesis, validate_m3_request


DEFAULT_ARC_LEWM_HYPOTHESES_OUTPUT_PATH = (
    Path("diagnostics") / "m2" / "arc_lewm_hypotheses.json"
)
DEFAULT_ARC_LEWM_M3_REQUESTS_OUTPUT_PATH = (
    Path("diagnostics") / "m2" / "arc_lewm_m3_candidate_requests_v2.json"
)
ARC_LEWM_HYPOTHESES_SCHEMA_VERSION = "m2.arc_lewm_hypotheses.v1"
ARC_LEWM_TO_M3_SCHEMA_VERSION = "m2.arc_lewm_to_m3.v2"
CANDIDATE_REVISION_STATUS = "CANDIDATE_ONLY"
ARC_LEWM_REPLAYABILITY = "OFFLINE_TRACE_CONTEXT_ONLY"
ARC_LEWM_CONTEXT_STATE_ORIGIN = "human_trace_frame_before"


def run_arc_lewm_hypothesis_adapter(
    *,
    signal_report_path: str | Path = DEFAULT_ARC_LEWM_SIGNAL_REPORT_OUTPUT_PATH,
    hypotheses_output_path: str | Path | None = None,
    m3_requests_output_path: str | Path | None = None,
    max_hypotheses: int = 4,
) -> Dict[str, Any]:
    signal_report = _load_json(signal_report_path)
    _validate_signal_report_contract(signal_report)
    hypotheses = generate_arc_lewm_hypotheses(signal_report, max_hypotheses=max_hypotheses)
    invalid = [
        {
            "hypothesis_id": hypothesis.hypothesis_id,
            "errors": list(validate_hypothesis(hypothesis).errors),
        }
        for hypothesis in hypotheses
        if not validate_hypothesis(hypothesis).valid
    ]
    if invalid:
        raise ValueError(f"invalid ARC-LeWM hypotheses generated: {invalid}")
    hypothesis_payload = build_arc_lewm_hypothesis_payload(
        signal_report_path=str(signal_report_path),
        signal_report=signal_report,
        hypotheses=hypotheses,
    )
    m3_payload = build_arc_lewm_m3_requests_payload(
        hypotheses,
        source_hypothesis_path=str(hypotheses_output_path or DEFAULT_ARC_LEWM_HYPOTHESES_OUTPUT_PATH),
    )
    if hypotheses_output_path is not None:
        write_arc_lewm_hypotheses(hypothesis_payload, hypotheses_output_path)
    if m3_requests_output_path is not None:
        write_m3_requests_payload(m3_payload, m3_requests_output_path)
    return {"hypotheses_payload": hypothesis_payload, "m3_payload": m3_payload}


def generate_arc_lewm_hypotheses(
    signal_report: Mapping[str, Any],
    *,
    max_hypotheses: int = 4,
) -> tuple[FrontierConditionedHypothesis, ...]:
    signals = dict(signal_report.get("signals", {}) or {})
    candidates: list[FrontierConditionedHypothesis] = []
    high_surprise = list(signals.get("high_surprise_transitions", []) or [])
    proxy_gaps = list(signals.get("proxy_completion_gap_candidates", []) or [])
    clusters = list(signals.get("action_conditioned_delta_clusters", []) or [])
    terminal = dict(signals.get("terminal_like_latent_neighborhoods", {}) or {})
    terminal_rows = list(terminal.get("highest_surprise_terminal_transitions", []) or [])

    if high_surprise:
        candidates.append(
            _make_hypothesis(
                index=1,
                family="latent_surprise_frontier",
                signal_family="high_surprise_transitions",
                signal=high_surprise[0],
                metric="local_patch_before_after",
                effect=(
                    "High ARC-LeWM latent prediction error marks a transition family "
                    "whose observed local effect should be tested against controls."
                ),
                expected_signal=(
                    "high_surprise_action_context_produces_distinct_grounded_local_effect"
                ),
                support_condition=(
                    "target_action_grounded_local_effect differs from matched controls"
                ),
                failure_condition=(
                    "target_action_grounded_local_effect does not differ from matched controls"
                ),
                priority=float(high_surprise[0].get("latent_surprise_score", 0.0)),
            )
        )
    if proxy_gaps:
        candidates.append(
            _make_hypothesis(
                index=2,
                family="proxy_completion_gap_candidate",
                signal_family="proxy_completion_gap_candidates",
                signal=proxy_gaps[0],
                metric="levels_completed_after_rollout",
                effect=(
                    "High latent surprise without level completion may identify a "
                    "proxy/completion gap that needs grounded M3 testing."
                ),
                expected_signal=(
                    "candidate_context_improves_completion_metric_over_dynamic_controls"
                ),
                support_condition=(
                    "target_context_levels_completed_after_rollout > controls"
                ),
                failure_condition=(
                    "target_context_levels_completed_after_rollout <= controls"
                ),
                priority=float(proxy_gaps[0].get("latent_surprise_score", 0.0)),
            )
        )
    if clusters:
        cluster = sorted(
            clusters,
            key=lambda row: float(row.get("latent_delta_norm_mean", 0.0)),
            reverse=True,
        )[0]
        candidates.append(
            _make_hypothesis(
                index=3,
                family="action_conditioned_delta_cluster",
                signal_family="action_conditioned_delta_clusters",
                signal={
                    "game_id": "unknown_game",
                    "episode_id": "action_cluster_summary",
                    "step": 0,
                    "action": str(cluster.get("action", "")),
                    "available_actions_t": _default_actions_for_cluster(cluster),
                    "latent_surprise_score": float(cluster.get("latent_surprise_mean", 0.0)),
                    "latent_delta_norm": float(cluster.get("latent_delta_norm_mean", 0.0)),
                    "aggregate_signal": True,
                },
                metric="contact_graph_before_after",
                effect=(
                    "An action-conditioned latent delta cluster may correspond to a "
                    "grounded relation/contact change rather than raw pixel change."
                ),
                expected_signal=(
                    "target_action_contact_graph_change_exceeds_dynamic_controls"
                ),
                support_condition=(
                    "target_action_contact_graph_delta > best_control_contact_graph_delta"
                ),
                failure_condition=(
                    "target_action_contact_graph_delta <= best_control_contact_graph_delta"
                ),
                priority=float(cluster.get("latent_delta_norm_mean", 0.0)),
            )
        )
    if terminal_rows:
        candidates.append(
            _make_hypothesis(
                index=4,
                family="terminal_like_latent_neighborhood",
                signal_family="terminal_like_latent_neighborhoods",
                signal=terminal_rows[0],
                metric="terminal_state_after_rollout",
                effect=(
                    "A terminal-like latent neighborhood may identify action contexts "
                    "that re-enter terminal risk under grounded replay."
                ),
                expected_signal=(
                    "target_action_terminal_rate_exceeds_matched_dynamic_controls"
                ),
                support_condition=(
                    "target_action_terminal_rate > best_control_terminal_rate"
                ),
                failure_condition=(
                    "target_action_terminal_rate <= best_control_terminal_rate"
                ),
                priority=float(terminal_rows[0].get("latent_surprise_score", 0.0)),
            )
        )
    return tuple(candidates[: max_hypotheses])


def build_arc_lewm_hypothesis_payload(
    *,
    signal_report_path: str,
    signal_report: Mapping[str, Any],
    hypotheses: Sequence[FrontierConditionedHypothesis],
) -> Dict[str, Any]:
    contract = candidate_only_contract()
    return {
        "config": {
            "schema_version": ARC_LEWM_HYPOTHESES_SCHEMA_VERSION,
            "base_m2_schema_version": M2_SCHEMA_VERSION,
            "source_signal_report_path": signal_report_path,
            "generator_mode": "arc_lewm_signal_adapter_candidate_only",
            "inputs_read": [
                "diagnostics/m2/arc_lewm_signal_report.json",
                "diagnostics/m2/object_world_model_invariant_packet.json",
            ],
            "artifacts_not_modified": ["M3", "A32", "A33"],
            "environment_step_performed": False,
            "policy_rollout_performed": False,
        },
        "arc_lewm_hypothesis_batches": [
            {
                "source_signal_report_path": signal_report_path,
                "candidate_signal_families": list(
                    signal_report.get("candidate_signal_families", []) or []
                ),
                "candidate_hypotheses": [
                    _hypothesis_dict(hypothesis) for hypothesis in hypotheses
                ],
            }
        ],
        "summary": {
            "signal_report_consumed": True,
            "hypotheses_generated": len(hypotheses),
            "testable_hypotheses": sum(
                1 for hypothesis in hypotheses if hypothesis.testability.testable
            ),
            "ready_for_m3_candidate_experiment_request": sum(
                1 for hypothesis in hypotheses if hypothesis.testability.testable
            ),
            "world_model_score_counted_as_support": False,
            "world_model_counted_as_evidence": False,
            **contract,
        },
        "contract": contract,
        **contract,
    }


def build_arc_lewm_m3_requests_payload(
    hypotheses: Sequence[FrontierConditionedHypothesis],
    *,
    source_hypothesis_path: str,
) -> Dict[str, Any]:
    requests = tuple(
        _arc_lewm_m3_request_from_hypothesis(hypothesis)
        for hypothesis in hypotheses
    )
    invalid = [
        {
            "request_id": request.request_id,
            "errors": list(validate_m3_request(request).errors),
        }
        for request in requests
        if not validate_m3_request(request).valid
    ]
    if invalid:
        raise ValueError(f"invalid ARC-LeWM M3 requests generated: {invalid}")

    ready = [
        request
        for request in requests
        if request.status == M2_READY_FOR_M3_STATUS
    ]
    blocked = [
        request
        for request in requests
        if request.status == M2_BLOCKED_FOR_M3_STATUS
    ]
    blocked_by_reason = Counter(
        request.blocking_reason or "unknown_blocking_reason"
        for request in blocked
    )
    return {
        "config": {
            "source_hypothesis_path": source_hypothesis_path,
            "schema_version": ARC_LEWM_TO_M3_SCHEMA_VERSION,
            "handoff_validator": "arc_lewm_strict_candidate_handoff_v1",
        },
        "experiment_requests": [request.to_dict() for request in requests],
        "summary": {
            "source_hypotheses": len(hypotheses),
            "experiment_requests": len(requests),
            "ready_for_m3": len(ready),
            "blocked_not_testable": len(blocked),
            "blocked_by_reason": dict(sorted(blocked_by_reason.items())),
            "truth_status": M2_TRUTH_STATUS,
            "revision_performed": False,
            "wrong_confirmations": 0,
            "world_model_score_counted_as_support": False,
            "world_model_counted_as_evidence": False,
            "support": 0,
            "revision_status": CANDIDATE_REVISION_STATUS,
            "reset_actions_marked_ready_for_m3": sum(
                1
                for request in ready
                if request.target_action == "RESET"
            ),
            "unknown_game_marked_ready_for_m3": sum(
                1
                for request in ready
                if request.game_id == "unknown_game"
            ),
        },
    }


def _arc_lewm_m3_request_from_hypothesis(
    hypothesis: FrontierConditionedHypothesis,
) -> M3CandidateExperimentRequest:
    target_args = _target_action_args_from_hypothesis(hypothesis)
    source_transition_id = _source_transition_id_from_hypothesis(hypothesis)
    source_episode_id, source_step = _parse_source_transition_id(source_transition_id)
    blocking_reason = _handoff_blocking_reason(
        hypothesis,
        target_action_args=target_args,
        source_transition_id=source_transition_id,
    )
    status = (
        M2_READY_FOR_M3_STATUS
        if blocking_reason is None and hypothesis.testability.testable
        else M2_BLOCKED_FOR_M3_STATUS
    )
    if status == M2_BLOCKED_FOR_M3_STATUS and blocking_reason is None:
        blocking_reason = hypothesis.testability.blocking_reason or "BLOCKED_NOT_TESTABLE"
    context_replay = tuple(hypothesis.testability.required_context_replay)
    context_replay_args = (
        tuple(hypothesis.context_snapshot.replay_action_args or ())
        if context_replay
        else None
    )
    return M3CandidateExperimentRequest(
        request_id=request_id_for_hypothesis(hypothesis.hypothesis_id),
        source_hypothesis_id=hypothesis.hypothesis_id,
        game_id=hypothesis.game_id,
        context_replay=context_replay,
        context_replay_args=context_replay_args,
        context_snapshot_hash=source_transition_id,
        target_action=hypothesis.testability.target_action,
        target_action_args=target_args,
        suggested_control_actions=tuple(hypothesis.testability.suggested_control_actions),
        control_policy=M2_DYNAMIC_CONTROL_POLICY,
        metric=hypothesis.testability.metric,
        expected_signal=hypothesis.testability.expected_signal_type,
        falsification_criterion=hypothesis.falsification,
        status=status,
        source_episode_id=source_episode_id,
        source_step=source_step,
        source_transition_id=source_transition_id,
        context_state_origin=(
            ARC_LEWM_CONTEXT_STATE_ORIGIN if source_transition_id else None
        ),
        replayability=ARC_LEWM_REPLAYABILITY if source_transition_id else None,
        blocking_reason=blocking_reason,
        truth_status=M2_TRUTH_STATUS,
        support=0,
        controlled_test_required=True,
        revision_performed=False,
        wrong_confirmations=0,
    )


def write_arc_lewm_hypotheses(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_ARC_LEWM_HYPOTHESES_OUTPUT_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _make_hypothesis(
    *,
    index: int,
    family: str,
    signal_family: str,
    signal: Mapping[str, Any],
    metric: str,
    effect: str,
    expected_signal: str,
    support_condition: str,
    failure_condition: str,
    priority: float,
) -> FrontierConditionedHypothesis:
    action = str(signal.get("action", "")) or "ACTION6"
    available_actions = tuple(
        str(action_name)
        for action_name in signal.get("available_actions_t", []) or []
        if str(action_name)
    )
    controls = default_controls_for_action(action, available_actions) if available_actions else ("ACTION3", "ACTION4")
    action_args = dict(signal.get("action_args", {}) or {})
    source_transition_id = _source_transition_id_from_signal(signal)
    blocking_reason = _signal_blocking_reason(
        signal_family=signal_family,
        signal=signal,
        metric=metric,
        action=action,
        available_actions=available_actions,
        action_args=action_args,
        source_transition_id=source_transition_id,
    )
    falsification = FalsificationCriterion(
        metric=metric,
        support_condition=support_condition,
        failure_condition=failure_condition,
        minimum_effect_size=1,
    )
    if not metric:
        falsification = default_falsification_for_metric(metric)
    testability = HypothesisTestability(
        testable=blocking_reason is None,
        recommended_test_type="controlled_action_vs_alternative",
        target_action=action,
        suggested_control_actions=tuple(control for control in controls if control != action),
        control_policy=M2_DYNAMIC_CONTROL_POLICY,
        metric=metric,
        required_context_replay=(),
        required_action_args=(
            (action_args,)
            if action_args
            else None
        ),
        expected_signal_type=expected_signal,
        measurable_by_existing_extractor=True,
        blocking_reason=blocking_reason,
    )
    context_snapshot = ContextSnapshot(
        replay_actions=(),
        replay_action_args=None,
        frame_before_hash=source_transition_id,
        live_state_signature=(
            f"arc_lewm_signal:{signal_family}:"
            f"{signal.get('game_id', '')}:{signal.get('episode_id', '')}:"
            f"{signal.get('step', '')}:surprise={float(signal.get('latent_surprise_score', 0.0)):.6f}"
        ),
        available_actions=available_actions or ("ACTION3", "ACTION4", "ACTION6"),
        local_patch=None,
        terminal_state=bool(signal.get("terminal_t1", False)),
    )
    rationale = (
        f"ARC-LeWM {signal_family} produced priority score "
        f"{float(priority):.6f}; this score is a candidate priority only, "
        "not evidence or support."
    )
    audit = SourceGenerationAudit(
        sources=("world_model",),
        raw_proposal_ids=(f"raw::arc_lewm::{signal_family}::{index:03d}",),
        rationales=(rationale,),
        normalization_warnings=(
            "arc_lewm_score_not_counted_as_support",
            "world_model_signal_not_counted_as_evidence",
        ),
        priority_score=float(priority),
        priority_score_counted_as_support=False,
    )
    return FrontierConditionedHypothesis(
        hypothesis_id=f"m2_14_lewm::{signal_family}::{index:03d}",
        source_request_id=f"m2_14d::{signal_family}",
        game_id=str(signal.get("game_id", "")) or "unknown_game",
        frontier_context_id=f"arc_lewm::{signal_family}",
        frontier_reason="ARC_LEWM_LATENT_SIGNAL_CANDIDATE_ONLY",
        frontier_step=int(signal.get("step", 0) or 0),
        hypothesis_family=family,
        candidate_action=action,
        predicted_metric=metric,
        predicted_effect=effect,
        rationale=rationale,
        testability=testability,
        falsification=falsification,
        context_snapshot=context_snapshot,
        source_generation=audit,
        status=M2_HYPOTHESIS_STATUS,
        support=0,
        controlled_test_required=True,
        truth_status=M2_TRUTH_STATUS,
        revision_performed=False,
        wrong_confirmations=0,
        trace_support_counted_as_proof=False,
        prior_counted_as_proof=False,
    )


def _hypothesis_dict(hypothesis: FrontierConditionedHypothesis) -> Dict[str, Any]:
    row = hypothesis.to_dict()
    row["revision_status"] = CANDIDATE_REVISION_STATUS
    row["ready_for_m3_candidate_experiment_request"] = bool(
        hypothesis.testability.testable
    )
    row["ready_for_a32"] = False
    row["ready_for_a33"] = False
    row["world_model_score_counted_as_support"] = False
    row["world_model_counted_as_evidence"] = False
    row["arc_lewm_hypothesis_counted_as_confirmation"] = False
    return row


def _target_action_args_from_hypothesis(
    hypothesis: FrontierConditionedHypothesis,
) -> Dict[str, Any] | None:
    action_args = hypothesis.testability.required_action_args
    if not action_args:
        return None
    first = action_args[0]
    return dict(first) if first else None


def _source_transition_id_from_hypothesis(
    hypothesis: FrontierConditionedHypothesis,
) -> str | None:
    if hypothesis.context_snapshot.frame_before_hash:
        return str(hypothesis.context_snapshot.frame_before_hash)
    signature = str(hypothesis.context_snapshot.live_state_signature or "")
    if signature.startswith("arc_lewm_signal|"):
        parts = signature.split("|")
        if len(parts) >= 6:
            game_id = parts[2]
            episode_id = parts[3]
            step = _int_or_none(parts[4])
            if step is not None and game_id and episode_id:
                return f"m2_14d::{game_id}::{episode_id}::{step:04d}"
    return None


def _source_transition_id_from_signal(signal: Mapping[str, Any]) -> str | None:
    if bool(signal.get("aggregate_signal", False)):
        return None
    for key in ("source_transition_id", "signal_id"):
        value = str(signal.get(key, "") or "")
        if value and len(value.split("::")) >= 4:
            return value
    game_id = str(signal.get("game_id", "") or "")
    episode_id = str(signal.get("episode_id", "") or "")
    step = _int_or_none(signal.get("step"))
    if not game_id or game_id == "unknown_game":
        return None
    if not episode_id or episode_id == "action_cluster_summary" or step is None:
        return None
    return f"m2_14d::{game_id}::{episode_id}::{step:04d}"


def _parse_source_transition_id(
    source_transition_id: str | None,
) -> tuple[str | None, int | None]:
    if not source_transition_id:
        return None, None
    parts = source_transition_id.split("::")
    if len(parts) < 4:
        return None, None
    step = _int_or_none(parts[-1])
    episode_id = "::".join(parts[2:-1])
    return (episode_id or None), step


def _signal_blocking_reason(
    *,
    signal_family: str,
    signal: Mapping[str, Any],
    metric: str,
    action: str,
    available_actions: Sequence[str],
    action_args: Mapping[str, Any],
    source_transition_id: str | None,
) -> str | None:
    game_id = str(signal.get("game_id", "") or "")
    if signal_family == "action_conditioned_delta_clusters" or bool(
        signal.get("aggregate_signal", False)
    ):
        return "DIAGNOSTIC_ONLY_AGGREGATE"
    if action == "RESET":
        return "BLOCKED_RESET_BOUNDARY"
    if not game_id or game_id == "unknown_game":
        return "BLOCKED_UNKNOWN_GAME"
    if available_actions and action not in set(available_actions):
        return "BLOCKED_ACTION_NOT_AVAILABLE"
    if action == "ACTION6" and not dict(action_args):
        return "BLOCKED_MISSING_ACTION_ARGS"
    if not source_transition_id:
        return "BLOCKED_MISSING_OFFLINE_TRACE_LOCATOR"
    if bool(signal.get("terminal_t1", False)) and metric != "terminal_state_after_rollout":
        return "BLOCKED_TERMINAL_SOURCE_STATE"
    return None


def _handoff_blocking_reason(
    hypothesis: FrontierConditionedHypothesis,
    *,
    target_action_args: Mapping[str, Any] | None,
    source_transition_id: str | None,
) -> str | None:
    signal_family = _signal_family_from_hypothesis(hypothesis)
    target_action = hypothesis.testability.target_action
    available_actions = tuple(hypothesis.context_snapshot.available_actions)
    if signal_family == "action_conditioned_delta_clusters":
        return "DIAGNOSTIC_ONLY_AGGREGATE"
    if target_action == "RESET":
        return "BLOCKED_RESET_BOUNDARY"
    if not hypothesis.game_id or hypothesis.game_id == "unknown_game":
        return "BLOCKED_UNKNOWN_GAME"
    if available_actions and target_action not in set(available_actions):
        return "BLOCKED_ACTION_NOT_AVAILABLE"
    if target_action == "ACTION6" and not dict(target_action_args or {}):
        return "BLOCKED_MISSING_ACTION_ARGS"
    if not source_transition_id:
        return "BLOCKED_MISSING_OFFLINE_TRACE_LOCATOR"
    if (
        hypothesis.context_snapshot.terminal_state
        and hypothesis.predicted_metric != "terminal_state_after_rollout"
    ):
        return "BLOCKED_TERMINAL_SOURCE_STATE"
    return hypothesis.testability.blocking_reason


def _signal_family_from_hypothesis(hypothesis: FrontierConditionedHypothesis) -> str:
    raw_prefix = "raw::arc_lewm::"
    for raw_id in hypothesis.source_generation.raw_proposal_ids:
        if raw_id.startswith(raw_prefix):
            remainder = raw_id[len(raw_prefix):]
            return remainder.rsplit("::", 1)[0]
    context_prefix = "arc_lewm::"
    if hypothesis.frontier_context_id.startswith(context_prefix):
        return hypothesis.frontier_context_id[len(context_prefix):]
    return ""


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _validate_signal_report_contract(signal_report: Mapping[str, Any]) -> None:
    summary = dict(signal_report.get("summary", {}) or {})
    if int(signal_report.get("support", summary.get("support", 0)) or 0) != 0:
        raise ValueError("ARC-LeWM signal report support must remain 0")
    for key in (
        "world_model_counted_as_evidence",
        "world_model_score_counted_as_support",
        "a32_write_performed",
        "a33_write_performed",
        "ready_for_a32",
        "ready_for_a33",
    ):
        if bool(signal_report.get(key, summary.get(key, False))):
            raise ValueError(f"ARC-LeWM signal report must not set {key}")


def _first_game_id(signal_report: Mapping[str, Any]) -> str:
    for section in dict(signal_report.get("signals", {}) or {}).values():
        if isinstance(section, list) and section:
            return str(section[0].get("game_id", "unknown_game"))
    return "unknown_game"


def _default_actions_for_cluster(cluster: Mapping[str, Any]) -> list[str]:
    action = str(cluster.get("action", "ACTION6"))
    actions = ["ACTION3", "ACTION4", "ACTION6"]
    if action not in actions:
        actions.append(action)
    return actions


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adapt ARC-LeWM signal reports into M2 hypotheses and M3 requests.",
    )
    parser.add_argument(
        "--signal-report",
        type=Path,
        default=DEFAULT_ARC_LEWM_SIGNAL_REPORT_OUTPUT_PATH,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_ARC_LEWM_HYPOTHESES_OUTPUT_PATH)
    parser.add_argument(
        "--m3-requests",
        type=Path,
        default=DEFAULT_ARC_LEWM_M3_REQUESTS_OUTPUT_PATH,
    )
    parser.add_argument("--max-hypotheses", type=int, default=4)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payloads = run_arc_lewm_hypothesis_adapter(
        signal_report_path=args.signal_report,
        hypotheses_output_path=args.out,
        m3_requests_output_path=args.m3_requests,
        max_hypotheses=args.max_hypotheses,
    )
    print(
        json.dumps(
            {
                "hypotheses_path": str(args.out),
                "m3_requests_path": str(args.m3_requests),
                "summary": payloads["hypotheses_payload"]["summary"],
                "m3_summary": payloads["m3_payload"]["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
