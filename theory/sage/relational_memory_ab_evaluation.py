"""SAGE.8b paired live-replay evaluation with and without relational memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from theory.a34.control_dependent_relational_usage_probe import (
    A34_2_RELATION_USEFUL,
    A34_2_SCHEMA_VERSION,
    A34_2_TRUTH_STATUS,
    DEFAULT_A34_CONTROL_DEPENDENT_RELATIONAL_USAGE_PROBE_PATH,
)
from theory.a34.parameterized_relational_usage_probe import (
    A34_3_RELATION_USEFUL,
    A34_3_SCHEMA_VERSION,
    A34_3_TRUTH_STATUS,
    DEFAULT_A34_PARAMETERIZED_RELATIONAL_USAGE_PROBE_PATH,
)
from theory.non_ar25_active_micro_run import _env_dir

from .live_mini_frontier_m3_executor import EnvFactory, _execute_request_arm
from .relational_memory_policy import (
    DEFAULT_SAGE8A_RELATIONAL_MEMORY_POLICY_PATH,
    LOWER_EFFECT_COMPARATOR_MATCH,
    SAGE8A_POLICY_READY,
    SAGE8A_SCHEMA_VERSION,
    SAGE8A_TRUTH_STATUS,
    PolicyActionOption,
    apply_relational_memory_policy,
)


DEFAULT_SAGE8B_RELATIONAL_MEMORY_AB_EVALUATION_PATH = (
    Path("diagnostics") / "sage" / "sage8b_relational_memory_ab_evaluation.json"
)

SAGE8B_SCHEMA_VERSION = "sage.relational_memory_ab_evaluation.v1"
SAGE8B_TRUTH_STATUS = "NOT_REEVALUATED_BY_SAGE_8B"
SAGE8B_ARC_GAIN = "SAGE_RELATIONAL_MEMORY_ARC_SCORE_GAIN_OBSERVED"
SAGE8B_LOCAL_ONLY_GAIN = "SAGE_RELATIONAL_MEMORY_LOCAL_GAIN_WITHOUT_ARC_SCORE_GAIN"
SAGE8B_NO_GAIN = "SAGE_RELATIONAL_MEMORY_NO_GAIN_OBSERVED"


def run_sage8b_relational_memory_ab_evaluation(
    *,
    policy_path: str | Path = DEFAULT_SAGE8A_RELATIONAL_MEMORY_POLICY_PATH,
    a34_2_path: str | Path = (
        DEFAULT_A34_CONTROL_DEPENDENT_RELATIONAL_USAGE_PROBE_PATH
    ),
    a34_3_path: str | Path = DEFAULT_A34_PARAMETERIZED_RELATIONAL_USAGE_PROBE_PATH,
    environments_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    """Run paired exact-context episodes and compare primary ARC metrics."""
    policy_source = _load_json(policy_path)
    a34_2 = _load_json(a34_2_path)
    a34_3 = _load_json(a34_3_path)
    validate_sage8b_sources(policy_source, a34_2, a34_3)
    specifications = build_sage8b_evaluation_specifications(policy_source, a34_2, a34_3)
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()
    episodes = tuple(
        execute_sage8b_paired_episode(
            specification,
            policy=policy_source["policy"],
            environments_dir=env_dir,
            env_factory=env_factory,
        )
        for specification in specifications
    )
    primary_metrics = summarize_primary_metrics(episodes)
    secondary_metrics = summarize_secondary_metrics(episodes)
    gate = build_sage8b_gate(specifications, episodes, primary_metrics)
    if not gate or not all(gate.values()):
        raise ValueError("SAGE.8b relational memory A/B evaluation gate did not pass")
    summary = summarize_sage8b(episodes, primary_metrics, secondary_metrics, gate)
    payload = {
        "config": {
            "schema_version": SAGE8B_SCHEMA_VERSION,
            "policy_path": str(policy_path),
            "a34_2_path": str(a34_2_path),
            "a34_3_path": str(a34_3_path),
            "environments_dir": str(env_dir),
            "inputs_read": ["SAGE.8a_POLICY", "A34.2_REPLAYS", "A34.3_REPLAYS"],
            "evaluation_design": {
                "paired_exact_context_replays": True,
                "same_prefix_between_arms": True,
                "no_memory_arm_uses_registered_lower_effect_comparator": True,
                "with_memory_arm_uses_sage8a_policy_decision": True,
                "policy_target_must_pass_live_action_selection": True,
                "all_eleven_registered_contexts_required": True,
                "technical_replays_not_counted_as_independent_episodes": True,
                "primary_metrics": ["levels_completed", "win_rate"],
                "secondary_metrics": ["local_patch_before_after"],
                "local_signal_gain_is_not_arc_score_gain": True,
            },
            "artifacts_not_modified": ["A33.3", "A33.4", "A34.2", "A34.3"],
        },
        "evaluation_specifications": [
            _public_specification(row) for row in specifications
        ],
        "paired_episodes": [dict(row) for row in episodes],
        "primary_metrics": primary_metrics,
        "secondary_metrics": secondary_metrics,
        "gate": gate,
        "summary": summary,
        "outcome_status": summary["outcome_status"],
        "status": "EVALUATED",
        "truth_status": SAGE8B_TRUTH_STATUS,
        "comparative_evaluation_performed": True,
        "live_replay_execution_performed": True,
        "scientific_review_performed": False,
        "revision_performed": False,
        "confirmation_performed": False,
        "refutation_performed": False,
        "support": 0,
        "registry_support_recounted": False,
        "a33_mutated": False,
        "scope_generalization_performed": False,
        "levels_completed": primary_metrics["levels_completed"][
            "with_memory_max_after"
        ],
        "win_rate": primary_metrics["win_rate"]["with_memory"],
        "primary_arc_progress_improved": summary["primary_arc_progress_improved"],
        "wrong_confirmations": 0,
    }
    if output_path is not None:
        write_sage8b_relational_memory_ab_evaluation(payload, output_path)
    return payload


def build_sage8b_evaluation_specifications(
    policy_source: Mapping[str, Any],
    a34_2: Mapping[str, Any],
    a34_3: Mapping[str, Any],
) -> Tuple[Dict[str, Any], ...]:
    """Join SAGE.8a choices to the exact outcome-independent replay manifests."""
    entries = [
        dict(row)
        for row in policy_source.get("policy_entries", []) or []
        if isinstance(row, Mapping)
    ]
    entries_by_game = {str(row.get("game_id", "")): row for row in entries}
    specifications: List[Dict[str, Any]] = []
    for source_name, source in (("A34.2", a34_2), ("A34.3", a34_3)):
        for context in source.get("replay_contexts", []) or []:
            game_id = str(context.get("game_id", ""))
            entry = dict(entries_by_game.get(game_id, {}) or {})
            request = dict(context.get("request", {}) or {})
            if not request:
                request = {
                    "request_id": str(context.get("source_request_id", "")),
                    "game_id": game_id,
                    "metric": str(context.get("metric", "")),
                    "context_snapshot_hash": str(
                        context.get("context_snapshot_hash", "")
                    ),
                    "context_replay": list(context.get("context_replay", []) or []),
                    "context_replay_args": [
                        dict(row)
                        for row in context.get("context_replay_args", []) or []
                    ],
                }
            specifications.append(
                {
                    "evaluation_id": (f"sage8b::paired::{len(specifications) + 1:03d}"),
                    "replay_source": source_name,
                    "game_id": game_id,
                    "context_snapshot_hash": str(
                        context.get("context_snapshot_hash", "")
                    ),
                    "source_request_id": str(context.get("source_request_id", "")),
                    "source_step": int(context.get("source_step", 0) or 0),
                    "budget": int(context.get("budget", 0) or 0),
                    "technical_replay_count": int(
                        context.get("technical_replay_count", 0) or 0
                    ),
                    "metric": str(context.get("metric", "")),
                    "policy_entry_id": str(entry.get("policy_entry_id", "")),
                    "no_memory_action": str(entry.get("lower_effect_action", "")),
                    "no_memory_action_args": dict(
                        entry.get("lower_effect_action_args", {}) or {}
                    ),
                    "memory_action": str(entry.get("selected_action", "")),
                    "memory_action_args": dict(
                        entry.get("selected_action_args", {}) or {}
                    ),
                    "equivalent_action": str(entry.get("equivalent_action", "")),
                    "equivalent_action_args": dict(
                        entry.get("equivalent_action_args", {}) or {}
                    ),
                    "request": request,
                    "replay_selected_without_outcome_read": True,
                }
            )
    return tuple(specifications)


def execute_sage8b_paired_episode(
    specification: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    environments_dir: str | Path,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    """Execute no-memory and memory actions from one identical live prefix."""
    proposed = PolicyActionOption(
        str(specification.get("no_memory_action", "")),
        dict(specification.get("no_memory_action_args", {}) or {}),
    )
    memory_option = PolicyActionOption(
        str(specification.get("memory_action", "")),
        dict(specification.get("memory_action_args", {}) or {}),
    )
    equivalent = PolicyActionOption(
        str(specification.get("equivalent_action", "")),
        dict(specification.get("equivalent_action_args", {}) or {}),
    )
    decision = apply_relational_memory_policy(
        policy,
        game_id=str(specification.get("game_id", "")),
        context_snapshot_hash=str(specification.get("context_snapshot_hash", "")),
        proposed_action_raw=proposed,
        valid_actions=(proposed, memory_option, equivalent),
        metric=str(specification.get("metric", "")),
    )
    if not bool(decision.get("relational_memory_applied", False)):
        raise ValueError(
            "SAGE.8b memory arm requires an applied SAGE.8a policy decision"
        )
    request = dict(specification.get("request", {}) or {})
    no_memory = _execute_request_arm(
        request,
        action_name=proposed.name,
        action_args=proposed.action_args,
        arm="sage8b_no_memory",
        environments_dir=environments_dir,
        env_factory=env_factory,
    )
    with_memory = _execute_request_arm(
        request,
        action_name=str(decision.get("selected_action", "")),
        action_args=dict(decision.get("selected_action_args", {}) or {}),
        arm="sage8b_with_relational_memory",
        environments_dir=environments_dir,
        env_factory=env_factory,
    )
    if any(
        str(arm.get("status", "")) != "EXECUTED" for arm in (no_memory, with_memory)
    ):
        reasons = [str(arm.get("reason", "")) for arm in (no_memory, with_memory)]
        raise ValueError(f"SAGE.8b paired replay arm blocked: {reasons}")
    memory_before = _signature_payload(with_memory.get("before_signature", ""))
    no_after = _signature_payload(no_memory.get("after_signature", ""))
    memory_after = _signature_payload(with_memory.get("after_signature", ""))
    before_levels = int(memory_before.get("levels_completed", 0) or 0)
    no_levels = int(no_after.get("levels_completed", 0) or 0)
    memory_levels = int(memory_after.get("levels_completed", 0) or 0)
    no_signal = _arm_signal(no_memory)
    memory_signal = _arm_signal(with_memory)
    return {
        "evaluation_id": str(specification.get("evaluation_id", "")),
        "replay_source": str(specification.get("replay_source", "")),
        "game_id": str(specification.get("game_id", "")),
        "context_snapshot_hash": str(specification.get("context_snapshot_hash", "")),
        "source_request_id": str(specification.get("source_request_id", "")),
        "source_step": int(specification.get("source_step", 0) or 0),
        "budget": int(specification.get("budget", 0) or 0),
        "metric": str(specification.get("metric", "")),
        "policy_entry_id": str(specification.get("policy_entry_id", "")),
        "policy_decision": _serializable_decision(decision),
        "relational_memory_consulted": True,
        "relational_memory_applied": True,
        "policy_target_live_action_selection_passed": True,
        "same_prefix_between_arms": (
            no_memory.get("before_signature") == with_memory.get("before_signature")
        ),
        "replay_exact_both_arms": all(
            bool(arm.get("context_snapshot_hash_verified", False))
            for arm in (no_memory, with_memory)
        ),
        "no_memory_action": proposed.name,
        "no_memory_action_args": dict(proposed.action_args),
        "with_memory_action": str(decision.get("selected_action", "")),
        "with_memory_action_args": dict(decision.get("selected_action_args", {}) or {}),
        "no_memory_arm": dict(no_memory),
        "with_memory_arm": dict(with_memory),
        "levels_completed_before": before_levels,
        "no_memory_levels_completed_after": no_levels,
        "with_memory_levels_completed_after": memory_levels,
        "no_memory_levels_completed_delta": no_levels - before_levels,
        "with_memory_levels_completed_delta": memory_levels - before_levels,
        "no_memory_win": _signature_won(no_after),
        "with_memory_win": _signature_won(memory_after),
        "no_memory_local_signal": no_signal,
        "with_memory_local_signal": memory_signal,
        "local_signal_gain": memory_signal - no_signal,
        "primary_metrics": ["levels_completed", "win_rate"],
        "local_signal_is_secondary_only": True,
        "truth_reevaluated": False,
        "support_counted": 0,
        "wrong_confirmations": 0,
    }


def summarize_primary_metrics(
    episodes: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    count = len(episodes)
    no_wins = sum(bool(row.get("no_memory_win", False)) for row in episodes)
    memory_wins = sum(bool(row.get("with_memory_win", False)) for row in episodes)
    no_total_delta = sum(
        int(row.get("no_memory_levels_completed_delta", 0) or 0) for row in episodes
    )
    memory_total_delta = sum(
        int(row.get("with_memory_levels_completed_delta", 0) or 0) for row in episodes
    )
    no_rate = no_wins / count if count else 0.0
    memory_rate = memory_wins / count if count else 0.0
    return {
        "primary_metric_order": ["levels_completed", "win_rate"],
        "levels_completed": {
            "no_memory_total_delta": no_total_delta,
            "with_memory_total_delta": memory_total_delta,
            "absolute_delta_gain": memory_total_delta - no_total_delta,
            "no_memory_max_after": max(
                (
                    int(row.get("no_memory_levels_completed_after", 0) or 0)
                    for row in episodes
                ),
                default=0,
            ),
            "with_memory_max_after": max(
                (
                    int(row.get("with_memory_levels_completed_after", 0) or 0)
                    for row in episodes
                ),
                default=0,
            ),
            "improved": memory_total_delta > no_total_delta,
        },
        "win_rate": {
            "episodes_per_arm": count,
            "no_memory_wins": no_wins,
            "with_memory_wins": memory_wins,
            "no_memory": no_rate,
            "with_memory": memory_rate,
            "absolute_gain": memory_rate - no_rate,
            "improved": memory_rate > no_rate,
        },
    }


def summarize_secondary_metrics(
    episodes: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    no_total = sum(
        float(row.get("no_memory_local_signal", 0.0) or 0.0) for row in episodes
    )
    memory_total = sum(
        float(row.get("with_memory_local_signal", 0.0) or 0.0) for row in episodes
    )
    return {
        "metric": "local_patch_before_after",
        "classification": "SECONDARY_DIAGNOSTIC_ONLY",
        "no_memory_total": no_total,
        "with_memory_total": memory_total,
        "absolute_gain": memory_total - no_total,
        "episodes_with_positive_gain": sum(
            float(row.get("local_signal_gain", 0.0) or 0.0) > 0 for row in episodes
        ),
        "improved": memory_total > no_total,
        "counted_as_level_completion": False,
        "counted_as_win": False,
    }


def build_sage8b_gate(
    specifications: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    primary_metrics: Mapping[str, Any],
) -> Dict[str, bool]:
    expected_hashes = {
        str(row.get("context_snapshot_hash", "")) for row in specifications
    }
    observed_hashes = {str(row.get("context_snapshot_hash", "")) for row in episodes}
    levels = dict(primary_metrics.get("levels_completed", {}) or {})
    wins = dict(primary_metrics.get("win_rate", {}) or {})
    return {
        "all_eleven_exact_contexts_evaluated_once": len(specifications)
        == len(episodes)
        == len(expected_hashes)
        == 11
        and expected_hashes == observed_hashes,
        "both_games_evaluated": {str(row.get("game_id", "")) for row in episodes}
        == {"wa30-ee6fef47", "tn36-ab4f63cc"},
        "memory_policy_applied_in_every_treatment_arm": all(
            bool(row.get("relational_memory_applied", False))
            and str(row.get("policy_decision", {}).get("decision_reason", ""))
            == LOWER_EFFECT_COMPARATOR_MATCH
            for row in episodes
        ),
        "all_paired_replays_exact": all(
            bool(row.get("replay_exact_both_arms", False)) for row in episodes
        ),
        "same_prefix_used_between_arms": all(
            bool(row.get("same_prefix_between_arms", False)) for row in episodes
        ),
        "policy_targets_pass_live_selection": all(
            bool(row.get("policy_target_live_action_selection_passed", False))
            for row in episodes
        ),
        "levels_completed_recorded_as_primary": all(
            key in levels
            for key in (
                "no_memory_total_delta",
                "with_memory_total_delta",
                "absolute_delta_gain",
                "improved",
            )
        ),
        "win_rate_recorded_as_primary": int(wins.get("episodes_per_arm", 0) or 0) == 11
        and 0.0 <= float(wins.get("no_memory", -1.0)) <= 1.0
        and 0.0 <= float(wins.get("with_memory", -1.0)) <= 1.0,
        "no_truth_reevaluation_or_support_counting": all(
            not bool(row.get("truth_reevaluated", True))
            and int(row.get("support_counted", -1) or 0) == 0
            for row in episodes
        ),
    }


def summarize_sage8b(
    episodes: Sequence[Mapping[str, Any]],
    primary_metrics: Mapping[str, Any],
    secondary_metrics: Mapping[str, Any],
    gate: Mapping[str, bool],
) -> Dict[str, Any]:
    levels = dict(primary_metrics.get("levels_completed", {}) or {})
    wins = dict(primary_metrics.get("win_rate", {}) or {})
    primary_improved = bool(levels.get("improved", False)) or bool(
        wins.get("improved", False)
    )
    secondary_improved = bool(secondary_metrics.get("improved", False))
    if primary_improved:
        outcome_status = SAGE8B_ARC_GAIN
    elif secondary_improved:
        outcome_status = SAGE8B_LOCAL_ONLY_GAIN
    else:
        outcome_status = SAGE8B_NO_GAIN
    gate_passed = bool(gate) and all(bool(value) for value in gate.values())
    return {
        "paired_episodes_evaluated": len(episodes),
        "games_evaluated": sorted({str(row.get("game_id", "")) for row in episodes}),
        "memory_policy_applications": sum(
            bool(row.get("relational_memory_applied", False)) for row in episodes
        ),
        "exact_paired_replays": sum(
            bool(row.get("replay_exact_both_arms", False)) for row in episodes
        ),
        "no_memory_levels_completed_delta_total": int(
            levels.get("no_memory_total_delta", 0) or 0
        ),
        "with_memory_levels_completed_delta_total": int(
            levels.get("with_memory_total_delta", 0) or 0
        ),
        "levels_completed_absolute_gain": int(
            levels.get("absolute_delta_gain", 0) or 0
        ),
        "levels_completed_improved": bool(levels.get("improved", False)),
        "no_memory_wins": int(wins.get("no_memory_wins", 0) or 0),
        "with_memory_wins": int(wins.get("with_memory_wins", 0) or 0),
        "no_memory_win_rate": float(wins.get("no_memory", 0.0) or 0.0),
        "with_memory_win_rate": float(wins.get("with_memory", 0.0) or 0.0),
        "win_rate_absolute_gain": float(wins.get("absolute_gain", 0.0) or 0.0),
        "win_rate_improved": bool(wins.get("improved", False)),
        "primary_arc_progress_improved": primary_improved,
        "secondary_local_signal_gain": float(
            secondary_metrics.get("absolute_gain", 0.0) or 0.0
        ),
        "secondary_local_signal_improved": secondary_improved,
        "local_signal_counted_as_arc_progress": False,
        "truth_reevaluations": 0,
        "support_counted": 0,
        "scope_generalization_performed": False,
        "wrong_confirmations": 0,
        "gate_passed": gate_passed,
        "outcome_status": outcome_status,
    }


def validate_sage8b_sources(
    policy_source: Mapping[str, Any],
    a34_2: Mapping[str, Any],
    a34_3: Mapping[str, Any],
) -> None:
    if (
        str(policy_source.get("config", {}).get("schema_version", ""))
        != SAGE8A_SCHEMA_VERSION
        or str(policy_source.get("outcome_status", "")) != SAGE8A_POLICY_READY
        or str(policy_source.get("truth_status", "")) != SAGE8A_TRUTH_STATUS
        or not bool(policy_source.get("ready_for_comparative_evaluation", False))
        or not bool(policy_source.get("summary", {}).get("gate_passed", False))
        or not all(bool(value) for value in policy_source.get("gate", {}).values())
    ):
        raise ValueError("SAGE.8b requires the completed SAGE.8a policy")
    source_expectations = (
        (a34_2, A34_2_SCHEMA_VERSION, A34_2_RELATION_USEFUL, A34_2_TRUTH_STATUS, 3),
        (a34_3, A34_3_SCHEMA_VERSION, A34_3_RELATION_USEFUL, A34_3_TRUTH_STATUS, 8),
    )
    for source, schema, outcome, truth, count in source_expectations:
        if (
            str(source.get("config", {}).get("schema_version", "")) != schema
            or str(source.get("outcome_status", "")) != outcome
            or str(source.get("truth_status", "")) != truth
            or not bool(source.get("summary", {}).get("gate_passed", False))
            or not all(bool(value) for value in source.get("gate", {}).values())
            or len(source.get("replay_contexts", []) or []) != count
            or int(source.get("support", -1) or 0) != 0
            or bool(source.get("scope_generalization_performed", True))
        ):
            raise ValueError("SAGE.8b requires completed scope-locked A34 replays")
    policy_hashes = {
        str(value)
        for entry in policy_source.get("policy_entries", []) or []
        for value in entry.get("context_snapshot_hashes", []) or []
    }
    replay_hashes = {
        str(row.get("context_snapshot_hash", ""))
        for source in (a34_2, a34_3)
        for row in source.get("replay_contexts", []) or []
    }
    if len(policy_hashes) != 11 or policy_hashes != replay_hashes:
        raise ValueError("SAGE.8a policy and A34 replay scopes must match exactly")


def write_sage8b_relational_memory_ab_evaluation(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SAGE8B_RELATIONAL_MEMORY_AB_EVALUATION_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _public_specification(specification: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in specification.items() if key != "request"}


def _serializable_decision(decision: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value for key, value in decision.items() if key != "selected_action_raw"
    }


def _arm_signal(arm: Mapping[str, Any]) -> float:
    return float(
        arm.get("measurement_for_delta", {}).get("local_changed_pixels", 0) or 0
    )


def _signature_payload(value: Any) -> Dict[str, Any]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _signature_won(signature: Mapping[str, Any]) -> bool:
    return str(signature.get("game_state", "")).upper() in {"WIN", "WON", "VICTORY"}


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy", default=str(DEFAULT_SAGE8A_RELATIONAL_MEMORY_POLICY_PATH)
    )
    parser.add_argument(
        "--a34-2",
        default=str(DEFAULT_A34_CONTROL_DEPENDENT_RELATIONAL_USAGE_PROBE_PATH),
    )
    parser.add_argument(
        "--a34-3", default=str(DEFAULT_A34_PARAMETERIZED_RELATIONAL_USAGE_PROBE_PATH)
    )
    parser.add_argument("--environments-dir", default=None)
    parser.add_argument(
        "--out", default=str(DEFAULT_SAGE8B_RELATIONAL_MEMORY_AB_EVALUATION_PATH)
    )
    args = parser.parse_args(argv)
    payload = run_sage8b_relational_memory_ab_evaluation(
        policy_path=args.policy,
        a34_2_path=args.a34_2,
        a34_3_path=args.a34_3,
        environments_dir=args.environments_dir,
        output_path=args.out,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
