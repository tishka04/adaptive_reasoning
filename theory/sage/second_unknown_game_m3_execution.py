"""SAGE.6b stratified M3 execution on the second unknown game."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from theory.non_ar25_active_micro_run import _env_dir

from .live_mini_frontier_m3_executor import (
    EnvFactory,
    execute_live_prefix_mini_frontier_request,
)
from .second_unknown_game_switch_frontier import (
    DEFAULT_SAGE6A_SWITCH_FRONTIER_PATH,
    SAGE6A_FRONTIER_GENERATED,
    SAGE6A_SCHEMA_VERSION,
    SAGE6A_TRUTH_STATUS,
)


DEFAULT_SAGE6B_M3_EXECUTION_PATH = (
    Path("diagnostics") / "sage" / "sage6b_second_unknown_game_m3_execution.json"
)

SAGE6B_SCHEMA_VERSION = "sage.second_unknown_game_m3_execution.v1"
SAGE6B_TRUTH_STATUS = "NOT_EVALUATED_BY_SAGE_6B"
SAGE6B_EXECUTION_COMPLETED = (
    "SAGE_SECOND_UNKNOWN_GAME_M3_EXECUTION_COMPLETED_CANDIDATE_ONLY"
)
SAGE6B_EXECUTION_INCOMPLETE = (
    "SAGE_SECOND_UNKNOWN_GAME_M3_EXECUTION_INCOMPLETE_CANDIDATE_ONLY"
)

DEFAULT_REQUESTS_PER_BUDGET = 2
SELECTION_POLICY = "PER_BUDGET_QUOTA_WITH_GLOBAL_CONTEXT_HASH_NOVELTY_AND_STEP_SPREAD"
EXPECTED_CONTEXT_STATE_ORIGIN = "sage6_second_game_live_prefix_frame_before"
EXPECTED_GENERATOR = "SAGE.6a_second_unknown_game_live_mini_frontier"


def run_sage6b_second_unknown_game_m3_execution(
    *,
    source_sage6a_path: str | Path = DEFAULT_SAGE6A_SWITCH_FRONTIER_PATH,
    environments_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    requests_per_budget: int = DEFAULT_REQUESTS_PER_BUDGET,
    env_factory: EnvFactory | None = None,
) -> Dict[str, Any]:
    """Execute a hash-diverse per-budget subset of SAGE.6a M3 requests."""
    source = _load_json(source_sage6a_path)
    validate_sage6b_source(source)
    game_id = str(source.get("summary", {}).get("game_id", ""))
    budgets = tuple(
        int(value) for value in source.get("summary", {}).get("budgets_evaluated", [])
    )
    selected = select_sage6b_execution_requests(
        source,
        requests_per_budget=requests_per_budget,
    )
    env_dir = Path(environments_dir) if environments_dir is not None else _env_dir()

    experiments: List[Dict[str, Any]] = []
    blocked: List[Dict[str, Any]] = []
    for request in selected:
        result = execute_live_prefix_mini_frontier_request(
            request,
            environments_dir=env_dir,
            env_factory=env_factory,
        )
        result = retag_sage6b_execution_result(result)
        if str(result.get("execution_status", "")) == "EXECUTED":
            experiments.append(result)
        else:
            blocked.append(result)

    per_budget = build_sage6b_budget_records(
        source=source,
        budgets=budgets,
        selected=selected,
        experiments=experiments,
        blocked=blocked,
    )
    gate = build_sage6b_gate(
        source=source,
        game_id=game_id,
        budgets=budgets,
        selected=selected,
        experiments=experiments,
        blocked=blocked,
        requests_per_budget=requests_per_budget,
    )
    outcome = (
        SAGE6B_EXECUTION_COMPLETED
        if gate and all(gate.values())
        else SAGE6B_EXECUTION_INCOMPLETE
    )
    summary = summarize_sage6b_execution(
        source=source,
        game_id=game_id,
        budgets=budgets,
        selected=selected,
        experiments=experiments,
        blocked=blocked,
        requests_per_budget=requests_per_budget,
        gate=gate,
        outcome=outcome,
    )
    payload = {
        "config": {
            "schema_version": SAGE6B_SCHEMA_VERSION,
            "source_sage6a_path": str(source_sage6a_path),
            "game_id": game_id,
            "budgets": list(budgets),
            "environments_dir": str(env_dir),
            "requests_per_budget": int(requests_per_budget),
            "selection_policy": SELECTION_POLICY,
            "execution_mode": "live_prefix_replay_context",
            "target_and_control_replay_same_prefix": True,
            "context_snapshot_hash_verification_required": True,
            "benchmark_run": False,
            "inputs_read": ["SAGE.6a"],
            "artifacts_not_modified": [
                "SAGE.6a",
                "SAGE.6",
                "SAGE.5",
                "A32.5",
                "A33.1",
                "A33.2",
                "M2",
                "M3",
                "A40",
                "P2",
            ],
            "scientific_policy": {
                "raw_events_are_candidate_only": True,
                "support_events_are_not_scientific_support": True,
                "contradiction_events_are_not_refutations": True,
                "source_scoped_mechanics_are_quarantined": True,
                "a32_a33_write_performed": False,
            },
        },
        "source_sage6a_context": build_source_sage6a_context(source),
        "selection_audit": build_sage6b_selection_audit(source, selected),
        "selected_execution_requests": [dict(row) for row in selected],
        "controlled_experiments": experiments,
        "blocked_replay_events": blocked,
        "per_budget_results": per_budget,
        "gate": gate,
        "summary": summary,
        "status": "UNRESOLVED",
        "outcome_status": outcome,
        "outcome_status_is_candidate_only": True,
        "truth_status": SAGE6B_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "support": 0,
        "revision_performed": False,
        "confirmation_performed": False,
        "refutation_performed": False,
        "wrong_confirmations": 0,
        "policy_result_counted_as_confirmation": False,
        "support_events_counted_as_support": False,
        "contradiction_events_counted_as_refutation": False,
        "mini_frontier_execution_counted_as_evidence": False,
        "source_scoped_mechanics_reused": 0,
        "cross_game_mechanics_imported": 0,
        "scope_generalization_performed": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }
    if output_path is not None:
        write_sage6b_second_unknown_game_m3_execution(payload, output_path)
    return payload


def select_sage6b_execution_requests(
    source: Mapping[str, Any],
    *,
    requests_per_budget: int = DEFAULT_REQUESTS_PER_BUDGET,
) -> tuple[Dict[str, Any], ...]:
    """Select an exact per-budget quota while maximizing context diversity."""
    limit = max(0, int(requests_per_budget))
    if limit == 0:
        return tuple()
    budgets = tuple(
        int(value) for value in source.get("summary", {}).get("budgets_evaluated", [])
    )
    ready = [
        dict(row)
        for row in source.get("mini_frontier_m3_requests", []) or []
        if str(row.get("status", "")) == "READY_FOR_M3"
        and str(row.get("replayability", "")) == "LIVE_PREFIX_REPLAY_CONTEXT"
        and str(row.get("context_state_origin", "")) == EXPECTED_CONTEXT_STATE_ORIGIN
    ]
    by_budget = {
        budget: sorted(
            (row for row in ready if budget_from_sage6a_request(row) == budget),
            key=request_sort_key,
        )
        for budget in budgets
    }
    selected_by_budget: Dict[int, List[Dict[str, Any]]] = {
        budget: [] for budget in budgets
    }
    selected_hashes: set[str] = set()

    # Allocate the least flexible budgets first.  This lets budget 50 and 300
    # split their four shared contexts before budget 150 consumes novel ones.
    allocation_order = sorted(budgets, key=lambda value: (len(by_budget[value]), value))
    for budget in allocation_order:
        spread = spread_requests_by_step(by_budget[budget])
        bucket = selected_by_budget[budget]
        for row in spread:
            context_hash = str(row.get("context_snapshot_hash", ""))
            if len(bucket) >= limit:
                break
            if context_hash and context_hash not in selected_hashes:
                bucket.append(row)
                selected_hashes.add(context_hash)
        for row in spread:
            if len(bucket) >= limit:
                break
            if row not in bucket:
                bucket.append(row)
                context_hash = str(row.get("context_snapshot_hash", ""))
                if context_hash:
                    selected_hashes.add(context_hash)

    selected = [
        row
        for budget in budgets
        for row in sorted(selected_by_budget[budget], key=request_sort_key)
    ]
    return tuple(selected)


def spread_requests_by_step(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Dict[str, Any], ...]:
    """Return first/last/second/penultimate requests in deterministic order."""
    ordered = sorted((dict(row) for row in rows), key=request_sort_key)
    spread: List[Dict[str, Any]] = []
    left = 0
    right = len(ordered) - 1
    while left <= right:
        spread.append(ordered[left])
        left += 1
        if left <= right:
            spread.append(ordered[right])
            right -= 1
    return tuple(spread)


def retag_sage6b_execution_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    row = dict(result)
    row.update(
        {
            "executed_by": "SAGE.6b_second_unknown_game_m3_execution",
            "truth_status": SAGE6B_TRUTH_STATUS,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "revision_performed": False,
            "wrong_confirmations": 0,
            "support_events_counted_as_support": False,
            "contradiction_events_counted_as_refutation": False,
            "observation_counted_as_confirmation": False,
        }
    )
    return row


def build_sage6b_budget_records(
    *,
    source: Mapping[str, Any],
    budgets: Sequence[int],
    selected: Sequence[Mapping[str, Any]],
    experiments: Sequence[Mapping[str, Any]],
    blocked: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    available = list(source.get("mini_frontier_m3_requests", []) or [])
    records: List[Dict[str, Any]] = []
    for budget in budgets:
        selected_rows = [
            row for row in selected if budget_from_sage6a_request(row) == int(budget)
        ]
        request_ids = {str(row.get("request_id", "")) for row in selected_rows}
        executed_rows = [
            row for row in experiments if str(row.get("request_id", "")) in request_ids
        ]
        blocked_rows = [
            row for row in blocked if str(row.get("request_id", "")) in request_ids
        ]
        records.append(
            {
                "budget": int(budget),
                "requests_available": sum(
                    1
                    for row in available
                    if budget_from_sage6a_request(row) == int(budget)
                ),
                "requests_selected": len(selected_rows),
                "selected_steps": [
                    int(row.get("source_step", 0) or 0) for row in selected_rows
                ],
                "selected_context_snapshot_hashes": [
                    str(row.get("context_snapshot_hash", "")) for row in selected_rows
                ],
                "requests_executed": len(executed_rows),
                "requests_blocked": len(blocked_rows),
                "live_prefix_replay_exact_events": sum(
                    1
                    for row in executed_rows
                    if bool(row.get("live_prefix_replay_exact", False))
                ),
                "context_snapshot_hash_verified_events": sum(
                    1 for row in executed_rows if experiment_hash_verified(row)
                ),
                "support_events": sum(
                    int(row.get("support_events", 0) or 0) for row in executed_rows
                ),
                "contradiction_events": sum(
                    int(row.get("contradiction_events", 0) or 0)
                    for row in executed_rows
                ),
                "neutral_events": sum(
                    int(row.get("neutral_events", 0) or 0) for row in executed_rows
                ),
                "target_signals": [
                    float(row.get("target_signal", 0.0) or 0.0) for row in executed_rows
                ],
                "control_signals": [
                    float(row.get("control_signal", 0.0) or 0.0)
                    for row in executed_rows
                ],
                "support": 0,
                "truth_status": SAGE6B_TRUTH_STATUS,
                "revision_status": "CANDIDATE_ONLY",
                "a32_write_performed": False,
                "a33_write_performed": False,
            }
        )
    return records


def build_sage6b_gate(
    *,
    source: Mapping[str, Any],
    game_id: str,
    budgets: Sequence[int],
    selected: Sequence[Mapping[str, Any]],
    experiments: Sequence[Mapping[str, Any]],
    blocked: Sequence[Mapping[str, Any]],
    requests_per_budget: int,
) -> Dict[str, bool]:
    request_ids = [str(row.get("request_id", "")) for row in selected]
    context_hashes = [str(row.get("context_snapshot_hash", "")) for row in selected]
    selected_by_budget = Counter(budget_from_sage6a_request(row) for row in selected)
    executed_ids = {str(row.get("request_id", "")) for row in experiments}
    available_by_budget = Counter(
        budget_from_sage6a_request(row)
        for row in source.get("mini_frontier_m3_requests", []) or []
    )
    expected = max(0, int(requests_per_budget))
    return {
        "source_sage6a_gate_passed": bool(
            source.get("summary", {}).get("gate_passed", False)
        )
        and all(bool(value) for value in source.get("gate", {}).values()),
        "all_source_budgets_have_sufficient_requests": all(
            int(available_by_budget.get(int(budget), 0)) >= expected
            for budget in budgets
        ),
        "all_source_budgets_selected": all(
            int(selected_by_budget.get(int(budget), 0)) == expected
            for budget in budgets
        ),
        "selected_request_ids_unique": "" not in request_ids
        and len(request_ids) == len(set(request_ids)),
        "selected_context_hashes_unique": "" not in context_hashes
        and len(context_hashes) == len(set(context_hashes)),
        "selected_requests_ready_for_m3": bool(selected)
        and all(str(row.get("status", "")) == "READY_FOR_M3" for row in selected),
        "selected_requests_live_prefix_replayable": bool(selected)
        and all(
            str(row.get("replayability", "")) == "LIVE_PREFIX_REPLAY_CONTEXT"
            and str(row.get("context_state_origin", ""))
            == EXPECTED_CONTEXT_STATE_ORIGIN
            for row in selected
        ),
        "selected_requests_scoped_to_second_game": bool(selected)
        and all(str(row.get("game_id", "")) == game_id for row in selected),
        "all_selected_requests_executed": bool(selected)
        and len(experiments) == len(selected)
        and set(request_ids) == executed_ids,
        "no_blocked_replays": len(blocked) == 0,
        "all_live_prefix_replays_exact": bool(experiments)
        and all(
            bool(row.get("live_prefix_replay_exact", False)) for row in experiments
        ),
        "all_context_snapshot_hashes_verified": bool(experiments)
        and all(experiment_hash_verified(row) for row in experiments),
        "all_outputs_candidate_only": all(
            int(row.get("support", 0) or 0) == 0
            and str(row.get("truth_status", "")) == SAGE6B_TRUTH_STATUS
            for row in experiments
        ),
        "source_registry_quarantine_preserved": int(
            source.get("source_scoped_mechanics_reused", 0) or 0
        )
        == 0
        and int(source.get("cross_game_mechanics_imported", 0) or 0) == 0
        and not bool(source.get("scope_generalization_performed", False)),
    }


def summarize_sage6b_execution(
    *,
    source: Mapping[str, Any],
    game_id: str,
    budgets: Sequence[int],
    selected: Sequence[Mapping[str, Any]],
    experiments: Sequence[Mapping[str, Any]],
    blocked: Sequence[Mapping[str, Any]],
    requests_per_budget: int,
    gate: Mapping[str, bool],
    outcome: str,
) -> Dict[str, Any]:
    available = list(source.get("mini_frontier_m3_requests", []) or [])
    selected_by_budget = Counter(budget_from_sage6a_request(row) for row in selected)
    executed_by_budget = Counter(budget_from_sage6a_request(row) for row in experiments)
    support_events = sum(int(row.get("support_events", 0) or 0) for row in experiments)
    contradiction_events = sum(
        int(row.get("contradiction_events", 0) or 0) for row in experiments
    )
    effect_sizes = [
        float(row.get("controlled_delta", {}).get("effect_size", 0.0) or 0.0)
        for row in experiments
    ]
    return {
        "source_sage6a_outcome_status": str(source.get("outcome_status", "")),
        "game_id": game_id,
        "budgets_available": [int(value) for value in budgets],
        "requests_available": len(available),
        "requests_per_budget": int(requests_per_budget),
        "requests_selected": len(selected),
        "requests_selected_by_budget": {
            str(key): value for key, value in sorted(selected_by_budget.items())
        },
        "unique_selected_context_snapshot_hashes": len(
            {str(row.get("context_snapshot_hash", "")) for row in selected}
        ),
        "selected_source_steps_by_budget": {
            str(budget): [
                int(row.get("source_step", 0) or 0)
                for row in selected
                if budget_from_sage6a_request(row) == int(budget)
            ]
            for budget in budgets
        },
        "requests_executed": len(experiments),
        "requests_executed_by_budget": {
            str(key): value for key, value in sorted(executed_by_budget.items())
        },
        "requests_blocked": len(blocked),
        "live_prefix_replay_exact_events": sum(
            1 for row in experiments if bool(row.get("live_prefix_replay_exact", False))
        ),
        "context_snapshot_hash_verified_events": sum(
            1 for row in experiments if experiment_hash_verified(row)
        ),
        "target_actions_executed": count_values(
            row.get("target_action", "") for row in experiments
        ),
        "control_actions_executed": count_values(
            row.get("control_action", "") for row in experiments
        ),
        "hypothesis_families_executed": count_values(
            row.get("hypothesis_family", "") for row in experiments
        ),
        "target_signal_total": sum(
            float(row.get("target_signal", 0.0) or 0.0) for row in experiments
        ),
        "control_signal_total": sum(
            float(row.get("control_signal", 0.0) or 0.0) for row in experiments
        ),
        "controlled_effect_sizes": effect_sizes,
        "positive_effect_events": sum(1 for value in effect_sizes if value > 0),
        "negative_effect_events": sum(1 for value in effect_sizes if value < 0),
        "zero_effect_events": sum(1 for value in effect_sizes if value == 0),
        "raw_support_events": support_events,
        "raw_contradiction_events": contradiction_events,
        "raw_neutral_events": sum(
            int(row.get("neutral_events", 0) or 0) for row in experiments
        ),
        "blocked_replay_events": len(blocked),
        "execution_failures": sum(
            1
            for row in blocked
            if "failed" in str(row.get("blocked_reason", "")).lower()
        ),
        "source_scoped_mechanics_reused": 0,
        "cross_game_mechanics_imported": 0,
        "gate_passed": bool(gate) and all(bool(value) for value in gate.values()),
        "outcome_status": outcome,
        "support": 0,
        "truth_status": SAGE6B_TRUTH_STATUS,
        "revision_status": "CANDIDATE_ONLY",
        "a32_write_performed": False,
        "a33_write_performed": False,
        "wrong_confirmations": 0,
    }


def build_source_sage6a_context(source: Mapping[str, Any]) -> Dict[str, Any]:
    summary = dict(source.get("summary", {}) or {})
    return {
        "source_outcome_status": str(source.get("outcome_status", "")),
        "game_id": str(summary.get("game_id", "")),
        "budgets": list(summary.get("budgets_evaluated", []) or []),
        "source_switches_expected": int(
            summary.get("source_switches_expected", 0) or 0
        ),
        "switches_reproduced": int(summary.get("switches_reproduced", 0) or 0),
        "effective_requests_generated": int(
            summary.get("effective_requests_generated", 0) or 0
        ),
        "unique_context_snapshot_hashes": int(
            summary.get("unique_context_snapshot_hashes", 0) or 0
        ),
        "source_scoped_mechanics_reused": int(
            source.get("source_scoped_mechanics_reused", 0) or 0
        ),
        "cross_game_mechanics_imported": int(
            source.get("cross_game_mechanics_imported", 0) or 0
        ),
        "scope_generalization_performed": bool(
            source.get("scope_generalization_performed", False)
        ),
        "source_counted_as_scientific_support": False,
        "support": 0,
        "truth_status": SAGE6B_TRUTH_STATUS,
    }


def build_sage6b_selection_audit(
    source: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    selected_index = {
        str(row.get("request_id", "")): index + 1 for index, row in enumerate(selected)
    }
    rows = sorted(
        (dict(row) for row in source.get("mini_frontier_m3_requests", []) or []),
        key=lambda row: (budget_from_sage6a_request(row), *request_sort_key(row)),
    )
    return [
        {
            "request_id": str(row.get("request_id", "")),
            "budget": budget_from_sage6a_request(row),
            "source_step": int(row.get("source_step", 0) or 0),
            "context_snapshot_hash": str(row.get("context_snapshot_hash", "")),
            "target_action": str(row.get("target_action", "")),
            "hypothesis_family": str(row.get("hypothesis_family", "")),
            "ready_for_m3": str(row.get("status", "")) == "READY_FOR_M3",
            "selected": str(row.get("request_id", "")) in selected_index,
            "selection_rank": int(
                selected_index.get(str(row.get("request_id", "")), 0)
            ),
            "outcome_metrics_read_for_selection": False,
            "support": 0,
            "truth_status": SAGE6B_TRUTH_STATUS,
        }
        for row in rows
    ]


def validate_sage6b_source(source: Mapping[str, Any]) -> None:
    config = dict(source.get("config", {}) or {})
    summary = dict(source.get("summary", {}) or {})
    if str(config.get("schema_version", "")) != SAGE6A_SCHEMA_VERSION:
        raise ValueError("SAGE.6a schema version is not supported by SAGE.6b")
    if str(source.get("outcome_status", "")) != SAGE6A_FRONTIER_GENERATED:
        raise ValueError("SAGE.6b requires a generated SAGE.6a frontier")
    if str(source.get("status", "")) != "UNRESOLVED":
        raise ValueError("SAGE.6a source must remain unresolved")
    if str(source.get("truth_status", "")) != SAGE6A_TRUTH_STATUS:
        raise ValueError("SAGE.6a truth must remain unevaluated")
    if str(source.get("revision_status", "")) != "CANDIDATE_ONLY":
        raise ValueError("SAGE.6a source must remain candidate-only")
    if int(source.get("support", 0) or 0) != 0:
        raise ValueError("SAGE.6a support must remain 0")
    if (
        bool(source.get("revision_performed", False))
        or bool(source.get("confirmation_performed", False))
        or bool(source.get("refutation_performed", False))
    ):
        raise ValueError("SAGE.6a cannot perform a scientific verdict")
    if int(source.get("wrong_confirmations", 0) or 0) != 0:
        raise ValueError("SAGE.6a wrong_confirmations must remain 0")
    if bool(source.get("a32_write_performed", False)) or bool(
        source.get("a33_write_performed", False)
    ):
        raise ValueError("SAGE.6a cannot write A32/A33")
    if (
        bool(source.get("policy_result_counted_as_confirmation", False))
        or bool(source.get("generated_requests_counted_as_support", False))
        or bool(source.get("mini_frontier_counted_as_evidence", False))
    ):
        raise ValueError("SAGE.6a generated frontier cannot count as evidence")
    if (
        int(source.get("source_scoped_mechanics_reused", 0) or 0) != 0
        or int(source.get("cross_game_mechanics_imported", 0) or 0) != 0
    ):
        raise ValueError("SAGE.6a cannot import source-game mechanics")
    if bool(source.get("scope_generalization_performed", False)):
        raise ValueError("SAGE.6a cannot generalize the source registry scope")
    gate = dict(source.get("gate", {}) or {})
    if (
        not gate
        or not all(bool(value) for value in gate.values())
        or not bool(summary.get("gate_passed", False))
    ):
        raise ValueError("every SAGE.6a source gate must pass")
    game_id = str(summary.get("game_id", ""))
    if not game_id or game_id != str(config.get("game_id", "")):
        raise ValueError("SAGE.6a selected game identity must align")
    budgets = [int(value) for value in summary.get("budgets_evaluated", []) or []]
    if not budgets or budgets != [
        int(value) for value in config.get("budgets", []) or []
    ]:
        raise ValueError("SAGE.6a budgets must align")
    if int(summary.get("switches_reproduced", 0) or 0) != int(
        summary.get("source_switches_expected", 0) or 0
    ) or not bool(summary.get("source_switch_count_reproduced_exactly", False)):
        raise ValueError("SAGE.6a must reproduce its source switch count exactly")
    requests = [
        row
        for row in source.get("mini_frontier_m3_requests", []) or []
        if isinstance(row, Mapping)
    ]
    if len(requests) != int(summary.get("effective_requests_generated", 0) or 0):
        raise ValueError("SAGE.6a request count must match its summary")
    request_ids = [str(row.get("request_id", "")) for row in requests]
    if "" in request_ids or len(request_ids) != len(set(request_ids)):
        raise ValueError("SAGE.6a request ids must be non-empty and unique")
    if not all(
        str(row.get("game_id", "")) == game_id
        and str(row.get("status", "")) == "READY_FOR_M3"
        and str(row.get("replayability", "")) == "LIVE_PREFIX_REPLAY_CONTEXT"
        and str(row.get("context_state_origin", "")) == EXPECTED_CONTEXT_STATE_ORIGIN
        and str(row.get("generated_by", "")) == EXPECTED_GENERATOR
        and int(row.get("support", 0) or 0) == 0
        for row in requests
    ):
        raise ValueError("SAGE.6a requests must remain ready, replayable, and scoped")


def write_sage6b_second_unknown_game_m3_execution(
    payload: Mapping[str, Any],
    output_path: str | Path = DEFAULT_SAGE6B_M3_EXECUTION_PATH,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def budget_from_sage6a_request(row: Mapping[str, Any]) -> int:
    parts = str(row.get("request_id", "")).split("::")
    try:
        return int(parts[-2])
    except (ValueError, IndexError):
        return 0


def request_sort_key(row: Mapping[str, Any]) -> tuple[int, str]:
    return (
        int(row.get("source_step", 0) or 0),
        str(row.get("request_id", "")),
    )


def experiment_hash_verified(row: Mapping[str, Any]) -> bool:
    return bool(row.get("target_context_signature_verified", False)) and bool(
        row.get("control_context_signature_verified", False)
    )


def count_values(values: Iterable[Any]) -> Dict[str, int]:
    result: Counter[str] = Counter(str(value) for value in values if str(value))
    return dict(sorted(result.items()))


def _load_json(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute a stratified SAGE.6a M3 subset on wa30."
    )
    parser.add_argument(
        "--source-sage6a", default=str(DEFAULT_SAGE6A_SWITCH_FRONTIER_PATH)
    )
    parser.add_argument("--environments-dir", default=None)
    parser.add_argument("--out", default=str(DEFAULT_SAGE6B_M3_EXECUTION_PATH))
    parser.add_argument(
        "--requests-per-budget",
        type=int,
        default=DEFAULT_REQUESTS_PER_BUDGET,
    )
    args = parser.parse_args(argv)
    run_sage6b_second_unknown_game_m3_execution(
        source_sage6a_path=args.source_sage6a,
        environments_dir=args.environments_dir,
        output_path=args.out,
        requests_per_budget=args.requests_per_budget,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
