"""Tests for M3.G3 safe-stop diversity acquisition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import pytest

from theory.m3.objective_conversion_multi_safe_stop_validation import (
    DEFAULT_OBJECTIVE_CONVERSION_MULTI_SAFE_STOP_VALIDATION_OUTPUT_PATH,
)
from theory.m3.objective_conversion_safe_stop_diversity_sampler import (
    ACCEPTED_DIVERSE_SAFE_STOP_CANDIDATE,
    DUPLICATE_OR_NEAR_DUPLICATE_REJECTED,
    INSUFFICIENT_DIVERSITY,
    REPLAY_INEXACT_REJECTED,
    SUFFICIENT_FOR_M3_G4,
    TERMINAL_OR_UNSAFE_REJECTED,
    SafeStopCandidatePlan,
    generate_safe_stop_candidate_plans,
    run_objective_conversion_safe_stop_diversity_sampling,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_G2_PATH = (
    REPO_ROOT / DEFAULT_OBJECTIVE_CONVERSION_MULTI_SAFE_STOP_VALIDATION_OUTPUT_PATH
)


def _plan(index: int) -> SafeStopCandidatePlan:
    return SafeStopCandidatePlan(
        plan_id=f"stub_plan_{index}",
        planned_prefix=(
            {"action": "ACTION3", "action_args": {}},
            {"action": "ACTION4", "action_args": {}},
        ),
        sampling_family="stub",
        anti_attractor_rationale="stub diversity fixture",
        tie_break_seed=index,
    )


def _record(
    plan: SafeStopCandidatePlan,
    *,
    state: str,
    prefix: str,
    relation: str,
    replay_exact: bool = True,
    non_terminal: bool = True,
    terminal_safe: bool = True,
    hold_measurable: bool = True,
) -> Mapping[str, object]:
    return {
        **plan.to_dict(),
        "status": "SAFE_STOP_CANDIDATE_MEASURED",
        "execution_performed": True,
        "blocked_reason": "",
        "captured_prefix": [dict(step) for step in plan.planned_prefix],
        "captured_prefix_len": len(plan.planned_prefix),
        "captured_prefix_hash": prefix,
        "safe_stop_state_hash": state,
        "relation_state_signature": relation,
        "terminal_horizon_estimate": {
            "estimated_moves_remaining": 40,
            "source": "stub",
        },
        "hold_baseline_terminal_adjusted_progress": 100.0,
        "hold_baseline_levels_completed": 0,
        "hold_baseline_terminal": False,
        "replay_exact": replay_exact,
        "non_terminal": non_terminal,
        "terminal_safe": terminal_safe,
        "hold_baseline_measurable": hold_measurable,
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
        "wrong_confirmations": 0,
    }


def test_generates_anti_attractor_plans_from_g2_prefix() -> None:
    payload = json.loads(SOURCE_G2_PATH.read_text(encoding="utf-8"))
    plans = generate_safe_stop_candidate_plans(payload)
    families = {plan.sampling_family for plan in plans}
    assert "base_prefix_truncation" in families
    assert "phase_shifted_relation_prefix" in families
    assert "action6_perturbation" in families
    assert all(plan.planned_prefix for plan in plans)


def test_sufficient_diversity_when_three_distinct_safe_stops_accepted() -> None:
    plans = tuple(_plan(index) for index in range(3))

    def executor(plan: SafeStopCandidatePlan):
        index = int(plan.tie_break_seed)
        return _record(
            plan,
            state=f"state::{index}",
            prefix=f"prefix::{index}",
            relation=f"relation::{index}",
        )

    payload = run_objective_conversion_safe_stop_diversity_sampling(
        source_g2_path=SOURCE_G2_PATH,
        candidate_plans=plans,
        candidate_executor=executor,
        min_diverse_safe_stops=3,
    )
    assert payload["diversity_status"] == SUFFICIENT_FOR_M3_G4
    assert payload["summary"]["accepted_diverse_safe_stops"] == 3
    assert payload["summary"]["ready_for_m3_g4"] is True


def test_duplicate_or_near_duplicate_is_rejected() -> None:
    plans = tuple(_plan(index) for index in range(3))

    def executor(plan: SafeStopCandidatePlan):
        return _record(
            plan,
            state="state::same",
            prefix=f"prefix::{plan.tie_break_seed}",
            relation=f"relation::{plan.tie_break_seed}",
        )

    payload = run_objective_conversion_safe_stop_diversity_sampling(
        source_g2_path=SOURCE_G2_PATH,
        candidate_plans=plans,
        candidate_executor=executor,
        min_diverse_safe_stops=3,
    )
    assert payload["diversity_status"] == INSUFFICIENT_DIVERSITY
    assert payload["summary"]["accepted_diverse_safe_stops"] == 1
    assert payload["summary"]["duplicate_or_near_duplicate_rejected"] == 2
    rejected = payload["rejected_safe_stop_candidates"]
    assert all(
        row["acceptance_status"] == DUPLICATE_OR_NEAR_DUPLICATE_REJECTED
        for row in rejected
    )


def test_unsafe_and_replay_inexact_candidates_rejected() -> None:
    plans = (_plan(0), _plan(1), _plan(2))

    def executor(plan: SafeStopCandidatePlan):
        if plan.tie_break_seed == 1:
            return _record(
                plan,
                state="state::unsafe",
                prefix="prefix::unsafe",
                relation="relation::unsafe",
                terminal_safe=False,
            )
        if plan.tie_break_seed == 2:
            return _record(
                plan,
                state="state::inexact",
                prefix="prefix::inexact",
                relation="relation::inexact",
                replay_exact=False,
            )
        return _record(
            plan,
            state="state::ok",
            prefix="prefix::ok",
            relation="relation::ok",
        )

    payload = run_objective_conversion_safe_stop_diversity_sampling(
        source_g2_path=SOURCE_G2_PATH,
        candidate_plans=plans,
        candidate_executor=executor,
        min_diverse_safe_stops=3,
    )
    statuses = {
        row["acceptance_status"] for row in payload["safe_stop_candidate_records"]
    }
    assert ACCEPTED_DIVERSE_SAFE_STOP_CANDIDATE in statuses
    assert TERMINAL_OR_UNSAFE_REJECTED in statuses
    assert REPLAY_INEXACT_REJECTED in statuses


def test_reads_g2_as_diversity_diagnostic_not_signal_refutation() -> None:
    payload = run_objective_conversion_safe_stop_diversity_sampling(
        source_g2_path=SOURCE_G2_PATH,
        candidate_plans=(_plan(0),),
        candidate_executor=lambda plan: _record(
            plan,
            state="state::0",
            prefix="prefix::0",
            relation="relation::0",
        ),
    )
    diagnostic = payload["source_g2_diversity_diagnostic"]
    assert diagnostic["source_unique_safe_stop_captures"] == 1
    assert diagnostic["source_g2_counted_as_signal_refutation"] is False
    assert diagnostic["source_g2_counted_as_confirmation"] is False


def test_guardrails_locked() -> None:
    payload = run_objective_conversion_safe_stop_diversity_sampling(
        source_g2_path=SOURCE_G2_PATH,
        candidate_plans=tuple(_plan(index) for index in range(3)),
        candidate_executor=lambda plan: _record(
            plan,
            state=f"state::{plan.tie_break_seed}",
            prefix=f"prefix::{plan.tie_break_seed}",
            relation=f"relation::{plan.tie_break_seed}",
        ),
    )
    assert payload["support"] == 0
    assert payload["revision_status"] == "CANDIDATE_ONLY"
    assert payload["truth_status"] == "NOT_EVALUATED_BY_M3"
    assert payload["objective_conversion_sequences_tested"] is False
    assert payload["a32_write_performed"] is False
    assert payload["a33_write_performed"] is False
    for record in payload["safe_stop_candidate_records"]:
        assert record["support"] == 0
        assert record["truth_status"] == "NOT_EVALUATED_BY_M3"


def test_rejects_source_g2_with_support(tmp_path: Path) -> None:
    source = json.loads(SOURCE_G2_PATH.read_text(encoding="utf-8"))
    source["support"] = 1
    bad_path = tmp_path / "bad_g2.json"
    bad_path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError):
        run_objective_conversion_safe_stop_diversity_sampling(
            source_g2_path=bad_path,
            candidate_plans=(_plan(0),),
            candidate_executor=lambda plan: _record(
                plan,
                state="state::0",
                prefix="prefix::0",
                relation="relation::0",
            ),
        )
