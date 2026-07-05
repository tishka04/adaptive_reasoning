"""Tests for M3.G7 state-conditioned LLM hypothesis execution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from theory.m2.local_llm_generator import (
    DEFAULT_M1_CANDIDATES_PATH,
    DEFAULT_M2G2_HYPOTHESES_PATH,
    DEFAULT_M3G6_RESULTS_PATH,
    LocalLLMConfig,
    run_state_conditioned_llm_generation,
)
from theory.m3.risk_aware_objective_completion_experiment_executor import (
    DEFAULT_RISK_AWARE_OBJECTIVE_EXPERIMENT_RESULTS_OUTPUT_PATH,
)
from theory.m3.state_conditioned_llm_hypothesis_execution import (
    LLM_REDUCES_TO_EXISTING_G6_FAILURE,
    SEMANTIC_COMPILATION_GAP,
    UNBOUND_CELL_STATUS,
    run_state_conditioned_llm_hypothesis_execution,
    state_conditioned_llm_execution_outcome_status,
)
from theory.m2.local_llm_generator import DEFAULT_FRONTIERS_PATH


REPO_ROOT = Path(__file__).resolve().parents[1]
G6_PATH = REPO_ROOT / DEFAULT_RISK_AWARE_OBJECTIVE_EXPERIMENT_RESULTS_OUTPUT_PATH


@pytest.fixture(scope="module")
def requests_path(tmp_path_factory) -> Path:
    """Compile the M2.13a requests from the real upstream artifacts."""
    outputs = run_state_conditioned_llm_generation(
        frontiers_path=REPO_ROOT / DEFAULT_FRONTIERS_PATH,
        m2g2_path=REPO_ROOT / DEFAULT_M2G2_HYPOTHESES_PATH,
        m1_candidates_path=REPO_ROOT / DEFAULT_M1_CANDIDATES_PATH,
        m3g6_results_path=REPO_ROOT / DEFAULT_M3G6_RESULTS_PATH,
        config=LocalLLMConfig(),
    )
    path = tmp_path_factory.mktemp("m3g7") / "requests.json"
    path.write_text(
        json.dumps(outputs["m3_payload"], indent=2, sort_keys=True), encoding="utf-8"
    )
    return path


def _run(requests_path: Path) -> dict:
    return run_state_conditioned_llm_hypothesis_execution(
        requests_path=requests_path,
        m3g6_results_path=G6_PATH,
    )


def test_consumes_five_ready_and_ignores_blocked_unlock(requests_path: Path) -> None:
    payload = _run(requests_path)
    summary = payload["summary"]

    assert summary["ready_requests_consumed"] == 5
    assert summary["blocked_requests_ignored"] == 1
    ignored = payload["ignored_requests"]
    assert len(ignored) == 1
    assert ignored[0]["metric"] == "available_actions_before_after"


def test_replays_action6_and_binds_to_g6_substrate(requests_path: Path) -> None:
    payload = _run(requests_path)
    cells = payload["execution_results"]

    assert len(cells) == 5
    for cell in cells:
        assert cell["context_replay"] == ["ACTION6"]
        assert cell["candidate_target_sequence"][0] == "ACTION6"
        assert cell["target_action"] in {"ACTION3", "ACTION4"}
        assert cell["bind_status"] == "BOUND"
        assert cell["candidate_condition_key"] in {"ACTION6,ACTION3", "ACTION6,ACTION4"}


def test_measures_both_metrics(requests_path: Path) -> None:
    payload = _run(requests_path)
    assert set(payload["summary"]["metrics_executed"]) == {
        "objective_completion_signal",
        "terminal_reentry_rate",
    }


def test_honest_outcome_reduces_to_existing_g6_failure(requests_path: Path) -> None:
    payload = _run(requests_path)
    summary = payload["summary"]

    assert (
        payload["state_conditioned_llm_execution_outcome_status"]
        == LLM_REDUCES_TO_EXISTING_G6_FAILURE
    )
    assert summary["objective_completion_signal"] is False
    assert summary["objective_completion_candidate_cells"] == 0
    assert summary["candidate_supported_cells"] == 0
    assert summary["cells_reduce_to_g6_failure"] == 5
    assert summary["adds_signal_beyond_g6"] is False
    assert summary["reproduces_g6_proxy_completion_divergence"] is True
    assert summary["recommends_m2_14_world_model"] is True
    assert summary["g6_objective_completion_signal"] is False


def test_candidate_only_guardrails_preserved(requests_path: Path) -> None:
    payload = _run(requests_path)

    assert payload["support"] == 0
    assert payload["revision_status"] == "CANDIDATE_ONLY"
    assert payload["truth_status"] == "NOT_EVALUATED_BY_M3"
    assert payload["policy_rollout_performed"] is False
    assert payload["environment_step_performed"] is False
    assert payload["experiment_result_counted_as_scientific_verdict"] is False
    assert payload["candidate_signal_counted_as_scientific_verdict"] is False
    assert payload["llm_request_counted_as_evidence"] is False
    assert payload["a32_write_performed"] is False
    assert payload["a33_write_performed"] is False
    for cell in payload["execution_results"]:
        assert cell["support"] == 0
        assert cell["revision_status"] == "CANDIDATE_ONLY"
        assert cell["truth_status"] == "NOT_EVALUATED_BY_M3"
        assert cell["cell_result_counted_as_scientific_verdict"] is False
        assert cell["a32_write_performed"] is False
        assert cell["a33_write_performed"] is False


def test_terminal_reentry_candidates_do_not_beat_dynamic_controls(
    requests_path: Path,
) -> None:
    payload = _run(requests_path)
    terminal_cells = [
        cell
        for cell in payload["execution_results"]
        if cell["metric"] == "terminal_reentry_rate"
    ]
    assert terminal_cells
    for cell in terminal_cells:
        # ACTION6,ACTION3 / ACTION6,ACTION4 re-enter terminal more than controls.
        assert cell["terminal_reentry_rate"] > cell["best_control_terminal_reentry_rate"]
        assert cell["candidate_supported"] is False
        assert cell["candidate_falsified"] is True


def test_outcome_helper_flags_semantic_gap_when_all_unbound() -> None:
    unbound = [
        {"bind_status": "UNBOUND", "cell_execution_status": UNBOUND_CELL_STATUS},
    ]
    assert (
        state_conditioned_llm_execution_outcome_status(execution_results=unbound)
        == SEMANTIC_COMPILATION_GAP
    )


def test_new_signal_outcome_when_candidate_supported() -> None:
    from theory.m3.state_conditioned_llm_hypothesis_execution import (
        LLM_ADDS_NEW_TESTABLE_SIGNAL,
    )

    bound = [
        {
            "bind_status": "BOUND",
            "candidate_supported": True,
            "reduces_to_g6_failure": False,
            "already_measured_in_g6": True,
        }
    ]
    assert (
        state_conditioned_llm_execution_outcome_status(execution_results=bound)
        == LLM_ADDS_NEW_TESTABLE_SIGNAL
    )


def test_input_validation_rejects_support_positive_requests(tmp_path: Path) -> None:
    bad = {
        "experiment_requests": [
            {
                "status": "READY_FOR_M3",
                "support": 1,
                "truth_status": "NOT_EVALUATED_BY_M2",
            }
        ],
        "summary": {"ready_for_m3": 1},
    }
    bad_path = tmp_path / "bad_requests.json"
    bad_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        run_state_conditioned_llm_hypothesis_execution(
            requests_path=bad_path, m3g6_results_path=G6_PATH
        )
