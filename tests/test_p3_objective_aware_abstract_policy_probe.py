import json
from pathlib import Path

import pytest

from theory.p3.abstract_mechanic_policy_probe import (
    ABSTRACT_MECHANIC_POLICY,
    GREEDY_CHANGED_PIXELS_POLICY,
    RANDOM_AVAILABLE_POLICY,
    TERMINAL_HORIZON_GUARD_POLICY,
    build_abstract_mechanic_policy_adapter,
)
from theory.p3.objective_aware_abstract_policy_probe import (
    POLICY_OBJECTIVE_USEFUL_CANDIDATE_ONLY,
    build_objective_aware_abstract_policy_adapter,
    consolidate_objective_aware_abstract_policy_utility,
    objective_condition_name,
    run_objective_aware_abstract_policy_rollout,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _symbolic_model_payload(*, support: int = 0) -> dict:
    return {
        "summary": {
            "ready_for_policy_probe_candidate_only": True,
            "support": support,
            "model_status": "CANDIDATE_ONLY",
        },
        "candidate_symbolic_model": {
            "candidate_symbolic_model_id": "m3g0_7::test::symbolic_model",
            "model_status": "CANDIDATE_ONLY",
            "actor_candidates": [{"entity_id": "E_actor"}],
            "action_models": {
                "ACTION3": {
                    "candidate_effects": ["move_entity"],
                    "relation_effects": ["distance_decreases"],
                    "status": "CONTEXT_STABLE_CANDIDATE_ONLY",
                    "support": 0,
                }
            },
            "relation_model": {
                "actor_relation_effects": [
                    {
                        "action": "ACTION3",
                        "source_entity": "E_actor",
                        "target_entity": "E_target",
                        "relation_delta_type": "distance_decreases",
                        "support": 0,
                    }
                ]
            },
            "dynamic_invariants": {},
            "caveats": [],
            "model_counted_as_confirmation": False,
            "support": support,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_M3",
        },
        "model_counted_as_confirmation": False,
        "support": support,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def _hud_payload(*, support: int = 0, verdict: bool = False) -> dict:
    return {
        "summary": {
            "support": support,
            "action6_prefix_count_used_as_decision_variable": False,
        },
        "support": support,
        "candidate_policy_counted_as_confirmation": verdict,
        "policy_result_counted_as_scientific_verdict": False,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_P3_AGENT_PROBE",
    }


def _objective_adapter_fixture(tmp_path: Path, *, lambdas=(0.0, 10.0)) -> Path:
    symbolic_path = _write_json(tmp_path / "symbolic.json", _symbolic_model_payload())
    abstract_payload = build_abstract_mechanic_policy_adapter(symbolic_model_path=symbolic_path)
    abstract_path = _write_json(tmp_path / "abstract_adapter.json", abstract_payload)
    hud_path = _write_json(tmp_path / "hud.json", _hud_payload())
    adapter = build_objective_aware_abstract_policy_adapter(
        abstract_adapter_path=abstract_path,
        hud_policy_probe_path=hud_path,
        symbolic_model_path=symbolic_path,
        terminal_risk_lambdas=lambdas,
    )
    return _write_json(tmp_path / "objective_adapter.json", adapter)


def test_objective_adapter_builds_risk_variants_without_confirmation(tmp_path: Path) -> None:
    adapter_path = _objective_adapter_fixture(tmp_path)
    payload = json.loads(adapter_path.read_text(encoding="utf-8"))

    adapter = payload["objective_aware_policy_adapter"]
    assert payload["summary"]["ready_for_objective_aware_policy_probe"] is True
    assert adapter["lambda_terminal_risk_values"] == [0.0, 10.0]
    assert [row["condition"] for row in adapter["policy_variants"]] == [
        objective_condition_name(0.0),
        objective_condition_name(10.0),
    ]
    assert adapter["terminal_horizon_source_counted_as_confirmation"] is False
    assert adapter["candidate_model_counted_as_confirmed_mechanic"] is False
    assert payload["support"] == 0
    assert payload["a32_write_performed"] is False
    assert payload["a33_write_performed"] is False


def test_objective_rollout_aggregates_terminal_adjusted_metric(tmp_path: Path) -> None:
    adapter_path = _objective_adapter_fixture(tmp_path)
    objective_safe = objective_condition_name(10.0)
    objective_risky = objective_condition_name(0.0)

    def fake_executor(condition, budget, seed, adapter, env_path, game_id):
        terminal = condition in {ABSTRACT_MECHANIC_POLICY, objective_risky}
        progress = 100.0
        if condition == objective_safe:
            progress = 100.0
            terminal = False
        elif condition == GREEDY_CHANGED_PIXELS_POLICY:
            progress = 95.0
        elif condition in {RANDOM_AVAILABLE_POLICY, TERMINAL_HORIZON_GUARD_POLICY}:
            progress = 50.0
            terminal = False
        return {
            "condition": condition,
            "budget": budget,
            "tie_break_seed": seed,
            "lambda_terminal_risk": 10.0 if condition == objective_safe else None,
            "progress_proxy": progress,
            "terminal_adjusted_progress": 0.0 if terminal else progress,
            "final_levels_completed": 0,
            "terminal_state_after_rollout": terminal,
            "steps_survived": budget,
            "actor_relation_delta_count": 4,
            "action_effect_usefulness": 4,
            "stale_action_rate": 0.0,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
        }

    payload = run_objective_aware_abstract_policy_rollout(
        objective_adapter_path=adapter_path,
        budgets=(8,),
        tie_break_seeds=(0,),
        condition_executor=fake_executor,
    )

    assert payload["summary"]["rollout_runs"] == 6
    assert payload["summary"]["best_objective_aware_condition"] == objective_safe
    assert payload["summary"]["objective_aware_terminal_rate_reduced_vs_p3g0"] is True
    assert payload["support"] == 0
    assert payload["policy_result_counted_as_scientific_verdict"] is False


def test_objective_consolidation_marks_objective_useful(tmp_path: Path) -> None:
    objective_safe = objective_condition_name(10.0)
    rollout = {
        "summary": {
            "condition_aggregates": {
                ABSTRACT_MECHANIC_POLICY: {
                    "condition": ABSTRACT_MECHANIC_POLICY,
                    "mean_progress_proxy": 100.0,
                    "mean_terminal_adjusted_progress": 0.0,
                    "terminal_rate": 1.0,
                    "mean_levels_completed": 0.0,
                },
                objective_safe: {
                    "condition": objective_safe,
                    "lambda_terminal_risk": 10.0,
                    "mean_progress_proxy": 100.0,
                    "mean_terminal_adjusted_progress": 100.0,
                    "terminal_rate": 0.0,
                    "mean_levels_completed": 0.0,
                },
            },
            "support": 0,
            "policy_result_counted_as_scientific_verdict": False,
            "candidate_model_counted_as_confirmed_mechanic": False,
        },
        "support": 0,
        "policy_result_counted_as_scientific_verdict": False,
        "candidate_model_counted_as_confirmed_mechanic": False,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }
    rollout_path = _write_json(tmp_path / "rollout.json", rollout)

    payload = consolidate_objective_aware_abstract_policy_utility(
        rollout_path=rollout_path
    )

    assert payload["summary"]["policy_utility_status"] == POLICY_OBJECTIVE_USEFUL_CANDIDATE_ONLY
    assert payload["summary"]["terminal_rate_reduced_vs_p3g0"] is True
    assert payload["summary"]["policy_result_counted_as_scientific_verdict"] is False
    assert payload["support"] == 0
    assert payload["a32_write_performed"] is False
    assert payload["a33_write_performed"] is False


@pytest.mark.parametrize(
    ("hud_support", "hud_verdict"),
    [
        (1, False),
        (0, True),
    ],
)
def test_objective_adapter_rejects_bad_hud_source(
    tmp_path: Path,
    hud_support: int,
    hud_verdict: bool,
) -> None:
    symbolic_path = _write_json(tmp_path / "symbolic.json", _symbolic_model_payload())
    abstract_payload = build_abstract_mechanic_policy_adapter(symbolic_model_path=symbolic_path)
    abstract_path = _write_json(tmp_path / "abstract_adapter.json", abstract_payload)
    hud_path = _write_json(
        tmp_path / "hud.json",
        _hud_payload(support=hud_support, verdict=hud_verdict),
    )

    with pytest.raises(ValueError):
        build_objective_aware_abstract_policy_adapter(
            abstract_adapter_path=abstract_path,
            hud_policy_probe_path=hud_path,
            symbolic_model_path=symbolic_path,
        )
