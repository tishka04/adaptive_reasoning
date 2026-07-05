import json
from pathlib import Path

import pytest

from theory.p3.abstract_mechanic_policy_probe import (
    ABSTRACT_MECHANIC_POLICY,
    GREEDY_CHANGED_PIXELS_POLICY,
    POLICY_USEFUL_CANDIDATE_ONLY,
    RANDOM_AVAILABLE_POLICY,
    TERMINAL_HORIZON_GUARD_POLICY,
    build_abstract_mechanic_policy_adapter,
    consolidate_abstract_mechanic_policy_utility,
    run_abstract_mechanic_policy_rollout,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _symbolic_model_payload(*, support: int = 0, confirmation: bool = False) -> dict:
    return {
        "summary": {
            "ready_for_policy_probe_candidate_only": True,
            "support": support,
            "model_status": "CANDIDATE_ONLY",
            "a32_write_performed": False,
            "a33_write_performed": False,
        },
        "candidate_symbolic_model": {
            "candidate_symbolic_model_id": "m3g0_7::test::symbolic_model",
            "model_status": "CANDIDATE_ONLY",
            "actor_candidates": [
                {
                    "entity_id": "E_actor",
                    "role": "controllable_actor_candidate",
                    "basis": "CONTEXT_STABLE_CANDIDATE_ONLY",
                    "support": 0,
                }
            ],
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
            "dynamic_invariants": {
                "E_drift": {
                    "family": "exogenous_motion",
                    "semantic_interpretation": "unknown",
                    "support": 0,
                }
            },
            "caveats": [
                {
                    "entity_id": "E_hud",
                    "family": "monotone_counter",
                    "semantic_interpretation": "unknown",
                    "support": 0,
                }
            ],
            "model_counted_as_confirmation": confirmation,
            "support": support,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_M3",
        },
        "model_counted_as_confirmation": confirmation,
        "support": support,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_M3",
        "a32_write_performed": False,
        "a33_write_performed": False,
    }


def test_adapter_builds_candidate_policy_without_confirmation(tmp_path: Path) -> None:
    source = _write_json(tmp_path / "symbolic.json", _symbolic_model_payload())

    payload = build_abstract_mechanic_policy_adapter(symbolic_model_path=source)

    adapter = payload["policy_adapter"]
    assert payload["summary"]["ready_for_abstract_policy_probe"] is True
    assert adapter["actor_candidates"] == ["E_actor"]
    assert [row["action"] for row in adapter["action_candidates"]] == ["ACTION3"]
    assert adapter["relation_targets"] == ["E_target"]
    assert adapter["ignored_or_caveated_entities"][0]["entity_id"] == "E_hud"
    assert adapter["dynamic_invariants_observed_not_semantic"] == ["E_drift"]
    assert adapter["candidate_model_counted_as_confirmed_mechanic"] is False
    assert adapter["policy_adapter_counted_as_confirmation"] is False
    assert payload["support"] == 0
    assert payload["a32_write_performed"] is False
    assert payload["a33_write_performed"] is False


def test_rollout_uses_executor_and_keeps_candidate_only(tmp_path: Path) -> None:
    source = _write_json(tmp_path / "symbolic.json", _symbolic_model_payload())
    adapter_payload = build_abstract_mechanic_policy_adapter(symbolic_model_path=source)
    adapter_path = _write_json(tmp_path / "adapter.json", adapter_payload)

    progress_by_condition = {
        RANDOM_AVAILABLE_POLICY: 1.0,
        GREEDY_CHANGED_PIXELS_POLICY: 2.0,
        TERMINAL_HORIZON_GUARD_POLICY: 1.5,
        ABSTRACT_MECHANIC_POLICY: 3.0,
    }

    def fake_executor(condition, budget, seed, adapter, env_path, game_id):
        return {
            "condition": condition,
            "budget": budget,
            "tie_break_seed": seed,
            "progress_proxy": progress_by_condition[condition],
            "final_levels_completed": 0,
            "terminal_state_after_rollout": False,
            "steps_survived": budget,
            "actor_relation_delta_count": 2 if condition == ABSTRACT_MECHANIC_POLICY else 0,
            "action_effect_usefulness": 2 if condition == ABSTRACT_MECHANIC_POLICY else 0,
            "stale_action_rate": 0.0,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
        }

    payload = run_abstract_mechanic_policy_rollout(
        policy_adapter_path=adapter_path,
        budgets=(8,),
        tie_break_seeds=(0,),
        condition_executor=fake_executor,
    )

    assert payload["summary"]["rollout_runs"] == 4
    assert payload["summary"]["candidate_beats_best_baseline_progress"] is True
    assert payload["summary"]["candidate_beats_best_baseline_relation_metric"] is True
    assert payload["summary"]["policy_result_counted_as_scientific_verdict"] is False
    assert payload["summary"]["candidate_model_counted_as_confirmed_mechanic"] is False
    assert payload["support"] == 0


def test_utility_consolidation_marks_useful_candidate_only(tmp_path: Path) -> None:
    rollout = {
        "summary": {
            "condition_aggregates": {
                RANDOM_AVAILABLE_POLICY: {
                    "mean_progress_proxy": 1.0,
                    "mean_actor_relation_delta_count": 0.0,
                },
                GREEDY_CHANGED_PIXELS_POLICY: {
                    "mean_progress_proxy": 2.0,
                    "mean_actor_relation_delta_count": 0.0,
                },
                TERMINAL_HORIZON_GUARD_POLICY: {
                    "mean_progress_proxy": 1.5,
                    "mean_actor_relation_delta_count": 0.0,
                },
                ABSTRACT_MECHANIC_POLICY: {
                    "mean_progress_proxy": 3.0,
                    "mean_actor_relation_delta_count": 2.0,
                },
            },
            "policy_result_counted_as_scientific_verdict": False,
            "candidate_model_counted_as_confirmed_mechanic": False,
            "support": 0,
        },
        "config": {"policy_adapter_path": "adapter.json"},
        "policy_result_counted_as_scientific_verdict": False,
        "candidate_model_counted_as_confirmed_mechanic": False,
        "support": 0,
        "a32_write_performed": False,
        "a33_write_performed": False,
    }
    rollout_path = _write_json(tmp_path / "rollout.json", rollout)

    payload = consolidate_abstract_mechanic_policy_utility(rollout_path=rollout_path)

    assert payload["summary"]["policy_utility_status"] == POLICY_USEFUL_CANDIDATE_ONLY
    assert payload["summary"]["policy_result_counted_as_scientific_verdict"] is False
    assert payload["summary"]["candidate_model_counted_as_confirmed_mechanic"] is False
    assert payload["support"] == 0
    assert payload["a32_write_performed"] is False
    assert payload["a33_write_performed"] is False


@pytest.mark.parametrize(
    ("support", "confirmation"),
    [
        (1, False),
        (0, True),
    ],
)
def test_adapter_rejects_support_or_confirmation(
    tmp_path: Path,
    support: int,
    confirmation: bool,
) -> None:
    source = _write_json(
        tmp_path / "symbolic.json",
        _symbolic_model_payload(support=support, confirmation=confirmation),
    )

    with pytest.raises(ValueError):
        build_abstract_mechanic_policy_adapter(symbolic_model_path=source)
