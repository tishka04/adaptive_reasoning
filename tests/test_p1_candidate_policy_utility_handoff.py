import theory.p1.bp35_candidate_policy_utility_handoff as handoff


def _probe_payload():
    return {
        "summary": {
            "candidate_policy_probe_ready": True,
            "candidate_policy_counted_as_confirmation": False,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_P1_AGENT_PROBE",
            "wrong_confirmations": 0,
        },
        "comparison": {
            "candidate_policy_better_than_baseline_on_any_axis": True,
            "candidate_progress_proxy_delta": 27.0,
            "candidate_policy_counted_as_confirmation": False,
            "support": 0,
        },
    }


def _matrix_payload():
    return {
        "aggregate": {
            "candidate_beats_baseline_runs": 18,
            "candidate_beats_action4_only_runs": 18,
            "candidate_beats_baseline_ratio": 1.0,
            "candidate_beats_action4_only_ratio": 1.0,
            "candidate_mean_progress_delta_vs_baseline": 57.2222,
            "candidate_mean_failure_like_selection_delta_vs_baseline": -4.4444,
            "robust_candidate_policy_utility_signal": True,
            "patch_similarity_attribution_signal_candidate_only": True,
            "policy_probe_result_is_not_scientific_verdict": True,
            "candidate_policy_counted_as_confirmation": False,
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_P1_AGENT_PROBE",
            "wrong_confirmations": 0,
        }
    }


def _scope_payload():
    return {
        "scope_consolidations": [
            {
                "scope_consolidation_id": "m3_24::bp35::patch_similarity_scope",
                "candidate_rule_family": "local_patch_transformability",
                "candidate_mechanic": (
                    "repositioning_opens_patch_similar_action6_affordances"
                ),
                "ready_for_agent_policy_probe": True,
                "a33_ready": False,
                "support": 0,
                "revision_status": "CANDIDATE_ONLY",
                "truth_status": "NOT_EVALUATED_BY_M3",
            }
        ]
    }


def test_candidate_policy_utility_handoff_is_candidate_only():
    payload = handoff.build_candidate_policy_utility_handoff(
        probe_payload=_probe_payload(),
        matrix_payload=_matrix_payload(),
        scope_payload=_scope_payload(),
    )

    assert payload["handoff_type"] == "AGENTIC_UTILITY_CANDIDATE_ONLY"
    assert payload["agentic_utility_status"] == "SUPPORTED_CANDIDATE_ONLY"
    assert payload["candidate_beats_baseline_runs"] == 18
    assert payload["candidate_beats_action4_only_runs"] == 18
    assert payload["patch_similarity_attribution_signal_candidate_only"] is True
    assert payload["policy_result_counted_as_mechanistic_confirmation"] is False
    assert payload["a33_ready"] is False
    assert payload["support"] == 0
    assert payload["revision_status"] == "CANDIDATE_ONLY"
    assert payload["truth_status"] == "NOT_EVALUATED_BY_P1_AGENT_PROBE"
    assert payload["summary"]["handoffs_produced"] == 1
    item = payload["candidate_policy_utility_handoffs"][0]
    assert item["ready_for_a32_utility_review"] is True
    assert item["ready_for_a32_utility_review_is_not_verdict"] is True
    assert item["support"] == 0


def test_candidate_policy_utility_handoff_rejects_nonzero_support():
    matrix = _matrix_payload()
    matrix["aggregate"]["support"] = 1

    payload = handoff.build_candidate_policy_utility_handoff(
        probe_payload=_probe_payload(),
        matrix_payload=matrix,
        scope_payload=_scope_payload(),
    )

    assert payload["agentic_utility_status"] == "BLOCKED_INVALID_INPUT"
    assert payload["summary"]["handoffs_produced"] == 0
    assert payload["summary"]["handoffs_rejected"] == 1
    rejected = payload["rejected_handoff_candidates"][0]
    assert "matrix_support_nonzero" in rejected["blocking_reasons"]
    assert rejected["policy_result_counted_as_mechanistic_confirmation"] is False
    assert rejected["support"] == 0
