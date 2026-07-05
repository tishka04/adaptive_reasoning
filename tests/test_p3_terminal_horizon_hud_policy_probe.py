import theory.p3.terminal_horizon_hud_policy_probe as hud_probe
import theory.p3.terminal_horizon_policy_probe as base_probe


def _payload():
    return {
        "config": {"schema_version": "p3.terminal_horizon_policy_probe.v1"},
        "terminal_horizon_estimator": {
            "source": "empirical_fallback",
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_P3_AGENT_PROBE",
        },
        "run_results": [
            {
                "candidate": {
                    "condition": base_probe.STOP_AT_HORIZON_POLICY,
                    "terminal_horizon_source": "hud_bar",
                    "terminal_horizon_triggered": True,
                    "horizon_trigger_log": {"source": "hud_bar"},
                }
            },
            {
                "candidate": {
                    "condition": base_probe.OBJECTIVE_MODE_POLICY,
                    "terminal_horizon_source": "hud_bar",
                    "terminal_horizon_triggered": True,
                    "horizon_trigger_log": {"source": "hud_bar"},
                }
            },
        ],
        "comparisons": [
            {
                "terminal_avoidance_signal": True,
                "objective_completion_signal": False,
                "terminal_avoidance_only": True,
            },
            {
                "terminal_avoidance_signal": True,
                "objective_completion_signal": False,
                "terminal_avoidance_only": True,
            },
        ],
        "summary": {
            "support": 0,
            "revision_status": "CANDIDATE_ONLY",
            "truth_status": "NOT_EVALUATED_BY_P3_AGENT_PROBE",
        },
        "support": 0,
        "revision_status": "CANDIDATE_ONLY",
        "truth_status": "NOT_EVALUATED_BY_P3_AGENT_PROBE",
    }


def test_summarize_hud_policy_sources_counts_hud_trigger_runs():
    summary = hud_probe.summarize_hud_policy_sources(_payload())

    assert summary["candidate_runs"] == 2
    assert summary["candidate_hud_bar_source_runs"] == 2
    assert summary["hud_bar_trigger_source_runs"] == 2
    assert summary["terminal_avoidance_signal_runs"] == 2
    assert summary["objective_completion_signal_runs"] == 0
    assert summary["action6_prefix_count_used_as_decision_variable"] is False
    assert summary["support"] == 0

