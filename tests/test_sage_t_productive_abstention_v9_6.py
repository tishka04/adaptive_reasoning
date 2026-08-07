from __future__ import annotations

import json
from types import SimpleNamespace

from theory.sage_t import paired_active_gate_v9_4 as active
from theory.sage_t import productive_abstention_v9_6 as abstention


def test_manifest_binds_failed_validation_and_keeps_firewall_closed() -> None:
    manifest = abstention.load_manifest()
    validation_report = json.loads(
        abstention.DEFAULT_T9_5_REPORT.read_text(encoding="utf-8")
    )

    assert validation_report["status"] == "T9_5_FAILED_CLOSED"
    assert (
        manifest["parent_t9_5_report_checksum"]
        == validation_report["report_checksum"]
    )
    assert manifest["experimental_authority"][
        "maximum_unproductive_interventions_per_branch"
    ] == 5
    assert manifest["firewall"]["source_train_only"] is True
    assert manifest["firewall"]["source_validation_opened"] is False
    assert manifest["firewall"]["holdout_opened"] is False


def test_applied_override_consumes_branch_budget(monkeypatch) -> None:
    controller = object.__new__(abstention.ProductiveAbstainingController)
    controller.maximum_unproductive_interventions_per_branch = 5
    controller._productive_budget_used = 0
    controller._productive_abstentions = 0

    monkeypatch.setattr(
        active.SafeActiveController,
        "decide",
        lambda self, **kwargs: SimpleNamespace(applied=True),
        raising=False,
    )

    result = controller.decide(symbolic_action_name="ACTION1")

    assert result.applied is True
    assert controller._productive_budget_used == 1
    assert controller._productive_abstentions == 0


def test_budgeted_controller_yields_before_pipeline_execution() -> None:
    controller = abstention.build_controller(abstention.load_manifest())
    controller._productive_budget_used = 5

    result = controller.decide(
        symbolic_action_name="action2",
        symbolic_action_data={"x": 4, "y": 7},
        observation=object(),
        legal_actions=(),
    )

    assert result.applied is False
    assert result.action_name == "ACTION2"
    assert result.action_data == {"x": 4, "y": 7}
    assert result.reason == "active_unproductive_abstention"
    assert controller.summary()["productive_abstention"]["abstentions"] == 1
    assert len(controller.decision_latencies_ms) == 1


def test_branch_start_restores_authority_budget() -> None:
    controller = abstention.build_controller(abstention.load_manifest())
    controller._productive_budget_used = 5

    controller.start_branch(regime_index=3)

    summary = controller.summary()["productive_abstention"]
    assert summary["budget_used"] == 0
    assert summary["budget_resets"] == 1
    assert controller._regime_index == 3


def test_gate_is_paired_to_exact_t9_4_source_train_protocol() -> None:
    manifest = abstention.load_manifest()
    parent = active.load_manifest()

    assert manifest["source_train_games"] == parent["source_train_games"]
    assert manifest["seeds"] == parent["seeds"]
    assert manifest["resets"] == parent["resets"]
    assert manifest["action_budget_per_reset"] == parent["action_budget_per_reset"]
    assert manifest["reference_t9_4"]["active_levels_completed"] == 3
    assert manifest["reference_t9_4"]["interventions"] == 114
    assert manifest["gate"]["minimum_intervention_reduction_fraction"] == 0.5
