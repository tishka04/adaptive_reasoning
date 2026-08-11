from __future__ import annotations

from pathlib import Path

import pytest

from theory.sage_t import t10_3_12b_protocol as protocol
from theory.sage_t import t10_3_12b_runtime as runtime
from theory.sage_t.factorial_invariants_v10_3_12b import (
    materialize_variant,
    signed,
    variant_recipes,
)


def _manifest() -> dict:
    return {
        "manifest_checksum": "synthetic-t10-3-12b",
        "matrix": {
            "physical_actions": 0,
            "variants": 128,
            "maximum_artifact_bytes": 5 * 1024 * 1024,
            "maximum_virtual_actions": 32_768,
            "maximum_wall_seconds": 600,
        },
        "gates": {
            "preflight_cases": 12,
            "full_source_correct": 128,
            "generic_correct": 128,
            "ambiguity_correct": 32,
            "source_role_decoupled_correct": 32,
            "minimum_distinct_state_hashes_per_context": 48,
            "minimum_factor_gap_per_context": 8,
            "minimum_factor_gap_per_challenge_context": 4,
            "maximum_source_to_generic_action_ratio": 0.80,
            "minimum_first_decision_divergence": 96,
            "program_hashes_per_arm_context": 1,
        },
        "firewall": {
            "sequence_games_opened": False,
            "holdout_opened": False,
            "production_authority": False,
        },
    }


def _write_parent_gate(root: Path) -> None:
    payload = signed(
        {
            "format_version": "synthetic-parent-audit",
            "manifest_checksum": "synthetic-t10-3-12b",
            "passed": True,
        },
        "audit_checksum",
    )
    protocol.write_json_once(root / runtime.PARENT_AUDIT_FILENAME, payload)


def test_preflight_and_variant_qa_are_zero_action(tmp_path, monkeypatch) -> None:
    output = tmp_path / "out"
    monkeypatch.setattr(protocol, "DEFAULT_OUTPUT_DIR", output)
    _write_parent_gate(output)
    preflight = runtime.preflight(tmp_path, _manifest())
    inventory = runtime.materialize_variants(tmp_path, _manifest())
    assert preflight["passed"] is True
    assert len(preflight["cases"]) == 12
    assert inventory["passed"] is True
    assert len(inventory["variants"]) == 128
    assert inventory["label_counts"] == {"execute_and_stop": 96, "abstain": 32}
    assert inventory["physical_actions"] == 0


def test_compile_factors_imports_no_parent_support(tmp_path, monkeypatch) -> None:
    repo_root = Path.cwd()
    output = tmp_path / "out"
    monkeypatch.setattr(protocol, "DEFAULT_OUTPUT_DIR", output)
    _write_parent_gate(output)
    runtime.preflight(repo_root, _manifest())
    runtime.materialize_variants(repo_root, _manifest())
    registry = runtime.compile_factors(repo_root, _manifest())
    assert registry["local_support_total"] == 0
    assert registry["promotion_count"] == 0
    assert registry["parent_support_imported"] is False
    assert registry["grounded_arguments_compiled"] is False


def test_intervention_aggregator_runs_on_unregistered_micro_sample(
    tmp_path, monkeypatch
) -> None:
    repo_root = Path.cwd()
    output = tmp_path / "out"
    monkeypatch.setattr(protocol, "DEFAULT_OUTPUT_DIR", output)
    recipes = [
        next(
            row
            for row in variant_recipes()
            if row["context"] == context
            and row["challenge"] == "long_positive"
            and row["transform"] == "rotate_90"
            and row["order"] == "canonical"
        )
        for context in ("repeat_context", "path_context")
    ]
    worlds = [materialize_variant(row) for row in recipes]
    inventory = signed(
        {
            "format_version": "synthetic-micro-inventory",
            "manifest_checksum": "synthetic-t10-3-12b",
            "passed": True,
            "distinct_state_hashes": {
                "repeat_context": 1,
                "path_context": 1,
            },
            "variants": [
                {
                    "recipe": recipe,
                    "state_hash": world.state_hash,
                    "expected_abstain": False,
                    "expected_steps": world.expected_steps,
                    "candidate_count": world.candidate_count,
                }
                for recipe, world in zip(recipes, worlds, strict=True)
            ],
            "physical_actions": 0,
        },
        "inventory_checksum",
    )
    protocol.write_json_once(output / runtime.VARIANT_INVENTORY_FILENAME, inventory)
    manifest = _manifest()
    manifest["gates"] = {
        **manifest["gates"],
        "full_source_correct": 2,
        "generic_correct": 2,
        "ambiguity_correct": 0,
        "source_role_decoupled_correct": 0,
        "minimum_distinct_state_hashes_per_context": 1,
        "minimum_factor_gap_per_context": 0,
        "minimum_factor_gap_per_challenge_context": 0,
        "maximum_source_to_generic_action_ratio": 1.0,
        "minimum_first_decision_divergence": 2,
    }
    runtime.compile_factors(repo_root, manifest)
    report = runtime.evaluate_interventions(repo_root, manifest)
    assert report["passed"] is True
    assert len(report["trials"]) == 2 * 6
    assert report["physical_actions"] == 0


def test_adjudication_never_claims_cross_game_generalization(tmp_path, monkeypatch) -> None:
    output = tmp_path / "out"
    monkeypatch.setattr(protocol, "DEFAULT_OUTPUT_DIR", output)
    report = signed(
        {
            "format_version": "synthetic-interventions",
            "manifest_checksum": "synthetic-t10-3-12b",
            "passed": True,
            "verdict": "PASS_T10_3_12B_FACTOR_INTERVENTIONS",
            "checks": {f"{factor}_identified": True for factor in ("operator", "role_binding", "transition", "termination")},
            "metrics": {"virtual_actions": 100},
            "physical_actions": 0,
        },
        "report_checksum",
    )
    protocol.write_json_once(output / runtime.INTERVENTION_REPORT_FILENAME, report)
    result = runtime.adjudicate(tmp_path, _manifest())
    assert result["passed"] is True
    assert result["cross_game_generalization_proven"] is False
    assert result["sequence_composition_authorized"] is False
    assert result["program_promoted"] is False
    assert result["cross_game_preregistration_authorized"] is True


def test_terminal_report_is_incomplete_before_adjudication(tmp_path, monkeypatch) -> None:
    output = tmp_path / "out"
    monkeypatch.setattr(protocol, "DEFAULT_OUTPUT_DIR", output)
    result = runtime.terminal_report(tmp_path, _manifest())
    assert result["verdict"] == "INCOMPLETE_T10_3_12B"
    assert result["cross_game_generalization_proven"] is False
    assert result["accounting"]["physical_actions_authorized"] == 0


def test_early_scientific_miss_can_be_adjudicated_without_integrity_relabel(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "out"
    monkeypatch.setattr(protocol, "DEFAULT_OUTPUT_DIR", output)
    miss = signed(
        {
            "format_version": "synthetic-preflight",
            "manifest_checksum": "synthetic-t10-3-12b",
            "passed": False,
            "verdict": "FACTOR_PREFLIGHT_MISS",
            "physical_actions": 0,
        },
        "preflight_checksum",
    )
    protocol.write_json_once(output / runtime.PREFLIGHT_FILENAME, miss)
    with pytest.raises(protocol.ScientificGateMiss, match="FACTOR_PREFLIGHT_MISS"):
        runtime.adjudicate(tmp_path, _manifest())
    result = runtime._read_signed(  # noqa: SLF001 - verify the sealed public artifact
        tmp_path, runtime.ADJUDICATION_FILENAME, "report_checksum"
    )
    assert result["passed"] is False
    assert result["verdict"] == "FACTOR_PREFLIGHT_MISS"
    assert result["evidence_phase"] == "preflight"


def test_role_proxy_rejection_preserves_other_candidate_evidence(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "out"
    monkeypatch.setattr(protocol, "DEFAULT_OUTPUT_DIR", output)
    report = signed(
        {
            "format_version": "synthetic-interventions",
            "manifest_checksum": "synthetic-t10-3-12b",
            "passed": False,
            "verdict": "SOURCE_ROLE_NOT_TRANSPORTABLE",
            "checks": {
                "operator_identified": True,
                "role_binding_identified": False,
                "transition_identified": True,
                "termination_identified": True,
                "source_role_transportable": False,
                "generic_exact": True,
            },
            "metrics": {"virtual_actions": 100},
            "physical_actions": 0,
        },
        "report_checksum",
    )
    protocol.write_json_once(output / runtime.INTERVENTION_REPORT_FILENAME, report)
    with pytest.raises(protocol.ScientificGateMiss, match="SOURCE_ROLE_NOT_TRANSPORTABLE"):
        runtime.adjudicate(tmp_path, _manifest())
    result = runtime._read_signed(  # noqa: SLF001 - verify the sealed public artifact
        tmp_path, runtime.ADJUDICATION_FILENAME, "report_checksum"
    )
    assert result["identified_factor_candidates"] == [
        "operator",
        "transition",
        "termination",
    ]
    assert result["generic_rebinding_required"] is True
    assert result["alternative_candidate_mechanisms"] == [
        "interventional_rebinding"
    ]
    assert result["evidence_grades"]["role_binding"] == (
        "rejected_by_structural_causal_decoupling"
    )
    assert result["evidence_grades"]["termination"] == (
        "counterfactual_only_no_parent_active_ablation"
    )
