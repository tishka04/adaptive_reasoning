"""Fail-closed runtime for SAGE.T10.3.12b factor identification."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import t10_3_12b_protocol as protocol
from .factorial_invariants_v10_3_12b import (
    ARMS,
    CONTEXTS,
    FACTORS,
    FactorProgram,
    FactorRegistry,
    assert_transfer_safe,
    compile_factor_registry,
    evaluate_trial,
    materialize_variant,
    median,
    sha256_payload,
    signed,
    variant_recipes,
)

PARENT_AUDIT_FILENAME = "parent_generic_rediscovery_audit.json"
PREFLIGHT_FILENAME = "factor_preflight.json"
VARIANT_INVENTORY_FILENAME = "counterfactual_variant_inventory.json"
FACTOR_REGISTRY_FILENAME = "factor_registry.json"
INTERVENTION_REPORT_FILENAME = "factorial_intervention_report.json"
ADJUDICATION_FILENAME = "factor_adjudication_report.json"
TERMINAL_REPORT_FILENAME = "terminal_report.json"

FACTOR_ARM = {
    "operator": "operator_ablation",
    "role_binding": "role_binding_ablation",
    "transition": "transition_ablation",
    "termination": "termination_ablation",
}


def _destination(root: Path) -> Path:
    return root.resolve() / protocol.DEFAULT_OUTPUT_DIR


def _path(root: Path, filename: str) -> Path:
    return _destination(root) / filename


def _write(root: Path, filename: str, payload: Mapping[str, Any]) -> None:
    protocol.write_json_once(_path(root, filename), payload)


def _read_signed(root: Path, filename: str, checksum_field: str) -> dict[str, Any]:
    path = _path(root, filename)
    if not path.is_file():
        raise protocol.IntegrityError(f"required T10.3.12b artifact is absent: {filename}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol.verify_signed(payload, checksum_field)
    return payload


def _require_gate(root: Path, phase: str) -> dict[str, Any]:
    contract = protocol.ARTIFACT_CONTRACT[phase]
    payload = _read_signed(root, str(contract["path"]), str(contract["checksum_field"]))
    gate = contract.get("gate_field")
    if gate and not bool(payload.get(str(gate))):
        raise protocol.ScientificGateMiss(str(payload.get("verdict", f"{phase.upper()}_MISS")))
    return payload


def _artifact_bytes(root: Path) -> int:
    destination = _destination(root)
    return sum(path.stat().st_size for path in destination.glob("*.json") if path.is_file())


def _parent_json(root: Path, name: str, checksum_field: str) -> dict[str, Any]:
    relative = protocol.PARENT_ARTIFACT_PATHS[name]
    payload = json.loads((root / relative).read_text(encoding="utf-8"))
    protocol.verify_signed(payload, checksum_field)
    return payload


def _parent_sequence_diagnostic(root: Path) -> dict[str, Any]:
    journal = root / protocol.PARENT_OUTPUT_DIR / "journal"
    rows: list[dict[str, Any]] = []
    for path in sorted((journal / "branches").glob("*/receipt.json")):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        protocol.verify_signed(receipt, "receipt_checksum")
        arm = str(receipt.get("arm", ""))
        if arm not in {"factorized_relational_source", "generic_grammar_source_free"}:
            continue
        work_id = str(receipt["work_id"])
        sequence = []
        for intent_path in sorted((journal / "intents" / work_id).glob("*.json")):
            intent = json.loads(intent_path.read_text(encoding="utf-8"))
            protocol.verify_signed(intent, "intent_checksum")
            action = intent.get("action", {})
            sequence.append(
                {
                    "name": str(action.get("name", "")),
                    "argument_checksum": str(action.get("argument_checksum", "")),
                }
            )
        rows.append(
            {
                "arm": arm,
                "game_id": str(receipt.get("game_id", "")),
                "sequence_hash": sha256_payload(sequence),
                "actions": len(sequence),
                "all_sage_t": int(receipt.get("sage_t_option_actions", 0)) == len(sequence),
            }
        )
    games = sorted({row["game_id"] for row in rows})
    by_game: dict[str, Any] = {}
    for game in games:
        source = sorted(
            {row["sequence_hash"] for row in rows if row["game_id"] == game and row["arm"] == "factorized_relational_source"}
        )
        generic = sorted(
            {row["sequence_hash"] for row in rows if row["game_id"] == game and row["arm"] == "generic_grammar_source_free"}
        )
        by_game[game] = {
            "source_unique_sequence_hashes": source,
            "generic_unique_sequence_hashes": generic,
            "exact_sequence_set_match": source == generic and len(source) == 1,
        }
    return {
        "work_rows": len(rows),
        "by_game": by_game,
        "all_source_and_generic_actions_sage_t": all(row["all_sage_t"] for row in rows),
        "all_games_exact_sequence_match": bool(by_game) and all(
            row["exact_sequence_set_match"] for row in by_game.values()
        ),
    }


def audit_parent(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    terminal = _parent_json(root, "terminal", "report_checksum")
    active = _parent_json(root, "active_report", "report_checksum")
    adjudication = _parent_json(root, "adjudication", "report_checksum")
    sequence = _parent_sequence_diagnostic(root)
    metrics = active.get("metrics", {})
    successes = metrics.get("successes", {})
    actions = metrics.get("actions_total_to_level", {})
    source_success = successes.get("factorized_relational_source", {})
    generic_success = successes.get("generic_grammar_source_free", {})
    checks = {
        "parent_verdict_is_generic_rediscovery": terminal.get("verdict") == "GENERIC_REDISCOVERY_ONLY",
        "parent_accounting_clean": bool(terminal.get("accounting", {}).get("equation_holds"))
        and bool(terminal.get("accounting", {}).get("inflight_valid")),
        "source_generic_success_tie": source_success == generic_success,
        "source_generic_action_tie": actions.get("factorized_relational_source")
        == actions.get("generic_grammar_source_free"),
        "source_generic_exact_trajectory_match": bool(sequence["all_games_exact_sequence_match"]),
        "source_generic_actions_all_sage_t": bool(sequence["all_source_and_generic_actions_sage_t"]),
        "control_fallback_contamination_recorded": not bool(
            adjudication.get("checks", {}).get("all_actions_sage_t", True)
        ),
        "parent_not_used_for_training": True,
        "parent_registry_support_not_imported": True,
        "sequence_games_closed": not bool(terminal.get("sequence_games_opened")),
        "production_authority_closed": not bool(terminal.get("production_authority")),
        "parent_journal_immutable": protocol.parent_journal_digest(root)
        == manifest["parent_journal_digest"],
    }
    passed = all(checks.values())
    payload = signed(
        {
            "format_version": "sage-t10.3.12b-parent-audit-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "passed": passed,
            "verdict": "PASS_T10_3_12B_PARENT_AUDIT" if passed else "INVALID_PARENT_PROVENANCE",
            "checks": checks,
            "parent_terminal_checksum": terminal["report_checksum"],
            "parent_active_checksum": active["report_checksum"],
            "parent_adjudication_checksum": adjudication["report_checksum"],
            "sequence_diagnostic": sequence,
            "parent_actions_used_for_training": False,
            "parent_registry_loaded_for_support": False,
            "physical_actions": 0,
        },
        "audit_checksum",
    )
    _write(root, PARENT_AUDIT_FILENAME, payload)
    if not passed:
        raise protocol.IntegrityError("T10.3.12 parent audit failed")
    return payload


def preflight(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require_gate(root, "audit-parent")
    recipes = variant_recipes()
    sample = FactorProgram(
        context="repeat_context",
        arm="factorized_source",
        operator="parameterized_apply",
        role_binding="boundary_prior_with_causal_verification",
        transition="same_role_until_progress",
        termination="stop_on_progress_or_ambiguity",
        safety_horizon=8,
        source_kind="factorized_projection",
    )
    round_trip = FactorProgram.from_payload(sample.safe_payload)
    forbidden_rejected = False
    try:
        assert_transfer_safe({"action_data": {"x": 4, "y": 29}})
    except ValueError:
        forbidden_rejected = True
    cases = [
        {"name": "recipe_count", "passed": len(recipes) == 128},
        {"name": "recipe_identity_unique", "passed": len({row["variant_id"] for row in recipes}) == 128},
        {"name": "identification_split", "passed": sum(row["split"] == "identification" for row in recipes) == 64},
        {"name": "challenge_split", "passed": sum(row["split"] == "challenge" for row in recipes) == 64},
        {"name": "repeat_balance", "passed": sum(row["context"] == "repeat_context" for row in recipes) == 64},
        {"name": "path_balance", "passed": sum(row["context"] == "path_context" for row in recipes) == 64},
        {"name": "ambiguity_not_universal", "passed": sum(row["challenge"] == "ambiguous_role" for row in recipes) == 32},
        {"name": "program_round_trip", "passed": round_trip.program_hash == sample.program_hash},
        {"name": "transfer_payload_safe", "passed": sample.safe_payload == round_trip.safe_payload},
        {"name": "grounded_payload_rejected", "passed": forbidden_rejected},
        {"name": "no_physical_phase", "passed": int(manifest["matrix"]["physical_actions"]) == 0},
        {"name": "sequence_and_authority_closed", "passed": not any(
            bool(manifest["firewall"][key])
            for key in ("sequence_games_opened", "holdout_opened", "production_authority")
        )},
    ]
    passed = len(cases) == int(manifest["gates"]["preflight_cases"]) and all(row["passed"] for row in cases)
    payload = signed(
        {
            "format_version": "sage-t10.3.12b-factor-preflight-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "passed": passed,
            "verdict": "PASS_T10_3_12B_PREFLIGHT" if passed else "FACTOR_PREFLIGHT_MISS",
            "cases": cases,
            "physical_actions": 0,
        },
        "preflight_checksum",
    )
    _write(root, PREFLIGHT_FILENAME, payload)
    if not passed:
        raise protocol.ScientificGateMiss("FACTOR_PREFLIGHT_MISS")
    return payload


def materialize_variants(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require_gate(root, "preflight")
    rows = []
    for recipe_payload in variant_recipes():
        world = materialize_variant(recipe_payload)
        rows.append(
            {
                "recipe": recipe_payload,
                "state_hash": world.state_hash,
                "expected_abstain": world.expected_abstain,
                "expected_steps": world.expected_steps,
                "candidate_count": world.candidate_count,
            }
        )
    distinct = {
        context: len({row["state_hash"] for row in rows if row["recipe"]["context"] == context})
        for context in CONTEXTS
    }
    label_counts = {
        "execute_and_stop": sum(not row["expected_abstain"] for row in rows),
        "abstain": sum(row["expected_abstain"] for row in rows),
    }
    checks = {
        "variant_count": len(rows) == int(manifest["matrix"]["variants"]),
        "state_diversity": all(
            count >= int(manifest["gates"]["minimum_distinct_state_hashes_per_context"])
            for count in distinct.values()
        ),
        "non_universal_expected_label": set(label_counts.values()) == {32, 96},
        "compact_no_raw_grid": all("grid" not in json.dumps(row).lower() for row in rows),
        "physical_actions_zero": True,
    }
    passed = all(checks.values())
    payload = signed(
        {
            "format_version": "sage-t10.3.12b-variant-inventory-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "passed": passed,
            "verdict": "PASS_T10_3_12B_VARIANT_QA" if passed else "COUNTERFACTUAL_DIVERSITY_MISS",
            "checks": checks,
            "distinct_state_hashes": distinct,
            "label_counts": label_counts,
            "variants": rows,
            "physical_actions": 0,
        },
        "inventory_checksum",
    )
    _write(root, VARIANT_INVENTORY_FILENAME, payload)
    if not passed:
        raise protocol.ScientificGateMiss("COUNTERFACTUAL_DIVERSITY_MISS")
    if _artifact_bytes(root) > int(manifest["matrix"]["maximum_artifact_bytes"]):
        raise protocol.IntegrityError("T10.3.12b compact artifact budget exceeded")
    return payload


def compile_factors(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    _require_gate(root, "materialize-variants")
    parent_path = root / protocol.PARENT_ARTIFACT_PATHS["candidate_registry"]
    parent_payload = json.loads(parent_path.read_text(encoding="utf-8"))
    protocol.verify_signed(parent_payload, "registry_checksum")
    registry = compile_factor_registry(parent_payload)
    snapshot = registry.snapshot()
    checks = {
        "program_count": len(snapshot["programs"]) == len(ARMS) * len(CONTEXTS),
        "support_zero": int(snapshot["local_support_total"]) == 0,
        "promotion_zero": int(snapshot["promotion_count"]) == 0,
        "parent_registry_pinned": snapshot["parent_registry_checksum"] == parent_payload["registry_checksum"],
    }
    if not all(checks.values()):
        raise protocol.IntegrityError("T10.3.12b factor compilation contract failed")
    payload = signed(
        {
            **snapshot,
            "manifest_checksum": manifest["manifest_checksum"],
            "checks": checks,
            "grounded_arguments_compiled": False,
            "parent_support_imported": False,
            "physical_actions": 0,
        },
        "registry_checksum",
    )
    _write(root, FACTOR_REGISTRY_FILENAME, payload)
    return payload


def _intervention_verdict(checks: Mapping[str, bool]) -> str:
    if not checks.get("source_role_transportable"):
        return "SOURCE_ROLE_NOT_TRANSPORTABLE"
    if not checks.get("full_source_exact"):
        return "FACTORIZED_SOURCE_INVARIANCE_MISS"
    if not checks.get("generic_exact"):
        return "GENERIC_BASELINE_INVALID"
    if not checks.get("ambiguity_exact"):
        return "AMBIGUITY_CONTROL_MISS"
    if not checks.get("source_value"):
        return "GENERIC_PRIOR_ONLY"
    for factor in FACTORS:
        if not checks.get(f"{factor}_identified"):
            return f"{factor.upper()}_FACTOR_NOT_IDENTIFIED"
    if not checks.get("counterfactual_state_diversity"):
        return "COUNTERFACTUAL_DIVERSITY_MISS"
    if not checks.get("program_hash_stability"):
        return "PROGRAM_HASH_INVARIANCE_MISS"
    if not checks.get("virtual_action_budget") or not checks.get("wall_budget"):
        return "COUNTERFACTUAL_BUDGET_MISS"
    if not checks.get("physical_actions_zero"):
        return "INVALID_PHYSICAL_ACTION_ACCOUNTING"
    return "PASS_T10_3_12B_FACTOR_INTERVENTIONS"


def evaluate_interventions(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    inventory = _require_gate(root, "materialize-variants")
    registry_payload = _read_signed(root, FACTOR_REGISTRY_FILENAME, "registry_checksum")
    registry = FactorRegistry(registry_payload)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for item in inventory["variants"]:
        world = materialize_variant(item["recipe"])
        for arm in ARMS:
            program = registry.program_for(arm, world.recipe.context)
            outcome = evaluate_trial(program, world)
            rows.append(
                {
                    "variant_id": world.recipe.variant_id,
                    "state_hash": world.state_hash,
                    "context": world.recipe.context,
                    "split": world.recipe.split,
                    "challenge": world.recipe.challenge,
                    "arm": arm,
                    "program_hash": program.program_hash,
                    **outcome.compact(),
                }
            )
    elapsed = time.perf_counter() - started
    virtual_actions = sum(int(row["virtual_actions"]) for row in rows)

    def selected(arm: str, *, context: str | None = None, split: str | None = None) -> list[dict[str, Any]]:
        return [
            row
            for row in rows
            if row["arm"] == arm
            and (context is None or row["context"] == context)
            and (split is None or row["split"] == split)
        ]

    correct_by_arm = {arm: sum(bool(row["correct"]) for row in selected(arm)) for arm in ARMS}
    correct_by_context = {
        arm: {
            context: sum(bool(row["correct"]) for row in selected(arm, context=context))
            for context in CONTEXTS
        }
        for arm in ARMS
    }
    correct_challenge = {
        arm: {
            context: sum(bool(row["correct"]) for row in selected(arm, context=context, split="challenge"))
            for context in CONTEXTS
        }
        for arm in ARMS
    }
    invariant_correct_by_context = {
        arm: {
            context: sum(
                bool(row["correct"])
                for row in selected(arm, context=context)
                if row["challenge"] != "relation_decoupled"
            )
            for context in CONTEXTS
        }
        for arm in ARMS
    }
    invariant_challenge_correct = {
        arm: {
            context: sum(
                bool(row["correct"])
                for row in selected(arm, context=context, split="challenge")
                if row["challenge"] != "relation_decoupled"
            )
            for context in CONTEXTS
        }
        for arm in ARMS
    }
    source_generic_efficiency: dict[str, Any] = {}
    for context in CONTEXTS:
        efficiency_challenges = {"short_positive", "long_positive"}
        source_rows = [
            row
            for row in selected(ARMS[0], context=context)
            if row["challenge"] in efficiency_challenges
        ]
        generic_rows = [
            row
            for row in selected(ARMS[1], context=context)
            if row["challenge"] in efficiency_challenges
        ]
        source_median = median([int(row["virtual_actions"]) for row in source_rows])
        generic_median = median([int(row["virtual_actions"]) for row in generic_rows])
        source_generic_efficiency[context] = {
            "source_median_virtual_actions": source_median,
            "generic_median_virtual_actions": generic_median,
            "ratio": source_median / generic_median if generic_median else 1.0,
        }
    source_by_id = {row["variant_id"]: row for row in selected(ARMS[0])}
    generic_by_id = {row["variant_id"]: row for row in selected(ARMS[1])}
    decision_divergence = sum(
        source_by_id[key]["first_decision_class"] != generic_by_id[key]["first_decision_class"]
        for key in source_by_id
        if source_by_id[key]["challenge"] != "ambiguous_role"
    )
    factor_gaps = {
        factor: {
            context: invariant_correct_by_context[ARMS[0]][context]
            - invariant_correct_by_context[arm][context]
            for context in CONTEXTS
        }
        for factor, arm in FACTOR_ARM.items()
    }
    challenge_factor_gaps = {
        factor: {
            context: invariant_challenge_correct[ARMS[0]][context]
            - invariant_challenge_correct[arm][context]
            for context in CONTEXTS
        }
        for factor, arm in FACTOR_ARM.items()
    }
    ambiguity_correct = sum(
        bool(row["correct"])
        for row in selected(ARMS[0])
        if row["challenge"] == "ambiguous_role"
    )
    source_role_decoupled_correct = sum(
        bool(row["correct"])
        for row in selected(ARMS[0])
        if row["challenge"] == "relation_decoupled"
    )
    program_hash_counts = {
        arm: {
            context: len({row["program_hash"] for row in selected(arm, context=context)})
            for context in CONTEXTS
        }
        for arm in ARMS
    }
    checks: dict[str, bool] = {
        "full_source_exact": correct_by_arm[ARMS[0]] == int(manifest["gates"]["full_source_correct"]),
        "generic_exact": correct_by_arm[ARMS[1]] == int(manifest["gates"]["generic_correct"]),
        "ambiguity_exact": ambiguity_correct == int(manifest["gates"]["ambiguity_correct"]),
        "source_role_transportable": source_role_decoupled_correct
        == int(manifest["gates"]["source_role_decoupled_correct"]),
        "source_value": all(
            row["ratio"] <= float(manifest["gates"]["maximum_source_to_generic_action_ratio"])
            for row in source_generic_efficiency.values()
        ) and decision_divergence >= int(manifest["gates"]["minimum_first_decision_divergence"]),
        "counterfactual_state_diversity": all(
            int(value) >= int(manifest["gates"]["minimum_distinct_state_hashes_per_context"])
            for value in inventory["distinct_state_hashes"].values()
        ),
        "program_hash_stability": all(
            count == int(manifest["gates"]["program_hashes_per_arm_context"])
            for by_context in program_hash_counts.values()
            for count in by_context.values()
        ),
        "virtual_action_budget": virtual_actions <= int(manifest["matrix"]["maximum_virtual_actions"]),
        "wall_budget": elapsed <= float(manifest["matrix"]["maximum_wall_seconds"]),
        "physical_actions_zero": True,
    }
    for factor in FACTORS:
        checks[f"{factor}_identified"] = all(
            factor_gaps[factor][context] >= int(manifest["gates"]["minimum_factor_gap_per_context"])
            and challenge_factor_gaps[factor][context]
            >= int(manifest["gates"]["minimum_factor_gap_per_challenge_context"])
            for context in CONTEXTS
        )
    checks["role_binding_identified"] = bool(
        checks["role_binding_identified"] and checks["source_role_transportable"]
    )
    verdict = _intervention_verdict(checks)
    passed = verdict == "PASS_T10_3_12B_FACTOR_INTERVENTIONS" and all(checks.values())
    payload = signed(
        {
            "format_version": "sage-t10.3.12b-factorial-intervention-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "passed": passed,
            "verdict": verdict,
            "checks": checks,
            "metrics": {
                "correct_by_arm": correct_by_arm,
                "correct_by_context": correct_by_context,
                "correct_challenge": correct_challenge,
                "invariant_correct_by_context": invariant_correct_by_context,
                "invariant_challenge_correct": invariant_challenge_correct,
                "factor_gaps": factor_gaps,
                "challenge_factor_gaps": challenge_factor_gaps,
                "source_generic_efficiency": source_generic_efficiency,
                "first_decision_divergence": decision_divergence,
                "ambiguity_correct": ambiguity_correct,
                "source_role_decoupled_correct": source_role_decoupled_correct,
                "program_hash_counts": program_hash_counts,
                "virtual_actions": virtual_actions,
                "elapsed_seconds": elapsed,
            },
            "trials": rows,
            "cross_game_generalization_proven": False,
            "physical_actions": 0,
        },
        "report_checksum",
    )
    _write(root, INTERVENTION_REPORT_FILENAME, payload)
    if _artifact_bytes(root) > int(manifest["matrix"]["maximum_artifact_bytes"]):
        raise protocol.IntegrityError("T10.3.12b compact artifact budget exceeded")
    if not passed:
        raise protocol.ScientificGateMiss(verdict)
    return payload


def adjudicate(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    intervention_path = _path(root, INTERVENTION_REPORT_FILENAME)
    if intervention_path.is_file():
        report = _read_signed(root, INTERVENTION_REPORT_FILENAME, "report_checksum")
        passed = bool(report.get("passed"))
        verdict = (
            "PASS_T10_3_12B_TRANSFERABLE_FACTOR_CANDIDATES_IDENTIFIED"
            if passed
            else str(report.get("verdict", "FACTOR_IDENTIFICATION_MISS"))
        )
        identified = list(FACTORS) if passed else [
            factor
            for factor in FACTORS
            if bool(report.get("checks", {}).get(f"{factor}_identified"))
        ]
        evidence_checksum = report["report_checksum"]
        evidence_phase = "evaluate-interventions"
    else:
        early_misses = (
            (PREFLIGHT_FILENAME, "preflight_checksum", "preflight"),
            (VARIANT_INVENTORY_FILENAME, "inventory_checksum", "materialize-variants"),
        )
        evidence = None
        evidence_phase = ""
        for filename, checksum_field, phase in early_misses:
            path = _path(root, filename)
            if not path.is_file():
                continue
            candidate = _read_signed(root, filename, checksum_field)
            if not bool(candidate.get("passed", True)):
                evidence = candidate
                evidence_phase = phase
        if evidence is None:
            raise protocol.IntegrityError(
                "T10.3.12b adjudication lacks a complete scientific artifact"
            )
        passed = False
        verdict = str(evidence.get("verdict", "FACTOR_IDENTIFICATION_MISS"))
        identified = []
        evidence_checksum = str(
            evidence.get("preflight_checksum") or evidence.get("inventory_checksum")
        )
    evidence_grades = {
        "operator": (
            "counterfactual_plus_parent_schema_swap"
            if "operator" in identified
            else "not_identified"
        ),
        "role_binding": (
            "counterfactual_plus_decoupling"
            if "role_binding" in identified
            else (
                "rejected_by_structural_causal_decoupling"
                if evidence_phase == "evaluate-interventions"
                and not bool(report.get("checks", {}).get("source_role_transportable"))
                else "not_identified"
            )
        ),
        "transition": (
            "counterfactual_plus_confounded_parent_relation_ablation"
            if "transition" in identified
            else "not_identified"
        ),
        "termination": (
            "counterfactual_only_no_parent_active_ablation"
            if "termination" in identified
            else "not_identified"
        ),
    }
    generic_rebinding_required = bool(
        evidence_phase == "evaluate-interventions"
        and report.get("checks", {}).get("generic_exact")
        and not report.get("checks", {}).get("source_role_transportable")
    )
    alternative_candidates = (
        ["interventional_rebinding"] if generic_rebinding_required else []
    )
    if passed:
        next_step = "preregister_t10_3_12c_cross_game_falsification"
    elif generic_rebinding_required:
        next_step = "replace_game_specific_role_proxy_with_interventional_rebinding"
    else:
        next_step = "retain_negative_result_without_retuning"
    payload = signed(
        {
            "format_version": "sage-t10.3.12b-factor-adjudication-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "passed": passed,
            "verdict": verdict,
            "identified_factor_candidates": identified,
            "evidence_grades": evidence_grades,
            "generic_rebinding_required": generic_rebinding_required,
            "alternative_candidate_mechanisms": alternative_candidates,
            "alternative_candidate_evidence": (
                "counterfactual_decoupling_robust_with_parent_aligned_behavioral_tie"
                if generic_rebinding_required
                else None
            ),
            "cross_game_generalization_proven": False,
            "program_promoted": False,
            "production_authority": False,
            "sequence_composition_authorized": False,
            "cross_game_preregistration_authorized": passed,
            "next_step": next_step,
            "evidence_phase": evidence_phase,
            "evidence_checksum": evidence_checksum,
            "physical_actions": 0,
        },
        "report_checksum",
    )
    _write(root, ADJUDICATION_FILENAME, payload)
    if not passed:
        raise protocol.ScientificGateMiss(verdict)
    return payload


def terminal_report(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    path = _path(root, TERMINAL_REPORT_FILENAME)
    if path.is_file():
        return _read_signed(root, TERMINAL_REPORT_FILENAME, "report_checksum")
    artifacts: dict[str, str | None] = {}
    for phase, contract in protocol.ARTIFACT_CONTRACT.items():
        if phase == "report":
            continue
        artifact = _path(root, str(contract["path"]))
        if not artifact.is_file():
            artifacts[phase] = None
            continue
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        field = str(contract["checksum_field"])
        protocol.verify_signed(payload, field)
        artifacts[phase] = str(payload[field])
    adjudication_path = _path(root, ADJUDICATION_FILENAME)
    if adjudication_path.is_file():
        adjudication = _read_signed(root, ADJUDICATION_FILENAME, "report_checksum")
        verdict = str(adjudication["verdict"])
        passed = bool(adjudication["passed"])
        identified = list(adjudication.get("identified_factor_candidates", ()))
        evidence_grades = dict(adjudication.get("evidence_grades", {}))
        generic_rebinding_required = bool(
            adjudication.get("generic_rebinding_required")
        )
        alternative_candidates = list(
            adjudication.get("alternative_candidate_mechanisms", ())
        )
    else:
        verdict = "INCOMPLETE_T10_3_12B"
        passed = False
        identified = []
        evidence_grades = {}
        generic_rebinding_required = False
        alternative_candidates = []
    intervention_path = _path(root, INTERVENTION_REPORT_FILENAME)
    virtual_actions = 0
    if intervention_path.is_file():
        intervention = _read_signed(root, INTERVENTION_REPORT_FILENAME, "report_checksum")
        virtual_actions = int(intervention.get("metrics", {}).get("virtual_actions", 0))
    payload = signed(
        {
            "format_version": "sage-t10.3.12b-terminal-report-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "passed": passed,
            "verdict": verdict,
            "identified_factor_candidates": identified,
            "evidence_grades": evidence_grades,
            "generic_rebinding_required": generic_rebinding_required,
            "alternative_candidate_mechanisms": alternative_candidates,
            "cross_game_generalization_proven": False,
            "cross_game_preregistration_authorized": passed,
            "sequence_composition_authorized": False,
            "accounting": {
                "physical_actions_authorized": 0,
                "physical_actions_sealed": 0,
                "physical_actions_replayed": 0,
                "virtual_actions": virtual_actions,
                "equation_holds": True,
            },
            "artifacts": artifacts,
            "parent_events_used_for_training": 0,
            "parent_registry_support_imported": 0,
            "production_authority": False,
            "sequence_games_opened": False,
            "firewall": dict(manifest["firewall"]),
        },
        "report_checksum",
    )
    _write(root, TERMINAL_REPORT_FILENAME, payload)
    return payload


def status(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for phase, contract in protocol.ARTIFACT_CONTRACT.items():
        artifact = _path(root, str(contract["path"]))
        if not artifact.is_file():
            artifacts[phase] = None
            continue
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        field = str(contract["checksum_field"])
        protocol.verify_signed(payload, field)
        artifacts[phase] = payload[field]
    drive = Path(root.resolve().anchor)
    try:
        import shutil

        free_bytes = int(shutil.disk_usage(drive).free)
    except OSError:
        free_bytes = -1
    return {
        "format_version": "sage-t10.3.12b-status-v1",
        "status": "READY",
        "manifest_checksum": manifest["manifest_checksum"],
        "artifacts": artifacts,
        "artifact_bytes": _artifact_bytes(root),
        "free_disk_bytes": free_bytes,
        "physical_actions_authorized": 0,
        "sequence_games_opened": False,
        "production_authority": False,
    }


def _emit(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "freeze",
            "status",
            "audit-parent",
            "preflight",
            "materialize-variants",
            "compile-factors",
            "evaluate-interventions",
            "adjudicate",
            "report",
        ),
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    try:
        if args.phase == "freeze":
            manifest, receipt = protocol.freeze_manifest(root)
            _emit(
                {
                    "status": "FROZEN",
                    "manifest_checksum": manifest["manifest_checksum"],
                    "freeze_receipt_checksum": receipt["receipt_checksum"],
                    "physical_actions": 0,
                }
            )
            return 0
        manifest = protocol.load_manifest(root)
        handlers = {
            "status": status,
            "audit-parent": audit_parent,
            "preflight": preflight,
            "materialize-variants": materialize_variants,
            "compile-factors": compile_factors,
            "evaluate-interventions": evaluate_interventions,
            "adjudicate": adjudicate,
            "report": terminal_report,
        }
        result = handlers[args.phase](root, manifest)
        _emit(result)
        if args.phase == "report" and not bool(result.get("passed")):
            return 3
        return 0
    except protocol.ScientificGateMiss as exc:
        _emit({"error": str(exc), "exit_code": 3, "phase": args.phase})
        return 3
    except (protocol.IntegrityError, ValueError, KeyError, OSError) as exc:
        _emit(
            {
                "error": "INVALID_PROVENANCE",
                "detail": f"{type(exc).__name__}:{str(exc)[:240]}",
                "exit_code": 2,
                "phase": args.phase,
            }
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADJUDICATION_FILENAME",
    "FACTOR_REGISTRY_FILENAME",
    "INTERVENTION_REPORT_FILENAME",
    "PARENT_AUDIT_FILENAME",
    "PREFLIGHT_FILENAME",
    "TERMINAL_REPORT_FILENAME",
    "VARIANT_INVENTORY_FILENAME",
    "adjudicate",
    "audit_parent",
    "compile_factors",
    "evaluate_interventions",
    "main",
    "materialize_variants",
    "preflight",
    "status",
    "terminal_report",
]
