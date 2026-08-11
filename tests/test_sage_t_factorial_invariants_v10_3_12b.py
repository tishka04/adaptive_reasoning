from __future__ import annotations

from theory.sage_t.factorial_invariants_v10_3_12b import (
    ARMS,
    CONTEXTS,
    FactorRegistry,
    compile_factor_registry,
    evaluate_trial,
    materialize_variant,
    variant_recipes,
)
from theory.sage_t.relational_program_v10_3_12 import (
    compile_candidate_registry,
    signed,
)


def _parent_registry() -> dict:
    registry = compile_candidate_registry(
        {
            "repeat": {
                "macro_schema": "repeat_target",
                "target_selector": "same_effect_distinct_target",
            },
            "path": {
                "macro_schema": "path_successor",
                "target_selector": "successor_toward_enclosure",
            },
        },
        repeat_projection_identified=True,
    ).snapshot()
    core = {key: value for key, value in registry.items() if key != "registry_checksum"}
    core.update({"local_support_total": 0, "physical_actions": 0})
    return signed(core, "registry_checksum")


def test_variant_matrix_is_balanced_and_non_universal() -> None:
    recipes = variant_recipes()
    assert len(recipes) == 128
    assert len({row["variant_id"] for row in recipes}) == 128
    assert sum(row["split"] == "identification" for row in recipes) == 64
    assert sum(row["split"] == "challenge" for row in recipes) == 64
    assert sum(row["challenge"] == "ambiguous_role" for row in recipes) == 32
    for context in CONTEXTS:
        worlds = [materialize_variant(row) for row in recipes if row["context"] == context]
        assert len(worlds) == 64
        assert len({world.state_hash for world in worlds}) == 48


def test_factor_registry_is_support_zero_and_round_trips() -> None:
    compiled = compile_factor_registry(_parent_registry())
    snapshot = compiled.snapshot()
    assert snapshot["local_support_total"] == 0
    assert snapshot["promotion_count"] == 0
    assert len(snapshot["programs"]) == len(ARMS) * len(CONTEXTS)
    restored = FactorRegistry(snapshot)
    for arm in ARMS:
        for context in CONTEXTS:
            assert restored.program_for(arm, context).program_hash == compiled.program_for(arm, context).program_hash


def test_source_changes_first_decision_and_single_factor_ablations_fail() -> None:
    registry = compile_factor_registry(_parent_registry())
    for context in CONTEXTS:
        recipe = next(
            row
            for row in variant_recipes()
            if row["context"] == context
            and row["challenge"] == "long_positive"
            and row["transform"] == "rotate_90"
            and row["order"] == "canonical"
        )
        world = materialize_variant(recipe)
        source = evaluate_trial(registry.program_for(ARMS[0], context), world)
        generic = evaluate_trial(registry.program_for(ARMS[1], context), world)
        assert source.correct is True
        assert generic.correct is True
        assert source.first_decision_class != generic.first_decision_class
        assert source.virtual_actions < generic.virtual_actions
        for arm in ("operator_ablation", "transition_ablation", "termination_ablation"):
            outcome = evaluate_trial(registry.program_for(arm, context), world)
            assert outcome.correct is False


def test_ambiguity_requires_abstention_and_role_binding() -> None:
    registry = compile_factor_registry(_parent_registry())
    for context in CONTEXTS:
        recipe = next(
            row
            for row in variant_recipes()
            if row["context"] == context and row["challenge"] == "ambiguous_role"
        )
        world = materialize_variant(recipe)
        source = evaluate_trial(registry.program_for(ARMS[0], context), world)
        ablated = evaluate_trial(registry.program_for("role_binding_ablation", context), world)
        assert source.correct is True and source.abstained is True
        assert ablated.correct is False and ablated.abstained is False


def test_verified_source_prior_rebinds_under_decoupling() -> None:
    registry = compile_factor_registry(_parent_registry())
    for context in CONTEXTS:
        recipe = next(
            row
            for row in variant_recipes()
            if row["context"] == context
            and row["challenge"] == "relation_decoupled"
            and row["transform"] == "mirror_x"
        )
        world = materialize_variant(recipe)
        source = evaluate_trial(registry.program_for(ARMS[0], context), world)
        generic = evaluate_trial(registry.program_for(ARMS[1], context), world)
        assert source.correct is True
        assert source.first_decision_class == "verify_source_prior"
        assert 2 <= source.probes <= generic.probes
        assert generic.correct is True
        assert source.virtual_actions < generic.virtual_actions
