from __future__ import annotations

from theory.sage_t.cross_game_transfer_v10_3_12c import (
    ARMS,
    FACTORS,
    CrossGameFactorRegistry,
    compile_cross_game_registry,
    select_grounding,
)
from theory.sage_t.factorial_invariants_v10_3_12b import FactorProgram, FactorRegistry
from theory.sage_t.contracts import ActionCandidate


def _parent_payload() -> dict:
    registry = FactorRegistry()
    for context in ("repeat_context", "path_context"):
        if context == "repeat_context":
            role = "boundary_prior_with_causal_verification"
            transition = "same_role_until_progress"
            generic_role = "probe_relations_then_bind_productive_role"
            horizon = 8
        else:
            role = "salient_end_prior_with_causal_verification"
            transition = "successor_toward_goal_end"
            generic_role = "probe_orientations_then_bind_goal_end"
            horizon = 16
        registry.register(
            FactorProgram(
                context=context,
                arm="factorized_source",
                operator="parameterized_apply",
                role_binding=role,
                transition=transition,
                termination="stop_on_progress_or_ambiguity",
                safety_horizon=horizon,
                source_kind="test_source",
            )
        )
        registry.register(
            FactorProgram(
                context=context,
                arm="generic_source_free",
                operator="parameterized_apply",
                role_binding=generic_role,
                transition=transition,
                termination="stop_on_progress_or_ambiguity",
                safety_horizon=horizon,
                source_kind="test_generic",
            )
        )
    return registry.snapshot()


def test_registry_has_full_generic_and_four_single_factor_ablations() -> None:
    registry = compile_cross_game_registry(_parent_payload())
    snapshot = registry.snapshot()
    assert len(snapshot["programs"]) == len(ARMS) * 2
    assert snapshot["local_support_total"] == 0
    for context in ("repeat_context", "path_context"):
        source = registry.program_for("factorized_source", context)
        for factor in FACTORS:
            ablated = registry.program_for(f"{factor}_ablation", context)
            changed = {
                name
                for name in FACTORS
                if getattr(source, name) != getattr(ablated, name)
            }
            assert changed == {factor}


def test_source_operator_abstains_without_parameters_but_controls_do_not() -> None:
    registry = compile_cross_game_registry(_parent_payload())
    candidates = (ActionCandidate("ACTION2"), ActionCandidate("ACTION1"))
    source = select_grounding(
        registry, arm="factorized_source", candidates=candidates,
        shape=(32, 32), step_index=0,
    )
    generic = select_grounding(
        registry, arm="generic_source_free", candidates=candidates,
        shape=(32, 32), step_index=0,
    )
    operator = select_grounding(
        registry, arm="operator_ablation", candidates=candidates,
        shape=(32, 32), step_index=0,
    )
    assert source.abstained
    assert not generic.abstained
    assert not operator.abstained


def test_relative_role_is_order_invariant_and_ambiguity_abstains() -> None:
    registry = compile_cross_game_registry(_parent_payload())
    unique = (
        ActionCandidate("ACTION6", {"x": 1, "y": 10}),
        ActionCandidate("ACTION6", {"x": 12, "y": 12}),
    )
    forward = select_grounding(
        registry, arm="factorized_source", candidates=unique,
        shape=(32, 32), step_index=0, forced_context="repeat_context",
    )
    reverse = select_grounding(
        registry, arm="factorized_source", candidates=tuple(reversed(unique)),
        shape=(32, 32), step_index=0, forced_context="repeat_context",
    )
    ambiguous = select_grounding(
        registry,
        arm="factorized_source",
        candidates=(
            ActionCandidate("ACTION6", {"x": 1, "y": 10}),
            ActionCandidate("ACTION6", {"x": 30, "y": 10}),
        ),
        shape=(32, 32),
        step_index=0,
        forced_context="repeat_context",
    )
    assert forward.candidate == reverse.candidate == unique[0]
    assert ambiguous.abstained
    assert ambiguous.reason == "ambiguous_relative_role"


def test_termination_ablation_is_a_preregistered_premature_stop() -> None:
    registry = CrossGameFactorRegistry(compile_cross_game_registry(_parent_payload()).snapshot())
    result = select_grounding(
        registry,
        arm="termination_ablation",
        candidates=(ActionCandidate("ACTION6", {"x": 1, "y": 10}),),
        shape=(32, 32),
        step_index=2,
        forced_context="repeat_context",
    )
    assert result.abstained
    assert result.reason == "termination_ablation_fixed_stop"
