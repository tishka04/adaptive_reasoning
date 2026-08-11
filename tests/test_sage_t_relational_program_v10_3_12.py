from __future__ import annotations

from theory.sage_t.relational_program_v10_3_12 import (
    ARMS,
    CONTEXTS,
    RelationalProgramRegistry,
    assert_transfer_safe,
    compile_candidate_registry,
    evaluate_fixture,
    fixture_correct,
    fixture_recipes,
    inverse_transform,
    materialize_fixture,
    transform_point,
)


def _registry() -> RelationalProgramRegistry:
    return compile_candidate_registry(
        {
            "a": {
                "macro_schema": "repeat_target",
                "target_selector": "same_effect_distinct_target",
            },
            "b": {
                "macro_schema": "path_successor",
                "target_selector": "successor_toward_enclosure",
            },
        },
        repeat_projection_identified=True,
    )


def test_registry_is_transfer_safe_support_zero_and_round_trips() -> None:
    snapshot = _registry().snapshot()
    assert_transfer_safe(snapshot)
    assert len(snapshot["programs"]) == 8
    assert all(not row["support_scopes"] for row in snapshot["programs"])
    assert all(not row["promoted"] for row in snapshot["programs"])
    assert RelationalProgramRegistry(snapshot).snapshot() == snapshot
    encoded = str(snapshot).lower()
    for token in (
        "game_id",
        "action_data",
        "entity_id",
        "raw_grid",
        "argument_checksum",
        "lp85",
        "su15",
    ):
        assert token not in encoded


def test_fixture_matrix_is_exact_and_factorized_arm_is_equivariant() -> None:
    recipes = fixture_recipes()
    assert len(recipes) == 96
    fixtures = [materialize_fixture(row) for row in recipes]
    assert sum(row.control == "positive" for row in fixtures) == 64
    assert sum(row.control != "positive" for row in fixtures) == 32

    registry = _registry()
    correct = {arm: 0 for arm in ARMS}
    inspections = {arm: {context: [] for context in CONTEXTS} for arm in ARMS}
    for fixture in fixtures:
        for arm in ARMS:
            outcome = evaluate_fixture(
                registry.program_for(arm, fixture.context), fixture
            )
            correct[arm] += fixture_correct(fixture, outcome)
            inspections[arm][fixture.context].append(outcome.inspections)

    assert correct[ARMS[0]] == 96
    assert correct[ARMS[0]] >= correct[ARMS[2]] + 8
    assert correct[ARMS[0]] >= correct[ARMS[3]] + 8
    for context in CONTEXTS:
        factorized = sorted(inspections[ARMS[0]][context])
        generic = sorted(inspections[ARMS[1]][context])
        assert factorized[len(factorized) // 2] <= 0.5 * generic[len(generic) // 2]


def test_all_d4_transforms_have_exact_inverses() -> None:
    transforms = {row["transform"] for row in fixture_recipes()}
    for transform in transforms:
        transformed = transform_point((11, 37), transform)
        assert transform_point(transformed, inverse_transform(transform)) == (11, 37)


def test_compiler_rejects_grounded_evidence() -> None:
    try:
        compile_candidate_registry(
            {
                "grounded_evidence": {},
                "a": {
                    "macro_schema": "repeat_target",
                    "target_selector": "same_effect_distinct_target",
                },
                "b": {
                    "macro_schema": "path_successor",
                    "target_selector": "successor_toward_enclosure",
                },
            },
            repeat_projection_identified=True,
        )
    except ValueError as exc:
        assert "grounded evidence" in str(exc)
    else:
        raise AssertionError("grounded evidence must never enter the compiler")

