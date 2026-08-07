from __future__ import annotations

from theory.sage12.bound_mechanic_pilot import load_pairs
from theory.sage_t import calibration_gate_v8_6 as v86
from theory.sage_t.goal_generation_v3 import (
    observed_goal_trigger_roles,
    programs_for_with_structural_goal_guard,
    structural_goal_guard_fragments,
)
from theory.sage_t.replay_gate import _programs_for as frozen_programs_for
from theory.sage_t.structural_roles import (
    WESTMOST_TARGET,
    StructuralRoleProgramExecutor,
)
from theory.sage_t.synthesis import DeterministicFragmentProposer


def _goal_sequences() -> list[dict[str, object]]:
    pairs = load_pairs(str(v86.DEFAULT_SHARD_DIR), v86.EXPECTED_GAMES)
    return [
        sequence
        for sequence in v86._signal_sequences(pairs)
        if sequence["positive_kind"] == "goal"
    ]


def _revealed(sequence: dict[str, object], *, include_positive: bool = True):
    panels = sequence["panels"]
    keys = sequence["keys"]
    pairs = list(zip(panels, keys))
    if not include_positive:
        pairs = pairs[:-1]
    return [
        next(arm for arm in panel.arms if arm.action.key == key)
        for panel, key in pairs
    ]


def _actions(sequence: dict[str, object]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                arm.action.action_name
                for panel in sequence["panels"]
                for arm in panel.arms
            }
        )
    )


def test_generator_is_bit_equivalent_before_structural_signal() -> None:
    manifest = v86.load_t7_manifest(verify_code=True)
    sequence = _goal_sequences()[0]
    neutral = _revealed(sequence, include_positive=False)
    actions = _actions(sequence)

    assert programs_for_with_structural_goal_guard(
        actions, neutral, manifest
    ) == frozen_programs_for(actions, neutral, manifest)


def test_all_goal_triggers_have_the_same_coordinate_free_role() -> None:
    sequences = _goal_sequences()

    assert len(sequences) == 3
    assert all(
        observed_goal_trigger_roles(_revealed(sequence))
        == {"ACTION6": WESTMOST_TARGET}
        for sequence in sequences
    )


def test_guard_fragments_are_support_free_and_conditioned() -> None:
    manifest = v86.load_t7_manifest(verify_code=True)
    sequence = _goal_sequences()[0]
    revealed = _revealed(sequence)
    proposal = DeterministicFragmentProposer(
        maximum_operator_candidates_per_action=int(
            manifest["generator"]["maximum_operator_candidates_per_action"]
        )
    ).propose(available_actions=_actions(sequence), transitions=revealed)

    guards = structural_goal_guard_fragments(
        proposal.fragments,
        {"ACTION6": WESTMOST_TARGET},
    )

    assert guards
    assert all(fragment.support == 0 for fragment in guards)
    assert all(
        fragment.payload[1].condition.terms
        == ("$target", WESTMOST_TARGET)
        for fragment in guards
    )
    assert all(
        fragment.prior_logprob
        == next(
            source.prior_logprob
            for source in proposal.fragments
            if source.fragment_id
            == fragment.fragment_id.removesuffix(
                f"_guard_{WESTMOST_TARGET}"
            )
        )
        - 0.10
        for fragment in guards
    )


def test_all_three_goal_signals_generate_compatible_guarded_programs() -> None:
    manifest = v86.load_t7_manifest(verify_code=True)
    executor = StructuralRoleProgramExecutor()
    sequences = _goal_sequences()

    for sequence in sequences:
        revealed = _revealed(sequence)
        programs = programs_for_with_structural_goal_guard(
            _actions(sequence), revealed, manifest
        )
        compatible = v86._compatible_programs(
            programs,
            sequence["positive"],
            executor,
        )

        assert len(programs) <= 64
        assert len(compatible) >= 1
        assert any(
            any(
                rule.condition.terms == ("$target", WESTMOST_TARGET)
                for rule in item.program.transition_rules
            )
            for item in compatible
        )
