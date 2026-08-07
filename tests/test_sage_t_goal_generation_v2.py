from __future__ import annotations

from theory.sage12.bound_mechanic_pilot import load_pairs
from theory.sage_t import calibration_gate_v8_6 as v86
from theory.sage_t.executor import ProgramExecutor
from theory.sage_t.goal_generation_v2 import (
    goal_progress_bridge_fragment,
    needs_goal_progress_bridge,
    programs_for_with_goal_progress_bridge,
)
from theory.sage_t.replay_gate import _programs_for as frozen_programs_for


def _goal_sequences() -> list[dict[str, object]]:
    pairs = load_pairs(str(v86.DEFAULT_SHARD_DIR), v86.EXPECTED_GAMES)
    return [
        sequence
        for sequence in v86._signal_sequences(pairs)
        if sequence["positive_kind"] == "goal"
    ]


def test_goal_bridge_fragment_has_zero_support_and_typed_bundle() -> None:
    fragment = goal_progress_bridge_fragment()
    progress, goal, terminal = fragment.payload

    assert fragment.support == 0
    assert fragment.prior_logprob == -0.05
    assert fragment.provenance == ("sage_t_deterministic_goal_progress_bridge",)
    assert progress.expression.op == "counter"
    assert progress.expression.value == "progress"
    assert goal.family == "level_completion"
    assert len(terminal) == 2


def test_generator_is_bit_equivalent_before_positive_goal_signal() -> None:
    manifest = v86.load_t7_manifest(verify_code=True)
    sequence = _goal_sequences()[0]
    panels = sequence["panels"]
    keys = sequence["keys"]
    neutral = [
        next(arm for arm in panel.arms if arm.action.key == key)
        for panel, key in zip(panels[:-1], keys[:-1])
    ]
    actions = ("ACTION6",)

    baseline = frozen_programs_for(actions, neutral, manifest)
    challenger = programs_for_with_goal_progress_bridge(
        actions,
        neutral,
        manifest,
    )

    assert not needs_goal_progress_bridge(neutral)
    assert challenger == baseline


def test_all_three_source_train_goal_signals_are_generated_compatibly() -> None:
    manifest = v86.load_t7_manifest(verify_code=True)
    executor = ProgramExecutor()
    sequences = _goal_sequences()

    assert len(sequences) == 3
    for sequence in sequences:
        revealed = [
            next(arm for arm in panel.arms if arm.action.key == key)
            for panel, key in zip(sequence["panels"], sequence["keys"])
        ]
        actions = tuple(
            sorted(
                {
                    arm.action.action_name
                    for panel in sequence["panels"]
                    for arm in panel.arms
                }
            )
        )
        programs = programs_for_with_goal_progress_bridge(
            actions,
            revealed,
            manifest,
        )
        compatible = v86._compatible_programs(
            programs,
            sequence["positive"],
            executor,
        )

        assert needs_goal_progress_bridge(revealed)
        assert len(compatible) >= 1
        assert all(
            item.program.progress_rule.expression.op == "counter"
            for item in compatible
        )
