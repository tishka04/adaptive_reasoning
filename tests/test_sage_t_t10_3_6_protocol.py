from __future__ import annotations

from theory.sage_t import t10_3_6_protocol as protocol


def test_frozen_matrix_is_functional_and_bounded() -> None:
    assert len(protocol.work_specs("witness-core")) == 4
    assert len(protocol.work_specs("discover-core")) == 8
    assert len(protocol.work_specs("reproduce-core")) == 4
    assert len(protocol.work_specs("discover-sequence")) == 6
    assert len(protocol.work_specs("confirm")) == 20
    assert protocol.TOTAL_RESETS == 42
    assert protocol.TOTAL_MAXIMUM_ACTIONS == 1856


def test_ambiguous_bindings_are_counterbalanced_by_reset_index() -> None:
    for phase in ("witness-core", "discover-core", "reproduce-core"):
        by_game: dict[str, list[int]] = {}
        for work in protocol.work_specs(phase):
            by_game.setdefault(work.game_id, []).append(work.reset_index)
        assert all(values == list(range(len(values))) for values in by_game.values())


def test_witness_descriptors_are_structural_only() -> None:
    lp = protocol.WITNESS_PROGRAMS["lp85-305b61c3"]
    su = protocol.WITNESS_PROGRAMS["su15-4c352900"]

    assert (lp["macro_schema"], lp["horizon"]) == ("repeat_target", 5)
    assert (su["macro_schema"], su["horizon"]) == ("path_successor", 10)
    for descriptor in protocol.WITNESS_PROGRAMS.values():
        assert "actions" not in descriptor
        assert "action_data" not in descriptor
        assert "x" not in descriptor and "y" not in descriptor


def test_confirmation_is_counterbalanced() -> None:
    rows = protocol.work_specs("confirm")
    assert sum(work.arm == "goal_directed_sage_t" for work in rows) == 10
    assert sum(work.arm == "unified_sage_t_off" for work in rows) == 10
    for game in protocol.ALL_SOURCE_GAMES:
        for seed in protocol.CONFIRMATION_SEEDS:
            pair = [work.arm for work in rows if work.game_id == game and work.seed == seed]
            assert sorted(pair) == ["goal_directed_sage_t", "unified_sage_t_off"]
