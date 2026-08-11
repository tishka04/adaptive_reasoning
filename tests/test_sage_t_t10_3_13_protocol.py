from __future__ import annotations

import pytest

from theory.sage_t import t10_3_13_protocol as protocol


def test_candidate_selection_is_frozen() -> None:
    assert protocol.candidate_pair(protocol.SOURCE_PASS) == (
        "source_closed_loop",
        "uniform_closed_loop",
    )
    assert protocol.candidate_pair(protocol.GENERIC_PASS) == (
        "uniform_closed_loop",
        "source_open_loop",
    )
    with pytest.raises(protocol.ScientificGateMiss):
        protocol.candidate_pair("CAUSAL_PROCEDURE_NO_TARGET_PROGRESS")


def test_prospective_matrix_is_one_paired_reset_per_game() -> None:
    specs = protocol.work_specs(
        "active-confirmation",
        candidate="source_closed_loop",
        control="uniform_closed_loop",
    )
    assert len(specs) == 10
    assert protocol.maximum_actions_for_specs(specs) == 480
    for game in protocol.PROTECTED_GAMES:
        rows = [work for work in specs if work.game_id == game]
        assert {row.role for row in rows} == {"candidate", "control"}
        assert {row.reset_index for row in rows} == {0}


def test_holdout_requires_exact_explicit_acknowledgement(tmp_path) -> None:
    with pytest.raises(protocol.IntegrityError):
        protocol.authorize_holdout(tmp_path, acknowledgement="yes")


def test_import_and_work_spec_do_not_claim_holdout_opened() -> None:
    assert protocol.PROTECTED_GAMES == ("s5i5", "vc33", "m0r0", "sk48", "r11l")
    assert protocol.TOTAL_RESETS == 10
    assert protocol.TOTAL_MAXIMUM_ACTIONS == 480
