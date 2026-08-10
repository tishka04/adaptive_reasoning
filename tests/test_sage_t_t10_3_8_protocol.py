from __future__ import annotations

from theory.sage_t import t10_3_8_protocol as protocol


def test_adjudication_removes_witness_recollection_from_matrix() -> None:
    assert protocol.TOTAL_RESETS == 38
    assert protocol.TOTAL_MAXIMUM_ACTIONS == 1792
    assert protocol.DISCOVERY_SEEDS == (3321, 3322, 3323, 3324)
    assert protocol.REPRODUCTION_SEEDS == (3331, 3332)
    assert protocol.SEQUENCE_SEEDS == (3341, 3342)
    assert protocol.CONFIRMATION_SEEDS == (3351, 3352)


def test_parent_snapshot_records_boolean_polarity_bug() -> None:
    parent = protocol.SUPERSEDED_T10_3_7
    assert parent["lp85_level_delta"] == 1
    assert parent["su15_level_delta"] == 2
    assert parent["false_checks"] == ("historical_grounded_actions_loaded",)
    assert parent["parent_passed"] is False
    assert parent["used_for_training"] is False
    assert parent["physical_actions_replayed"] == 0


def test_physical_work_starts_at_blank_discovery() -> None:
    try:
        protocol.work_specs("witness-core")
    except ValueError:
        pass
    else:
        raise AssertionError("T10.3.8 must not recollect the witness")
    assert len(protocol.work_specs("discover-core")) == 8
    assert len(protocol.work_specs("reproduce-core")) == 4
    assert len(protocol.work_specs("discover-sequence")) == 6
    assert len(protocol.work_specs("confirm")) == 20
