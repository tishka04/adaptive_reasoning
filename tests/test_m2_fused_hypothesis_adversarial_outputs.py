import pytest

from theory.m2.fused_hypothesis_generator import (
    build_fused_input_packet,
    validate_fused_candidate_contract,
)
from tests.test_m2_fused_hypothesis_generator import _sources


def test_fused_input_packet_rejects_source_support_positive():
    sources = _sources()
    sources["arc_lewm_signal_report"]["support"] = 1

    with pytest.raises(ValueError, match="support must remain 0"):
        build_fused_input_packet(sources)


def test_fused_input_packet_rejects_source_world_model_evidence_flag():
    sources = _sources()
    sources["arc_lewm_signal_report"]["summary"][
        "world_model_counted_as_evidence"
    ] = True

    with pytest.raises(ValueError, match="world_model_counted_as_evidence"):
        build_fused_input_packet(sources)


def test_fused_input_packet_rejects_downstream_write_flag():
    sources = _sources()
    sources["m3_arc_lewm_replication_results"]["summary"][
        "a32_write_performed"
    ] = True

    with pytest.raises(ValueError, match="a32_write_performed"):
        build_fused_input_packet(sources)


def test_fused_candidate_contract_rejects_verdict_like_status():
    with pytest.raises(ValueError, match="status must remain UNRESOLVED"):
        validate_fused_candidate_contract(
            {
                "status": "CONFIRMED",
                "support": 0,
                "truth_status": "NOT_EVALUATED_BY_M2",
                "revision_performed": False,
                "wrong_confirmations": 0,
            }
        )


def test_fused_candidate_contract_rejects_support_positive():
    with pytest.raises(ValueError, match="support must remain 0"):
        validate_fused_candidate_contract(
            {
                "status": "UNRESOLVED",
                "support": 1,
                "truth_status": "NOT_EVALUATED_BY_M2",
                "revision_performed": False,
                "wrong_confirmations": 0,
            }
        )


def test_fused_candidate_contract_rejects_ready_for_a32_or_a33():
    for key in ("ready_for_a32", "ready_for_a33"):
        row = {
            "status": "UNRESOLVED",
            "support": 0,
            "truth_status": "NOT_EVALUATED_BY_M2",
            "revision_performed": False,
            "wrong_confirmations": 0,
            key: True,
        }
        with pytest.raises(ValueError, match="must not be ready for A32/A33"):
            validate_fused_candidate_contract(row)


def test_fused_candidate_contract_rejects_counted_evidence_flags():
    for key in (
        "llm_output_counted_as_evidence",
        "world_model_score_counted_as_support",
        "world_model_counted_as_evidence",
        "m3_observation_counted_as_confirmation",
        "terminal_risk_support_events_counted_as_support",
    ):
        row = {
            "status": "UNRESOLVED",
            "support": 0,
            "truth_status": "NOT_EVALUATED_BY_M2",
            "revision_performed": False,
            "wrong_confirmations": 0,
            key: True,
        }
        with pytest.raises(ValueError, match=key):
            validate_fused_candidate_contract(row)
