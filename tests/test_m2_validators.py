from dataclasses import replace

from theory.m2.validators import (
    validate_hypothesis,
    validate_m3_request,
    validate_testability,
)
from theory.m2.testability_compiler import compile_m3_request

from m2_test_helpers import valid_hypothesis


def test_validate_hypothesis_rejects_missing_source_request_id():
    hypothesis = replace(valid_hypothesis(), source_request_id="")

    assert not validate_hypothesis(hypothesis).valid


def test_validate_hypothesis_rejects_missing_frontier_context_id():
    hypothesis = replace(valid_hypothesis(), frontier_context_id="")

    assert not validate_hypothesis(hypothesis).valid


def test_validate_hypothesis_rejects_confirmation_and_support_fields():
    base = valid_hypothesis()

    assert not validate_hypothesis(replace(base, status="CONFIRMED")).valid
    assert not validate_hypothesis(replace(base, support=1)).valid
    assert not validate_hypothesis(replace(base, truth_status="CONFIRMED")).valid
    assert not validate_hypothesis(replace(base, revision_performed=True)).valid
    assert not validate_hypothesis(replace(base, wrong_confirmations=1)).valid


def test_validate_hypothesis_rejects_missing_falsification():
    data = valid_hypothesis().to_dict()
    data["falsification"] = None

    assert not validate_hypothesis(data).valid


def test_validate_testability_rejects_testable_without_metric_or_target():
    assert not validate_testability({"testable": True, "target_action": "ACTION3"}).valid
    assert not validate_testability({"testable": True, "metric": "changed_pixels"}).valid


def test_validate_m3_request_keeps_support_zero_and_unresolved():
    request = compile_m3_request(valid_hypothesis())
    assert validate_m3_request(request).valid

    data = request.to_dict()
    data["support"] = 1
    assert not validate_m3_request(data).valid
