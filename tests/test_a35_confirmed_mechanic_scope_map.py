import json

import theory.a35.confirmed_mechanic_scope_map as scope


KEY = (
    "mechanic_prediction::bp35-0a0ad940::ACTION6::"
    "position_effect_candidate::local_patch_before_after"
)


def _registry_entry():
    return {
        "key": KEY,
        "game_id": "bp35-0a0ad940",
        "action": "ACTION6",
        "mechanic_family": "position_effect_candidate",
        "predicted_metric": "local_patch_before_after",
        "known_scope": "local_context",
        "status": "confirmed",
    }


def _usage_probe():
    return {
        "key": KEY,
        "utility_assessment": "USEFUL",
        "functional_progress": True,
        "truth_status": "NOT_REEVALUATED_BY_A34",
    }


def _probe(context_sequence, *, useful=True, contradiction=False):
    signal = 1 if useful else 0
    return scope.ContextualMechanicScopeProbe(
        key=KEY,
        game_id="bp35-0a0ad940",
        context_id=scope.context_id(context_sequence),
        context_sequence=tuple(context_sequence),
        predicted_metric="local_patch_before_after",
        baseline_action="ACTION3",
        treatment_action="ACTION6",
        baseline_measurement={"local_changed_pixels": 0, "changed": True},
        treatment_measurement={"local_changed_pixels": signal, "changed": bool(signal)},
        baseline_signal=0.0,
        treatment_signal=float(signal),
        utility_assessment="USEFUL" if useful else "NOT_USEFUL",
        local_patch_before_after_observed=bool(signal),
        useful_new_state=useful,
        functional_progress=useful,
        usage_contradiction=contradiction,
    )


def test_scope_map_reports_precondition_dependent_without_revising(monkeypatch, tmp_path):
    registry_path = tmp_path / "confirmed_mechanics_registry.json"
    usage_path = tmp_path / "confirmed_mechanic_usage_probe.json"
    registry_path.write_text(
        json.dumps({"confirmed_mechanics": [_registry_entry()]}),
        encoding="utf-8",
    )
    usage_path.write_text(
        json.dumps({"usage_probes": [_usage_probe()]}),
        encoding="utf-8",
    )

    useful_contexts = {(), ("ACTION3",)}

    def fake_probe(entry, context_sequence, *, environments_dir, baseline_order):
        return _probe(
            tuple(context_sequence),
            useful=tuple(context_sequence) in useful_contexts,
        )

    monkeypatch.setattr(scope, "probe_context_for_mechanic", fake_probe)

    payload = scope.run_confirmed_mechanic_scope_map(
        registry_path=registry_path,
        usage_probe_path=usage_path,
        environments_dir=tmp_path,
    )
    scope_map = payload["scope_maps"][0]

    assert payload["summary"]["mechanics_mapped"] == 1
    assert payload["summary"]["contexts_tested"] == 4
    assert payload["summary"]["functional_progress_contexts"] == 2
    assert scope_map["scope_assessment"] == "PRECONDITION_DEPENDENT"
    assert scope_map["a34_usage_reference_loaded"] is True
    assert scope_map["truth_status"] == "NOT_REEVALUATED_BY_A35"
    assert scope_map["revision_performed"] is False
    assert scope_map["wrong_confirmations"] == 0


def test_scope_assessment_distinguishes_local_only_and_stable():
    local_only = (
        _probe((), useful=True),
        _probe(("ACTION3",), useful=False),
        _probe(("ACTION4",), useful=False),
    )
    stable = (
        _probe((), useful=True),
        _probe(("ACTION3",), useful=True),
        _probe(("ACTION4",), useful=True),
    )

    assert scope.scope_assessment_from_contexts(local_only) == "LOCAL_ONLY"
    assert scope.scope_assessment_from_contexts(stable) == "CONTEXTUALLY_STABLE"


def test_contextual_probe_can_report_usage_contradiction_without_verdict():
    probes = (
        _probe((), useful=False, contradiction=True),
        _probe(("ACTION3",), useful=False),
    )

    assert scope.scope_assessment_from_contexts(probes) == "UNSTABLE_OR_NOT_USEFUL"
    assert all(probe.truth_status == "NOT_REEVALUATED_BY_A35" for probe in probes)
