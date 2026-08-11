from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from theory.sage_t import t10_3_12f_protocol as protocol
from theory.sage_t import t10_3_12f_runtime as runtime
from theory.sage_t.causal_procedure_v10_3_12f import (
    MODEL_FAMILIES,
    CausalOutcome,
    ProcedureUpdate,
    signed,
    uniform_prior,
)


def _manifest() -> dict[str, Any]:
    return {
        "manifest_checksum": "synthetic-t10-3-12f-manifest",
        "gates": {
            "holm_familywise_alpha": 0.05,
            "minimum_candidate_success_games": 2,
            "minimum_identification_verified_games": 2,
            "minimum_identification_better_games_each_control": 5,
        },
        "firewall": {
            "holdout_opened": False,
            "ar25_opened": False,
            "source_validation_opened": False,
            "sequence_games_opened": False,
            "production_authority": False,
            "t10_3_13_authorized": False,
        },
    }


def _write_signed_receipts(
    root: Path,
    manifest: dict[str, Any],
    *,
    utility_by_arm: dict[str, float],
    successful_arms: frozenset[str],
    verified_context_diversity: int = 0,
    source_identification_advantage: bool = True,
) -> None:
    destination = runtime._destination(root)
    for work in protocol.work_specs("active-historical"):
        receipt = signed(
            {
                "format_version": "sage-t10.3.12f-branch-receipt-v1",
                "manifest_checksum": manifest["manifest_checksum"],
                **work.as_dict(),
                "work_id": work.work_id,
                "status": "COMPLETE",
                "complete": True,
                "sealed_events": 0,
                "observed_updates": 0,
                "level_delta": int(work.arm in successful_arms),
                "utility": float(utility_by_arm[work.arm]),
                "prequential_log_loss": (
                    0.1
                    if source_identification_advantage
                    and work.arm == "source_closed_loop"
                    else 1.0
                ),
                "procedure_summary": {
                    "verified_context_diversity": verified_context_diversity,
                    "entered_control": verified_context_diversity >= 2,
                    "interventions_before_verification": (
                        1
                        if verified_context_diversity >= 2
                        and source_identification_advantage
                        and work.arm == "source_closed_loop"
                        else 8 if verified_context_diversity >= 2 else None
                    ),
                },
                "illegal_actions": 0,
                "legacy_fallback_actions": 0,
                "physical_actions_replayed": 0,
            },
            "receipt_checksum",
        )
        protocol.write_json_once(runtime._receipt_path(destination, work), receipt)

    active = signed(
        {
            "format_version": "sage-t10.3.12f-active-historical-v1",
            "manifest_checksum": manifest["manifest_checksum"],
            "collection_complete": True,
            "holdout_opened": False,
            "production_authority": False,
        },
        "report_checksum",
    )
    protocol.write_json_once(
        runtime._path(root, runtime.ACTIVE_REPORT_FILENAME),
        active,
    )


def test_status_cli_emits_exactly_one_json_object(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = runtime.main(["status", "--repo-root", str(tmp_path)])

    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert code == 0
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["format_version"] == "sage-t10.3.12f-status-v1"
    assert payload["manifest_frozen"] is False
    assert payload["accounting"]["authorized_actions"] == 0
    assert payload["holdout_opened"] is False
    assert payload["t10_3_13_authorized"] is False


def test_incomplete_existing_receipt_is_rejected_before_reuse() -> None:
    work = protocol.work_specs("active-historical")[0]
    receipt = {
        "complete": False,
        "manifest_checksum": "manifest",
        "prior_checksum_loaded": "prior",
        "work_id": work.work_id,
        **work.as_dict(),
        "issued_intents": 0,
        "sealed_events": 0,
        "observed_updates": 0,
    }

    with pytest.raises(protocol.IntegrityError, match="not complete"):
        runtime._validate_receipt_binding(
            receipt,
            work=work,
            manifest_checksum="manifest",
            prior_checksum="prior",
        )


def test_action_intent_retains_checksums_but_no_raw_action_name_or_data() -> None:
    work = protocol.WorkSpec(
        phase="active-historical",
        game_id=protocol.TARGET_GAMES[0],
        scope_index=0,
        tie_break_seed=312_600,
        arm="source_closed_loop",
        reset_index=0,
        action_budget=1,
    )
    selected = SimpleNamespace(
        name="ULTRA_SECRET_GROUNDED_OPERATOR",
        action_args={"row": 17, "column": 23, "color": "vermillion"},
    )
    decision = SimpleNamespace(
        safe_payload={
            "phase": "IDENTIFY",
            "predicted_family": "stable_repeat",
            "predicted_probability": 0.5,
        }
    )

    intent = runtime._intent(
        _manifest(),
        work,
        step_index=0,
        selected=selected,
        decision=decision,
        prior_checksum="synthetic-prior",
    )
    encoded = json.dumps(intent, sort_keys=True)

    protocol.verify_signed(intent, "intent_checksum")
    assert "ULTRA_SECRET_GROUNDED_OPERATOR" not in encoded
    assert "vermillion" not in encoded
    assert "action_name" not in encoded
    assert "action_data" not in encoded
    assert intent["physical_action"] == {
        "operator_checksum": protocol.sha256_payload(
            "ULTRA_SECRET_GROUNDED_OPERATOR"
        ),
        "parameter_arity": 3,
        "argument_checksum": protocol.sha256_payload(selected.action_args),
    }
    assert intent["raw_arguments_retained"] is False


def test_one_synthetic_work_seals_event_before_observe_and_matches_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work = protocol.WorkSpec(
        phase="active-historical",
        game_id=protocol.TARGET_GAMES[0],
        scope_index=0,
        tie_break_seed=312_600,
        arm="source_closed_loop",
        reset_index=0,
        action_budget=1,
    )
    destination = runtime._destination(tmp_path)
    selected = SimpleNamespace(
        name="SECRET_GROUNDED_ACTION",
        action_args={"x": 4, "y": 9},
    )
    before = SimpleNamespace(
        grid=np.zeros((3, 3), dtype=np.int64),
        game_state="NOT_FINISHED",
        levels_completed=0,
        available_actions=("SECRET_GROUNDED_ACTION",),
    )
    after = SimpleNamespace(
        grid=np.ones((3, 3), dtype=np.int64),
        game_state="NOT_FINISHED",
        levels_completed=1,
        available_actions=("SECRET_GROUNDED_ACTION",),
    )
    outcome = CausalOutcome(level_delta=1, quality=1.0)
    event_was_sealed_when_observed: list[bool] = []

    class FakeEnvironment:
        def close(self) -> None:
            return None

    class FakeController:
        def observe(self, *_args: Any, **kwargs: Any) -> ProcedureUpdate:
            event_path = runtime._work_path(
                destination,
                "events",
                work,
                "0000.json",
            )
            event_was_sealed_when_observed.append(event_path.is_file())
            return ProcedureUpdate(
                phase_before="VERIFY",
                phase_after="CONTROL",
                predicted_family="stable_repeat",
                predicted_probability=0.9,
                outcome=kwargs["outcome"],
                posterior={family: 0.25 for family in MODEL_FAMILIES},
                mismatch=False,
                revised=False,
                abstained=False,
                reason="synthetic observation",
            )

    class FakeActiveProcedure:
        def __init__(self, _work: protocol.WorkSpec, _prior: Any) -> None:
            self.controller = FakeController()
            self.interventions_before_verification: int | None = None

        def decide(self, **_kwargs: Any) -> tuple[Any, Any, Any]:
            candidate = SimpleNamespace(
                action_name="abstract_operator",
                action_data={},
            )
            decision = SimpleNamespace(
                candidate=candidate,
                safe_payload={
                    "phase": "VERIFY",
                    "predicted_family": "stable_repeat",
                    "predicted_probability": 0.9,
                    "program_hash": "abstract-program",
                    "candidates_inspected": 1,
                    "abstained": False,
                },
            )
            cognitive = runtime.CognitiveDecision(
                action_name="SECRET_GROUNDED_ACTION",
                action_data={"x": 4, "y": 9},
                source="sage_t_causal_procedure",
                reason="synthetic",
                confidence=0.9,
                option_id="abstract-program",
            )
            state = SimpleNamespace(signature="abstract-state-before")
            return cognitive, decision, state

        def summary(self) -> dict[str, Any]:
            return {
                "interventions_before_verification": (
                    self.interventions_before_verification
                ),
                "verified_context_diversity": 2,
            }

    class FakeLock:
        def heartbeat(self) -> None:
            return None

    monkeypatch.setattr(runtime, "_ActiveProcedure", FakeActiveProcedure)
    monkeypatch.setattr(runtime.durable.live, "_make_real_env", lambda *_args: FakeEnvironment())
    monkeypatch.setattr(runtime.durable.live, "_reset_env", lambda _env: before)
    monkeypatch.setattr(
        runtime.durable.live,
        "snapshot_frame",
        lambda frame, **_kwargs: frame,
    )
    monkeypatch.setattr(runtime.durable.live, "_is_terminal", lambda _state: False)
    monkeypatch.setattr(runtime.durable.live, "_valid_actions", lambda _env: (selected,))
    monkeypatch.setattr(
        runtime.durable.live,
        "_materialize_decision",
        lambda _legal, _decision: selected,
    )
    monkeypatch.setattr(
        runtime.durable.live,
        "_step_env_action",
        lambda _env, _selected: after,
    )
    monkeypatch.setattr(
        runtime.durable.live,
        "_available_action_names",
        lambda legal: tuple(item.name for item in legal),
    )
    monkeypatch.setattr(runtime, "build_observation", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        runtime,
        "compile_observation",
        lambda _observation: SimpleNamespace(signature="abstract-state-after"),
    )
    monkeypatch.setattr(runtime, "abstract_context_signature", lambda _state: "context")
    monkeypatch.setattr(
        runtime,
        "_compile_causal_transition",
        lambda **_kwargs: (
            object(),
            outcome,
            {
                "correspondence_quality": 1.0,
                "persistent_one_to_one": True,
                "ambiguous_correspondence": False,
                "relation_conflict_rejected": False,
                "birth_death_relation_evidence_used": False,
            },
        ),
    )

    receipt = runtime._run_work(
        tmp_path,
        destination,
        _manifest(),
        work,
        uniform_prior(),
        "synthetic-prior",
        FakeLock(),
    )

    intent = json.loads(
        runtime._work_path(destination, "intents", work, "0000.json").read_text(
            encoding="utf-8"
        )
    )
    event = json.loads(
        runtime._work_path(destination, "events", work, "0000.json").read_text(
            encoding="utf-8"
        )
    )
    update = json.loads(
        runtime._work_path(destination, "updates", work, "0000.json").read_text(
            encoding="utf-8"
        )
    )

    assert event_was_sealed_when_observed == [True]
    assert intent["event_id"] == event["event_id"] == update["event_id"]
    assert receipt["issued_intents"] == 1
    assert receipt["sealed_events"] == 1
    assert receipt["observed_updates"] == 1
    assert receipt["complete"] is True
    assert receipt["level_delta"] == 1
    assert receipt["first_level_action"] == 1
    assert receipt["utility"] == 1.0
    assert update["phase_after"] == "CONTROL"
    assert update["posterior_digest"]
    assert event["grounded_arguments_retained"] is False
    assert update["grounded_payload_retained"] is False


def test_exact_sign_permutation_and_holm_correction() -> None:
    assert runtime._exact_positive_sign_permutation([1.0] * 6) == pytest.approx(1 / 64)
    assert runtime._exact_positive_sign_permutation([0.0, 0.0]) == 1.0

    corrected = runtime._holm(
        {
            "first": {"raw_p_value": 0.01},
            "second": {"raw_p_value": 0.02},
            "third": {"raw_p_value": 0.20},
        },
        alpha=0.05,
    )
    assert corrected["first"]["holm_reject"] is True
    assert corrected["second"]["holm_reject"] is True
    assert corrected["third"]["holm_reject"] is False
    assert corrected["first"]["holm_adjusted_p_value"] == pytest.approx(0.03)
    assert corrected["second"]["holm_adjusted_p_value"] == pytest.approx(0.04)
    assert corrected["third"]["holm_adjusted_p_value"] == pytest.approx(0.20)


def test_adjudication_can_select_source_candidate_and_report_stays_firewalled(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    _write_signed_receipts(
        tmp_path,
        manifest,
        utility_by_arm={
            "source_closed_loop": 1.0,
            "uniform_closed_loop": 0.0,
            "permuted_source_closed_loop": 0.0,
            "source_open_loop": 0.0,
        },
        successful_arms=frozenset({"source_closed_loop"}),
    )

    adjudication = runtime.adjudicate(tmp_path, manifest)
    report = runtime.terminal_report(tmp_path, manifest)

    assert adjudication["passed"] is True
    assert adjudication["verdict"] == runtime.SOURCE_PASS
    assert adjudication["candidate_arm"] == "source_closed_loop"
    assert adjudication["control_arm"] == "uniform_closed_loop"
    assert all(
        row["holm_reject"] is True
        for row in adjudication["source_contrasts_holm"].values()
    )
    assert report["passed"] is True
    assert report["verdict"] == runtime.SOURCE_PASS
    assert report["historical_candidate_only"] is True
    for field in (
        "confirmatory_evidence",
        "prospective_generalization_proven",
        "program_promoted",
        "t10_3_13_authorized",
        "holdout_opened",
        "ar25_opened",
        "source_validation_opened",
        "sequence_games_opened",
        "production_authority",
    ):
        assert report[field] is False
    assert report["parent_events_used_for_training"] == 0
    assert report["target_history_events_used_for_initialization"] == 0
    assert report["physical_actions_replayed"] == 0
    assert report["legacy_fallback_actions"] == 0


def test_adjudication_all_zero_is_plain_negative(tmp_path: Path) -> None:
    manifest = _manifest()
    _write_signed_receipts(
        tmp_path,
        manifest,
        utility_by_arm={arm: 0.0 for arm in protocol.ARMS},
        successful_arms=frozenset(),
    )

    adjudication = runtime.adjudicate(tmp_path, manifest)

    assert adjudication["passed"] is False
    assert adjudication["verdict"] == "CAUSAL_PROCEDURE_NO_TARGET_PROGRESS"
    assert adjudication["candidate_arm"] is None
    assert adjudication["control_arm"] is None
    assert adjudication["historical_candidate_only"] is True
    assert adjudication["confirmatory_evidence"] is False
    assert adjudication["prospective_generalization_proven"] is False
    assert adjudication["t10_3_13_authorized"] is False
    assert adjudication["holdout_opened"] is False


def test_identification_without_control_requires_cross_arm_advantage(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    _write_signed_receipts(
        tmp_path,
        manifest,
        utility_by_arm={arm: 0.0 for arm in protocol.ARMS},
        successful_arms=frozenset(),
        verified_context_diversity=2,
        source_identification_advantage=False,
    )

    adjudication = runtime.adjudicate(tmp_path, manifest)

    assert adjudication["verdict"] == "CAUSAL_PROCEDURE_NO_TARGET_PROGRESS"
    assert adjudication["identification_advantage"]["passed"] is False


def test_identification_advantage_remains_a_negative_diagnostic(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    _write_signed_receipts(
        tmp_path,
        manifest,
        utility_by_arm={arm: 0.0 for arm in protocol.ARMS},
        successful_arms=frozenset(),
        verified_context_diversity=2,
    )

    adjudication = runtime.adjudicate(tmp_path, manifest)

    assert adjudication["passed"] is False
    assert adjudication["verdict"] == "CAUSAL_IDENTIFICATION_WITHOUT_CONTROL"
    assert adjudication["identification_advantage"]["passed"] is True
    assert (
        adjudication["identification_advantage"]["candidate_arm"]
        == "source_closed_loop"
    )
