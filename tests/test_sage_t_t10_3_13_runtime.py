from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from theory.sage_t import t10_3_13_protocol as protocol
from theory.sage_t import t10_3_13_runtime as runtime


def _manifest(*, candidate: str = "source_closed_loop") -> dict[str, Any]:
    control = "uniform_closed_loop" if candidate == "source_closed_loop" else "source_open_loop"
    return {
        "manifest_checksum": "manifest-13",
        "authorization_checksum": "authorization-13",
        "parent_state": {"prior_checksum": "prior-12f"},
        "matrix": {
            "candidate_arm": candidate,
            "control_arm": control,
            "maximum_artifact_bytes": 32 * 1024 * 1024,
        },
        "gates": {
            "minimum_candidate_success_games": 3,
            "minimum_net_success_advantage": 2,
            "minimum_games_with_higher_utility": 4,
            "maximum_games_with_lower_utility": 0,
            "minimum_games_with_better_log_loss": 4,
        },
    }


def _empty_accounting() -> dict[str, Any]:
    return {
        "authorized_actions": 0,
        "sealed_events": 0,
        "unresolved_intents": 0,
        "inflight_intents": 0,
        "inflight_paths": [],
        "live_collector_lock": False,
        "equation_holds": True,
        "inflight_valid": True,
        "incomplete_work_ids": [],
    }


def test_status_without_manifest_proves_holdout_not_opened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        protocol,
        "load_manifest",
        lambda _root: pytest.fail("status must not load a manifest"),
    )
    monkeypatch.setattr(
        runtime,
        "_make_environment",
        lambda *_args, **_kwargs: pytest.fail("status opened a protected environment"),
    )
    result = runtime.status(tmp_path)
    assert result["manifest_frozen"] is False
    assert result["authorization_present"] is False
    assert result["holdout_opened"] is False
    assert result["holdout_not_opened_proven"] is True
    assert result["protected_frames_read"] == 0
    assert result["physical_actions"] == 0
    assert result["accounting"] == _empty_accounting()


def test_status_cli_emits_one_json_object_while_dormant(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = runtime.main(["status", "--repo-root", str(tmp_path)])
    lines = capsys.readouterr().out.splitlines()
    assert code == 0
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["holdout_not_opened_proven"] is True
    assert payload["holdout_opened"] is False


def test_incomplete_protected_receipt_is_rejected_before_reuse() -> None:
    manifest = _manifest()
    work = protocol.work_specs(
        "active-confirmation",
        candidate=str(manifest["matrix"]["candidate_arm"]),
        control=str(manifest["matrix"]["control_arm"]),
    )[0]

    with pytest.raises(protocol.IntegrityError, match="not complete"):
        runtime._validate_receipt_binding(
            {"complete": False},
            manifest=manifest,
            work=work,
            prior_checksum="prior-12f",
        )


def test_authorize_holdout_forwards_only_the_exact_phrase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []

    def fake_authorize(
        _root: Path,
        *,
        acknowledgement: str,
    ) -> dict[str, Any]:
        calls.append(acknowledgement)
        assert acknowledgement == protocol.AUTHORIZATION_PHRASE
        return {"authorization_checksum": "authorized"}

    monkeypatch.setattr(protocol, "authorize_holdout", fake_authorize)
    code = runtime.main(
        [
            "authorize-holdout",
            "--repo-root",
            str(tmp_path),
            "--acknowledgement",
            protocol.AUTHORIZATION_PHRASE,
        ]
    )
    lines = capsys.readouterr().out.splitlines()
    assert code == 0
    assert calls == [protocol.AUTHORIZATION_PHRASE]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["status"] == "HOLDOUT_AUTHORIZED_NOT_OPENED"
    assert payload["protected_frames_read"] == 0
    assert payload["holdout_opened"] is False


def test_wrong_authorization_phrase_fails_before_parent_or_holdout_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        protocol,
        "verify_parent_candidate",
        lambda _root: pytest.fail("wrong phrase must fail before parent inspection"),
    )
    monkeypatch.setattr(
        runtime,
        "_make_environment",
        lambda *_args, **_kwargs: pytest.fail("authorization opened the holdout"),
    )
    code = runtime.main(
        [
            "authorize-holdout",
            "--repo-root",
            str(tmp_path),
            "--acknowledgement",
            "yes",
        ]
    )
    lines = capsys.readouterr().out.splitlines()
    assert code == 2
    assert len(lines) == 1
    assert json.loads(lines[0])["phase"] == "authorize-holdout"


def test_preflight_is_zero_frame_and_zero_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePrior:
        prior_checksum = "prior-12f"

    class FakeController:
        def __init__(self, arm: str, *, scope: int, prior: Any) -> None:
            del scope, prior
            self.arm = arm

        def summary(self) -> dict[str, Any]:
            return {"arm": self.arm, "actions": 0, "program_hash": f"program-{self.arm}"}

    writes: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(runtime, "_require_gate", lambda *_args: {"passed": True})
    monkeypatch.setattr(
        runtime,
        "_read_parent_prior",
        lambda *_args: (FakePrior(), {"prior_checksum": "prior-12f"}),
    )
    monkeypatch.setattr(
        runtime,
        "_causal_api",
        lambda: (
            FakePrior,
            FakeController,
            (object, lambda _prior: {"passed": True}),
        ),
    )
    monkeypatch.setattr(runtime, "_accounting", lambda _root: _empty_accounting())
    monkeypatch.setattr(runtime, "_write", lambda _r, name, payload: writes.append((name, payload)))
    monkeypatch.setattr(
        runtime,
        "_make_environment",
        lambda *_args, **_kwargs: pytest.fail("preflight opened a protected environment"),
    )
    result = runtime.preflight(tmp_path, _manifest())
    assert result["passed"] is True
    assert result["protected_games_instantiated"] == 0
    assert result["protected_frames_read"] == 0
    assert result["physical_actions"] == 0
    assert result["holdout_opened"] is False
    assert writes[0][0] == runtime.PREFLIGHT_FILENAME


def test_synthetic_transition_compiles_without_opening_a_game() -> None:
    selected = SimpleNamespace(name="ACTION6", action_args={"x": 1, "y": 1})
    before = SimpleNamespace(
        grid=np.asarray([[0, 1], [0, 0]], dtype=np.int32),
        levels_completed=0,
        game_state="NOT_FINISHED",
    )
    after = SimpleNamespace(
        grid=np.asarray([[0, 0], [0, 1]], dtype=np.int32),
        levels_completed=0,
        game_state="NOT_FINISHED",
    )
    observed, outcome = runtime._compile_transition(
        before=before,
        after=after,
        selected=selected,
        legal_before=(selected,),
        legal_after=(selected,),
    )
    assert observed.action.action_name == "ACTION6"
    assert 0.0 <= outcome.quality <= 1.0
    assert "frame" not in outcome.safe_payload


def test_one_action_is_sealed_before_observe_and_never_replayed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    legal = SimpleNamespace(name="ACTION6", action_args={"x": 1, "y": 1})
    selected = SimpleNamespace(name="ACTION6", action_args={"x": 1, "y": 1})
    before = SimpleNamespace(
        grid=np.asarray([[0, 1], [0, 0]], dtype=np.int32),
        levels_completed=0,
        game_state="NOT_FINISHED",
        available_actions=("ACTION6",),
    )
    after = SimpleNamespace(
        grid=np.asarray([[0, 0], [0, 1]], dtype=np.int32),
        levels_completed=1,
        game_state="WIN",
        available_actions=("ACTION6",),
    )

    class FakeDecision:
        candidate = legal
        abstained = False
        safe_payload = {
            "reason": "test",
            "phase": "IDENTIFY",
            "predicted_family": "stable_repeat",
            "predicted_probability": 0.5,
            "intervention": {
                "action_family": "spatial_operator",
                "target_role": "unbound_role",
                "argument_schema": "point_binding",
            },
            "program_hash": "program",
            "candidates_inspected": 1,
            "abstained": False,
        }

    class FakeOutcome:
        level_delta = 1
        game_over = False
        noop = False
        safe_payload = {
            "mode": "level_progress",
            "level_delta": 1,
            "game_over": False,
            "noop": False,
            "quality": 1.0,
        }

    class FakeUpdate:
        predicted_probability = 0.8
        safe_payload = {
            "phase_before": "IDENTIFY",
            "phase_after": "ABSTAIN",
            "predicted_family": "stable_repeat",
            "predicted_probability": 0.8,
            "outcome": FakeOutcome.safe_payload,
            "posterior": {
                "stable_repeat": 0.7,
                "relational_successor": 0.1,
                "state_conditioned_switch": 0.1,
                "null_or_unsafe": 0.1,
            },
            "mismatch": False,
            "revised": False,
            "abstained": True,
            "reason": "level_progress_observed",
        }

    event_path: Path | None = None

    class FakeController:
        def __init__(self, _arm: str, *, scope: int, prior: Any) -> None:
            del scope, prior

        def propose(self, *_args: Any, **_kwargs: Any) -> FakeDecision:
            order.append("propose")
            return FakeDecision()

        def observe(self, *_args: Any, **_kwargs: Any) -> FakeUpdate:
            order.append("observe")
            assert event_path is not None and event_path.is_file()
            return FakeUpdate()

        def summary(self) -> dict[str, Any]:
            return {
                "arm": "source_closed_loop",
                "actions": 1,
                "program_hash": "program",
                "grounded_payload_persisted": False,
                "legacy_fallback_actions": 0,
            }

    class FakeLock:
        def heartbeat(self) -> None:
            return None

    monkeypatch.setattr(
        runtime,
        "_causal_api",
        lambda: (object, FakeController, (FakeOutcome, lambda _prior: {"passed": True})),
    )
    monkeypatch.setattr(runtime, "_make_environment", lambda *_args: object())
    monkeypatch.setattr(runtime, "_reset_environment", lambda _env: before)
    monkeypatch.setattr(runtime, "_snapshot", lambda frame, **_kwargs: frame)
    monkeypatch.setattr(runtime, "_legal_actions", lambda _env: (legal,))
    monkeypatch.setattr(runtime, "_compile_state", lambda *_args: object())
    monkeypatch.setattr(runtime, "_materialize", lambda *_args: selected)

    def step(_environment: Any, _selected: Any) -> Any:
        order.append("env")
        return after

    monkeypatch.setattr(runtime, "_step_environment", step)
    monkeypatch.setattr(
        runtime,
        "_compile_transition",
        lambda **_kwargs: (object(), FakeOutcome()),
    )
    monkeypatch.setattr(runtime, "_close_environment", lambda _env: None)
    original_write = protocol.write_json_once

    def traced_write(path: Path, payload: dict[str, Any]) -> None:
        nonlocal event_path
        parts = set(path.parts)
        if "intents" in parts:
            order.append("intent")
        elif "events" in parts:
            order.append("event")
            event_path = path
        elif "updates" in parts:
            order.append("update")
        original_write(path, payload)

    monkeypatch.setattr(protocol, "write_json_once", traced_write)
    work = protocol.WorkSpec(
        phase="active-confirmation",
        game_id="s5i5",
        arm="source_closed_loop",
        role="candidate",
        reset_index=0,
        action_budget=48,
    )
    receipt = runtime._run_work(
        tmp_path,
        tmp_path / protocol.DEFAULT_OUTPUT_DIR,
        _manifest(),
        work,
        object(),
        "prior-12f",
        FakeLock(),
    )
    assert receipt["complete"] is True
    assert receipt["level_delta"] == 1
    assert receipt["first_level_action"] == 1
    assert receipt["utility"] == 1.0
    assert receipt["causal_proposals"] == 1
    assert receipt["causal_observations"] == 1
    assert receipt["sealed_events"] == 1
    assert receipt["raw_frames_persisted"] is False
    assert receipt["grounded_arguments_persisted"] is False
    assert order.index("propose") < order.index("intent")
    assert order.index("intent") < order.index("env")
    assert order.index("env") < order.index("event")
    assert order.index("event") < order.index("observe")
    assert order.index("observe") < order.index("update")

    monkeypatch.setattr(
        runtime,
        "_make_environment",
        lambda *_args: pytest.fail("sealed work was physically replayed"),
    )
    replay = runtime._run_work(
        tmp_path,
        tmp_path / protocol.DEFAULT_OUTPUT_DIR,
        _manifest(),
        work,
        object(),
        "prior-12f",
        FakeLock(),
    )
    assert replay["receipt_checksum"] == receipt["receipt_checksum"]


def _prospective_receipts() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, game in enumerate(protocol.PROTECTED_GAMES):
        candidate_success = index < 4
        control_success = index == 0
        rows.extend(
            [
                {
                    "manifest_checksum": "manifest-13",
                    "authorization_checksum": "authorization-13",
                    "game_id": game,
                    "role": "candidate",
                    "arm": "source_closed_loop",
                    "level_delta": int(candidate_success),
                    "utility": 0.8 if candidate_success else 0.0,
                    "prequential_log_loss": 0.2 if index < 4 else 0.4,
                    "errors": [],
                    "illegal_actions": 0,
                    "legacy_fallback_actions": 0,
                    "physical_actions_replayed": 0,
                },
                {
                    "manifest_checksum": "manifest-13",
                    "authorization_checksum": "authorization-13",
                    "game_id": game,
                    "role": "control",
                    "arm": "uniform_closed_loop",
                    "level_delta": int(control_success),
                    "utility": 0.5 if control_success else 0.0,
                    "prequential_log_loss": 0.5 if index < 4 else 0.4,
                    "errors": [],
                    "illegal_actions": 0,
                    "legacy_fallback_actions": 0,
                    "physical_actions_replayed": 0,
                },
            ]
        )
    return rows


def test_adjudication_enforces_all_five_prospective_outcome_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = {
        "collection_complete": True,
        "report_checksum": "active",
        "metrics": {"initial_hashes_match_within_pairs": True},
    }
    writes: list[dict[str, Any]] = []
    monkeypatch.setattr(runtime, "_require_gate", lambda *_args: active)
    monkeypatch.setattr(runtime, "_load_receipts", lambda _destination: _prospective_receipts())
    monkeypatch.setattr(runtime, "_write", lambda _r, _name, payload: writes.append(payload))
    result = runtime.adjudicate(tmp_path, _manifest())
    assert result["passed"] is True
    assert result["verdict"] == "PASS_PROSPECTIVE_SOURCE_INFORMED_CAUSAL_PROCEDURE"
    assert len(result["candidate_success_games"]) == 4
    assert result["net_success_advantage"] == 3
    assert len(result["higher_utility_games"]) == 4
    assert result["lower_utility_games"] == []
    assert len(result["better_log_loss_games"]) == 4
    assert all(result["checks"].values())
    assert writes == [result]


def test_log_loss_gate_is_a_scientific_miss_not_a_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _prospective_receipts()
    for row in rows:
        if row["role"] == "candidate":
            row["prequential_log_loss"] = 0.9
    active = {
        "collection_complete": True,
        "report_checksum": "active",
        "metrics": {"initial_hashes_match_within_pairs": True},
    }
    monkeypatch.setattr(runtime, "_require_gate", lambda *_args: active)
    monkeypatch.setattr(runtime, "_load_receipts", lambda _destination: rows)
    monkeypatch.setattr(runtime, "_write", lambda *_args: None)
    result = runtime.adjudicate(tmp_path, _manifest())
    assert result["passed"] is False
    assert result["checks"]["better_log_loss_on_at_least_four_games"] is False
    assert result["verdict"] == "PROSPECTIVE_CAUSAL_PREDICTION_GATE_MISS"


def test_main_returns_code_three_only_for_adjudicated_scientific_miss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(protocol, "load_manifest", lambda _root: _manifest())
    monkeypatch.setattr(
        runtime,
        "adjudicate",
        lambda *_args: {"passed": False, "verdict": "PROSPECTIVE_SUCCESS_GATE_MISS"},
    )
    code = runtime.main(["adjudicate", "--repo-root", str(tmp_path)])
    lines = capsys.readouterr().out.splitlines()
    assert code == 3
    assert len(lines) == 1
    assert json.loads(lines[0])["verdict"] == "PROSPECTIVE_SUCCESS_GATE_MISS"


def test_report_cannot_seal_an_incomplete_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "_accounting", lambda _root: _empty_accounting())
    with pytest.raises(protocol.ScientificGateMiss, match="incomplete"):
        runtime.terminal_report(tmp_path, _manifest())
    assert not (
        tmp_path / protocol.DEFAULT_OUTPUT_DIR / runtime.TERMINAL_REPORT_FILENAME
    ).exists()
