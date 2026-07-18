import copy
import json

import pytest

import theory.sage.second_unknown_game_transfer as transfer


class FakeAction:
    def __init__(self, name, data=None):
        self.id = name
        self.data = dict(data or {})


class FakeGame:
    def _get_valid_actions(self):
        return [
            FakeAction("ACTION3"),
            FakeAction("ACTION4"),
            FakeAction("ACTION7"),
            FakeAction("ACTION6", {"x": 2, "y": 2}),
            FakeAction("ACTION6", {"x": 8, "y": 8}),
        ]


class FakeFrame:
    def __init__(self, grid, step=0):
        self.frame = grid
        self.available_actions = ["ACTION3", "ACTION4", "ACTION6", "ACTION7"]
        self.state = "NOT_FINISHED"
        self.levels_completed = 0
        self.step = step


class FakeEnv:
    def __init__(self):
        self._game = FakeGame()
        self.grid = [[0] * 12 for _ in range(12)]
        self.step_count = 0

    def step(self, action, data=None):
        name = str(getattr(action, "name", action)).split(".")[-1].upper()
        if name == "RESET":
            self.grid = [[0] * 12 for _ in range(12)]
            self.step_count = 0
            return FakeFrame([list(row) for row in self.grid])
        self.step_count += 1
        if name == "ACTION3":
            self.grid[0][0] = self.step_count % 10
        elif name == "ACTION4":
            self.grid[0][1] = self.step_count % 10
        elif name == "ACTION7":
            self.grid[0][2] = self.step_count % 10
        elif name == "ACTION6":
            x = int((data or {}).get("x", 0)) % 12
            y = int((data or {}).get("y", 0)) % 12
            self.grid[y][x] = self.step_count % 10
        self.grid[11][11] = self.step_count % 10
        return FakeFrame([list(row) for row in self.grid], self.step_count)


@pytest.fixture(scope="module")
def real_source_sage5():
    return json.loads(
        transfer.DEFAULT_SAGE5_UNKNOWN_GAME_RESULTS_PATH.read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def real_source_a33_2():
    return json.loads(
        transfer.DEFAULT_A33_SCOPED_UNKNOWN_GAME_REGISTRY_OUTPUT_PATH.read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture(scope="module")
def real_payload():
    return json.loads(
        transfer.DEFAULT_SAGE6_SECOND_UNKNOWN_GAME_TRANSFER_PATH.read_text(
            encoding="utf-8"
        )
    )


def test_sage6_selects_wa30_by_fixed_order_before_execution(real_payload):
    audit = real_payload["selection_audit"]

    assert [row["short_game_id"] for row in audit] == [
        "wa30",
        "tn36",
        "ft09",
        "cn04",
        "sb26",
    ]
    assert audit[0]["eligibility_status"] == transfer.ELIGIBLE_UNKNOWN_GAME
    assert audit[0]["eligible_rank"] == 1
    assert audit[0]["selected"] is True
    assert audit[1]["eligibility_status"] == transfer.ELIGIBLE_UNKNOWN_GAME
    assert audit[1]["selected"] is False
    assert audit[2]["eligibility_status"] == transfer.EXCLUDED_KNOWN_GAME
    assert audit[3]["eligibility_status"] == transfer.EXCLUDED_KNOWN_GAME
    assert audit[4]["eligibility_status"] == transfer.EXCLUDED_SOURCE_GAME
    assert all(row["outcome_metrics_read_for_selection"] is False for row in audit)

    selected = real_payload["selected_second_unknown_game"]
    assert selected["game_id"] == "wa30-ee6fef47"
    assert selected["unknown_game"] is True
    assert selected["no_human_trace_for_game"] is True
    assert selected["no_m2_arc_lewm_trace_for_game"] is True
    assert selected["no_game_specific_prior"] is True
    assert selected["selected_before_execution"] is True
    assert selected["selected_from_outcome_metrics"] is False
    assert selected["truth_status"] == transfer.SAGE6_TRUTH_STATUS


def test_sage6_real_second_game_passes_all_bounded_gates(real_payload):
    summary = real_payload["summary"]

    assert summary["source_game_id"] == "sb26-7fbdac44"
    assert summary["selected_second_game_id"] == "wa30-ee6fef47"
    assert summary["candidate_games_audited"] == 5
    assert summary["eligible_unknown_games"] == 2
    assert summary["budgets_evaluated"] == [50, 150, 300]
    assert summary["budgets_gate_passed"] == 3
    assert summary["budgets_total"] == 3
    assert summary["all_budgets_gate_passed"] is True
    assert summary["subgoal_switches_total"] == 98
    assert summary["budgets_with_progress_stall_detected"] == [50, 150, 300]
    assert summary["max_terminal_rate"] == 0.005
    assert summary["max_repeated_action_arg_rate"] == 0.5
    assert summary["min_unique_state_signatures"] == 37
    assert summary["max_unique_state_signatures"] == 132
    assert summary["levels_completed_max"] == 0
    assert summary["gate_passed"] is True
    assert summary["outcome_status"] == transfer.SAGE6_ALL_BUDGETS_PASSED
    assert all(real_payload["gate"].values())


def test_sage6_quarantines_the_sb26_scoped_registry(real_payload):
    guard = real_payload["cross_game_transfer_guard"]

    assert guard["source_game_id"] == "sb26-7fbdac44"
    assert guard["selected_game_id"] == "wa30-ee6fef47"
    assert guard["different_game"] is True
    assert guard["scope_locked_registry_entries_available"] == 1
    assert guard["scope_locked_registry_game_ids"] == ["sb26-7fbdac44"]
    assert guard["selected_game_outside_registry_scopes"] is True
    assert guard["source_scoped_mechanics_reused"] == 0
    assert guard["cross_game_mechanics_imported"] == 0
    assert guard["source_action5_prior_applied"] is False
    assert guard["source_action6_candidate_applied"] is False
    assert guard["scope_generalization_performed"] is False
    assert guard["registry_read_only"] is True
    assert guard["quarantine_passed"] is True


def test_sage6_remains_candidate_only_without_scientific_write(real_payload):
    assert real_payload["status"] == "UNRESOLVED"
    assert real_payload["outcome_status"] == transfer.SAGE6_ALL_BUDGETS_PASSED
    assert real_payload["outcome_status_is_candidate_only"] is True
    assert real_payload["truth_status"] == transfer.SAGE6_TRUTH_STATUS
    assert real_payload["revision_status"] == "CANDIDATE_ONLY"
    assert real_payload["support"] == 0
    assert real_payload["revision_performed"] is False
    assert real_payload["confirmation_performed"] is False
    assert real_payload["refutation_performed"] is False
    assert real_payload["policy_result_counted_as_confirmation"] is False
    assert real_payload["source_scoped_mechanics_reused"] == 0
    assert real_payload["cross_game_mechanics_imported"] == 0
    assert real_payload["scope_generalization_performed"] is False
    assert real_payload["a32_write_performed"] is False
    assert real_payload["a33_write_performed"] is False
    assert real_payload["wrong_confirmations"] == 0


def test_sage6_runner_uses_first_eligible_game_with_fake_environment(tmp_path):
    out = tmp_path / "sage6.json"

    payload = transfer.run_sage6_second_unknown_game_transfer(
        output_path=out,
        candidate_game_ids=("fake-second", "fake-third"),
        budgets=(10, 20),
        env_factory=lambda game_id: FakeEnv(),
    )

    assert out.exists()
    assert payload["selected_second_unknown_game"]["game_id"] == "fake-second"
    assert payload["selection_audit"][0]["selected"] is True
    assert payload["selection_audit"][1]["selected"] is False
    assert payload["summary"]["budgets_evaluated"] == [10, 20]
    assert payload["summary"]["budgets_gate_passed"] == 2
    assert payload["outcome_status"] == transfer.SAGE6_ALL_BUDGETS_PASSED
    assert payload["support"] == 0


def test_second_game_selection_excludes_source_scope_and_known_games(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "known-20260718.steps.jsonl").write_text("{}\n", encoding="utf-8")

    audit, selected = transfer.select_second_unknown_game(
        source_game_id="source-11111111",
        candidate_game_ids=(
            "source-11111111",
            "scoped-22222222",
            "known-33333333",
            "eligible-44444444",
        ),
        registry_game_ids=("scoped-22222222",),
        human_traces_dir=traces,
        m2_dataset_manifest={},
    )

    assert selected == "eligible-44444444"
    assert [row["eligibility_status"] for row in audit] == [
        transfer.EXCLUDED_SOURCE_GAME,
        transfer.EXCLUDED_SCOPE_LOCKED_REGISTRY_GAME,
        transfer.EXCLUDED_KNOWN_GAME,
        transfer.ELIGIBLE_UNKNOWN_GAME,
    ]


def test_second_game_selection_fails_when_no_candidate_is_eligible(tmp_path):
    with pytest.raises(ValueError, match="no eligible second unknown game"):
        transfer.select_second_unknown_game(
            source_game_id="source-11111111",
            candidate_game_ids=("source-11111111", "known-22222222"),
            registry_game_ids=(),
            human_traces_dir=tmp_path,
            m2_dataset_manifest={"per_game_counts": {"known-22222222": 3}},
        )


def test_sage6_rejects_mutated_sage5_scientific_state(real_source_sage5):
    source = copy.deepcopy(real_source_sage5)
    source["support"] = 1
    with pytest.raises(ValueError, match="support must remain 0"):
        transfer.validate_sage5_transfer_source(source)

    source = copy.deepcopy(real_source_sage5)
    source["a33_write_performed"] = True
    with pytest.raises(ValueError, match="cannot write A32/A33"):
        transfer.validate_sage5_transfer_source(source)


def test_sage6_rejects_unlocked_or_generalized_a33_scope(
    real_source_sage5,
    real_source_a33_2,
):
    source = copy.deepcopy(real_source_a33_2)
    source["scope_generalization_performed"] = True
    with pytest.raises(ValueError, match="cannot generalize"):
        transfer.validate_a33_2_transfer_source(
            source,
            source_sage5=real_source_sage5,
        )

    source = copy.deepcopy(real_source_a33_2)
    source["scoped_confirmed_mechanics"][0]["scope_contexts_locked"] = False
    with pytest.raises(ValueError, match="scope must remain fully locked"):
        transfer.validate_a33_2_transfer_source(
            source,
            source_sage5=real_source_sage5,
        )


def test_sage6_writer_round_trips_payload(tmp_path, real_payload):
    path = tmp_path / "round_trip.json"

    transfer.write_sage6_second_unknown_game_transfer(real_payload, path)

    assert json.loads(path.read_text(encoding="utf-8")) == real_payload


def test_sage_package_exports_second_unknown_game_api():
    import theory.sage as sage

    assert (
        sage.DEFAULT_SAGE6_SECOND_UNKNOWN_GAME_TRANSFER_PATH
        == transfer.DEFAULT_SAGE6_SECOND_UNKNOWN_GAME_TRANSFER_PATH
    )
    assert (
        sage.run_sage6_second_unknown_game_transfer
        is transfer.run_sage6_second_unknown_game_transfer
    )
