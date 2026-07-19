import copy
import json

import pytest

import theory.sage.third_unknown_game_transfer as transfer


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
def real_source_sage6():
    return json.loads(
        transfer.DEFAULT_SAGE6_SECOND_UNKNOWN_GAME_TRANSFER_PATH.read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture(scope="module")
def real_source_a33_2():
    return json.loads(
        transfer.DEFAULT_A33_SCOPED_UNKNOWN_GAME_REGISTRY_OUTPUT_PATH.read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture(scope="module")
def real_source_a33_3():
    return json.loads(
        transfer.DEFAULT_A33_CONTROL_DEPENDENT_RELATIONAL_REGISTRY_OUTPUT_PATH.read_text(
            encoding="utf-8"
        )
    )


@pytest.fixture(scope="module")
def real_payload():
    return json.loads(
        transfer.DEFAULT_SAGE7_THIRD_UNKNOWN_GAME_TRANSFER_PATH.read_text(
            encoding="utf-8"
        )
    )


def test_sage7_selects_tn36_by_fixed_order_before_execution(real_payload):
    audit = real_payload["selection_audit"]

    assert [row["short_game_id"] for row in audit] == [
        "wa30",
        "tn36",
        "ft09",
        "cn04",
        "sb26",
    ]
    assert audit[0]["eligibility_status"] == transfer.EXCLUDED_PRIOR_UNKNOWN_GAME
    assert audit[1]["eligibility_status"] == transfer.ELIGIBLE_UNKNOWN_GAME
    assert audit[1]["eligible_rank"] == 1
    assert audit[1]["selected"] is True
    assert audit[2]["eligibility_status"] == transfer.EXCLUDED_KNOWN_GAME
    assert audit[3]["eligibility_status"] == transfer.EXCLUDED_KNOWN_GAME
    assert audit[4]["eligibility_status"] == transfer.EXCLUDED_PRIOR_UNKNOWN_GAME
    assert all(row["outcome_metrics_read_for_selection"] is False for row in audit)

    selected = real_payload["selected_third_unknown_game"]
    assert selected["game_id"] == "tn36-ab4f63cc"
    assert selected["unknown_game"] is True
    assert selected["no_human_trace_for_game"] is True
    assert selected["no_m2_arc_lewm_trace_for_game"] is True
    assert selected["no_game_specific_prior"] is True
    assert selected["selected_before_execution"] is True
    assert selected["selected_from_outcome_metrics"] is False
    assert selected["truth_status"] == transfer.SAGE7_TRUTH_STATUS


def test_sage7_real_third_game_passes_all_bounded_gates(real_payload):
    summary = real_payload["summary"]

    assert summary["first_unknown_game_id"] == "sb26-7fbdac44"
    assert summary["second_unknown_game_id"] == "wa30-ee6fef47"
    assert summary["selected_third_game_id"] == "tn36-ab4f63cc"
    assert summary["candidate_games_audited"] == 5
    assert summary["eligible_unknown_games"] == 1
    assert summary["budgets_evaluated"] == [50, 150, 300]
    assert summary["budgets_gate_passed"] == 3
    assert summary["budgets_total"] == 3
    assert summary["all_budgets_gate_passed"] is True
    assert summary["env_steps_total"] == 172
    assert summary["subgoal_switches_total"] == 172
    assert summary["new_candidate_targets_discovered_total"] == 27
    assert summary["rerun_m2_m3_requested_total"] == 145
    assert summary["rerun_m2_m3_effective_requests_generated_total"] == 0
    assert summary["budgets_with_progress_stall_detected"] == []
    assert summary["max_terminal_rate"] == 0.016393
    assert summary["max_repeated_action_arg_rate"] == 0.0
    assert summary["min_unique_state_signatures"] == 51
    assert summary["max_unique_state_signatures"] == 62
    assert summary["levels_completed_max"] == 0
    assert summary["gate_passed"] is True
    assert summary["outcome_status"] == transfer.SAGE7_ALL_BUDGETS_PASSED
    assert all(real_payload["gate"].values())


def test_sage7_audits_single_action6_parameterized_surface(real_payload):
    surface = real_payload["action_surface_audit"]
    summary = real_payload["summary"]

    assert surface["action_families"] == ["ACTION6"]
    assert surface["distinct_action_families"] == 1
    assert surface["legal_action_options_count"] == 11
    assert surface["parameterized_action_options_count"] == 11
    assert surface["single_action_family_only"] is True
    assert surface["parameterized_control_design_required"] is True
    assert surface["parameterized_action_variants_counted_as_distinct_actions"] is False
    assert all(row["action"] == "ACTION6" for row in surface["legal_action_options"])
    assert summary["ready_for_parameterized_mini_frontier"] is True
    assert summary["required_next_step"] == transfer.SAGE7_PARAMETERIZED_FRONTIER_REQUIRED


def test_sage7_quarantines_both_scoped_registries(real_payload):
    guard = real_payload["cross_game_transfer_guard"]

    assert guard["prior_unknown_game_ids"] == [
        "sb26-7fbdac44",
        "wa30-ee6fef47",
    ]
    assert guard["selected_game_id"] == "tn36-ab4f63cc"
    assert guard["different_from_all_prior_unknown_games"] is True
    assert guard["a33_2_scope_locked_entries_available"] == 1
    assert guard["a33_3_relational_entries_available"] == 1
    assert guard["a33_2_registry_game_ids"] == ["sb26-7fbdac44"]
    assert guard["a33_3_registry_game_ids"] == ["wa30-ee6fef47"]
    assert guard["selected_game_outside_registry_scopes"] is True
    assert guard["source_scoped_mechanics_reused"] == 0
    assert guard["cross_game_mechanics_imported"] == 0
    assert guard["sb26_action5_prior_applied"] is False
    assert guard["wa30_action2_relational_contrast_applied"] is False
    assert guard["wa30_action1_universal_baseline_applied"] is False
    assert guard["wa30_standalone_action2_effect_applied"] is False
    assert guard["scope_generalization_performed"] is False
    assert guard["a33_2_registry_read_only"] is True
    assert guard["a33_3_registry_read_only"] is True
    assert guard["quarantine_passed"] is True


def test_sage7_remains_candidate_only_without_scientific_write(real_payload):
    assert real_payload["status"] == "UNRESOLVED"
    assert real_payload["outcome_status"] == transfer.SAGE7_ALL_BUDGETS_PASSED
    assert real_payload["outcome_status_is_candidate_only"] is True
    assert real_payload["truth_status"] == transfer.SAGE7_TRUTH_STATUS
    assert real_payload["revision_status"] == "CANDIDATE_ONLY"
    assert real_payload["support"] == 0
    assert real_payload["execution_performed"] is True
    assert real_payload["revision_performed"] is False
    assert real_payload["confirmation_performed"] is False
    assert real_payload["refutation_performed"] is False
    assert real_payload["policy_result_counted_as_confirmation"] is False
    assert real_payload["source_scoped_mechanics_reused"] == 0
    assert real_payload["cross_game_mechanics_imported"] == 0
    assert real_payload["scope_generalization_performed"] is False
    assert real_payload["parameterized_action_variants_counted_as_distinct_actions"] is False
    assert real_payload["a32_intake_requested"] is False
    assert real_payload["a32_write_performed"] is False
    assert real_payload["a33_write_performed"] is False
    assert real_payload["wrong_confirmations"] == 0


def test_sage7_runner_uses_first_unused_eligible_game_with_fake_environment(tmp_path):
    out = tmp_path / "sage7.json"

    payload = transfer.run_sage7_third_unknown_game_transfer(
        output_path=out,
        candidate_game_ids=("fake-third", "fake-fourth"),
        budgets=(10, 20),
        env_factory=lambda game_id: FakeEnv(),
    )

    assert out.exists()
    assert payload["selected_third_unknown_game"]["game_id"] == "fake-third"
    assert payload["selection_audit"][0]["selected"] is True
    assert payload["selection_audit"][1]["selected"] is False
    assert payload["summary"]["budgets_evaluated"] == [10, 20]
    assert payload["summary"]["budgets_gate_passed"] == 2
    assert payload["outcome_status"] == transfer.SAGE7_ALL_BUDGETS_PASSED
    assert payload["support"] == 0


def test_third_game_selection_excludes_prior_scope_and_known_games(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "known-20260719.steps.jsonl").write_text("{}\n", encoding="utf-8")

    audit, selected = transfer.select_third_unknown_game(
        prior_unknown_game_ids=("first-11111111", "second-22222222"),
        candidate_game_ids=(
            "first-11111111",
            "scoped-33333333",
            "known-44444444",
            "eligible-55555555",
        ),
        registry_game_ids=("scoped-33333333",),
        human_traces_dir=traces,
        m2_dataset_manifest={},
    )

    assert selected == "eligible-55555555"
    assert [row["eligibility_status"] for row in audit] == [
        transfer.EXCLUDED_PRIOR_UNKNOWN_GAME,
        transfer.EXCLUDED_SCOPE_LOCKED_REGISTRY_GAME,
        transfer.EXCLUDED_KNOWN_GAME,
        transfer.ELIGIBLE_UNKNOWN_GAME,
    ]


def test_third_game_selection_fails_when_no_candidate_is_eligible(tmp_path):
    with pytest.raises(ValueError, match="no eligible third unknown game"):
        transfer.select_third_unknown_game(
            prior_unknown_game_ids=("first-11111111",),
            candidate_game_ids=("first-11111111", "known-22222222"),
            registry_game_ids=(),
            human_traces_dir=tmp_path,
            m2_dataset_manifest={"per_game_counts": {"known-22222222": 3}},
        )


def test_sage7_rejects_mutated_sage6_scientific_state(real_source_sage6):
    source = copy.deepcopy(real_source_sage6)
    source["support"] = 1
    with pytest.raises(ValueError, match="support must remain 0"):
        transfer.validate_sage6_transfer_source(source)

    source = copy.deepcopy(real_source_sage6)
    source["cross_game_mechanics_imported"] = 1
    with pytest.raises(ValueError, match="cannot import"):
        transfer.validate_sage6_transfer_source(source)


def test_sage7_rejects_unlocked_a33_2_scope(real_source_a33_2):
    source = copy.deepcopy(real_source_a33_2)
    source["scoped_confirmed_mechanics"][0]["scope_contexts_locked"] = False
    with pytest.raises(ValueError, match="sb26 entry scope"):
        transfer.validate_a33_2_quarantine_source(source)


def test_sage7_rejects_generalized_or_unlocked_a33_3_scope(
    real_source_sage6,
    real_source_a33_3,
):
    source = copy.deepcopy(real_source_a33_3)
    source["standalone_action2_effect_registered"] = True
    with pytest.raises(ValueError, match="excluded claims"):
        transfer.validate_a33_3_quarantine_source(
            source,
            source_sage6=real_source_sage6,
        )

    source = copy.deepcopy(real_source_a33_3)
    source["control_dependent_relational_contrasts"][0][
        "scope_contexts_locked"
    ] = False
    with pytest.raises(ValueError, match="scope must remain fully locked"):
        transfer.validate_a33_3_quarantine_source(
            source,
            source_sage6=real_source_sage6,
        )


def test_sage7_writer_round_trips_payload(tmp_path, real_payload):
    path = tmp_path / "round_trip.json"

    transfer.write_sage7_third_unknown_game_transfer(real_payload, path)

    assert json.loads(path.read_text(encoding="utf-8")) == real_payload


def test_sage_package_exports_third_unknown_game_api():
    import theory.sage as sage

    assert (
        sage.DEFAULT_SAGE7_THIRD_UNKNOWN_GAME_TRANSFER_PATH
        == transfer.DEFAULT_SAGE7_THIRD_UNKNOWN_GAME_TRANSFER_PATH
    )
    assert (
        sage.run_sage7_third_unknown_game_transfer
        is transfer.run_sage7_third_unknown_game_transfer
    )
