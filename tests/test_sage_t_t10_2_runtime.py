from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from theory.sage12.action_target_data import grid_sha256
from theory.sage_t import t10_2_protocol as protocol
from theory.sage_t import t10_2_runtime as runtime_adapter
from theory.sage_t.contracts import (
    AbstractEntity,
    AbstractState,
    ActionCandidate,
    GroundFact,
    ObservedTransition,
    PredictionPacket,
)
from theory.sage_t.frame_adapters_v10_2 import (
    project_transition_with_frozen_frames,
)
from theory.sage_t.frame_transport_v10_2 import TransportMap, TransportOrbitWitness
from theory.sage_t.gauge_inference_v10_2 import (
    JointGaugeHypothesis,
    rank_option_sequence_signatures,
)
from theory.sage_t.mixed_automata_v10_2 import alternate, repeat, until_then
from theory.sage_t.observer_frames_v10_2 import (
    OBSERVER_FRAME_SPECS,
    FrameProjection,
    PhysicalEventBundle,
    ProjectedTransition,
)
from theory.sage_t.progress_witness_v10 import compile_progress_program
from theory.sage_t.t10_2_runtime import (
    T10_1BehaviorFrozenPolicyFactory,
    T10_2GaugePolicyFactory,
    T10_2SourceFactory,
    T10_2ValidationFactory,
    build_v4_3_replay_ledger,
    run_source_trainer,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_runtime_canonical_json_refuses_nonfinite_numbers(value: float) -> None:
    with pytest.raises(ValueError):
        runtime_adapter._canonical_json({"value": value})


@dataclass(frozen=True)
class FakeAction:
    name: str
    action_args: dict[str, int]


@dataclass(frozen=True)
class FakeSnapshot:
    levels_completed: int = 0
    game_state: str = "NOT_FINISHED"


class FakeRuntime:
    def __init__(self, *, progress: bool = True) -> None:
        self.progress = progress
        self.opens: list[tuple[str, int]] = []
        self.closed = 0

    def open(self, game_id: str, seed: int) -> dict[str, Any]:
        self.opens.append((game_id, seed))
        return {"level": 0, "step": 0}

    @staticmethod
    def reset(environment: dict[str, Any]) -> FakeSnapshot:
        environment["level"] = 0
        environment["step"] = 0
        return FakeSnapshot()

    @staticmethod
    def legal_actions(_environment: dict[str, Any]) -> tuple[FakeAction, ...]:
        return (
            FakeAction("ACTION1", {}),
            FakeAction("ACTION6", {"x": 3, "y": 8}),
        )

    def step(self, environment: dict[str, Any], _action: FakeAction) -> FakeSnapshot:
        environment["step"] += 1
        if self.progress:
            environment["level"] += 1
        return FakeSnapshot(levels_completed=environment["level"])

    @staticmethod
    def snapshot(frame: FakeSnapshot, **_kwargs: Any) -> FakeSnapshot:
        return frame

    def close(self, _environment: dict[str, Any]) -> None:
        self.closed += 1


def _bundle_builder(
    before: FakeSnapshot,
    after: FakeSnapshot,
    action: FakeAction,
    event_id: str,
    **_kwargs: Any,
):
    entity = AbstractEntity(
        "branch-local",
        ("object", "target"),
        attributes=(("kind", "shape"),),
        center=(1.0, 1.0),
    )
    exists = GroundFact("exists", ("branch-local",))
    changed = GroundFact("changed", ("branch-local",))
    state_before = AbstractState(
        entities=(entity,),
        true_facts=frozenset({exists}),
        false_facts=frozenset({changed}),
    )
    progressed = after.levels_completed > before.levels_completed
    state_after = AbstractState(
        entities=(entity,),
        true_facts=frozenset({exists, changed}),
    )
    packet = PredictionPacket(
        object_deltas={"changed": 1.0},
        progress_mean=float(progressed),
        progress_distribution={f"value:{int(progressed)}": 1.0},
        terminal_probability=0.0,
        goal_probability=1.0 if progressed else None,
        known_channels=frozenset(
            {"objects", "progress", "terminal", *(("goal",) if progressed else ())}
        ),
        state_after=state_after,
    )
    evidence = ObservedTransition(
        state_before=state_before,
        action=ActionCandidate(action.name),
        state_after=state_after,
        observation=packet,
        events=(
            "state_change",
            *(("progress", "level_complete") if progressed else ()),
        ),
    )
    return project_transition_with_frozen_frames(evidence, event_id=event_id)


def test_default_bundle_dispatch_excludes_custom_builder_only_keywords(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def strict_default_bundle(
        *,
        before: FakeSnapshot,
        after: FakeSnapshot,
        action: FakeAction,
        legal_actions: tuple[FakeAction, ...],
        event_id: str,
        step_index: int,
        game_id: str,
    ) -> PhysicalEventBundle:
        captured.update(
            {
                "legal_actions": legal_actions,
                "step_index": step_index,
                "game_id": game_id,
            }
        )
        return _bundle_builder(before, after, action, event_id)

    monkeypatch.setattr(
        runtime_adapter,
        "_default_bundle",
        strict_default_bundle,
    )
    action = FakeAction("ACTION1", {})
    bundle = runtime_adapter._make_bundle(
        None,
        before=FakeSnapshot(levels_completed=0),
        after=FakeSnapshot(levels_completed=1),
        action=action,
        legal_actions=(action,),
        event_id="default-bundle-dispatch",
        step_index=3,
        game_id=protocol.SOURCE_GAMES[0],
    )

    assert bundle.event_id == "default-bundle-dispatch"
    assert captured == {
        "legal_actions": (action,),
        "step_index": 3,
        "game_id": protocol.SOURCE_GAMES[0],
    }


def _collect(
    factory: T10_2SourceFactory,
    game: str,
    *,
    seed: int,
    split: str,
    held_out: str | None,
    total_action_budget: int | None = None,
) -> list[dict[str, Any]]:
    donors = tuple(item for item in protocol.SOURCE_GAMES if item != held_out)
    environment = factory(
        game_id=game,
        seed=seed,
        phase="collect",
        split=split,
        held_out_game=held_out,
        training_games=donors,
    )
    try:
        return environment.collect_events(
            game_id=game,
            seed=seed,
            split=split,
            held_out_game=held_out,
            training_games=donors,
            resets=4,
            action_budget=2,
            stop_on_progress=True,
            stop_on_game_over=True,
            total_action_budget=total_action_budget,
        )
    finally:
        environment.close()


def test_source_lane_total_budget_stops_before_the_next_runtime_step() -> None:
    factory = T10_2SourceFactory(
        runtime_loader=lambda: FakeRuntime(progress=False),
        bundle_builder=_bundle_builder,
    )

    rows = _collect(
        factory,
        protocol.SOURCE_GAMES[0],
        seed=0,
        split="discovery",
        held_out=None,
        total_action_budget=3,
    )

    assert len(rows) == 3
    assert [(row["reset_index"], row["step_index"]) for row in rows] == [
        (0, 0),
        (0, 1),
        (1, 0),
    ]


def _walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key).casefold())
            keys.update(_walk_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_walk_keys(item))
    return keys


def test_source_factory_is_lazy_cross_fits_and_persists_only_compact_events() -> None:
    runtime = FakeRuntime()
    loads: list[bool] = []

    def load_runtime() -> FakeRuntime:
        loads.append(True)
        return runtime

    factory = T10_2SourceFactory(
        runtime_loader=load_runtime,
        bundle_builder=_bundle_builder,
    )
    with pytest.raises(protocol.FirewallError):
        factory(
            game_id=protocol.VALIDATION_GAMES[0],
            seed=0,
            phase="collect",
            split="discovery",
        )
    assert loads == []

    held_out = protocol.SOURCE_GAMES[0]
    for donor in protocol.SOURCE_GAMES[1:]:
        rows = _collect(
            factory,
            donor,
            seed=0,
            split="discovery",
            held_out=None,
        )
        assert len(rows) == 4
    confirmation = _collect(
        factory,
        held_out,
        seed=3,
        split="leave_one_game_out_confirmation",
        held_out=held_out,
    )
    assert len(confirmation) == 4
    controllers = [row["selection"]["controller"] for row in confirmation]
    assert controllers.count("learned") == 2
    assert controllers.count("capacity_matched_independent") == 2
    assert all(row["selection"]["donor_game_count"] == 2 for row in confirmation)
    assert (
        len({row["selection"]["posterior_candidate_count"] for row in confirmation})
        == 1
    )
    assert confirmation[0]["selection"]["posterior_candidate_count"] > 0
    assert {
        row["selection"]["cross_fit_model"]
        for row in confirmation
        if row["selection"]["controller"] == "capacity_matched_independent"
    } == {"capacity_matched_factorized_independent_posterior"}
    independent_rows = [
        row
        for row in confirmation
        if row["selection"]["controller"] == "capacity_matched_independent"
    ]
    assert all(
        row["selection"]["posterior_family"] == "strict_five_factor_variational_control"
        and row["selection"]["factorized_bank_capacity_matched"] is True
        and row["selection"]["factorized_mdl_prior_preserved"] is True
        and not row["selection"]["factorized_control_refusal"]
        and row["selection"]["factorized_target_particles"]
        == row["selection"]["posterior_candidate_count"]
        and row["selection"]["factorized_target_classes"]
        == row["selection"]["posterior_class_capacity"]
        for row in independent_rows
    )
    audit_unit = factory.cross_fit_audit[-1]
    assert audit_unit["held_out_prefit_events_used"] == 0
    assert audit_unit["held_out_game"] == held_out
    assert audit_unit["seed"] == 3
    assert len(audit_unit["resets"]) == 4
    assert [row["reset_index"] for row in audit_unit["resets"]] == [0, 1, 2, 3]
    assert (
        sum(
            row["online_observations"]
            for row in audit_unit["resets"]
            if row["controller"] == "learned"
        )
        == 2
    )
    assert (
        sum(
            row["online_observations"]
            for row in audit_unit["resets"]
            if row["controller"] == "capacity_matched_independent"
        )
        == 2
    )
    learned_capacity = {
        (row["initial_particle_count"], row["initial_class_count"])
        for row in audit_unit["resets"]
        if row["controller"] == "learned"
    }
    independent_capacity = {
        (row["initial_particle_count"], row["initial_class_count"])
        for row in audit_unit["resets"]
        if row["controller"] == "capacity_matched_independent"
    }
    assert learned_capacity == independent_capacity
    assert all(row["selection"]["decision_engine_used"] for row in confirmation)
    learned_progress = [
        row
        for row in confirmation
        if row["selection"]["controller"] == "learned" and row["labels"]["progress"]
    ]
    assert learned_progress
    assert all(
        int(row["selection"]["progressing_sequence_rank"]) > 0
        for row in learned_progress
    )
    assert all(
        float(row["selection"]["compatible_option_sequence_mass"]) > 0.0
        for row in learned_progress
    )
    assert all("cross_fit_rank" not in row["selection"] for row in confirmation)

    forbidden = {
        "grid",
        "raw_frame",
        "frame_pixels",
        "color",
        "x",
        "y",
        "entity_id",
        "entities",
        "true_facts",
        "false_facts",
        "nodes",
        "edges",
    }
    for row in confirmation:
        assert set(row["model_view"]["frames"]) == {
            frame.frame_id for frame in OBSERVER_FRAME_SPECS
        }
        assert not (_walk_keys(row) & forbidden)
        assert "outcome" in row
        assert len(row["transport_certificates"]) == 4
        assert row["transport"]["identity_root_certificate_exact"] is True
        assert row["transport"]["exact_certificate_count"] >= 1
    retained = factory.discovery_events
    assert retained[held_out] == ()
    assert all("model_view" in row for game in retained.values() for row in game)


def test_compact_runtime_event_has_four_structural_projections_and_boolean_outcome() -> (
    None
):
    factory = T10_2SourceFactory(
        runtime_loader=lambda: FakeRuntime(progress=False),
        bundle_builder=_bundle_builder,
    )
    rows = _collect(
        factory,
        protocol.SOURCE_GAMES[0],
        seed=0,
        split="discovery",
        held_out=None,
    )
    assert rows
    first = rows[0]
    assert first["outcome"] == {
        "progression": 0.0,
        "terminal": False,
        "goal": False,
    }
    assert set(first["projections"]) == {
        frame.frame_id for frame in OBSERVER_FRAME_SPECS
    }
    assert all(
        {
            "before_hash",
            "after_hash",
            "observation",
            "observation_hash",
        }
        <= set(projection)
        for projection in first["projections"].values()
    )
    for frame in first["model_view"]["frames"].values():
        assert frame["before"]["format_version"] == (
            "sage-t10.2-structural-quotient-v2"
        )
        assert frame["after"]["format_version"] == ("sage-t10.2-structural-quotient-v2")
        assert not (
            _walk_keys(frame)
            & {"entities", "true_facts", "false_facts", "nodes", "edges"}
        )
    assert "projection_hashes" not in first["provenance"]
    assert (
        len(
            json.dumps(
                first["model_view"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        <= runtime_adapter.MAXIMUM_MODEL_VIEW_BYTES
    )
    assert (
        len(
            json.dumps(
                protocol.seal_event(first),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        <= runtime_adapter.MAXIMUM_COMPACT_EVENT_BYTES
    )


def test_posterior_action_scores_follow_option_state_after_shared_prefix() -> None:
    program = compile_progress_program(
        sequence_length=2,
        action_names=("ACTION1", "ACTION6"),
        positive=True,
    )
    alternating = alternate("action1", "action6", maximum_horizon=4)
    repeating = repeat("action1", maximum_horizon=4)

    def after_action1(option: Any) -> Any:
        issued = option.execute_one(
            option.new_execution(),
            {"action1": ActionCandidate("ACTION1")},
        )
        return option.observe(
            issued.execution_after_action,
            issued,
            events=("state_changed",),
        )

    posterior = SimpleNamespace(
        particles=(
            SimpleNamespace(
                probability=0.8,
                option_state=after_action1(alternating),
                hypothesis=SimpleNamespace(
                    world_program=program,
                    option=alternating,
                ),
            ),
            SimpleNamespace(
                probability=0.2,
                option_state=after_action1(repeating),
                hypothesis=SimpleNamespace(
                    world_program=program,
                    option=repeating,
                ),
            ),
        )
    )
    scores = runtime_adapter._posterior_action_scores(posterior)
    assert scores[("ACTION6", 0)] > scores[("ACTION1", 0)]


def test_option_sequence_rank_is_not_primitive_action_rank() -> None:
    program = compile_progress_program(
        sequence_length=2,
        action_names=("ACTION1", "ACTION6"),
        positive=True,
    )
    action1 = ActionCandidate("ACTION1")
    action6 = ActionCandidate("ACTION6")
    first_events = ("no_effect",)
    progressing_events = ("state_changed", "progress", "level_complete")

    def after_first(option: Any) -> Any:
        state = option.new_execution()
        if "action1" not in option.allowed_action_schemas(state):
            return state
        issued = option.execute_one(state, {"action1": action1})
        return option.observe(
            issued.execution_after_action,
            issued,
            events=first_events,
        )

    def particle(option: Any, mass: float) -> SimpleNamespace:
        return SimpleNamespace(
            probability=mass,
            option_state=after_first(option),
            hypothesis=SimpleNamespace(world_program=program, option=option),
        )

    def primitive_rank(posterior: Any, action_name: str) -> int:
        scores = runtime_adapter._posterior_action_scores(posterior)
        selected = scores[(action_name, 0)]
        return 1 + sum(score > selected for score in scores.values())

    prefix = (
        (action1, first_events),
        (action6, progressing_events),
    )

    # The high-mass repeat option likes the latest primitive action but cannot
    # explain ACTION1 earlier in the same reset.  The progressing sequence is
    # therefore second even though ACTION6 itself is ranked first.
    action_first_sequence_second = SimpleNamespace(
        particles=(
            particle(repeat("action6", maximum_horizon=4), 0.70),
            particle(alternate("action1", "action6", maximum_horizon=4), 0.15),
            particle(alternate("action1", "action6", maximum_horizon=4), 0.15),
        ),
        residual_mass=0.0,
    )
    assert primitive_rank(action_first_sequence_second, "ACTION6") == 1
    ranking = rank_option_sequence_signatures(
        action_first_sequence_second,
        prefix,
    )
    assert ranking.best_compatible_rank == 2
    assert ranking.compatible_signature_count == 1
    assert ranking.signature_count == 2
    assert ranking.compatible_posterior_mass == pytest.approx(0.30)

    # Two distinct decoys make ACTION1 the preferred one-step action in the
    # mixture, but neither decoy explains the whole prefix.  Aggregation is by
    # option-sequence signature, so the compatible alternate option is rank 1.
    action_second_sequence_first = SimpleNamespace(
        particles=(
            particle(alternate("action1", "action6", maximum_horizon=4), 0.45),
            particle(repeat("action1", maximum_horizon=4), 0.30),
            particle(
                until_then(
                    "action1",
                    "progress",
                    "action6",
                    maximum_horizon=4,
                ),
                0.25,
            ),
        ),
        residual_mass=0.0,
    )
    assert primitive_rank(action_second_sequence_first, "ACTION6") == 2
    ranking = rank_option_sequence_signatures(
        action_second_sequence_first,
        prefix,
    )
    assert ranking.best_compatible_rank == 1
    assert ranking.compatible_signature_count == 1


def _paired_oracle_fixture() -> tuple[
    tuple[JointGaugeHypothesis, ...],
    tuple[tuple[dict[str, Any], PhysicalEventBundle], ...],
]:
    positive_program = compile_progress_program(
        sequence_length=2,
        action_names=("ACTION1", "ACTION6"),
        positive=True,
    )
    negative_program = compile_progress_program(
        sequence_length=2,
        action_names=("ACTION1", "ACTION6"),
        positive=False,
    )
    frame = OBSERVER_FRAME_SPECS[0]
    candidates = (
        JointGaugeHypothesis(
            positive_program,
            frame,
            (),
            alternate("action1", "action6", maximum_horizon=4),
        ),
        JointGaugeHypothesis(
            negative_program,
            frame,
            (),
            repeat("action1", maximum_horizon=4),
        ),
    )
    first_bundle = _bundle_builder(
        FakeSnapshot(),
        FakeSnapshot(),
        FakeAction("ACTION1", {}),
        "paired-oracle-0",
    )
    second_bundle = _bundle_builder(
        FakeSnapshot(),
        FakeSnapshot(levels_completed=1),
        FakeAction("ACTION6", {}),
        "paired-oracle-1",
    )
    rows = (
        (
            {
                "game_id": protocol.SOURCE_GAMES[0],
                "seed": 0,
                "selection": {"reset_index": 0, "step_index": 0},
                "outcome": {"progression": 0},
            },
            first_bundle,
        ),
        (
            {
                "game_id": protocol.SOURCE_GAMES[0],
                "seed": 0,
                "selection": {"reset_index": 0, "step_index": 1},
                "outcome": {"progression": 1},
            },
            second_bundle,
        ),
    )
    return candidates, rows


@pytest.mark.parametrize("name", ("dynamics_swap", "goal_swap", "option_swap"))
def test_paired_factor_oracle_is_capacity_prior_and_evidence_matched(
    name: str,
) -> None:
    candidates, rows = _paired_oracle_fixture()
    result = runtime_adapter._paired_ablation_result(
        name,
        candidates=candidates,
        rows=rows,
    )

    assert result["attempted"] is True
    assert result["evaluable"] is True
    assert result["passed"] is True
    assert result["degradation"] > 0.0
    assert result["same_evidence"] is True
    assert result["same_observations"] is True
    assert result["same_capacity"] is True
    assert result["same_priors"] is True
    assert result["discriminant_classes"] is True
    assert result["component_multiset_preserved"] is True
    assert result["ablation_contract_preserved"] is True
    reference = result["reference"]
    ablated = result["ablated"]
    assert reference["observations"] == ablated["observations"] == 2
    assert reference["candidate_count"] == ablated["candidate_count"] == 2
    assert reference["positive_candidate_count"] == 1
    assert reference["negative_candidate_count"] == 1
    assert reference["evidence_fingerprint"] == ablated["evidence_fingerprint"]
    assert reference["prior_vector_sha256"] == ablated["prior_vector_sha256"]
    assert reference["positive_prefix_count"] == 1
    assert reference["compatible_positive_prefix_count"] == 1
    if name == "option_swap":
        assert (
            reference["mean_compatible_positive_prefix_mass"]
            > ablated["mean_compatible_positive_prefix_mass"]
        )


def test_swap_permutation_preserves_factor_multiset_and_refuses_fixed_points() -> None:
    candidates, rows = _paired_oracle_fixture()
    pairs, refusal = runtime_adapter._ablation_pairs(candidates, "option_swap")
    assert refusal == ""
    assert len(pairs) == len(candidates)
    reference_tokens = sorted(
        runtime_adapter._stable_hash(
            runtime_adapter._ablation_component(reference, "option_swap")
        )
        for reference, _ in pairs
    )
    altered_tokens = sorted(
        runtime_adapter._stable_hash(
            runtime_adapter._ablation_component(altered, "option_swap")
        )
        for _, altered in pairs
    )
    assert reference_tokens == altered_tokens
    assert all(
        runtime_adapter._ablation_component(reference, "option_swap")
        != runtime_adapter._ablation_component(altered, "option_swap")
        for reference, altered in pairs
    )

    refused = runtime_adapter._paired_ablation_result(
        "frame_swap",
        candidates=candidates,
        rows=rows,
    )
    assert refused["attempted"] is True
    assert refused["evaluable"] is False
    assert refused["passed"] is False
    assert refused["refusal"] == ("no_valid_fixed_point_free_component_permutation")


def test_confirmation_metrics_uses_only_progressing_sequence_rank() -> None:
    cross_fit_audit = {
        "registered_unit_count": len(protocol.SOURCE_GAMES)
        * len(protocol.CONFIRMATION_SEEDS),
        "checks": {"exact_confirmation_units": True},
        "passed": True,
    }
    learned = {
        "game_id": protocol.SOURCE_GAMES[0],
        "seed": 3,
        "split": "leave_one_game_out_confirmation",
        "outcome": {"progression": 1},
        "selection": {
            "controller": "learned",
            "progressing_sequence_rank": 4,
            "cross_fit_rank": 1,
        },
    }
    independent = {
        "game_id": protocol.SOURCE_GAMES[0],
        "seed": 3,
        "split": "leave_one_game_out_confirmation",
        "outcome": {"progression": 0},
        "selection": {"controller": "capacity_matched_independent"},
    }
    metrics = runtime_adapter._confirmation_metrics(
        (learned, independent), cross_fit_audit
    )
    assert metrics["positive_fold_ranks"] == {protocol.SOURCE_GAMES[0]: 4}

    learned["selection"].pop("progressing_sequence_rank")
    metrics = runtime_adapter._confirmation_metrics(
        (learned, independent), cross_fit_audit
    )
    assert metrics["positive_fold_ranks"] == {}


def test_state_changed_predicate_advances_until_option() -> None:
    option = until_then(
        "action1",
        "state_changed",
        "action6",
        maximum_horizon=4,
    )
    issued = option.execute_one(
        option.new_execution(),
        {"action1": ActionCandidate("ACTION1")},
    )
    replanned = option.observe(
        issued.execution_after_action,
        issued,
        events=("state_changed",),
    )
    assert option.allowed_action_schemas(replanned) == ("action6",)


def test_confirmation_observes_then_replans_after_each_learned_action() -> None:
    factory = T10_2SourceFactory(
        runtime_loader=lambda: FakeRuntime(progress=False),
        bundle_builder=_bundle_builder,
    )
    held_out = protocol.SOURCE_GAMES[0]
    for donor in protocol.SOURCE_GAMES[1:]:
        rows = _collect(
            factory,
            donor,
            seed=0,
            split="discovery",
            held_out=None,
        )[:2]
        for row in rows:
            action_name = (
                "ACTION1" if row["selection"]["step_index"] == 0 else "ACTION6"
            )
            row["action"]["name"] = action_name
            row["selection"]["action_name"] = action_name
        factory._discovery_events[donor] = rows

    confirmation = _collect(
        factory,
        held_out,
        seed=3,
        split="leave_one_game_out_confirmation",
        held_out=held_out,
    )
    learned_first_reset = [
        row
        for row in confirmation
        if row["selection"]["controller"] == "learned"
        and row["selection"]["reset_index"] == 0
    ]
    assert len(learned_first_reset) == 2
    assert learned_first_reset[0]["selection"]["action_name"] == "ACTION1"
    assert [
        row["selection"]["online_posterior_observation_index"]
        for row in learned_first_reset
    ] == [1, 2]
    assert all(row["selection"]["decision_engine_used"] for row in learned_first_reset)
    assert learned_first_reset[1]["selection"]["decision_reason"] == "selected"


def test_exact_observed_transport_builds_receipt_bound_orbit_witnesses() -> None:
    action = ActionCandidate("ACTION1")
    entity = AbstractEntity("branch-local", ("object", "target"))
    before_state = AbstractState(
        entities=(entity,),
        true_facts=frozenset({GroundFact("exists", ("branch-local",))}),
    )
    after_state = AbstractState(
        entities=(entity,),
        true_facts=frozenset(
            {
                GroundFact("exists", ("branch-local",)),
                GroundFact("changed", ("branch-local",)),
            }
        ),
    )
    event_id = "stable-orbit-fixture"
    projections = []
    for frame in OBSERVER_FRAME_SPECS:
        before = FrameProjection(
            frame=frame,
            state=before_state,
            action=action,
            stage="before",
            provenance=("exact_structural_fixture",),
        )
        after = FrameProjection(
            frame=frame,
            state=after_state,
            action=action,
            stage="after",
            provenance=("exact_structural_fixture",),
        )
        projections.append(
            ProjectedTransition(event_id=event_id, before=before, after=after)
        )
    bundle = PhysicalEventBundle(
        event_id=event_id,
        action=action,
        common_outcome=PredictionPacket(
            progress_mean=0.0,
            terminal_probability=0.0,
            goal_probability=0.0,
            known_channels=frozenset({"progress", "terminal", "goal"}),
        ),
        projections=tuple(projections),
        events=("state_changed",),
    )
    compact = runtime_adapter._compact_event(
        bundle,
        controller="balanced_discovery",
        reset_index=0,
        step_index=0,
        progressing_sequence_rank=None,
        donor_game_count=0,
        capacity_slots=0,
    )
    compact.update(
        {
            "game_id": protocol.SOURCE_GAMES[0],
            "seed": 0,
            "split": "discovery",
        }
    )
    assert any(
        certificate["target_frame"] != "root_only"
        and certificate["exact"] is True
        and certificate["certifies_gauge_equivalence"] is True
        and certificate["commutativity"]["exact"] is True
        for certificate in compact["transport_certificates"]
    )
    assert len(compact["transport_orbits"]) == 3
    witnesses = runtime_adapter._observed_orbit_witnesses((compact,))
    assert len(witnesses) == 3
    assert all(witness.certification_receipt for witness in witnesses)

    candidates = runtime_adapter._synthesize_gauge_candidates((compact,))
    orbit_candidates = [
        candidate
        for candidate in candidates
        if candidate.transports
        and all(
            isinstance(transport, TransportOrbitWitness)
            for transport in candidate.transports
        )
    ]
    assert orbit_candidates
    # The strict posterior merges frame copies only when the witness also
    # covers every D/G/A symbol required by the synthesized program/option.
    # This tiny fixture attests its observed vocabulary, not those extra
    # learned predicates, so retaining separate gauge keys is fail-closed.
    by_key: dict[str, set[str]] = {}
    for candidate in orbit_candidates:
        by_key.setdefault(candidate.gauge_equivalence_key, set()).add(
            candidate.frame.frame_id
        )
    assert all(len(frames) == 1 for frames in by_key.values())
    posterior = runtime_adapter.GaugeProgramPosterior()
    posterior.seed(orbit_candidates)
    assert all(len(gauge_class.particles) == 1 for gauge_class in posterior.classes)

    raw_candidates = [
        candidate
        for candidate in candidates
        if candidate.transports
        and all(
            isinstance(transport, TransportMap) for transport in candidate.transports
        )
    ]
    assert raw_candidates
    raw_same_program = {}
    for candidate in raw_candidates:
        key = (
            candidate.world_program.canonical_hash,
            candidate.option.canonical_hash,
        )
        raw_same_program.setdefault(key, []).append(candidate)
    assert any(
        len({item.gauge_equivalence_key for item in group})
        == len({item.frame.frame_id for item in group})
        for group in raw_same_program.values()
        if len({item.frame.frame_id for item in group}) >= 2
    )


def _exact_orbit_compact_event() -> dict[str, Any]:
    action = ActionCandidate("ACTION1")
    entity = AbstractEntity("local", ("actor", "object", "target"))
    before_state = AbstractState(
        entities=(entity,),
        true_facts=frozenset({GroundFact("exists", ("local",))}),
        counters=(("step_count", 0.0),),
        topology=(("component_count", 1),),
    )
    after_state = AbstractState(
        entities=(entity,),
        true_facts=frozenset(
            {
                GroundFact("exists", ("local",)),
                GroundFact("changed", ("local",)),
            }
        ),
        counters=(("step_count", 1.0),),
        topology=(("component_count", 1),),
    )
    event_id = "persisted-orbit-fixture"
    projections = []
    for frame in OBSERVER_FRAME_SPECS:
        projections.append(
            ProjectedTransition(
                event_id=event_id,
                before=FrameProjection(
                    frame=frame,
                    state=before_state,
                    action=action,
                    stage="before",
                    provenance=("exact_structural_fixture",),
                ),
                after=FrameProjection(
                    frame=frame,
                    state=after_state,
                    action=action,
                    stage="after",
                    provenance=("exact_structural_fixture",),
                ),
            )
        )
    bundle = PhysicalEventBundle(
        event_id=event_id,
        action=action,
        common_outcome=PredictionPacket(
            progress_mean=0.0,
            terminal_probability=0.0,
            goal_probability=0.0,
            known_channels=frozenset({"progress", "terminal", "goal"}),
        ),
        projections=tuple(projections),
        events=("state_changed",),
    )
    return runtime_adapter._compact_event(
        bundle,
        controller="balanced_discovery",
        reset_index=0,
        step_index=0,
        progressing_sequence_rank=None,
        donor_game_count=0,
        capacity_slots=0,
    )


def test_summary_replay_bundle_has_no_invented_graph_and_is_deterministic() -> None:
    compact = _exact_orbit_compact_event()
    bundle = runtime_adapter._bundle_from_compact_event(compact)

    assert bundle is not None
    assert all(
        isinstance(projection, runtime_adapter.SummaryReplayTransition)
        for projection in bundle.projections
    )
    for projection in bundle.projections:
        assert not projection.complete
        assert projection.missing == ("endpoint_incidence_intentionally_omitted",)
        for state in (projection.before.state, projection.after.state):
            assert state.entities == ()
            assert state.true_facts == frozenset()
            assert state.false_facts == frozenset()
            assert state.registers == ()
        persisted = compact["model_view"]["frames"][projection.frame_id]
        assert (
            runtime_adapter._structural_observation_payload(projection.observation)
            == persisted["observation"]
        )
        assert {"counters", "regime", "topology"} <= set(projection.covered_channels)

    first = runtime_adapter._synthesize_gauge_candidates((compact,))
    second = runtime_adapter._synthesize_gauge_candidates((compact,))
    assert [item.canonical_hash for item in first] == [
        item.canonical_hash for item in second
    ]
    runtime_source = (REPO_ROOT / "theory" / "sage_t" / "t10_2_runtime.py").read_text(
        encoding="utf-8"
    )
    assert "state_from_quotient_payload" not in runtime_source
    assert "AbstractEntity(" not in runtime_source
    assert "GroundFact(" not in runtime_source
    assert '"q0"' not in runtime_source


@pytest.mark.parametrize(
    "tamper",
    ("edge", "domain", "orbit_hash", "receipt", "summary_hash"),
)
def test_persisted_orbit_replay_rejects_tampering(tamper: str) -> None:
    compact = _exact_orbit_compact_event()
    corrupted = copy.deepcopy(compact)
    envelope = corrupted["transport_orbits"][0]
    if tamper == "edge":
        envelope["orbit_payload"]["symbol_edges"][0][0][2] = "same_color"
    elif tamper == "domain":
        envelope["orbit_payload"]["certified_domain"].pop()
    elif tamper == "orbit_hash":
        envelope["orbit_hash"] = "0" * 64
    elif tamper == "receipt":
        envelope["attestation"]["receipt"] = "0" * 64
    else:
        envelope["attestation"]["source_before_summary_hash"] = "f" * 64

    with pytest.raises(protocol.DataGateError):
        runtime_adapter._observed_orbit_witnesses((corrupted,))


def test_runtime_size_guards_reject_model_and_event_overages() -> None:
    with pytest.raises(protocol.DataGateError, match="model_view exceeds"):
        runtime_adapter._assert_compact_event_budget(
            {"model_view": {"padding": "x" * (32 * 1_024)}}
        )
    with pytest.raises(protocol.DataGateError, match="compact event exceeds"):
        runtime_adapter._assert_compact_event_budget(
            {
                "model_view": {},
                "padding": "x" * (48 * 1_024),
            }
        )


def test_decision_paths_never_replace_engine_abstention_with_raw_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AbstainingEngine:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def decide(self, *_args: Any, **_kwargs: Any) -> Any:
            return SimpleNamespace(
                action=None,
                reason="all_vetoed",
                normalized_entropy=0.0,
            )

    monkeypatch.setattr(runtime_adapter, "GaugeDecisionEngine", AbstainingEngine)
    factory = T10_2SourceFactory(
        runtime_loader=lambda: FakeRuntime(),
        bundle_builder=_bundle_builder,
    )
    environment = factory(
        game_id=protocol.SOURCE_GAMES[0],
        seed=0,
        phase="collect",
        split="discovery",
        held_out_game=None,
        training_games=protocol.SOURCE_GAMES,
    )
    legal = (FakeAction("ACTION1", {}),)
    selected, metadata = environment._posterior_decision(
        legal,
        posterior=object(),
        frame_states={},
        fallback=legal[0],
    )
    assert selected is None
    assert metadata["reason"] == "all_vetoed"

    posterior = SimpleNamespace(start_branch=lambda: None)
    policy = runtime_adapter._FrozenGaugeValidationPolicy(
        game_id=protocol.VALIDATION_GAMES[0],
        seed=0,
        runtime=FakeRuntime(),
        posterior=posterior,
        candidate_bank=(),
        bundle_builder=_bundle_builder,
    )
    policy.reset(
        reset_index=0,
        posterior_reset=True,
        learning_enabled=False,
    )
    assert (
        policy.select_action(
            legal_actions=legal,
            environment=object(),
            reset_index=0,
            step_index=0,
            learning_enabled=False,
        )
        is None
    )
    environment.close()


def test_confirmation_refuses_before_both_discovery_donors() -> None:
    factory = T10_2SourceFactory(
        runtime_loader=FakeRuntime,
        bundle_builder=_bundle_builder,
    )
    held_out = protocol.SOURCE_GAMES[0]
    with pytest.raises(protocol.GateRefusalError, match="discovery donors"):
        factory(
            game_id=held_out,
            seed=3,
            phase="collect",
            split="leave_one_game_out_confirmation",
            held_out_game=held_out,
            training_games=protocol.SOURCE_GAMES[1:],
        )
    assert factory.runtime_loaded is False


def _v4_3_row(short_game: str) -> dict[str, Any]:
    digest = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()

    def arm(name: str) -> dict[str, Any]:
        frame_before = [[1, 2]]
        frame_after = [[2, 1]]
        action = {"name": "ACTION6", "action_args": {"x": 9, "y": 4}}
        trace = {
            "anchor": {
                "action_family": "click",
                "actor_relation": "far",
                "actor_relative_direction": "north",
                "col": 9,
                "row": 4,
                "target_object_id": 77,
                "kind": "free_slot",
                "occupied": False,
                "path_status": "open",
            },
            "effects": {
                "ambiguity_reasons": [],
                "labels": {
                    "actor_displaced": False,
                    "target_created": True,
                    "target_moved": False,
                    "target_removed": False,
                },
                "game_over": False,
                "level_complete": False,
                "noop": False,
            },
            "format_version": "sage12-action-target-trace-v3",
            "game_id": short_game,
            "source_split": "source_train",
            "selected_action_name": action["name"],
            "selected_action_data": action["action_args"],
            "available_action_names": ["ACTION6"],
            "frame_before": frame_before,
            "frame_after": frame_after,
            "frame_before_sha256": grid_sha256(frame_before),
            "frame_after_sha256": grid_sha256(frame_after),
            "game_state_before": "NOT_FINISHED",
            "game_state_after": "NOT_FINISHED",
            "levels_completed_before": 0,
            "levels_completed_after": 0,
            "policy_seed": 857,
            "reset_index": 0,
            "step_index": 0,
        }
        trace["trace_digest"] = protocol.canonical_sha256(
            {
                "game_id": short_game,
                "source_split": "source_train",
                "policy_seed": 857,
                "reset_index": 0,
                "step_index": 0,
                "frame_before_sha256": trace["frame_before_sha256"],
                "action_name": action["name"],
                "action_data": action["action_args"],
            }
        )
        return {
            "action": action,
            "arm": name,
            "post_state_sha256": digest(f"post-{name}"),
            "replay_pre_state_sha256": digest("pre"),
            "trace": trace,
        }

    row = {
        "format_version": "sage12-bound-trajectory-v4.3",
        "game_id": short_game,
        "source_split": "source_train",
        "policy_seed": 857,
        "reset_index": 0,
        "root_index": 0,
        "path": "",
        "depth": 0,
        "context": [{} for _ in range(8)],
        "expected_pre_state_sha256": digest("pre"),
        "replay_pre_state_sha256": digest("pre"),
        "left": arm("left"),
        "right": arm("right"),
    }
    row["pair_digest"] = protocol.canonical_sha256(
        {
            "game_id": short_game,
            "source_split": "source_train",
            "policy_seed": 857,
            "reset_index": 0,
            "root_index": 0,
            "path": "",
            "pre": row["expected_pre_state_sha256"],
            "left": row["left"]["trace"]["trace_digest"],
            "right": row["right"]["trace"]["trace_digest"],
        }
    )
    return row


def _replay_fixture(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    descriptors: dict[str, dict[str, str]] = {}
    for game in protocol.SOURCE_GAMES:
        path = tmp_path / f"{game.split('-', 1)[0]}.jsonl"
        path.write_text(
            json.dumps(_v4_3_row(game.split("-", 1)[0])) + "\n", encoding="utf-8"
        )
        paths[game] = path
        descriptors[game] = {"path": path.name, "sha256": protocol.file_sha256(path)}
    manifest = {
        "manifest_checksum": "manifest",
        "environment_sha256": "environment",
        "baseline_commit": protocol.BASELINE_COMMIT,
        "baseline_frozen_code_sha256": dict(protocol.BASELINE_FROZEN_SHA256),
        "frozen_source_shards": descriptors,
        "code_sha256": {
            relative: protocol.file_sha256(
                Path(__file__).resolve().parents[1] / relative
            )
            for relative in (
                "theory/sage_t/contracts.py",
                "theory/sage_t/executor.py",
                "theory/sage_t/posterior.py",
                "theory/sage_t/compiler.py",
                "theory/sage_t/progress_witness_v10.py",
                "theory/sage_t/observer_frames_v10_2.py",
                "theory/sage_t/frame_adapters_v10_2.py",
                "theory/sage_t/frame_transport_v10_2.py",
                "theory/sage_t/mixed_automata_v10_2.py",
                "theory/sage_t/gauge_inference_v10_2.py",
                "theory/sage_t/compact_quotient_v10_2.py",
                "theory/sage_t/t10_2_protocol.py",
                "theory/sage_t/t10_2_runtime.py",
            )
        },
    }
    return manifest, paths


def _write_cross_fit_audit(
    output_dir: Path,
    *,
    manifest: dict[str, Any],
    fresh: list[dict[str, Any]],
    units: Any = None,
) -> dict[str, Any]:
    source_path = output_dir / "source_events.jsonl"
    selected_units = (
        units if units is not None else protocol._fallback_cross_fit_units(fresh)
    )
    audit = protocol.build_cross_fit_audit(
        manifest=manifest,
        source_event_path=source_path,
        source_events=fresh,
        units=selected_units,
        factory_binding={
            "module": "theory.sage_t.t10_2_runtime",
            "class": "T10_2SourceFactory",
            "source_sha256": manifest["code_sha256"]["theory/sage_t/t10_2_runtime.py"],
            "manifest_checksum": manifest["manifest_checksum"],
            "code_bound": True,
        },
    )
    protocol.write_compact_json(
        output_dir / protocol.CROSS_FIT_AUDIT_FILENAME,
        audit,
    )
    return audit


def test_replay_builder_verifies_three_shards_and_strips_frames_graphs_and_coords(
    tmp_path: Path,
) -> None:
    manifest, paths = _replay_fixture(tmp_path)
    output = tmp_path / "replay.jsonl"
    rows = build_v4_3_replay_ledger(
        manifest=manifest,
        shard_paths=paths,
        output_path=output,
    )
    assert len(rows) == 6
    assert output.exists()
    for row in rows:
        assert row["provenance"]["kind"] == "frozen_source_replay"
        assert row["provenance"]["raw_frames_retained"] is False
        assert row["provenance"]["graphs_retained"] is False
        assert (
            row["provenance"]["compiler_sha256"]
            == row["provenance"]["conversion_code_sha256"]["compiler"]
        )
        assert set(row["projections"]) == {
            frame.frame_id for frame in OBSERVER_FRAME_SPECS
        }
        assert set(row["model_view"]["frames"]) == {
            frame.frame_id for frame in OBSERVER_FRAME_SPECS
        }
        replay_bundle = runtime_adapter._bundle_from_compact_event(row)
        assert replay_bundle is not None
        assert len(replay_bundle.projections) == 4
        assert not (
            _walk_keys(row["model_view"])
            & {
                "x",
                "y",
                "row",
                "col",
                "grid",
                "entities",
                "true_facts",
                "false_facts",
                "nodes",
                "edges",
            }
        )
        serialized = json.dumps(row)
        assert '"frame_before":' not in serialized
        assert '"frame_after":' not in serialized

    paths[protocol.SOURCE_GAMES[0]].write_text("{}\n", encoding="utf-8")
    with pytest.raises(protocol.ManifestDriftError, match="shard drifted"):
        build_v4_3_replay_ledger(manifest=manifest, shard_paths=paths)


def _fresh_event(
    manifest: dict[str, Any],
    *,
    event_id: str,
    game: str,
    seed: int,
    split: str,
    controller: str,
    progress: bool,
    reset: int,
) -> dict[str, Any]:
    event = _exact_orbit_compact_event()
    event.update(
        {
            "event_id": event_id,
            "game_id": game,
            "seed": seed,
            "split": split,
            "reset_index": reset,
            "step_index": 0,
            "outcome": {
                "progression": int(progress),
                "terminal": False,
                "goal": progress,
            },
        }
    )
    event["prefix"].update({"nonterminal": not progress, "evaluable": True})
    event["labels"].update(
        {
            "progress": progress,
            "game_over": False,
            "no_effect": not progress,
            "state_changed": progress,
        }
    )
    event["selection"].update(
        {
            "controller": controller,
            "reset_index": reset,
            "step_index": 0,
            "action_name": "ACTION1",
            "progressing_sequence_rank": (
                2 if controller == "learned" and progress else None
            ),
            "legal_grounding": True,
        }
    )
    event["provenance"].update(
        {
            "kind": "fresh_source_trajectory",
            "game_id": game,
            "seed": seed,
            "split": split,
            "manifest_checksum": manifest["manifest_checksum"],
            "environment_sha256": manifest["environment_sha256"],
        }
    )
    return protocol.seal_event(event)


def test_grammar_oracle_uses_best_executed_sequence_per_game_and_exact_gate() -> None:
    def executed_event(
        *,
        event_id: str,
        game: str,
        reset: int,
        step: int,
        progress: bool,
        error: str = "",
        game_over: bool = False,
    ) -> dict[str, Any]:
        return {
            "event_id": event_id,
            "game_id": game,
            "seed": 0,
            "action": {"executed": True, "name": "ACTION1"},
            "outcome": {"progression": int(progress)},
            "labels": {"progress": progress, "game_over": game_over},
            "selection": {
                "reset_index": reset,
                "step_index": step,
                "action_name": "ACTION1",
                "legal_grounding": True,
                "posterior_update_error": error,
            },
        }

    events = [
        executed_event(
            event_id="g1-best-prefix",
            game=protocol.SOURCE_GAMES[0],
            reset=0,
            step=0,
            progress=False,
        ),
        executed_event(
            event_id="g1-best-progress",
            game=protocol.SOURCE_GAMES[0],
            reset=0,
            step=1,
            progress=True,
        ),
        executed_event(
            event_id="g1-worse-error",
            game=protocol.SOURCE_GAMES[0],
            reset=1,
            step=0,
            progress=False,
            error="PosteriorError",
        ),
        executed_event(
            event_id="g2-best-progress",
            game=protocol.SOURCE_GAMES[1],
            reset=0,
            step=0,
            progress=True,
        ),
    ]
    grammar = runtime_adapter._executed_grammar(events)
    assert grammar["oracle_sequences"] == 2
    assert grammar["actions"] == 3
    assert grammar["levels"] == 2
    assert grammar["progress_games"] == 2
    assert grammar["errors"] == 0
    assert grammar["all_executed_errors"] == 1
    best = grammar["best_executed_sequence"]
    assert best["all_actions_executed"] is True
    assert best["steps_contiguous"] is True
    assert best["progress_at_end"] is True
    assert best["oracle_eligible"] is True
    assert best["sequence_hash"] in grammar["executed_sequence_hashes"]
    assert runtime_adapter._grammar_oracle_passes(grammar) is True

    unsafe_events = [dict(event) for event in events]
    unsafe_events[-1] = {
        **unsafe_events[-1],
        "labels": {"progress": True, "game_over": True},
    }
    unsafe = runtime_adapter._executed_grammar(unsafe_events)
    assert unsafe["all_executed_game_overs"] == 1
    assert unsafe["progress_games"] == 1
    assert runtime_adapter._grammar_oracle_passes(unsafe) is False

    gapped_events = copy.deepcopy(events)
    gapped_events[1]["selection"]["step_index"] = 2
    gapped = runtime_adapter._executed_grammar(gapped_events)
    assert gapped["eligible_sequences"] == 1
    assert runtime_adapter._grammar_oracle_passes(gapped) is False

    one_game = runtime_adapter._executed_grammar(events[:-1])
    assert one_game["executed_sequences"] > 0
    assert runtime_adapter._grammar_oracle_passes(one_game) is False


class SpyPosterior:
    instances: ClassVar[list[SpyPosterior]] = []

    def __init__(self) -> None:
        self.seeded: tuple[Any, ...] = ()
        self.observed: list[Any] = []
        self.classes = (object(),)
        self.instances.append(self)

    def seed(self, candidates: tuple[Any, ...]) -> None:
        self.seeded = tuple(candidates)

    def observe(self, bundle: Any) -> None:
        self.observed.append(bundle)


def test_source_trainer_uses_posterior_reports_every_control_and_fails_closed(
    tmp_path: Path,
) -> None:
    manifest, shard_paths = _replay_fixture(tmp_path / "shards")
    manifest["manifest_checksum"] = "m"
    manifest["environment_sha256"] = "e"
    fresh = [
        _fresh_event(
            manifest,
            event_id="learned",
            game=protocol.SOURCE_GAMES[0],
            seed=3,
            split="leave_one_game_out_confirmation",
            controller="learned",
            progress=True,
            reset=0,
        ),
        _fresh_event(
            manifest,
            event_id="independent",
            game=protocol.SOURCE_GAMES[0],
            seed=3,
            split="leave_one_game_out_confirmation",
            controller="capacity_matched_independent",
            progress=False,
            reset=1,
        ),
    ]
    replay = build_v4_3_replay_ledger(
        manifest=manifest,
        shard_paths=shard_paths,
    )
    protocol.write_event_ledger(tmp_path / "source_events.jsonl", fresh)
    protocol.write_event_ledger(tmp_path / "replay_events.jsonl", replay)
    _write_cross_fit_audit(tmp_path, manifest=manifest, fresh=fresh)

    metrics = run_source_trainer(
        manifest=manifest,
        compile_report={"status": "PASS_T10_2_QA"},
        replay_report={"status": "T10_2_SOURCE_REPLAY_COMPLETE"},
        output_dir=tmp_path,
        candidates=("candidate",),
        posterior_factory=SpyPosterior,
        bundle_builder=lambda event: {"event": event["event_id"]},
    )
    assert set(metrics["completed_controls"]) == set(
        protocol.REGISTERED_SOURCE_CONTROLS
    )
    assert metrics["completed_controls"]["grammar_oracle"] is True
    assert metrics["completed_controls"]["no_transport"] is False
    assert metrics["control_results"]["no_transport"]["execution_ok"] is False
    assert metrics["control_results"]["no_transport"]["passed"] is False
    assert metrics["learned"]["positive_fold_ranks"] == {protocol.SOURCE_GAMES[0]: 2}
    assert metrics["posterior"]["used"] is True
    assert metrics["posterior"]["observation_count"] == 2 + len(replay)
    assert SpyPosterior.instances[-1].seeded == ("candidate",)
    assert metrics["controls"]["transport_oracle_passed"] is False
    assert metrics["learned"]["game_seed_probe_accuracy_increment"] > 0.0
    assert metrics["control_results"]["grammar_oracle"]["passed"] is False
    assert metrics["control_results"]["option_oracle"]["passed"] is False
    assert metrics["control_results"]["complete_program_oracle"]["passed"] is False
    transport_oracle = metrics["control_results"]["transport_oracle"]
    assert transport_oracle["attempted"] is True
    assert transport_oracle["execution_ok"] is False
    assert transport_oracle["completed"] is False
    assert transport_oracle["passed"] is False
    assert transport_oracle["nontrivial_exact_commutative_certificate_count"] == 6
    assert transport_oracle["certified_orbit_witness_candidate_count"] == 0
    assert transport_oracle["posterior_merged_gauge_class_count"] == 0
    assert metrics["controls"]["transport_oracle_passed"] is False
    for name in (
        "dynamics_oracle",
        "goal_oracle",
        "option_oracle",
        "complete_program_oracle",
    ):
        result = metrics["control_results"][name]
        assert result["attempted"] is True
        assert result["completed"] is result["execution_ok"]
        assert result["scientific_pass"] is result["passed"]
        assert result["active_execution"] is False
        assert result["executed_actions"] == len(fresh)
        assert "observation_count" in result
    frozen = metrics["control_results"]["t10_1_behavior_frozen_baseline"]
    assert frozen["execution_mode"] == "offline_behavior_frozen_replay"
    assert frozen["active_execution"] is False


def _seal_runtime_rows(
    rows: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    game: str,
    seed: int,
    split: str,
) -> list[dict[str, Any]]:
    sealed = []
    for raw in rows:
        event = dict(raw)
        event.update({"game_id": game, "seed": seed, "split": split})
        event["provenance"] = {
            **event["provenance"],
            "kind": "fresh_source_trajectory",
            "game_id": game,
            "seed": seed,
            "split": split,
            "manifest_checksum": manifest["manifest_checksum"],
            "environment_sha256": manifest["environment_sha256"],
        }
        sealed.append(protocol.seal_event(event))
    return sealed


def test_default_source_trainer_synthesizes_and_observes_real_gauge_posterior(
    tmp_path: Path,
) -> None:
    manifest, shard_paths = _replay_fixture(tmp_path / "shards")
    factory = T10_2SourceFactory(
        runtime_loader=lambda: FakeRuntime(),
        bundle_builder=_bundle_builder,
    )
    fresh: list[dict[str, Any]] = []
    for game in protocol.SOURCE_GAMES:
        raw = _collect(
            factory,
            game,
            seed=0,
            split="discovery",
            held_out=None,
        )
        fresh.extend(
            _seal_runtime_rows(
                raw,
                manifest=manifest,
                game=game,
                seed=0,
                split="discovery",
            )
        )
    for held_out in protocol.SOURCE_GAMES:
        for seed in protocol.CONFIRMATION_SEEDS:
            raw_confirmation = _collect(
                factory,
                held_out,
                seed=seed,
                split="leave_one_game_out_confirmation",
                held_out=held_out,
            )
            fresh.extend(
                _seal_runtime_rows(
                    raw_confirmation,
                    manifest=manifest,
                    game=held_out,
                    seed=seed,
                    split="leave_one_game_out_confirmation",
                )
            )
    replay = build_v4_3_replay_ledger(
        manifest=manifest,
        shard_paths=shard_paths,
    )
    protocol.write_event_ledger(tmp_path / "source_events.jsonl", fresh)
    protocol.write_event_ledger(tmp_path / "replay_events.jsonl", replay)
    cross_fit_audit = _write_cross_fit_audit(
        tmp_path,
        manifest=manifest,
        fresh=fresh,
        units=factory.cross_fit_audit,
    )
    assert cross_fit_audit["passed"] is True

    metrics = run_source_trainer(
        manifest=manifest,
        compile_report={
            "status": "T10_2_FRESH_INTEGRITY_COMPLETE",
            "integrity_passed": True,
            "fresh_scientific_qa": {"passed": False},
        },
        replay_report={"status": "T10_2_SOURCE_REPLAY_COMPLETE"},
        output_dir=tmp_path,
    )
    assert metrics["posterior"]["implementation"] == "GaugeProgramPosterior"
    assert metrics["posterior"]["candidate_count"] > 0
    assert metrics["posterior"]["observation_count"] == len(fresh) + len(replay)
    assert metrics["posterior"]["errors"] == []
    assert metrics["learned"]["common_posterior_passed"] is True
    assert all(metrics["attempted_controls"].values())
    assert not all(metrics["completed_controls"].values())
    assert metrics["grammar_oracle"]["positive_folds"]
    assert metrics["controls"]["option_oracle_passed"] is False
    assert metrics["controls"]["complete_program_oracle_passed"] is False
    assert metrics["posterior"]["candidate_count"] <= 256
    assert metrics["posterior"]["decision_class_budget"] == 64
    assert {"repeat", "alternate"} <= set(metrics["posterior"]["option_schemas"])
    assert metrics["posterior"]["candidates_with_transports"] > 0
    grammar_control = metrics["control_results"]["grammar_oracle"]
    assert grammar_control["passed"] is True
    assert grammar_control["progress_games"] >= 2
    assert grammar_control["levels"] >= 2
    assert grammar_control["actions"] > 0
    assert grammar_control["errors"] == 0
    assert grammar_control["illegal_actions"] == 0
    assert grammar_control["game_overs"] == 0
    transport_oracle = metrics["control_results"]["transport_oracle"]
    assert transport_oracle["attempted"] is True
    assert transport_oracle["execution_ok"] is False
    assert transport_oracle["completed"] is False
    assert transport_oracle["passed"] is False
    assert transport_oracle["nontrivial_exact_commutative_certificate_count"] == 0
    assert transport_oracle["certified_orbit_witness_candidate_count"] == 0
    assert transport_oracle["posterior_merged_gauge_class_count"] == 0
    assert metrics["controls"]["transport_oracle_passed"] is False
    for name in (
        "dynamics_oracle",
        "goal_oracle",
        "option_oracle",
        "complete_program_oracle",
    ):
        result = metrics["control_results"][name]
        assert result["attempted"] is True
        assert result["passed"] is False, (name, result)
        assert result["active_execution"] is False
        assert result["executed_actions"] > 0
        assert result["observation_count"] >= 0
    assert metrics["source_evidence"]["control_results_sha256"] == (
        protocol.canonical_sha256(metrics["control_results"])
    )
    assert (
        metrics["control_results"]["t10_1_behavior_frozen_baseline"]["passed"] is True
    )
    assert (
        metrics["control_results"]["t10_1_behavior_frozen_baseline"]["executed_actions"]
        > 0
    )
    recipe_binding = metrics["challenger_recipe"]
    recipe_path = tmp_path / protocol.CHALLENGER_RECIPE_FILENAME
    assert recipe_binding["bound"] is True
    assert recipe_binding["artifact"] == protocol.artifact_descriptor(recipe_path)
    assert (
        recipe_path.stat().st_size
        <= protocol.DEFAULT_RESOURCE_LIMITS.maximum_checkpoint_bytes
    )
    recipe = protocol.read_checked_json(
        recipe_path,
        checksum_key="recipe_checksum",
    )
    assert recipe["candidate_bank"]["grammar_input"] == "source_events_only"
    assert recipe["candidate_bank"]["grammar_retuned"] is False
    assert recipe["candidate_bank"]["prior_retuned"] is False
    assert recipe["posterior_fit"]["observation_count"] == len(fresh) + len(replay)

    source_report = protocol.signed_payload(
        {
            "format_version": protocol.FORMAT_VERSION,
            "phase": "source-train",
            "status": "PASS_T10_2_SOURCE_GATE",
            "verdict": "PASS_T10_2_SOURCE_GATE",
            "manifest_checksum": manifest["manifest_checksum"],
            "metrics": metrics,
            "checks": {"synthetic_complete_source_pass": True},
            "passed": True,
            "inputs": {
                "fresh_events": protocol.artifact_descriptor(
                    tmp_path / "source_events.jsonl"
                ),
                "replay_events": protocol.artifact_descriptor(
                    tmp_path / "replay_events.jsonl"
                ),
            },
            "firewall": {
                "source_validation_opened": True,
                "ar25_opened": False,
                "holdout_opened": False,
                "production_authority": False,
            },
        },
        checksum_key="report_checksum",
    )
    source_report_path = tmp_path / "source_report.json"
    protocol.write_compact_json(source_report_path, source_report)
    policy_factory = T10_2GaugePolicyFactory(
        source_report=source_report_path,
        manifest=manifest,
        output_dir=tmp_path,
        bundle_builder=_bundle_builder,
    )
    validation_runtime = FakeRuntime(progress=False)
    policy = policy_factory(
        controller="t10_2",
        game_id=protocol.VALIDATION_GAMES[0],
        seed=protocol.VALIDATION_SEEDS[0],
        posterior_reset=True,
        learning_enabled=False,
        runtime=validation_runtime,
    )
    policy.reset(reset_index=0, posterior_reset=True, learning_enabled=False)
    environment = validation_runtime.open(
        protocol.VALIDATION_GAMES[0], protocol.VALIDATION_SEEDS[0]
    )
    legal = validation_runtime.legal_actions(environment)
    before = FakeSnapshot()
    selected = policy.select_action(
        legal_actions=legal,
        environment=environment,
        reset_index=0,
        step_index=0,
        learning_enabled=False,
    )
    assert selected in legal
    policy.observe(
        before=before,
        after=FakeSnapshot(),
        action=selected,
        legal_actions=legal,
        reset_index=0,
        step_index=0,
        learning_enabled=False,
    )
    assert policy.online_observations == 1
    assert policy.behavior_projection == (
        "frozen_source_gauge_posterior_observe_update_replan"
    )

    source_event_path = tmp_path / "source_events.jsonl"
    source_event_path.write_text(
        source_event_path.read_text(encoding="utf-8") + "{}\n",
        encoding="utf-8",
    )
    with pytest.raises(protocol.ManifestDriftError, match="ledger drifted"):
        T10_2GaugePolicyFactory(
            source_report=source_report_path,
            manifest=manifest,
            output_dir=tmp_path,
            bundle_builder=_bundle_builder,
        )
    assert (
        metrics["control_results"]["t10_1_behavior_frozen_baseline"]["execution_mode"]
        == "offline_behavior_frozen_replay"
    )
    assert (
        metrics["control_results"]["t10_1_behavior_frozen_baseline"]["active_execution"]
        is False
    )


def _source_pass() -> dict[str, Any]:
    return {
        "status": "PASS_T10_2_SOURCE_GATE",
        "verdict": "PASS_T10_2_SOURCE_GATE",
        "passed": True,
        "checks": {"all": True},
        "firewall": {
            "source_validation_opened": True,
            "ar25_opened": False,
            "holdout_opened": False,
            "production_authority": False,
        },
    }


class FrozenPolicy:
    def __init__(self) -> None:
        self.learning_flags: list[bool] = []

    @staticmethod
    def select_action(
        legal_actions: tuple[FakeAction, ...], **_kwargs: Any
    ) -> FakeAction:
        return legal_actions[0]

    def observe(self, *, learning_enabled: bool, **_kwargs: Any) -> None:
        self.learning_flags.append(learning_enabled)


@dataclass(frozen=True)
class GridSnapshot:
    grid: tuple[tuple[int, ...], ...]
    levels_completed: int = 0
    game_state: str = "NOT_FINISHED"


class ExactProjectionRuntime:
    def __init__(self, *, action_count: int = 2) -> None:
        self.action_count = action_count
        self.steps = 0
        self.opens: list[tuple[str, int]] = []
        self.closed = 0

    def open(self, game_id: str, seed: int) -> dict[str, Any]:
        self.opens.append((game_id, seed))
        return {"grid": [0] * self.action_count}

    def reset(self, environment: dict[str, Any]) -> GridSnapshot:
        environment["grid"] = [0] * self.action_count
        return self.snapshot(environment)

    def legal_actions(self, _environment: dict[str, Any]) -> tuple[FakeAction, ...]:
        return tuple(
            FakeAction(f"ACTION{index + 1}", {}) for index in range(self.action_count)
        )

    def step(self, environment: dict[str, Any], action: FakeAction) -> GridSnapshot:
        self.steps += 1
        index = int(action.name.removeprefix("ACTION")) - 1
        environment["grid"][index] = 1
        return self.snapshot(environment)

    @staticmethod
    def snapshot(frame: Any, **_kwargs: Any) -> GridSnapshot:
        if isinstance(frame, GridSnapshot):
            return frame
        return GridSnapshot(grid=(tuple(frame["grid"]),))

    def close(self, _environment: dict[str, Any]) -> None:
        self.closed += 1


class IdleFrozenPolicy:
    @staticmethod
    def reset(**_kwargs: Any) -> None:
        return None

    @staticmethod
    def select_action(**_kwargs: Any) -> None:
        return None


def test_t10_1_factory_runs_literal_scan_then_candidate_macros_in_14_resets() -> None:
    runtime = ExactProjectionRuntime(action_count=2)
    baseline_factory = T10_1BehaviorFrozenPolicyFactory(repo_root=REPO_ROOT)
    factory = T10_2ValidationFactory(
        source_report=_source_pass(),
        runtime_loader=lambda: runtime,
        t10_1_policy_factory=baseline_factory,
        t10_2_policy_factory=lambda **_kwargs: IdleFrozenPolicy(),
    )
    game = protocol.VALIDATION_GAMES[0]
    environment = factory(game_id=game, seed=2101, phase="validate")
    result = environment.run_validation(
        game_id=game,
        seed=2101,
        controller_order=("t10_1", "t10_2"),
        resets=14,
        action_budget=96,
        posterior_reset=True,
        learning_enabled=False,
    )
    baseline = result["baseline"]
    assert baseline["behavior_projection"] == (
        "exact_one_step_scan_then_frozen_candidate_macros_across_14_resets"
    )
    assert baseline["legal_actions"] == 34
    assert baseline["planned_actions"] == baseline["completed_actions"] == 34
    assert baseline["illegal_actions"] == 0
    assert baseline["errors"] == 0
    assert baseline["unregistered_stops"] == 0
    assert len(baseline["reset_summaries"]) == 14
    assert {
        reset["stop_reason"] for reset in baseline["reset_summaries"]
    } <= protocol.VALIDATION_REGISTERED_STOP_REASONS
    assert baseline["deduplicated_immediate_noops"] > 0
    candidate = result["t10_2"]
    assert candidate["completed_actions"] == 0
    assert candidate["planned_actions"] == 14 * 96
    assert candidate["unregistered_stops"] == 14
    assert {reset["stop_reason"] for reset in candidate["reset_summaries"]} == {
        "policy_abstained"
    }
    assert baseline_factory.binding["manifest_checksum"] == (
        runtime_adapter.T10_1_FROZEN_MANIFEST_CHECKSUM
    )


def test_t10_1_exact_projection_refuses_before_first_action_when_resets_do_not_fit() -> (
    None
):
    runtime = ExactProjectionRuntime(action_count=8)
    factory = T10_2ValidationFactory(
        source_report=_source_pass(),
        runtime_loader=lambda: runtime,
        t10_1_policy_factory=T10_1BehaviorFrozenPolicyFactory(repo_root=REPO_ROOT),
        t10_2_policy_factory=lambda **_kwargs: IdleFrozenPolicy(),
    )
    game = protocol.VALIDATION_GAMES[0]
    environment = factory(game_id=game, seed=2101, phase="validate")
    with pytest.raises(protocol.RuntimeUnavailableError, match="14 resets"):
        environment.run_validation(
            game_id=game,
            seed=2101,
            controller_order=("t10_1", "t10_2"),
            resets=14,
            action_budget=96,
            posterior_reset=True,
            learning_enabled=False,
        )
    assert runtime.steps == 0


def test_validation_factory_runs_exact_paired_budget_without_learning() -> None:
    runtime = FakeRuntime(progress=True)
    policies: list[FrozenPolicy] = []

    def policy_factory(**_kwargs: Any) -> FrozenPolicy:
        policy = FrozenPolicy()
        policies.append(policy)
        return policy

    factory = T10_2ValidationFactory(
        source_report=_source_pass(),
        runtime_loader=lambda: runtime,
        t10_1_policy_factory=policy_factory,
        t10_2_policy_factory=policy_factory,
    )
    game = protocol.VALIDATION_GAMES[0]
    environment = factory(game_id=game, seed=2101, phase="validate")
    result = environment.run_validation(
        game_id=game,
        seed=2101,
        controller_order=("t10_2", "t10_1"),
        resets=14,
        action_budget=96,
        posterior_reset=True,
        learning_enabled=False,
    )
    assert result["controller_order"] == ["t10_2", "t10_1"]
    assert result["baseline"]["levels"] == 14
    assert result["t10_2"]["levels"] == 14
    assert result["baseline"]["legal_actions"] == 14
    assert result["t10_2"]["legal_actions"] == 14
    assert result["baseline"]["planned_actions"] == 14
    assert result["t10_2"]["planned_actions"] == 14
    assert result["baseline"]["unregistered_stops"] == 0
    assert result["t10_2"]["unregistered_stops"] == 0
    assert {reset["stop_reason"] for reset in result["t10_2"]["reset_summaries"]} == {
        "progression"
    }
    assert result["learning_between_controllers"] is False
    assert len(runtime.opens) == 2
    assert runtime.closed == 2
    assert all(flag is False for policy in policies for flag in policy.learning_flags)


def test_validation_has_no_approximate_t10_1_fallback_and_never_opens_runtime() -> None:
    loads: list[bool] = []
    factory = T10_2ValidationFactory(
        source_report=_source_pass(),
        runtime_loader=lambda: loads.append(True),
        t10_2_policy_factory=lambda: FrozenPolicy(),
    )
    game = protocol.VALIDATION_GAMES[0]
    environment = factory(game_id=game, seed=2101, phase="validate")
    with pytest.raises(protocol.RuntimeUnavailableError, match="exact behavior-frozen"):
        environment.run_validation(
            game_id=game,
            seed=2101,
            controller_order=("t10_1", "t10_2"),
            resets=14,
            action_budget=96,
            posterior_reset=True,
            learning_enabled=False,
        )
    assert loads == []


def test_validation_rejects_failed_source_before_factory_exists() -> None:
    failed = _source_pass()
    failed["passed"] = False
    with pytest.raises(protocol.GateRefusalError, match="source PASS"):
        T10_2ValidationFactory(source_report=failed)
