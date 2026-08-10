from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from theory.sage_t import t10_3_1_protocol as protocol
from theory.sage_t import t10_3_1_runtime as runtime
from theory.sage_t.contracts import AbstractEntity, AbstractState
from theory.sage_t.contracts import ActionCandidate, ObservedTransition, PredictionPacket
from theory.sage_t.frame_adapters_v10_3_1 import project_goal_transition


@dataclass(frozen=True)
class _Action:
    name: str
    action_args: dict


def test_dynamic_bp35_choice_uses_current_legal_inventory() -> None:
    state = AbstractState(entities=(AbstractEntity("actor", ("actor", "player")),))
    work = next(
        item
        for item in runtime.build_work_specs("panel")
        if item.game_id.startswith("bp35") and item.controller == "canonical_option"
    )
    first = [_Action("ACTION1", {}), _Action("ACTION4", {})]
    second = [_Action("ACTION2", {}), _Action("ACTION3", {})]
    assert runtime.choose_regrounded_action(work, state, first, step_index=0).name == "ACTION1"
    assert runtime.choose_regrounded_action(work, state, second, step_index=1).name == "ACTION2"


def test_ambiguous_parameter_anchor_is_not_used_as_adaptive_substitute() -> None:
    state = AbstractState(
        entities=(
            AbstractEntity("a", ("target",), center=(0, 0)),
            AbstractEntity("b", ("target",), center=(0, 2)),
        )
    )
    work = next(
        item
        for item in runtime.build_work_specs("panel")
        if item.game_id.startswith("lp85") and item.controller == "canonical_option"
    )
    assert (
        runtime.choose_regrounded_action(
            work, state, [_Action("ACTION6", {"x": 1, "y": 0})], step_index=0
        )
        is None
    )


def test_only_initial_grounding_miss_makes_branch_unknown() -> None:
    initial_status, initial_error = runtime.classify_grounding_stop(
        "binding_swap", 0
    )
    later_status, later_error = runtime.classify_grounding_stop(
        "binding_swap", 7
    )
    assert initial_status == "OPTION_GROUNDING_MISS"
    assert initial_error is True
    assert later_status == "OPTION_TERMINATED_NO_GROUNDING"
    assert later_error is False


def test_registry_has_sixty_unique_counterbalanced_resets() -> None:
    panel = runtime.build_work_specs("panel")
    confirmation = runtime.build_work_specs("confirmation")
    assert len(panel) == 48
    assert len(confirmation) == 12
    assert len({item.work_id for item in (*panel, *confirmation)}) == 60
    first = {}
    for item in confirmation:
        first.setdefault((item.game_id, item.seed), item.controller)
    assert set(first.values()) == set(protocol.CONFIRMATION_CONTROLLERS)


def _synthetic_rows():
    rows = []
    receipts = []
    reset = 0
    for game in protocol.SOURCE_GAMES:
        for seed in protocol.PANEL_SEEDS:
            for controller in protocol.PANEL_ARMS:
                positive = controller == "canonical_option" and game in protocol.POSITIVE_WITNESS_GAMES
                transport_active = controller == "canonical_option"
                work_id = f"work-{reset}"
                event_ids = []
                for step in range(8):
                    event_id = f"event-{reset}-{step}"
                    event_ids.append(event_id)
                    rows.append(
                        {
                            "event_id": event_id,
                            "work_id": work_id,
                            "source_game": game,
                            "seed": seed,
                            "controller": controller,
                            "reset_index": reset,
                            "step_index": step,
                            "complete": True,
                            "model_view": {
                                "frames": {
                                    name: {"complete": True}
                                    for name in ("root_only", "allocentric", "aligned", "topological")
                                }
                            },
                            "binding": {
                                "complete": True,
                                "pre_action_complete": True,
                                "after_root_available": True,
                            },
                            "correspondence": {
                                "fraction_denominator": 1,
                                "confident_matches": 1,
                                "fully_ambiguous_matches": 0,
                            },
                            "transport": {
                                "multiframe_exact_nonidentity": True,
                                "exact_nonidentity_certificate_count": 1,
                                "common_quotient_changed": transport_active,
                                "common_quotient_delta_count": 8 if transport_active else 0,
                            },
                            "transport_certificates": [
                                {
                                    "comparable": True,
                                    "exact": True,
                                    "projection_complete": True,
                                    "round_trip_exact": True,
                                    "commutativity": {"exact": True},
                                }
                            ],
                            "observations": {
                                "root_effect": False,
                                "physical_no_effect": False,
                                "level_complete": positive and step == 7,
                                "game_over": False,
                            },
                            "labels": {"goal_reachable_within_option": positive},
                            "selection": {"parameter_arity": 1},
                        }
                    )
                receipts.append(
                    {
                        "work_id": work_id,
                        "phase": "panel",
                        "game_id": game,
                        "seed": seed,
                        "controller": controller,
                        "reset_index": reset,
                        "complete": True,
                        "goal_reachable_within_option": positive,
                        "issued_intents": 8,
                        "sealed_events": 8,
                        "unresolved_intents": 0,
                        "event_ids": event_ids,
                        "level_delta": int(positive),
                        "errors": [],
                        "illegal_actions": 0,
                        "game_over": False,
                        "receipt_checksum": "r" * 64,
                    }
                )
                reset += 1
    return rows, receipts


def test_accounting_and_unknown_branch_labels_are_distinct_gates() -> None:
    rows, receipts = _synthetic_rows()
    rows[0]["labels"]["goal_reachable_within_option"] = None
    checkpoint = {
        "equation_holds": True,
        "explicitly_unresolved_intent_count": 0,
    }
    qa = runtime.build_qa_report(
        rows, receipts, checkpoint, manifest_checksum="m" * 64
    )
    assert qa["checks"]["intent_accounting_equation"] is True
    assert qa["checks"]["branch_label_completeness"] is False


def test_synthetic_causal_panel_passes_qa_and_transport_ablation() -> None:
    rows, receipts = _synthetic_rows()
    checkpoint = {
        "equation_holds": True,
        "explicitly_unresolved_intent_count": 0,
    }
    qa = runtime.build_qa_report(
        rows, receipts, checkpoint, manifest_checksum="m" * 64
    )
    assert qa["passed"] is True
    migration = protocol.build_migration_receipt(repo_root=".")
    manifest = {"manifest_checksum": "m" * 64, "migration_receipt": migration}
    recipe = runtime.build_model_recipe(rows, manifest)
    assert recipe["passed"] is True
    assert recipe["cross_fit"]["lp85-305b61c3"]["auroc"] >= 0.75
    assert recipe["cross_fit"]["su15-4c352900"]["brier_improvement"] > 0
    assert recipe["full_mean_auroc"] > recipe["no_transport_mean_auroc"]
    assert recipe["parent_t10_3_events_used"] == 0


def test_interrupted_intent_is_never_replayed(tmp_path: Path) -> None:
    destination = tmp_path / "pilot"
    manifest = {"manifest_checksum": "m" * 64}
    work = runtime.build_work_specs("panel")[0]
    intent = runtime._intent_payload(
        manifest,
        work,
        step_index=0,
        action={"name": "ACTION1", "action_args": {}},
        binding={
            "method": "movement_actor",
            "structural_signature": "s" * 64,
            "unique": True,
            "complete": True,
        },
    )
    runtime._write_signed_once(
        runtime._work_path(destination, "intents", work, "00.json"), intent
    )
    receipt = runtime._recover_interrupted_work(destination, manifest, work)
    assert receipt["unresolved_intents"] == 1
    assert receipt["physical_actions_replayed"] == 0


def test_synthetic_runtime_seals_sixteen_dynamically_regrounded_actions(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeRuntime:
        def __init__(self):
            self.steps = []

        def open(self, game_id, seed):
            return self

        def reset(self, environment):
            return SimpleNamespace()

        def legal_actions(self, environment):
            return [_Action(f"ACTION{index}", {}) for index in range(1, 5)]

        def step(self, environment, action):
            self.steps.append(action.name)
            return SimpleNamespace()

        def snapshot(self, frame, fallback_available_actions=()):
            return SimpleNamespace(
                grid=[[0, 0], [0, 0]],
                game_state="NOT_FINISHED",
                levels_completed=0,
            )

        def close(self, environment):
            return None

    state = AbstractState(entities=(AbstractEntity("actor", ("actor", "player")),))
    monkeypatch.setattr(runtime, "_abstract_state", lambda snapshot, legal: state)

    def projection_builder(**kwargs):
        action = ActionCandidate(kwargs["action"].name)
        return project_goal_transition(
            ObservedTransition(state, action, state, PredictionPacket()),
            event_id=kwargs["event_id"],
        )

    monkeypatch.setattr(runtime, "_build_goal_projection", projection_builder)
    fake = FakeRuntime()
    work = next(
        item
        for item in runtime.build_work_specs("panel")
        if item.game_id.startswith("bp35") and item.controller == "canonical_option"
    )
    receipt = runtime._run_work(
        tmp_path,
        {"manifest_checksum": "m" * 64},
        work,
        fake,
    )
    assert receipt["complete"] is True
    assert receipt["issued_intents"] == 16
    assert receipt["sealed_events"] == 16
    assert receipt["goal_reachable_within_option"] is False
    assert fake.steps == ["ACTION1", "ACTION2", "ACTION3", "ACTION4"] * 4
