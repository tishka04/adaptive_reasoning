from __future__ import annotations

from pathlib import Path

import pytest

from theory.sage_t import t10_3_protocol as protocol
from theory.sage_t import t10_3_runtime as runtime


def _synthetic_rows():
    rows = []
    receipts = []
    reset_index = 0
    for game in protocol.SOURCE_GAMES:
        for seed in protocol.PANEL_SEEDS:
            for controller in protocol.PANEL_ARMS:
                positive = controller == "canonical_option"
                work_id = f"work-{reset_index}"
                event_ids = []
                for step in range(8):
                    event_id = f"event-{reset_index}-{step}"
                    event_ids.append(event_id)
                    exact_count = 3 if positive else 1
                    rows.append(
                        {
                            "event_id": event_id,
                            "work_id": work_id,
                            "source_game": game,
                            "seed": seed,
                            "controller": controller,
                            "reset_index": reset_index,
                            "step_index": step,
                            "complete": True,
                            "model_view": {
                                "frames": {
                                    name: {"complete": True}
                                    for name in ("root_only", "allocentric", "aligned", "topological")
                                }
                            },
                            "binding": {"complete": True},
                            "correspondence": {
                                "fraction_denominator": 10,
                                "confident_matches": 10,
                                "fully_ambiguous_matches": 0,
                            },
                            "transport": {
                                "multiframe_exact_nonidentity": True,
                                "exact_nonidentity_certificate_count": exact_count,
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
                        "reset_index": reset_index,
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
                reset_index += 1
    return rows, receipts


def test_work_registry_covers_budget_and_counterbalances_confirmation() -> None:
    panel = runtime.build_work_specs("panel")
    confirmation = runtime.build_work_specs("confirmation")
    assert len(panel) == 48
    assert len(confirmation) == 12
    assert len({work.work_id for work in (*panel, *confirmation)}) == 60
    first_orders = {}
    for work in confirmation:
        first_orders.setdefault((work.game_id, work.seed), work.controller)
    assert set(first_orders.values()) == set(protocol.CONFIRMATION_CONTROLLERS)


def test_branch_backfill_cannot_cross_reset_and_handles_positive_negative_truncated() -> None:
    receipt = {
        "work_id": "w1",
        "phase": "panel",
        "game_id": protocol.SOURCE_GAMES[0],
        "seed": 3101,
        "controller": "canonical_option",
        "reset_index": 0,
        "complete": True,
        "goal_reachable_within_option": True,
        "receipt_checksum": "r" * 64,
    }
    event = {
        "event_id": "e1",
        "event_checksum": "e" * 64,
        "work_id": "w1",
        "step_index": 0,
        "binding": {"complete": True},
        "model_view": {"frames": {"root_only": {}}},
        "correspondence": {},
        "transport": {},
        "transport_certificates": [],
        "observations": {},
        "prefix": {},
        "action": {"name": "ACTION1", "data": {"parameter_arity": 0}},
    }
    assert runtime.backfill_branch_labels([event], receipt)[0]["labels"] == {
        "goal_reachable_within_option": True
    }
    negative = dict(receipt, goal_reachable_within_option=False)
    assert runtime.backfill_branch_labels([event], negative)[0]["labels"] == {
        "goal_reachable_within_option": False
    }
    truncated = dict(receipt, complete=False, goal_reachable_within_option=None)
    assert runtime.backfill_branch_labels([event], truncated)[0]["labels"] == {
        "goal_reachable_within_option": None
    }
    with pytest.raises(runtime.IntegrityError, match="reset boundary"):
        runtime.backfill_branch_labels([dict(event, work_id="w2")], receipt)


def test_incomplete_transport_can_never_retain_exact_attestation() -> None:
    event = {
        "transport_certificates": [
            {
                "source_frame": "root_only",
                "target_frame": "action_aligned_relational",
                "projection_complete": False,
                "exact": True,
                "round_trip_exact": True,
                "certifies_gauge_equivalence": True,
                "commutativity": {"exact": True},
            }
        ],
        "transport_orbits": [],
    }
    normalized = runtime.normalize_transport_evidence(event)
    certificate = normalized["transport_certificates"][0]
    assert certificate["comparable"] is False
    assert certificate["exact"] is False
    assert normalized["transport"]["incomplete_projections_attested_exact"] is False


def test_persistence_firewall_rejects_coordinates_colors_grids_and_entity_ids() -> None:
    runtime.assert_no_forbidden_persistence(
        {"binding": {"method": "movement_actor", "structural_signature": "a" * 64}}
    )
    for forbidden in ("x", "color", "grid", "entity_id"):
        with pytest.raises(runtime.IntegrityError, match="forbidden persisted field"):
            runtime.assert_no_forbidden_persistence({"nested": {forbidden: "raw"}})


def test_crash_between_intent_and_seal_is_marked_without_replay(tmp_path: Path) -> None:
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
    assert receipt is not None
    assert receipt["status"] == "INTERRUPTED_UNKNOWN"
    assert receipt["unresolved_intents"] == 1
    assert receipt["physical_actions_replayed"] == 0


def test_synthetic_causal_success_passes_qa_and_model_controls() -> None:
    rows, receipts = _synthetic_rows()
    qa = runtime.build_qa_report(rows, receipts, manifest_checksum="m" * 64)
    assert qa["passed"] is True
    handoff = protocol.build_handoff_receipt(repo_root=".")
    manifest = {"manifest_checksum": "m" * 64, "handoff_receipt": handoff}
    recipe = runtime.build_model_recipe(rows, manifest)
    assert recipe["passed"] is True
    assert recipe["cross_fit"]["lp85-305b61c3"]["auroc"] >= 0.75
    assert recipe["cross_fit"]["su15-4c352900"]["brier_improvement"] > 0
    assert recipe["full_mean_auroc"] > recipe["no_transport_mean_auroc"]
    assert recipe["feature_firewall"]["game_id"] is False
