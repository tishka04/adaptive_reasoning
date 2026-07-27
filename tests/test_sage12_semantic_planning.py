from __future__ import annotations

import json

import numpy as np
import pytest

from theory.live_transition_loop import build_transition_record
from theory.sage12 import (
    EntityRef,
    HierarchicalSubgoal,
    HypothesisCompiler,
    LocalHypothesisGenerator,
    PairwiseTrajectoryEBM,
    Sage12Config,
    Sage12Mode,
    SemanticActionCandidate,
    SemanticEffect,
    SemanticHypothesis,
    SemanticPlanningController,
    SemanticPredicate,
    TemplateHypothesisGenerator,
    build_scene_graph,
)
from theory.sage12.llm import HypothesisGenerationResult
from theory.unified_cognitive_controller import UnifiedCognitiveController
from v3.schemas import (
    GameObservation,
    ObjectInfo,
    PlayerHypothesis,
)


def _observation() -> GameObservation:
    grid = np.zeros((5, 5), dtype=np.int64)
    grid[2, 1] = 1
    grid[2, 2] = 2
    return GameObservation(
        raw_grid=grid,
        grid_hash=123,
        game_state="NOT_FINISHED",
        levels_completed=0,
        available_actions=["ACTION1", "ACTION2"],
        objects=[
            ObjectInfo(
                object_id=1,
                value=1,
                cells=[(2, 1)],
                bbox=(2, 1, 2, 1),
                center=(2.0, 1.0),
                area=1,
            ),
            ObjectInfo(
                object_id=2,
                value=2,
                cells=[(2, 2)],
                bbox=(2, 2, 2, 2),
                center=(2.0, 2.0),
                area=1,
            ),
        ],
        player_candidates=[
            PlayerHypothesis(
                value=1,
                position=(2, 1),
                confidence=0.95,
            )
        ],
    )


def _hypothesis(
    hypothesis_id: str,
    action_name: str,
    effect: str,
    *,
    confidence: float = 0.9,
) -> SemanticHypothesis:
    return SemanticHypothesis(
        hypothesis_id=hypothesis_id,
        action_name=action_name,
        effects=(
            SemanticEffect(
                SemanticPredicate(effect),
            ),
        ),
        confidence=confidence,
        support=0,
    )


class _StaticGenerator:
    def generate(self, **_: object) -> HypothesisGenerationResult:
        return HypothesisGenerationResult(
            hypotheses=(
                _hypothesis("complete", "ACTION1", "level_complete"),
                _hypothesis("change", "ACTION2", "changed"),
            )
        )


def _controller(mode: Sage12Mode) -> SemanticPlanningController:
    return SemanticPlanningController(
        game_id="fixture",
        generator=_StaticGenerator(),
        config=Sage12Config(
            mode=mode,
            proposal_gate_passed=True,
            world_model_gate_passed=True,
            energy_gate_passed=True,
            active_gate_passed=True,
            maximum_depth=2,
            beam_width=8,
        ),
    )


def _arbitrate(
    controller: SemanticPlanningController,
    *,
    protected: bool = False,
    veto_action: str = "",
):
    return controller.arbitrate(
        symbolic_action_name="ACTION2",
        symbolic_action_data={},
        symbolic_source="symbolic",
        observation=_observation(),
        candidates=(
            SemanticActionCandidate("ACTION1"),
            SemanticActionCandidate("ACTION2"),
        ),
        protected_competence_available=protected,
        danger_veto=lambda name, data: name == veto_action,
        subgoals=(
            HierarchicalSubgoal("finish", "level_complete"),
        ),
    )


def test_scene_graph_contains_grounded_relations_without_game_identity() -> None:
    graph = build_scene_graph(_observation())

    assert graph.entities_for_role("player")
    assert any(relation.kind == "aligned" for relation in graph.relations)
    assert any(relation.kind == "contact" for relation in graph.relations)
    assert "fixture" not in graph.signature


def test_proposed_hypothesis_cannot_claim_observed_support() -> None:
    with pytest.raises(ValueError, match="support=0"):
        SemanticHypothesis(
            hypothesis_id="bad_support",
            action_name="ACTION1",
            effects=(
                SemanticEffect(SemanticPredicate("progress")),
            ),
            support=1,
        )


def test_compiler_rejects_unbound_roles_and_illegal_actions() -> None:
    graph = build_scene_graph(_observation())
    unbound = SemanticHypothesis(
        hypothesis_id="unbound",
        action_name="ACTION1",
        preconditions=(
            SemanticPredicate(
                "exists",
                subject=EntityRef("door"),
            ),
        ),
        effects=(
            SemanticEffect(SemanticPredicate("progress")),
        ),
    )
    illegal = _hypothesis("illegal", "RESET", "progress")

    result = HypothesisCompiler().compile(
        (unbound, illegal),
        graph=graph,
        legal_candidates=(
            SemanticActionCandidate("ACTION1"),
        ),
    )

    assert result.options == ()
    assert any("unbound_roles:door" in item for item in result.rejected)
    assert any("illegal_action" in item for item in result.rejected)


def test_local_llm_adapter_accepts_json_only_and_keeps_zero_support() -> None:
    class _Model:
        def generate_json(self, **_: object) -> str:
            return json.dumps(
                {
                    "hypotheses": [
                        {
                            "hypothesis_id": "move_toward_goal",
                            "action_name": "ACTION1",
                            "effects": [
                                {
                                    "predicate": {
                                        "name": "progress",
                                    }
                                }
                            ],
                            "confidence": 0.6,
                            "support": 0,
                        }
                    ]
                }
            )

    result = LocalHypothesisGenerator(_Model()).generate(
        graph=build_scene_graph(_observation()),
        available_actions=("ACTION1",),
        subgoal="progress",
    )

    assert result.parse_error == ""
    assert result.hypotheses[0].support == 0


def test_shadow_ranks_but_preserves_symbolic_action() -> None:
    result = _arbitrate(_controller(Sage12Mode.SHADOW))

    assert result.applied is False
    assert result.action_name == "ACTION2"
    assert result.selected_option_id
    assert result.trajectory_length >= 1


def test_active_executes_only_first_action_of_best_trajectory() -> None:
    result = _arbitrate(_controller(Sage12Mode.ACTIVE))

    assert result.applied is True
    assert result.action_name == "ACTION1"
    assert result.source == "sage12_semantic_planner"
    assert result.trajectory_length >= 1


def test_authority_gates_downgrade_to_shadow() -> None:
    controller = SemanticPlanningController(
        generator=_StaticGenerator(),
        config=Sage12Config(
            mode=Sage12Mode.ACTIVE,
            proposal_gate_passed=False,
            world_model_gate_passed=True,
            energy_gate_passed=True,
            active_gate_passed=True,
        ),
    )

    result = _arbitrate(controller)

    assert result.applied is False
    assert result.effective_mode == "shadow"
    assert controller.summary()["gate_downgrades"] == 1


def test_proposal_backend_exception_fails_closed() -> None:
    class _BrokenGenerator:
        def generate(self, **_: object):
            raise RuntimeError("model unavailable")

    controller = SemanticPlanningController(
        generator=_BrokenGenerator(),
        config=Sage12Config(mode=Sage12Mode.SHADOW),
    )

    result = _arbitrate(controller)

    assert result.action_name == "ACTION2"
    assert result.applied is False
    assert "failed closed" in result.reason
    assert controller.summary()["parse_failures"] == 1


def test_protected_competence_and_danger_veto_keep_symbolic_supremacy() -> None:
    protected = _arbitrate(
        _controller(Sage12Mode.ACTIVE),
        protected=True,
    )
    vetoed = _arbitrate(
        _controller(Sage12Mode.ACTIVE),
        veto_action="ACTION1",
    )

    assert protected.applied is False
    assert protected.action_name == "ACTION2"
    assert vetoed.action_name == "ACTION2"


def test_observed_transition_updates_evidence_after_execution_only() -> None:
    controller = _controller(Sage12Mode.ACTIVE)
    decision = _arbitrate(controller)
    before = np.zeros((3, 3), dtype=np.int64)
    after = before.copy()
    after[1, 1] = 1
    record = build_transition_record(
        action=decision.action_name,
        grid_before=before,
        grid_after=after,
        available_actions=("ACTION1", "ACTION2"),
        game_state_after="WIN",
        levels_completed_after=1,
        infer_players=False,
    )

    controller.observe_transition(record)
    summary = controller.summary()

    assert summary["observed_outcomes"] == 1
    assert summary["world_model"]["supported_effects"] >= 1
    assert summary["semantic_memory"]["observations"] == 1


def test_unified_controller_exposes_off_by_default_sage12_summary() -> None:
    controller = UnifiedCognitiveController(
        "fixture",
        available_actions=("ACTION1",),
    )

    summary = controller.summary()["sage12_semantic_planning"]

    assert summary["configured_mode"] == "off"
    assert summary["evaluations"] == 0


def test_unified_controller_can_apply_guarded_sage12_decision() -> None:
    semantic = _controller(Sage12Mode.ACTIVE)
    controller = UnifiedCognitiveController(
        "fixture",
        available_actions=("ACTION1", "ACTION2"),
        semantic_controller=semantic,
    )
    grid = np.zeros((5, 5), dtype=np.int64)
    grid[1, 1] = 1
    grid[3, 3] = 2

    decision = controller.select_action(
        current_grid=grid,
        available_actions=("ACTION1", "ACTION2"),
        legacy_action="ACTION2",
    )

    assert decision.action_name == "ACTION1"
    assert decision.source == "sage12_semantic_planner"


def test_pairwise_ebm_can_learn_lower_energy_for_preferred_examples() -> None:
    pytest.importorskip("torch")
    model = PairwiseTrajectoryEBM(hidden_width=8, seed=7)
    preferred = [[0.0, 0.0, 0.1, 0.1, 1.0, 0.0]] * 8
    rejected = [[1.0, 1.0, 0.9, 2.0, 3.0, 1.0]] * 8

    losses = model.fit_pairs(
        preferred,
        rejected,
        epochs=80,
        learning_rate=0.02,
    )
    energies = model.energies((preferred[0], rejected[0]))

    assert losses[-1] < losses[0]
    assert energies[0] < energies[1]
    assert model.trained_pairs == 8


def test_template_generator_is_a_deterministic_baseline() -> None:
    result = TemplateHypothesisGenerator().generate(
        graph=build_scene_graph(_observation()),
        available_actions=("ACTION2", "ACTION1"),
        subgoal="progress",
    )

    assert [item.action_name for item in result.hypotheses] == [
        "ACTION2",
        "ACTION1",
    ]
    assert all(item.support == 0 for item in result.hypotheses)
