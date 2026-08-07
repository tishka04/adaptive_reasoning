from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from theory.sage_t.consolidation import (
    ConsolidationRegistry,
    LegacyArbiter,
)
from theory.sage_t.contracts import ActionCandidate
from theory.sage_t.controller import (
    SageTArbitration,
    SageTConfig,
    SageTController,
)
from theory.sage_t.decision import (
    BayesianDecision,
    CandidateSequence,
    DisagreementBreakdown,
    SequenceAssessment,
)
from theory.unified_cognitive_controller import (
    UnifiedCognitiveConfig,
    UnifiedCognitiveController,
)
from v3.schemas import GameObservation, ObjectInfo, PlayerHypothesis


def _observation() -> GameObservation:
    grid = np.zeros((3, 3), dtype=np.int64)
    grid[1, 0] = 1
    grid[1, 2] = 2
    return GameObservation(
        raw_grid=grid,
        grid_hash=321,
        game_state="NOT_FINISHED",
        levels_completed=0,
        available_actions=["ACTION1", "ACTION2"],
        objects=[
            ObjectInfo(
                object_id=1,
                value=1,
                cells=[(1, 0)],
                bbox=(1, 0, 1, 0),
                center=(1.0, 0.0),
                area=1,
            ),
            ObjectInfo(
                object_id=2,
                value=2,
                cells=[(1, 2)],
                bbox=(1, 2, 1, 2),
                center=(1.0, 2.0),
                area=1,
            ),
        ],
        player_candidates=[
            PlayerHypothesis(
                value=1,
                position=(1, 0),
                confidence=0.95,
            ),
        ],
    )


class _ChooseActionTwo:
    def __init__(self, *, risk: float = 0.0) -> None:
        self.risk = risk

    def decide(self, *_: object, **__: object) -> BayesianDecision:
        chosen = SequenceAssessment(
            candidate=CandidateSequence((ActionCandidate("ACTION2"),)),
            utility=1.0,
            expected_goal=0.5,
            expected_progress=0.0,
            terminal_risk=self.risk,
            information_gain=0.5,
            beta=1.0,
            disagreement=DisagreementBreakdown(
                observational=0.2,
                causal=0.3,
                teleological=0.4,
                planning=0.5,
            ),
            residual_mass=0.0,
        )
        return BayesianDecision(
            chosen=chosen,
            assessments=(chosen,),
            normalized_entropy=0.8,
            reason="selected",
        )


class _StubSageT:
    def __init__(self) -> None:
        self.posterior = SimpleNamespace(particles=())
        self.observations = 0
        self.branches = 0

    def decide(self, **_: object) -> SageTArbitration:
        return SageTArbitration(
            action_name="ACTION2",
            action_data={},
            applied=True,
            requested_mode="active",
            effective_mode="active",
            reason="active_override",
        )

    def observe_transition(self, _: object) -> None:
        self.observations += 1

    def start_branch(self) -> None:
        self.branches += 1

    def note_level_change(self) -> None:
        self.branches += 1

    def summary(self):
        return {
            "requested_mode": "active",
            "effective_mode": "active",
        }


def _decide(controller: SageTController, *, protected: bool = False):
    return controller.decide(
        symbolic_action_name="ACTION1",
        symbolic_action_data={},
        observation=_observation(),
        legal_actions=(
            ActionCandidate("ACTION1"),
            ActionCandidate("ACTION2"),
        ),
        protected_route=protected,
    )


def test_bounded_and_active_requests_fail_closed_until_their_gates() -> None:
    bounded = SageTConfig(mode="bounded")
    active = SageTConfig(
        mode="active",
        counterfactual_gate_passed=True,
    )
    incomplete_active = SageTConfig(
        mode="active",
        active_gate_passed=True,
    )

    assert bounded.effective_mode.value == "shadow"
    assert active.effective_mode.value == "bounded"
    assert incomplete_active.effective_mode.value == "shadow"
    with pytest.raises(ValueError, match="risk"):
        SageTConfig(
            mode="bounded",
            bounded_maximum_terminal_risk=0.051,
        )


def test_shadow_records_but_never_changes_the_symbolic_action() -> None:
    controller = SageTController(
        config=SageTConfig(mode="shadow"),
        decision_engine=_ChooseActionTwo(),
    )

    result = _decide(controller)

    assert result.action_name == "ACTION1"
    assert result.applied is False
    assert result.reason == "shadow"
    assert controller.summary()["shadow_decisions"] == 1


def test_bounded_mode_allows_one_safe_intervention_per_unknown_context() -> None:
    controller = SageTController(
        config=SageTConfig(
            mode="bounded",
            counterfactual_gate_passed=True,
        ),
        decision_engine=_ChooseActionTwo(),
    )

    first = _decide(controller)
    second = _decide(controller)

    assert first.applied is True
    assert first.action_name == "ACTION2"
    assert second.applied is False
    assert second.reason == "bounded_context_budget"


def test_bounded_risk_and_protected_route_are_absolute_vetoes() -> None:
    risky = SageTController(
        config=SageTConfig(
            mode="bounded",
            counterfactual_gate_passed=True,
        ),
        decision_engine=_ChooseActionTwo(risk=0.06),
    )
    active = SageTController(
        config=SageTConfig(
            mode="active",
            counterfactual_gate_passed=True,
            active_gate_passed=True,
        ),
        decision_engine=_ChooseActionTwo(),
    )

    assert _decide(risky).reason == "bounded_risk_veto"
    protected = _decide(active, protected=True)
    assert protected.applied is False
    assert protected.reason == "protected_route_veto"


def test_jsonl_audit_contains_posterior_and_sequence_matrix(tmp_path) -> None:
    trace = tmp_path / "sage_t.jsonl"
    controller = SageTController(
        config=SageTConfig(
            mode="shadow",
            trace_path=str(trace),
        ),
        decision_engine=_ChooseActionTwo(),
    )

    _decide(controller)
    record = json.loads(trace.read_text(encoding="utf-8").splitlines()[0])

    assert record["kind"] == "decision"
    assert "posterior_before" in record
    assert record["sequences"][0]["disagreement"]["causal"] == 0.3


def test_unified_controller_defaults_to_sage_t_off() -> None:
    controller = UnifiedCognitiveController(
        "unit",
        available_actions=["ACTION1", "ACTION2"],
        config=UnifiedCognitiveConfig(),
    )

    summary = controller.summary()["sage_t_joint_program_posterior"]

    assert summary["requested_mode"] == "off"
    assert summary["effective_mode"] == "off"


def test_t6_consolidation_requires_both_proof_and_rollback() -> None:
    registry = ConsolidationRegistry()

    rejected = registry.retire(
        LegacyArbiter.GOAL_GENERATOR,
        active_gate_passed=True,
        paired_ablation_passed=False,
    )
    accepted = registry.retire(
        LegacyArbiter.FRONTIER_SCORE,
        active_gate_passed=True,
        paired_ablation_passed=True,
    )
    registry.rollback(LegacyArbiter.FRONTIER_SCORE)

    assert rejected is False
    assert registry.is_authoritative(LegacyArbiter.GOAL_GENERATOR)
    assert accepted is True
    assert registry.is_authoritative(LegacyArbiter.FRONTIER_SCORE)
    assert set(registry.snapshot()["a32_a40_adapters"]) == {
        f"A{number}" for number in range(32, 41)
    }


def test_unified_controller_selection_observation_and_reset_seams() -> None:
    sage_t = _StubSageT()
    controller = UnifiedCognitiveController(
        "unit",
        available_actions=["ACTION1", "ACTION2"],
        sage_t_controller=sage_t,
    )
    grid = np.array(
        [
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.int64,
    )

    decision = controller.select_action(
        current_grid=grid,
        available_actions=["ACTION1", "ACTION2"],
        legacy_action="ACTION1",
    )
    controller.observe_transition(
        action=decision.action_name,
        grid_before=grid,
        grid_after=grid,
        available_actions=["ACTION1", "ACTION2"],
    )
    controller.on_reset()

    assert decision.action_name == "ACTION2"
    assert decision.source == "sage_t_joint_program"
    assert sage_t.observations == 1
    assert sage_t.branches == 1
