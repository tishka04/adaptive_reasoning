from __future__ import annotations

from theory.sage_t.causal.archive import ArchiveEdge, GoExploreArchive
from theory.sage_t.causal.contracts import GroundedAction
from theory.sage_t.causal.novelty import FEATURE_DIM, OnlineNoveltyPredictor
from theory.sage_t.causal.terminal_shield import MultiStepTerminalShield
from theory.sage_t.contracts import AbstractState, GroundFact


def state(index: int) -> AbstractState:
    return AbstractState(
        true_facts=frozenset({GroundFact("changed", (f"e{index}",))}),
        counters=(("progress", float(index)),),
    )


def test_symbolic_archive_preserves_exact_variants_and_prefixes() -> None:
    archive = GoExploreArchive(maximum_cells=8)
    action = GroundedAction("ACTION1")
    root, novel = archive.observe_state(
        state=state(0),
        exact_hash="raw-a",
        level=0,
        legal_actions=(action,),
    )
    assert novel
    same, novel = archive.observe_state(
        state=state(0),
        exact_hash="raw-b",
        level=0,
        legal_actions=(action,),
    )
    assert not novel
    assert same.cell_id == root.cell_id
    assert set(root.variants) == {"raw-a", "raw-b"}

    edge = archive.add_transition(
        source_cell_id=root.cell_id,
        source_exact_hash="raw-a",
        action=action,
        target_state=state(1),
        target_exact_hash="raw-c",
        target_level=0,
        target_legal_actions=(action,),
        terminal=False,
        success=False,
        changed=True,
    )
    target = archive.cells[edge.target_cell_id]
    assert archive.prefixes.actions(target.variants["raw-c"].prefix_id) == (action,)
    restored = GoExploreArchive.from_dict(archive.to_dict())
    assert restored.to_dict() == archive.to_dict()


def _edge(index: int, *, terminal: bool = False, progress: bool = False) -> ArchiveEdge:
    action = GroundedAction(f"ACTION{index + 1}")
    return ArchiveEdge(
        edge_id=f"edge-{index}",
        ordinal=index,
        source_cell_id=f"cell-{index}",
        source_exact_hash=f"exact-{index}",
        action=action,
        target_cell_id=f"cell-{index + 1}",
        target_exact_hash=f"exact-{index + 1}",
        level_delta=int(progress),
        terminal=terminal,
        success=progress,
        changed=True,
        novel=True,
        prefix_id=f"prefix-{index}",
    )


def test_terminal_shield_propagates_a_confirmed_delayed_failure() -> None:
    trace = tuple(_edge(index, terminal=index == 32) for index in range(33))
    shield = MultiStepTerminalShield(horizon=64, minimum_support=2)
    supports = shield.record_terminal_trace(trace, exact_replay_confirmed=True)
    assert supports[0].maximum_failure_distance == 33
    assert not shield.allows(trace[0].source_cell_id, trace[0].action)
    assert shield.metrics()["multi_step_hazard_observed"]

    shield.observe_progress(_edge(0, progress=True))
    assert shield.allows(trace[0].source_cell_id, trace[0].action)


def test_novelty_predictor_is_small_symbolic_and_checkpointable(tmp_path) -> None:
    predictor = OnlineNoveltyPredictor(seed=7, batch_size=2)
    action = GroundedAction("ACTION1", {"x": 4, "y": 5})
    prediction = predictor.observe(
        state(0), action, changed=True, novel=True, update=True
    )
    predictor.observe(state(1), action, changed=False, novel=False, update=True)
    assert 0.0 <= prediction.change_probability <= 1.0
    assert predictor.parameter_count < 15_000
    assert predictor.model.network[0].in_features == FEATURE_DIM
    checkpoint = tmp_path / "novelty.pt"
    predictor.save(checkpoint)
    restored = OnlineNoveltyPredictor.load(checkpoint)
    assert restored.parameter_count == predictor.parameter_count
    assert len(restored.examples) == 2
