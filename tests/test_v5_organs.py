"""Tests for the V5 reactive organs ported from the greedy controller:
danger memory (lethal-action veto) and anti-attractor (no-op ban + escape).
"""

import numpy as np
import pytest

from v5.control.anti_attractor import AntiAttractor
from v5.control.danger_memory import DangerMemoryV5, action_key
from v5.schemas import FrameDiff, GameObservation, PrimitiveAction


# ----------------------------------------------------------------------
# DangerMemoryV5
# ----------------------------------------------------------------------
def test_action_key_distinguishes_clicks():
    assert action_key(PrimitiveAction("ACTION1")) == "ACTION1"
    assert action_key(PrimitiveAction("ACTION6", x=3, y=4)) == "ACTION6@3,4"


def test_danger_memory_vetoes_only_observed_lethal_pairs():
    mem = DangerMemoryV5()
    mem.record_primitive_death(123, PrimitiveAction("ACTION2"))

    assert mem.is_primitive_lethal(123, PrimitiveAction("ACTION2"))
    # Same action, different state -> not lethal.
    assert not mem.is_primitive_lethal(999, PrimitiveAction("ACTION2"))
    # Different action, same state -> not lethal.
    assert not mem.is_primitive_lethal(123, PrimitiveAction("ACTION3"))
    assert "ACTION2" in mem.lethal_names(123)
    assert mem.lethal_names(123, available=["ACTION3"]) == set()


def test_danger_memory_roundtrip_serialisation():
    mem = DangerMemoryV5()
    mem.record_death(7, "ACTION1")
    mem.record_death(8, "ACTION6@1,2")
    restored = DangerMemoryV5.from_dict(mem.to_dict())
    assert restored.is_lethal(7, "ACTION1")
    assert restored.is_lethal(8, "ACTION6@1,2")
    assert len(restored) == 2


# ----------------------------------------------------------------------
# AntiAttractor — channel 1 (no-op ban)
# ----------------------------------------------------------------------
def test_noop_ban_after_threshold():
    aa = AntiAttractor(no_effect_ban_after=3)
    for _ in range(2):
        aa.note_no_effect(42, "ACTION1")
    assert not aa.is_banned_noop(42, "ACTION1")
    aa.note_no_effect(42, "ACTION1")
    assert aa.is_banned_noop(42, "ACTION1")
    # Different state not affected.
    assert not aa.is_banned_noop(43, "ACTION1")


# ----------------------------------------------------------------------
# AntiAttractor — channel 2 (preventive escape)
# ----------------------------------------------------------------------
def test_should_escape_on_strict_noop_stall():
    aa = AntiAttractor(stagnation_escape_after=8, escape_cooldown=5)
    for step in range(8):
        aa.observe(grid_hash=1, action_name="ACTION1", is_noop=True)
    assert aa.should_escape(current_step=100)


def test_should_escape_on_repeat_loop():
    aa = AntiAttractor(repeat_window=8, repeat_max_distinct=2)
    # 8 actions, only 2 distinct, but states keep changing (not a strict stall).
    for i in range(8):
        name = "ACTION1" if i % 2 == 0 else "ACTION2"
        aa.observe(grid_hash=i, action_name=name, is_noop=False)
    assert aa.should_escape(current_step=100)


def test_should_escape_on_novelty_stall():
    aa = AntiAttractor(novelty_window=12, novelty_min_unique=3)
    # 12 states cycling over only 3 unique hashes, with varied actions.
    names = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
    for i in range(12):
        aa.observe(grid_hash=i % 3, action_name=names[i % 4], is_noop=False)
    assert aa.should_escape(current_step=100)


def test_escape_respects_cooldown():
    aa = AntiAttractor(stagnation_escape_after=8, escape_cooldown=5)
    for _ in range(8):
        aa.observe(grid_hash=1, action_name="ACTION1", is_noop=True)
    aa.note_escape(current_step=100)
    assert not aa.should_escape(current_step=103)  # within cooldown
    assert aa.should_escape(current_step=106)       # past cooldown


def test_pick_escape_avoids_lethal_recent_and_clicks():
    aa = AntiAttractor(escape_ban_last_k=2)
    # Recent actions ACTION1, ACTION2 -> banned; ACTION3 lethal; expect ACTION4.
    for name in ["ACTION1", "ACTION2"]:
        aa.observe(grid_hash=5, action_name=name, is_noop=False)
    lethal = {("5", "ACTION3")}

    def is_lethal(gh, a):
        return (str(gh), a) in lethal

    choice = aa.pick_escape_action(
        available=["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION6"],
        grid_hash=5,
        is_lethal=is_lethal,
    )
    assert choice == "ACTION4"  # non-click, not banned, not lethal


def test_pick_escape_falls_back_to_click_when_only_option():
    aa = AntiAttractor()
    choice = aa.pick_escape_action(
        available=["ACTION6"], grid_hash=1, is_lethal=lambda gh, a: False
    )
    assert choice == "ACTION6"


# ----------------------------------------------------------------------
# Agent-level guard integration
# ----------------------------------------------------------------------
def _obs(grid_hash=1, actions=None):
    return GameObservation(
        raw_grid=np.zeros((4, 4), dtype=np.int32),
        grid_hash=grid_hash,
        game_state="NOT_FINISHED",
        levels_completed=0,
        available_actions=actions or ["ACTION1", "ACTION2", "ACTION3"],
    )


def _make_agent():
    pytest.importorskip("v5.adaptive_reasoning_agent_v5")
    from v5.adaptive_reasoning_agent_v5 import AdaptiveReasoningAgentV5

    return AdaptiveReasoningAgentV5(cross_game=None)


def test_guard_vetoes_lethal_chosen_action():
    agent = _make_agent()
    obs = _obs(grid_hash=77, actions=["ACTION1", "ACTION2", "ACTION3"])
    agent.danger_memory.record_primitive_death(77, PrimitiveAction("ACTION1"))

    guarded = agent._guard_action(obs, PrimitiveAction("ACTION1"))
    assert guarded.name != "ACTION1"
    assert guarded.name in obs.available_actions
    assert agent._danger_vetoes == 1


def test_guard_passes_safe_action_unchanged():
    agent = _make_agent()
    obs = _obs(grid_hash=5)
    guarded = agent._guard_action(obs, PrimitiveAction("ACTION2"))
    assert guarded.name == "ACTION2"
    assert agent._danger_vetoes == 0


def test_record_transition_learns_wall_on_game_over():
    agent = _make_agent()
    prev = _obs(grid_hash=200)
    agent._prev_obs = prev
    agent._level_action_trace = [PrimitiveAction("ACTION2")]
    after = _obs(grid_hash=201)
    after.frame_diff = FrameDiff(
        changed_cells=[], changed_values_before=[], changed_values_after=[],
        created_objects=[], removed_objects=[], moved_objects=[],
        game_over=True, num_changed=0,
    )
    agent._record_transition(after, "GAME_OVER")
    assert agent.danger_memory.is_lethal(200, "ACTION2")
