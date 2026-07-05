"""
Regression tests for the human-trace → TaskProgram → Actioner pipeline.

Pins the behavior validated in smoke runs on ar25:
  - probe subgoals get HARD-filtered prefer_actions
  - achieve / unknown subgoals get SOFT-reorder (all actions preserved)
  - movement alignment skips fall-through for probes (preserves
    hard-filter concentration during cycling fallback)

These tests must keep passing across refactors; if they fail, the
winning ar25 behavior has regressed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# Make the package importable without an install.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ==================================================================
# Fixtures
# ==================================================================

@pytest.fixture(scope="module")
def actioner():
    from v4_1_reasoning_system.arc_agi.actioner import Actioner
    return Actioner()


@pytest.fixture(scope="module")
def subgoal_cls():
    from v4_1_reasoning_system.arc_agi.goal_decomposer import SubGoal
    return SubGoal


def _make_sg(subgoal_cls, *, task_program_id: str, prefer_actions, max_actions: int):
    return subgoal_cls(
        id=0, priority=0,
        description="test", success_hint="x",
        max_actions=max_actions,
        metadata={
            "task_program_id": task_program_id,
            "prefer_actions": list(prefer_actions),
        },
    )


# ==================================================================
# _is_probe_subgoal — classification
# ==================================================================

def test_probe_id_classified_as_probe(actioner, subgoal_cls):
    sg = _make_sg(subgoal_cls, task_program_id="probe_change_player_action5",
                  prefer_actions=["ACTION5"], max_actions=8)
    assert actioner._is_probe_subgoal(sg) is True


def test_achieve_id_classified_as_non_probe(actioner, subgoal_cls):
    sg = _make_sg(subgoal_cls, task_program_id="achieve_correspond_shapes",
                  prefer_actions=["ACTION1", "ACTION2"], max_actions=60)
    assert actioner._is_probe_subgoal(sg) is False


def test_identify_id_classified_as_probe(actioner, subgoal_cls):
    sg = _make_sg(subgoal_cls, task_program_id="identify_boundary",
                  prefer_actions=["ACTION3"], max_actions=6)
    assert actioner._is_probe_subgoal(sg) is True


def test_no_id_small_budget_fallback_classified_as_probe(actioner, subgoal_cls):
    sg = subgoal_cls(
        id=0, priority=0, description="d", success_hint="x", max_actions=8,
        metadata={"prefer_actions": ["ACTION5"]},
    )
    assert actioner._is_probe_subgoal(sg) is True


def test_no_id_large_budget_classified_as_non_probe(actioner, subgoal_cls):
    sg = subgoal_cls(
        id=0, priority=0, description="d", success_hint="x", max_actions=40,
        metadata={"prefer_actions": ["ACTION1"]},
    )
    assert actioner._is_probe_subgoal(sg) is False


def test_none_subgoal_not_probe(actioner):
    assert actioner._is_probe_subgoal(None) is False


# ==================================================================
# _biased_available — hard vs soft
# ==================================================================

POOL = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"]


def test_probe_hard_filters_to_preferred_only(actioner, subgoal_cls):
    sg = _make_sg(subgoal_cls, task_program_id="probe_change_player_action5",
                  prefer_actions=["ACTION5"], max_actions=8)
    biased = actioner._biased_available(POOL, sg)
    assert biased == ["ACTION5"], (
        "Probe subgoals MUST hard-filter — this is what made ar25 win. "
        "Any change here needs to be validated against the smoke benchmark."
    )


def test_achieve_soft_reorders_keeps_all_actions(actioner, subgoal_cls):
    sg = _make_sg(subgoal_cls, task_program_id="achieve_correspond_shapes",
                  prefer_actions=["ACTION1", "ACTION2"], max_actions=60)
    biased = actioner._biased_available(POOL, sg)
    assert biased == ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"]
    assert set(biased) == set(POOL), "achieve bias must preserve every action"


def test_probe_empty_intersection_falls_back_to_full_pool(actioner, subgoal_cls):
    sg = _make_sg(subgoal_cls, task_program_id="probe_weird",
                  prefer_actions=["ACTION99"], max_actions=8)
    biased = actioner._biased_available(POOL, sg)
    assert biased == POOL, "empty preferred-intersection must not starve the agent"


def test_no_preference_returns_pool_unchanged(actioner, subgoal_cls):
    sg = subgoal_cls(id=0, priority=0, description="d", success_hint="x",
                     max_actions=30, metadata={})
    assert actioner._biased_available(POOL, sg) == POOL


def test_none_subgoal_returns_pool_unchanged(actioner):
    assert actioner._biased_available(POOL, None) == POOL


# ==================================================================
# Preferred-action normalisation
# ==================================================================

def test_preferred_actions_normalises_numeric_and_lowercase(actioner, subgoal_cls):
    sg = subgoal_cls(
        id=0, priority=0, description="d", success_hint="x", max_actions=8,
        metadata={"prefer_actions": ["5", "action2", "ACTION3"]},
    )
    pref = actioner._preferred_actions(sg)
    assert pref == ["ACTION5", "ACTION2", "ACTION3"]


# ==================================================================
# End-to-end: canonical ar25 TaskProgram attaches correctly
# ==================================================================

AR25_PATH = ROOT / "task_programs" / "ar25.json"


@pytest.mark.skipif(not AR25_PATH.exists(), reason="canonical ar25.json not present")
def test_canonical_ar25_loads_and_first_subgoal_is_action5_probe(actioner):
    """
    Smoke-tests the full handshake:
      TaskProgram JSON → GoalDecomposer._decompose_from_program
      → SubGoal with metadata task_program_id + prefer_actions
      → Actioner classifies it as probe and hard-filters to [ACTION5].

    This is the invariant that makes ar25 win. If this test fails,
    the ar25 smoke benchmark will almost certainly regress.
    """
    from human_trace.task_program import try_load_task_program
    from v4_1_reasoning_system.arc_agi.goal_decomposer import GoalDecomposer

    program = try_load_task_program(AR25_PATH.parent, "ar25")
    assert program is not None
    assert program.goal_family == "correspondence"
    assert program.subgoal_tests, "canonical program must have at least one subgoal_test"

    # Attach and decompose. `_decompose_from_program` does not use the
    # observation / memory arguments when a TaskProgram is attached, so
    # we can pass None for both (keeps this test GPU- and LLM-free).
    decomposer = GoalDecomposer()
    decomposer.set_task_program(program)
    goal = decomposer._decompose_from_program(
        program=program,
        observation=None,
        memory=None,
        previous_goal=None,
    )
    assert goal is not None
    assert goal.subgoals, "decomposition must produce subgoals"

    # Find the first subgoal that came from the program (skip the
    # injected "probe every action" warm-up subgoal with id=-1).
    program_subgoals = [sg for sg in goal.subgoals
                        if (sg.metadata or {}).get("task_program_id", "").startswith("probe_change_player_action5")]
    assert program_subgoals, "ACTION5 probe subgoal must be present"
    sg5 = program_subgoals[0]

    # Classify and bias.
    assert actioner._is_probe_subgoal(sg5) is True
    biased = actioner._biased_available(POOL, sg5)
    assert biased == ["ACTION5"], (
        "Regression: the ACTION5 probe must hard-filter to [ACTION5]. "
        "This is the exact bias that produced level-1 wins on ar25."
    )


def test_level_specific_loader_prefers_live_variant_and_prefix_match(tmp_path):
    from human_trace.task_program import (
        TaskProgram,
        SubgoalTest,
        try_load_task_program_for_level,
    )

    def _program(desc: str) -> TaskProgram:
        return TaskProgram(
            game_id="ar25-e3c63847",
            goal_family="correspondence",
            macro_goal=desc,
            subgoal_tests=[
                SubgoalTest(
                    id="achieve_progress",
                    description=desc,
                    verification="advance",
                    max_actions=40,
                    expected_signal="level_advance",
                    prefer_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
                )
            ],
        )

    _program("generic").save(tmp_path / "ar25.json")
    _program("level2 plain").save(tmp_path / "ar25.lvl2.json")
    _program("level2 live").save(tmp_path / "ar25.lvl2.live1.json")

    loaded = try_load_task_program_for_level(tmp_path, "ar25-e3c63847", 2)

    assert loaded is not None
    assert loaded.macro_goal == "level2 live"


def test_level_specific_program_skips_bootstrap_probe_above_level1():
    from human_trace.task_program import TaskProgram, SubgoalTest
    from v4_1_reasoning_system.arc_agi.goal_decomposer import GoalDecomposer
    from v4_1_reasoning_system.arc_agi.game_memory import GameMemory, ActionProfile

    program = TaskProgram(
        game_id="ar25-e3c63847",
        goal_family="correspondence",
        macro_goal="level 2",
        subgoal_tests=[
            SubgoalTest(
                id="probe_level2_action5_switch",
                description="probe",
                verification="switch",
                max_actions=8,
                expected_signal="role_switch",
                prefer_actions=["ACTION5"],
            ),
            SubgoalTest(
                id="achieve_level2_correspondence",
                description="achieve",
                verification="advance",
                max_actions=50,
                expected_signal="level_advance",
                prefer_actions=["ACTION2", "ACTION5"],
            ),
        ],
    )
    setattr(program, "_target_level", 2)
    setattr(program, "_attachment_kind", "level-specific")

    memory = GameMemory()
    memory.action_profiles["ACTION5"] = ActionProfile(
        action_name="ACTION5",
        times_tried=4,
        times_changed_grid=4,
    )

    goal = GoalDecomposer()._decompose_from_program(program, None, memory, None)

    assert goal is not None
    ids = [(sg.metadata or {}).get("task_program_id", "") for sg in goal.subgoals]
    assert "probe_level2_action5_switch" not in ids
    assert "achieve_level2_correspondence" in ids
    assert all(not sg.description.startswith("[human-program] Probe: try each available action")
               for sg in goal.subgoals)


def test_program_decomposition_exports_action_role_metadata():
    from human_trace.task_program import ActionRole, TaskProgram, SubgoalTest
    from v4_1_reasoning_system.arc_agi.goal_decomposer import GoalDecomposer
    from v4_1_reasoning_system.arc_agi.game_memory import GameMemory

    program = TaskProgram(
        game_id="ar25-e3c63847",
        goal_family="correspondence",
        macro_goal="level 2",
        anti_patterns=["do not spam ACTION5 without following it with movement"],
        action_roles=[
            ActionRole(
                action="ACTION5",
                role="control_switch",
                evidence="switches controlled entity",
                confidence=0.8,
            ),
            ActionRole(
                action="ACTION2",
                role="movement",
                evidence="moves through corridor",
                confidence=0.7,
            ),
        ],
        subgoal_tests=[
            SubgoalTest(
                id="achieve_level2_correspondence",
                description="switch once, then move",
                verification="advance",
                max_actions=50,
                expected_signal="level_advance",
                prefer_actions=["ACTION2", "ACTION5"],
            ),
        ],
    )
    setattr(program, "_target_level", 2)

    goal = GoalDecomposer()._decompose_from_program(program, None, GameMemory(), None)

    assert goal is not None
    metadata = goal.subgoals[0].metadata
    assert metadata["control_switch_actions"] == ["ACTION5"]
    assert metadata["movement_actions"] == ["ACTION2"]
    assert metadata["program_anti_patterns"] == [
        "do not spam ACTION5 without following it with movement"
    ]


def test_level1_program_keeps_bootstrap_probe():
    from human_trace.task_program import TaskProgram, SubgoalTest
    from v4_1_reasoning_system.arc_agi.actioner import Actioner
    from v4_1_reasoning_system.arc_agi.goal_decomposer import GoalDecomposer
    from v4_1_reasoning_system.arc_agi.game_memory import GameMemory

    program = TaskProgram(
        game_id="ar25-e3c63847",
        goal_family="correspondence",
        macro_goal="level 1",
        subgoal_tests=[
            SubgoalTest(
                id="achieve_match_shapes",
                description="achieve",
                verification="advance",
                max_actions=50,
                expected_signal="level_advance",
                prefer_actions=["ACTION2"],
            ),
        ],
    )
    setattr(program, "_target_level", 1)
    setattr(program, "_attachment_kind", "generic")

    goal = GoalDecomposer()._decompose_from_program(program, None, GameMemory(), None)

    assert goal is not None
    assert goal.subgoals
    assert goal.subgoals[0].description.startswith("[human-program] Probe: try each available action")
    assert goal.subgoals[0].metadata.get("probe") is True
    assert str(goal.subgoals[0].metadata.get("task_program_id", "")).startswith("probe_")
    assert Actioner()._is_probe_subgoal(goal.subgoals[0]) is True


# ==================================================================
# Compiler invariant: action-role preservation
# ==================================================================
# These tests pin the GENERAL invariant the user demanded:
#   "Every explicit ACTION<N> mechanic mentioned in human traces
#    must become a dedicated probe_* subgoal in the TaskProgram,
#    irrespective of which specific game it is."
# They are parameterised over fictitious mechanic strings so the
# behaviour cannot regress to an ar25-specific hard-code.

def _make_pack_with_mechanic(mechanic: str):
    """Build a minimal HumanPriorPack whose only ACTION evidence is
    the given mechanic string (e.g. 'change_player_action5')."""
    from human_trace.integration import HumanPriorPack
    from human_trace.schema import EpisodeRecord

    ep = EpisodeRecord(
        game_id="x", episode_id="ep0",
        started_at="t0", ended_at="t1",
        n_steps=1, final_state="WIN", levels_completed=1,
        objective_guess="reach goal",
        discovered_mechanics=[mechanic],
    )
    return HumanPriorPack(game_id="x", steps=[], episodes=[ep])


def _make_program_without(action: str, *, with_achieve: bool = True):
    """A TaskProgram whose subgoal_tests do NOT cover ``action``."""
    from human_trace.task_program import TaskProgram, SubgoalTest
    sgs = [
        SubgoalTest(
            id="probe_action1",
            description="probe ACTION1",
            verification="moves",
            max_actions=6,
            expected_signal="object_moved",
            prefer_actions=["ACTION1"],
        ),
    ]
    if with_achieve:
        sgs.append(SubgoalTest(
            id="achieve_progress",
            description="combine primitives",
            verification="level advances",
            max_actions=40,
            expected_signal="level_advance",
            prefer_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"],
        ))
    return TaskProgram(
        game_id="x",
        goal_family="correspondence",
        macro_goal="reach goal",
        subgoal_tests=sgs,
    )


@pytest.mark.parametrize(
    "mechanic, expected_action, expected_signal",
    [
        ("change_player_action5", "ACTION5", "role_switch"),
        ("action3_rotates_shape", "ACTION3", "object_moved"),
        ("click_red_action6",     "ACTION6", "click_triggered_change"),
        ("action7_opens_door",    "ACTION7", "click_triggered_change"),
    ],
)
def test_action_role_preservation_injects_missing_probe(
    mechanic, expected_action, expected_signal,
):
    """Compiler invariant: any ACTION<N> mentioned in trace evidence
    must end up as a dedicated probe in the TaskProgram, even if the
    LLM forgot to emit it."""
    from human_trace.compile_trace import _enforce_action_role_preservation

    pack = _make_pack_with_mechanic(mechanic)
    program = _make_program_without(expected_action)

    repaired, log = _enforce_action_role_preservation(program, pack)

    # Find the synthesized probe.
    matches = [sg for sg in repaired.subgoal_tests
               if list(sg.prefer_actions) == [expected_action]]
    assert matches, (
        f"invariant violated: ACTION mention {mechanic!r} did not "
        f"produce a dedicated probe for {expected_action} "
        f"(subgoals={[(s.id, s.prefer_actions) for s in repaired.subgoal_tests]})"
    )
    probe = matches[0]
    assert probe.id.startswith("probe_"), (
        "synthesized probe id must start with 'probe_' so the actioner "
        "applies the hard-filter (see Actioner._is_probe_subgoal)"
    )
    assert probe.expected_signal == expected_signal, (
        f"signal classification regressed: {mechanic!r} → expected "
        f"{expected_signal!r}, got {probe.expected_signal!r}"
    )
    # Repair log must record the injection so it shows up in compile logs.
    assert any(expected_action in m for m in log), log


def test_action_role_preservation_inserts_before_achieve():
    """Synthesized probes must land BEFORE the achieve subgoal, so
    primitive discovery happens before exploitation."""
    from human_trace.compile_trace import _enforce_action_role_preservation

    pack = _make_pack_with_mechanic("change_player_action5")
    program = _make_program_without("ACTION5", with_achieve=True)

    repaired, _ = _enforce_action_role_preservation(program, pack)

    ids = [sg.id for sg in repaired.subgoal_tests]
    achieve_idx = next(i for i, sid in enumerate(ids) if sid.startswith("achieve"))
    probe_idx = next(i for i, sid in enumerate(ids) if sid == "probe_action5")
    assert probe_idx < achieve_idx, (
        f"order regressed: synthesized probe at {probe_idx} must come "
        f"before achieve at {achieve_idx}; full order={ids}"
    )


def test_action_role_preservation_idempotent_when_already_covered():
    """If the LLM already emitted BOTH a dedicated probe AND an
    ActionRole entry for the action, the auto-repair must be a no-op
    (no duplicate injection on either surface)."""
    from human_trace.compile_trace import _enforce_action_role_preservation
    from human_trace.task_program import SubgoalTest, ActionRole

    pack = _make_pack_with_mechanic("change_player_action5")
    program = _make_program_without("ACTION5", with_achieve=True)
    # Hand-rolled probe + matching ActionRole, so both surfaces are
    # populated by the LLM.
    program.subgoal_tests.insert(1, SubgoalTest(
        id="probe_change_player_action5",
        description="use ACTION5 to switch controlled entity",
        verification="controlled entity changes",
        max_actions=8,
        expected_signal="role_switch",
        prefer_actions=["ACTION5"],
    ))
    program.action_roles.append(ActionRole(
        action="ACTION5",
        role="control_switch",
        evidence="change_player_action5",
        confidence=0.9,
    ))
    n_sgs_before = len(program.subgoal_tests)
    n_roles_before = len(program.action_roles)

    repaired, log = _enforce_action_role_preservation(program, pack)

    assert log == [], f"unexpected repair log: {log}"
    assert len(repaired.subgoal_tests) == n_sgs_before
    assert len(repaired.action_roles) == n_roles_before


def test_action_role_preservation_does_not_credit_achieve_membership():
    """Appearing in achieve.prefer_actions does NOT count as coverage —
    the actioner's hard-filter is per-subgoal, so a primitive only
    counts as 'probed' if it has its own dedicated single-action probe."""
    from human_trace.compile_trace import _enforce_action_role_preservation

    pack = _make_pack_with_mechanic("change_player_action5")
    program = _make_program_without("ACTION5", with_achieve=True)
    # ACTION5 is in achieve.prefer_actions but no dedicated probe.
    achieve = program.subgoal_tests[-1]
    assert "ACTION5" in achieve.prefer_actions, "test setup expects this"

    repaired, log = _enforce_action_role_preservation(program, pack)

    assert any("ACTION5" in m for m in log), (
        "ACTION5 only appearing in achieve.prefer_actions must STILL "
        "trigger probe injection: see _subgoal_pins_single_action contract"
    )
    matches = [sg for sg in repaired.subgoal_tests
               if list(sg.prefer_actions) == ["ACTION5"]]
    assert matches, "dedicated ACTION5 probe must be injected"


def test_action_roles_field_populated_from_evidence():
    """The new structured `action_roles` schema field must be filled
    from explicit trace evidence even when the LLM omitted it."""
    from human_trace.compile_trace import _enforce_action_role_preservation

    pack = _make_pack_with_mechanic("change_player_action5")
    program = _make_program_without("ACTION5", with_achieve=True)
    assert program.action_roles == []  # LLM omitted

    repaired, log = _enforce_action_role_preservation(program, pack)

    roles = {r.action: r for r in repaired.action_roles}
    assert "ACTION5" in roles
    assert roles["ACTION5"].role == "control_switch"
    assert "change_player_action5" in roles["ACTION5"].evidence
    assert any("ActionRole" in m and "ACTION5" in m for m in log)


def test_action_roles_llm_emitted_role_label_is_preserved():
    """If the LLM already supplied an ActionRole, keep its role label
    (the LLM may understand the semantics better than our regex)."""
    from human_trace.compile_trace import _enforce_action_role_preservation
    from human_trace.task_program import ActionRole

    pack = _make_pack_with_mechanic("change_player_action5")
    program = _make_program_without("ACTION5", with_achieve=True)
    # LLM said it's "interact" (semantically wrong, but pinning intent).
    program.action_roles.append(ActionRole(
        action="ACTION5",
        role="interact",
        evidence="change_player_action5",
        confidence=0.6,
    ))

    repaired, _ = _enforce_action_role_preservation(program, pack)

    roles = {r.action: r for r in repaired.action_roles}
    assert roles["ACTION5"].role == "interact", (
        "LLM-emitted role label must NOT be overwritten by the heuristic "
        "classifier (the LLM is the source of truth for semantics)"
    )


# ==================================================================
# Invariant 3: goal-object preservation
# ==================================================================

@pytest.mark.parametrize(
    "macro_goal, expected_pairs",
    [
        ("Match yellow and purple shapes",
         [("yellow", "shapes"), ("purple", "shapes")]),
        ("Click the red dot",
         [("red", "dots")]),
        # Colour without a recognised noun → emit (color, None); the
        # downstream entity-preservation pass will inject an entity
        # named after just the colour. Pinned because dropping the
        # colour entirely silently throws away goal-relevant info.
        ("Avoid the green hazard",
         [("green", None)]),
        ("Move the blue block to the goal",
         [("blue", "blocks")]),
        ("Just go to the goal",
         []),
        ("",
         []),
    ],
)
def test_extract_macro_goal_objects(macro_goal, expected_pairs):
    """Deterministic extractor: which (color, noun) pairs appear in
    the macro_goal text. Pinned over a range of phrasings so the
    invariant generalises across games."""
    from human_trace.compile_trace import _extract_macro_goal_objects
    pairs = _extract_macro_goal_objects(macro_goal)
    # We accept (color, None) as well-formed (just a color mentioned),
    # but the parametrisation here only lists pairs we expect to fire.
    assert pairs == expected_pairs, (
        f"macro_goal={macro_goal!r} → expected {expected_pairs}, got {pairs}"
    )


def test_entity_preservation_injects_missing_colored_objects():
    """Entities referenced by colour in macro_goal MUST appear in
    `program.entities` after the post-compile pass."""
    from human_trace.compile_trace import _enforce_entity_preservation
    from human_trace.task_program import TaskProgram, Entity

    program = TaskProgram(
        game_id="x",
        goal_family="correspondence",
        macro_goal="Match yellow and purple shapes",
        entities=[Entity(name="player", role="player")],  # missing the colors!
    )

    repaired, log = _enforce_entity_preservation(program)

    names = [e.name for e in repaired.entities]
    assert any("yellow" in n for n in names), names
    assert any("purple" in n for n in names), names
    # The original player entity must be preserved.
    assert "player" in names
    # Repair log must be informative.
    assert any("yellow" in m for m in log)
    assert any("purple" in m for m in log)


def test_entity_preservation_idempotent_when_already_present():
    """If the LLM already emitted entities matching the colour terms,
    no new entities are injected."""
    from human_trace.compile_trace import _enforce_entity_preservation
    from human_trace.task_program import TaskProgram, Entity

    program = TaskProgram(
        game_id="x",
        goal_family="correspondence",
        macro_goal="Match yellow and purple shapes",
        entities=[
            Entity(name="yellow_shape", role="movable"),
            Entity(name="purple_shape", role="reference"),
        ],
    )
    n_before = len(program.entities)

    repaired, log = _enforce_entity_preservation(program)

    assert log == []
    assert len(repaired.entities) == n_before


def test_entity_preservation_noop_when_macro_goal_has_no_objects():
    """A macro_goal without colour/object words leaves entities alone."""
    from human_trace.compile_trace import _enforce_entity_preservation
    from human_trace.task_program import TaskProgram, Entity

    program = TaskProgram(
        game_id="x", goal_family="navigation",
        macro_goal="Reach the exit",
        entities=[Entity(name="player", role="player")],
    )

    repaired, log = _enforce_entity_preservation(program)

    assert log == []
    assert [e.name for e in repaired.entities] == ["player"]


# ==================================================================
# Invariant 4: specific achieve subgoal id
# ==================================================================

@pytest.mark.parametrize(
    "goal_family, generic_id, expected_id",
    [
        ("correspondence", "achieve_progress", "achieve_correspondence"),
        ("navigation",     "achieve_goal",     "achieve_navigation"),
        ("collection",     "achieve",          "achieve_collection"),
        ("click_sequence", "achieve_level",    "achieve_click_sequence"),
    ],
)
def test_specific_achieve_id_renames_generic_when_family_known(
    goal_family, generic_id, expected_id,
):
    """When goal_family is concrete, generic achieve ids get renamed."""
    from human_trace.compile_trace import _enforce_specific_achieve_id
    from human_trace.task_program import TaskProgram, SubgoalTest

    program = TaskProgram(
        game_id="x",
        goal_family=goal_family,
        macro_goal="g",
        subgoal_tests=[SubgoalTest(
            id=generic_id,
            description="d", verification="v",
            max_actions=40, expected_signal="level_advance",
            prefer_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
        )],
    )

    repaired, log = _enforce_specific_achieve_id(program)

    assert repaired.subgoal_tests[0].id == expected_id, (
        f"goal_family={goal_family} generic {generic_id!r} should rename "
        f"to {expected_id!r}, got {repaired.subgoal_tests[0].id!r}"
    )
    assert any(generic_id in m for m in log)


def test_specific_achieve_id_noop_when_already_specific():
    """A descriptive achieve id like 'achieve_match_shapes' is left alone."""
    from human_trace.compile_trace import _enforce_specific_achieve_id
    from human_trace.task_program import TaskProgram, SubgoalTest

    program = TaskProgram(
        game_id="x", goal_family="correspondence", macro_goal="g",
        subgoal_tests=[SubgoalTest(
            id="achieve_match_shapes",
            description="d", verification="v",
            max_actions=40, expected_signal="level_advance",
            prefer_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
        )],
    )

    repaired, log = _enforce_specific_achieve_id(program)

    assert log == []
    assert repaired.subgoal_tests[0].id == "achieve_match_shapes"


def test_specific_achieve_id_noop_when_family_unknown():
    """If the LLM gave up and set goal_family='unknown', don't rename
    — there's no semantic content to inject."""
    from human_trace.compile_trace import _enforce_specific_achieve_id
    from human_trace.task_program import TaskProgram, SubgoalTest

    program = TaskProgram(
        game_id="x", goal_family="unknown", macro_goal="g",
        subgoal_tests=[SubgoalTest(
            id="achieve_progress",
            description="d", verification="v",
            max_actions=40, expected_signal="level_advance",
            prefer_actions=["ACTION1", "ACTION2", "ACTION3", "ACTION4"],
        )],
    )

    repaired, log = _enforce_specific_achieve_id(program)

    assert log == []
    assert repaired.subgoal_tests[0].id == "achieve_progress"
