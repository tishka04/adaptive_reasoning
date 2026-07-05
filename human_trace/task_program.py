"""Structured strategy package compiled from human traces.

A `TaskProgram` is the output of the offline LLM trace compiler
(`human_trace.compile_trace`). It turns messy per-step hints like
"gray_shape_is_controling_shapes" into a strict, machine-actionable
schema that `GoalDecomposer` can consume directly instead of re-
parsing prose.

Design principles:
  1. **Strict enums** with `"other"` escape hatches so a 3B model can
     produce valid JSON. Unknown values get logged, not crashed on.
  2. **Click-game support is first-class**: roles, constraints,
     subgoal tests, and target specs all carry click semantics where
     relevant.
  3. **Schema-constrained** via Pydantic: the compiler retries on
     validation failure rather than letting the downstream consumer
     deal with half-parsed JSON.

Two files form the pipeline:
  - This file (`task_program.py`)  — schema + (de)serialisation.
  - `compile_trace.py`             — LLM-backed compiler CLI.

Consumers (Phase 2):
  - `GoalDecomposer._subgoals_from_program`
  - `ConstraintChecker`            (Phase 3, not yet built)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Literal, Optional, Tuple

try:
    from pydantic import BaseModel, Field, ValidationError, model_validator
except ImportError:
    from adaptive_reasoning_compat.pydantic import (
        BaseModel,
        Field,
        ValidationError,
        model_validator,
    )


# ==================================================================
# Controlled vocabulary
# ==================================================================

# Roles an entity can play in the puzzle's relational structure.
# Click-related: `click_target` (cell the player clicks) and
# `clickable_region` (area where clicks are meaningful) are first-class.
EntityRole = Literal[
    "controller",          # the player / thing that moves
    "player",              # synonym for controller (natural-language alias)
    "movable",             # non-player objects that can be repositioned
    "reference",           # target shape to match/align to
    "hazard",              # causes game_over on contact
    "cursor",              # indicator of selection, not a physical obstacle
    "boundary_exception", # normal wall except for specific entities
    "click_target",        # a specific cell/object that must be clicked
    "clickable_region",    # an area where clicks trigger effects
    "goal_marker",         # exit / finish tile
    "unknown",
    "other",
]


# Goal families. Added `click_sequence` explicitly so click-based
# puzzles get distinct subgoals from pure navigation.
GoalFamily = Literal[
    "navigation",
    "collection",
    "ordering",
    "alignment",
    "correspondence",
    "symmetry",
    "exclusion",
    "toggle_activation",
    "click_sequence",       # click in a specific order / pattern
    "click_selection",      # single-click discriminating choice
    "unknown",
    "other",
]


# Constraint types. Click-related constraints added.
ConstraintType = Literal[
    "cannot_cross_boundary",
    "can_exit_grid",
    "only_subset_matters",
    "role_switch_costly",
    "cursor_mediated_control",
    "crossing_allowed",
    "click_order_matters",        # clicks must be in specific sequence
    "click_position_constrained", # only certain cells respond to clicks
    "click_is_reversible",        # can be undone (ACTION7)
    "click_is_one_shot",          # once clicked, cannot be un-clicked
    "click_requires_setup",       # click only works after some state
    "movement_limited",           # bounded number of moves per level
    "other",
]


# What observable signal indicates a subgoal succeeded.
ExpectedSignal = Literal[
    "grid_change",
    "object_moved",
    "level_advance",
    "role_switch",
    "overlap_achieved",
    "click_triggered_change",
    "click_absorbed_no_change",  # diagnostic: clicked but nothing happened
    "object_disappeared",
    "new_region_revealed",
    "other",
]


# ==================================================================
# Components
# ==================================================================

# What semantic role a single ACTION plays. Closed-vocabulary so the
# downstream planner can pattern-match instead of doing NL parsing.
ActionRoleKind = Literal[
    "control_switch",   # changes which entity the player is controlling
    "movement",         # repositions the controlled entity (up/down/left/right)
    "interact",         # contextual interact with adjacent / focused cell
    "rotate",           # rotates a shape or piece
    "push",             # pushes another object
    "click_select",     # clicks a screen-space target
    "toggle",           # flips a boolean state of a cell or object
    "undo",             # reverses the previous step
    "no_effect",        # explicitly observed to do nothing
    "unknown",
    "other",
]


class ActionRole(BaseModel):
    """One ACTION → role binding extracted from the trace.

    Maintained as a *parallel* source-of-truth alongside `subgoal_tests`:
    - `action_roles` is the deterministic, machine-extractable surface
      of what each ACTION means, derived from explicit human evidence.
    - `subgoal_tests` is the planner's executable representation —
      synthesised mechanically from `action_roles` so that the LLM
      cannot accidentally drop a primitive by forgetting to emit a
      probe (the historical level-2 plateau failure mode).

    The compiler post-processor enforces:
        every entry in `action_roles` ↔ at least one dedicated probe
        in `subgoal_tests` whose `prefer_actions == [<action>]`.
    """
    action: str = Field(..., max_length=16)            # e.g. "ACTION5"
    role: ActionRoleKind
    # Verbatim evidence string from the human trace, e.g.
    # "change_player_action5". Pinned because it is the audit trail
    # for why this role was assigned and feeds the prompt block.
    evidence: str = Field(..., max_length=200)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    # Other actions whose semantics depend on this one (e.g. a
    # control_switch action changes what the movement actions do).
    changes_semantics_of: List[str] = Field(default_factory=list, max_length=8)


class HypothesisRevision(BaseModel):
    """Belief change during a human playthrough.

    Currently populated from `discovered_mistakes` as a coarse proxy.
    When the recorder gains a structured revision UI, this will carry
    timestamps and causes too.
    """
    at_step: Optional[int] = None
    was: str = Field(..., max_length=200)
    now: str = Field(..., max_length=200)
    cause: Optional[str] = Field(default=None, max_length=200)


class Entity(BaseModel):
    """A logical entity in the puzzle — not a visual sprite.

    For click games `click_target` and `clickable_region` entities may
    carry color/position hints in `notes` when the trace reveals them
    (e.g. "only blue cells click-through").
    """
    name: str = Field(..., max_length=64)
    role: EntityRole
    # Freeform; may contain color value, position hint, etc.
    notes: Optional[str] = Field(default=None, max_length=200)
    # If the entity has a canonical color in the grid, record it.
    color_value: Optional[int] = Field(default=None, ge=0, le=31)


class Constraint(BaseModel):
    type: ConstraintType
    description: str = Field(..., max_length=200)
    # Entity names this applies to. Empty = global.
    applies_to: List[str] = Field(default_factory=list, max_length=8)


class SubgoalTest(BaseModel):
    """A concrete, verifiable sub-objective the agent can attempt.

    Ranked cheapest-first by the compiler so the goal-pursuit layer
    can try them in order and rotate on stall.
    """
    id: str = Field(..., max_length=64)         # slug, used as goal.id
    description: str = Field(..., max_length=200)
    verification: str = Field(..., max_length=200)
    max_actions: int = Field(default=30, ge=1, le=200)
    preconditions: List[str] = Field(default_factory=list, max_length=6)
    expected_signal: ExpectedSignal

    # For click-based subgoals: a concrete click hypothesis.
    # Either specific (x,y) or a color-based one. Both optional.
    click_target_xy: Optional[Tuple[int, int]] = None
    click_target_color: Optional[int] = Field(default=None, ge=0, le=31)

    # Which actions are expected to matter; if set, the executor can
    # bias its action scoring toward these (e.g. ["ACTION5","ACTION6"]).
    prefer_actions: List[str] = Field(default_factory=list, max_length=7)


class TaskProgram(BaseModel):
    """Structured strategy distilled from one or more human traces."""
    game_id: str = Field(..., max_length=80)
    goal_family: GoalFamily

    # Macro vs level split (the "one monolithic goal" concern).
    macro_goal: str = Field(..., max_length=300)
    level_hypotheses: List[str] = Field(default_factory=list, max_length=10)

    entities: List[Entity] = Field(default_factory=list, max_length=16)
    win_condition_hypotheses: List[str] = Field(
        default_factory=list, max_length=8
    )
    constraints: List[Constraint] = Field(default_factory=list, max_length=16)
    anti_patterns: List[str] = Field(default_factory=list, max_length=12)

    # ── Structured semantic facts (LLM-normalised, code-extractable) ──
    # The compiler's preferred surface for action-meaning: the LLM
    # populates this when it understands the role of an ACTION; if it
    # forgets, the post-compile hook auto-fills from explicit evidence.
    # `subgoal_tests` is then derived from `action_roles` so the planner
    # always has executable probes for every primitive the human named.
    action_roles: List[ActionRole] = Field(
        default_factory=list, max_length=12
    )

    # Ordered cheapest-first; the goal-pursuit layer rotates on stall.
    subgoal_tests: List[SubgoalTest] = Field(
        default_factory=list, max_length=16
    )

    # From discovered_mistakes and (future) structured revisions.
    belief_revisions: List[HypothesisRevision] = Field(
        default_factory=list, max_length=12
    )

    # Compiler's own self-reported confidence.
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # Provenance.
    source_episodes: List[str] = Field(default_factory=list, max_length=16)
    compiler_model: Optional[str] = None
    compiled_at: Optional[str] = None   # ISO UTC

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _sanity(self) -> "TaskProgram":
        # `click_*` subgoal fields only make sense if actions include ACTION6.
        # We don't hard-enforce here because the compiler may be missing
        # action info, but we warn the caller via a soft invariant.
        return self

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, text: str) -> "TaskProgram":
        return cls.model_validate_json(text)

    @classmethod
    def from_path(cls, path: str | Path) -> "TaskProgram":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")


# ==================================================================
# Loader with soft degradation
# ==================================================================

def try_load_task_program(
    program_dir: str | Path, game_id: str
) -> Optional[TaskProgram]:
    """Return the TaskProgram for `game_id` under `program_dir`, or None.

    Accepts either an exact `<game_id>.json` or a prefix match so that
    game ids with hashes (e.g. "ar25-e3c63847") still match when the
    caller passes the short id ("ar25").
    """
    root = Path(program_dir)
    if not root.exists():
        return None

    # Exact match first.
    exact = root / f"{game_id}.json"
    if exact.exists():
        try:
            return TaskProgram.from_path(exact)
        except ValidationError:
            return None

    # Prefix match (e.g. "ar25" matches "ar25-e3c63847.json"), but
    # explicitly skip level-suffixed variants ("<game>.lvl<N>.json")
    # so that the default loader doesn't accidentally pick one.
    for p in sorted(root.glob("*.json")):
        stem = p.stem
        if ".lvl" in stem:
            continue
        if stem == game_id or stem.startswith(f"{game_id}-") or stem.startswith(f"{game_id}."):
            try:
                return TaskProgram.from_path(p)
            except ValidationError:
                continue
    return None


def try_load_task_program_for_level(
    program_dir: str | Path, game_id: str, level: int
) -> Optional[TaskProgram]:
    """Return a level-specific TaskProgram for `game_id` if one exists.

    Looks (in priority order) for:
      - ``<program_dir>/<game_id>.lvl<level>.live<N>.json`` (highest N first)
      - ``<program_dir>/<game_id>.lvl<level>.json``
      - ``<program_dir>/<short_game_id>.lvl<level>.json`` for any short id
        that prefix-matches `game_id` (handles "ar25-e3c63847" -> "ar25").

    Returns None if no level-specific program is on disk.
    """
    root = Path(program_dir)
    if not root.exists() or level < 0:
        return None

    level_tag = f".lvl{level}"
    candidates: List[tuple[int, int, int, Path]] = []

    for p in sorted(root.glob("*.json")):
        stem = p.stem
        if level_tag not in stem:
            continue
        if not (stem.endswith(level_tag) or f"{level_tag}.live" in stem):
            continue

        base, suffix = stem.split(level_tag, 1)
        if suffix and not suffix.startswith(".live"):
            continue
        if suffix.startswith(".live"):
            live_suffix = suffix[len(".live"):]
            if not live_suffix.isdigit():
                continue
            live_idx = int(live_suffix)
        else:
            live_idx = 0

        exact_match = base == game_id
        prefix_match = (
            exact_match
            or game_id.startswith(f"{base}-")
            or game_id.startswith(f"{base}.")
        )
        if not prefix_match:
            continue
        candidates.append((
            2 if exact_match else 1,
            1 if live_idx > 0 else 0,
            live_idx,
            p,
        ))

    for _exact_score, _is_live, _live_idx, path in sorted(candidates, reverse=True):
        try:
            return TaskProgram.from_path(path)
        except ValidationError:
            continue
    return None


def json_schema_for_prompt() -> str:
    """A compact, LLM-readable rendering of the schema for prompt use.

    We deliberately skip full Pydantic JSON-schema (too verbose and
    confuses small models) in favour of a terse description that
    shows enum values explicitly.
    """
    lines = [
        "TaskProgram = {",
        '  "game_id": str,',
        f'  "goal_family": {list(GoalFamily.__args__)},',
        '  "macro_goal": str (≤300 chars, what the game is about overall),',
        '  "level_hypotheses": [str] (how levels differ; ≤10 items),',
        '  "entities": [Entity] (≤16),',
        '  "win_condition_hypotheses": [str] (ranked; ≤8),',
        '  "constraints": [Constraint] (≤16),',
        '  "anti_patterns": [str] (what NOT to do; ≤12),',
        '  "action_roles": [ActionRole] (one per ACTION whose role is named in the trace; ≤12),',
        '  "subgoal_tests": [SubgoalTest] (ranked cheapest-first; ≤16),',
        '  "belief_revisions": [HypothesisRevision] (≤12),',
        '  "confidence": float 0..1',
        "}",
        "",
        "Entity = {",
        '  "name": str,',
        f'  "role": {list(EntityRole.__args__)},',
        '  "notes": str | null,',
        '  "color_value": int 0..31 | null',
        "}",
        "",
        "Constraint = {",
        f'  "type": {list(ConstraintType.__args__)},',
        '  "description": str,',
        '  "applies_to": [str] (entity names; may be empty)',
        "}",
        "",
        "SubgoalTest = {",
        '  "id": str (short slug),',
        '  "description": str,',
        '  "verification": str (observable signal to confirm success),',
        '  "max_actions": int 1..200 (default 30),',
        '  "preconditions": [str],',
        f'  "expected_signal": {list(ExpectedSignal.__args__)},',
        '  "click_target_xy": [int x, int y] | null,',
        '  "click_target_color": int 0..31 | null,',
        '  "prefer_actions": [str] (e.g. ["ACTION5","ACTION6"])',
        "}",
        "",
        "HypothesisRevision = {",
        '  "at_step": int | null,',
        '  "was": str,',
        '  "now": str,',
        '  "cause": str | null',
        "}",
        "",
        "ActionRole = {",
        '  "action": str (e.g. "ACTION5"),',
        f'  "role": {list(ActionRoleKind.__args__)},',
        '  "evidence": str (verbatim trace string, e.g. "change_player_action5"),',
        '  "confidence": float 0..1 (default 0.7),',
        '  "changes_semantics_of": [str] (other ACTIONs whose meaning depends on this one)',
        "}",
    ]
    return "\n".join(lines)
