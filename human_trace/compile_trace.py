"""Offline trace → TaskProgram compiler.

Uses a local causal LM (default: Qwen2.5-3B-Instruct) to distil the
rich semantic content of human traces (objective guesses, discovered
mechanics, mistakes, sticky per-step hypotheses) into the strict
`TaskProgram` schema.

This runs **once per game, offline**. The resulting JSON file is then
consumed at runtime by `GoalDecomposer` (Phase 2).

Usage:
    python -m human_trace.compile_trace \\
        --traces human_traces \\
        --game ar25 \\
        --out task_programs/ar25.json \\
        [--model Qwen/Qwen2.5-3B-Instruct] \\
        [--device cpu|cuda] \\
        [--max-retries 3]

Design notes:
  - Retry-until-valid loop: if the LLM returns malformed JSON or
    violates the schema, we feed the error back and try again.
  - Evidence is aggregated from ALL episodes for the game, with raw
    step records summarised (not included verbatim) to keep the
    prompt small.
  - Sticky hypothesis *changes* across consecutive steps are
    surfaced as candidate belief revisions.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .integration import HumanPriorPack, build_prior_pack, _summarise_actions, _derive_goal_hints, _derive_hypothesis_priors
from .loader import load_traces
from .task_program import TaskProgram, json_schema_for_prompt


def filter_pack_by_min_level(pack: HumanPriorPack, min_level: int) -> HumanPriorPack:
    """Return a subset pack containing only the parts of the human trace
    that pertain to level `min_level + 1` and above.

    - Steps kept: those with ``levels_completed_after >= min_level``.
      That includes steps taken on the level the human is *currently
      working on after* having completed `min_level` previous levels.
    - Episodes kept: those whose final ``levels_completed >= min_level``
      (i.e., the human got at least past level `min_level`).
    - Action stats / goal hints / hypotheses are recomputed on the
      filtered view.
    """
    if min_level <= 0:
        return pack
    sub_steps = [s for s in pack.steps if s.levels_completed_after >= min_level]
    sub_eps = [e for e in pack.episodes if e.levels_completed >= min_level]
    if not sub_steps and not sub_eps:
        # Nothing to compile from; return an empty (but valid) pack.
        return HumanPriorPack(game_id=pack.game_id)
    goal_hints, failure_hints = _derive_goal_hints(sub_eps, sub_steps)
    return HumanPriorPack(
        game_id=pack.game_id,
        steps=sub_steps,
        episodes=sub_eps,
        action_stats=_summarise_actions(sub_steps),
        goal_hints=goal_hints,
        failure_hints=failure_hints,
        hypothesis_priors=_derive_hypothesis_priors(sub_eps, sub_steps),
    )

logger = logging.getLogger(__name__)


# ==================================================================
# Prompt construction
# ==================================================================

_SYSTEM_PROMPT = """\
You are a strategy compiler for ARC-AGI-3 puzzle games.

Given evidence from human playthroughs — their stated objectives,
discovered mechanics, mistakes, sticky in-play hypotheses, action
statistics — produce a STRICT JSON object matching the TaskProgram
schema below.

Your output will be read by a reasoning agent that cannot see the
original traces. It can only act on structured, machine-readable
fields. So:

1. EXTRACT FROM EVIDENCE. Only include facts that appear in the
   evidence blocks. If a mechanic isn't stated, do NOT invent one.
   When ambiguous, lower `confidence` rather than guess.

2. KEY-FACT PARSING. Evidence strings shaped like `X_is_Y` or
   `X_does_Y` encode a role directly:
     "gray_shape_is_controlling_shapes"   -> Entity(name="gray_shape", role="controller")
     "black_doted_shape_is_player"        -> Entity(name="black_dotted_shape", role="player")
     "shapes_can_cross_doted_line"        -> Constraint(type="can_exit_grid", ...)
     "all_shapes_do_not_must_match"       -> Constraint(type="only_subset_matters", ...)
   Do NOT contradict these (e.g. don't add a constraint that shapes
   cannot cross if the evidence says they can).

3. CONCRETE SUBGOAL TESTS. Each SubgoalTest must either
     (a) set `prefer_actions` to 1-3 action names the human used
         heavily and effectively (see "High-signal actions" block), OR
     (b) set `click_target_color` or `click_target_xy` when the goal
         is to click a specific object, OR
     (c) pick a specific `expected_signal` from the schema enum
         (not "other" unless truly unclear).

3a. ACTION-ROLE SUBGOALS + STRUCTURED FACTS. For EVERY `ACTION<N>`
    that appears in the "Action-role hints" block:

    (i) Emit ONE entry in the top-level `action_roles` list:
        - `action`: e.g. "ACTION5"
        - `role`: pick from the ActionRoleKind enum (control_switch,
          movement, interact, rotate, push, click_select, toggle,
          undo, no_effect, unknown, other). Choose the BEST fit.
        - `evidence`: the verbatim trace string (e.g. "change_player_action5")
        - `confidence`: 0..1 (start at 0.7 unless evidence is overwhelming)
        - `changes_semantics_of`: list other ACTIONs whose meaning
          depends on this one (e.g. a control_switch action changes
          what the movement actions do).

    (ii) Emit ONE dedicated SubgoalTest:
      - `id`: e.g. "probe_action5"
      - `description`: paraphrase of the evidence string
      - `verification`: one sentence stating what observable change
        confirms this action's role
      - `expected_signal`: REQUIRED. Pick from the schema enum that
        best matches (e.g. `role_switch` for control-switch,
        `grid_change` for generic effect, `object_moved` for movement)
      - `max_actions`: small (4-8) since this is a probe
      - `prefer_actions`: exactly `[ACTION<N>]`
    Place these probes FIRST in `subgoal_tests`.

3b. ACHIEVE SUBGOAL (REQUIRED, exactly ONE). You MUST append a
    single "achieve_<goal>" entry as the LAST element of
    `subgoal_tests`. This subgoal is the EXPLOITATION phase where
    the agent combines the primitives discovered by sibling probes
    into a winning sequence; without it the agent stays locked in
    primitive-discovery mode forever and cannot progress past a
    level once the relevant primitives are known.
    Requirements for the achieve subgoal:
      - `id` MUST start with "achieve_" (e.g. "achieve_match_shapes",
        "achieve_progress").
      - `prefer_actions` MUST list AT LEAST 4 distinct actions,
        covering BOTH movement and interact/switch primitives, so
        the agent can chain them. Do NOT pin a single action.
      - `max_actions` MUST be between 30 and 60.
      - `expected_signal` SHOULD be a high-level progress indicator
        (`level_advance`, `grid_cell_change`, `unique_states_increasing`).
      - If the exact win condition is unclear, emit it anyway with
        a generic description like "combine discovered primitives
        until level advances" — an imperfect achieve subgoal is far
        better than none.

4. RANK: probes first (cheap), then any achieve goal (expensive).
   The agent executes tests in order and rotates on stall.

5. ANTI-PATTERNS FROM MISTAKES. Convert each discovered mistake to
   an imperative directive: "do not assume all shapes must match".

6. BELIEF REVISIONS. If the "Timeline of belief changes" shows the
   human's hypothesis changing, emit at least one HypothesisRevision
   whose `at_step`, `was`, `now`, `cause` match the timeline.

7. GOAL FAMILY: use `click_sequence` / `click_selection` for games
   where the human used ACTION6 heavily; `correspondence` /
   `alignment` / `symmetry` when the evidence talks about matching
   shapes; `navigation` for move-to-goal; etc.

Return ONLY valid JSON — no prose, no markdown fences.

=== STRUCTURAL EXAMPLE (use only for SHAPE — never copy its content) ===
The example below uses deliberately abstract placeholder names
(<ENTITY_A>, <COLOR_Z>, <ACTION_X>) so you understand the FIELD
SHAPES, not the domain. Your output's `goal_family`, `macro_goal`,
entity names, and subgoal descriptions MUST come from the evidence
blocks in the user message, NOT from this example.

Evidence would say something like:
  - type_guess: "<FAMILY_GUESS>"
  - mechanics: ["<ENTITY_A>_is_<ROLE_A>", "<ENTITY_B>_can_<ACTION_Z>"]
  - mistakes: ["assumed_<WRONG_ASSUMPTION>"]
  - High-signal actions: <ACTION_X> (high chg%), <ACTION_Y> (high chg%)

Compiled JSON (shape only):
{
  "goal_family": "<one of the enum values that matches type_guess and evidence>",
  "macro_goal": "<one sentence paraphrasing the human's objective_guess>",
  "entities": [
    {"name": "<ENTITY_A>", "role": "<ROLE_A>"},
    {"name": "<ENTITY_B>", "role": "<ROLE_B>"}
  ],
  "constraints": [
    {"type": "<schema_constraint_type>", "description": "<evidence_string_verbatim>"}
  ],
  "anti_patterns": ["do not <directive from mistakes>"],
  "action_roles": [
    {"action":"<ACTION_X>","role":"<ActionRoleKind>",
     "evidence":"<verbatim_trace_string_naming_action_x>",
     "confidence":0.7,"changes_semantics_of":[]}
  ],
  "subgoal_tests": [
    {"id":"probe_<actionx>","description":"confirm <ACTION_X> does <effect>",
     "verification":"<observable_change>","expected_signal":"object_moved",
     "max_actions":8,"prefer_actions":["<ACTION_X>"]},
    {"id":"achieve_<goal_family>","description":"<goal stated by human objective_guess>",
     "verification":"<concrete win-condition test>","expected_signal":"level_advance",
     "max_actions":40,"prefer_actions":["<ACTION_X>","<ACTION_Y>","<ACTION_Z>","<ACTION_W>"]}
  ],
  "confidence": 0.7
}

CRITICAL: Do NOT emit the literal strings "<ENTITY_A>", "<ACTION_X>",
"reach_right", "green_block", or any other name from the example
structure. Fill every <placeholder> and every field with values
derived from THIS GAME's evidence.
"""


def _action_stats_block(pack: HumanPriorPack) -> str:
    """Compact per-action summary the LLM can reason over."""
    lines = ["Action statistics (from human playthroughs):"]
    if not pack.action_stats:
        return "  (no action statistics available)"
    # Sort by tries desc.
    for name, s in sorted(
        pack.action_stats.items(), key=lambda kv: -kv[1].get("tries", 0)
    ):
        tries = int(s.get("tries", 0))
        chg = s.get("change_rate", 0.0)
        die = int(s.get("deaths", 0))
        win = int(s.get("wins", 0))
        lines.append(
            f"  {name}: tries={tries}, change_rate={chg:.0%}, "
            f"deaths={die}, wins={win}"
        )
    return "\n".join(lines)


def _episode_block(pack: HumanPriorPack) -> str:
    lines = ["Episode summaries (what the human reported):"]
    for i, ep in enumerate(pack.episodes):
        lines.append(
            f"- ep{i} ({ep.episode_id[:8]}): "
            f"{ep.n_steps} steps, final={ep.final_state}, "
            f"levels={ep.levels_completed}, "
            f"type_guess='{ep.game_type_guess or '?'}', "
            f"objective_guess='{ep.objective_guess or '?'}'"
        )
        if ep.discovered_mechanics:
            lines.append(
                "    mechanics: " + "; ".join(ep.discovered_mechanics[:8])
            )
        if ep.discovered_mistakes:
            lines.append(
                "    mistakes: " + "; ".join(ep.discovered_mistakes[:8])
            )
        if ep.notes:
            lines.append(f"    notes: {ep.notes[:200]}")
    return "\n".join(lines)


def _sticky_hypotheses_block(pack: HumanPriorPack) -> str:
    """Surface the most-frequent and most-recent in-play hypotheses.

    Also flag CHANGES across consecutive steps as candidate
    belief revisions — the compiler can use these to populate
    TaskProgram.belief_revisions.
    """
    hyp_counter: Counter = Counter()
    sequence: List[tuple[str, int]] = []   # (hypothesis, step_idx)
    for s in pack.steps:
        h = (s.hypothesis or "").strip()
        if not h:
            continue
        hyp_counter[h] += 1
        if not sequence or sequence[-1][0] != h:
            sequence.append((h, s.step))

    lines = ["In-play sticky hypotheses (human's evolving beliefs):"]
    if hyp_counter:
        lines.append("  Most frequent:")
        for h, n in hyp_counter.most_common(6):
            lines.append(f"    - \"{h}\" ({n} steps)")
    if len(sequence) >= 2:
        lines.append("  Timeline of belief changes:")
        for h, step in sequence[:10]:
            lines.append(f"    step {step}: \"{h}\"")
    if len(lines) == 1:
        lines.append("  (none recorded)")
    return "\n".join(lines)


_KEY_FACT_RE = re.compile(
    r"^([a-z0-9_]+?)_(is|are|does|do|can|cannot|must|must_not|should|should_not)_([a-z0-9_]+)$",
    flags=re.IGNORECASE,
)


def _key_facts_block(pack: HumanPriorPack) -> str:
    """Extract `X_<predicate>_Y`-shaped evidence strings verbatim.

    These are the highest-signal inputs we have: short sentences the
    human wrote down during play that encode a direct entity/role/
    constraint mapping. Small LLMs benefit a lot from seeing them
    enumerated separately from the per-episode prose.
    """
    facts: set[tuple[str, str]] = set()  # (raw_string, source_tag)

    # From discovered_mechanics / discovered_mistakes per episode.
    for ep in pack.episodes:
        for m in (ep.discovered_mechanics or []):
            s = m.strip()
            if _KEY_FACT_RE.match(s):
                facts.add((s, "mechanic"))
        for m in (ep.discovered_mistakes or []):
            s = m.strip()
            if _KEY_FACT_RE.match(s):
                facts.add((s, "mistake"))

    # From sticky step-level hypotheses.
    for st in pack.steps:
        h = (st.hypothesis or "").strip()
        if h and _KEY_FACT_RE.match(h):
            facts.add((h, "belief"))

    lines = ["Key evidence strings (X_is_Y form — highest signal):"]
    if not facts:
        lines.append("  (none extracted)")
        return "\n".join(lines)
    # Group by source tag for readability.
    for tag in ("mechanic", "mistake", "belief"):
        group = sorted(s for s, t in facts if t == tag)
        if group:
            lines.append(f"  [{tag}]")
            for s in group[:12]:
                lines.append(f"    - {s}")
    return "\n".join(lines)


_ACTION_MENTION_RE = re.compile(r"action\s*_?\s*(\d)", flags=re.IGNORECASE)


def _extract_explicit_action_mentions(
    pack: HumanPriorPack,
) -> Dict[str, List[str]]:
    """Deterministically collect every evidence string that names an
    ``ACTION<N>`` from the human trace.

    This is the single source of truth for the compiler invariant:
    *every* ACTION whose role is explicitly stated by the human must
    survive into the final TaskProgram as a dedicated probe. Both the
    prompt hint block and the post-compile auto-repair step consume
    this mapping.

    Returns
    -------
    dict
        ``{"ACTION5": ["change_player_action5", ...], ...}`` — keys are
        canonical ACTION names, values are deduplicated evidence
        strings in deterministic order.
    """
    hits: Dict[str, set[str]] = {}

    def _add(s: str) -> None:
        s = (s or "").strip()
        if not s:
            return
        for m in _ACTION_MENTION_RE.finditer(s):
            key = f"ACTION{m.group(1)}"
            hits.setdefault(key, set()).add(s)

    for ep in pack.episodes:
        for m in (ep.discovered_mechanics or []):
            _add(m)
        for m in (ep.discovered_mistakes or []):
            _add(m)
        _add(ep.objective_guess or "")
        _add(ep.notes or "")
    for st in pack.steps:
        _add(getattr(st, "hypothesis", "") or "")

    return {act: sorted(strs) for act, strs in hits.items()}


def _action_role_hints_block(pack: HumanPriorPack) -> str:
    """Render the ACTION<N>-mention evidence as a prompt block.

    Derived from :func:`_extract_explicit_action_mentions`.
    """
    hits = _extract_explicit_action_mentions(pack)
    lines = [
        "Action-role hints (evidence strings naming a specific ACTION — "
        "MUST become dedicated SubgoalTests):"
    ]
    if not hits:
        lines.append("  (no action-specific mechanic strings found)")
        return "\n".join(lines)
    for act in sorted(hits.keys()):
        for s in hits[act][:4]:
            lines.append(f"  {act}: \"{s}\"")
    lines.append(
        "  → For each ACTION above, emit a SubgoalTest with "
        "`prefer_actions=[<that action>]` and a description that "
        "paraphrases the evidence string (e.g. \"change_player_action5\" "
        "→ description=\"use ACTION5 to switch controlled entity\")."
    )
    return "\n".join(lines)


# ------------------------------------------------------------------
# Post-compile invariants: deterministic auto-repair of the LLM output.
# ------------------------------------------------------------------
# The compiler follows the user's "LLM normalizes semantics, code builds
# structure" principle: we let the LLM name things and pick a goal_family,
# but we own the skeleton. Two invariants are enforced post-compile:
#
#   1. Action-role preservation. Every ACTION<N> the human explicitly
#      named must survive as a dedicated probe (prefer_actions == [that
#      action]) placed BEFORE any achieve_* subgoal.
#   2. Achieve presence. At least one achieve_*/solve_*/win_* subgoal
#      with >=4 distinct prefer_actions and max_actions in [30, 60].
#
# Invariant (1) auto-repairs by INJECTING synthetic probes when missing,
# because LLM retries cost ~240s each on a 3B model and the skeleton is
# well-defined enough to synthesize deterministically. Invariant (2) is
# enforced by the retry/reject loop since the achieve subgoal's
# description/prefer_actions depend on LLM semantic understanding.

# Heuristic: classify an evidence string's likely expected_signal so the
# synthesized probe fires the right actioner path. We keep the list tiny
# and conservative — the fallback is "object_moved" which the actioner
# handles via the standard movement-alignment path.
# Each tuple: (regex over evidence string, ActionRoleKind, ExpectedSignal,
# verification_verb). Order matters: more specific matches come first so a
# string like "change_player_action5" classifies as control_switch + role_switch
# rather than the more generic "click" or "move" patterns.
_ROLE_SIGNAL_HEURISTICS: List[Tuple[re.Pattern, str, str, str]] = [
    (re.compile(r"change[_\s]*player|switch[_\s]*control|role[_\s]*switch",
                re.IGNORECASE),
     "control_switch", "role_switch",
     "switch the controlled entity"),
    (re.compile(r"\brotate|spin|turn\b", re.IGNORECASE),
     "rotate", "object_moved",
     "rotate a shape or piece"),
    (re.compile(r"\bpush|shove\b", re.IGNORECASE),
     "push", "object_moved",
     "push another object"),
    (re.compile(r"\bundo|revert\b", re.IGNORECASE),
     "undo", "grid_change",
     "reverse the previous step"),
    (re.compile(r"\btoggle|flip\b", re.IGNORECASE),
     "toggle", "grid_change",
     "flip a boolean cell state"),
    (re.compile(r"click|select|activate|open|tap", re.IGNORECASE),
     "click_select", "click_triggered_change",
     "trigger a change by interaction"),
    (re.compile(r"move|shift|walk|step", re.IGNORECASE),
     "movement", "object_moved",
     "cause an observable object motion"),
]


def _classify_evidence(evidence_strs: List[str]) -> Tuple[str, str, str]:
    """Pick ``(role_kind, expected_signal, verification_verb)`` for a
    synthesized probe given the evidence strings that mention the action.
    Falls back to ``("unknown", "object_moved", ...)``.
    """
    for pat, kind, sig, verb in _ROLE_SIGNAL_HEURISTICS:
        for s in evidence_strs:
            if pat.search(s):
                return kind, sig, verb
    return "unknown", "object_moved", "produce an observable change"


def _subgoal_pins_single_action(sg: Any, action: str) -> bool:
    """True iff ``sg`` is a DEDICATED probe for ``action``.

    Appearance of ``action`` in a multi-action prefer_actions list (e.g.
    the achieve subgoal) does NOT count: the actioner's hard-filter is
    per-subgoal, so only a single-action subgoal concentrates discovery
    on that primitive.
    """
    pref = list(getattr(sg, "prefer_actions", []) or [])
    return len(pref) == 1 and pref[0] == action


def _enforce_action_role_preservation(
    program: "TaskProgram",
    pack: HumanPriorPack,
) -> Tuple["TaskProgram", List[str]]:
    """Auto-fill ``action_roles`` and inject matching probes.

    Two coupled repairs:
      1. **action_roles** (structured schema field): for every ACTION<N>
         the human named, ensure ``program.action_roles`` contains an
         entry. If the LLM emitted one already, keep it; otherwise
         synthesize from the evidence strings + role-classifier heuristic.
      2. **subgoal_tests** (executable plan): for every ``ActionRole``
         in the (now-complete) ``program.action_roles``, ensure there is
         at least one dedicated probe (``prefer_actions == [action]``)
         placed BEFORE any ``achieve_*``/``solve_*``/``win_*`` subgoal.

    This is the "LLM normalises semantics, code builds structure"
    contract: the LLM may forget to emit either surface, but the
    deterministic post-processor always reconciles the two from the
    explicit human evidence.

    Returns ``(maybe_modified_program, repair_log)``.
    """
    mentions = _extract_explicit_action_mentions(pack)
    if not mentions:
        return program, []

    from human_trace.task_program import ActionRole, SubgoalTest

    repair_log: List[str] = []

    # ── Step 1: reconcile action_roles ─────────────────────────────
    existing_roles: Dict[str, ActionRole] = {
        r.action: r for r in (program.action_roles or [])
    }
    new_roles_needed: List[ActionRole] = []
    for act in sorted(mentions.keys()):
        if act in existing_roles:
            continue
        ev = mentions[act]
        kind, _signal, _verb = _classify_evidence(ev)
        ev_preview = ev[0] if ev else act.lower()
        new_roles_needed.append(ActionRole(
            action=act,
            role=kind,  # type: ignore[arg-type]
            evidence=ev_preview[:200],
            confidence=0.7,
            changes_semantics_of=[],
        ))
        repair_log.append(
            f"injected ActionRole for {act} (role={kind}, "
            f"evidence={ev_preview!r})"
        )
    if new_roles_needed:
        # Respect the schema cap (≤12). Newly-injected entries are kept
        # verbatim because they came from explicit evidence; LLM-emitted
        # entries are kept first.
        merged = list(program.action_roles or []) + new_roles_needed
        program.action_roles = merged[:12]

    # ── Step 2: reconcile subgoal_tests ────────────────────────────
    # Coverage is decided over the FULL action_roles list (LLM-emitted +
    # auto-injected) so the LLM gets credit for any role it described
    # itself, even if the evidence string was paraphrased.
    all_actions = sorted({r.action for r in (program.action_roles or [])})
    covered: set = set()
    for sg in program.subgoal_tests:
        for act in all_actions:
            if _subgoal_pins_single_action(sg, act):
                covered.add(act)

    missing = [a for a in all_actions if a not in covered]
    if not missing:
        return program, repair_log

    # Find the first achieve_* / solve_* / win_* so we insert probes
    # BEFORE it (probes-before-achieve invariant).
    insert_at = len(program.subgoal_tests)
    for i, sg in enumerate(program.subgoal_tests):
        sid = str(getattr(sg, "id", "")).lower()
        if sid.startswith(("achieve", "solve", "win")):
            insert_at = i
            break

    # Index roles for quick lookup so we copy through the LLM's role
    # description when one was supplied.
    role_by_act: Dict[str, ActionRole] = {
        r.action: r for r in (program.action_roles or [])
    }

    synth: List[SubgoalTest] = []
    for act in missing:
        role = role_by_act.get(act)
        ev_preview = role.evidence if role else (mentions.get(act, [act])[0])
        # Re-derive signal/verb from the evidence so the synthesized
        # probe is signal-correct even when the LLM-emitted ActionRole
        # used a different role label than our heuristic chose.
        ev_strs = mentions.get(act, [ev_preview])
        _kind, signal, verb = _classify_evidence(ev_strs)
        sg_id = f"probe_{act.lower()}"[:64]
        desc = f"{act}: {verb} (evidence: {ev_preview})"[:200]
        verif = f"observable {signal.replace('_', ' ')} after {act}"[:200]
        synth.append(SubgoalTest(
            id=sg_id,
            description=desc,
            verification=verif,
            max_actions=6,
            expected_signal=signal,  # type: ignore[arg-type]
            prefer_actions=[act],
        ))
        repair_log.append(
            f"injected probe for {act} (signal={signal}, "
            f"evidence={ev_preview!r})"
        )

    # Respect the schema's max_length=16 on subgoal_tests.
    new_list = list(program.subgoal_tests)
    new_list[insert_at:insert_at] = synth
    if len(new_list) > 16:
        # Trim the TAIL (lowest-priority extras) rather than dropping
        # achieve_* or the fresh probes.
        new_list = new_list[:16]
    program.subgoal_tests = new_list
    return program, repair_log


# ------------------------------------------------------------------
# Invariant 3: goal-object preservation
# ------------------------------------------------------------------
# If the macro_goal text references concrete objects (typically by
# colour: "match yellow and purple shapes"), those objects MUST appear
# as entities. Otherwise downstream consumers that try to follow the
# goal-family-specific subgoal logic (e.g. "go to the purple_shape")
# silently fall back to generic exploration.

# Closed list of colour words. Conservative on purpose: extending it
# is cheap, and we don't want to mistake e.g. "block" for an object
# without the LLM's say-so. Order doesn't matter.
_GOAL_OBJECT_COLOR_WORDS: Tuple[str, ...] = (
    "red", "blue", "green", "yellow", "purple", "pink", "gray", "grey",
    "orange", "cyan", "magenta", "brown", "white", "black",
)
# Object/role nouns that we'll attach to a colour to form an entity name.
_GOAL_OBJECT_NOUNS: Tuple[str, ...] = (
    "shape", "shapes", "block", "blocks", "tile", "tiles", "cell", "cells",
    "dot", "dots", "line", "lines", "arrow", "arrows", "square", "squares",
    "circle", "circles", "triangle", "triangles", "cursor", "cursors",
    "marker", "markers",
)


def _extract_macro_goal_objects(macro_goal: str) -> List[Tuple[str, Optional[str]]]:
    """Find ``(colour, noun_or_None)`` pairs referenced in the macro_goal.

    Examples
    --------
    >>> _extract_macro_goal_objects("Match yellow and purple shapes")
    [('yellow', 'shapes'), ('purple', 'shapes')]
    >>> _extract_macro_goal_objects("Click the red dot")
    [('red', 'dot')]
    >>> _extract_macro_goal_objects("Just go to the goal")
    []
    """
    text = (macro_goal or "").lower()
    if not text:
        return []
    # Tokenize on whitespace and punctuation.
    tokens = re.findall(r"[a-z]+", text)
    if not tokens:
        return []

    pairs: List[Tuple[str, Optional[str]]] = []
    seen: set = set()
    for i, tok in enumerate(tokens):
        if tok not in _GOAL_OBJECT_COLOR_WORDS:
            continue
        # Find the next noun within the next 3 tokens (skipping
        # conjunctions like "and" / "or" / "the" / colour words).
        noun: Optional[str] = None
        for nxt in tokens[i + 1: i + 5]:
            if nxt in _GOAL_OBJECT_NOUNS:
                noun = nxt.rstrip("s") + "s"  # canonical plural form
                break
            if nxt in _GOAL_OBJECT_COLOR_WORDS:
                # "yellow and purple shapes" — keep scanning past the
                # second colour to find the shared noun.
                continue
            if nxt in {"and", "or", "the", "a", "an"}:
                continue
            # Hit a non-skip word that isn't a noun → stop scanning so
            # we don't pull in unrelated text.
            break
        key = (tok, noun)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


def _enforce_entity_preservation(
    program: "TaskProgram",
) -> Tuple["TaskProgram", List[str]]:
    """Auto-inject placeholder entities for concrete objects named in
    ``program.macro_goal`` but missing from ``program.entities``.

    Why: if the human writes "match yellow and purple shapes" but the
    LLM produces only ``[{name: "player"}, {name: "controller"}]``, the
    downstream goal-pursuit layer never grounds the actual objects of
    the puzzle. We inject minimal stubs (role=unknown) so the
    information at least round-trips and is available to later layers.
    """
    pairs = _extract_macro_goal_objects(program.macro_goal)
    if not pairs:
        return program, []

    from human_trace.task_program import Entity

    existing_names: set = {
        (e.name or "").lower() for e in (program.entities or [])
    }
    repair_log: List[str] = []
    new_entities: List[Entity] = []
    for color, noun in pairs:
        # Canonical name: "<color>_<noun>" or just "<color>" if no noun.
        if noun:
            ent_name = f"{color}_{noun.rstrip('s')}"  # singular form
        else:
            ent_name = color
        # Match liberally: an existing entity called "yellow" or
        # "yellow_shapes" or "yellow_shape" all satisfy a "yellow_shape"
        # requirement.
        if any(ent_name in n or n in ent_name for n in existing_names if n):
            continue
        new_entities.append(Entity(
            name=ent_name[:64],
            role="unknown",  # type: ignore[arg-type]
            notes=f"auto-injected from macro_goal phrase '{color} {noun or ''}'".strip(),
        ))
        existing_names.add(ent_name)
        repair_log.append(
            f"injected entity {ent_name!r} (from macro_goal mention of "
            f"{color!r}{' ' + noun if noun else ''})"
        )
    if not new_entities:
        return program, []
    merged = list(program.entities or []) + new_entities
    program.entities = merged[:16]  # respect schema cap
    return program, repair_log


# ------------------------------------------------------------------
# Invariant 4: specific achieve subgoal (no generic id when goal_family is known)
# ------------------------------------------------------------------
# `achieve_progress`, `achieve_goal`, `achieve_level` — these convey no
# semantic information and confuse the downstream planner that keys on
# the id prefix to pick exploitation strategies. If the goal_family is
# concrete (e.g. "correspondence"), force the achieve id to carry it.

# Generic achieve names that should be specialised.
_GENERIC_ACHIEVE_IDS: Tuple[str, ...] = (
    "achieve",
    "achieve_progress",
    "achieve_goal",
    "achieve_level",
    "achieve_win",
    "achieve_unknown",
    "achieve_other",
)


def _enforce_specific_achieve_id(
    program: "TaskProgram",
) -> Tuple["TaskProgram", List[str]]:
    """Rename a generic ``achieve_*`` id to ``achieve_<goal_family>``
    when ``goal_family`` is a non-trivial enum value.

    No-op when:
      - there is no achieve_* subgoal (handled by the achieve validator),
      - ``goal_family`` is "unknown" / "other",
      - the achieve id is already specific (e.g. ``achieve_match_shapes``).
    """
    if not program.subgoal_tests:
        return program, []
    family = (program.goal_family or "").lower()
    if family in ("", "unknown", "other"):
        return program, []

    repair_log: List[str] = []
    for sg in program.subgoal_tests:
        sid = (sg.id or "").lower()
        if not sid.startswith(("achieve", "solve", "win")):
            continue
        if sid not in _GENERIC_ACHIEVE_IDS:
            continue
        new_id = f"achieve_{family}"[:64]
        if new_id == sid:
            continue
        repair_log.append(
            f"renamed generic achieve subgoal {sg.id!r} → {new_id!r} "
            f"(goal_family={family})"
        )
        sg.id = new_id
    return program, repair_log


def _high_signal_actions_block(pack: HumanPriorPack) -> str:
    """Which actions should appear in SubgoalTest.prefer_actions.

    Definition: high `tries` + high `change_rate` + low death rate.
    Compact list, max 4 entries. Referenced by name in the system
    prompt example so the model knows to cite these exactly.
    """
    if not pack.action_stats:
        return "High-signal actions: (no data)"
    scored: list[tuple[str, float, dict]] = []
    for name, s in pack.action_stats.items():
        tries = int(s.get("tries", 0))
        chg = float(s.get("change_rate", 0.0))
        die = int(s.get("deaths", 0))
        if tries < 5:
            continue
        # Simple score: change rate weighted by log(tries), penalised
        # by death rate.
        import math
        score = chg * math.log(1 + tries) - 0.3 * (die / max(tries, 1))
        scored.append((name, score, s))
    scored.sort(key=lambda kv: -kv[1])
    lines = ["High-signal actions (candidates for SubgoalTest.prefer_actions):"]
    for name, sc, s in scored[:4]:
        lines.append(
            f"  {name} (tries={int(s['tries'])}, "
            f"change_rate={float(s['change_rate']):.0%}, "
            f"deaths={int(s.get('deaths', 0))}) — score={sc:.2f}"
        )
    if len(lines) == 1:
        lines.append("  (none qualified: all actions have <5 tries)")
    return "\n".join(lines)


def _goal_family_hint(pack: HumanPriorPack) -> str:
    """Expose both the human's mechanism guess AND objective guess.

    CRITICAL: `type_guess` describes the *mechanism* ("move", "click",
    "drag") while `objective_guess` describes the *goal* ("collect
    all coins", "correspond_shapes"). `goal_family` must match the
    objective, not the mechanism. E.g. a game where you move shapes
    to match colors has type_guess="move" but goal_family="correspondence".
    """
    type_guesses: Counter = Counter()
    obj_guesses: Counter = Counter()
    for ep in pack.episodes:
        g = (ep.game_type_guess or "").lower().strip()
        if g:
            type_guesses[g] += 1
        o = (getattr(ep, "objective_guess", "") or "").lower().strip()
        if o:
            obj_guesses[o] += 1

    lines: List[str] = []
    if type_guesses:
        top = ", ".join(f"{g} ({n}x)" for g, n in type_guesses.most_common(3))
        lines.append(f"Human type_guess (mechanism — NOT goal_family): {top}")
    if obj_guesses:
        top = ", ".join(f'"{g}" ({n}x)' for g, n in obj_guesses.most_common(3))
        lines.append(
            f"Human objective_guess (the GOAL — drives goal_family + macro_goal): {top}"
        )
        lines.append(
            "  → map this phrase to a goal_family enum value. "
            "If it mentions matching/pairing → 'correspondence' or 'alignment'. "
            "If it mentions collecting → 'collection'. "
            "If it mentions clicking in order → 'click_sequence'. "
            "If it mentions reaching/exiting a location → 'navigation'."
        )
    return "\n".join(lines)


def build_user_prompt(
    pack: HumanPriorPack,
    retry_error: Optional[str] = None,
    agent_experience: Optional[str] = None,
    revision_context: Optional[str] = None,
) -> str:
    """Assemble the user prompt with all evidence blocks.

    Optional arguments:
      - ``agent_experience``: a free-form block of runtime observations
        produced by the live agent (action stats on the current level,
        discovered mechanics, etc.). Injected as an additional evidence
        block so the LLM can incorporate agent-specific learning when
        revising a TaskProgram mid-game.
      - ``revision_context``: a short directive (e.g. "This is REVISION 1.
        Produce a TaskProgram for LEVEL 2 only; probe new mechanics.")
        placed at the top of the prompt so the model tailors its output
        to the ongoing run.
    """
    parts = [
        f"Game ID: {pack.game_id}",
        "",
    ]
    if revision_context:
        parts.extend([
            "=== REVISION CONTEXT ===",
            revision_context.strip(),
            "",
        ])
    parts.extend([
        _goal_family_hint(pack),
        "",
        _key_facts_block(pack),
        "",
        _action_role_hints_block(pack),
        "",
        _high_signal_actions_block(pack),
        "",
        _episode_block(pack),
        "",
        _action_stats_block(pack),
        "",
        _sticky_hypotheses_block(pack),
        "",
    ])
    if agent_experience:
        parts.extend([
            "=== AGENT RUNTIME OBSERVATIONS ===",
            "(Collected by the live agent on the level you are compiling for.",
            " These supersede or refine the human-trace evidence when they",
            " disagree, because they reflect the agent's own interactions.)",
            agent_experience.strip(),
            "",
        ])
    parts.extend([
        "=== SCHEMA ===",
        json_schema_for_prompt(),
        "",
        "=== TASK ===",
        "Produce the TaskProgram JSON for this game. Return JSON only.",
    ])
    if retry_error:
        parts.extend([
            "",
            "=== PREVIOUS ATTEMPT REJECTED ===",
            f"Reason: {retry_error[:800]}",
            "Fix the issues and return valid JSON matching the schema.",
        ])
    return "\n".join(p for p in parts if p is not None)


# ==================================================================
# LLM invocation
# ==================================================================

def _extract_json_object(raw: str) -> Optional[str]:
    """Pull the first balanced top-level JSON object out of the response.

    Handles small-model quirks like markdown fences or leading prose.
    """
    # Strip common fences.
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)

    start = raw.find("{")
    if start < 0:
        return None
    # Walk to matching close brace.
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(raw[start:], start=start):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]
    return None


def _generate_once(
    model, tokenizer, system_prompt: str, user_prompt: str, device: str,
    max_new_tokens: int = 1200, temperature: float = 0.3,
) -> str:
    """Generate a single completion, stopping early when a balanced JSON
    object has been produced. The model often rambles past the closing
    brace into trailing prose — on CPU that's pure wasted decoding time
    (at ~10-15 tok/s a 900-token preamble adds ~1-2 minutes per attempt).
    A lightweight brace-balance StoppingCriteria halts generation as
    soon as we've emitted a complete top-level JSON object.
    """
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=8192
    ).to(device)
    prompt_len = inputs["input_ids"].shape[1]

    class _BalancedJSONStop(StoppingCriteria):
        """Stop once the generated tokens contain a complete
        balanced {...} JSON object. Checks every 16 new tokens.
        """
        def __init__(self, tokenizer, prompt_len: int):
            self._tok = tokenizer
            self._prompt_len = prompt_len
            self._check_every = 16
            self._last_check = 0

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            gen_len = input_ids.shape[1] - self._prompt_len
            if gen_len - self._last_check < self._check_every:
                return False
            self._last_check = gen_len
            text = self._tok.decode(
                input_ids[0][self._prompt_len:], skip_special_tokens=True
            )
            # Find first '{' and walk balance.
            start = text.find("{")
            if start < 0:
                return False
            depth = 0
            in_str = False
            escape = False
            for ch in text[start:]:
                if escape:
                    escape = False; continue
                if ch == "\\":
                    escape = True; continue
                if ch == '"':
                    in_str = not in_str; continue
                if in_str:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return True  # complete JSON object
            return False

    stopping = StoppingCriteriaList([_BalancedJSONStop(tokenizer, prompt_len)])
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=max(temperature, 0.01),
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping,
        )
    gen = out[0][prompt_len:]
    return tokenizer.decode(gen, skip_special_tokens=True)


def compile_with_llm(
    pack: HumanPriorPack,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    device: str = "cpu",
    max_retries: int = 3,
    agent_experience: Optional[str] = None,
    revision_context: Optional[str] = None,
) -> TaskProgram:
    """Compile a HumanPriorPack into a validated TaskProgram.

    Optional ``agent_experience`` and ``revision_context`` are forwarded
    verbatim to :func:`build_user_prompt`. They allow the live agent to
    request a *revised* TaskProgram mid-run by augmenting the human-prior
    evidence with its own runtime observations.
    """
    from v4_1_reasoning_system.arc_agi.llm_cache import get_shared_llm

    logger.info("Loading compiler LLM: %s (device=%s)", model_name, device)
    t0 = time.time()
    model, tokenizer = get_shared_llm(model_name, device)
    if model is None or tokenizer is None:
        raise RuntimeError(
            f"Could not load LLM {model_name!r}. "
            "Install transformers / check HF cache."
        )
    logger.info("LLM loaded in %.1fs", time.time() - t0)

    last_error: Optional[str] = None
    for attempt in range(1, max_retries + 1):
        user_prompt = build_user_prompt(
            pack,
            retry_error=last_error,
            agent_experience=agent_experience,
            revision_context=revision_context,
        )
        logger.info("Compile attempt %d/%d ...", attempt, max_retries)
        t0 = time.time()
        raw = _generate_once(
            model, tokenizer, _SYSTEM_PROMPT, user_prompt, device=device,
            temperature=0.2 if attempt == 1 else 0.5,  # diversify on retry
        )
        logger.info("  generation took %.1fs", time.time() - t0)
        logger.debug("Raw response:\n%s", raw[:2000])

        json_str = _extract_json_object(raw)
        if json_str is None:
            last_error = "No JSON object found in response."
            continue

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e}"
            continue

        # Force compiler metadata regardless of what the model emitted.
        data["game_id"] = pack.game_id
        data["compiler_model"] = model_name
        data["compiled_at"] = datetime.now(timezone.utc).isoformat()
        data["source_episodes"] = [ep.episode_id for ep in pack.episodes][:16]

        try:
            program = TaskProgram.model_validate(data)
        except Exception as e:   # ValidationError or custom
            last_error = f"Schema validation failed: {e}"
            logger.info("  attempt %d rejected: %s", attempt, last_error[:200])
            continue

        # ── Post-compile invariants (deterministic auto-repair) ───
        # The "LLM normalises semantics, code builds structure" pipeline:
        # rather than burn a ~240s LLM retry on a structural omission,
        # we patch the program from the explicit human evidence.
        #
        # Order matters:
        #   (1) action_roles + probes — depends only on `pack`.
        #   (2) entities — depends on `macro_goal` (LLM-generated).
        #   (3) achieve id specialisation — depends on `goal_family`.
        #
        # Each repair returns its own log so the diagnostic trail is
        # explicit and grep-friendly.
        repair_log_total: List[str] = []
        program, log1 = _enforce_action_role_preservation(program, pack)
        program, log2 = _enforce_entity_preservation(program)
        program, log3 = _enforce_specific_achieve_id(program)
        repair_log_total.extend(log1)
        repair_log_total.extend(log2)
        repair_log_total.extend(log3)
        for msg in repair_log_total:
            logger.info("  post-compile repair: %s", msg)

        # Enforce rule 3b: at least one achieve_* subgoal with broad
        # prefer_actions. Without it, the agent's actioner hard-filters
        # every subgoal to a single action and can never combine
        # primitives — this is the documented failure mode on level-N
        # plateaus (see ar25 lvl2.live1/live2 history).
        achieve_sgs = [
            s for s in program.subgoal_tests
            if str(getattr(s, "id", "")).lower().startswith("achieve")
        ]
        if not achieve_sgs:
            last_error = (
                "missing required achieve_* subgoal: rule 3b mandates "
                "exactly ONE subgoal whose id starts with 'achieve_' as "
                "the LAST element of subgoal_tests, with prefer_actions "
                "listing >=4 distinct actions and max_actions in [30, 60]"
            )
            logger.info("  attempt %d rejected: %s", attempt, last_error)
            continue
        # Validate the achieve subgoal's shape too — a degenerate
        # achieve (one action, max=8) defeats the exploitation goal.
        achv = achieve_sgs[-1]
        achv_pref = list(getattr(achv, "prefer_actions", []) or [])
        achv_max = int(getattr(achv, "max_actions", 0) or 0)
        if len(set(achv_pref)) < 4 or achv_max < 30:
            last_error = (
                f"achieve_* subgoal {achv.id!r} is degenerate: "
                f"prefer_actions={achv_pref} (need >=4 distinct), "
                f"max_actions={achv_max} (need 30-60). Rule 3b requires "
                f"the achieve subgoal to chain multiple primitives."
            )
            logger.info("  attempt %d rejected: %s", attempt, last_error[:200])
            continue

        logger.info(
            "Compile succeeded on attempt %d: goal_family=%s, "
            "%d entities, %d subgoals, %d constraints",
            attempt, program.goal_family, len(program.entities),
            len(program.subgoal_tests), len(program.constraints),
        )
        return program

    raise RuntimeError(
        f"Compiler failed after {max_retries} attempts. "
        f"Last error: {last_error}"
    )


# ==================================================================
# CLI
# ==================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True,
                        help="Directory with *.episodes.jsonl and *.steps.jsonl")
    parser.add_argument("--game", required=True,
                        help="Game id prefix (e.g. 'ar25' or 'ar25-e3c63847')")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output path for the TaskProgram JSON")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct",
                        help="HF model id for the compiler LLM")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device for the compiler LLM. 'auto' picks "
                             "cuda when torch.cuda.is_available(), else cpu.")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--from-level",
        type=int,
        default=0,
        help=(
            "If > 0, only compile from human trace data where the human had "
            "already completed at least N levels. Use to produce per-level "
            "TaskPrograms (e.g. --from-level 1 -> level-2-onwards program "
            "saved to task_programs/<game>.lvl2.json)."
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )

    corpus = load_traces(args.traces)
    # Resolve short game id to full id by prefix.
    full_ids = sorted(corpus.by_game.keys())
    match = None
    for gid in full_ids:
        if gid == args.game or gid.startswith(f"{args.game}-") or gid.startswith(f"{args.game}."):
            match = gid
            break
    if match is None:
        print(f"ERROR: no traces found matching game={args.game!r}. "
              f"Available: {full_ids}", file=sys.stderr)
        return 2

    pack = build_prior_pack(corpus, match)
    if not pack.episodes:
        print(f"ERROR: no episodes for {match}", file=sys.stderr)
        return 2

    if args.from_level > 0:
        pre = (len(pack.episodes), len(pack.steps))
        pack = filter_pack_by_min_level(pack, args.from_level)
        post = (len(pack.episodes), len(pack.steps))
        print(
            f"Filtering by from_level={args.from_level} "
            f"(human must have completed >= {args.from_level} levels): "
            f"{pre[0]}->{post[0]} episodes, {pre[1]}->{post[1]} steps."
        )
        if not pack.episodes and not pack.steps:
            print(
                f"ERROR: no traces survive --from-level {args.from_level}. "
                f"Human did not progress past level {args.from_level} in this game.",
                file=sys.stderr,
            )
            return 2

    print(f"Compiling {match}: {len(pack.episodes)} episodes, "
          f"{len(pack.steps)} steps, {len(pack.action_stats)} actions")

    # Resolve "auto" -> cuda if available, else cpu.
    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
        print(f"  [device] resolved 'auto' -> {device}")

    try:
        program = compile_with_llm(
            pack, model_name=args.model, device=device,
            max_retries=args.max_retries,
        )
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    program.save(args.out)
    print(f"Wrote {args.out}")
    print(f"  goal_family={program.goal_family} "
          f"confidence={program.confidence:.2f}")
    print(f"  macro_goal: {program.macro_goal}")
    print(f"  {len(program.entities)} entities, "
          f"{len(program.constraints)} constraints, "
          f"{len(program.subgoal_tests)} subgoal tests, "
          f"{len(program.anti_patterns)} anti-patterns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
