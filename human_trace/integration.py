"""Turn human traces into v4_1 priors.

Two integration levels (corresponding to the user's chosen
"Priors + goal hypotheses" option):

1. **Per-game priors (GameMemory)** — replay every StepRecord through
   `GameMemory.record_action(...)`. This populates action profiles,
   click effectiveness, and player hypotheses exactly as if the agent
   itself had played the human's trajectory. Also attaches the
   human's sticky hypotheses as `GameMemory.hypotheses` entries and
   their game-type / objective guesses as high-confidence hypothesis
   keys the GoalDecomposer template fallback can inspect.

2. **Cross-game priors (CrossGameMemory)** — summarises each episode
   into a lightweight dict stored under
   `cross_game.goal_strategy_hints[goal_id]`, following the same
   schema v4_1 already uses for self-generated hints. Losing
   episodes contribute to `cross_game.failure_patterns`. Trust is
   preserved: these stay priors, not authoritative.

The loader leaves v4_1 source untouched — all we do here is call its
public API.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .loader import TraceCorpus, load_traces
from .schema import EpisodeRecord, IntentTag, StepRecord

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Canonical goal-id mapping (matches CrossGameMemory conventions)
# ------------------------------------------------------------------
# v4_1's CrossGameMemory normalises to ~7 canonical goal types (see the
# related memory about memory-leak fix). Keep this tight so we don't
# explode the hint dict with bespoke human strings.

_CANONICAL_GOALS: List[str] = [
    "click_puzzle",
    "sequence_puzzle",
    "navigate_exit",
    "collection",
    "push_puzzle",
    "transform_puzzle",
    "navigate_puzzle",
    "unknown",
]

# Keyword → canonical mapping for best-effort parsing of human guesses.
_GOAL_KEYWORDS: Dict[str, str] = {
    "click": "click_puzzle",
    "toggle": "click_puzzle",
    "activate": "click_puzzle",
    "sequence": "sequence_puzzle",
    "order": "sequence_puzzle",
    "navigate": "navigate_exit",
    "exit": "navigate_exit",
    "reach": "navigate_exit",
    "collect": "collection",
    "gather": "collection",
    "push": "push_puzzle",
    "slot": "push_puzzle",
    "transform": "transform_puzzle",
    "apply": "transform_puzzle",
    "move": "navigate_puzzle"
}


def canonicalise_goal(human_guess: str) -> str:
    """Map a free-form human game_type_guess to a canonical goal id."""
    if not human_guess:
        return "unknown"
    g = human_guess.lower().strip()
    if g in _CANONICAL_GOALS:
        return g
    for kw, canon in _GOAL_KEYWORDS.items():
        if kw in g:
            return canon
    return "unknown"


# ------------------------------------------------------------------
# Prior pack
# ------------------------------------------------------------------

@dataclass
class HumanPriorPack:
    """Derived priors ready to be handed to a v4_1 agent.

    This is an intermediate, inspection-friendly form of the corpus.
    `seed_game_memory` / `seed_cross_game_memory` consume it directly.
    """
    game_id: str
    # StepRecords in chronological order, aggregated across episodes.
    steps: List[StepRecord] = field(default_factory=list)
    episodes: List[EpisodeRecord] = field(default_factory=list)
    # Action-effect summary per action name.
    action_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Aggregated goal hint dicts keyed by canonical goal id.
    goal_hints: List[Dict[str, Any]] = field(default_factory=list)
    # Known-bad strategies keyed by goal id.
    failure_hints: List[Dict[str, Any]] = field(default_factory=list)
    # Human hypotheses to inject into GameMemory (hypothesis_key, confidence).
    hypothesis_priors: List[Tuple[str, float]] = field(default_factory=list)
    # Compact human trajectory fragments for proposal seeding.
    trajectory_fragments: List[Dict[str, Any]] = field(default_factory=list)


# ------------------------------------------------------------------
# Prior construction
# ------------------------------------------------------------------

def _summarise_actions(steps: Iterable[StepRecord]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for s in steps:
        if s.action == "RESET":
            continue
        b = stats.setdefault(s.action, {"tries": 0.0, "changes": 0.0, "deaths": 0.0, "wins": 0.0})
        b["tries"] += 1
        changed = s.frame_before != s.frame_after
        if changed:
            b["changes"] += 1
        if s.game_state_after == "GAME_OVER":
            b["deaths"] += 1
        if s.game_state_after == "WIN":
            b["wins"] += 1
    for b in stats.values():
        n = max(b["tries"], 1.0)
        b["change_rate"] = b["changes"] / n
        b["death_rate"] = b["deaths"] / n
        b["win_rate"] = b["wins"] / n
    return stats


def _derive_goal_hints(
    episodes: Iterable[EpisodeRecord],
    steps: Iterable[StepRecord],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (goal_hints, failure_hints) both keyed by canonical goal id."""
    goal_hints: List[Dict[str, Any]] = []
    failure_hints: List[Dict[str, Any]] = []

    # Group steps by episode for intent-sequence summaries.
    steps_by_ep: Dict[str, List[StepRecord]] = {}
    for s in steps:
        steps_by_ep.setdefault(s.episode_id, []).append(s)

    for ep in episodes:
        goal_id = canonicalise_goal(ep.game_type_guess)
        ep_steps = steps_by_ep.get(ep.episode_id, [])
        # Build a short action summary: ordered dominant actions.
        action_counts = Counter(s.action for s in ep_steps if s.action != "RESET")
        top_actions = [a for a, _ in action_counts.most_common(4)]
        # Intent tag distribution (for diagnostics / debugging).
        intent_counts = Counter(s.intent for s in ep_steps)

        hint: Dict[str, Any] = {
            "source": "human",
            "game_id": ep.game_id[:80],
            "goal_id": goal_id,
            "objective": (ep.objective_guess or "")[:80],
            "top_actions": top_actions,
            "n_steps": ep.n_steps,
            "levels_completed": ep.levels_completed,
            "final_state": ep.final_state,
            "won": ep.final_state == "WIN",
            "progress_score": 1.0 if ep.final_state == "WIN" else min(
                0.75, 0.15 * ep.levels_completed + 0.05
            ),
            "intent_distribution": dict(intent_counts.most_common(5)),
            "mechanics": [m[:80] for m in ep.discovered_mechanics[:6]],
        }

        # Classify: win/partial → goal_hint; loss with mistakes → failure_hint.
        is_failure = ep.final_state == "GAME_OVER" or bool(ep.discovered_mistakes)
        if ep.final_state == "WIN" or ep.levels_completed > 0:
            goal_hints.append(hint)
        if is_failure and (ep.discovered_mistakes or ep.final_state == "GAME_OVER"):
            fh = dict(hint)
            fh["reason"] = ";".join(m[:60] for m in ep.discovered_mistakes[:3]) or "game_over"
            failure_hints.append(fh)

    return goal_hints, failure_hints


def _derive_hypothesis_priors(
    episodes: Iterable[EpisodeRecord],
    steps: Iterable[StepRecord],
) -> List[Tuple[str, float]]:
    """Pull durable hypotheses out of sticky-hypothesis strings and discoveries."""
    out: List[Tuple[str, float]] = []
    # Count sticky hypothesis strings — more occurrences → more confidence.
    counter: Counter[str] = Counter()
    for s in steps:
        if s.hypothesis:
            counter[s.hypothesis.strip()[:80]] += 1
    for text, n in counter.most_common(12):
        conf = min(0.8, 0.35 + 0.05 * n)
        out.append((f"human::{text}", conf))

    for ep in episodes:
        if ep.game_type_guess:
            out.append((f"game_type::{canonicalise_goal(ep.game_type_guess)}", 0.65))
        if ep.objective_guess:
            out.append((f"objective::{ep.objective_guess.strip()[:60]}", 0.6))
        for m in ep.discovered_mechanics:
            out.append((f"mechanic::{m.strip()[:60]}", 0.55))
    return out


def _derive_trajectory_fragments(
    episodes: Iterable[EpisodeRecord],
    steps: Iterable[StepRecord],
) -> List[Dict[str, Any]]:
    """Build compact action fragments for proposal seeding."""
    steps_by_ep: Dict[str, List[StepRecord]] = {}
    for step in steps:
        steps_by_ep.setdefault(step.episode_id, []).append(step)

    out: List[Dict[str, Any]] = []
    for ep in episodes:
        goal_id = canonicalise_goal(ep.game_type_guess)
        ep_steps = [s for s in steps_by_ep.get(ep.episode_id, []) if s.action != "RESET"]
        if not ep_steps:
            continue
        actions = [s.action for s in ep_steps]
        action_data = [dict(s.action_args) if s.action_args else None for s in ep_steps]
        base_score = 1.0 if ep.final_state == "WIN" else min(0.75, 0.15 * ep.levels_completed + 0.05)
        max_len = min(5, len(actions))
        for frag_len in range(2, max_len + 1):
            out.append({
                "goal_family": goal_id,
                "objective_id": ep.objective_guess[:80] if ep.objective_guess else goal_id,
                "actions": actions[:frag_len],
                "action_data": action_data[:frag_len],
                "source": "human_trace",
                "score": base_score,
                "progress_delta": base_score,
                "prediction_match": 1.0 if ep.final_state == "WIN" else 0.5,
                "success": ep.final_state == "WIN" or ep.levels_completed > 0,
                "metadata": {
                    "game_id": ep.game_id[:80],
                    "episode_id": ep.episode_id[:32],
                },
            })
    return out


def build_prior_pack(
    corpus: TraceCorpus | str | Path,
    game_id: str,
) -> HumanPriorPack:
    """Build a `HumanPriorPack` for a single game.

    `corpus` can be either an already-loaded `TraceCorpus` or a directory
    path; in the latter case traces are loaded on demand.
    """
    if not isinstance(corpus, TraceCorpus):
        corpus = load_traces(corpus, game_id=game_id)

    entries = corpus.by_game.get(game_id, [])
    if not entries:
        logger.warning("No traces for game_id=%r", game_id)
        return HumanPriorPack(game_id=game_id)

    all_steps: List[StepRecord] = []
    all_eps: List[EpisodeRecord] = []
    for ep_bucket in entries:
        all_steps.extend(ep_bucket["steps"])  # type: ignore[arg-type]
        ep = ep_bucket["episode"]
        if ep is not None:
            all_eps.append(ep)  # type: ignore[arg-type]

    goal_hints, failure_hints = _derive_goal_hints(all_eps, all_steps)
    pack = HumanPriorPack(
        game_id=game_id,
        steps=all_steps,
        episodes=all_eps,
        action_stats=_summarise_actions(all_steps),
        goal_hints=goal_hints,
        failure_hints=failure_hints,
        hypothesis_priors=_derive_hypothesis_priors(all_eps, all_steps),
        trajectory_fragments=_derive_trajectory_fragments(all_eps, all_steps),
    )
    return pack


# ------------------------------------------------------------------
# Seeding v4_1 memories
# ------------------------------------------------------------------

def _grid_np(grid: List[List[int]]) -> np.ndarray:
    if not grid:
        return np.zeros((1, 1), dtype=np.int32)
    return np.array(grid, dtype=np.int32)


def seed_game_memory(pack: HumanPriorPack, game_memory: Any) -> Dict[str, int]:
    """Replay a human trace through `GameMemory.record_action(...)`.

    We construct `FrameDiff` objects on the fly using v4_1's public
    `GridAnalyzer.compute_diff` so the memory sees the same shape of
    input as during live play.

    Returns a small stats dict for logging.
    """
    # Lazy import so the human_trace package stays importable without
    # v4_1 on the path (e.g. for unit-testing the recorder).
    from v4_1_reasoning_system.arc_agi import GridAnalyzer  # noqa: WPS433

    stats = {"steps_replayed": 0, "clicks_replayed": 0, "hypotheses_added": 0}

    for step in pack.steps:
        if step.action == "RESET":
            game_memory.on_reset()
            continue
        g_before = _grid_np(step.frame_before)
        g_after = _grid_np(step.frame_after)
        # Align shapes — some games emit varying grid sizes across steps.
        if g_before.shape != g_after.shape:
            pad_h = max(g_before.shape[0], g_after.shape[0])
            pad_w = max(g_before.shape[1], g_after.shape[1])
            def _pad(g: np.ndarray) -> np.ndarray:
                out = np.zeros((pad_h, pad_w), dtype=g.dtype)
                out[: g.shape[0], : g.shape[1]] = g
                return out
            g_before = _pad(g_before)
            g_after = _pad(g_after)

        try:
            diff = GridAnalyzer.compute_diff(g_before, g_after)
        except Exception as e:  # pragma: no cover
            logger.debug("compute_diff failed on step %s: %s", step.step, e)
            continue

        game_memory.record_action(
            action_name=step.action,
            grid_before=g_before,
            grid_after=g_after,
            diff=diff,
            game_state=step.game_state_after,
            levels_completed=step.levels_completed_after,
        )
        stats["steps_replayed"] += 1

        if step.action == "ACTION6" and step.action_args:
            x = int(step.action_args.get("x", 0))
            y = int(step.action_args.get("y", 0))
            changed = diff.anything_changed if diff is not None else (step.frame_before != step.frame_after)
            level_changed = step.levels_completed_after > game_memory.current_level - 1
            try:
                game_memory.record_click(
                    pos=(y, x),  # record_click takes (y, x)
                    grid_before=g_before,
                    changed=bool(changed),
                    level_changed=bool(level_changed),
                )
                stats["clicks_replayed"] += 1
            except Exception:  # pragma: no cover
                logger.debug("record_click failed", exc_info=True)

    for key, conf in pack.hypothesis_priors:
        try:
            game_memory.add_hypothesis(key, float(conf))
            stats["hypotheses_added"] += 1
        except Exception:  # pragma: no cover
            continue

    # Reset online-only counters/history so the live agent starts with human
    # semantics and hypotheses, but without inheriting the human's "age" or
    # recent-step context. This keeps action profiles, click priors, player
    # identity, level action sequences, and visited-state priors intact.
    seeded_total_actions = int(getattr(game_memory, "total_actions", 0) or 0)
    seeded_total_resets = int(getattr(game_memory, "total_resets", 0) or 0)
    seeded_total_game_overs = int(getattr(game_memory, "total_game_overs", 0) or 0)
    seeded_max_level = int(getattr(game_memory, "max_level_reached", 0) or 0)
    try:
        game_memory.total_actions = 0
        game_memory.total_resets = 0
        game_memory.total_game_overs = 0
        game_memory.max_level_reached = 0
        game_memory.current_level = 0
        if hasattr(game_memory, "action_history"):
            game_memory.action_history.clear()
        if hasattr(game_memory, "click_history"):
            game_memory.click_history.clear()
        if hasattr(game_memory, "level_attempts"):
            game_memory.level_attempts.clear()
        if hasattr(game_memory, "_prev_grid"):
            game_memory._prev_grid = None
        if hasattr(game_memory, "_prev_grid_hash"):
            game_memory._prev_grid_hash = 0
    except Exception:  # pragma: no cover
        pass
    stats["seeded_total_actions"] = seeded_total_actions
    stats["seeded_total_resets"] = seeded_total_resets
    stats["seeded_total_game_overs"] = seeded_total_game_overs
    stats["seeded_max_level"] = seeded_max_level

    logger.info(
        "seed_game_memory(%s): %d steps, %d clicks, %d hypotheses replayed "
        "(seeded actions=%d resets=%d game_overs=%d max_level=%d; "
        "online counters reset for clean agent measurement)",
        pack.game_id, stats["steps_replayed"], stats["clicks_replayed"],
        stats["hypotheses_added"], seeded_total_actions, seeded_total_resets,
        seeded_total_game_overs, seeded_max_level,
    )
    return stats


def seed_cross_game_memory(
    pack: HumanPriorPack,
    cross_game: Any,
    max_hints_per_goal: int = 3,
    max_failures_per_goal: int = 5,
) -> Dict[str, int]:
    """Inject goal hints + failure patterns into `CrossGameMemory`.

    Respects v4_1's per-goal caps and the "lightweight dicts only" rule
    from the CrossGameMemory memory-leak fix (no StrategyOutcome objects).
    """
    stats = {"goal_hints": 0, "failure_hints": 0, "trajectory_priors": 0}

    # Goal hints
    for hint in pack.goal_hints:
        goal_id = hint.get("goal_id", "unknown")
        bucket = cross_game.goal_strategy_hints.setdefault(goal_id, [])
        # De-duplicate by (source, game_id, top_actions) signature.
        sig = (hint.get("source"), hint.get("game_id"), tuple(hint.get("top_actions", [])))
        existing_sigs = {
            (h.get("source"), h.get("game_id"), tuple(h.get("top_actions", [])))
            for h in bucket if isinstance(h, dict)
        }
        if sig in existing_sigs:
            continue
        bucket.append(hint)
        # Cap: keep top N by progress_score.
        if len(bucket) > max_hints_per_goal:
            bucket.sort(key=lambda h: -float(h.get("progress_score", 0.0)))
            del bucket[max_hints_per_goal:]
        stats["goal_hints"] += 1

    # Failure patterns
    for fh in pack.failure_hints:
        goal_id = fh.get("goal_id", "unknown")
        bucket = cross_game.failure_patterns.setdefault(goal_id, [])
        sig = (fh.get("source"), fh.get("game_id"), fh.get("reason"))
        existing = {(h.get("source"), h.get("game_id"), h.get("reason")) for h in bucket}
        if sig in existing:
            continue
        bucket.append(fh)
        if len(bucket) > max_failures_per_goal:
            del bucket[: len(bucket) - max_failures_per_goal]
        stats["failure_hints"] += 1

    # Trajectory fragments
    priors = getattr(cross_game, "trajectory_priors", None)
    if priors is None:
        cross_game.trajectory_priors = {}
        priors = cross_game.trajectory_priors
    for frag in pack.trajectory_fragments:
        goal_id = frag.get("goal_family", "unknown")
        bucket = priors.setdefault(goal_id, [])
        sig = (
            tuple(frag.get("actions", [])),
            frag.get("source"),
            frag.get("metadata", {}).get("game_id"),
        )
        existing = {
            (
                tuple(item.get("actions", [])),
                item.get("source"),
                item.get("metadata", {}).get("game_id"),
            )
            for item in bucket if isinstance(item, dict)
        }
        if sig in existing:
            continue
        bucket.append(frag)
        bucket.sort(key=lambda item: -float(item.get("progress_delta", 0.0)))
        del bucket[8:]
        stats["trajectory_priors"] += 1

    logger.info(
        "seed_cross_game_memory(%s): +%d hints, +%d failures, +%d trajectory priors",
        pack.game_id, stats["goal_hints"], stats["failure_hints"], stats["trajectory_priors"],
    )
    return stats
