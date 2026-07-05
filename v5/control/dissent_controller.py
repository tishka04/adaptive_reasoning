"""Dissent controller for V5.

A separate, skeptical controller that runs in parallel with the main
decision loop and has explicit authority to interrupt at most once per
8 actions. It watches for:

  - sterile branches (from ProgressTracker)
  - false-progress patterns (LP high, SP low; SP high, TP low)
  - ontology monoculture (one kind dominating for too long)
  - loop warnings (repeated state hashes)

When it interrupts, it returns a `RedirectIntent` with a concrete
primitive action (typically RESET, or a least-tried probe).
"""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, List, Optional

from ..mechanics.action_profiler import ActionProfiler
from ..schemas import GameObservation, PrimitiveAction
from ..schemas_ext import DissentReport, OntologyHypothesis, RedirectIntent


class DissentController:
    """Skeptical controller. Interrupts coherent-but-sterile loops."""

    def __init__(
        self,
        min_gap_between_interrupts: int = 8,
        max_ontology_flips: int = 2,
        ontology_flip_window: int = 30,
    ) -> None:
        self._last_interrupt_action: int = -999
        self._min_gap = int(min_gap_between_interrupts)
        self._recent_hashes: Deque[int] = deque(maxlen=20)
        self._ontology_run: List[str] = []
        self._last_report: DissentReport = DissentReport()
        # Ontology flip budget & state
        self._max_ontology_flips = int(max_ontology_flips)
        self._ontology_flips_used: int = 0
        self._forced_ontology_kind: Optional[str] = None
        self._forced_ontology_until_action: int = -1
        self._ontology_flip_window = int(ontology_flip_window)
        self._last_sp_stagnation_since: int = 0
        self._last_sp_score: float = 0.0

    # -----------------------------------------------------------------
    def update(
        self,
        obs: GameObservation,
        *,
        action_counter: int,
        lp: float,
        sp: float,
        tp: float,
        top_ontology: Optional[OntologyHypothesis],
        branch_kill_flag: bool,
    ) -> DissentReport:
        """Recompute the dissent report based on latest progress/ontology."""
        self._recent_hashes.append(obs.grid_hash)

        ontology_kind = top_ontology.kind if top_ontology else "unknown"
        self._ontology_run.append(ontology_kind)
        if len(self._ontology_run) > 40:
            self._ontology_run = self._ontology_run[-40:]

        # SP stagnation tracker — when sp has not improved for K steps
        if sp > self._last_sp_score + 0.02:
            self._last_sp_score = sp
            self._last_sp_stagnation_since = action_counter
        sp_stagnant_for = max(0, action_counter - self._last_sp_stagnation_since)

        # loop warning: same state hash dominates the recent window
        loop_warning = False
        if len(self._recent_hashes) >= 10:
            top_hash_count = max(
                self._recent_hashes.count(h) for h in set(self._recent_hashes)
            )
            if top_hash_count >= len(self._recent_hashes) * 0.5:
                loop_warning = True

        # ontology monoculture: one kind for >85% of the last 20 steps
        ontology_warnings: List[str] = []
        if len(self._ontology_run) >= 20:
            recent = self._ontology_run[-20:]
            dominant_count = max(recent.count(k) for k in set(recent))
            if dominant_count / len(recent) > 0.85:
                ontology_warnings.append(
                    f"ontology '{ontology_kind}' monoculture ({dominant_count}/{len(recent)})"
                )

        # false-progress signals
        false = {}
        if lp > 0.35 and sp < 0.12:
            false["high_lp_low_sp"] = round(lp - sp, 3)
        if sp > 0.35 and tp < 0.08:
            false["high_sp_low_tp"] = round(sp - tp, 3)
        if loop_warning:
            false["repeat_pressure"] = round(
                top_hash_count / max(len(self._recent_hashes), 1), 3
            )

        suggestions: List[str] = []
        if loop_warning or branch_kill_flag:
            suggestions.append("diversify")
        if false.get("high_sp_low_tp", 0.0) > 0.25:
            suggestions.append("closure")
        if false.get("high_lp_low_sp", 0.0) > 0.20:
            suggestions.append("experiment")
        if ontology_warnings:
            suggestions.append("reframe")

        report = DissentReport(
            false_progress=false,
            ontology_warnings=ontology_warnings,
            loop_warning=loop_warning,
            suggested_actions=suggestions,
        )
        # Stash latest stats for later decisions
        self._last_sp_stagnant_for = sp_stagnant_for
        self._last_top_ontology_kind = ontology_kind
        self._last_report = report
        return report

    # -----------------------------------------------------------------
    # Ontology flip API
    # -----------------------------------------------------------------
    def suggest_ontology_flip(self, action_counter: int) -> bool:
        """Should the agent flip to a different ontology right now?

        Triggers when:
          - SP has been stagnant for >= 60 actions
          - AND ontology monoculture was flagged in the latest report
          - AND budget remains
        """
        if self._ontology_flips_used >= self._max_ontology_flips:
            return False
        if self._last_report is None or not self._last_report.ontology_warnings:
            return False
        if getattr(self, "_last_sp_stagnant_for", 0) < 60:
            return False
        # Avoid flipping inside an existing forced window
        if action_counter <= self._forced_ontology_until_action:
            return False
        return True

    def force_ontology(
        self,
        kind: str,
        action_counter: int,
        steps: Optional[int] = None,
    ) -> None:
        """Clamp the top ontology to `kind` for up to `steps` actions.

        The agent is responsible for honouring this via its ontology
        competition (e.g. calling `downweight` on all other kinds).
        """
        if self._ontology_flips_used >= self._max_ontology_flips:
            return
        window = int(steps if steps is not None else self._ontology_flip_window)
        self._forced_ontology_kind = str(kind)
        self._forced_ontology_until_action = action_counter + window
        self._ontology_flips_used += 1

    def forced_ontology_active(self, action_counter: int) -> Optional[str]:
        """Return the forced kind if still in window, else None."""
        if self._forced_ontology_kind is None:
            return None
        if action_counter > self._forced_ontology_until_action:
            self._forced_ontology_kind = None
            return None
        return self._forced_ontology_kind

    @property
    def ontology_flips_used(self) -> int:
        return self._ontology_flips_used

    # -----------------------------------------------------------------
    def should_interrupt(self, action_counter: int) -> bool:
        if action_counter - self._last_interrupt_action < self._min_gap:
            return False
        r = self._last_report
        if r.loop_warning:
            return True
        if r.false_progress.get("repeat_pressure", 0.0) > 0.45:
            return True
        if r.ontology_warnings:
            return True
        return False

    # -----------------------------------------------------------------
    def interrupt_and_redirect(
        self,
        obs: GameObservation,
        profiler: ActionProfiler,
        action_counter: int,
        *,
        branch_kill_flag: bool,
        ontology_downweight_fn=None,
        top_ontology_id: Optional[str] = None,
    ) -> RedirectIntent:
        """Produce a redirect. Caller must respect it."""
        self._last_interrupt_action = action_counter
        r = self._last_report

        # 1. On branch kill, force a RESET
        if branch_kill_flag:
            return RedirectIntent(
                reason="branch_kill",
                primitive=PrimitiveAction("RESET"),
                metadata={"origin": "dissent"},
            )

        # 2. If ontology monoculture, downweight the dominant ontology
        if r.ontology_warnings and ontology_downweight_fn and top_ontology_id:
            try:
                ontology_downweight_fn(top_ontology_id, amount=0.5)
            except Exception:
                pass

        # 3. Probe with a least-tried action (or a random click if objects exist)
        available = [a for a in obs.available_actions if a != "RESET"]
        if obs.objects and "ACTION6" in obs.available_actions:
            target = random.choice(obs.objects[: min(len(obs.objects), 4)])
            y, x = int(round(target.center[0])), int(round(target.center[1]))
            return RedirectIntent(
                reason="skeptical_probe",
                primitive=PrimitiveAction("ACTION6", x=x, y=y),
                metadata={"origin": "dissent"},
            )

        # Fall back to least-tried non-reset action
        least_tried = _least_tried(profiler, available, top_k=3)
        chosen = least_tried[0] if least_tried else (available[0] if available else "ACTION1")
        return RedirectIntent(
            reason="diversify",
            primitive=PrimitiveAction(chosen),
            metadata={"origin": "dissent"},
        )

    # -----------------------------------------------------------------
    @property
    def last_report(self) -> DissentReport:
        return self._last_report


def _least_tried(
    profiler: ActionProfiler, available: List[str], top_k: int = 3
) -> List[str]:
    """Return the top_k least-tried actions among `available`.

    Uses the V3 ActionProfiler (has `least_tried_actions`) if the method
    exists, otherwise reads stats directly.
    """
    if hasattr(profiler, "least_tried_actions"):
        try:
            return profiler.least_tried_actions(available, top_k=top_k)
        except TypeError:
            # V4 signature uses `k=` instead of `top_k=`
            return profiler.least_tried_actions(available, k=top_k)

    stats = getattr(profiler, "stats", {})
    scored = []
    for a in available:
        s = stats.get(a)
        tries = getattr(s, "total_tries", 0) if s is not None else 0
        scored.append((tries, a))
    scored.sort()
    return [a for _, a in scored[:top_k]]
