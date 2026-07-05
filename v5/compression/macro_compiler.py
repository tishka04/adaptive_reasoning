"""Macro compiler — compile repeated action sequences into reusable options.

Once short reliable sequences are discovered, they become parameterized
macros that can be reused in search and reactive control.
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

from ..schemas import (
    Effect,
    MacroAction,
    PlannedAction,
    Predicate,
    PrimitiveAction,
    SolvedTrajectory,
)

logger = logging.getLogger(__name__)

MIN_SEQUENCE_LENGTH = 2
MAX_SEQUENCE_LENGTH = 8
MIN_OCCURRENCES = 2
MAX_PROMOTED = 10          # max promoted (active) macros
MAX_EPHEMERAL = 30         # max ephemeral pattern cache
PROMOTION_REPEATS = 3      # min repeats to promote
PROMOTION_GAIN = 0.0       # min mean structural gain to promote


class MacroCompiler:
    """Two-tier macro system: ephemeral patterns → promoted macros.

    Ephemeral: frequent subsequences, cached cheaply, do NOT affect control.
    Promoted: validated subsequences that precede structural progress.
    Only promoted macros are returned to the agent for use.
    """

    def __init__(self) -> None:
        self.macros: Dict[str, MacroAction] = {}         # promoted only
        self._ephemeral: Dict[str, Dict] = {}            # pattern cache
        self._all_traces: List[List[str]] = []

    def observe_trace(
        self,
        actions: List[PrimitiveAction],
        success: bool,
    ) -> None:
        """Record an observed action trace for pattern mining."""
        names = [a.name for a in actions]
        self._all_traces.append(names)

        # Limit memory
        if len(self._all_traces) > 100:
            self._all_traces = self._all_traces[-100:]

        if success:
            self._mine_from_success(actions)

    def compile_from_trace(
        self,
        actions: List[PrimitiveAction],
        control_success_rate: float = 0.0,
        structural_gain: float = 0.0,
    ) -> List[MacroAction]:
        """Mine ephemeral patterns from trace. Promote if they have structural gain.

        Returns only newly promoted macros (not ephemeral patterns).
        """
        if len(self.macros) >= MAX_PROMOTED:
            return []

        names = [a.name for a in actions]
        self._all_traces.append(names)
        if len(self._all_traces) > 50:
            self._all_traces = self._all_traces[-50:]

        new_macros: List[MacroAction] = []

        # Mine short patterns (2-4 steps) into ephemeral cache
        for length in range(2, min(5, len(names) // 2 + 1)):
            seen: Dict[Tuple[str, ...], int] = {}
            for i in range(len(names) - length + 1):
                subseq = tuple(names[i:i + length])
                seen[subseq] = seen.get(subseq, 0) + 1

            for subseq, count in seen.items():
                if count < 2:
                    continue
                key = "_".join(subseq)
                eph = self._ephemeral.get(key)
                if eph is None:
                    eph = {"subseq": subseq, "count": 0,
                           "gains": [], "first_actions": None}
                    self._ephemeral[key] = eph
                eph["count"] += count
                if structural_gain > 0:
                    eph["gains"].append(structural_gain)
                if eph["first_actions"] is None:
                    # Store first occurrence actions for macro creation
                    for i in range(len(names) - length + 1):
                        if tuple(names[i:i + length]) == subseq:
                            eph["first_actions"] = actions[i:i + length]
                            break

        # Try to promote ephemeral patterns
        to_remove = []
        for key, eph in self._ephemeral.items():
            if eph["count"] < PROMOTION_REPEATS:
                continue
            mean_gain = (sum(eph["gains"]) / len(eph["gains"])
                         if eph["gains"] else 0.0)
            if mean_gain < PROMOTION_GAIN:
                continue
            if eph["first_actions"] is None:
                continue
            if len(self.macros) >= MAX_PROMOTED:
                break

            macro = self._create_macro(
                eph["subseq"], eph["first_actions"],
                name_prefix="promoted",
            )
            if macro.macro_id not in self.macros:
                macro.success_rate = max(0.3, control_success_rate)
                macro.times_used = eph["count"]
                macro.times_succeeded = max(1, int(eph["count"] * 0.5))
                self.macros[macro.macro_id] = macro
                new_macros.append(macro)
                to_remove.append(key)
                logger.info(
                    f"Promoted macro: {macro.name} "
                    f"({len(macro.steps)} steps, {eph['count']}×, "
                    f"gain={mean_gain:.2f})"
                )

        for key in to_remove:
            self._ephemeral.pop(key, None)

        # Prune ephemeral cache
        if len(self._ephemeral) > MAX_EPHEMERAL:
            # Keep highest-count patterns
            ranked = sorted(self._ephemeral.items(),
                            key=lambda kv: kv[1]["count"], reverse=True)
            self._ephemeral = dict(ranked[:MAX_EPHEMERAL])

        return new_macros

    def compile_from_solution(self, solved: SolvedTrajectory) -> List[MacroAction]:
        """Extract macros from a solved trajectory."""
        actions = solved.primitive_actions
        if len(actions) < MIN_SEQUENCE_LENGTH:
            return []

        new_macros: List[MacroAction] = []

        # Find repeated subsequences in the solution
        names = [a.name for a in actions]
        for length in range(MIN_SEQUENCE_LENGTH,
                            min(MAX_SEQUENCE_LENGTH, len(names) // 2) + 1):
            for i in range(len(names) - length + 1):
                subseq = tuple(names[i:i + length])
                # Count occurrences in this solution
                count = 0
                for j in range(len(names) - length + 1):
                    if tuple(names[j:j + length]) == subseq:
                        count += 1

                if count >= MIN_OCCURRENCES:
                    macro = self._create_macro(subseq, actions[i:i + length])
                    if macro.macro_id not in self.macros:
                        self.macros[macro.macro_id] = macro
                        new_macros.append(macro)
                        logger.info(
                            f"Compiled macro: {macro.name} "
                            f"({len(macro.steps)} steps, {count}× in solution)"
                        )

        # Also compile the entire solution as a macro
        if len(actions) <= MAX_SEQUENCE_LENGTH:
            full_macro = self._create_macro(
                tuple(names), actions,
                name_prefix=f"solution_L{solved.level_index}",
            )
            if full_macro.macro_id not in self.macros:
                self.macros[full_macro.macro_id] = full_macro
                full_macro.success_rate = 1.0
                full_macro.times_succeeded = 1
                full_macro.times_used = 1
                new_macros.append(full_macro)

        self._prune_macros()
        return new_macros

    def _mine_from_success(self, actions: List[PrimitiveAction]) -> None:
        """Mine patterns from a successful trace across all stored traces."""
        names = [a.name for a in actions]

        # Find subsequences that appear in multiple traces
        for length in range(MIN_SEQUENCE_LENGTH,
                            min(MAX_SEQUENCE_LENGTH, len(names)) + 1):
            for i in range(len(names) - length + 1):
                subseq = tuple(names[i:i + length])

                # Count across all traces
                total_count = 0
                for trace in self._all_traces:
                    for j in range(len(trace) - length + 1):
                        if tuple(trace[j:j + length]) == subseq:
                            total_count += 1
                            break  # count once per trace

                if total_count >= MIN_OCCURRENCES:
                    macro = self._create_macro(subseq, actions[i:i + length])
                    if macro.macro_id not in self.macros:
                        self.macros[macro.macro_id] = macro

    def _create_macro(
        self,
        name_seq: Tuple[str, ...],
        actions: List[PrimitiveAction],
        name_prefix: str = "macro",
    ) -> MacroAction:
        """Create a MacroAction from an action sequence."""
        seq_str = "_".join(name_seq)
        macro_id = hashlib.md5(seq_str.encode()).hexdigest()[:8]

        steps = [PlannedAction(primitive=a, purpose=f"step {i}")
                 for i, a in enumerate(actions)]

        # Simple name: most common action type
        action_counts = Counter(name_seq)
        dominant = action_counts.most_common(1)[0][0]

        return MacroAction(
            macro_id=macro_id,
            name=f"{name_prefix}_{dominant}_{len(actions)}",
            steps=steps,
            avg_cost=float(len(actions)),
        )

    def _prune_macros(self) -> None:
        """Keep only the top promoted macros."""
        if len(self.macros) <= MAX_PROMOTED:
            return

        ranked = sorted(
            self.macros.values(),
            key=lambda m: (m.success_rate, m.times_succeeded),
            reverse=True,
        )
        self.macros = {m.macro_id: m for m in ranked[:MAX_PROMOTED]}

    def record_usage(self, macro_id: str, success: bool) -> None:
        """Update macro stats after use."""
        m = self.macros.get(macro_id)
        if m is None:
            return
        m.times_used += 1
        if success:
            m.times_succeeded += 1
        m.success_rate = m.times_succeeded / max(m.times_used, 1)

    def get_applicable(
        self,
        obs: "GameObservation",
    ) -> List[MacroAction]:
        """Return macros whose initiation conditions are met."""
        return [
            m for m in self.macros.values()
            if all(p.check(obs) for p in m.initiation_conditions)
            and m.success_rate >= 0.3
        ]

    def summary(self) -> str:
        lines = []
        for m in sorted(self.macros.values(),
                        key=lambda m: m.success_rate, reverse=True):
            lines.append(
                f"  {m.name}: {len(m.steps)} steps, "
                f"success={m.success_rate:.2f}, used={m.times_used}"
            )
        return "\n".join(lines) if lines else "  (no macros)"
