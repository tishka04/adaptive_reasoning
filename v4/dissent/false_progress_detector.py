"""False-progress analysis for V4."""

from __future__ import annotations


class FalseProgressDetector:
    """Detect locally competent but globally sterile behavior."""

    def analyze(self, memory) -> dict[str, float]:
        lp, sp, tp = memory.game.progress.scores()
        recent = list(memory.fast.recent_transitions)[-15:]
        removal_pressure = 0.0
        if recent:
            removal_steps = sum(
                1 for transition in recent if transition.metadata.get("removed_values")
            )
            removal_pressure = removal_steps / len(recent)

        metrics = {
            "high_lp_low_sp": max(0.0, lp - sp),
            "high_sp_low_tp": max(0.0, sp - tp),
            "removal_without_terminal": removal_pressure * max(0.0, 1.0 - tp),
            "repeat_pressure": memory.game.progress.state.sterile_repeats / 100.0,
            "macro_without_closure": max(0.0, min(1.0, len(memory.game.rituals) / 6.0) - tp),
        }
        return metrics
