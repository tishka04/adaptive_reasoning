"""Phase-2 label generators: turn heuristics into continuous training labels.

The heuristics are used here as *label generators* (the teacher), not as a
policy. All progress signals are kept continuous (not just boolean) so the
value model can learn a smooth gradient.

Label fields produced per transition:
    break_progress           - largest component shrank (continuous)
    fragmentation_progress   - second family count up & second-largest down
    correspondence_progress  - top-pair global correspondence increased
    danger                   - game over event (1.0/0.0)
    no_op                    - grid unchanged (continuous, 1.0 -> 0.0)
    macro_label              - argmax affordance among the macro vocabulary

``auto_levelup_progress`` and the Monte-Carlo value targets (future_level_up,
steps_to_level_up, progress_score) are computed in the dataset builder because
they need episode/game level context (reference states, future outcomes).
"""

from __future__ import annotations

from typing import Dict

MACRO_LABELS = [
    "BREAK_LARGEST_COMPONENT",
    "ALIGN_COMPONENTS",
    "CORRESPOND",
    "EXPLORE_ACTION",
    "AVOID",
    "UNKNOWN",
]

MACRO_SCORE_NAMES = [
    "break",
    "align",
    "correspond",
    "explore",
    "avoid",
]

# A signal must clear this magnitude to claim a non-trivial macro affordance.
MACRO_EPSILON = 0.5


def break_progress(delta: Dict[str, float]) -> float:
    """Positive when the largest component shrinks (breaking apart)."""

    return -float(delta.get("delta_largest_component_size", 0.0))


def fragmentation_progress(delta: Dict[str, float]) -> float:
    """Positive when the second family fragments: more pieces, smaller max."""

    d_count = float(delta.get("delta_top_pair_0_b_count", 0.0))
    d_largest = float(delta.get("delta_top_pair_0_b_largest", 0.0))
    return d_count - d_largest


def correspondence_progress(delta: Dict[str, float]) -> float:
    """Positive when the top color pair becomes structurally more similar."""

    return float(delta.get("delta_top_pair_0_global_correspondence", 0.0))


def no_op_score(changed_cells: int, total_cells: int) -> float:
    """1.0 when nothing changed, decaying to 0.0 as more cells change."""

    if changed_cells <= 0:
        return 1.0
    # ~2% of the grid changing already counts as a clearly active step.
    denom = max(1.0, 0.02 * float(total_cells))
    return float(max(0.0, 1.0 - changed_cells / denom))


def macro_scores(
    *,
    delta: Dict[str, float],
    changed_cells: int,
    total_cells: int,
    game_over: bool,
    auto_levelup_progress: float = 0.0,
) -> Dict[str, float]:
    """Continuous affordance targets that avoid collapsing into UNKNOWN.

    These are teacher scores, not policy decisions. Consumers can train a
    multi-output regressor or threshold individual affordances instead of
    forcing every transition into one mutually-exclusive class.
    """

    return {
        "break": max(0.0, break_progress(delta)),
        "align": max(0.0, fragmentation_progress(delta)),
        "correspond": max(
            0.0,
            correspondence_progress(delta),
            float(auto_levelup_progress),
        ),
        "explore": no_op_score(changed_cells, total_cells),
        "avoid": 1.0 if game_over else 0.0,
    }


def macro_label(
    *,
    delta: Dict[str, float],
    changed_cells: int,
    total_cells: int,
    game_over: bool,
    auto_levelup_progress: float = 0.0,
) -> str:
    """Argmax affordance the teacher would assign to this transition."""

    if game_over:
        return "AVOID"
    if no_op_score(changed_cells, total_cells) >= 0.999:
        return "EXPLORE_ACTION"

    candidates = {
        "BREAK_LARGEST_COMPONENT": break_progress(delta),
        "ALIGN_COMPONENTS": fragmentation_progress(delta),
        "CORRESPOND": max(
            correspondence_progress(delta),
            float(auto_levelup_progress),
        ),
    }
    best_macro = max(candidates, key=lambda k: candidates[k])
    if candidates[best_macro] <= MACRO_EPSILON:
        return "UNKNOWN"
    return best_macro
