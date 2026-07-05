"""Cognitive phase taxonomy mappings for human trace experiments."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from build_cognitive_trace_dataset import PHASE_LABELS

TAXONOMY_V1 = "v1"
TAXONOMY_V2 = "v2"

TAXONOMY_V2_LABELS: List[str] = [
    "explore_movement",
    "explore_click",
    "explore_boundary",
    "explore_object",
    "explore_global_rule",
    "test_move",
    "test_control_or_activation",
    "test_object_interaction",
    "probe_control_object",
    "probe_color_object",
    "probe_hazard_object",
    "probe_bridge_or_path",
    "probe_matching_object",
    "reach_target",
    "avoid_danger",
    "repeat_success",
    "recover_after_failure",
]


def taxonomy_labels(taxonomy: str) -> List[str]:
    if taxonomy == TAXONOMY_V1:
        return list(PHASE_LABELS)
    if taxonomy == TAXONOMY_V2:
        return list(TAXONOMY_V2_LABELS)
    raise ValueError(f"Unknown cognitive taxonomy: {taxonomy}")


def map_cognitive_phase(row: Dict[str, Any], taxonomy: str) -> Optional[str]:
    """Map a cognitive dataset row to the requested taxonomy label.

    Taxonomy v2 is intentionally conservative: it only splits the two known
    toxic meta-classes (explore_unknown/probe_object), renames the click and
    interaction tests, and drops the technical ``none`` label.
    """

    phase = str(row.get("cognitive_phase", ""))
    if taxonomy == TAXONOMY_V1:
        return phase if phase in PHASE_LABELS else None
    if taxonomy != TAXONOMY_V2:
        raise ValueError(f"Unknown cognitive taxonomy: {taxonomy}")

    if phase == "none":
        return None
    if phase == "test_click":
        return "test_control_or_activation"
    if phase == "test_interaction":
        return "test_object_interaction"
    if phase == "explore_unknown":
        return _map_explore_unknown(row)
    if phase == "probe_object":
        return _map_probe_object(row)
    if phase in {
        "test_move",
        "reach_target",
        "avoid_danger",
        "repeat_success",
        "recover_after_failure",
    }:
        return phase
    return None


def _row_text(row: Dict[str, Any]) -> str:
    parts: List[str] = [
        str(row.get("hypothesis_label", "")),
        str(row.get("hypothesis", "")),
        str(row.get("episode_game_type_guess", "")),
        str(row.get("episode_objective_guess", "")),
    ]
    return " ".join(parts).lower()


def _hypothesis_text(row: Dict[str, Any]) -> str:
    return " ".join(
        [
            str(row.get("hypothesis_label", "")),
            str(row.get("hypothesis", "")),
        ]
    ).lower()


def _has_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _map_explore_unknown(row: Dict[str, Any]) -> str:
    text = _row_text(row)
    action = str(row.get("action", ""))
    available = set(str(item) for item in row.get("available_actions", []) or [])
    core_changed = int(row.get("core_changed_cells") or row.get("changed_cells") or 0)

    if _has_any(
        text,
        (
            "boundary",
            "border",
            "wall",
            "barrier",
            "blocked",
            "limit",
            "timer",
            "moves left",
            "steps left",
            "game_over",
            "kill",
            "danger",
            "hazard",
        ),
    ) or (action in {"ACTION1", "ACTION2", "ACTION3", "ACTION4"} and core_changed == 0):
        return "explore_boundary"
    if _has_any(
        text,
        (
            "action",
            "move",
            "moves",
            "cursor",
            "left",
            "right",
            "up",
            "down",
            "rotate",
        ),
    ) or action in {"ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5"}:
        return "explore_movement"
    if _has_any(
        text,
        (
            "must",
            "need",
            "rule",
            "match",
            "objective",
            "target",
            "goal",
            "reproduce",
            "replicate",
        ),
    ):
        return "explore_global_rule"
    if _has_any(
        text,
        (
            "object",
            "shape",
            "cell",
            "square",
            "color",
            "bridge",
            "teleporter",
            "clamp",
            "button",
            "switch",
        ),
    ):
        return "explore_object"
    if action == "ACTION6" or row.get("click_x") is not None or available == {"ACTION6"}:
        return "explore_click"
    return "explore_global_rule"


def _map_probe_object(row: Dict[str, Any]) -> str:
    text = _hypothesis_text(row)
    game_type = str(row.get("episode_game_type_guess", "")).lower()
    hypothesis = str(row.get("hypothesis_label", "") or "")

    if _has_any(
        text,
        (
            "hazard",
            "danger",
            "kill",
            "black_cells_kill",
            "game_over",
            "timer",
            "moves left",
            "steps left",
        ),
        ):
        return "probe_hazard_object"
    if _has_any(
        text,
        (
            "bridge",
            "path",
            "way",
            "teleporter",
            "gravity",
            "barrier",
            "open",
            "reach",
            "target",
            "clamp",
            "door",
        ),
    ):
        return "probe_bridge_or_path"
    if _has_any(
        text,
        (
            "match",
            "matching",
            "model",
            "mirror",
            "miror",
            "replicate",
            "reproduce",
            "centered",
            "overlap",
            "correspond",
        ),
    ):
        return "probe_matching_object"
    if _has_any(
        text,
        (
            "control",
            "action",
            "cursor",
            "click",
            "button",
            "switch",
            "select",
            "activate",
        ),
    ):
        return "probe_control_object"
    if _has_any(
        text,
        (
            "color",
            "colored",
            "paint",
            "blue",
            "green",
            "red",
            "yellow",
            "purple",
            "black",
            "grey",
            "gray",
            "fancy",
        ),
    ):
        return "probe_color_object"
    if hypothesis == "__none__":
        if "matching" in game_type or "shape" in game_type:
            return "probe_matching_object"
        if "transform" in game_type:
            return "probe_bridge_or_path"
        return "probe_control_object"
    if str(row.get("action", "")) == "ACTION6":
        return "probe_control_object"
    return "probe_control_object"
