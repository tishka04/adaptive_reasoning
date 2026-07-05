"""
State Describer — converts ARC-AGI-3 game observations into
structured natural-language descriptions for the LLM.

This is the "observation → language" bridge. The LLM receives these
descriptions and generates candidate strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .grid_analyzer import GridAnalyzer, GridObject, FrameDiff
from .game_memory import GameMemory


@dataclass
class GameObservation:
    """Structured observation of the current game state."""
    grid_description: str
    objects: List[Dict[str, Any]]
    player_info: Optional[Dict[str, Any]]
    action_semantics: Dict[str, str]
    memory_summary: Dict[str, Any]
    raw_grid: np.ndarray
    level: int
    game_state: str
    action_counter: int
    visual_cortex_summary: Optional[str] = None

    def to_prompt(self) -> str:
        """Format as a prompt string for the LLM."""
        parts = [
            f"## Game State (action {self.action_counter}, level {self.level})",
            f"State: {self.game_state}",
            "",
            "### Grid",
            self.grid_description,
            "",
        ]

        if self.player_info:
            parts.append("### Player")
            parts.append(
                f"- Position: ({self.player_info['y']:.0f}, {self.player_info['x']:.0f})"
            )
            parts.append(f"- Color value: {self.player_info['value']}")
            parts.append("")

        if self.objects:
            parts.append("### Objects")
            for obj in self.objects[:10]:  # cap to avoid huge prompts
                parts.append(
                    f"- Color {obj['value']}: {obj['size']} cells at "
                    f"({obj['center_y']:.0f}, {obj['center_x']:.0f})"
                )
            parts.append("")

        if self.action_semantics:
            parts.append("### Known Actions")
            for act, sem in self.action_semantics.items():
                parts.append(f"- {act}: {sem}")
            parts.append("")

        parts.append("### Memory")
        ms = self.memory_summary
        parts.append(f"- Actions taken: {ms.get('total_actions', 0)}")
        parts.append(f"- Unique states seen: {ms.get('states_visited', 0)}")
        parts.append(f"- Max level reached: {ms.get('max_level', 0)}")
        parts.append(f"- Game-overs: {ms.get('total_game_overs', 0)}")
        if ms.get("effective_click_values"):
            parts.append(
                f"- Clickable object colors: {sorted(ms['effective_click_values'])}"
            )

        if self.visual_cortex_summary:
            parts.append("")
            parts.append("### Visual Cortex Predictions")
            parts.append(self.visual_cortex_summary)

        return "\n".join(parts)


class StateDescriber:
    """
    Converts raw grid observations + game memory into structured
    natural-language descriptions.

    Adapts the v4_1 LLMParser concept: instead of parsing NL→task,
    this does observation→NL for the game domain.
    """

    # Color names for ARC-AGI grid values 0-15
    _COLOR_NAMES = {
        0: "black/empty", 1: "blue", 2: "red", 3: "green",
        4: "yellow", 5: "gray", 6: "magenta", 7: "orange",
        8: "cyan", 9: "maroon", 10: "teal", 11: "lime",
        12: "purple", 13: "pink", 14: "brown", 15: "white",
    }

    def describe(
        self,
        grid: np.ndarray,
        memory: GameMemory,
        game_state: str,
        levels_completed: int,
        action_counter: int,
        diff: Optional[FrameDiff] = None,
    ) -> GameObservation:
        """Build a full structured observation of the current game state."""
        grid_desc = self._describe_grid(grid)
        objects = self._describe_objects(grid, memory)
        player_info = self._describe_player(memory)
        action_sem = memory.infer_action_semantics()
        mem_summary = self._describe_memory(memory)

        return GameObservation(
            grid_description=grid_desc,
            objects=objects,
            player_info=player_info,
            action_semantics=action_sem,
            memory_summary=mem_summary,
            raw_grid=grid,
            level=levels_completed,
            game_state=game_state,
            action_counter=action_counter,
        )

    def _describe_grid(self, grid: np.ndarray) -> str:
        """Produce a compact text description of the grid."""
        h, w = grid.shape
        unique_vals = np.unique(grid)
        color_counts = {}
        for v in unique_vals:
            v = int(v)
            count = int(np.sum(grid == v))
            name = self._COLOR_NAMES.get(v, f"color_{v}")
            color_counts[name] = count

        lines = [f"Size: {h}x{w}"]
        # Sort by count descending; skip dominant background
        sorted_colors = sorted(color_counts.items(), key=lambda x: -x[1])
        bg_name, bg_count = sorted_colors[0]
        total = h * w
        lines.append(
            f"Background: {bg_name} ({bg_count}/{total} cells, "
            f"{100 * bg_count / total:.0f}%)"
        )
        for name, count in sorted_colors[1:]:
            lines.append(f"  {name}: {count} cells ({100 * count / total:.1f}%)")
        return "\n".join(lines)

    def _describe_objects(
        self, grid: np.ndarray, memory: GameMemory
    ) -> List[Dict[str, Any]]:
        """Find and describe non-background objects."""
        objects = GridAnalyzer.find_objects(grid, ignore_values={0}, min_size=2)
        result = []
        player_val = memory.player.value if memory.player.identified else None
        for obj in objects:
            cy, cx = obj.center
            is_player = (obj.value == player_val) if player_val is not None else False
            result.append({
                "value": obj.value,
                "color": self._COLOR_NAMES.get(obj.value, f"color_{obj.value}"),
                "size": obj.size,
                "center_y": cy,
                "center_x": cx,
                "bbox": obj.bbox,
                "is_player": is_player,
            })
        return result

    def _describe_player(self, memory: GameMemory) -> Optional[Dict[str, Any]]:
        """Describe the player entity if identified."""
        if not memory.player.identified or memory.player.position is None:
            return None
        py, px = memory.player.position
        return {
            "y": py,
            "x": px,
            "value": memory.player.value,
            "color": self._COLOR_NAMES.get(memory.player.value or 0, "unknown"),
            "confidence": memory.player.confidence,
        }

    def _describe_memory(self, memory: GameMemory) -> Dict[str, Any]:
        """Summarise game memory for the observation."""
        summary = memory.summary()
        summary["effective_click_values"] = list(memory.get_effective_click_values())

        # Add action effect summaries
        action_effects = {}
        for name, profile in memory.action_profiles.items():
            go_rate = profile.times_caused_game_over / max(profile.times_tried, 1)
            action_effects[name] = {
                "tried": profile.times_tried,
                "change_rate": profile.change_rate,
                "move_rate": profile.move_rate,
                "game_over_rate": go_rate,
            }
        summary["action_effects"] = action_effects
        return summary
