"""Loop criticism for V4."""

from __future__ import annotations


class AntiLoopCritic:
    """Detect repeated states, repeated projects, and repeated plans."""

    def analyze(self, memory) -> bool:
        hashes = list(memory.fast.recent_hashes)[-30:]
        if len(hashes) >= 20:
            unique_ratio = len(set(hashes)) / max(len(hashes), 1)
            if unique_ratio < 0.25:
                return True

        projects = memory.game.selected_projects[-20:]
        if len(projects) >= 8:
            top_count = max(projects.count(project_id) for project_id in set(projects))
            if top_count / len(projects) > 0.60:
                return True

        actions = list(memory.fast.recent_actions)[-16:]
        if len(actions) >= 8:
            half = len(actions) // 2
            if actions[:half] == actions[half:half * 2]:
                return True
        return False
