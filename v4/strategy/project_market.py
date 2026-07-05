"""Project market for V4."""

from __future__ import annotations

from typing import Any

from ..schemas import Project


class ProjectMarket:
    """Maintain a market of candidate projects and their evolving dignity."""

    def __init__(self) -> None:
        self.projects: dict[str, Project] = {}
        self._suspended_kinds: dict[str, int] = {}
        self._step: int = 0
        self._last_memory: Any = None
        self._last_obs: Any = None
        self._last_feedback_key: tuple[object, ...] | None = None

    def update(self, new_projects: list[Project], memory: Any, obs: Any | None = None) -> None:
        self._step += 1
        self._last_memory = memory
        self._last_obs = obs
        self._tick_suspensions()
        self._apply_latest_feedback(memory)

        for project in new_projects:
            existing = self.projects.get(project.project_id)
            if existing is None:
                project.metadata["last_seen_step"] = self._step
                self.projects[project.project_id] = project
                continue
            existing.expected_info_gain = max(existing.expected_info_gain, project.expected_info_gain)
            existing.expected_structural_gain = max(
                existing.expected_structural_gain, project.expected_structural_gain
            )
            existing.expected_terminal_gain = max(
                existing.expected_terminal_gain, project.expected_terminal_gain
            )
            existing.estimated_cost = min(existing.estimated_cost, project.estimated_cost)
            existing.fragility = min(existing.fragility * 0.8 + project.fragility * 0.2, 1.0)
            existing.dignity = min(1.0, max(existing.dignity, project.dignity))
            existing.metadata.update(project.metadata)
            existing.metadata["last_seen_step"] = self._step

        stale_ids = []
        for project_id, project in self.projects.items():
            last_seen = int(project.metadata.get("last_seen_step", self._step))
            if self._step - last_seen > 120:
                project.dignity *= 0.92
            if project.dignity < 0.05:
                stale_ids.append(project_id)
        for project_id in stale_ids:
            self.projects.pop(project_id, None)

    def ranked(self) -> list[Project]:
        ranked = [
            project
            for project in self.projects.values()
            if project.active and project.kind not in self._suspended_kinds
        ]
        ranked.sort(key=self._score, reverse=True)
        return ranked

    def suspend_family(self, kind: str, steps: int = 20) -> None:
        self._suspended_kinds[kind] = max(self._suspended_kinds.get(kind, 0), steps)

    def apply_outcome(self, project_id: str, sp_delta: float, tp_delta: float) -> None:
        project = self.projects.get(project_id)
        if project is None:
            return
        before_dignity = project.dignity
        before_score = self._score(project)
        if tp_delta > 0.01:
            project.dignity = min(1.0, project.dignity + 0.10 + 0.5 * tp_delta)
        elif sp_delta > 0.01:
            project.dignity = min(1.0, project.dignity + 0.06 + 0.3 * sp_delta)
        else:
            project.dignity = max(0.0, project.dignity - 0.05 - 0.2 * project.fragility)
        after_score = self._score(project)
        project.metadata["last_outcome"] = {
            "sp_delta": round(sp_delta, 4),
            "tp_delta": round(tp_delta, 4),
            "before_dignity": round(before_dignity, 4),
            "after_dignity": round(project.dignity, 4),
            "before_score": round(before_score, 4),
            "after_score": round(after_score, 4),
        }

    def _apply_latest_feedback(self, memory: Any) -> None:
        transition = memory.fast.last_transition
        if transition is None:
            return
        project_id = transition.metadata.get("project_id")
        if not project_id:
            return
        feedback_key = self._feedback_key(transition)
        if feedback_key == self._last_feedback_key:
            return
        self.apply_outcome(
            str(project_id),
            float(transition.metadata.get("sp_delta", 0.0)),
            float(transition.metadata.get("tp_delta", 0.0)),
        )
        self._last_feedback_key = feedback_key

    def _tick_suspensions(self) -> None:
        expired = []
        for kind, remaining in self._suspended_kinds.items():
            remaining -= 1
            if remaining <= 0:
                expired.append(kind)
            else:
                self._suspended_kinds[kind] = remaining
        for kind in expired:
            self._suspended_kinds.pop(kind, None)

    def _score(self, project: Project) -> float:
        memory = self._last_memory
        reliability = 0.50
        learned = None
        if memory is not None and getattr(memory.game, "learning", None) is not None:
            reliability = memory.game.learning.world_reliability.estimate(memory)
            learned = memory.game.learning.project_value.estimate(project, self._last_obs, memory)
            project.metadata["learned_value"] = round(learned, 3)
            project.metadata["world_reliability"] = round(reliability, 3)

        project.survival_score = (
            0.25 * project.expected_info_gain
            + 0.25 * project.expected_structural_gain
            + 0.30 * project.expected_terminal_gain * (0.55 + 0.45 * reliability)
            + 0.10 * project.dignity
            - 0.10 * project.fragility
            - 0.10 * project.estimated_cost
        )
        if learned is not None:
            project.survival_score += 0.28 * (learned - 0.50)
        if project.kind == "closure_probe" and reliability < 0.30:
            project.survival_score -= 0.05
        return project.survival_score

    def prune(self, max_active: int = 6, memory: Any | None = None) -> list[str]:
        if memory is not None:
            self._last_memory = memory
        if len(self.projects) <= max_active:
            for project in self.projects.values():
                self._score(project)
            return []
        ranked = sorted(self.projects.values(), key=self._score, reverse=True)
        previous_ids = list(self.projects)
        self.projects = {
            project.project_id: project
            for project in ranked[:max_active]
            if project.survival_score > 0.10
        }
        return [project_id for project_id in previous_ids if project_id not in self.projects]

    def snapshot(self, project_id: str | None = None, limit: int = 3) -> dict[str, Any]:
        if project_id is not None:
            project = self.projects.get(project_id)
            if project is None:
                return {"exists": False}
            return {
                "exists": True,
                "project_id": project.project_id,
                "kind": project.kind,
                "dignity": round(project.dignity, 4),
                "score": round(self._score(project), 4),
                "expected_info_gain": round(project.expected_info_gain, 4),
                "expected_structural_gain": round(project.expected_structural_gain, 4),
                "expected_terminal_gain": round(project.expected_terminal_gain, 4),
            }

        ranked = self.ranked()[:limit]
        return {
            "top": [
                {
                    "project_id": project.project_id,
                    "kind": project.kind,
                    "dignity": round(project.dignity, 4),
                    "score": round(self._score(project), 4),
                }
                for project in ranked
            ]
        }

    def mark_feedback_applied(self, transition: Any) -> None:
        self._last_feedback_key = self._feedback_key(transition)

    def _feedback_key(self, transition: Any) -> tuple[object, ...]:
        return (
            transition.prev_hash,
            transition.next_hash,
            transition.action.name,
            transition.metadata.get("project_id"),
        )
