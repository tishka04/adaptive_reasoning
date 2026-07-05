"""Learned world and episode embeddings for Phase 4 retrieval."""

from __future__ import annotations

from collections import deque
from math import sqrt
from typing import Any

from ..ontology.ontology_hypotheses import ONTOLOGY_KINDS
from .common import clamp, entropy


class WorldEmbeddingModel:
    """Compact online embeddings for worlds, frames, and episode memories."""

    def __init__(self) -> None:
        self.frame_prototypes: dict[str, list[float]] = {}
        self.frame_counts: dict[str, float] = {}
        self.episode_prototypes: dict[str, dict[str, Any]] = {}
        self._episode_trace: deque[list[float]] = deque(maxlen=192)

    def embed_query(self, query: dict[str, object], obs, memory) -> list[float]:
        ontology_scores = {kind: 0.0 for kind in ONTOLOGY_KINDS}
        for item in getattr(memory.game, "current_ontologies", [])[:6]:
            ontology_scores[item.kind] = float(item.confidence)

        lp, sp, tp = memory.game.progress.scores()
        repeat_ratio = self._repeat_ratio(memory)
        object_count = len(obs.objects)
        small_objects = sum(1 for obj in obs.objects if obj.value != 0 and obj.area <= 12)
        unlocked = len(obs.topology.unlocked_regions)
        surprise_total = obs.surprise.total
        diff_size = float(obs.frame_diff.num_changed) if obs.frame_diff is not None else 0.0
        rules = len(memory.game.constraints.rules)
        validated_tel = len(memory.game.teleology.hypotheses())
        speculative_tel = len(memory.game.teleology.speculative_hypotheses())
        move_ops = len(memory.game.inducer.get_by_kind("move"))
        click_ops = len(memory.game.inducer.get_by_kind("click"))
        transform_ops = len(memory.game.inducer.get_by_kind("global_transform"))
        project_concentration = self._project_concentration(memory)
        phase_value = self._phase_value(memory.fast.current_phase)
        knowledge = memory.game.knowledge_level()
        player_present = 1.0 if obs.best_player is not None else 0.0
        clickable_ratio = (
            sum(1 for obj in obs.objects if obj.area <= 10) / max(object_count, 1)
            if object_count > 0 else 0.0
        )

        vector = [
            *(ontology_scores[kind] for kind in ONTOLOGY_KINDS),
            lp,
            sp,
            tp,
            surprise_total,
            min(1.0, object_count / 24.0),
            min(1.0, small_objects / max(object_count, 1)),
            min(1.0, unlocked / 4.0),
            min(1.0, diff_size / 12.0),
            repeat_ratio,
            min(1.0, move_ops / 5.0),
            min(1.0, click_ops / 4.0),
            min(1.0, transform_ops / 3.0),
            min(1.0, rules / 6.0),
            min(1.0, validated_tel / 3.0),
            min(1.0, speculative_tel / 4.0),
            project_concentration,
            phase_value,
            knowledge,
            player_present,
            clickable_ratio,
        ]
        return [clamp(value) for value in vector]

    def observe_frame(self, frame, obs, memory, query: dict[str, object]) -> list[float]:
        embedding = self.embed_query(query, obs, memory)
        frame.embedding = self._blend(frame.embedding, frame.prototype_weight, embedding)
        frame.prototype_weight += 1.0

        old = self.frame_prototypes.get(frame.frame_id)
        count = self.frame_counts.get(frame.frame_id, 0.0)
        self.frame_prototypes[frame.frame_id] = self._blend(old, count, embedding)
        self.frame_counts[frame.frame_id] = count + 1.0
        self._episode_trace.append(embedding)
        return embedding

    def observe_episode(self, memory, won: bool) -> None:
        if self._episode_trace:
            vector = self._mean(list(self._episode_trace))
        elif memory.fast.current_obs is not None:
            vector = self.embed_query(
                memory.game.frame_memory.build_query(memory.fast.current_obs, memory),
                memory.fast.current_obs,
                memory,
            )
        else:
            return

        top = memory.game.current_ontologies[0].kind if memory.game.current_ontologies else "unknown"
        lp, sp, tp = memory.game.progress.scores()
        if won or tp >= 0.25:
            terminal_style = "closure"
        elif sp >= 0.12:
            terminal_style = "structural"
        else:
            terminal_style = "explore"
        outcome = "win" if won else "loss"
        key = f"{top}|{terminal_style}|{outcome}"
        existing = self.episode_prototypes.get(key)
        count = float(existing.get("count", 0.0)) if existing is not None else 0.0
        blended = self._blend(existing.get("vector") if existing is not None else None, count, vector)
        self.episode_prototypes[key] = {
            "vector": blended,
            "count": count + 1.0,
            "ontology_kind": top,
            "terminal_style": terminal_style,
            "won": won,
        }

    def frame_similarity(self, query_embedding: list[float], frame) -> float:
        vector = frame.embedding or self.frame_prototypes.get(frame.frame_id)
        if not vector:
            return 0.0
        return self._cosine(query_embedding, vector)

    def episode_support(
        self,
        query_embedding: list[float],
        frame,
        *,
        prefer_success: bool = False,
        prefer_contrast: bool = False,
        query_ontology: str = "unknown",
        query_terminal: str = "explore",
    ) -> float:
        best = 0.0
        for data in self.episode_prototypes.values():
            vector = data.get("vector")
            if not vector:
                continue
            similarity = self._cosine(query_embedding, vector)
            if similarity <= 0.0:
                continue
            weight = 1.0
            ontology_kind = str(data.get("ontology_kind", "unknown"))
            terminal_style = str(data.get("terminal_style", "explore"))
            if terminal_style == frame.terminal_style or terminal_style == query_terminal:
                weight += 0.20
            if prefer_success and bool(data.get("won")):
                weight += 0.18
            if prefer_contrast:
                if ontology_kind != query_ontology:
                    weight += 0.22
                else:
                    weight -= 0.12
            elif ontology_kind == frame.ontology_kind:
                weight += 0.16
            score = similarity * max(0.0, weight)
            if score > best:
                best = score
        return clamp(best)

    def bridge_affinity(self, query_embedding: list[float], primary_frame, secondary_frame) -> float:
        if not self.episode_prototypes:
            return 0.0
        same = self.episode_support(
            query_embedding,
            secondary_frame,
            prefer_success=True,
            prefer_contrast=False,
            query_ontology=primary_frame.ontology_kind,
            query_terminal=secondary_frame.terminal_style,
        )
        contrast = self.episode_support(
            query_embedding,
            secondary_frame,
            prefer_success=False,
            prefer_contrast=True,
            query_ontology=primary_frame.ontology_kind,
            query_terminal=secondary_frame.terminal_style,
        )
        return clamp(0.55 * same + 0.45 * contrast)

    def seed_from_cross_game(self, cross_game) -> None:
        self.frame_prototypes = {
            key: [float(item) for item in value.get("vector", [])]
            for key, value in getattr(cross_game, "learned_world_frame_embeddings", {}).items()
        }
        self.frame_counts = {
            key: float(value.get("count", 0.0))
            for key, value in getattr(cross_game, "learned_world_frame_embeddings", {}).items()
        }
        self.episode_prototypes = {
            key: {
                "vector": [float(item) for item in value.get("vector", [])],
                "count": float(value.get("count", 0.0)),
                "ontology_kind": str(value.get("ontology_kind", "unknown")),
                "terminal_style": str(value.get("terminal_style", "explore")),
                "won": bool(value.get("won", False)),
            }
            for key, value in getattr(cross_game, "learned_world_episode_embeddings", {}).items()
        }

    def export_to_cross_game(self, cross_game) -> None:
        cross_game.learned_world_frame_embeddings = {
            key: {"vector": value, "count": self.frame_counts.get(key, 0.0)}
            for key, value in list(self.frame_prototypes.items())[:64]
        }
        cross_game.learned_world_episode_embeddings = {
            key: value
            for key, value in list(self.episode_prototypes.items())[:48]
        }

    def _project_concentration(self, memory) -> float:
        projects = memory.game.selected_projects[-12:]
        if not projects:
            return 0.0
        return max(projects.count(item) for item in set(projects)) / max(len(projects), 1)

    def _repeat_ratio(self, memory) -> float:
        hashes = list(memory.fast.recent_hashes)
        if not hashes:
            return 0.0
        return max(hashes.count(item) for item in set(hashes)) / max(len(hashes), 1)

    def _phase_value(self, phase: str) -> float:
        phases = {
            "sensory_ignorance": 0.0,
            "mechanical_stabilization": 0.20,
            "project_emergence": 0.40,
            "closure_pressure": 0.65,
            "compression": 0.85,
            "crisis": 1.0,
        }
        return phases.get(phase, 0.0)

    def _blend(self, current: list[float] | None, count: float, new: list[float]) -> list[float]:
        if not current:
            return list(new)
        total = count + 1.0
        return [
            ((old * count) + fresh) / max(total, 1e-9)
            for old, fresh in zip(current, new)
        ]

    def _mean(self, vectors: list[list[float]]) -> list[float]:
        if not vectors:
            return []
        totals = [0.0 for _ in vectors[0]]
        for vector in vectors:
            for index, value in enumerate(vector):
                totals[index] += value
        return [value / max(len(vectors), 1) for value in totals]

    def _cosine(self, left: list[float], right: list[float]) -> float:
        if not left or not right:
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = sqrt(sum(a * a for a in left))
        right_norm = sqrt(sum(b * b for b in right))
        if left_norm <= 1e-9 or right_norm <= 1e-9:
            return 0.0
        return clamp((dot / (left_norm * right_norm) + 1.0) * 0.5)
