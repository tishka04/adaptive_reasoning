"""Frame memory for bissociative V4 retrieval."""

from __future__ import annotations

from collections import deque

from ..schemas import MemoryFrame


class FrameMemory:
    """Store coherent interpretive bundles rather than isolated facts only."""

    def __init__(self) -> None:
        self.frames: dict[str, MemoryFrame] = {}
        self.recent_frame_ids: deque[str] = deque(maxlen=80)

    def build_query(self, obs, memory) -> dict[str, object]:
        ontologies = getattr(memory.game, "current_ontologies", [])
        top = ontologies[0] if ontologies else None
        ontology_kind = top.kind if top is not None else "unknown"
        ranked_rules = getattr(memory.game.laws, "ranked_rules", [])
        dominant_laws = [rule.rule_id for rule in ranked_rules[:2]]
        current_project_id = memory.fast.current_project_id
        project = None
        if current_project_id and memory.game.project_market is not None:
            project = memory.game.project_market.projects.get(current_project_id)
        project_kind = project.kind if project is not None else "none"
        lp, sp, tp = memory.game.progress.scores()
        if tp >= 0.20:
            terminal_style = "closure"
        elif sp >= 0.12:
            terminal_style = "structural"
        else:
            terminal_style = "explore"
        salient_entities = [
            f"v{value}"
            for value in sorted({obj.value for obj in obs.objects if obj.value != 0})[:4]
        ]
        return {
            "ontology_kind": ontology_kind,
            "dominant_laws": dominant_laws,
            "project_kind": project_kind,
            "salient_entities": salient_entities,
            "terminal_style": terminal_style,
        }

    def observe(self, obs, memory) -> MemoryFrame:
        query = self.build_query(obs, memory)
        project_kind = str(query["project_kind"])
        frame_id = f"{query['ontology_kind']}|{project_kind}|{query['terminal_style']}"
        frame = self.frames.get(frame_id)
        if frame is None:
            frame = MemoryFrame(
                frame_id=frame_id,
                ontology_kind=str(query["ontology_kind"]),
                dominant_laws=list(query["dominant_laws"]),
                dominant_projects=[project_kind] if project_kind != "none" else [],
                salient_entities=list(query["salient_entities"]),
                terminal_style=str(query["terminal_style"]),
                atoms=list(query["dominant_laws"]),
                reliability=0.25,
            )
            self.frames[frame_id] = frame

        frame.dominant_laws = list(query["dominant_laws"])
        frame.dominant_projects = [project_kind] if project_kind != "none" else []
        frame.salient_entities = list(query["salient_entities"])
        frame.terminal_style = str(query["terminal_style"])
        frame.usage_count += 1

        transition = memory.fast.last_transition
        if transition is not None:
            sp_gain = float(transition.metadata.get("sp_delta", 0.0))
            tp_gain = float(transition.metadata.get("tp_delta", 0.0))
            predicted_ok = bool(transition.metadata.get("predicted_ok"))
            if tp_gain > 0.01 or transition.level_completed:
                frame.success_count += 1
            elif sp_gain <= 0.0 and not predicted_ok:
                frame.failure_count += 1
            reliability_gain = 0.10 * sp_gain + 0.18 * tp_gain + (0.03 if predicted_ok else -0.02)
            frame.reliability = min(0.95, max(0.05, frame.reliability * 0.92 + reliability_gain))

        if getattr(memory.game, "learning", None) is not None:
            memory.game.learning.world_embedding.observe_frame(frame, obs, memory, query)

        self.recent_frame_ids.append(frame_id)
        return frame

    def retrieve_best(self, query: dict[str, object], memory, limit: int = 3) -> list[MemoryFrame]:
        query_embedding = None
        if getattr(memory.game, "learning", None) is not None and memory.fast.current_obs is not None:
            query_embedding = memory.game.learning.world_embedding.embed_query(
                query,
                memory.fast.current_obs,
                memory,
            )
        ranked = sorted(
            self.frames.values(),
            key=lambda frame: self._score_match(
                frame,
                query,
                prefer_distance=False,
                query_embedding=query_embedding,
                memory=memory,
            ),
            reverse=True,
        )
        return ranked[:limit]

    def retrieve_distant_outcome_match(self, query: dict[str, object], memory, limit: int = 3) -> list[MemoryFrame]:
        query_embedding = None
        if getattr(memory.game, "learning", None) is not None and memory.fast.current_obs is not None:
            query_embedding = memory.game.learning.world_embedding.embed_query(
                query,
                memory.fast.current_obs,
                memory,
            )
        ranked = sorted(
            self.frames.values(),
            key=lambda frame: self._score_match(
                frame,
                query,
                prefer_distance=True,
                query_embedding=query_embedding,
                memory=memory,
            ),
            reverse=True,
        )
        return ranked[:limit]

    def _score_match(
        self,
        frame: MemoryFrame,
        query: dict[str, object],
        prefer_distance: bool,
        query_embedding,
        memory,
    ) -> float:
        score = 0.0
        ontology_kind = str(query["ontology_kind"])
        project_kind = str(query["project_kind"])
        terminal_style = str(query["terminal_style"])
        salient_entities = set(query["salient_entities"])

        if frame.ontology_kind == ontology_kind:
            score += 0.40 if not prefer_distance else -0.10
        else:
            score += 0.20 if prefer_distance else 0.02

        if frame.terminal_style == terminal_style:
            score += 0.22
        if project_kind != "none" and project_kind in frame.dominant_projects:
            score += 0.20

        entity_overlap = len(salient_entities & set(frame.salient_entities))
        score += 0.10 * entity_overlap
        score += 0.20 * frame.reliability
        if prefer_distance and frame.ontology_kind == ontology_kind:
            score -= 0.15
        if query_embedding is not None and getattr(memory.game, "learning", None) is not None:
            model = memory.game.learning.world_embedding
            similarity = model.frame_similarity(query_embedding, frame)
            episode_hint = model.episode_support(
                query_embedding,
                frame,
                prefer_success=not prefer_distance,
                prefer_contrast=prefer_distance,
                query_ontology=ontology_kind,
                query_terminal=terminal_style,
            )
            score += (0.30 if not prefer_distance else 0.20) * similarity
            score += (0.12 if not prefer_distance else 0.18) * episode_hint
        return score
