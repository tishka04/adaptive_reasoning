"""Cross-game memory for V4."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..schemas import Effect, Operator, Predicate, Ritual

INITIAL_TRUST = 0.10
MAX_TRUST = 0.20
MAX_FILE_SIZE_MB = 12


@dataclass
class CrossGameMemoryV4:
    """Persistent, trust-gated abstractions for V4."""

    ontology_priors: dict[str, float] = field(default_factory=dict)
    operator_templates: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    law_families: dict[str, dict[str, Any]] = field(default_factory=dict)
    terminal_motifs: dict[str, dict[str, Any]] = field(default_factory=dict)
    ritual_signatures: dict[str, dict[str, Any]] = field(default_factory=dict)
    anti_patterns: dict[str, dict[str, Any]] = field(default_factory=dict)
    mind_reliability: dict[str, float] = field(default_factory=dict)
    learned_project_priors: dict[str, float] = field(default_factory=dict)
    learned_ontology_priors: dict[str, float] = field(default_factory=dict)
    learned_bridge_priors: dict[str, float] = field(default_factory=dict)
    learned_compression_priors: dict[str, float] = field(default_factory=dict)
    learned_world_frame_embeddings: dict[str, dict[str, Any]] = field(default_factory=dict)
    learned_world_episode_embeddings: dict[str, dict[str, Any]] = field(default_factory=dict)
    game_digests: dict[str, dict[str, Any]] = field(default_factory=dict)
    trust: float = INITIAL_TRUST
    games_played: int = 0
    games_won: int = 0
    version: str = "v4"

    def seed_game(self, memory: Any) -> None:
        memory.game.ontology_priors.update(self.ontology_priors)
        if hasattr(memory.game, "learning"):
            memory.game.learning.seed_from_cross_game(self)
        for kind, templates in self.operator_templates.items():
            for template in templates[:2]:
                operator_id = str(template.get("operator_id", ""))
                if not operator_id or operator_id in memory.game.inducer.operators:
                    continue
                operator = Operator(
                    operator_id=operator_id,
                    kind=kind,
                    primitive_action=str(template.get("primitive_action", "ACTION1")),
                    parameters=dict(template.get("parameters", {})),
                    preconditions=[
                        Predicate(item["kind"], dict(item.get("args", {})))
                        for item in template.get("preconditions", [])
                    ],
                    expected_effects=[
                        Effect(item["kind"], dict(item.get("args", {})))
                        for item in template.get("expected_effects", [])
                    ],
                    confidence=min(0.35, float(template.get("confidence", 0.35)) * self.trust),
                    support=max(1, int(template.get("support", 2) * self.trust)),
                    contradictions=0,
                    risk_estimate=float(template.get("risk_estimate", 0.1)),
                    contexts_supported=list(template.get("contexts_supported", [])),
                )
                memory.game.inducer.operators[operator.operator_id] = operator

        for name, reliability in self.mind_reliability.items():
            stats = memory.game.mind_stats[name]
            stats["used"] += 2.0
            stats["correct"] += 2.0 * max(0.0, min(1.0, reliability))
            stats["progress"] += 1.5 * max(0.0, min(1.0, reliability))

        for ritual_id, data in list(self.ritual_signatures.items())[:1]:
            prefix = []
            for item in data.get("prefix", []):
                action_name = str(item.get("name", "ACTION1"))
                x = item.get("x")
                y = item.get("y")
                from ..schemas import PrimitiveAction

                prefix.append(PrimitiveAction(action_name, x=x, y=y))
            ritual = Ritual(
                ritual_id=ritual_id,
                ontology_kind=str(data.get("ontology_kind", "token_world")),
                prefix=prefix,
                terminal_signature=dict(data.get("terminal_signature", {})),
                success_rate=min(0.35, float(data.get("success_rate", 0.4)) * self.trust),
            )
            memory.game.add_ritual(ritual)

    def export_game(self, memory: Any, won: bool) -> None:
        self.games_played += 1
        if won:
            self.games_won += 1
        if hasattr(memory.game, "learning"):
            memory.game.learning.export_to_cross_game(self)

        ranked = getattr(memory.game, "ontology_history", [])
        if ranked:
            latest = ranked[-1]
            for kind, confidence in latest:
                old = self.ontology_priors.get(kind, 0.0)
                self.ontology_priors[kind] = 0.8 * old + 0.2 * float(confidence)

        pred_accuracy = memory.game.inducer.operator_predictive_accuracy()
        control_success = memory.game.inducer.operator_control_success()
        for operator in memory.game.inducer.operators.values():
            if operator.confidence < 0.55 or operator.survival_score < 0.12:
                continue
            bucket = self.operator_templates.setdefault(operator.kind, [])
            template = {
                "operator_id": operator.operator_id,
                "primitive_action": operator.primitive_action,
                "parameters": operator.parameters,
                "preconditions": [
                    {"kind": predicate.kind, "args": predicate.args}
                    for predicate in operator.preconditions
                ],
                "expected_effects": [
                    {"kind": effect.kind, "args": effect.args}
                    for effect in operator.expected_effects
                ],
                "confidence": operator.confidence,
                "support": operator.support,
                "risk_estimate": operator.risk_estimate,
                "contexts_supported": operator.contexts_supported,
            }
            existing_ids = {item["operator_id"] for item in bucket}
            if template["operator_id"] not in existing_ids:
                bucket.append(template)
            bucket.sort(key=lambda item: (item["confidence"], item["support"]), reverse=True)
            self.operator_templates[operator.kind] = bucket[:4]

        for rule in memory.game.teleology.hypotheses()[:5]:
            self.law_families[rule.rule_id] = {
                "family": rule.family,
                "confidence": rule.confidence,
                "support": rule.support,
                "ontology_tags": rule.ontology_tags,
                "stage": rule.stage,
            }

        for motif in memory.game.motifs.values():
            if motif.survival_score < 0.18:
                continue
            self.terminal_motifs[motif.motif_id] = {
                "kind": motif.kind,
                "content": motif.content,
                "support": motif.support,
                "utility": motif.utility,
                "terminal_association": motif.terminal_association,
                "structural_association": motif.structural_association,
            }

        for ritual in memory.game.rituals.values():
            if ritual.survival_score < 0.30:
                continue
            self.ritual_signatures[ritual.ritual_id] = {
                "ontology_kind": ritual.ontology_kind,
                "prefix": [
                    {"name": action.name, "x": action.x, "y": action.y}
                    for action in ritual.prefix
                ],
                "terminal_signature": ritual.terminal_signature,
                "success_rate": ritual.success_rate,
            }

        for name in memory.game.mind_stats:
            self.mind_reliability[name] = memory.game.mind_reliability(name)

        if won:
            self.trust = min(MAX_TRUST, self.trust + 0.03)
        elif pred_accuracy >= 0.55 and control_success >= 0.45:
            self.trust = min(MAX_TRUST, self.trust + 0.005)
        else:
            self.trust = max(INITIAL_TRUST, self.trust * 0.98)

    def save(self, path: str) -> None:
        data = {
            "ontology_priors": self.ontology_priors,
            "operator_templates": self.operator_templates,
            "law_families": self.law_families,
            "terminal_motifs": self.terminal_motifs,
            "ritual_signatures": self.ritual_signatures,
            "anti_patterns": self.anti_patterns,
            "mind_reliability": self.mind_reliability,
            "learned_project_priors": self.learned_project_priors,
            "learned_ontology_priors": self.learned_ontology_priors,
            "learned_bridge_priors": self.learned_bridge_priors,
            "learned_compression_priors": self.learned_compression_priors,
            "learned_world_frame_embeddings": self.learned_world_frame_embeddings,
            "learned_world_episode_embeddings": self.learned_world_episode_embeddings,
            "game_digests": self.game_digests,
            "trust": self.trust,
            "games_played": self.games_played,
            "games_won": self.games_won,
            "version": self.version,
        }
        with open(path, "wb") as handle:
            pickle.dump(data, handle)
        size_mb = Path(path).stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            print(f"Warning: cross-game memory is {size_mb:.1f}MB")

    @classmethod
    def load(cls, path: str) -> "CrossGameMemoryV4":
        try:
            with open(path, "rb") as handle:
                data = pickle.load(handle)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            return cls()

        if data.get("version") != "v4":
            return cls()

        memory = cls()
        memory.ontology_priors = data.get("ontology_priors", {})
        memory.operator_templates = data.get("operator_templates", {})
        memory.law_families = data.get("law_families", {})
        memory.terminal_motifs = data.get("terminal_motifs", {})
        memory.ritual_signatures = data.get("ritual_signatures", {})
        memory.anti_patterns = data.get("anti_patterns", {})
        memory.mind_reliability = data.get("mind_reliability", {})
        memory.learned_project_priors = data.get("learned_project_priors", {})
        memory.learned_ontology_priors = data.get("learned_ontology_priors", {})
        memory.learned_bridge_priors = data.get("learned_bridge_priors", {})
        memory.learned_compression_priors = data.get("learned_compression_priors", {})
        memory.learned_world_frame_embeddings = data.get("learned_world_frame_embeddings", {})
        memory.learned_world_episode_embeddings = data.get("learned_world_episode_embeddings", {})
        memory.game_digests = data.get("game_digests", {})
        memory.trust = min(MAX_TRUST, data.get("trust", INITIAL_TRUST))
        memory.games_played = data.get("games_played", 0)
        memory.games_won = data.get("games_won", 0)
        return memory
