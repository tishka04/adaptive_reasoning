"""Strict local-model adapter for bounded SAGE12 hypothesis generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, Tuple

from .hypotheses import (
    ALLOWED_PREDICATES,
    SemanticHypothesis,
    hypotheses_from_json,
)
from .scene_graph import SceneGraph


class LocalJSONModel(Protocol):
    """Minimal interface supported by llama.cpp, Transformers, or test doubles."""

    def generate_json(
        self,
        *,
        prompt: str,
        schema: Mapping[str, Any],
        maximum_tokens: int,
    ) -> str:
        ...


@dataclass(frozen=True)
class TransformersModelConfig:
    """Configuration for a local open-weight Hugging Face causal model."""

    model_path: str = "models/qwen2_5_0.5b_instruct"
    device: str = "auto"
    temperature: float = 0.0
    local_files_only: bool = True


class TransformersJSONModel:
    """Lazy, cached local Transformers backend with optional CUDA placement."""

    def __init__(
        self,
        config: TransformersModelConfig | None = None,
    ) -> None:
        self.config = config or TransformersModelConfig()
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def generate_json(
        self,
        *,
        prompt: str,
        schema: Mapping[str, Any],
        maximum_tokens: int,
    ) -> str:
        tokenizer, model = self._load()
        messages = [
            {
                "role": "system",
                "content": (
                    "Return one JSON value only. It must satisfy this schema: "
                    + json.dumps(schema, sort_keys=True)
                ),
            },
            {"role": "user", "content": prompt},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        attention_mask = (
            (inputs != tokenizer.pad_token_id).long()
            if tokenizer.pad_token_id is not None
            else None
        )
        generation_kwargs = {
            "max_new_tokens": max(32, int(maximum_tokens)),
            "do_sample": self.config.temperature > 0.0,
            "pad_token_id": (
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            ),
        }
        if self.config.temperature > 0.0:
            generation_kwargs["temperature"] = float(
                self.config.temperature
            )
        output = model.generate(
            inputs,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
        return tokenizer.decode(
            output[0][inputs.shape[-1] :],
            skip_special_tokens=True,
        ).strip()

    def _load(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        path = Path(self.config.model_path)
        if not path.exists():
            raise RuntimeError(f"local model path does not exist: {path}")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "transformers is required for the local SAGE12 LLM"
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(path),
            local_files_only=self.config.local_files_only,
        )
        device_map = (
            "auto"
            if self.config.device == "auto"
            else {"": self.config.device}
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            str(path),
            torch_dtype="auto",
            device_map=device_map,
            local_files_only=self.config.local_files_only,
        )
        self._model.eval()
        return self._tokenizer, self._model


@dataclass(frozen=True)
class HypothesisGenerationResult:
    hypotheses: Tuple[SemanticHypothesis, ...]
    parse_error: str = ""
    raw_response: str = ""


class LocalHypothesisGenerator:
    """Proposal-only adapter. Its output cannot directly execute an action."""

    def __init__(
        self,
        model: LocalJSONModel,
        *,
        maximum_hypotheses: int = 8,
        maximum_tokens: int = 1_024,
    ) -> None:
        self.model = model
        self.maximum_hypotheses = max(1, int(maximum_hypotheses))
        self.maximum_tokens = max(128, int(maximum_tokens))

    def generate(
        self,
        *,
        graph: SceneGraph,
        available_actions: Sequence[str],
        subgoal: str,
    ) -> HypothesisGenerationResult:
        prompt = _proposal_prompt(graph, available_actions, subgoal)
        raw = self.model.generate_json(
            prompt=prompt,
            schema=HYPOTHESIS_RESPONSE_SCHEMA,
            maximum_tokens=self.maximum_tokens,
        )
        try:
            parsed = hypotheses_from_json(
                raw,
                maximum=self.maximum_hypotheses,
            )
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            return HypothesisGenerationResult(
                hypotheses=(),
                parse_error=f"{type(exc).__name__}: {exc}",
                raw_response=str(raw)[:4_096],
            )
        return HypothesisGenerationResult(
            hypotheses=parsed,
            raw_response=str(raw)[:4_096],
        )


class TemplateHypothesisGenerator:
    """Cheap deterministic control and fallback; it never fabricates support."""

    def generate(
        self,
        *,
        graph: SceneGraph,
        available_actions: Sequence[str],
        subgoal: str,
    ) -> HypothesisGenerationResult:
        hypotheses = []
        for index, action in enumerate(available_actions):
            hypotheses.append(
                SemanticHypothesis.from_mapping(
                    {
                        "hypothesis_id": f"template_{index}",
                        "action_name": action,
                        "effects": [
                            {
                                "predicate": {
                                    "name": (
                                        "level_complete"
                                        if subgoal == "level_complete"
                                        else "progress"
                                    )
                                }
                            }
                        ],
                        "confidence": 0.25,
                        "source": "template",
                        "support": 0,
                    }
                )
            )
        return HypothesisGenerationResult(hypotheses=tuple(hypotheses))


_ENTITY_REF_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": ["role"],
    "properties": {
        "role": {"type": "string"},
        "selector": {"type": "string"},
    },
    "additionalProperties": False,
}
_PREDICATE_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": ["name"],
    "properties": {
        "name": {"type": "string", "enum": list(ALLOWED_PREDICATES)},
        "subject": _ENTITY_REF_SCHEMA,
        "object": _ENTITY_REF_SCHEMA,
        "value": {"type": "string"},
    },
    "additionalProperties": False,
}
HYPOTHESIS_RESPONSE_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": ["hypotheses"],
    "properties": {
        "hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "hypothesis_id",
                    "action_name",
                    "effects",
                    "confidence",
                ],
                "properties": {
                    "hypothesis_id": {"type": "string"},
                    "action_name": {"type": "string"},
                    "action_data": {"type": "object"},
                    "preconditions": {
                        "type": "array",
                        "items": _PREDICATE_SCHEMA,
                    },
                    "effects": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "required": ["predicate"],
                            "properties": {
                                "predicate": _PREDICATE_SCHEMA,
                                "operation": {
                                    "type": "string",
                                    "enum": ["assert", "retract"],
                                },
                            },
                            "additionalProperties": False,
                        },
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "rationale": {"type": "string"},
                    "support": {"const": 0},
                },
            },
        }
    },
}


def _proposal_prompt(
    graph: SceneGraph,
    available_actions: Sequence[str],
    subgoal: str,
) -> str:
    scene = {
        "entities": [
            {
                "id": entity.entity_id,
                "roles": entity.roles,
                "area": entity.area_bucket,
                "aspect": entity.aspect_bucket,
            }
            for entity in graph.entities
        ],
        "relations": [relation.key for relation in graph.relations],
        "available_actions": list(available_actions),
        "subgoal": subgoal,
    }
    return (
        "Propose bounded causal hypotheses for the supplied abstract scene. "
        "Use only supplied actions, structural roles, and these predicates: "
        f"{', '.join(ALLOWED_PREDICATES)}. Return JSON only. "
        "All support fields must be 0 because "
        "a proposal is not observed evidence. Do not infer game identity, "
        "hidden rules, success, or safety as fact.\n"
        + json.dumps(scene, sort_keys=True)
    )


__all__ = [
    "HYPOTHESIS_RESPONSE_SCHEMA",
    "HypothesisGenerationResult",
    "LocalHypothesisGenerator",
    "LocalJSONModel",
    "TemplateHypothesisGenerator",
    "TransformersJSONModel",
    "TransformersModelConfig",
]
