"""
Semantic parser — uses an instruction-tuned LLM to convert a
natural-language problem statement into a structured TaskObject.

Designed for 7-8B instruct models (Llama 3.1 8B, Qwen2.5 7B, Mistral 7B).
Supports HuggingFace transformers with optional quantisation.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .task_schema import (
    AmbiguityMarker,
    Constraint,
    ConstraintType,
    DomainGuess,
    Objective,
    ObjectiveSense,
    TaskObject,
)

# ------------------------------------------------------------------
# System prompt for structured parsing
# ------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are a structured-reasoning parser. Given a user problem, output a JSON
object with exactly these keys:
{
  "domain": one of "planning", "scheduling", "optimization", "coding", "unknown",
  "description": short summary of the problem,
  "entities": list of named entities / objects involved,
  "constraints": [
    {"name": str, "description": str, "ctype": "hard"|"soft", "params": {}}
  ],
  "objective": {
    "description": str,
    "sense": "minimize"|"maximize"|"satisfy",
    "metric": str or null,
    "params": {}
  } or null,
  "ambiguities": [
    {"field": str, "reason": str, "severity": float 0-1}
  ],
  "structured_data": {}
}

CRITICAL: The "structured_data" field must contain machine-readable data:
- For scheduling: {"type":"scheduling","jobs":[{"name":str,"duration":int,"dependencies":[str]}],"horizon":int,"resources":{}}
- For optimization/knapsack: {"type":"knapsack","items":[{"name":str,"weight":number,"value":number}],"capacity":number}
- For planning: {"subgoals":[{"name":str,"description":str,"dependencies":[str],"params":{}}]}
- For assignment: {"type":"assignment","costs":[[int]]}
Extract ALL numeric values (durations, weights, capacities) from the problem text into structured_data.
Return ONLY valid JSON. No markdown, no explanation."""


class LLMParser:
    """Wraps a causal LM to produce TaskObjects from natural language."""

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        device: Optional[str] = None,
        load_in_4bit: bool = True,
        max_new_tokens: int = 1024,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        try:
            import torch as _torch
            self.device = device or ("cuda" if _torch.cuda.is_available() else "cpu")
        except ImportError:
            self.device = device or "cpu"

        # Lazy-loaded
        self._tokenizer = None
        self._model = None
        self._load_in_4bit = load_in_4bit

    # ----- lazy loading -----
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        quant_kwargs: Dict[str, Any] = {}
        if self._load_in_4bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig

            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            **quant_kwargs,
        )
        if self.device != "cuda":
            self._model = self._model.to(self.device)

    # ----- generation -----
    def _generate(self, user_text: str) -> str:
        import torch

        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"<s>[INST] {_SYSTEM_PROMPT}\n\n{user_text} [/INST]"

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        decoded = self._tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return decoded.strip()

    # ----- JSON extraction -----
    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # Try to find JSON block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not extract valid JSON from LLM output:\n{text[:500]}")

    # ----- public API -----
    def parse(
        self,
        problem: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskObject:
        """Parse a natural-language problem into a TaskObject."""
        user_text = problem
        if context:
            user_text += f"\n\nAdditional context:\n{json.dumps(context, indent=2)}"

        raw = self._generate(user_text)
        data = self._extract_json(raw)

        # Build task object defensively
        constraints = []
        for c in data.get("constraints", []):
            raw_ctype = c.get("ctype", "hard")
            try:
                ctype = ConstraintType(raw_ctype)
            except ValueError:
                ctype = ConstraintType.HARD
            constraints.append(Constraint(
                name=c.get("name", "unnamed"),
                description=c.get("description", ""),
                ctype=ctype,
                params=c.get("params", {}),
            ))

        obj_data = data.get("objective")
        objective = None
        if obj_data:
            raw_sense = obj_data.get("sense", "satisfy")
            try:
                sense = ObjectiveSense(raw_sense)
            except ValueError:
                sense = ObjectiveSense.SATISFY
            objective = Objective(
                description=obj_data.get("description", ""),
                sense=sense,
                metric=obj_data.get("metric"),
                params=obj_data.get("params", {}),
            )

        ambiguities = [
            AmbiguityMarker(
                field=a.get("field", "unknown"),
                reason=a.get("reason", ""),
                severity=float(a.get("severity", 0.5)),
            )
            for a in data.get("ambiguities", [])
        ]

        domain_str = data.get("domain", "unknown")
        try:
            domain = DomainGuess(domain_str)
        except ValueError:
            domain = DomainGuess.UNKNOWN

        return TaskObject(
            raw_input=problem,
            domain=domain,
            description=data.get("description", ""),
            entities=data.get("entities", []),
            constraints=constraints,
            objective=objective,
            ambiguities=ambiguities,
            structured_data=data.get("structured_data", {}),
            context=context or {},
        )

    def parse_offline(self, problem: str, parsed_json: Dict[str, Any]) -> TaskObject:
        """Build a TaskObject from pre-computed JSON (for testing without GPU)."""
        data = parsed_json
        constraints = [
            Constraint(
                name=c.get("name", "unnamed"),
                description=c.get("description", ""),
                ctype=ConstraintType(c.get("ctype", "hard")),
                params=c.get("params", {}),
            )
            for c in data.get("constraints", [])
        ]
        obj_data = data.get("objective")
        objective = None
        if obj_data:
            objective = Objective(
                description=obj_data.get("description", ""),
                sense=ObjectiveSense(obj_data.get("sense", "satisfy")),
                metric=obj_data.get("metric"),
                params=obj_data.get("params", {}),
            )
        ambiguities = [
            AmbiguityMarker(
                field=a.get("field", "unknown"),
                reason=a.get("reason", ""),
                severity=float(a.get("severity", 0.5)),
            )
            for a in data.get("ambiguities", [])
        ]
        domain_str = data.get("domain", "unknown")
        try:
            domain = DomainGuess(domain_str)
        except ValueError:
            domain = DomainGuess.UNKNOWN

        return TaskObject(
            raw_input=problem,
            domain=domain,
            description=data.get("description", ""),
            entities=data.get("entities", []),
            constraints=constraints,
            objective=objective,
            ambiguities=ambiguities,
            structured_data=data.get("structured_data", {}),
        )
