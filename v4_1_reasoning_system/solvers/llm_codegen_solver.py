"""
LLM code-generation solver — uses an instruction-tuned LLM to generate
code, solver formulations, critique, and repair prompts.

Use cases:
  - generate solver code from a task description
  - generate test cases
  - critique and repair failing code
  - produce solver skeletons for other modules

The LLM is treated as a reasoning operator, NOT as the source of truth.
All outputs must be verified externally.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseSolver, SolverResult


# ------------------------------------------------------------------
# Prompt templates
# ------------------------------------------------------------------
_CODEGEN_SYSTEM = """\
You are a code generator. Given a problem description, produce a Python
solution that solves it. Output ONLY the Python code, no explanation.
The code should:
- Be self-contained
- Print the result as JSON to stdout
- Handle edge cases
- Be correct and efficient"""

_SOLVER_SKELETON_SYSTEM = """\
You are a solver architect. Given a problem description, produce a Python
function using ortools CP-SAT that models and solves the problem.
Output ONLY the Python code. The function signature should be:
def solve(data: dict) -> dict:
    # returns solution dict or raises if infeasible"""

_REPAIR_SYSTEM = """\
You are a code repair agent. Given failing code and error output,
produce the corrected Python code. Output ONLY the fixed code.
Do not explain, just fix."""

_CRITIQUE_SYSTEM = """\
You are a code reviewer. Given code and its output, identify any bugs
or issues. Output a JSON object:
{"has_bugs": bool, "issues": [str], "suggested_fix": str or null}"""


class LLMCodegenSolver(BaseSolver):
    """
    Uses an LLM to generate / repair / critique code solutions.
    """

    name = "llm_codegen"

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        device: Optional[str] = None,
        load_in_4bit: bool = True,
        max_new_tokens: int = 2048,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        try:
            import torch as _torch
            self.device = device or ("cuda" if _torch.cuda.is_available() else "cpu")
        except ImportError:
            self.device = device or "cpu"
        self._load_in_4bit = load_in_4bit
        self._tokenizer = None
        self._model = None

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

    def _generate_text(self, system: str, user: str) -> str:
        import torch

        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = f"<s>[INST] {system}\n\n{user} [/INST]"

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        decoded = self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return decoded.strip()

    # ----- main interface -----
    def solve(
        self,
        task: Dict[str, Any],
        budget: str = "medium",
        strictness: str = "verified",
        tool_hint: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> SolverResult:
        t0 = time.time()
        logs: List[str] = []
        context = context or {}

        # Determine sub-mode
        if tool_hint in ("patch_failing_code", "fix_last_failure"):
            mode = "repair"
        elif tool_hint == "make_solver_skeleton":
            mode = "skeleton"
        elif tool_hint == "critique":
            mode = "critique"
        else:
            mode = "generate"

        logs.append(f"LLM codegen mode: {mode}")

        try:
            if mode == "generate":
                result = self._generate_code(task, context, logs)
            elif mode == "skeleton":
                result = self._generate_skeleton(task, context, logs)
            elif mode == "repair":
                result = self._repair_code(task, context, logs)
            elif mode == "critique":
                result = self._critique_code(task, context, logs)
            else:
                result = self._generate_code(task, context, logs)
        except Exception as e:
            result = SolverResult(
                success=False,
                violations=[f"LLM codegen error: {str(e)}"],
            )

        # Optionally execute and verify
        if result.success and strictness in ("verified", "strict") and mode in ("generate", "repair"):
            result = self._execute_and_check(result, task, logs)

        result.elapsed_seconds = time.time() - t0
        result.logs = logs
        return result

    # ------------------------------------------------------------------
    # Sub-modes
    # ------------------------------------------------------------------
    def _generate_code(self, task: Dict, context: Dict, logs: List[str]) -> SolverResult:
        desc = task.get("description", task.get("raw_input", ""))
        constraints = task.get("constraints", [])
        prompt = f"Problem: {desc}\n"
        if constraints:
            prompt += "Constraints:\n"
            for c in constraints:
                prompt += f"  - {c.get('name', '')}: {c.get('description', '')}\n"
        if task.get("objective"):
            prompt += f"Objective: {task['objective'].get('description', '')}\n"
        if context.get("structured_data"):
            prompt += f"Data: {json.dumps(context['structured_data'])}\n"

        code = self._generate_text(_CODEGEN_SYSTEM, prompt)
        code = self._extract_code(code)
        logs.append(f"Generated {len(code)} chars of code")

        return SolverResult(
            success=bool(code),
            solution={"code": code, "type": "generated"},
            metadata={"mode": "generate"},
        )

    def _generate_skeleton(self, task: Dict, context: Dict, logs: List[str]) -> SolverResult:
        desc = task.get("description", task.get("raw_input", ""))
        prompt = f"Problem: {desc}\n"
        if task.get("structured_data"):
            prompt += f"Data schema: {json.dumps(task['structured_data'])}\n"

        code = self._generate_text(_SOLVER_SKELETON_SYSTEM, prompt)
        code = self._extract_code(code)
        logs.append(f"Generated solver skeleton: {len(code)} chars")

        return SolverResult(
            success=bool(code),
            solution={"code": code, "type": "skeleton"},
            metadata={"mode": "skeleton"},
        )

    def _repair_code(self, task: Dict, context: Dict, logs: List[str]) -> SolverResult:
        failing_code = context.get("previous_solution", {}).get("code", "")
        error_output = context.get("verifier_feedback", {}).get("error", "")

        if not failing_code:
            return SolverResult(
                success=False,
                violations=["No failing code provided for repair"],
            )

        prompt = f"Failing code:\n```python\n{failing_code}\n```\n\nError:\n{error_output}"
        code = self._generate_text(_REPAIR_SYSTEM, prompt)
        code = self._extract_code(code)
        logs.append(f"Repaired code: {len(code)} chars")

        return SolverResult(
            success=bool(code),
            solution={"code": code, "type": "repaired"},
            metadata={"mode": "repair"},
        )

    def _critique_code(self, task: Dict, context: Dict, logs: List[str]) -> SolverResult:
        code = context.get("previous_solution", {}).get("code", "")
        output = context.get("verifier_feedback", {}).get("output", "")

        prompt = f"Code:\n```python\n{code}\n```\n\nOutput:\n{output}"
        critique_text = self._generate_text(_CRITIQUE_SYSTEM, prompt)
        logs.append("Critique generated")

        try:
            critique = json.loads(self._extract_json_str(critique_text))
        except (json.JSONDecodeError, ValueError):
            critique = {"has_bugs": None, "issues": [critique_text], "suggested_fix": None}

        return SolverResult(
            success=True,
            solution=critique,
            metadata={"mode": "critique"},
        )

    # ------------------------------------------------------------------
    # Code execution
    # ------------------------------------------------------------------
    def _execute_and_check(
        self, result: SolverResult, task: Dict, logs: List[str]
    ) -> SolverResult:
        """Execute generated code in a subprocess and check output."""
        code = result.solution.get("code", "") if isinstance(result.solution, dict) else ""
        if not code:
            return result

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(code)
                f.flush()
                tmp_path = f.name

            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if proc.returncode == 0:
                logs.append(f"Code executed successfully")
                result.metadata["stdout"] = proc.stdout[:2000]
                result.feasible = True
                # Try to parse JSON output as the solution value
                try:
                    parsed_output = json.loads(proc.stdout.strip())
                    result.solution["output"] = parsed_output
                except (json.JSONDecodeError, ValueError):
                    result.solution["output"] = proc.stdout.strip()
            else:
                logs.append(f"Code execution failed: {proc.stderr[:500]}")
                result.success = False
                result.violations.append(f"Runtime error: {proc.stderr[:200]}")
                result.metadata["stderr"] = proc.stderr[:2000]

            Path(tmp_path).unlink(missing_ok=True)

        except subprocess.TimeoutExpired:
            result.success = False
            result.violations.append("Code execution timed out (30s)")
            logs.append("Execution timeout")
        except Exception as e:
            result.violations.append(f"Execution error: {str(e)}")
            logs.append(f"Execution error: {e}")

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_code(text: str) -> str:
        """Extract Python code from LLM output."""
        # Try fenced code block first
        match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: treat entire output as code if it looks like Python
        lines = text.strip().split("\n")
        code_lines = [l for l in lines if not l.startswith("```")]
        return "\n".join(code_lines).strip()

    @staticmethod
    def _extract_json_str(text: str) -> str:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return match.group()
        return text
