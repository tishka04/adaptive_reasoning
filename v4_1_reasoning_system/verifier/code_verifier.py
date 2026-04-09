"""
Code verifier — checks generated code solutions for:
  - syntax validity
  - successful execution (no runtime errors)
  - test case pass/fail
  - output correctness
  - timeout compliance
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseVerifier, VerificationResult


class CodeVerifier(BaseVerifier):
    """
    Verifies code solutions by parsing, executing, and checking against
    test cases.
    """

    name = "code"

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout

    def verify(
        self,
        task: Dict[str, Any],
        solution: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        logs: List[str] = []
        violations: List[Dict[str, Any]] = []

        # Extract code from solution
        if isinstance(solution, dict):
            code = solution.get("code", "")
        elif isinstance(solution, str):
            code = solution
        else:
            return VerificationResult(
                valid=False,
                violations=[{"name": "no_code", "description": "No code found in solution", "severity": 1.0}],
            )

        if not code.strip():
            return VerificationResult(
                valid=False,
                violations=[{"name": "empty_code", "description": "Code is empty", "severity": 1.0}],
            )

        logs.append(f"Verifying code: {len(code)} chars")

        # Check 1: Syntax validity
        syntax_ok = self._check_syntax(code, violations, logs)

        if not syntax_ok:
            return VerificationResult(
                valid=False,
                feasible=False,
                violations=violations,
                tests_passed=0,
                tests_total=1,
                logs=logs,
            )

        # Check 2: Execution
        exec_result = self._execute_code(code, logs)

        if not exec_result["success"]:
            violations.append({
                "name": "runtime_error",
                "description": exec_result.get("error", "Unknown runtime error"),
                "severity": 1.0,
            })

        # Check 3: Test cases
        test_cases = task.get("structured_data", {}).get("test_cases", [])
        if not test_cases and context:
            test_cases = context.get("test_cases", [])

        tests_passed = 0
        tests_total = len(test_cases)

        for i, tc in enumerate(test_cases):
            passed = self._run_test_case(code, tc, logs)
            if passed:
                tests_passed += 1
            else:
                violations.append({
                    "name": f"test_fail_{i}",
                    "description": f"Test case {i} failed: input={tc.get('input', '?')}",
                    "severity": 0.8,
                })

        # If no test cases but execution succeeded, count execution as a test
        if tests_total == 0 and exec_result["success"]:
            tests_total = 1
            tests_passed = 1

        # Check 4: Output format (if expected)
        expected_format = task.get("structured_data", {}).get("expected_output_format")
        if expected_format and exec_result["success"]:
            format_ok = self._check_output_format(exec_result.get("stdout", ""), expected_format, logs)
            if not format_ok:
                violations.append({
                    "name": "output_format",
                    "description": f"Output does not match expected format: {expected_format}",
                    "severity": 0.5,
                })

        hard_violations = [v for v in violations if v.get("severity", 0) >= 1.0]
        valid = len(hard_violations) == 0

        return VerificationResult(
            valid=valid,
            feasible=valid,
            score=tests_passed / max(tests_total, 1),
            violations=violations,
            tests_passed=tests_passed,
            tests_total=tests_total,
            metadata={
                "stdout": exec_result.get("stdout", "")[:1000],
                "stderr": exec_result.get("stderr", "")[:1000],
                "returncode": exec_result.get("returncode", -1),
            },
            logs=logs,
        )

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------
    def _check_syntax(self, code: str, violations: List[Dict], logs: List[str]) -> bool:
        try:
            ast.parse(code)
            logs.append("  Syntax: OK")
            return True
        except SyntaxError as e:
            violations.append({
                "name": "syntax_error",
                "description": f"Syntax error at line {e.lineno}: {e.msg}",
                "severity": 1.0,
            })
            logs.append(f"  Syntax error: {e}")
            return False

    def _execute_code(self, code: str, logs: List[str]) -> Dict[str, Any]:
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                f.flush()
                tmp_path = f.name

            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            Path(tmp_path).unlink(missing_ok=True)

            success = proc.returncode == 0
            logs.append(f"  Execution: {'OK' if success else 'FAILED'} (rc={proc.returncode})")

            return {
                "success": success,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode,
                "error": proc.stderr[:500] if not success else "",
            }

        except subprocess.TimeoutExpired:
            logs.append(f"  Execution: TIMEOUT ({self.timeout}s)")
            return {"success": False, "error": f"Timeout after {self.timeout}s", "stdout": "", "stderr": "", "returncode": -1}
        except Exception as e:
            logs.append(f"  Execution error: {e}")
            return {"success": False, "error": str(e), "stdout": "", "stderr": "", "returncode": -1}

    def _run_test_case(self, code: str, test_case: Dict, logs: List[str]) -> bool:
        """Run code with test input and check against expected output."""
        tc_input = test_case.get("input", "")
        expected = test_case.get("expected_output", test_case.get("expected", ""))

        # Create wrapper that provides input
        wrapper = f"""
import sys
import io
sys.stdin = io.StringIO({json.dumps(str(tc_input))})
{code}
"""
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(wrapper)
                f.flush()
                tmp_path = f.name

            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            Path(tmp_path).unlink(missing_ok=True)

            if proc.returncode != 0:
                return False

            actual = proc.stdout.strip()
            expected_str = str(expected).strip()

            # Try numeric comparison
            try:
                return abs(float(actual) - float(expected_str)) < 1e-6
            except (ValueError, TypeError):
                pass

            # String comparison
            return actual == expected_str

        except Exception:
            return False

    @staticmethod
    def _check_output_format(output: str, expected_format: str, logs: List[str]) -> bool:
        """Check if output matches expected format."""
        if expected_format == "json":
            try:
                json.loads(output.strip())
                return True
            except (json.JSONDecodeError, ValueError):
                logs.append("  Output format: not valid JSON")
                return False
        elif expected_format == "number":
            try:
                float(output.strip())
                return True
            except ValueError:
                logs.append("  Output format: not a number")
                return False
        return True
