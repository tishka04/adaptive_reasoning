"""Launch and compare the v4_1 sampling-agent setups.

Usage:
    python compare_sampling_setups.py --game ar25 --time-budget 60

This wraps `run_with_human_priors.py` and compares:
  1. baseline: no human priors, no task program
  2. hypothesis_planner: no human priors, no task program, hypothesis planner
  3. latent_program: trajectory-scorer generated runtime program
  4. task_program_latent: static task program + latent runtime repair
  5. task_program_hypothesis: task program enabled, hypothesis planner
  6. task_program_only: task program enabled, no human priors
  7. human_and_task_program: human priors + task program
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
RUNNER = PROJECT_ROOT / "run_with_human_priors.py"
DEFAULT_TRACES_DIR = PROJECT_ROOT / "human_traces"
DEFAULT_TASK_PROGRAMS_DIR = PROJECT_ROOT / "task_programs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "diagnostics" / "sampling_setup_compare"
NO_TRACES_SENTINEL = PROJECT_ROOT / "__no_traces__"
WIDTH = 108


@dataclass
class SetupSpec:
    name: str
    label: str
    use_human_priors: bool
    use_task_program: bool
    sampler_stage: str
    planner_mode: str = "prior"


@dataclass
class RunResult:
    name: str
    label: str
    command: list[str]
    returncode: int
    wall_clock: float
    parsed: dict[str, Any]
    stdout_path: str
    stderr_path: str


SETUPS = [
    SetupSpec(
        name="baseline",
        label="No Priors, No Program",
        use_human_priors=False,
        use_task_program=False,
        sampler_stage="v0",
    ),
    SetupSpec(
        name="hypothesis_planner",
        label="Hypothesis Planner",
        use_human_priors=False,
        use_task_program=False,
        sampler_stage="v0",
        planner_mode="hypothesis",
    ),
    SetupSpec(
        name="latent_program",
        label="Latent Program From Scorer",
        use_human_priors=False,
        use_task_program=False,
        sampler_stage="v2",
    ),
    SetupSpec(
        name="task_program_latent",
        label="Task Program + Latent",
        use_human_priors=False,
        use_task_program=True,
        sampler_stage="v2",
    ),
    SetupSpec(
        name="task_program_hypothesis",
        label="Task Program + Hypothesis",
        use_human_priors=False,
        use_task_program=True,
        sampler_stage="v1",
        planner_mode="hypothesis",
    ),
    SetupSpec(
        name="task_program_only",
        label="Task Program Only",
        use_human_priors=False,
        use_task_program=True,
        sampler_stage="v1",
    ),
    SetupSpec(
        name="human_and_task_program",
        label="Human Priors + Task Program",
        use_human_priors=True,
        use_task_program=True,
        sampler_stage="v1",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the v4_1 sampling-agent setups and compare them."
    )
    parser.add_argument("--game", default="ar25")
    parser.add_argument("--time-budget", type=float, default=60.0)
    parser.add_argument("--mode", choices=["development", "competition"], default="development")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES_DIR)
    parser.add_argument("--task-programs", type=Path, default=DEFAULT_TASK_PROGRAMS_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for logs and JSON summary. Default: diagnostics/sampling_setup_compare/<timestamp>",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional explicit JSON summary path. Defaults inside --output-dir.",
    )
    parser.add_argument("--keep-memory", action="store_true", help="Persist cross-game memory files for each setup.")
    parser.add_argument("--stall-fraction", type=float, default=None)
    parser.add_argument("--max-wins", type=int, default=None)
    parser.add_argument("--level-up-bonus", type=float, default=None)
    parser.add_argument("--no-preserve-env", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def parse_runner_output(stdout_text: str) -> dict[str, Any]:
    def find(pattern: str, cast: Any = str, default: Any = None) -> Any:
        match = re.search(pattern, stdout_text, flags=re.MULTILINE)
        if not match:
            return default
        value = match.group(1).strip()
        if cast is bool:
            return value.lower() == "true"
        try:
            return cast(value)
        except Exception:
            return value

    raw_total_actions = find(r"total actions:\s+(\d+)$", cast=int, default=0)
    human_steps_replayed = find(
        r"\[game-memory\] human replay:\s+(\d+)\s+steps,",
        cast=int,
        default=0,
    )
    agent_only_actions = raw_total_actions
    if human_steps_replayed and raw_total_actions >= human_steps_replayed:
        agent_only_actions = raw_total_actions - human_steps_replayed

    return {
        "resolved_game_id": find(r"resolved game_id = (.+)$"),
        "won": find(r"won:\s+(.+)$", cast=bool, default=False),
        "max_level": find(r"max level:\s+(\d+)$", cast=int, default=0),
        "total_actions": raw_total_actions,
        "raw_total_actions": raw_total_actions,
        "human_steps_replayed": human_steps_replayed,
        "agent_only_actions": agent_only_actions,
        "assoc_wins": find(r"assoc wins:\s+(\d+)$", cast=int, default=0),
        "elapsed_seconds": find(r"elapsed:\s+([0-9.]+)s$", cast=float, default=0.0),
        "sampler_stage": find(r"sampler stage:\s+(\w+)$", cast=str, default="unknown"),
        "planner_mode": find(r"planner mode:\s+(\w+)$", cast=str, default="unknown"),
        "final_goal": find(r"final goal:\s+(.+)$", cast=str, default=""),
        "task_program_attached": "[task-program] attached" in stdout_text,
        "human_pack_loaded": "[human] pack:" in stdout_text,
        "runner_error": find(r"\[error\] (.+)$", cast=str, default=None),
    }


def build_command(args: argparse.Namespace, spec: SetupSpec, memory_path: Path) -> list[str]:
    command = [
        args.python_exe,
        str(RUNNER),
        "--game",
        args.game,
        "--time-budget",
        str(args.time_budget),
        "--mode",
        args.mode,
        "--memory-path",
        str(memory_path),
        "--sampler-stage",
        spec.sampler_stage,
        "--planner-mode",
        spec.planner_mode,
        "--traces",
        str(args.traces if spec.use_human_priors else NO_TRACES_SENTINEL),
        "--task-programs",
        str(args.task_programs),
    ]
    if not args.keep_memory:
        command.append("--no-save-memory")
    if not spec.use_task_program:
        command.append("--no-task-program")
    if args.stall_fraction is not None:
        command.extend(["--stall-fraction", str(args.stall_fraction)])
    if args.max_wins is not None:
        command.extend(["--max-wins", str(args.max_wins)])
    if args.level_up_bonus is not None:
        command.extend(["--level-up-bonus", str(args.level_up_bonus)])
    if args.no_preserve_env:
        command.append("--no-preserve-env")
    return command


def run_setup(
    args: argparse.Namespace,
    spec: SetupSpec,
    output_dir: Path,
) -> RunResult:
    memory_path = output_dir / f"{spec.name}_memory.pt"
    stdout_path = output_dir / f"{spec.name}.stdout.txt"
    stderr_path = output_dir / f"{spec.name}.stderr.txt"
    command = build_command(args, spec, memory_path)

    print()
    print("-" * WIDTH)
    print(f"  RUNNING {spec.label}  |  setup={spec.name}")
    print("-" * WIDTH)
    print("  " + " ".join(command))

    started = time.time()
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    wall_clock = time.time() - started

    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")

    parsed = parse_runner_output(completed.stdout)
    parsed["wall_clock_seconds"] = round(wall_clock, 2)
    parsed["returncode"] = completed.returncode

    print(
        f"  returncode={completed.returncode} "
        f"won={parsed['won']} "
        f"max_level={parsed['max_level']} "
        f"actions={parsed['agent_only_actions']} "
        f"assoc_wins={parsed['assoc_wins']} "
        f"elapsed={parsed['elapsed_seconds']:.1f}s "
        f"wall={wall_clock:.1f}s"
    )
    if parsed["raw_total_actions"] != parsed["agent_only_actions"]:
        print(
            f"  raw_actions={parsed['raw_total_actions']} "
            f"(includes {parsed['human_steps_replayed']} replayed human steps)"
        )
    print(f"  logs: stdout={stdout_path.name} stderr={stderr_path.name}")

    if completed.returncode != 0:
        print("  stderr preview:")
        for line in completed.stderr.strip().splitlines()[-10:]:
            print(f"    {line}")

    return RunResult(
        name=spec.name,
        label=spec.label,
        command=command,
        returncode=completed.returncode,
        wall_clock=round(wall_clock, 2),
        parsed=parsed,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )


def rank_key(result: RunResult) -> tuple[int, int, int, int, float]:
    parsed = result.parsed
    return (
        1 if parsed.get("won") else 0,
        int(parsed.get("max_level", 0)),
        int(parsed.get("assoc_wins", 0)),
        -int(parsed.get("agent_only_actions", parsed.get("total_actions", 0))),
        -float(parsed.get("elapsed_seconds", 0.0)),
    )


def print_comparison(results: list[RunResult]) -> None:
    baseline = next((item for item in results if item.name == "baseline"), None)
    name_width = max(30, max(len(item.label) for item in results))
    print()
    print("=" * WIDTH)
    print("  SAMPLING SETUP COMPARISON")
    print("=" * WIDTH)
    print(
        f"  {'Setup':<{name_width}} {'Won':>5} {'Level':>7} {'Assoc':>7} {'AgentActs':>10} {'RawActs':>9} {'Elapsed':>9} {'TaskProg':>9} {'Human':>7} {'Stage':>6} {'Planner':>10}"
    )
    print("  " + "-" * (WIDTH - 4))
    for result in results:
        parsed = result.parsed
        print(
            f"  {result.label:<{name_width}} "
            f"{str(parsed.get('won', False))[:5]:>5} "
            f"{int(parsed.get('max_level', 0)):>7} "
            f"{int(parsed.get('assoc_wins', 0)):>7} "
            f"{int(parsed.get('agent_only_actions', parsed.get('total_actions', 0))):>10} "
            f"{int(parsed.get('raw_total_actions', parsed.get('total_actions', 0))):>9} "
            f"{float(parsed.get('elapsed_seconds', 0.0)):>8.1f}s "
            f"{str(parsed.get('task_program_attached', False))[:9]:>9} "
            f"{str(parsed.get('human_pack_loaded', False))[:7]:>7} "
            f"{str(parsed.get('sampler_stage', 'unknown'))[:6]:>6} "
            f"{str(parsed.get('planner_mode', 'unknown'))[:10]:>10}"
        )

    if baseline is not None:
        print()
        print("  Delta vs baseline:")
        for result in results:
            if result is baseline:
                continue
            parsed = result.parsed
            base = baseline.parsed
            delta_level = int(parsed.get("max_level", 0)) - int(base.get("max_level", 0))
            delta_actions = int(parsed.get("agent_only_actions", parsed.get("total_actions", 0))) - int(
                base.get("agent_only_actions", base.get("total_actions", 0))
            )
            delta_assoc = int(parsed.get("assoc_wins", 0)) - int(base.get("assoc_wins", 0))
            print(
                f"  {result.label:<{name_width}} "
                f"level {delta_level:+d}  "
                f"assoc {delta_assoc:+d}  "
                f"agent_actions {delta_actions:+d}"
            )

    best = max(results, key=rank_key)
    print()
    print(
        f"  Best setup by solved/level/assoc/agent-actions: {best.label} "
        f"(won={best.parsed.get('won')}, level={best.parsed.get('max_level')}, "
        f"agent_actions={best.parsed.get('agent_only_actions', best.parsed.get('total_actions'))})"
    )


def main() -> int:
    args = parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (DEFAULT_OUTPUT_DIR / f"{args.game}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_json or (output_dir / "summary.json")

    print("=" * WIDTH)
    print(
        f"  v4_1 setup comparison  |  game={args.game!r}  |  budget={args.time_budget}s  |  runs={len(SETUPS)}"
    )
    print("=" * WIDTH)
    print(f"  output_dir={output_dir}")

    results: list[RunResult] = []
    started = time.time()
    for spec in SETUPS:
        result = run_setup(args, spec, output_dir)
        results.append(result)
        if args.fail_fast and result.returncode != 0:
            break

    print_comparison(results)

    payload = {
        "game": args.game,
        "time_budget": args.time_budget,
        "mode": args.mode,
        "python_exe": args.python_exe,
        "traces": str(args.traces),
        "task_programs": str(args.task_programs),
        "output_dir": str(output_dir),
        "wall_clock_seconds": round(time.time() - started, 2),
        "results": [asdict(item) for item in results],
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print()
    print(f"  Summary JSON saved to {output_json}")
    return 0 if all(item.returncode == 0 for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
