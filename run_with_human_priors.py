"""Run the v4_1 AdaptiveReasoning agent seeded from human traces.

Usage:
    python run_with_human_priors.py [--game ar25] [--time-budget 60]
                                    [--traces human_traces]
                                    [--memory-path cross_game_memory.pt]
                                    [--no-save-memory]

What this does
--------------
1. Loads JSONL human traces from `--traces` for the requested game_id prefix.
2. Builds a `HumanPriorPack` (action stats + goal hints + failure patterns
   + hypothesis priors).
3. Loads the on-disk `CrossGameMemory` (or creates a new one), then merges
   the cross-game priors from the pack into it.
4. Creates the `AdaptiveReasoning` agent for the game.
5. Replays the human trace through the agent's `GameMemory` (so it boots
   up with pre-learned action profiles and hypotheses).
6. Runs `agent.main()` under the given time budget.
7. Persists the updated `CrossGameMemory` back to disk, so the priors stay
   in effect for subsequent runs.

This does NOT modify any v4_1 code or the ARC-AGI-3-Agents launcher — it
wraps them.
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
import time
import types
from pathlib import Path

# Ensure stdout/stderr can emit non-ASCII characters under Windows PowerShell
# pipes (which default to cp1252 and crash on em-dash / arrow glyphs).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Route INFO-level log lines from the agent to stderr so diagnostic
# messages (e.g. LEVEL-UP, GS.WIN) are visible without extra flags.
# Silence known-noisy libraries below INFO so we don't drown in HTTP logs.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    force=True,  # override any config set by imported packages
)
logging.getLogger().setLevel(logging.INFO)
for _noisy in ("urllib3", "httpx", "transformers", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_DIR = PROJECT_ROOT / "environment_files"
AGENTS_DIR = PROJECT_ROOT / "ARC-AGI-3-Agents"
BUNDLED_SITE_PACKAGES = AGENTS_DIR / ".venv" / "Lib" / "site-packages"

# The ARC-AGI-3-Agents package expects to be discoverable on sys.path and
# uses env vars to pick transport / data directories.
os.environ.setdefault("OPERATION_MODE", "offline")
os.environ.setdefault("ENVIRONMENTS_DIR", str(ENV_DIR))
os.environ.setdefault("ARC_API_KEY", "test")
os.environ.setdefault("RECORDINGS_DIR", str(PROJECT_ROOT / "recordings"))
os.environ.setdefault("TQDM_DISABLE", "1")

# Silence chatty loggers — keep our own prints only.
logging.disable(logging.CRITICAL)

for _mod in ("numpy", "torch"):
    try:
        __import__(_mod)
    except Exception:
        pass

if importlib.util.find_spec("pydantic") is None:
    import adaptive_reasoning_compat.pydantic as _compat_pydantic

    sys.modules.setdefault("pydantic", _compat_pydantic)

if importlib.util.find_spec("dotenv") is None:
    _dotenv = types.ModuleType("dotenv")

    def _load_dotenv(*_args, **_kwargs):
        return False

    _dotenv.load_dotenv = _load_dotenv
    sys.modules.setdefault("dotenv", _dotenv)

for p in (BUNDLED_SITE_PACKAGES, PROJECT_ROOT, AGENTS_DIR):
    if not p.exists():
        continue
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# --- Imports that depend on sys.path being set up ---------------------
try:
    from arc_agi import Arcade, OperationMode                             # noqa: E402
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Could not import the ARC environment package expected by this repo. "
        "The PyPI package named 'arc-agi' is not the same API. "
        f"Tried bundled site-packages at {BUNDLED_SITE_PACKAGES}."
    ) from exc
from agents import AVAILABLE_AGENTS                                       # noqa: E402
from human_trace import (                                                 # noqa: E402
    build_prior_pack,
    HumanTraceMemory,
    load_traces,
    seed_cross_game_memory,
    seed_game_memory,
)
from v4_1_reasoning_system.arc_agi.associative_memory import CrossGameMemory  # noqa: E402
from v4_1_reasoning_system.arc_agi.reasoning_loop import ABLATION_STAGES  # noqa: E402
from v4_1_reasoning_system.arc_agi.trajectory_sampler import TrajectorySampler  # noqa: E402


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _resolve_full_game_id(arc: Arcade, game_id: str) -> str:
    """Promote a 4-char game_id (`ar25`) to its full versioned form
    (`ar25-e3c63847`) by consulting the local env registry."""
    if "-" in game_id:
        return game_id
    for env in arc.get_environments():
        if env.game_id.startswith(game_id + "-") or env.game_id == game_id:
            return env.game_id
    return game_id


def _load_human_pack(traces_dir: Path, game_id_prefix: str, full_game_id: str):
    """Try the full game_id first; fall back to the 4-char prefix."""
    corpus = load_traces(traces_dir)
    if not corpus.by_game:
        print(f"  [human] no trace files under {traces_dir}")
        return None

    # Prefer an exact match on the resolved full game_id.
    if full_game_id in corpus.by_game:
        return build_prior_pack(corpus, full_game_id)

    # Fallback: match by prefix (catches "ar25" vs "ar25-e3c63847").
    for gid in corpus.by_game:
        if gid.startswith(game_id_prefix):
            return build_prior_pack(corpus, gid)

    print(f"  [human] no traces matching {game_id_prefix!r}; found {list(corpus.by_game)}")
    return None


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="ar25")
    parser.add_argument("--time-budget", type=float, default=60)
    parser.add_argument("--traces", type=Path, default=PROJECT_ROOT / "human_traces")
    parser.add_argument(
        "--memory-path",
        type=Path,
        default=PROJECT_ROOT / "cross_game_memory.pt",
        help="On-disk path for CrossGameMemory (shared with test_full_agent.py).",
    )
    parser.add_argument(
        "--mode",
        choices=["development", "competition"],
        default="development",
    )
    parser.add_argument(
        "--no-save-memory",
        action="store_true",
        help="Don't write back the updated cross-game memory (dry run).",
    )
    parser.add_argument(
        "--task-programs",
        type=Path,
        default=PROJECT_ROOT / "task_programs",
        help=(
            "Directory holding compiled TaskProgram JSON files "
            "(produced by `python -m human_trace.compile_trace`). "
            "If a program matching --game exists, it overrides template "
            "goal decomposition at runtime."
        ),
    )
    parser.add_argument(
        "--stall-fraction",
        type=float,
        default=None,
        help=(
            "Fraction of remaining time that counts as a stall per iteration. "
            "Default: keep class value (0.25). Raise to give longer per-iter budgets."
        ),
    )
    parser.add_argument(
        "--max-wins",
        type=int,
        default=None,
        help=(
            "Outer-loop cap on level-ups before exiting. Default: keep class value (999). "
            "Set low to exit early after a few wins; raise to keep pushing for deeper levels."
        ),
    )
    parser.add_argument(
        "--level-up-bonus",
        type=float,
        default=None,
        help=(
            "Seconds of stall-free budget granted after each in-iter level-up. "
            "Default: keep class value (30.0). Raise to give the agent more time on new levels."
        ),
    )
    parser.add_argument(
        "--no-preserve-env",
        action="store_true",
        help=(
            "Disable env-persistence across iterations. When off, every outer-loop "
            "iteration rebuilds the env + RESETs, even if the previous iter made progress."
        ),
    )
    parser.add_argument(
        "--no-task-program",
        action="store_true",
        help="Disable TaskProgram loading even if one exists on disk.",
    )
    parser.add_argument(
        "--sampler-stage",
        choices=["v0", "v1", "v2", "v3"],
        default="v0",
        help="Trajectory sampler stage to run. V0=minimal core, V1=task-program/human priors.",
    )
    parser.add_argument(
        "--planner-mode",
        choices=["prior", "hypothesis"],
        default="prior",
        help="Trajectory planner family to run.",
    )
    parser.add_argument(
        "--reasoning-mode",
        choices=["full", "symbolic_core", "ascetic"],
        default="full",
        help="Reasoning preset. symbolic_core/ascetic default to the robust short-horizon core.",
    )
    parser.add_argument(
        "--ablation-stage",
        choices=list(ABLATION_STAGES),
        default=None,
        help=(
            "Explicit ablation stage: symbolic_only -> game_memory -> "
            "goal_pursuit -> trajectory_memory -> short_horizon -> "
            "visual_cortex -> jepa_ebm."
        ),
    )
    args = parser.parse_args()

    print("=" * 90)
    print(f"  AdaptiveReasoning agent, human-primed   |   game={args.game!r}")
    print(f"  traces={args.traces}   memory={args.memory_path}")
    print(f"  time_budget={args.time_budget}s   mode={args.mode}")
    print(f"  sampler stage: {args.sampler_stage}")
    print(f"  planner mode:  {args.planner_mode}")
    print(f"  reasoning:     {args.reasoning_mode} / {args.ablation_stage or 'preset-default'}")
    print("=" * 90)

    # 1. Arcade + env
    arc = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ENV_DIR),
    )
    full_game_id = _resolve_full_game_id(arc, args.game)
    env = arc.make(full_game_id)
    if env is None:
        print(f"  [error] could not make env for {full_game_id}")
        return 2
    print(f"  resolved game_id = {full_game_id}")

    # 2. Human prior pack
    pack = _load_human_pack(args.traces, args.game, full_game_id)
    human_trace_memory = None
    if pack is None:
        print("  [warn] continuing without human priors (no traces found).")
    else:
        print(
            f"  [human] pack: {len(pack.steps)} steps, "
            f"{len(pack.episodes)} eps, "
            f"{len(pack.goal_hints)} goal hints, "
            f"{len(pack.failure_hints)} failure hints, "
            f"{len(pack.hypothesis_priors)} hypotheses"
        )
        human_trace_memory = HumanTraceMemory.from_prior_pack(pack)
        print(
            f"  [human] runtime trace memory: {len(human_trace_memory.traces)} reusable traces",
        )

    # 3. Cross-game memory (load + seed + hold)
    cross_game = CrossGameMemory.load(str(args.memory_path))
    cross_game.mode = args.mode
    if cross_game.games_played:
        print(
            f"  [cross-game] resumed: {cross_game.games_played} games, "
            f"{cross_game.games_won} wins, trust={cross_game.prior_trust:.2f}"
        )
    else:
        print("  [cross-game] fresh memory")

    if pack is not None:
        seed_stats = seed_cross_game_memory(pack, cross_game)
        print(
            f"  [cross-game] human seeding: +{seed_stats['goal_hints']} hints, "
            f"+{seed_stats['failure_hints']} failures"
        )

    # 4. Agent
    agent_cls = AVAILABLE_AGENTS["adaptivereasoning"]
    original_time = agent_cls.TIME_BUDGET
    original_reasoning_mode = getattr(agent_cls, "REASONING_MODE", "full")
    original_ablation_stage = getattr(agent_cls, "ABLATION_STAGE", None)
    agent_cls.TIME_BUDGET = float(args.time_budget)
    if hasattr(agent_cls, "REASONING_MODE"):
        agent_cls.REASONING_MODE = str(args.reasoning_mode)
    if hasattr(agent_cls, "ABLATION_STAGE"):
        agent_cls.ABLATION_STAGE = args.ablation_stage

    # ── Optional CLI overrides for the strategic iteration budget ─
    _orig_stall = getattr(agent_cls, "STALL_FRACTION", None)
    _orig_max_wins = getattr(agent_cls, "MAX_WINS", None)
    _orig_bonus = getattr(agent_cls, "LEVEL_UP_BONUS_SECONDS", None)
    _orig_preserve = getattr(agent_cls, "PRESERVE_ENV_ACROSS_ITERS", None)
    if args.stall_fraction is not None:
        agent_cls.STALL_FRACTION = float(args.stall_fraction)
    if args.max_wins is not None:
        agent_cls.MAX_WINS = int(args.max_wins)
    if args.level_up_bonus is not None:
        agent_cls.LEVEL_UP_BONUS_SECONDS = float(args.level_up_bonus)
    if args.no_preserve_env:
        agent_cls.PRESERVE_ENV_ACROSS_ITERS = False
    print(
        f"  [tuning] stall_fraction={agent_cls.STALL_FRACTION} "
        f"max_wins={agent_cls.MAX_WINS} "
        f"level_up_bonus={agent_cls.LEVEL_UP_BONUS_SECONDS}s "
        f"preserve_env={agent_cls.PRESERVE_ENV_ACROSS_ITERS}"
    )

    agent = agent_cls(
        card_id=None,
        game_id=full_game_id,
        agent_name="adaptivereasoning",
        ROOT_URL="",
        record=False,
        arc_env=env,
        cross_game=cross_game,
        arcade=arc,
    )
    print(
        f"  [ablation] stage={agent.reasoning.config.ablation_stage} "
        f"features={agent.reasoning.config.enabled_features()}"
    )
    agent.reasoning.config.sampler_stage = str(args.sampler_stage).lower()
    agent.reasoning.config.planner_mode = str(args.planner_mode).lower()
    agent.reasoning.trajectory_sampler = TrajectorySampler(
        stage=agent.reasoning.config.sampler_stage,
        planner_mode=agent.reasoning.config.planner_mode,
        enable_continuation=agent.reasoning.config.enable_trajectory_continuation,
    )
    # Tell the agent where to PERSIST live-revised TaskPrograms
    # produced by the online LLM revision mechanism (saved as
    # `<game>.lvl<N>.live<k>.json`). The legacy offline auto-revise
    # mechanism (pre-compiled `<game>.lvl<N>.json` files swapped in on
    # plateau) has been removed; the agent now only uses the online
    # path.
    if not args.no_task_program:
        agent._task_program_dir = args.task_programs
    # Path to the human-trace corpus, used by the LIVE LLM revision
    # mechanism to rebuild a filtered HumanPriorPack on demand.
    if hasattr(args, "traces"):
        agent._human_traces_dir = args.traces
    if human_trace_memory is not None:
        agent.reasoning.set_human_trace_memory(human_trace_memory)

    # 5. Seed the agent's per-game GameMemory from the same pack.
    #    The agent creates `self.memory = GameMemory()` in __init__; we
    #    warm it up here, before `main()`.
    if pack is not None:
        gm_stats = seed_game_memory(pack, agent.memory)
        print(
            f"  [game-memory] human replay: "
            f"{gm_stats['steps_replayed']} steps, "
            f"{gm_stats['clicks_replayed']} clicks, "
            f"{gm_stats['hypotheses_added']} hypotheses, "
            f"human_hit_level={gm_stats.get('seeded_max_level', 0)} "
            f"seeded_actions={gm_stats.get('seeded_total_actions', 0)} "
            f"(online counters reset → agent actions/max_level now measure agent only)"
        )
        # Log a tiny profile snapshot so we can see what the agent starts with.
        prof = agent.memory.action_profiles
        for name in sorted(prof):
            p = prof[name]
            print(
                f"     {name}: tries={p.times_tried:3d} "
                f"chg={p.change_rate:.0%} die={p.times_caused_game_over} "
                f"mv={p.times_moved_player}"
            )
        print(f"     hypotheses: {len(agent.memory.hypotheses)} keys")

    # 5.5 Load compiled TaskProgram (if any) and attach to GoalDecomposer.
    #     Accepts either a full game id prefix match or bare `--game`.
    if not args.no_task_program:
        try:
            from human_trace.task_program import (
                try_load_task_program,
            )
            program = (
                try_load_task_program(args.task_programs, full_game_id)
                or try_load_task_program(args.task_programs, args.game)
            )
        except Exception as e:  # pragma: no cover
            program = None
            print(f"  [task-program] load error: {e}")
        if program is not None:
            setattr(program, "_target_level", 1)
            setattr(program, "_attachment_kind", "generic")
            agent.reasoning.decomposer.set_task_program(program)
            if hasattr(agent, "_attached_task_program_target_level"):
                agent._attached_task_program_target_level = 1
            if hasattr(agent, "_attached_task_program_kind"):
                agent._attached_task_program_kind = "generic"
            print(
                f"  [task-program] attached {program.game_id} "
                f"(family={program.goal_family}, "
                f"{len(program.subgoal_tests)} subgoal tests, "
                f"conf={program.confidence:.2f})"
            )
        else:
            print(
                f"  [task-program] none found under {args.task_programs} "
                f"— agent will use template heuristics"
            )
    else:
        print("  [task-program] disabled via --no-task-program")

    # 6. Run
    print(f"\n  running agent.main() with {args.time_budget}s budget ...\n")
    t0 = time.time()
    try:
        agent.main()
    except Exception as e:  # pragma: no cover
        print(f"  [error] agent.main() raised: {e!r}")
    elapsed = time.time() - t0

    # 7. Report + persist
    summary = agent.memory.summary()
    loop_stats = agent.reasoning.get_stats()
    assoc_wins = loop_stats.get("assoc_memory", {}).get("total_wins", 0)
    won = assoc_wins > 0 or summary["max_level"] > 0

    print()
    print("-" * 90)
    print("  RESULT")
    print("-" * 90)
    print(f"  won:           {won}")
    print(f"  max level:     {summary['max_level']}")
    print(f"  total actions: {summary['total_actions']}")
    print(f"  assoc wins:    {assoc_wins}")
    print(f"  elapsed:       {elapsed:.1f}s")
    print(f"  final goal:    {loop_stats.get('overarching_goal')}")

    agent_cls.TIME_BUDGET = original_time
    if hasattr(agent_cls, "REASONING_MODE"):
        agent_cls.REASONING_MODE = original_reasoning_mode
    if hasattr(agent_cls, "ABLATION_STAGE"):
        agent_cls.ABLATION_STAGE = original_ablation_stage
    if _orig_stall is not None:
        agent_cls.STALL_FRACTION = _orig_stall
    if _orig_max_wins is not None:
        agent_cls.MAX_WINS = _orig_max_wins
    if _orig_bonus is not None:
        agent_cls.LEVEL_UP_BONUS_SECONDS = _orig_bonus
    if _orig_preserve is not None:
        agent_cls.PRESERVE_ENV_ACROSS_ITERS = _orig_preserve

    if not args.no_save_memory:
        cross_game.save(str(args.memory_path))
        print(f"\n  cross-game memory saved to {args.memory_path}")
    else:
        print("\n  --no-save-memory set; cross-game memory NOT written back")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
