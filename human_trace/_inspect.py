"""Quick report on the human traces recorded under human_traces/."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from human_trace import load_traces, build_prior_pack


def human_bytes(n: int) -> str:
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def main() -> int:
    trace_dir = Path("human_traces")
    print(f"# Trace inventory: {trace_dir.resolve()}\n")
    files = sorted(trace_dir.glob("*.jsonl"))
    for f in files:
        print(f"  {human_bytes(f.stat().st_size):>10}  {f.name}")
    print()

    corpus = load_traces(trace_dir)
    print(f"Games : {len(corpus.by_game)}")
    print(f"Eps   : {corpus.n_episodes}")
    print(f"Steps : {corpus.n_steps}\n")

    for game_id, eps in corpus.by_game.items():
        print(f"## {game_id}  ({len(eps)} episodes)")
        for i, ep_bucket in enumerate(eps):
            ep = ep_bucket["episode"]
            steps = ep_bucket["steps"]
            header = f"  ep{i}  ({len(steps)} steps)"
            if ep is None:
                print(header + "  [no EpisodeRecord — session killed?]")
                continue
            print(
                f"{header}  final={ep.final_state}  levels={ep.levels_completed}"
                f"  type={ep.game_type_guess or '-'}  obj={ep.objective_guess or '-'}"
            )
            if ep.discovered_mechanics:
                for m in ep.discovered_mechanics:
                    print(f"        mech: {m}")
            if ep.discovered_mistakes:
                for m in ep.discovered_mistakes:
                    print(f"        miss: {m}")
        print()

        # Action + intent summaries across all steps for this game.
        all_steps = [s for b in eps for s in b["steps"]]
        action_counts = Counter(s.action for s in all_steps)
        intent_counts = Counter(s.intent for s in all_steps)
        hyp_counts = Counter(s.hypothesis for s in all_steps if s.hypothesis)
        state_counts = Counter(s.game_state_after for s in all_steps)

        print("  Actions : ", dict(action_counts))
        print("  Intents : ", dict(intent_counts))
        print("  States  : ", dict(state_counts))
        if hyp_counts:
            print("  Hypotheses (top 5):")
            for h, n in hyp_counts.most_common(5):
                print(f"    - x{n}  {h[:70]}")
        print()

        # Click heatmap (ACTION6)
        clicks = [s for s in all_steps if s.action == "ACTION6" and s.action_args]
        if clicks:
            effective = sum(1 for s in clicks if s.frame_before != s.frame_after)
            print(f"  Clicks  : {len(clicks)} total, {effective} changed the grid "
                  f"({effective/len(clicks):.0%})")
            pos_counts = Counter((s.action_args["y"], s.action_args["x"]) for s in clicks)
            print("    most-clicked (y,x):", pos_counts.most_common(5))
        print()

        # Prior pack preview
        pack = build_prior_pack(corpus, game_id)
        print("  Derived priors:")
        print(f"    action_stats    = {len(pack.action_stats)} actions")
        for a, st in pack.action_stats.items():
            print(f"      {a:8s}  tries={int(st['tries']):3d}  "
                  f"chg={st['change_rate']:.0%}  die={st['death_rate']:.0%}  "
                  f"win={st['win_rate']:.0%}")
        print(f"    goal_hints      = {len(pack.goal_hints)}")
        for h in pack.goal_hints:
            print(f"      -> {h['goal_id']}  won={h['won']}  top={h['top_actions']}")
        print(f"    failure_hints   = {len(pack.failure_hints)}")
        for h in pack.failure_hints:
            print(f"      -> {h['goal_id']}  reason={h['reason'][:60]}")
        print(f"    hypothesis_priors = {len(pack.hypothesis_priors)}")
        for key, conf in pack.hypothesis_priors[:8]:
            print(f"      [{conf:.2f}]  {key[:80]}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
