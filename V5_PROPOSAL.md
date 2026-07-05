# V5 — Proposal: Blending V3, V4 and V4_1

> Status: **proposal / design doc**. Nothing has been implemented yet.
> Purpose: distil the strongest, validated ideas from `v3/`, `v4/` and
> `v4_1_reasoning_system/` into a single coherent agent that is smaller
> than V4, stronger than V3, and free of V4_1's LLM/latency dependencies.

---

## 1. Lessons from the three systems

### 1.1 V3 — "operator-centric, mechanic-inducing" (symbolic)

What actually worked in runs:
- `OperatorInducer` + `RuleEngine` — lean, auditable symbolic induction
- Six `SpecialistMinds` with competing inductive biases
- `ProgressTracker` (LP/SP/TP) + sterile branch killer
- `OperatorSearch` (beam) + `SolutionShortener` + two-tier `MacroCompiler`
- `BeliefDebugger` — demotes false abstractions over time
- Compositional evidence gating — exploit only when warranted

Weak points:
- No perception priors (cold-starts every game)
- Single worldview — no way to reframe when stuck
- No compact per-game memory beyond rules/operators

Headline result: **solved level 1 of cn04**, compositional evidence in
every game, 91.5% pred accuracy, 79.2% ctrl success.

### 1.2 V4 — "chambered ecology" (baroque)

Genuinely novel ideas worth keeping:
- **Ontology competition** — competing worldviews (token_world,
  navigator, click, transform, …) with evidence-weighted ranking
- **Rituals** — compiled `(prefix, terminal_signature)` templates;
  strictly a refinement of macros, but meaningfully compact
- **Dissent controller** — separate, skeptical controller with authority
  to interrupt "coherent but sterile" behaviour
- **Surprise field** — 5 parallel surprise channels (pixel / object /
  causal / topology / semantic)
- **Bissociation engine** — crossing two frames to propose hybrid plans
- Compact per-game digest (just landed in `v4/memory/win_digest.py`)

What clearly hurt V4:
- `project_market` inserted a whole extra layer between minds and
  operators with little payoff — minds have to route through projects
- `LearningSuite` has 14 sub-models (bandits, value models, reliability
  predictors, embeddings…) that are poorly grounded and add variance
- Too many budgets and thresholds to tune in parallel
- Observed in practice: `tr87-cd924810` burns 34K actions with
  `op_kinds=[global_transform, noop]` and zero level transitions. The
  agent "thinks" a lot but never touches meaningful physics.

### 1.3 V4_1 — "LLM + deep learning"

What really works as a *prior* (not as a controller):
- **Visual Cortex** (CNN frame predictor, ~450K params) — produces
  per-action change rates, direction fields, danger scores. Genuinely
  fast and genuinely useful when injected into the associative memory.
- Structured `analyze_action_effects()` + `compute_action_similarity()`
- **CrossGameMemory** that actually transfers NN weights + action priors

What should NOT come into V5:
- LLM goal decomposition — heavy, brittle, requires network / API
- EBM router as main selector — training instability, underperforms
  symbolic arbiter on interactive games
- JEPA/EBM online training — the default is already `False` in V4_1
  because it added noise

---

## 2. V5 — Design principles

1. **Symbolic spine, learned priors.** Control decisions (operators,
   rules, minds, progress) are symbolic. Learned models are allowed to
   *inject priors*, never to make the final call.
2. **Minimal necessary layers.** Every layer must have a measurable
   contribution in a held-out game; otherwise it is cut.
3. **Compact memory, per-game digests as the primary unit.** The
   "what was useful / useless" digest we just added becomes the
   canonical cross-game record. Operator templates are derived from
   digests, not stored independently.
4. **Three budgets, one scheduler.** One clock (time budget), one
   action counter, and one progress tracker (LP/SP/TP). Everything else
   subscribes to these.
5. **Dissent is first-class.** A skeptical controller runs in parallel
   with the main loop and has explicit authority to interrupt the mind
   arbiter once per ~10 actions.
6. **Multiple ontologies, but few.** Not an open hypothesis space.
   Exactly four seeded ontologies (navigator / click / token / physics)
   compete, with VC-derived evidence. Extra kinds must be promoted from
   hard evidence, not speculated.
7. **No LLM, no online deep training.** The VC checkpoint is loaded
   once and used as a prior generator. No gradient steps during play.

---

## 3. V5 architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  V5 ADAPTIVE REASONING AGENT                                      │
│                                                                   │
│  1. Perception Layer                                              │
│     - ObjectExtractor            (from V3)                        │
│     - FrameDiff                  (from V3)                        │
│     - VisualCortex (CNN priors)  (from V4_1, frozen checkpoint)   │
│     - SurpriseField (5-channel)  (from V4)                        │
│                                                                   │
│  2. Ontology Layer (slim)                                         │
│     - OntologyCompetition        (from V4, ≤4 kinds)              │
│     - AffordanceReframing        (from V4)                        │
│                                                                   │
│  3. Mechanics Layer                                               │
│     - ActionProfiler (VC-primed) (V3 + V4_1 priors)               │
│     - OperatorInducer            (from V3)                        │
│     - RuleEngine                 (from V3)                        │
│     - ExperimentDesigner         (from V3)                        │
│                                                                   │
│  4. Memory Layer                                                  │
│     - GameMemory                 (V3 base)                        │
│     - EpisodicGraph              (from V3)                        │
│     - BeliefDebugger             (from V3)                        │
│     - RitualStore                (from V4, compact templates)     │
│     - CrossGameMemoryV5          (digests as primary unit)        │
│                                                                   │
│  5. Control Layer                                                 │
│     - GoalSkeleton               (new, 2-level symbolic goal)     │
│     - ProgressTracker LP/SP/TP   (from V3)                        │
│     - SpecialistMinds ×6         (from V3)                        │
│     - Arbiter                    (from V3, Bayesian)              │
│     - OperatorSearch             (from V3, beam)                  │
│     - ReactiveController         (from V3)                        │
│     - DissentController          (from V4, interrupts arbiter)    │
│                                                                   │
│  6. Execution / Compression Layer                                 │
│     - Executor                   (from V3)                        │
│     - SolutionShortener          (from V3)                        │
│     - MacroCompiler (two-tier)   (from V3)                        │
│     - Ritualizer                 (from V4)                        │
└──────────────────────────────────────────────────────────────────┘
```

### 3.1 What we explicitly drop

From V4:
- `project_market` (ProjectMarket, project_generators)
- `bissociation_engine` + `bridge_memory` (interesting, but unvalidated)
- Entire `learning/` suite (14 sub-models) — kept only if a specific
  sub-model shows measurable lift on held-out games (see §6)

From V4_1:
- `goal_decomposer` (LLM calls)
- `strategy_generator` / LLM pathway
- `energy_scorer` (EBM)
- `reasoning_loop` phases (EXPLORE/DECOMPOSE/STRATEGIZE/EXECUTE/UPDATE)

From V3:
- Nothing. V3 is the spine.

### 3.2 What we port and refine

| Capability | Source | Notes |
|---|---|---|
| ObjectExtractor, FrameDiff | V3 | unchanged |
| VisualCortex (frozen) | V4_1 | load checkpoint, disable training, expose `analyze_action_effects`, `compute_action_similarity` |
| SurpriseField | V4 | unchanged, consumed by ProgressTracker as SP signal |
| OntologyCompetition | V4 | cap at 4 seeded kinds; only promote via hard evidence |
| OperatorInducer, RuleEngine | V3 | unchanged |
| ProgressTracker | V3 | unchanged |
| SpecialistMinds (6) | V3 | unchanged |
| Arbiter | V3 | add ontology-aware mind priors |
| OperatorSearch | V3 | unchanged |
| DissentController | V4 | remove its project_market hooks |
| Ritualizer | V4 | retained, stored in RitualStore |
| MacroCompiler | V3 | unchanged |
| Digest + CrossGameMemoryV5 | new (based on V4 win_digest) | primary unit |

### 3.3 New component: GoalSkeleton

A tiny, symbolic 2-level goal tree computed from ontology + TP:

```
GoalSkeleton
├── goal            : "complete level N"
└── active_subgoal  : chosen from ontology kind:
                       navigator → "reach remaining target"
                       click     → "exhaust interactive objects"
                       token     → "apply transform until terminal state"
                       physics   → "push X into slot"
```

Subgoals are *hints* for the minds, not constraints. Subgoal satisfaction
advances TP. This replaces V4_1's LLM decomposer with 20 lines of code.

---

## 4. Decision loop (simplified)

```
on each observation o_t:
  perceive(o_t)                             # objects, diff, surprise, VC priors
  ontology_competition.update(o_t)          # 4 kinds re-ranked
  profiler.update(o_t)                      # primed by VC priors
  inducer.update(), rule_engine.update()
  progress.on_action(o_t)                   # LP/SP/TP + branch kill flag

  dissent_report = dissent_controller.update(o_t)
  if dissent_controller.should_interrupt():
      return dissent_controller.interrupt_and_redirect()

  goal_skeleton.refresh(top_ontology, TP)

  # forced cadences for diversity
  if action_counter % 10 == 0: return search_mode(goal_skeleton)
  if progress.should_kill_branch(): return experiment_mode()

  # stuck patterns force specific minds
  if lp > 0.3 and sp < 0.15: return force_mind("sequence")
  if tp > 0.1:              return force_mind("closure")

  # normal mode select: experiment / search / exploit
  mode = choose_mode(knowledge, confidence, coverage)
  return mode.act(goal_skeleton)

on level_complete:
  compressed = solution_shortener(level_trace)
  macro_compiler.compile(compressed)
  ritualizer.compile(compressed, terminal_signature)

on game_end:
  digest = build_digest(memory, won, stop_reason)
  merge_into_cross_game(cross_game, digest)
```

---

## 5. Cross-game memory (V5)

`CrossGameMemoryV5` is intentionally narrow:

```
game_digests     : dict[game_id -> GameWinDigest]   # primary unit
operator_priors  : derived-at-load from digests
                   (kinds that appeared as useful_operator_kinds
                    across ≥2 games gain +prior)
ontology_priors  : derived-at-load from digests
                   (top ontology in won games weighted stronger)
ritual_signatures: dict[ritual_id -> compact dict]  # retained from V4
trust            : single scalar, grows only with WINs
```

Everything that can be derived from digests is *derived on load* rather
than stored separately. The pickle file stays under 100 KB even after
hundreds of games.

### 5.1 Digest-driven transfer

On `seed_game(memory)`:
1. Pick top-3 digests by similarity (same ontology as currently top) or
   by raw games-won count.
2. Pre-seed operator kinds that appeared as `useful_operator_kinds` with
   a small confidence prior (`0.30 × trust`).
3. Downweight primitives listed in `useless_primitives` across ≥2 games.
4. If a ritual_id has `success_rate ≥ 0.5` in ≥1 digest, inject it at
   confidence `0.30 × trust`.

No operator template soup, no mind reliability soup — the digest is the
source of truth.

---

## 6. Keep-or-cut ablation plan

Before declaring V5 done, every ported V4 layer must earn its seat:

| Component | Keep if | Cut if |
|---|---|---|
| SurpriseField | adds ≥ +5% SP recall vs. diff-only | otherwise |
| OntologyCompetition | Mind selection differs meaningfully across ontologies | dominated by one ontology always |
| DissentController | Interrupt→recovery solves ≥1 game that V3 alone doesn't | never escapes |
| Ritualizer | Ritual-seeded game wins ≥1 level faster than non-seeded | slower or equal |
| GoalSkeleton | Mind accuracy goes up when subgoal is set | no effect |

Run the held-out set (e.g. 10 games) with each component on/off. Commit
the minimal subset that monotonically improves over V3 baseline.

---

## 7. Directory layout

```
v5/
├── __init__.py
├── schemas.py                    # flat, small; reuse V3 + 3 V4 dataclasses
├── adaptive_reasoning_agent_v5.py
├── perception/
│   ├── object_extractor.py       # from v3
│   ├── frame_diff.py             # from v3
│   ├── surprise_field.py         # from v4
│   └── visual_cortex_adapter.py  # wraps v4_1's VisualCortex, freeze-only
├── ontology/
│   ├── ontology_competition.py   # from v4 (slim)
│   └── affordance_reframing.py   # from v4
├── mechanics/
│   ├── action_profiler.py        # from v3, +VC priors
│   ├── operator_inducer.py       # from v3
│   ├── rule_engine.py            # from v3
│   └── experiment_designer.py    # from v3
├── memory/
│   ├── game_memory.py            # v3 base
│   ├── episodic_graph.py         # from v3
│   ├── belief_debugger.py        # from v3
│   ├── ritual_store.py           # from v4 (just the store, not ritualizer)
│   ├── win_digest.py             # from v4 (already exists, move here)
│   └── cross_game_memory.py      # new, digest-centric
├── control/
│   ├── progress_tracker.py       # from v3
│   ├── goal_skeleton.py          # NEW
│   ├── specialist_minds.py       # from v3
│   ├── arbiter.py                # from v3, ontology-aware
│   ├── operator_search.py        # from v3
│   ├── reactive_controller.py    # from v3
│   └── dissent_controller.py     # from v4 (hooks removed)
├── execution/
│   └── executor.py               # from v3
└── compression/
    ├── solution_shortener.py     # from v3
    ├── macro_compiler.py         # from v3
    └── ritualizer.py             # from v4
```

Rough line counts (copied, not newly written):
- from v3: ~5,500 lines (the whole v3 tree minus legacy options/reactive)
- from v4: ~1,600 lines (ontology, dissent, ritualizer, surprise, digest)
- from v4_1: ~600 lines (just visual_cortex.py, frozen-forward paths)
- new in v5: ~400 lines (agent, goal_skeleton, cross_game_memory,
  arbiter ontology hook, VC adapter)

Total ≈ 8 kLoC (vs. V4's ≈ 14 kLoC).

---

## 8. Recommended build order

1. Scaffold `v5/` directory, copy V3 files wholesale
2. Port V4 `OntologyCompetition`, `AffordanceReframing`, `SurpriseField`,
   `DissentController` (with its project-market hooks stripped)
3. Port V4 `Ritualizer` + new `RitualStore`
4. Move `win_digest.py` from v4 → v5, rewrite `CrossGameMemoryV5`
   around digests
5. Add `GoalSkeleton` and wire it into `SpecialistMinds.propose()`
6. Add `VisualCortexAdapter` (forward-only wrapper around V4_1 VC)
7. Wire VC priors into `ActionProfiler.update()`
8. Add ontology-aware mind priors in `Arbiter`
9. Top-level `AdaptiveReasoningAgentV5` that runs the loop in §4
10. `run_until_win_v5.py` (clone of the v4 runner, swap imports)
11. Run ablations (§6), cut any component that doesn't earn its seat

Estimated effort: 1–2 focused days to scaffold, +1 day per ablation
sweep, +iterative tuning.

---

## 9. Expected advantages over V3 and V4

Over V3:
- VC priors → faster operator induction (fewer wasted actions
  disambiguating obvious movers vs. no-ops)
- Ontology reframing → reframe stuck games without code changes
- Dissent → actually escapes sterile loops rather than relying on
  branch-kill alone
- Rituals → compact templates transfer across games
- Digest-driven cross-game memory → compact, interpretable, fast to
  seed

Over V4:
- ~6 kLoC less code, ~3× fewer layers between perception and action
- No `project_market` indirection — minds plan operators directly
- No untrained learning suite noise
- Dissent kept, but ontology space is bounded and small
- A real cross-game memory based on digests, not a sprawling ecology

Over V4_1:
- No LLM, no online deep training, no latency variance
- Symbolic control remains auditable
- VC retained as the one genuinely valuable learned component

---

## 10. Open questions

- Should `GoalSkeleton` be 2-level or a slightly richer 3-level
  (goal → subgoal → micro-target)? Start with 2.
- Should `DissentController` be allowed to flip ontologies on its own,
  or only suggest? V4 lets it downweight; keep that.
- Should cross-game memory keep `operator_templates` at all or always
  re-derive from digests? Re-derive by default; only cache if derivation
  proves expensive.
- Do we need a V5 VC checkpoint trained on ARC grids specifically, or
  reuse V4_1's? Reuse first; retrain only if ablation shows benefit.

---

## 12. Refinements (post-review)

Five concrete tweaks addressing "too much pruning, too little exploratory
pressure":

### 12.1 Partial-credit digests

Pure win-gating starves cross-game learning because wins are rare.
V5 digests already record LP/SP/TP, but the cross-game export will now:

- Treat digests with `tp >= 0.15` or `sp >= 0.30` as *partial credit* —
  contribute to operator-kind priors and ontology priors at a fraction
  of a win's weight (e.g. `weight = min(0.8, 0.3 + tp + 0.5*sp)`).
- `useful_operator_kinds` from partial-credit games still propagate
  but with dampened influence.
- Trust still only grows meaningfully on real wins — this is about
  priors, not authority.

### 12.2 Lightweight bissociation (frame-crossing probe)

A cheap hybrid-proposal generator, not a full subsystem. Every ~25
actions, if the current ontology has plateaued (no SP/TP delta for N
steps), pick:

- the current top ontology,
- one distant digest from cross-game memory whose ontology kind
  differs and whose `confirmed_operator_ids` is non-empty,

and emit a short hybrid probe: one primitive from the distant digest,
followed by a click/transform under the current ontology. Bounded
budget (≤ 1 probe per 25 actions), bounded cost (2–3 actions),
quarantined to never become the dominant strategy.

### 12.3 VC influences more than the profiler

Rather than only seeding `ActionProfiler`, the VC adapter will expose
three priors consumed at three places:

- `ActionProfiler`: change-rate prior (current plan).
- `OperatorSearch`: expansion bias — penalise operators whose
  primitive has near-zero predicted change under VC.
- `ExperimentDesigner`: prefer probing actions the VC predicts to be
  informative but high-uncertainty.

Still no online training. The VC remains a frozen, forward-only
checkpoint.

### 12.4 Dissent can force an ontology flip

Current dissent: downweight + interrupt. Add:

- `DissentController.force_ontology(kind, steps=K)` — clamp the top
  ontology to `kind` for K actions, overriding evidence decay.
- Triggered when loops coincide with ontology monoculture AND
  persistent SP stagnation.
- The forced window is short (K ≈ 30) and burns an internal budget
  (max 2 flips per game) to prevent flip-flopping.

### 12.5 Productive-minority metrics

The agent summary and runner will surface, per batch:

- `%_games_with_SP_positive` (SP > 0)
- `%_games_with_TP_positive` (TP > 0)
- `%_games_with_LP_above_0.5`
- `%_games_with_partial_credit` (any of the above at digest-export)

These tell us whether V5 is actually improving where it matters,
independent of the win rate.

---

## 11. TL;DR

> **V5 = V3 spine + V4 ontology/dissent/ritual/surprise/digest + V4_1
> visual cortex as frozen prior generator.**
> Nothing trains online. The agent has one brain (V3's minds), one
> referee (the arbiter), one critic (the dissent controller), one memory
> (digests), and one perception enhancer (VC). Anything that doesn't
> survive a head-to-head ablation against V3 is cut.

