# SAGE12 V4.15 — demonstration-conditioned milestone policy result

## Verdict

**`BEHAVIOR_PRIOR_BOTTLENECK`**

V4.15 executed the full registered chain:

```text
complete human demonstrations
  → causal object-relative sequence policy
  → learned milestone and return conditioning
  → V4.14 semantic rollout
  → learned trajectory EBM
  → bounded receding-horizon controller
```

The result is negative for the proposed implementation. The sequence model
does learn a substantial, cross-game human-choice signal, but that signal is
mostly insensitive to object relations, strongly identifies the source game,
and does not become a competent milestone policy. The EBM makes the learned
policy worse offline, and neither learned controller completes a level in the
active panel.

This result does not refute the global hypothesis → world model → EBM →
controller architecture. The true-world lane still lets the same learned EBM
capture all eight offline completion opportunities. It does show that
demonstration-conditioned behavior cloning over the current object-relative
graph and milestone targets is not yet the missing semantic interface.

No final-confirmation game was opened and no controller authority was
promoted.

## Demonstration corpus

The frozen compiler used every non-reset transition from the six games with
human traces: `ar25`, `bp35`, `cd82`, `cn04`, `dc22`, and `ft09`.

It produced:

- 5,661 candidate-choice records in 41 causal sequences;
- 5,620/5,620 matching adjacent before/after states;
- 10–16 candidates per decision, with a 15.84 mean;
- 60,231 duplicate semantic candidates removed;
- 4,962 positive suffix returns;
- all eight registered milestone classes;
- no forbidden identity, coordinate, colour, future-frame, or audit field in
  the student view.

The milestone counts were:

| Milestone | Rows |
|---|---:|
| `target_moved` | 1,331 |
| `target_removed` | 1,192 |
| `path_opened` | 1,078 |
| `none_within_64` | 719 |
| `level_complete` | 657 |
| `productive` | 383 |
| `actor_approached_root` | 152 |
| `target_created` | 149 |

Candidate compilation initially exposed a severe redundant-observation cost.
The final compiler extracts each structured observation once per human state
and derives every candidate graph from that cached observation. A focused
equivalence test proves exact graph equality for movement, click, and
targetless actions. The complete corpus then compiled in 482.43 seconds.
This is a runtime optimization only: the frozen candidates, labels, order,
and manifest did not change.

One segment has no matching episode-summary row in the raw human files. It is
retained because all of its transition rows are present and its internal
state continuity passes. This is reported as `orphan_sequences: 1` rather
than silently repaired.

## Device selection and training

The repository's ARC environment was required both for the compatible game
API and CUDA:

| Device | Registered benchmark |
|---|---:|
| CPU | 0.35812 s |
| RTX 4050 (`cuda:0`) | 0.05150 s |

CUDA was **6.95× faster**, above the frozen 1.2× threshold, and was therefore
used. The final all-game fit took 15.95 seconds for 30 epochs. The six LOGO
fits plus the final model took 170.3 seconds end to end.

## Leave-one-human-game-out result

Every prediction below comes from a model trained on the other five human
games.

| Condition | Top-1 | Productive rows | Within-16 rows | Mean NLL |
|---|---:|---:|---:|---:|
| action-only | 0.00866 | 0.00563 | 0.00439 | **2.86112** |
| deterministic template | 0.20738 | 0.26689 | 0.17384 | 2.83914 |
| behavior policy | **0.43915** | **0.45826** | 0.46795 | 3.19661 |
| learned milestone policy | 0.43597 | 0.44325 | 0.46181 | 3.19560 |
| oracle milestone policy | 0.44285 | 0.45732 | **0.47059** | 3.23631 |
| relation shuffle | 0.43614 | 0.44606 | 0.44952 | 3.08265 |
| history shuffle | 0.39958 | 0.41370 | 0.42318 | 3.48353 |

The learned milestone policy beats action-only by **+0.42731** top-1 with a
95% paired-bootstrap interval of **[+0.41459, +0.44109]**. It is nonnegative
on all six held games. Exact imitation on decisions with exactly three
candidates also rises from 0.00072 action-only and 0.04732 template to
0.18874.

This is real progress: complete causal histories provide a much stronger
behavioral prior than the earlier per-action marginals.

It is not the registered semantic success:

- relation shuffling changes top-1 by **−0.00018**, with interval
  `[-0.01255, +0.01131]`;
- the learned milestone condition is slightly worse than ordinary behavior
  cloning;
- even the oracle milestone improves behavior cloning by only 0.00371;
- milestone macro balanced accuracy is **0.07717**, below the 0.125 majority
  baseline and far below the required 0.175;
- the output-based game probe reaches **86.29%**, versus 33.92% majority,
  exceeding the allowed +0.20 gain by a wide margin.

The model is therefore learning transferable action/history regularities,
but not the intended relation-grounded milestone abstraction.

## Untraced-game transfer and EBM

All 768 V4.11 transfer panels and 2,831 action arms were evaluated. Future
frames were used only to score the selected arms.

| Controller condition | Mean utility | Mean regret | Completion captured |
|---|---:|---:|---:|
| action-sequence-only | 0.41690 | 0.48740 | 1/8 |
| action-only | 0.55395 | 0.35035 | 1/8 |
| deterministic template | **0.58161** | **0.32268** | 0/8 |
| behavior policy | 0.39645 | 0.50785 | 2/8 |
| learned milestone policy | 0.39318 | 0.51112 | 2/8 |
| relation-shuffled policy | 0.39426 | 0.51004 | **8/8** |
| policy + temporal EBM | 0.37462 | 0.52967 | 1/8 |
| oracle milestone + temporal EBM | 0.39262 | 0.51168 | 1/8 |
| true world + learned EBM | 0.50758 | 0.39672 | **8/8** |
| exact oracle | 0.90430 | 0.00000 | **8/8** |

Adding the temporal EBM to the learned milestone policy changes utility by
**−0.01855**, with a 95% interval of `[-0.04805, +0.00908]`. It is
nonnegative against the policy alone on only 4/8 games and captures 1/8
completion opportunities, below the registered requirement of 4/8.

The striking relation-shuffle completion result is not evidence for the
learned relations: scrambling them captures 8/8 while preserving almost the
same mean utility. It is another sign that the graph relations are not
controlling the decision reliably.

The true-world control remains important. Supplying observed future effects
to the same learned EBM improves over the learned milestone policy by
**+0.11440**, interval `[+0.05405, +0.17381]`, and captures 8/8 completions.
The EBM can use correct semantic trajectories; the deployable policy/world
model does not predict them accurately enough.

## Bounded active validation

V4.15 reused the checksummed V4.14 action-sequence baseline and executed 18
fresh runs:

- games `re86`, `ls20`, and `sc25`;
- seeds 0–2;
- 1,000 actions per run;
- at most 14 resets;
- ordinary behavior policy and milestone policy + temporal EBM.

All 27 registered rows are present: nine frozen baselines and 18 fresh runs.
The fresh panel took 1,095.64 seconds.

| Active result | Action sequence | Behavior policy | Milestone + EBM |
|---|---:|---:|---:|
| actions | 8,184 | 8,184 | 8,184 |
| levels | 0 | 0 | 0 |
| WINs | 0 | 0 | 0 |
| GAME_OVERs | **72** | 93 | 93 |
| illegal proposals | 0 | 0 | 0 |
| mean decision latency | **0.00025 s** | 0.04396 s | 0.07737 s |

No controller completed a level on any game or seed. Both learned policies
added 21 GAME_OVERs on `ls20`, where the baseline had none. The EBM did not
change the aggregate active behavior relative to ordinary behavior cloning.

## Gate audit

The behavior-prior gate passed two checks and failed three:

| Check | Threshold | Result |
|---|---|---|
| learned over action-only | paired 95% lower bound > 0 | pass |
| nonnegative held games | at least 4/6 | pass: 6/6 |
| relation sensitivity | shuffle-degradation lower bound > 0 | fail |
| milestone quality | balanced accuracy ≥ majority + 0.05 | fail |
| game signature | probe gain ≤ 0.20 | fail |

The global gate also failed:

| Check | Threshold | Result |
|---|---|---|
| EBM utility gain | paired 95% lower bound > 0 | fail |
| nonnegative transfer games | at least 5/8 | fail: 4/8 |
| completion | at least 1 and at least 50% of oracle | fail: 1/8 vs 4 required |
| all conditions executed | true | pass |
| future outcomes scoring-only | true | pass |

The final verdict is consequently `BEHAVIOR_PRIOR_BOTTLENECK`, with
`authority_promoted: false`, `holdout_opened: false`, and
`all_conditions_executed: true`.

## Interpretation

The iteration separates two ideas that were previously conflated:

1. **Demonstrations do teach useful sequential regularity.** Moving from an
   action marginal to a causal history raises held-game choice accuracy from
   effectively zero to about 44%.
2. **That is not yet high-level semantics.** The learned policy is nearly
   unchanged when relations are shuffled, cannot predict balanced milestones,
   carries strong game signatures, and fails to maintain a productive
   subgoal in live control.

The likely failure is the supervision interface. A single “next milestone
within 64” token collapses many different causal plans into the same label,
while the candidate loss rewards reproducing exact human clicks even when
multiple actions are functionally equivalent. The GRU can exploit repeated
source-game routines without learning an invariant transformation or
precondition structure.

The next iteration should not merely enlarge this network or lower the gates.
It should represent and predict transformations: what object or free-space
topology must change, which preconditions make that change possible, and how
the current action advances that transformation. Candidate evaluation should
credit causal equivalence classes and multi-step option progress, not only an
exact next action. V4.15's behavior policy can remain a proposal prior, but it
should not receive authority or define the semantic state.

## Reproducibility

```powershell
python -m theory.sage12.demonstration_milestone_policy_v4_15 freeze
python -m theory.sage12.demonstration_milestone_policy_v4_15 compile
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.demonstration_milestone_policy_v4_15 train --device auto
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.demonstration_milestone_policy_v4_15 evaluate
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.demonstration_milestone_policy_v4_15 active --device auto
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest -q tests/test_sage12_demonstration_milestone_policy_v4_15.py
```

Checksums:

- frozen manifest checksum:
  `bc9a96fe6642533facad1dc4a11d9b782ec7ed633d4629e4651c6d81b11cdf6a`;
- teacher corpus SHA-256:
  `3aefc5436cf2b0c50a74a5b09d5c487b0533b1e6dd75a72992b77f5e91329fa7`;
- policy checkpoint SHA-256:
  `456adc082b2249d2b392789021404cbcdc60890da6ddd5b33463063d1b724d3b`;
- semantic result checksum:
  `c51fed7be89e2d30a9b0837ef4b3feaf6e31b30b196c6248724488250a7f5a27`;
- active validation checksum:
  `48bdecc30f5b2509d57e035102d5e47c454b8e2f1a2712ba188e97a4fd4c64c0`;
- integrated result checksum:
  `c0605301b1c551453eb8bf389eecad788a5664c9118972d15a66f0404861a1a5`.
