# SAGE12 V4.14 — human-trajectory temporal semantics result

## Verdict

**`TEMPORAL_SEMANTIC_PREDICTOR_BOTTLENECK`**

The iteration ran the complete registered chain rather than stopping at a
local semantic gate:

```text
human temporal teacher
  → object-relative DeepSets + GRU
  → Qwen constrained semantic prior
  → deployable depth-three rollout
  → learned trajectory EBM
  → receding-horizon active controller
```

The result is negative. The temporal controller improves offline utility over
the action-sequence baseline, but it does not learn transferable relational
semantics, misses seven of eight available completion choices, and wins no
level in the bounded active panel. This rejects the V4.14 semantic
student/rollout as an adequate implementation. It does not refute the full
architecture: the true-world lane still captures all eight offline completion
opportunities with the same learned EBM.

No final-confirmation game was opened and no authority was promoted.

## Corpus and temporal teacher

All six games with human trajectories were used for training:
`ar25`, `bp35`, `cd82`, `cn04`, `dc22`, and `ft09`.

The compiler produced:

- 5,661 non-reset transitions;
- 41 causal sequences;
- 5,620/5,620 adjacent continuity matches;
- 74 immediate level-completion transitions;
- 296, 1,139, and 2,919 positive progress targets within 4, 16, and 64
  actions respectively;
- 101 danger-within-eight targets;
- 2,132 immediate productive effects.

This confirms the motivation for temporal labels. Completion remains rare as
an immediate transition, while successful prefixes yield substantially denser
long-horizon supervision. Earlier actions were not relabelled as immediate
wins.

The student firewall passed: human text, post-action grids, raw colours,
absolute coordinates, state hashes, persistent object identifiers, and game
identity are absent from the model view.

## Training and held-human-game diagnostics

The laptop GPU was selected after a like-for-like benchmark:

| Device | Benchmark time |
|---|---:|
| CPU | 0.65038 s |
| RTX 4050 (`cuda:0`) | 0.02288 s |

The measured GPU speed-up was **28.42×**. The all-six final fit took 57.55 s
for 30 epochs over 5,661 rows and 672 suffix-ranking pairs.

Outer leave-one-human-game-out results were:

| Prediction | Macro balanced accuracy | Macro Brier |
|---|---:|---:|
| action-only immediate effects | 0.47249 | **0.11950** |
| temporal immediate effects | **0.58605** | 0.12389 |
| relation-shuffled temporal effects | 0.58317 | 0.12714 |
| history-shuffled temporal effects | 0.56103 | 0.13880 |
| persistent roles | 0.63040 | 0.17465 |
| progress/danger heads | 0.48667 | 0.15891 |

There is a real but narrow temporal signal. The full model recovers positives
for `moved` (0.602 recall), `local_change` (0.589), `productive` (0.579), and
`contact_lost` (0.561). It still has zero completion recall, only 0.183 risk
recall, and 0.276 target-removal recall.

The controls are decisive:

- relation shuffling changes balanced accuracy by only 0.00288;
- history shuffling removes only 0.02502 balanced-accuracy points;
- a probe recovers the held game from semantic outputs with 100% accuracy,
  versus a 33.92% majority baseline.

The student therefore encodes strong game signatures while making little use
of the relations that were intended to transfer.

## Untraced-game transfer

All 1,056 V4.11 panels from the eight untraced transfer games were evaluated,
covering 2,831 action arms. True continuation frames were used only to score
outcomes. V4.14 received the current graph and proposed action names and
generated every future semantic belief itself.

| Controller condition | Mean utility | Gain over action-sequence | 95% CI | Nonnegative games | Completion |
|---|---:|---:|---:|---:|---:|
| action-sequence baseline | 0.41690 | — | — | 8/8 | 1/8 |
| action-only | 0.55395 | +0.13704 | descriptive | 7/8 | 1/8 |
| deterministic template | **0.58162** | +0.16471 | descriptive | 5/8 | 0/8 |
| V4.12 snapshot + EBM | 0.38967 | −0.02723 | descriptive | 4/8 | 1/8 |
| V4.14 temporal rollout + EBM | 0.48844 | **+0.07154** | **[+0.02067, +0.12300]** | 5/8 | 1/8 |
| relation-shuffled V4.14 + EBM | 0.52650 | **+0.10960** | **[+0.05343, +0.16487]** | 5/8 | 0/8 |
| true world + learned EBM | 0.50758 | +0.09068 | [+0.02471, +0.15553] | 7/8 | **8/8** |
| oracle energy | 0.90430 | +0.48740 | [+0.42108, +0.55260] | 8/8 | **8/8** |

The temporal lane passes the utility confidence interval and the per-game
coverage checks. It fails the frozen completion requirement: one completion
was selected where four were required.

More importantly, scrambling relations improves mean utility by 0.03806 over
the full temporal lane. On held-game semantic effects, the temporal model is
also worse than action-only:

| Semantics | Macro balanced accuracy | Macro Brier |
|---|---:|---:|
| action-only | **0.50000** | **0.07759** |
| V4.12 snapshot | 0.48842 | 0.07550 |
| V4.14 temporal | 0.49437 | 0.11233 |
| V4.14 relation shuffle | 0.49298 | 0.11517 |

The positive offline utility delta cannot be credited to transferable
object-relation semantics. The likely source is the learned EBM combined with
action and temporal marginals.

## Bounded active validation

The active panel used `re86`, `ls20`, and `sc25`, seeds 0–2, 1,000 actions per
run, and at most 14 resets. Both controllers were run for every game/seed
pair.

Qwen2.5 0.5B retained the V4.7 decoder: atomic constrained `0`/`1` tokens, no
sampling, no temperature, 14 bits per two candidates, maximum 512 input
tokens, and batch size 32. It refreshed an action-level semantic prior every
128 executed actions and after resets. At every refresh, all legal candidates
were included. The prior was blended at weight 0.5 with the learned temporal
effects and carried through the recurrent belief between refreshes.

Execution audit:

- 18/18 registered runs completed;
- 16,368 non-reset actions were executed;
- both controllers executed 8,184 actions;
- `sc25` stopped at the 14-reset bound, hence 728 rather than 1,000 actions
  per run;
- 0 illegal proposals;
- 474 Qwen pair requests covering 876 candidates;
- 100% strict bitstream validity;
- 100% compiler candidate coverage at Qwen refreshes;
- 66.14 s total Qwen inference;
- 566.78 s total active-panel runtime.

| Active result | Action-sequence baseline | Qwen + temporal rollout + EBM |
|---|---:|---:|
| levels completed | 0 | 0 |
| WINs | 0 | 0 |
| GAME_OVERs | 72 | **93** |
| mean decision latency | 0.00025 s | 0.07166 s |
| illegal proposals | 0 | 0 |

Per game, neither controller completed a level. On `ls20`, the semantic
controller caused seven additional GAME_OVERs in every seed (21 total versus
zero for the baseline). Results on `re86` and `sc25` were tied at zero
progress.

This is the first V4.14 result that directly answers the live question: this
implementation does **not** currently win levels.

## Interpretation

The iteration resolves the earlier ambiguity about rare terminal transitions.
Temporal credit makes the training target much denser, but target density was
not the only bottleneck.

Three failures remain:

1. **The learned semantics do not transfer.** Relation shuffling is almost
   harmless in held-human-game evaluation and beneficial in the global
   controller panel.
2. **The temporal heads do not identify successful prefixes.** Their aggregate
   balanced accuracy is below chance and completion recall is zero.
3. **The controller has no competent behavioral prior.** Both the human
   action-sequence marginal and the semantic controller fall into repetitive
   non-winning behavior when deployed from scratch.

The true-world lane is the important architectural control. With the same
learned EBM it selects all eight offline completion opportunities, so the EBM
and selection mechanism can exploit correct information. V4.14 therefore
refutes the current teacher-to-student semantics and deployable rollout, not
the hypothesis → world model → energy → controller architecture in principle.

## Recommended next iteration

Do not add another head to the same effect classifier. The next experiment
should establish a competent trajectory prior first:

1. train an object-relative sequence policy on complete human prefixes using
   causal next-action imitation and success-conditioned return-to-go;
2. represent subgoals as tracked object-state changes and distances to
   demonstrated milestones, not only independent effect bits;
3. use the semantic world model and EBM as a reranker over the sequence
   policy's candidate plans;
4. compare behavior cloning, sequence policy + EBM, oracle milestone
   semantics, relation shuffle, and the full learned chain on held games;
5. require non-zero live level progress before expanding the active panel or
   opening final confirmation.

This directly tests whether demonstrations can provide the missing behavioral
scaffold while preserving the higher-semantic architecture.

## Reproducibility

```powershell
python -m theory.sage12.human_temporal_semantics_v4_14 freeze
python -m theory.sage12.human_temporal_semantics_v4_14 compile
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.human_temporal_semantics_v4_14 train --device auto
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.human_temporal_semantics_v4_14 evaluate --device auto
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.human_temporal_semantics_v4_14 active --device auto
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m pytest -q tests/test_sage12_human_temporal_semantics_v4_14.py
```

Checksums:

- frozen manifest:
  `9a6d1f59189e2b84cadd5217cca2da489574e255208fbc9bfd7a73609142e6b1`
- teacher corpus:
  `c515272bc8e40f7bdc8fd15395a2e8dc2ae093a3c1e2d7dd639582878088efac`
- temporal checkpoint:
  `6636caf4cdb374b671f9efc9bbbad6a008291332033b083dbc74dcf73d53ef30`
- semantic result:
  `aea425506c3e894d2262acd17289d8d9da8c03b997ac9ef6b84f20842fbf881a`
- active validation:
  `3b5a2ecf467db20a6a42441099806b7819ed69115549fe494b67205651234a6e`
- integrated result:
  `981e63e100df83814e9a777a31c2e9932edd0b410bceec9698c400239a41b76b`
