# SAGE12 V4.13 — unconditional semantic-bottleneck curve

Status at freeze: **protocol frozen; no V4.13 model result observed**.

## Why this iteration exists

Several recent iterations tested semantic representations but stopped before
the full architecture whenever an intermediate gate failed. Those failures
identified a real representation problem, but they could not determine which
accuracy level the complete architecture needs or whether another module
would fail even with perfect semantics.

V4.13 therefore removes the intermediate stop. Every registered condition is
run through:

`semantic input → world model → depth-3 trajectories → EBM → controller`

The result must locate the first failing component rather than returning only
another local semantic verdict.

## Frozen data and firewall

No new transition is collected. V4.13 fingerprints and reuses:

- the V4.12 outer-LOGO semantic predictions, including the failed learned
  relation model and its root/relation-shuffle controls;
- the V4.11 deterministic post-transition teacher;
- the complete V4.3 source-only depth-three trees;
- the unchanged V4.7 world-model, trajectory and EBM hyperparameters;
- the same eleven SAGE11 source-train games.

Source-validation, holdout, historical and live environments remain closed.
Every learned world-model prediction is outer leave-one-game-out.

## Semantic accuracy curve

The oracle condition exposes the true eleven-effect semantic vector:

- seven `SlotAnnotation` effects;
- the four additional active functional effects `local_change`,
  `contact_lost`, `productive` and `risk`.

The curve contains:

| Condition | Expected bit accuracy |
|---|---:|
| `oracle_100` | 100% |
| `oracle_90` | 90% |
| `oracle_75` | 75% |
| `oracle_50` | 50% |

Noise is not probability softening that a downstream linear model could
trivially rescale. Individual semantic bits are flipped using the frozen hash
`sha256(seed|slot_id|effect)`. The same hash and seed are used at every noise
level, so the error sets are nested. Corruption is applied to both training
and held-game slots; world fitting remains strict LOGO. All four oracle-curve
conditions share the same world-model and EBM seeds so that semantic
corruption is the only intended difference.

V4.13 also fits:

- `structured`: V4.7 structural features without teacher semantics;
- `learned_v4_12`: the actual held-game V4.12 probabilities, despite their
  failed local gate.

## Complete component ladder

Every root receives decisions from:

- deterministic-left;
- action-only;
- action-sequence-only;
- the fold-selected primary baseline;
- structured world model + depth-three EBM;
- learned V4.12 world model + depth-three EBM;
- all four oracle-accuracy world models + depth-three EBMs;
- learned and oracle fixed-heuristic controllers;
- learned root-only and relation-shuffle perturbations;
- learned and oracle root-reuse topology stress tests;
- true executed effects/utility + learned EBM;
- exact maximum executed leaf return (`oracle_energy`).

The true-world lane isolates the EBM/controller from both semantic and world
model error. The semantic-oracle lane tests the learned world model with
perfect semantic inputs. The learned lane then isolates the semantic
predictor.

## Frozen measurements

For every condition V4.13 reports:

- immediate utility and regret;
- depth-three leaf utility and regret;
- oracle first-action and oracle-leaf accuracy;
- unsafe-first-action rate;
- per-game transfer;
- completion-opportunity capture;
- paired bootstrap gain over the fold-selected primary baseline;
- world-model Brier, ECE, recall and utility error.

The semantic curve additionally reports observed bit accuracy, Spearman
correlation between semantic accuracy and utility, and the lowest observed
semantic accuracy whose utility confidence interval remains above the primary
baseline while preserving completion capture.

## Diagnostic classification

The checks do not stop execution. They classify the completed result:

1. `TRUE_WORLD_EBM_CONTROLLER_NOT_SUPPORTED` if the true-world lane does not
   beat the primary baseline with a positive paired-bootstrap lower bound or
   misses all available completion opportunities.
2. `SEMANTIC_WORLD_MODEL_BOTTLENECK` if true world succeeds but perfect
   semantic input through the learned world model does not beat both the
   primary and structured controls or misses completion.
3. `LEARNED_GLOBAL_CHAIN_SUPPORTED` if the learned V4.12 chain beats the
   primary baseline with a positive lower bound, captures completion and is
   nonnegative on at least 6/11 games.
4. `SEMANTIC_PREDICTOR_BOTTLENECK` when the two oracle ladders succeed but the
   learned semantics do not.

Curve monotonicity is separately considered supported at Spearman `≥ 0.80`.
It informs the required semantic precision but cannot prevent any condition
from running.

## Topology and real-win boundary

The frozen V4.3 trees contain true descriptors for future candidate nodes.
This is a deliberate candidate-complete topology oracle shared by the V4.7
line of experiments. It is valid for component attribution but not deployable
in an unseen live state.

V4.13 adds a root-reuse stress test that removes future-node descriptors, but
reusing current slots is not itself a learned state-transition rollout.
Therefore V4.13 will not claim a live win rate. A legitimate gameplay
evaluation requires a deployable world model that generates its own next
semantic state; building that component is justified only if the oracle
component ladder succeeds here.

## Reproduction

```powershell
python -m theory.sage12.semantic_bottleneck_curve_v4_13 freeze
.\ARC-AGI-3-Agents\.venv\Scripts\python.exe -m theory.sage12.semantic_bottleneck_curve_v4_13 evaluate
```

No result may promote authority, open the holdout or authorize live control.
