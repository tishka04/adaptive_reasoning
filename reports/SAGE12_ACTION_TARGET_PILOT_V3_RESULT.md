# SAGE12 action-target effect pilot V3 result

Date: 2026-07-27.

Outcome: `FAIL_CLOSED`. The structured action-target classifier did not
transfer beyond the stronger frozen baseline. No semantic world model, EBM,
shadow controller, bounded probe, holdout, historical, or `ar25` evaluation
was authorized.

Result checksum:
`10b1d84b6ff675c3fd05f73ad853d0618658b79045824ad4c2f9e79e6466fdb4`.

## Executive result

V3 repaired the main scientific weaknesses of the earlier pilots: it used
fresh executed transitions, exact action anchors, independent effect labels,
partial credit, conservative before/after matching, and stronger controls.
The interface was fully valid and grounded. The resulting representation was
still not predictive enough across games.

The primary shallow gradient-boosting model reached 0.232 macro-F1. This was
below action-only at 0.237 and below the deterministic template at 0.371.
Its gain against the stronger baseline was -0.140, with a paired bootstrap
95% interval of [-0.155, -0.125]. Target shuffling reduced macro-F1 by only
0.0005 instead of the required 0.05. Calibration was also poor at 0.397
macro ECE.

The correct conclusion is narrow: the current one-step `coarse`
action-target projection is not a transferable semantic effect
representation. This result does not reject sequence-conditioned mechanism
induction, richer persistent object tracking, or the complete
hypothesis/world-model/energy architecture.

## Preregistered execution

The protocol, code, tests, and frozen manifest were committed before
collection. The source-training projection and model family were then frozen
before source validation was scored.

- frozen manifest checksum:
  `8aff373b2896b13dfafe88a8a8d37d9399088f881386a19c81cb63acb3f487bf`;
- source-training collection manifest checksum:
  `1ba0b41b2595a1c9f18f613696e97b4397066194fe700117104d1eaa930d3331`;
- source-only preflight checksum:
  `1ea27b59159bb138cfa7321fbf40d2a5abf6d20e3302c02a05b1ba4c14fccc5a`;
- projection-freeze checksum:
  `7e1a93970b5502873bce6c3659ba46f671752adce81a8b2da829a6485b36ce9c`;
- source-validation collection report checksum:
  `3a5eb3bb97f1ad8505456eca7652a37ac61428808ef171036ddbc253bcde25a3`;
- combined validation-shard checksum:
  `71f18d8d30cc6b500e0c5fafa0e2edf17e0b8ba1edb9b433882421fef3d4c5af`.

Three execution amendments were made without changing the scientific design:

1. atomic file replacement was retried after transient OneDrive locks;
2. an 11-row saturated-game shortfall was deterministically reallocated
   within the frozen global top-up and per-game caps;
3. the frozen liblinear identity probe was explicitly wrapped one-vs-rest for
   scikit-learn 1.9 compatibility.

After validation collection but before any validation metric, the secondary
Qwen shuffle path was corrected to serialize the same raw semantic mapping as
the unshuffled path. The structured-model control, data, projection, gates,
and model choices did not change. All amendments are recorded in the frozen
protocol.

## Data evidence

The final corpus contains exactly 4,000 unique source-only rows.

| Split | Rows | Games | Exact duplicates | Non-ambiguous |
| --- | ---: | ---: | ---: | ---: |
| Source training | 3,040 | 11 | 0 | 0.822 |
| Source validation | 960 | 3 | 0 | 1.000 |

Observed label capacity was:

| Label | Train positive / negative | Train games with positive | Validation positive / negative | Validation games with positive |
| --- | ---: | ---: | ---: | ---: |
| `actor_displaced` | 721 / 1,779 | 10 | 345 / 615 | 3 |
| `target_created` | 160 / 2,343 | 4 | 62 / 831 | 1 |
| `target_moved` | 147 / 1,808 | 5 | 264 / 626 | 3 |
| `target_removed` | 369 / 1,586 | 4 | 62 / 828 | 1 |

The training label-capacity gate passed. The validation capacity gate failed
because creation and removal positives occurred only in `sc25`, not in the
required two validation games. Actor identification was unavailable on 540
source-training rows, mostly `cd82` and `sp80`; this reduced global
non-ambiguity below the frozen 0.95 threshold.

## Source-training freeze

The source-only leakage ladder rejected `full` at +0.3793 identity accuracy
over action-only and `no_shape` at +0.1289. It selected `coarse` at +0.0987,
inside the frozen +0.10 limit. Source-training leave-one-game-out macro-F1
selected shallow gradient boosting at 0.2012 over logistic regression at
0.1903.

No source-validation label, score, projection change, or model-family change
was used in this selection.

## Predictive results

| Method or control | Macro-F1 | Macro ECE |
| --- | ---: | ---: |
| Structured action-target model | 0.2319 | 0.3970 |
| Action-only baseline | 0.2372 | 0.3958 |
| Deterministic template | 0.3714 | 0.2648 |
| Qwen frozen-embedding ablation | 0.2372 | 0.3778 |
| Target-shuffled structured control | 0.2314 | 0.3954 |
| Action-shuffled structured control | 0.2592 | 0.3733 |
| Label-permutation control | 0.2123 | 0.3032 |

The structured model predicted neither `target_created` nor
`target_removed` on validation, giving both labels F1 0. Its
`actor_displaced` F1 was 0.535 and `target_moved` F1 was 0.393. The template
was strong because it predicted `target_removed` at 0.984 F1 and
`target_moved` at 0.484 F1.

Per-game transfer exposed the failure rather than hiding it in the aggregate:

| Validation game | Structured | Action-only | Template | Stronger baseline | Gain |
| --- | ---: | ---: | ---: | --- | ---: |
| `ls20` | 0.0440 | 0.0431 | 0.0092 | action-only | +0.0009 |
| `re86` | 0.3889 | 0.4167 | 0.2500 | action-only | -0.0278 |
| `sc25` | 0.0149 | 0.0115 | 0.4167 | template | -0.4017 |

The fixed typed renderer emitted 2,419 hypotheses. Strict JSON validity,
`support=0`, and compiler grounding were all 1.000.

## Gate ledger

| Frozen gate | Required | Observed | Pass |
| --- | --- | --- | --- |
| Source-training label capacity | at least 100 positive and negative per label | met for all four labels | yes |
| Validation label capacity | at least 20 positive and negative and positives in two games | creation/removal positive in one game | no |
| Exact duplicates | zero | zero | yes |
| Global non-ambiguous rate | at least 0.95 | 0.822 | no |
| Per-game non-ambiguous rate | at least 0.90 | `cd82` 0.040; `sp80` 0.198 | no |
| Strict JSON validity | 1.00 | 1.00 | yes |
| `support=0` | 1.00 | 1.00 | yes |
| Grounded hypotheses | at least 0.99 | 1.00 | yes |
| Macro-F1 gain | at least +0.10 | -0.140 | no |
| Bootstrap lower bound | above zero | -0.155 | no |
| Target-shuffle degradation | at least 0.05 | 0.0005 | no |
| Every-game gain | non-negative | negative on `re86` and `sc25` | no |
| Macro ECE | at most 0.10 | 0.397 | no |
| Game-identity gain | at most +0.10 | +0.0987 | yes |

The result artifact records 14 named checks because JSON validity and
`support=0` are represented separately. Six passed and eight failed. Every
gate was conjunctive, so any one failure was sufficient to stop promotion.

## Qwen diagnostic

Qwen2.5 0.5B Instruct was used only as the frozen secondary encoder ablation.
The 26 unique prompts were embedded on `cuda:0` using the laptop NVIDIA
GeForce RTX 4050. One batch took 0.689 seconds, with a maximum prompt length
of 84 tokens.

- prompt checksum:
  `f6d95cf27c2f5366a350f60e2934e6bf576a5add2f1f4aa3f4385d9746fc5db0`;
- embedding checksum:
  `33c58adce975fccebdc71bb3c0ddc1d25879232dc51e3502b2660b43fb6b84a8`;
- model-weight checksum:
  `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.

Its 0.2372 macro-F1 exactly matched the action-only baseline and did not alter
the primary verdict.

## Post-hoc diagnosis

The post-hoc diagnostic is explanatory only and changes no gate. Its checksum
is
`ba8a2b51d1dbf10efddc73247fb690fdd773fb0ea62c35b6f7eb8b1d5bc4dd02`.

The selected coarse projection had only 26 unique source-training signatures
and eight validation signatures. Every validation signature had already been
seen in training. The within-game/action/anchor target shuffle changed only
12 of 960 validation rows, or 1.25%, which explains why its degradation was
nearly zero: the target descriptors had almost no remaining conditional
variation to perturb.

The same shared signatures also mapped to materially different effects across
games. Mean absolute training-to-validation effect-rate shifts were 0.061 for
actor displacement, 0.156 for creation, 0.224 for movement, and 0.289 for
removal. On `re86`, actor displacement and applicable target movement occurred
on every row; on `sc25`, creation and removal were concentrated in click
actions; on `ls20`, all effects were rare. A global one-step classifier
therefore learned source-game marginals that did not identify the held-out
mechanic.

## Decision and next scientific step

World-model fitting remains unauthorized. Collecting more rows with the same
26-signature projection is also not justified by this result.

The next repair should first establish stable actor/object identity and
represent changes over a short observed history. It should test whether a
mechanic hypothesis such as "this action moves the actor" or "clicking this
role removes and replaces an object" can be inferred from a few within-game
transitions and then predict a later transition. That sequence-conditioned
question is closer to the intended high-semantic architecture than another
global one-step classifier.

Any such experiment requires a new versioned protocol, training-only
preflight, meaningful counterfactual shuffle, and fresh promotion gates. It
must not fit the semantic world model or EBM until its own transfer gate
passes.
