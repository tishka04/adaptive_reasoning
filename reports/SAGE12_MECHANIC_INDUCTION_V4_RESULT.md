# SAGE12 temporal mechanic-induction pilot V4 result

Date: 2026-07-27.

Outcome: `FAIL_CLOSED`, with strong positive evidence for the temporal
mechanic-induction premise. No world model, EBM, controller, holdout,
historical, or `ar25` evaluation was authorized.

Result checksum:
`5987eb9531f568dc814dad46eb9e78d13a3813a9c30db3d6cb1fa8a319e16927`.

## Executive result

V4 is the first SAGE12 pilot in which the learned semantic representation
clearly beat a strong, equally informed baseline on fresh prospective
transitions. Given eight observed transitions, the structured mechanic
inducer achieved:

- macro Brier score 0.0377 versus 0.0708 for the local action-only Beta table;
- macro Brier skill +0.4676;
- bootstrap 95% interval [+0.4089, +0.5232];
- macro-F1 0.5109 versus 0.3583, a +0.1526 gain;
- positive per-game Brier skill on all three validation games;
- +0.7637 Brier skill over the same model with no observed context;
- a 0.3987 skill loss when context effects were detached from their actions.

This is direct evidence that recent observed transitions contain transferable
mechanic information beyond action identity. It supports the architectural
link `history -> mechanic hypothesis -> predicted effect`.

Promotion still failed for two preregistered reasons. The source-training
preflight had already failed persistent actor-role quality in `cd82` and
`sp80`, and prospective macro ECE was 0.1056 against the frozen maximum 0.10.
The gates are conjunctive, so the result remains closed.

## Frozen execution

- manifest checksum:
  `24fc301460c015fbfa9d4647bf13733caa2ca373e07782c8c6d2bda11fdad901`;
- source preflight checksum:
  `5ae964387078c0b0f0ef529fc8d5bb96f05daed697540e1034ab8f5600fff44b`;
- source-priors checksum:
  `7f9d62dddd392387d90c31409c203e0b0d23f5e7432e218f7d011d8ddc08042a`;
- prospective collection report checksum:
  `4b46a8c60f34bfb8900d99ff80c57b3d125bc30f08822ac70d0957409e41bd93`;
- combined prospective-shard checksum:
  `7a2604e3f73d606ad741e6b0d40563ddb58bc4382ed8fab462364167bf7b83cc`.

The implementation, thresholds, protocol, and manifest were published before
preflight. The 1,911 source-training windows and priors were then published
before prospective collection. No rule, prior, feature, calibration method,
threshold, gate, or model was changed after a prospective outcome was opened.

## Data

Source development reconstructed 1,911 unique contiguous eight-transition
windows from the immutable V3 source-training traces.

| Effect | Source positive / negative | Prospective positive / negative |
| --- | ---: | ---: |
| `actor_displaced` | 454 / 1,134 | 213 / 363 |
| `target_created` | 87 / 1,490 | 37 / 498 |
| `target_moved` | 90 / 1,190 | 151 / 384 |
| `target_removed` | 226 / 1,054 | 37 / 498 |

Prospective collection executed exactly 768 new transitions, 256 per game,
using the frozen new seeds. It produced 576 unique scored windows, 192 per
game. Action counts were balanced. Eighty-nine exact chronological repeats
were retained in the raw audit stream as required; collection was not
outcome-adaptive.

The prospective tracker reported actor-role coverage 1.00 in all three games.
The source preflight remained below threshold at 0.831 globally because
`cd82` reached only 0.049 and `sp80` 0.244. Source static-query identity gain
was +0.0816 beyond action-only, inside the frozen +0.10 limit.

## Predictive comparison

| Method | Macro Brier | Macro-F1 | Macro ECE |
| --- | ---: | ---: | ---: |
| Structured temporal mechanic inducer | 0.0377 | 0.5109 | 0.1056 |
| Local action-only Beta | 0.0708 | 0.3583 | 0.1609 |
| Global action-only | 0.1602 | 0.0807 | 0.1556 |
| Deterministic template | 0.2719 | 0.3663 | 0.2719 |
| No-context structured ablation | 0.1596 | 0.0807 | 0.1308 |
| Outcome-shuffled context | 0.0660 | 0.4833 | 0.1183 |
| Binding-shuffled context | 0.0395 | 0.5096 | 0.1085 |

Per-game Brier skill against each game's stronger baseline was:

| Game | Windows | Structured Brier | Baseline Brier | Skill |
| --- | ---: | ---: | ---: | ---: |
| `ls20` | 192 | 0.0082 | 0.0144 | +0.4258 |
| `re86` | 192 | 0.0419 | 0.1354 | +0.6906 |
| `sc25` | 192 | 0.0631 | 0.0728 | +0.1327 |

The representation predicted actor displacement at 0.938 F1 and target
movement at 1.000 F1. Creation and removal each remained at 0.053 F1 at the
fixed 0.5 threshold. Their Brier scores improved only modestly, so V4 does not
establish that all semantic effects are solved.

The binding shuffle caused only a small loss relative to the unshuffled model.
The primary signal therefore comes mainly from binding observed effects to
action scopes, with less demonstrated dependence on the current coarse anchor
condition.

The source-prior label permutation retained high skill because the structured
posterior is driven mainly by the correctly aligned local context. It is not
an outcome-binding control and must not be read as evidence that label
identity is irrelevant. The outcome shuffle above is the causal control for
that question.

## Gate ledger

The final artifact contains 15 top-level structured checks: 13 passed and two
failed.

| Gate | Required | Observed | Pass |
| --- | --- | --- | --- |
| Source windows | at least 1,500 | 1,911 | yes |
| Source label capacity | at least 75 positive and negative | all labels | yes |
| Source actor-role quality | global 0.95 and every game 0.90 | 0.831; two games below | no |
| Static identity leakage | at most action-only +0.10 | +0.0816 | yes |
| Prospective windows | at least 500 | 576 | yes |
| Prospective label capacity | at least 30 positive and negative | all labels | yes |
| Prospective actor-role quality | global 0.95 and every game 0.90 | 1.00 | yes |
| JSON / support / grounding | exactly 1.00 | 1.00 / 1.00 / 1.00 | yes |
| Macro Brier skill | at least +0.10 | +0.4676 | yes |
| Bootstrap lower bound | above zero | +0.4089 | yes |
| Macro-F1 gain | at least +0.05 | +0.1526 | yes |
| Outcome-shuffle skill loss | at least 0.05 | 0.3987 | yes |
| Context gain | at least +0.05 | +0.7637 | yes |
| Every-game skill | non-negative | all positive | yes |
| Macro ECE | at most 0.10 | 0.1056 | no |

The final JSON represents the two source actor checks through the combined
`source_preflight_passed` gate. The source preflight artifact retains their
individual values.

## Qwen diagnostic

Qwen did not receive a quality evaluation. All 128 frozen prompts tokenized to
879 tokens, exceeding the preregistered 512-token input cap. The backend
therefore rejected every prompt before generation:

- strict JSON validity: 0.00;
- emitted hypotheses: 0;
- grounded hypotheses: 0;
- recorded setup/rejection time: 19.51 seconds on `cuda:0`.

The cap was not raised after seeing prospective results. This is a frozen
interface/configuration failure, not evidence that Qwen cannot infer the
mechanics from a correctly bounded history. Qwen was explicitly non-gating,
so this does not change the structured result.

## Interpretation and next decision

V1-V3 asked whether static representations transferred. They did not. V4
shows that a short observed history changes the answer substantially: an
explicit mechanic posterior predicts later effects much better than action
identity, including an online action table using the same observations.

This supports the central architecture more strongly than any previous
SAGE12 result, but it does not yet validate hypothesis-to-trajectory rollout,
energy ranking, or control. The strict protocol also prevents treating the
near-miss calibration result as a pass.

The next experiment should be a small V4.1 replication, not the world model:

1. repair actor tracking on source training without using prospective labels;
2. preregister source-only calibration rather than adjusting the 0.5 outputs
   after validation;
3. compact the Qwen prompt below a frozen verified token cap;
4. use new prospective seeds and require the same predictive, shuffle, and
   every-game results.

Only that clean replication should authorize V5 semantic trajectory fitting.
