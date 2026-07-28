# SAGE12 candidate-complete semantic-slot pilot V4.7 result

## Verdict

V4.7 completed:

`CURRENT_STACK_NEGATIVE_QWEN_SEMANTICS_BOTTLENECK`

This is a negative result for the present frozen Qwen annotation stack, not a
refutation of the proposed global architecture. Candidate coverage was fixed,
the downstream chain recovered strongly when only the annotations were
replaced by true semantic effects, and the learned EBM exactly recovered the
oracle when supplied true world outputs.

No validation or holdout game was opened, no environment was executed, and no
authority was promoted.

## Frozen population and runtime

The manifest admitted all and only the complete V4.3 source trees:

- 340 complete roots;
- 2,380 pre-action nodes;
- 4,760 legal semantic slots;
- 11 source games;
- 2,380 original Qwen prompts;
- 340 additional root relation-shuffle prompts.

The planning estimate had been 2,396 nodes. The implementation did not fill
the 16-node difference with incomplete trees.

The local frozen Qwen2.5 0.5B model ran on the NVIDIA RTX 4050 Laptop GPU.
All 2,720 requests produced exactly 14 constrained bits. Model inference took
205.636 seconds in 85 batches, with 2.407 seconds median per batch. Prompt
length averaged 456.2 tokens and peaked at 470 under the frozen 512-token cap.
The atomic token IDs were 15 for `0` and 16 for `1`.

Strict bitstream validity, compiler coverage, and `support=0` compliance were
all 1.00. This removes the V4.6 parsing/grounding ambiguity: every legal action
reached the world model and energy ranker.

## Direct Qwen annotation quality

The constrained format solved syntax, not semantics. Before fitting any world
model, Qwen's seven effects had:

- macro Brier: 0.2653;
- macro ECE: 0.3951;
- macro recall at 0.5: 0.4242.

The most consequential common labels were weak:

| Effect | Positives | Recall | Hard-positive rate | Brier |
|---|---:|---:|---:|---:|
| changed | 4,438 | 0.2729 | 0.2645 | 0.4250 |
| moved | 1,077 | 0.2219 | 0.1523 | 0.2133 |
| target_created | 188 | 0.6649 | 0.6413 | 0.2386 |
| target_removed | 619 | 0.3990 | 0.6318 | 0.2982 |
| target_moved | 8 | 0.2500 | 0.4104 | 0.2210 |
| level_complete | 3 | 0.6667 | 0.3983 | 0.2256 |
| game_over | 87 | 0.4943 | 0.5765 | 0.2353 |

The completion recall is based on only three positives and is not evidence of
transfer. Qwen heavily under-called ordinary changes while over-calling
several rare effects.

At the 340 roots, relation shuffling changed 14.83% of hard bits and shifted
probabilities by 0.0694 on average. That sensitivity was not productive
downstream.

## Primary comparison

The fold-local baseline selector chose action-sequence-only on all 11 outer
folds. This baseline used training games only and scored:

- first-branch utility: 7.615955;
- informative-root action accuracy: 0.5400;
- exact selected-leaf utility: 6.965376;
- exact-leaf accuracy: 0.3912.

The full Qwen + calibrated world model + depth-three EBM stack scored:

- first-branch utility: 7.178692;
- informative-root action accuracy: 0.4040;
- exact selected-leaf utility: 6.138756;
- exact-leaf accuracy: 0.2294;
- coverage: 1.00;
- unsafe first-action rate: 0.

Its paired utility gain was **-0.437263**, bootstrap 95% interval
**[-0.728540, -0.192237]**. It was non-negative on only 4 of 11 games, below
the frozen 6-game requirement. The negative interval and cross-game count
both reject `EXPLORATORY_SUPPORT`.

The four non-negative games were bp35 and dc22, where both methods tied, plus
lf52 and sp80. The largest regressions were lp85 (-3.1892), g50t (-1.4071),
and tu93 (-0.9366).

## Factorial ablations

### Qwen adds no useful information

The structured-only stack scored 7.194545. Adding Qwen changed utility by
-0.015853, interval [-0.077378, +0.054728]. The learned world metrics were
also nearly identical:

| World input | Mean Brier | Mean ECE | Utility RMSE |
|---|---:|---:|---:|
| structured only | 0.078057 | 0.088018 | 3.236094 |
| structured + Qwen | 0.077959 | 0.088105 | 3.240042 |

The tiny Brier difference did not carry decision-relevant information. Qwen
slightly worsened calibration and utility prediction.

### Learned hierarchy did not help

Depth one scored 7.279769. Depth three was -0.101077 lower, interval
[-0.204197, +0.007152]. Exact depth-three leaves were also substantially
worse than the optimistic first-branch target.

The deployable root-slot-reuse control scored 7.290395. The non-deployable
future-slot topology was -0.111704 lower, interval
[-0.246229, +0.014548]. Oracle topology did not rescue noisy semantic
predictions.

### Relations were not used productively

The relation-shuffled stack scored 7.233933, so original minus shuffled was
-0.055241, interval [-0.183622, +0.030109]. Only 4.71% of first actions
changed. As in earlier pilots, relation sensitivity exists at the raw output
level but does not align with utility.

## Oracle ladder

The oracle replacements localize the first recoverable bottleneck:

| Method | Utility | Gain vs baseline | Action accuracy | Leaf accuracy |
|---|---:|---:|---:|---:|
| full Qwen stack | 7.178692 | -0.437263 | 0.404 | 0.229 |
| true annotations, learned world + EBM | 8.013718 | +0.397764 | 0.776 | 0.603 |
| true world outputs, learned EBM | 8.045671 | +0.429716 | 1.000 | 1.000 |
| true energy | 8.045671 | +0.429716 | 1.000 | 1.000 |

True annotations alone produced a strictly positive paired interval
[+0.291728, +0.501150]. The corresponding calibrated world model reduced mean
Brier to 0.029225, mean ECE to 0.025796, and utility RMSE to 2.729826.

With true world outputs, the learned EBM matched true energy exactly on every
root and leaf. The current energy architecture is therefore capable of the
required ranking in this finite tree view. The remaining gap between true
annotations and true world outputs shows a secondary world-model limitation,
but the first positive recovery is unambiguously the semantic annotation
replacement.

## Game-signature leakage

Random five-fold game classification reached 0.999790 accuracy from structured
slot/context features, versus 0.094118 majority accuracy. Adding Qwen left it
at 0.999790. The binding/context representation still encodes near-perfect
game signatures even without explicit IDs, coordinates, or raw values.

This does not violate the outer leave-one-game-out firewall, but it explains
why source interpolation can look much easier than cross-game transfer and is
a material representation defect.

## Interpretation

V4.7 answers three questions cleanly:

1. Candidate-complete slots fix the V4.6 proposal/grounding coverage failure.
2. Frozen Qwen2.5 0.5B token logits on the present slot prompt do not supply
   useful causal semantics beyond the structured representation.
3. Correct high-level semantics are sufficient for the learned world/energy
   chain to beat the strong baseline, and exact world outputs let the learned
   EBM recover the full oracle.

The result therefore supports the architectural hypothesis conditionally:
the composition can work when semantic effects are correct. It rejects this
particular zero-shot Qwen semantic interface and exposes game-signature
leakage plus a secondary world-model error. It does not justify live control,
held-out validation, a larger world model, or lower decision thresholds.

## Runtime incident

The first complete evaluation wrote all fold models and decisions but failed
during the final identity probe. On Windows, the current scikit-learn version
first rejected 64-bit sparse indices and then rejected `liblinear` for 11-class
classification. The diagnostic was corrected to use 32-bit sparse indices
and the standard multiclass `lbfgs` solver. This change affects only the
descriptive identity probe, not Qwen, the folds, world models, calibration,
baseline selection, EBM training, decisions, bootstraps, or verdict rules.

The corrected full evaluation ran from the frozen inputs in 502.4 seconds.
Focused tests and all artifact checksums were rerun after the correction.

## Artifacts

- manifest checksum:
  `ad08752af3f8b87afd07efd4aa49d9cf1c4f3a81a06d9cb175b11beea4b10ecc`;
- Qwen output checksum:
  `2f90e634e01455c166004e56c3d31e3b797f17f8c947f0f54d9a95cb245b801b`;
- decisions checksum:
  `f60393d8a7d652ed75b5965a22cb1bc865a167f72908dd8d2a8d97de8c7973bd`;
- folds checksum:
  `900d50ea3d36003645c4cf57450347c965ba0dada22af3ed152e4860e44d5fed`;
- combined model checksum:
  `48a8dc422dcce9407735a9a97385bf8d37b5f9534ff40441c4b524fb34f07f81`;
- Qwen diagnostic checksum:
  `2df5376b0e0fedf535efb69aac94a1edf29e2539554d45fa4ea5967803f40cf5`;
- final result checksum:
  `c81cca6d92b40eed9b880fbb20d57e0f624de02e245641ac1ab7afd5a4cb8c42`.

The machine-readable result is
`training/sage12/integration_pilot_v4_7/result.json`.
