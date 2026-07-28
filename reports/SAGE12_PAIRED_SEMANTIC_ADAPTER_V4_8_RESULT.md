# SAGE12 paired semantic adapter V4.8 result

## Verdict

V4.8 completed:

`EXPLORATORY_ARCHITECTURE_NOT_SUPPORTED`

The paired adaptation contains decision-relevant signal: it improves the
unchanged structured-only world/EBM/controller stack by +0.206779 utility with
an entirely positive bootstrap interval. That gain is not enough. The
adapted stack remains -0.243501 below the stronger fold-selected
action-sequence baseline, selects none of the three completion trajectories,
and passes only two of seven frozen exploratory checks.

This rejects the frozen-Qwen plus external low-rank adapter as the current
semantic acquisition mechanism. It does not refute the global architecture:
the unchanged true-annotation diagnostic again beats the baseline by
+0.397764.

No validation, holdout, or historical outcome was opened, no environment was
executed, and no authority was promoted.

## Population and firewall

The experiment used:

- 5,128 source-train same-prestate pairs;
- 2,748 sampled SAGE11 pairs;
- all 2,380 V4.3 tree nodes and 4,760 slots;
- 340 complete depth-three roots across 11 source-train games;
- 340 relation-shuffled root pairs.

Every semantic annotation, world prediction, and decision for a held game was
produced without semantic outcomes from that game. Baseline selection also
used training games only. The V4.3 future tree remains a non-deployable
topology oracle.

## Semantic checkpoint

The selected `invariant_context` adapter was already weaker than action-only
on direct seven-effect prediction:

- adapted LOGO macro Brier: 0.093676;
- action-only LOGO macro Brier: 0.067048;
- skill: -0.026628;
- completion recall: 0/3;
- adapted semantic-output identity: 0.914286 at the pair checkpoint.

The full semantic checkpoint and GPU runtime are documented separately in
`SAGE12_PAIRED_SEMANTIC_ADAPTER_V4_8_SEMANTIC_RESULT.md`.

## End-to-end comparison

The fold selector chose action-sequence-only as the primary baseline on all
11 folds.

| Method | First-action utility | Action accuracy | Leaf utility | Leaf accuracy |
|---|---:|---:|---:|---:|
| primary action-sequence baseline | 7.615955 | 0.540 | 6.965376 | 0.3912 |
| structured-only V4.7 stack | 7.165675 | 0.380 | 6.079387 | 0.2471 |
| adapted semantic stack | 7.372454 | 0.484 | 6.605327 | 0.3324 |
| true annotations, learned world + EBM | 8.013718 | 0.776 | 7.939827 | 0.6029 |

The adapted-versus-baseline mean gain was **-0.243501**, bootstrap 95%
interval **[-0.534479, -0.004103]**. Even though the confidence interval was
not required by the relaxed exploratory rule, it is entirely negative.

The adapter was non-negative on 5/11 games:

- positive: bp35 +0.0313, g50t +0.7736, lf52 +0.6164, sp80 +0.0535;
- tied: dc22;
- negative: cd82 -0.1951, ka59 -0.3556, su15 -0.1244, tr87 -0.0509,
  tu93 -1.1513, and lp85 -3.1892.

The largest failure is `lp85`, the only game containing the three observed
completion trajectories.

## Useful but insufficient semantic increment

Against structured-only, adapted semantics gained **+0.206779**, interval
**[+0.095707, +0.330483]**. This is the strongest evidence of progress in
V4.8: the semantic adapter is not ignored by the downstream system.

The adapted stack also improves the V4.7 zero-shot stack's utility from
7.178692 to 7.372454, a post-hoc difference of +0.193762. It improves:

- first-action accuracy from 0.404 to 0.484;
- leaf utility from 6.138756 to 6.605327;
- leaf accuracy from 0.2294 to 0.3324.

This is meaningful engineering progress, but the strong action-sequence
baseline remains ahead.

## Completion behavior

All three completion events occur at the third action of `RRR` trajectories
in `lp85`.

| Method | Completion trajectories selected |
|---|---:|
| adapted semantic stack | 0/3 |
| adapted relation shuffle | 0/3 |
| structured-only | 0/3 |
| primary action-sequence baseline | 3/3 |
| true annotations | 3/3 |

The initial diagnostic implementation inspected only the root action and
therefore reported zero opportunities. It was corrected to traverse the full
selected depth-three path, a focused regression test was added, and the
entire 11-fold evaluation was rerun. The decision and fold checksums were
unchanged; the completion gate and final result checksum changed.

## Relation use and identity leakage

Original semantics beat their relation shuffle by +0.018551 utility, but the
interval [-0.006094, +0.054339] crosses zero. Only 2.35% of first actions
changed. This passes the deliberately relaxed positive-mean check, but is weak
evidence that relations are used productively.

Game classification from the seven adapted slot probabilities reached
0.856092, versus 0.094118 majority and 0.282563 from the V4.7 zero-shot
probabilities alone. The adapter made output identity leakage substantially
worse, even though explicit identifiers and raw coordinates were excluded.

The context-free candidate also leaked heavily during SAGE11-only selection
(0.985082), so this is not caused solely by the selected unordered context.
Action/binding effect distributions themselves remain game signatures.

## World-model effects

The adapted semantic features slightly improve mean effect Brier and ECE:

| World input | Mean Brier | Mean ECE | Utility RMSE |
|---|---:|---:|---:|
| structured only | 0.078057 | 0.088018 | 3.236094 |
| adapted semantics | **0.076649** | **0.072814** | 3.280141 |
| true annotations | **0.029225** | **0.025796** | **2.729826** |

Calibration gains do not translate into sufficient utility prediction. The
adapted world has worse utility RMSE and zero 0.5-recall for movement,
creation, removal, movement-of-target, completion, and game-over after
calibration.

## Frozen checks

| Check | Result |
|---|---|
| direct Brier beats action-only | FAIL |
| adapted utility beats structured-only | PASS |
| adapted utility beats primary baseline | FAIL |
| non-negative on at least 6/11 games | FAIL (5/11) |
| original utility beats relation shuffle | PASS |
| select at least 2 completion trajectories | FAIL (0/3) |
| semantic-output identity at most 0.60 | FAIL (0.8561) |

The final verdict therefore passes 2/7 conjunctive checks.

## Interpretation

V4.8 answers the question that earlier component gates could not:

1. A paired, cross-game semantic adapter can improve the integrated
   world-model/EBM/controller stack over structured-only.
2. This specific adapter still loses to a strong action-sequence baseline and
   fails exactly where solving matters: rare completion trajectories.
3. Correct semantic annotations continue to make the same downstream
   architecture decisively positive.

The next iteration should not tune the V4.7 world model or EBM. The evidence
supports replacing the semantic learner or its supervision:

- acquire genuinely object-relative, same-state completion contrasts;
- train a model with explicit invariance or adversarial game-identity removal;
- use a stronger semantic backbone or a non-LLM relational learner;
- make completion/progress ranking, rather than average effect reconstruction,
  the training objective.

The 0.5B frozen-encoder/external-adapter route should not be scaled unchanged.

## Reproducibility and artifacts

Focused tests pass, including the full-path completion regression. The
complete SAGE12 test group also passes.

- manifest:
  `143fe10e1b35f7fa2dc1dc1078f86beb21042d6a647b6235c1c1ca23bccbce67`;
- semantic result:
  `2b90c09c04a44c3e436e45fe5089c515001d25a9279cc636b9b8706c61895151`;
- semantic annotations:
  `ae37faa5290b0d2cd9b978da10c7c589b2f329964f38d38d34ebcec533d60454`;
- decisions:
  `b94b96c68ee8c35603d14bff46f1cc87837275a0e235761ec890c74fc531c082`;
- folds:
  `c411bfe9f201413662b354aa46d305c20cbb49fc3652a8c55583eac19e10d525`;
- final result:
  `be7d25af9d270f982bfca25ad1f95c10f2a7f5955677aa2e05f2d39a16f34bb0`.

The machine-readable result is
`training/sage12/semantic_adapter_v4_8/result.json`.
