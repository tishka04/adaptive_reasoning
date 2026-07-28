# SAGE12 paired semantic adapter V4.8 semantic checkpoint

## Status

The source-only semantic stage completed with a negative direct result. This
checkpoint is published before fitting the V4.7 semantic world model:

`DIRECT_SEMANTIC_ADAPTATION_NEGATIVE`

The result does not stop the exploratory end-to-end run. V4.8 was explicitly
designed to test the complete architecture even when a cheap intermediate
metric fails. No validation, holdout, historical outcome, or live environment
was opened.

## Frozen population

The published manifest admitted:

- 5,128 same-prestate pairs;
- 2,748 sampled SAGE11 pairs;
- all 2,380 replay-verified V4.3 nodes;
- 1,153 changed, 1,461 moved, 1,172 progress, 351 game-over, and 22
  level-completion contrasts;
- 172 target-creation, 517 target-removal, and 8 target-movement contrasts.

Every semantic output for a game was trained without either SAGE11 or V4.3
rows from that game.

## GPU encoding and adapter

The frozen Qwen2.5 0.5B model encoded 4,395 unique pair prompts on the NVIDIA
RTX 4050 Laptop GPU:

- 92 batches;
- 91.586 seconds of model inference;
- 174.52 mean input tokens;
- 253 maximum tokens under the frozen 256-token cap;
- 896-dimensional representations.

Qwen's weights were not updated. Each fold trained only a 59,168-parameter
rank-16 external residual adapter and eight four-class heads.

The first encoding attempt stopped before producing predictions because the
verbose serialization reached 314 tokens. The same frozen fields were
serialized as fixed-order arrays and compact unordered counts; the data,
representation candidates, token limit, folds, optimizer, thresholds, and
model weights were unchanged. A separate aggregation defect was also found
before interpretation: unavailable SAGE11 target labels produced `NaN` macro
means. They are now explicitly excluded from source-only macro averages. Both
corrections are covered by tests and the full adaptation was rerun.

## Representation selection

Selection used SAGE11 pairs only:

| Representation | Macro Brier | Pair-class accuracy | Output identity |
|---|---:|---:|---:|
| minimal | 0.175610 | 0.480277 | 0.985082 |
| invariant context | **0.170230** | 0.479258 | 0.993086 |
| action-only baseline | **0.117618** | **0.747016** | n/a |

`invariant_context` therefore won the frozen comparison between the two Qwen
views, but neither view beat action identity. The selected context also made
game identity almost perfectly readable at this source-selection stage.

Both views had zero recall for the 19 explicit SAGE11 completion positives.

## Cross-game V4.3 semantics

On all 4,760 V4.3 arms, the selected adapted Qwen representation had:

- seven-effect macro Brier: 0.093676;
- action-only seven-effect macro Brier: 0.067048;
- Brier skill versus action-only: **-0.026628**;
- semantic-output game identity: 0.914286 versus 0.094118 majority.

Important per-effect diagnostics were:

| Effect | Positives | Brier | Recall |
|---|---:|---:|---:|
| changed | 4,438 | 0.118240 | 0.984452 |
| moved | 1,077 | 0.283817 | 0.068709 |
| target created | 188 | 0.059527 | 0.000000 |
| target removed | 619 | 0.159963 | 0.172859 |
| target moved | 8 | 0.008833 | 0.000000 |
| level complete | 3 | 0.001522 | 0.000000 |
| game over | 87 | 0.023833 | 0.000000 |

The adapter learned the very common `changed` marginal but not the rare
productive effects the architecture needs for level solving.

## Interpretation before end-to-end fitting

This checkpoint rejects the direct hypothesis that a frozen Qwen embedding
plus a cheap external low-rank pair adapter is already a better transferable
effect model than action identity. Same-state pairing and coordinate removal
did not eliminate game signatures, and rare effect recall remains the
dominant semantic defect.

It does not yet answer whether the calibrated world model and learned energy
can extract a small amount of decision-relevant signal hidden by macro Brier.
The frozen V4.8 protocol therefore requires the full downstream evaluation to
run and report that answer rather than stopping here.

## Artifacts

- manifest checksum:
  `143fe10e1b35f7fa2dc1dc1078f86beb21042d6a647b6235c1c1ca23bccbce67`;
- embedding cache checksum:
  `0d669aa5736f78af3adc2d467acb1746bc694653f3a352600d55187e14f67345`;
- annotation checksum:
  `ae37faa5290b0d2cd9b978da10c7c589b2f329964f38d38d34ebcec533d60454`;
- semantic result checksum:
  `2b90c09c04a44c3e436e45fe5089c515001d25a9279cc636b9b8706c61895151`.

The machine-readable result is
`training/sage12/semantic_adapter_v4_8/semantic_result.json`.
