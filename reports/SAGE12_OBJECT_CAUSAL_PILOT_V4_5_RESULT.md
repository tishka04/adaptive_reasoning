# SAGE12 V4.5 rooted intervention-event result

Status: **FAIL_CLOSED at source-only feasibility**

- manifest checksum:
  `cfae89ac0de9f263af52dbb042e352869324f301a633012d44ad7b85ec028741`
- feasibility checksum:
  `1e19f7df0cdb315dc473e9a430c29e0cd29feda562e69d7a873d4d289b4099e6`
- event-vocabulary checksum:
  `fb52747ce17b074bd82b67eeeeb311358ce51570fb03c99a08a162caf8ac93e8`
- derived-delta SHA-256:
  `db2d2ed284dd11650af65fa2ad57d6c3f790a8c13e1bb6299ab347c16ba0cc17`
- final closure checksum:
  `1e162e1b03afba04e49b58abce066fb2b68e5fb8f2ebef6022d15abe2dc56b17`

No fresh source trace or validation shard was created. No source-validation
game was opened. Qwen, GNN, world-model, EBM, and controller authority remain
closed.

## Outcome

V4.5 found abundant source event capacity but did not find a sufficiently
grounded or transferable causal representation. Both the compiler gate and
the predictive gate failed.

### Compiler quality

| Metric | Requirement | Result | Gate |
|---|---:|---:|---|
| Identical pair pre-state | 1.0000 | 1.0000 | pass |
| Confident correspondence rate | ≥0.9000 | 0.9840 | pass |
| Ambiguity rate | ≤0.1000 | 0.0160 | pass |
| Root grounding rate | ≥0.9000 | 0.8130 | fail |
| Exclusive-event localization | ≥0.9000 | 0.8370 | fail |
| Promoted events | ≥2 | 11 | pass |

The confidence and ambiguity numbers describe how decisively the deterministic
matcher selected an assignment; they do not prove that the assignment is the
correct causal object. The failed localization gate is the stronger audit:
16.3% of cells assigned to intervention-exclusive events were absent from the
actual left-versus-right post-state differential.

All 11 promoted tokens were local or collateral. No direct target event
survived the source capacity rule. The vocabulary was dominated by large
appearance/disappearance and long east/west displacement events. This is the
signature of whole-layout or segmentation dynamics, not a stable manipulated
object vocabulary.

## Prediction

The strongest baseline was the root graph without history.

| Method | Macro accuracy | Macro Brier | Macro ECE |
|---|---:|---:|---:|
| Structured root + history | 0.7306 | 0.1448 | 0.1087 |
| Root without history | 0.7267 | 0.1346 | 0.0871 |
| Action only | 0.7244 | 0.1382 | 0.0874 |
| Action + history, no root | 0.7097 | 0.1466 | 0.1037 |
| Deterministic template | 0.4759 | 0.2349 | 0.1052 |

The structured model achieved:

- macro-Brier skill `−0.0757` versus the required `+0.10`;
- macro-accuracy gain `+0.0039` versus the required `+0.10`;
- root-swap accuracy drop `−0.0039` versus the required `+0.10`;
- relation-shuffle accuracy drop `−0.0222` versus the required `+0.10`;
- game-identity gain `+0.2287` versus the maximum `+0.05`;
- macro ECE `0.1087` versus the maximum `0.10`;
- paired-bootstrap lower 95% bound `−0.0120`;
- exact complete-arm inversion error `1.54e-16`, which passed.

The root-swap control was rerun after correcting the implementation to swap
only root semantics while preserving action identity. Both root swapping and
relation shuffling improved rather than degraded accuracy. The exact
antisymmetry control passed, so this negative sensitivity is a property of the
compiled representation rather than the pairwise model implementation.

Per-game transfer was negative on `dc22` (`−0.0199`) and `sp80` (`−0.1383`).
`ka59` had no scoreable promoted event. The apparent gains on `bp35` and
`su15` did not transfer across the source split, and the bootstrap interval
included zero.

## Interpretation

V4.5 rejects the current combination of:

1. independent connected-component segmentation in each frame;
2. score-based pre/post object matching;
3. event discovery over those matched components;
4. the deterministic two-hop target-rooted graph.

It does not reject the higher-level SAGE12 architecture. The pilot stopped
before any semantic language model, world model, energy model, or controller
was trained. It shows that a GNN over these compiled objects would inherit
incorrect roots, layout-wide event labels, and strong game signatures.

The next representation experiment should start from the direct
left-versus-right changed-cell mask, form intervention-exclusive regions
before object matching, and require those regions to connect to the action
argument or a verified actor path. A manually audited or synthetic
correspondence set should measure correctness rather than matcher confidence
before another predictive gate is attempted.

## Fail-closed ledger

- fresh source collection:
  `SKIPPED_FEASIBILITY`,
  checksum
  `6204eb5357f6aed6cd1960301b5b335937baacf984e3862d755a143b19c1e840`;
- fresh source preflight:
  `SKIPPED_FEASIBILITY`,
  checksum
  `310a93015b29b56ee26499129312d60ec60952b51574e961b7217f116c80c4a6`;
- validation collection:
  `SKIPPED_SOURCE_PREFLIGHT`,
  checksum
  `4cfbb170819e7c844e19c05142e5e0504c8bdf8657e7968c76b38de5e0519e15`;
- final evaluation:
  `SKIPPED_VALIDATION_COLLECTION`.

The frozen design is in
`reports/SAGE12_OBJECT_CAUSAL_PILOT_V4_5_PROTOCOL.md`.
