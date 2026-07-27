# SAGE12 invariant target-mechanic replication V4.2

Date frozen: 2026-07-27

Status: `FROZEN_BEFORE_SOURCE_PREFLIGHT`

Manifest checksum:
`fba242f31cbc492f44333bcae8c5f9228baee0b79c15f0c68009dce7a76a6210`

## Question and authority

V4.2 asks whether eight contiguous semantic transitions predict the next
target effect beyond action identity after game-specific anchor detail is
removed.

The authoritative effects are exactly `target_created`, `target_removed`,
and `target_moved`. `actor_displaced` remains an audit-only count and is
absent from model views, prompts, rule induction, calibration, macro metrics,
gates, and V5 authority. V4.2 can authorize only a separately frozen V5
protocol for the three target effects. It cannot fit a world model, EBM, or
controller.

## Representation repair

The public anchor vocabulary is frozen to:

- `occupied`: prior `occupied_actor` or `occupied_object`;
- `free`: prior `empty` or `open`;
- `none`: prior `targetless` or `unknown`.

The V4.1 causal role tracker remains audit-only. A private compatibility
adapter pads the old four-effect engine with `actor_displaced=false` and
`applicable=false`; no actor outcome reaches the model. V4 and V4.1 formats,
code paths, artifacts, and checksums remain independently reproducible.

Design used only the already opened V4.1 source-training preflight. On its
1,911 windows, the coarse representation reduced static game-identity gain
from +0.1293 to +0.0387. For the three target effects, calibrated structured
Brier was 0.0479 versus 0.0586 for the stronger action baseline, macro-F1 was
0.6891, and context Brier skill was +0.4328. These values motivate V4.2 but
are not prospective evidence.

## Data and firewall

Source development re-derives eight-transition windows from the immutable
3,040 V3 source-training traces. Windows are game-local, reset-local,
frame-contiguous, horizon one, and deduplicated only by complete audit digest.
The old V3 validation outcomes and V4 prospective outcomes are non-gating.
Holdout, historical, and `ar25` data remain closed.

Game identity, seeds, indices, coordinates, values, colours, object IDs, raw
grids, hashes, query outcomes, actor labels, and actor-role state remain
outside the model view.

## Frozen source preflight

Prospective collection is forbidden unless every check passes:

- at least 1,500 unique windows;
- at least 75 positives and 75 negatives for every target effect;
- static action/family/coarse-anchor identity gain at most +0.05 over action;
- calibrated leave-one-source-game-out macro-ECE at most 0.10;
- calibration macro-Brier degradation at most 0.005;
- calibrated Brier skill at least +0.10 over the stronger calibrated
  local-action, global-action, or deterministic template baseline;
- calibrated macro-F1 gain at least +0.05;
- eight-transition context Brier skill at least +0.10 over no context;
- maximum complete Qwen prompt at most 384 tokens;
- model-view firewall and actor exclusion pass.

A failure is published as `FAIL_SOURCE_TRAIN_PREFLIGHT` and ends V4.2.

## Prospective collection

After a passing preflight only, collect exactly 768 fresh transitions: 256
each from `re86`, `ls20`, and `sc25`, using unused policy seeds 479, 523, 569,
and 617. Each reset has at most 32 actions and each game at most 24 resets.
Legal actions are balanced, selection is never outcome-adaptive, and
chronological repeats remain in the audit stream.

## Structured evaluation

The structured branch passes only when all checks succeed:

- at least 500 windows and 30 positives/negatives per target effect;
- strict JSON, grounding, and `support=0` exactly 1.00;
- raw and calibrated macro Brier skill at least +0.10;
- run-cluster bootstrap 95% lower bound above zero;
- calibrated macro-F1 gain at least +0.05;
- outcome-shuffle skill loss at least 0.05;
- anchor-binding-shuffle skill loss at least 0.02;
- context Brier skill at least +0.05 over no context;
- non-negative Brier skill in every validation game;
- calibrated macro-ECE at most 0.10;
- prospective identity gain at most +0.05;
- every target effect has best-method Brier at most 0.10, F1 at least 0.20,
  and at least 30 positives and negatives.

Baselines, Platt mappings, thresholds, bootstrap seed, and the stronger
baseline selection are fitted or chosen from source data only.

## Separate Qwen authority

Qwen2.5 0.5B runs on the already benchmarked `cuda:0` device with temperature
zero, a 512-token runtime input cap, and 256 output tokens. The compact schema
allows only the three anchors, three effects, exact/family scope, at most
eight rules, and `support=0`.

The 128 contexts are selected without outcomes by game, action, anchor, and
window digest. Qwen passes separately only with JSON and grounding at least
0.95, `support=0` exactly 1.00, productive-effect recall@8 at least 0.70,
outcome-shuffle loss at least 0.05, and non-negative per-game skill. CUDA or
runtime failure closes only Qwen authority and does not change the structured
verdict.

## Execution and immutable publication

```powershell
python -m theory.sage12.target_mechanic_replication preflight
python -m theory.sage12.target_mechanic_replication_collection
python -m theory.sage12.target_mechanic_replication evaluate
```

Publish four direct checkpoints on `main`:

1. implementation, tests, protocol, and frozen manifest;
2. source windows, priors, calibration, token audit, and preflight;
3. prospective collection manifest and raw shards, if authorized;
4. predictions, Qwen outputs, controls, result, tests, and documentation.

No code, threshold, seed, prompt, effect, calibrator, or gate may change after
its outcome boundary is opened. Every pass or failure is published.

## Executed source preflight

The frozen preflight completed `PASS_SOURCE_TRAIN_PREFLIGHT` with all 11
gates passing, checksum
`68747717f45289775cd543aaa027eb24164200b255b42b57368e4c6fba0816ff`.
It derived 1,911 windows; static identity gain was +0.0387, calibrated Brier
skill +0.1821, macro-F1 gain +0.0749, context skill +0.4328, macro-ECE
0.0365, and Qwen prompts 295–317 tokens. Prospective collection is now
authorized under the unchanged manifest. Full ledger:
`reports/SAGE12_TARGET_MECHANIC_REPLICATION_V4_2_PREFLIGHT.md`.

## Executed prospective collection

The authorized collector produced exactly 768 transitions, 256 per game,
under report checksum
`6bdec774c744061e3e5014ced8d3d0191d1cdc13243130817ea9ec84fd50dce7`.
All games used eight resets, legal-action counts were balanced, the policy
was outcome-independent, and 91 chronological exact repeats were retained.
No prospective metric or Qwen output was computed before publishing the raw
shards. Collection ledger:
`reports/SAGE12_TARGET_MECHANIC_REPLICATION_V4_2_COLLECTION.md`.

## Executed evaluation result

The frozen evaluator stopped at `FAIL_RUNTIME_CLOSED` while serializing a
generic rule with internal anchor `any`. The public compatibility map covered
only the three concrete anchors, causing `KeyError('any')` after outcomes
were opened. The protocol therefore forbade a patch or rerun. No structured
or Qwen verdict is available and no authority was granted. Failure checksum:
`17934d7b576ac11c36abcac6235e7bc259247f225f49edf5e05126971390be6a`.
Full result:
`reports/SAGE12_TARGET_MECHANIC_REPLICATION_V4_2_RESULT.md`.
