# SAGE12 target-mechanic runtime recovery V4.2.1

Date frozen: 2026-07-27

Status: `FROZEN_BEFORE_SOURCE_REHEARSAL`

Manifest checksum:
`81f14c655dc6b824970b2ecd8638ca62360abedcc7f4dcf3abed2b86cdd3a3c8`

## Purpose and authority

V4.2.1 is a clean, separately versioned recovery of V4.2's runtime failure.
V4.2 produced no prospective metric verdict because its public serializer
could not encode the structured engine's generic `any` rule anchor. V4.2 and
its opened shards remain immutable and non-authoritative.

V4.2.1 changes only the public structured-rule contract and artifact
transaction order. It does not change the scientific question,
representation, source/validation game split, target effects, model,
calibration, baselines, controls, thresholds, Qwen decoding, or promotion
gates. A complete structured pass may authorize only a separately frozen V5
protocol for target creation, removal, and movement. It cannot fit a world
model, EBM, or controller.

## Minimal repair

Observed state and query anchors remain exactly `occupied`, `free`, and
`none`. Public structured rules additionally admit `any`, matching the
generic rule already used by the unchanged V4/V4.1 inference engine.

The public serializer and inverse loader must round-trip every generated
rule, including exact/`any` and family/`any`. The Qwen schema and prompt are
unchanged from V4.2 and continue to expose only concrete anchors. Qwen cannot
emit `any`; this repair therefore adds no new LLM hypothesis.

## Mandatory source rehearsal

Before the ordinary source preflight, a source-only rehearsal must:

1. derive the same 1,911 windows from immutable V3 source-training traces;
2. enumerate every rule for every unique query and target effect;
3. public-serialize and restore every rule with a 100% round-trip rate;
4. explicitly exercise exact/`any` and family/`any`;
5. run the complete structured prediction/evidence writer over all 1,911
   source windows;
6. require at least one selected evidence rule using `any`;
7. publish the predictions, SHA-256 checksum, checks, and rehearsal checksum.

Failure stops before source validation or prospective collection. A passing
rehearsal is a new conjunctive source-preflight gate.

## Unchanged source preflight and gates

The source preflight re-derives source windows, priors, leave-one-game-out
Platt calibration, thresholds, identity leakage, context utility, model-view
firewall, and Qwen token budget. In addition to the rehearsal checks, all
V4.2 scientific gates remain unchanged:

- at least 1,500 source windows and 75 positives/negatives per target effect;
- static identity gain at most +0.05 over action identity;
- source macro-ECE at most 0.10 and calibration Brier degradation at most
  0.005;
- Brier skill at least +0.10, macro-F1 gain at least +0.05, and context
  Brier skill at least +0.10;
- complete Qwen prompts at most 384 tokens;
- actor exclusion and model-view firewall pass.

## Fresh prospective collection

Only a passing rehearsal and preflight may authorize collection. V4.2.1
collects exactly 768 new chronological transitions: 256 each from `re86`,
`ls20`, and `sc25`, using unused policy seeds 661, 709, 757, and 809.
Legal-action balancing, 32 actions per reset, 24-reset maximum, retained
chronological repeats, and outcome-independent selection are unchanged.

No V4.2 prospective shard may be copied, relabelled, or used for a V4.2.1
gate. Holdout, historical, and `ar25` outcomes remain closed.

## Transactional prospective evaluation

After building validation windows, the evaluator computes the unchanged
structured metrics and controls. It must then write and checksum, in order:

1. every structured prediction and its evidence;
2. a complete `structured_intermediate.json` verdict;
3. the Qwen clean and outcome-shuffled streams;
4. the combined final result.

This ordering guarantees that a later Qwen or final-serialization failure
cannot erase the structured verdict. Any uncaught runtime error writes a
checksummed `runtime_failure.json` containing the stage, exception, available
artifact hashes, and explicit revocation of V5, world-model, and EBM
authority. A Qwen-only error is recorded in the separate Qwen branch and does
not alter the structured verdict.

## Unchanged prospective gates

The structured branch retains all V4.2 requirements:

- at least 500 windows and 30 positives/negatives per target effect;
- JSON, grounding, and `support=0` exactly 1.00;
- raw and calibrated Brier skill at least +0.10;
- positive run-cluster bootstrap 95% lower bound;
- macro-F1 gain at least +0.05;
- outcome-shuffle loss at least 0.05;
- anchor-binding-shuffle loss at least 0.02;
- context Brier skill at least +0.05;
- non-negative transfer in every game;
- macro-ECE at most 0.10 and identity gain at most +0.05;
- every target effect meets its capacity, Brier, and F1 authority checks.

Qwen remains a separate 128-context diagnostic with the same local
Qwen2.5-0.5B weights, `cuda:0`, temperature zero, prompt/schema, token caps,
selection, shuffle, and gates as V4.2.

## Execution and immutable checkpoints

```powershell
python -m theory.sage12.target_mechanic_recovery rehearsal
python -m theory.sage12.target_mechanic_recovery preflight
python -m theory.sage12.target_mechanic_recovery_collection
python -m theory.sage12.target_mechanic_recovery evaluate
```

Publish direct checkpoints on `main` before crossing each boundary:

1. implementation, tests, protocol, and frozen manifest;
2. source rehearsal and predictions;
3. source windows, priors, calibration, and passing/failing preflight;
4. fresh collection manifest and shards, only if authorized;
5. predictions, structured intermediate, Qwen artifacts, final result or
   automatic runtime failure, tests, and documentation.

Once fresh prospective outcomes are opened, no code, prompt, seed, threshold,
calibrator, baseline, control, or gate may change. Every pass or failure is
published. World-model and EBM fitting remain forbidden throughout V4.2.1.

## Executed source rehearsal

The frozen rehearsal completed `PASS_SOURCE_REHEARSAL`. It serialized all
1,911 source predictions, round-tripped all 168 enumerated rules, covered 42
exact and 42 family `any` rules, and serialized 2,120 selected generic-rule
evidence entries. All seven checks passed without opening source validation.
Rehearsal checksum:
`cd2164ecdfab094d99364cfdec213767987e974e9fd5b4dc01f98db423873b92`.
The source preflight is now authorized; no prospective collection is yet
authorized.

## Executed source preflight

The source preflight completed `PASS_SOURCE_TRAIN_PREFLIGHT` with all 14
conjunctive gates passing. It retained 1,911 source windows; identity gain was
+0.038723, calibrated Brier skill +0.182060, macro-F1 gain +0.074908,
context skill +0.432771, macro-ECE 0.036452, and Qwen prompts 295-317 tokens.
Preflight checksum:
`4ce44b0a0eacaa041106813649d6782be44c21790385c31fda03dbe605abecdb`.
Only the frozen fresh collection is now authorized.
