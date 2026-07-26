# SAGE.11 graph world model — model card

Status: architecture implemented and software-validated; untrained. The
original joint-target pilot failed. The separately pre-registered factorized
v2 pilot passed its cheap gate on 2026-07-26, but its 77-feature streaming
interface is not the current graph model's 19-atom interface. Training remains
blocked until that interface mismatch is resolved; no checkpoint is promoted.

## Model

The default model has 1,540,953 trainable parameters, below the strict
5,000,000-parameter limit. It uses a permutation-invariant typed-atom graph
encoder and five independently initialized bootstrap heads. Inputs are
structural state atoms plus a six-value action/argument vector. Outputs are
next-state latent, symbolic effect class, changed, progress, terminal, risk,
and no-op predictions with bootstrap variance.

The model format is `sage11-world-model-v1`. It rejects legacy M2/v4 state
loading. The concrete trainer uses bootstrap-resampled head losses, gradient
clipping, and checkpoint metadata containing the split and dataset-manifest
checksums. The encoder is frozen during target-side online adaptation.

## Intended use

The model ranks legal, symbolically admissible counterfactual actions. It does
not assert facts, award support, override observed danger, or replace the
symbolic controller. Every bridged hypothesis begins with `support=0`.

## Training preconditions

1. Checksummed source-only dataset with at least 100,000 transitions.
2. Cheap effect-predictability pilot improves macro-F1 by at least 0.10 over
   its frozen primary baseline (per-action majority in v1, learned
   action-only in v2); otherwise revisit labels/features.
3. The trained model must consume the same versioned input/target interface
   that passed its cheap pilot. Pilot v2 does not authorize training the old
   19-atom/joint-class interface.
4. Terminal head remains disabled until 100 strong terminal/level events.
5. All tuning is restricted to the 11 source-train and three
   source-validation games.

## Cheap-pilot evidence

The fixed CPU `HistGradientBoostingClassifier` used 76,908 source-training
rows and evaluated once on 23,092 rows from `re86`, `ls20`, and `sc25`.
Inputs were 19 train-fitted binary pre-action typed atoms plus the existing
six-value action vector; game identity, outcomes, policy arm, historical data,
and holdout data were excluded.

Overall classifier macro-F1 was 0.0779 versus 0.0490 for the train-only
per-action majority baseline, a gain of 0.0288 rather than the required 0.10.
Per-game gains were -0.0070 on `re86`, -0.0543 on `ls20`, and +0.0234 on
`sc25`; all failed. Action shuffling degraded overall macro-F1 by only 0.0059.
All validation effect labels and typed atoms existed in training, so unseen
vocabulary does not explain the no-go.

The machine-readable result checksum is
`c724aeb6d2ab71154a7c72fa381f3f5f4347a5135644ba64ac82a5542e528136`.
See `reports/SAGE11_EFFECT_PILOT_RESULT.md`. The graph architecture remains a
software artifact only until a separately pre-registered representation or
label revision passes a new cheap pilot.

### Factorized pilot v2

Protocol v2 was pushed in commit `2660f4b` before fitting. It replaced the
joint class with independently scored changed-cells and player-moved heads,
replaced raw coordinates with categorical/topological action features, and
added leakage-free streaming context. Its learned action-only comparator used
10 features; the full representation used 77.

The full composite reached 0.5506 macro-F1 versus 0.3431 action-only, a
+0.2075 gain. Both core heads were non-negative and all three validation games
improved against action-only, so the frozen v2 gate formally passed. Result
checksum:
`45f58d1537a1b1a6800636b77df401ab3bf1f94f4ed6dc3bcf2d107864f0328f`.

The pass is qualified. Player-moved supplied nearly all the gain (0.9450 F1,
+0.3720), while changed-cells reached only 0.1562 and remained 0.0154 below
the per-action majority. Current-action shuffling degraded the composite by
only 0.0078 and argument shuffling by approximately zero. The later
world-model action-shuffle gate therefore remains unmet, and the result may
largely reflect implicit game-regime signatures in near-constant atoms.

See `reports/SAGE11_EFFECT_PILOT_V2_PROTOCOL.md` and
`reports/SAGE11_EFFECT_PILOT_V2_RESULT.md`. V2 permits implementation of its
input interface, not training or promotion of the unmodified graph model.

## Required gates

- change-weighted next-state accuracy beats persistence;
- at least a 15 percentage-point gain on changed transitions;
- action shuffling degrades performance by at least 10%;
- effect macro-F1 is at least majority macro-F1 + 0.10;
- risk and no-op ECE are each at most 0.10;
- latent feature standard deviation is at least 0.01;
- validation games are exactly source-side games.

Passing these gates permits shadow evaluation only. Bounded mode additionally
requires byte-identical shadow actions, zero hypothetical preemption of a route
that later succeeds, pre-registered top-1/top-3 productivity improvement,
calibration, and inference-cost compliance. Active mode requires the complete
bounded protocol and final holdout promotion gate.

## Safety and limitations

ECE does not bound tail risk, so the model's risk output is advisory. Symbolic
danger memory is a hard veto in every authority mode. Protected competence
always retains authority. One neural probe is allowed per branch/context,
followed by immediate return to symbolic control. Two non-productive probes
demote the context until context change, a new confirmed effect, route
refutation, or level change.

The current random initialization is not useful for acting. Pilot v1 is
evidence against the original representation/target pairing. Pilot v2
demonstrates factorized predictability but not robust changed-cell or
current-action sensitivity. There is no claim of a useful learned world
model, cross-game competence, score gain, or holdout generalization.
Historical and holdout games remain untouched.
