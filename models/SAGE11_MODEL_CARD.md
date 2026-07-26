# SAGE.11 graph world model — model card

Status: architecture implemented and software-validated; untrained. No
checkpoint is promoted by this change.

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
2. Cheap effect-predictability pilot improves macro-F1 by at least 0.05 over
   the per-action majority baseline; otherwise revisit labels/features.
3. Terminal head remains disabled until 100 strong terminal/level events.
4. All tuning is restricted to the 11 source-train and three
   source-validation games.

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

The current random initialization is not useful for acting. There is no claim
of cross-game competence, score gain, or holdout generalization until a
checksummed checkpoint and the required paired reports are published.
