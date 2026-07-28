# SAGE12 V4.4 paired causal-contrast result

Status: **FAIL_CLOSED at source preflight**

- source preflight checksum:
  `d58cc5825ab0932629496a1206b478ce310be88136e21e4b8799cc902dc18935`
- validation-collection closure checksum:
  `888e07cbb5402c14867fd95bc7b5ebb3d845efe3354a66f320c3560f17d250d2`
- final closure checksum:
  `29b1ad1ef81435b359c28e94ce6dff9ee1aaf7edadb95cd007bc8560b08953ca`

No validation shard was created and no source-validation game was opened.
No world model, Qwen model, GNN, EBM, or controller was fit or executed.

## Outcome

The pairwise reformulation did not rescue the current binding representation.
All capacity and source-integrity gates passed:

- 172 creation-discordant pairs;
- 189 removal-discordant pairs;
- two source games with at least ten discordant pairs for each effect;
- movement remained diagnostic-only with zero discordant pairs;
- the V4.3 source checksum, JSON contract, and support-zero semantics passed.

Every projection nevertheless failed all predictive, binding-sensitivity,
calibration, identity, per-game, and bootstrap gates:

| Projection | Brier skill | Accuracy gain | Binding-swap accuracy drop | Identity gain | ECE | Bootstrap lower |
|---|---:|---:|---:|---:|---:|---:|
| minimal | -0.0432 | -0.1049 | -0.2645 | +0.0755 | 0.1794 | -0.1563 |
| relational | -0.0337 | +0.0143 | -0.1160 | +0.0906 | 0.1246 | -0.0547 |
| typed | -0.0397 | -0.0211 | -0.1746 | +0.1565 | 0.1250 | -0.0984 |

The frozen requirements were +0.10 Brier skill, +0.10 directional-accuracy
gain, +0.10 binding-swap degradation, identity gain at most +0.05, ECE at most
0.10, and a positive bootstrap lower bound. The stronger baseline for every
projection was action plus shared history without binding.

## Controls

The exact structural control passed: complete arm swapping inverted every
model probability to floating-point error between `1.39e-16` and `1.67e-16`.
The antisymmetric implementation therefore behaved as specified.

The causal binding control failed in the opposite direction. Swapping only
the two bindings improved accuracy by 0.116 to 0.265 depending on projection.
This is stronger evidence than a merely insensitive shuffle: the frozen
binding descriptions systematically encouraged the wrong cross-game
direction once action and shared temporal evidence were controlled.

Scoreable-game transfer also failed. Effect evidence was concentrated in
`g50t`, `su15`, and `tu93`, with only three removal examples in `lp85`.
The structured model was negative versus the stronger baseline in important
folds, including `tu93`; the paired bootstrap lower bound remained negative
for every projection.

## Interpretation

V4.4 directly tested the most favorable use of the V4.3 corpus: comparing two
executed interventions from an identical verified state while cancelling
shared background. The negative result therefore rejects more than the V4.3
absolute Beta learner. It rejects the current manually engineered
`BindingSignature` as a transferable causal discriminator for creation and
removal under this source split.

It still does not evaluate a semantic world model, energy model, or controller:
the protocol stopped before those stages. Nor does it rule out richer
persistent object-relative events. The evidence instead points away from
coarse occupancy/relation/shape buckets and toward discovering the actual
changed object and intervention equivalence class from paired before/after
graphs.

## Consequence

No additional rows should be collected under V4.3/V4.4. A successor must
first learn or induce source-only event correspondences that distinguish:

- the manipulated object from nearby decoys;
- object-relative argument identity across both arms;
- transformation, creation, removal, and displacement of matched components;
- mechanic-specific intervention families rather than global binding buckets.

That representation must demonstrate source LOGO pairwise utility and low
identity leakage before any new validation collection or world-model protocol
is considered.

The frozen design is in
`reports/SAGE12_PAIRWISE_CAUSAL_PILOT_V4_4_PROTOCOL.md`.

Final validation passed 48 focused V4.4/V4.3/V4.2 tests in 24.44 seconds and
focused Ruff checks. It reloaded all closure artifacts and confirmed that
neither `source_model.json` nor a validation shard directory exists.
