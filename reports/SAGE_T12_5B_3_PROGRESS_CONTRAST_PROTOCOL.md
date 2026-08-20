# SAGE.T12.5b.3 — Prospective same-prefix progress contrasts

## Scientific status

T12.5b.2 is a signed, integrity-clean negative audit. It reconstructed 30
local affordances from the sealed T12.5b-r1 branches, preserved unavailable
actions as missing interventions, and found no hard contrast. Causal progress
and raw effect magnitude therefore remained observationally confounded.

T12.5b.3 is a separately frozen prospective source-train collection. It does
not amend, rerun, or retune T12.5b.2. The progress posterior, ordered
milestones, route witnesses, action catalog, detour schedule, branch schedule,
thresholds, and negative-result routes are fixed before any new ARC call.

## Question

> In a new, exactly reproducible context at the same causal progress stage,
> does an observed non-progress action produce a larger structured change than
> the next milestone while the frozen causal posterior still ranks the
> milestone effect higher?

The unit of comparison remains two observed one-step interventions executed
from one exact context. Effects from different states are never composed into
a contrast.

## Bound parent and adaptive target selection

Freeze requires the signed parent status
`FAIL_T12_5B_2_INSUFFICIENT_DISCRIMINATIVE_CONTRASTS`, classification
`INSUFFICIENT_DISCRIMINATIVE_CONTRASTS`, and the exact three failed scientific
checks recorded by T12.5b.2. All parent integrity checks must remain true.

The parent affordance registry is used once, before collection, to select the
prospective target. Stage 3 is the unique closest contest in both lineages:

- the sealed next milestone is produced by `ACTION3`, magnitude 8;
- the largest observed non-progress effect is `ACTION4`, magnitude 6;
- the same result holds in lineages 8701 and 8705.

The target criterion uses sealed milestone labels and raw magnitude only. It
does not use the causal progress score being evaluated. The complete selection
and its input checksums are persisted in the manifest.

## Fixed detour contexts

Starting from the exact stage-3 prefix in each lineage, the collector creates
three prospective contexts by executing `ACTION4` exactly 1, 2, or 3 times.
No depth is selected or dropped after observing an outcome.

A context is admissible only when:

- the original witness and three-step option prefix replay exactly;
- every detour action is locally executable;
- every detour effect matches none of the five sealed milestones;
- no detour is terminal;
- all six scheduled candidate branches reproduce the same exact detour state
  and the same structured prefix.

An inadmissible context remains in the signed trial matrix and contributes no
candidate effect. At least one admissible context must exist in each lineage,
and at least one context identifier must be admissible in both lineages.

## Fixed candidate branches

From each detour context, `ACTION3`, `ACTION4`, and `ACTION6` are attempted
twice. The matrix is therefore:

`2 lineages × 3 detour depths × 3 actions × 2 repetitions = 36 trials`.

`ACTION7` remains excluded because the SDK did not expose it as executable at
the sealed anchors. A locally unavailable action is missing evidence, never a
zero-effect observation. Each admissible context must expose at least two
executable candidates, availability must agree across repetitions, and every
executed effect must be deterministic.

The frozen posterior never selects a branch. All branches execute in the fixed
order independently of any causal or magnitude ranking.

## Semantic progress binding and hard contrasts

The progress affordance is whichever local action matches the sealed stage-3
milestone signature. Its action name is provenance only. Cross-lineage
bindings use exactly `(stage, milestone_signature)`; hashes, coordinates,
grounded identities, and action-name equality are excluded.

For a local progress affordance `p` and distractor `d`, a hard contrast
requires:

1. `p` matches the stage-3 milestone and `d` does not;
2. both effects are observed and deterministic from the same exact context;
3. `magnitude(d) >= magnitude(p) + 1`.

Only after registration is the frozen causal score inspected. Success requires
`progress_gain(p) > progress_gain(d)` on every hard contrast.

## Gates and negative-result routes

The positive gate requires:

- exact original-prefix replay, complete fixed schedule, exact repetitions,
  deterministic contexts/effects, zero terminal failures, and all bounds;
- at least one admissible context per lineage and one shared admissible context;
- a transported semantic progress affordance;
- at least one hard contrast in each lineage and one shared contrast context;
- causal hard-contrast accuracy 1.0;
- causal-minus-magnitude hard-contrast accuracy at least 0.5.

Outcomes are classified without retuning:

- integrity failure → `FAIL_T12_5B_3_COLLECTION_INTEGRITY_GATE`;
- no shared reproducible neutral detour → `FAIL_T12_5B_3_DETOUR_CONTEXT_GATE`;
- no transported progress affordance → `FAIL_T12_5B_3_PROGRESS_AFFORDANCE_GATE`;
- no prospective hard contrast →
  `FAIL_T12_5B_3_INSUFFICIENT_DISCRIMINATIVE_CONTRASTS`;
- observed contrasts but inadequate causal ranking →
  `FAIL_T12_5B_3_DISCRIMINATION_GATE`;
- full pass → `PASS_T12_5B_3_PROSPECTIVE_CONTRAST_GATE`.

No same-version rerun, threshold change, action removal, or post-hoc detour
selection is authorized after a miss.

## Bounds and claim boundary

- maximum 3,500 SDK calls;
- maximum 7,200 wall-clock seconds;
- maximum 3 GiB of artifacts;
- no raw frame persistence;
- source-train game `bp35` only;
- lineages 8701 and 8705 only.

A pass authorizes only a separately frozen T12.5c paired-control protocol.
It does not authorize that control run. Source validation, holdout, neural
training, target-game transfer, production use, and controller authority remain
closed in every T12.5b.3 outcome.
