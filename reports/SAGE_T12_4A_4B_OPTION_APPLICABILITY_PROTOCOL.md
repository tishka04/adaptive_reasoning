# SAGE.T12.4a.4b — Option applicability audit

## Scientific question

T12.4a.4 established a clean negative result: the five-action option is
deterministic and available at level 1, but it produces no progress in any of
four exact-prefix repetitions. T12.4a.4b asks **which contract failed to
transfer**. It does not retune the option and does not test policy control.

The frozen alternatives are:

1. `INITIATION_AND_DYNAMICS_SHIFT`: the anchor representation and first
   structured effect differ;
2. `DYNAMICS_CONTEXT_SHIFT`: at least one structured transition effect differs;
3. `INITIATION_GOAL_CONTEXT_SHIFT`: effects match, but the structural initiation
   context and completion outcome differ;
4. `TERMINATION_PREDICATE_CONTEXT_SHIFT`: anchors and effects match, but the
   completion predicate does not transfer;
5. `REPRESENTATION_INSUFFICIENT`: pixels change while the current object-centric
   state exposes no mechanism change.

An integrity failure or failure to reproduce the original contrast is not a
sixth scientific explanation; it fails the audit closed.

## Frozen design

- Game and split: `bp35`, `source_train` only.
- Lineages: 8701 and 8705.
- Contexts: the two sealed level-0 initiation states where the option succeeds,
  and the common level-1 entry where T12.4a.4 failed.
- Branches: the complete option and a null control.
- Repetitions: two per context, lineage and branch.
- Total: 16 exact-prefix trials.
- Maximum ARC SDK calls: 1,200.
- Maximum persisted data: 3 GiB.
- Persisted state: bounded object-centric descriptors and structured deltas;
  raw frames are forbidden.

The object-centric comparison removes grounded entity identifiers and direct
level/terminal fields from the mechanism signature. It retains entity roles,
attributes, relations, topology and non-level counters. Full signatures remain
available for the separate goal/termination comparison.

## Gate

The audit passes only if all 16 prefixes are exact, all actions are available,
repetitions are deterministic within each lineage, null controls preserve their
anchors, the level-0 option progresses four times, the level-1 option progresses
zero times, no terminal failure occurs, budgets hold, and exactly one frozen
diagnosis is selected.

A pass grants no neural training, active evaluation, option control, validation,
holdout or production authority. It can authorize only one diagnostic child:

- an option initiation/effect-contract freeze when the symbolic representation
  is sufficient; or
- a representation-extension freeze when it is insufficient.

The checksum-linked manifest, trials, diagnosis, report and receipt make the
decision auditable. A failed gate remains a scientific negative result and
stops the chain.

