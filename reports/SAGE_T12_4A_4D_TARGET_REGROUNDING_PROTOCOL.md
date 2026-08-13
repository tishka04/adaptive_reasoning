# SAGE.T12.4a.4d — Target-local re-grounding protocol

## Scientific question

T12.4a.4 showed that the five-action level-0 option does not transfer to the
common level-1 entry. T12.4a.4b localized the miss to initiation and dynamics,
and T12.4a.4c compiled that negative result as sparse initiation guards and
typed step-effect contracts. T12.4a.4d asks a narrower question:

> Can the failed contract be used to re-ground exploration locally at the
> exact level-1 entry and discover a replay-confirmable level-1-to-level-2
> progress witness?

This is a source-train discovery experiment. It is not a transfer, validation,
holdout or production experiment.

## Frozen inputs

The freeze binds by SHA-256:

- the passed T12.4a.4c manifest and receipt;
- all 24 contracted causal-program particles and their posterior snapshot;
- the contracted option registry with six initiation guards and five effect
  contracts;
- the two T12.4a.2 exact routes (seeds 8701 and 8705) to the same level-1 hash;
- the multi-step terminal shield and its passed T12.3e lineage/shield receipt.

The old option remains unavailable at the target anchor. Its posterior
applicable mass and materialized action count are recorded as a shadow control
before each search arm.

## Paired design

Fresh search seeds are 9101, 9102 and 9103. Each seed is crossed with both
exact route lineages. Every condition runs two arms:

1. `local_archive_control`: lineage-preserving symbolic Go-Explore with the
   frozen terminal shield;
2. `contract_regrounded`: the identical archive, action catalogue, replay
   anchor, shield, burst schedule and budget, with deterministic action
   reranking from contract mismatch and branch-local object roles.

The candidate actions are not generated differently in the treatment. The
treatment may only reorder the exact same grounded catalogue. Absolute
coordinates are used only to ground an already available branch-local entity;
they are never stored in the transferable contract.

The burst schedule is 4/8/16. Each arm receives at most 2,048 SDK calls and 64
excursions. The complete run is capped at 26,000 SDK calls, 10,000 cells per
archive and exactly 3 GiB of artifacts. Raw frames are not persisted.

## Primary gate

The primary T12.4a.4d gate passes only if all of the following hold:

- both arms start from the expected exact level-1 hash on every lineage/seed;
- every archive restoration is exact;
- paired arms expose the same initial grounded-action catalogue;
- the contracted option is blocked at every target anchor;
- all per-arm and total SDK limits hold;
- terminal failures remain at or below 10% of explored transitions;
- one bounded local suffix reaches a higher level;
- that suffix is replayed successfully twice from seed 8701 and twice from
  seed 8705;
- all four confirmations use exact prefixes, execute every suffix action,
  avoid terminal failure and reach one common final exact hash.

A pass authorizes only a separate T12.4a.4e freeze to minimize and compile the
new option. It does not authorize the old option, active neural control,
validation, holdout access or production control.

## Secondary guidance claim

Contract guidance is evaluated separately from witness discovery. The
secondary claim is authorized only if the treatment reaches first progress
where its paired control does not or reaches it in fewer SDK calls. At least
one paired win, no paired loss and no higher aggregate terminal-failure rate
are required.

If the control also finds the selected witness with equal or better efficiency,
the primary witness gate may pass but no causal-guidance advantage is claimed.
This prevents a generic Go-Explore success from being misattributed to the
T12.4a.4c contracts.

## Negative-result policy

Any replay mismatch, catalogue mismatch, budget violation, unblocked old
option, excessive terminal rate or unconfirmed witness fails closed. A clean
search miss is also a negative result. In either case T12.4a.4e remains closed;
there is no threshold retuning, seed substitution, holdout opening or automatic
rerun under this protocol.
