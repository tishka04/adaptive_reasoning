# SAGE.T12.5 — Goal-relative causal progress

## Scientific question

T12.5 tests whether SAGE can recognize that a state is closer to a solution
without using a terminal event, raw novelty, an action label, a pixel hash or a
level identity as the progress signal.

The hypothesis is that useful progress is history-dependent. The relevant
belief state is therefore `(causal state, progress stage)`, where the stage is
advanced by a sequence of typed causal effects. A frame alone is not assumed to
contain a universal distance-to-goal.

## Sealed evidence

The phase is a checksum-bound child of two passed experiments:

- T12.4a.4c supplies five identity-free typed effect contracts, the 24 complete
  dynamics-plus-goal owner programs and their posterior masses;
- T12.4a.3 supplies all 64 exact-prefix subsequences and the reversed control,
  repeated on lineages 8701 and 8705.

The structured success and failed-context traces from T12.4a.4b are also bound
through the T12.4a.4c manifest. They are essential because they apply the same
five action labels in both contexts while only the successful context produces
the required effects. This is the direct control against an action-only value
function.

The typed effect vocabulary was already induced using both lineages in
T12.4a.4c. Consequently, T12.5 calls seed 8705 a replication of the **ordering
hypothesis**, not a held-out validation of the effect representation. Target
games, validation splits and holdout remain closed.

## Rival progress programs

Each of the 24 complete causal programs is paired with four explicit rivals:

1. `terminal_only`: value is flat until success is observed;
2. `change_count`: any recognized change advances a counter;
3. `unordered_effects`: all milestones matter, but their order does not;
4. `ordered_effects`: only the next expected typed effect advances the stage.

This produces 96 lightweight joint particles. Each particle contains the hash
of its complete owner program and its own goal-progress program. Neural modules
are not introduced. Owner dynamics and goal predicates are never recombined
across particles.

For an ordered program with five milestones, the progress potential is
`stage / 5`. Unrelated actions may leave the stage unchanged, but an effect
from a later stage cannot advance it early. Action labels are ignored by the
progress executor; only typed effect deltas are consumed.

## Induction and replication

The ordering posterior is updated first with the exact evidence from lineage
8701. It is then evaluated, without weight updates, on lineage 8705. Only after
the replication accuracy is recorded may lineage 8705 consolidate the common
posterior.

The exact-prefix ablations identify action order but do not contain observed
typed deltas for every branch. They are therefore retained as a separate
`exact_replay_action_order_proxy` modality. The successful and failed full
traces are retained as `observed_typed_transition_trace`. Reports and receipts
must not merge those evidence classes or claim that proxy effects were directly
observed.

## Preregistered baselines and gate

The ordered program is compared against:

- terminal-only value;
- change/novelty count;
- unordered milestone completion;
- exact action-sequence matching;
- state-only initiation classification.

The offline gate requires:

- 100% ordering replication accuracy on lineage 8705;
- 100% posterior classification accuracy on that lineage before consolidation;
- consolidated ordered-program mass of at least 0.95;
- ordered accuracy strictly above every preregistered baseline;
- strictly increasing values across the five observed successful prefixes;
- zero progress across the same-action failed-context trace;
- invariance to arbitrary action relabeling;
- correct ranking of the next expected effect above an out-of-order and an
  irrelevant effect;
- preservation of every complete owner-program posterior mass within `1e-12`;
- exactly 96 or fewer joint particles, no forbidden semantic input, zero SDK
  calls and at most 3 GiB of artifacts.

A pass establishes only source reconstruction plus cross-lineage replication
of the ordering hypothesis. It authorizes a separately frozen shadow
experiment in which predicted action effects can be ranked by posterior goal
proximity. It does not authorize environment control, target-game transfer,
validation, holdout, neural training or production use.

