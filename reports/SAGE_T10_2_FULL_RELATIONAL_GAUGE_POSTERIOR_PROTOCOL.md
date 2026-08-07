# SAGE.T10.2 full relational gauge posterior — preregistered protocol

Status: `PREREGISTERED_BEFORE_COLLECTION`

Date frozen: 2026-08-07

Baseline: `05c1c91b82054af55a03ef962745f4f101cd3c0e`

Baseline PR: <https://github.com/tishka04/adaptive_reasoning/pull/6>

## Scientific hypothesis

T10.2 tests a complete particle

\[
H=(D,G,F,\Tau,A),
\]

where dynamics, goal, observer frame, partial transport, and finite option
automaton are inferred jointly.  The registered prediction is that a common
posterior over these complete programs can rank and actively ground mixed
progress sequences that T10.1 missed with monolithic repetition.

The experiment is supported only by causal and active utility.  Better factor
F1, likelihood, calibration, or correspondence alone cannot pass any gate.

## Immutable boundaries

- `JointProgramHypothesis`, `ProgramPosterior`, `ProgramExecutor`,
  `CounterfactualDecisionEngine`, and `SageTController` are frozen and must
  retain their baseline byte hashes.
- T10.2 is additive and has no unified-controller or production authority.
- `ar25`, the final holdout, and production authority remain closed under
  every verdict, including success.
- T10.1 validation outputs for `re86`, `ls20`, and `sc25` may be cited as
  motivation and used as a behavior-frozen comparator only.  They are never
  available to source collection, synthesis, priors, posterior updates, or
  model selection.
- No raw frame, full graph, color, absolute coordinate, persistent entity ID,
  game identity, or seed identity may enter a transferable program.
- Live projectors may construct an anonymous graph transiently.  The persisted
  ledger uses the closed `sage-t10.2-structural-quotient-v2` schema: counts of
  role signatures, counts of fact predicate/truth/arity signatures, allowlisted
  counters, register occupancy, and allowlisted global topology.  It contains
  no endpoint, term, literal, entity token, register value, incidence, or full
  projection hash and cannot reconstruct an `AbstractState` graph.
- Formal non-injectivity is a release gate.  The non-isomorphic seven-node
  contact trees with branch lengths `(4,1,1)` and `(3,2,1)` around their unique
  degree-three node must have the same persisted summary and checksum.

## Frozen representation bank

| Frame | Registered source | Principal transport |
|---|---|---|
| `root_only` | structural root summary | identity |
| `allocentric_object_relative` | V4.9 `ObjectRelativeGraph` | entity permutation and structural translation |
| `action_aligned_relational` | V4.10 intervention frame | allocentric roles to action axis |
| `action_rooted_topological` | V4.16/V4.19 MT graph | persistent correspondence and role-preserving graph isomorphism |

Raw grids, rotations, and reflections are outside T10.2.  A missing projection
is explicit unknown coverage and triggers a registered fallback; it is never
fabricated from another frame.

## Registered interfaces

- `PhysicalEventBundle`: one physical event ID, one executed action, one
  common progression/terminal/goal outcome, and one or more frame projections.
  Fresh and converted source evidence must contain exactly the four frozen
  frames.  The common outcome contributes once to likelihood.
- `ObserverFrameSpec` and `FrameProjection`: deterministic, versioned
  projections with covered channels, provenance, and canonical checksum.
- `TransportMap`: a partial role/fact/action transport with domain, coverage,
  ambiguity, and optional inverse.
- `OptionAutomaton`: initiation, bounded state, structural policy,
  termination, and horizon.
- `JointGaugeHypothesis`: the frozen world program plus one frame, transports,
  and one option automaton.
- `GaugeProgramPosterior`: joint log-space update, stateful diversity, and
  gauge-class marginalization.
- `GaugeDecisionEngine`: bounded counterfactual decision over gauge classes,
  executing one real action before observing and replanning.

## Posterior and decision rules

- Physical likelihood is scored exactly once per event.
- Available projection likelihoods are averaged, never summed.
- Non-commutativity on comparable channels is penalized; incomplete mappings
  are reported as not comparable rather than as successes.
- Channel weights are frozen to objects/relations/topology `1`, progress/goal
  `2`, terminal `4`, with unknown-coverage penalty `0.75`.
- The baseline MDL prior is preserved.  Each registered transport or option
  AST node adds `-0.05` log prior.
- Gauge-equivalent copies contribute summed mass to one decision class; their
  priors are not divided by the number of copies.  A class is quotientable only
  through a versioned `TransportOrbitWitness` whose certified domain covers all
  role, fact, and action symbols referenced by its complete hypothesis.
- At most 256 gauge classes are retained; at most 64 are evaluated per
  decision, with omitted mass retained as one residual class.
- MAP collapse is forbidden before maximum class mass `>=0.90`, Bayes factor
  `>=20`, and the same winning class persists for three transitions.
- The T7 utility and entropy exploration switch are reused without retuning.
- Only legal actions are proposed.  A danger veto always dominates utility.

## Registered option grammar

T10.1 repetition and successor macros remain available.  T10.2 adds:

- `alternate(A,B)`;
- `prime_then_repeat(A,n,B)`;
- `A_until(predicate)->B`;
- `follow_relation_then_apply(A,B)`;
- distinct no-op probes whose stateful signatures are not merged before their
  later behavior diverges.

An option has at most four states, two action schemas, and horizon 16.  A
relation-labelled transition requires both action-bound provenance and a
matching fact or topology witness in the current structural projection; a
free-form `relation` string is never a certificate.  Initiation and relation
evidence are checked again while ranking an executed prefix.  The controller
executes exactly one action, observes the physical event, updates the posterior
and option state, then replans.  A fallback remains subject to the same option
and danger-veto checks.

## Compact replay and transport attestations

Each state summary is limited to 8 KiB and each event model view to 32 KiB;
the complete canonical event is limited to 48 KiB.  Replay may materialize an
aggregate state containing only allowlisted counters, topology, and regime,
with empty entity/fact/register collections and explicit incomplete incidence.
It must never invent anonymous nodes or endpoints.

An exact live graph certificate is discarded after collection.  The ledger
retains only its stable symbolic orbit, endpoint-free before/after summary
hashes, structural observation hashes, the live-certificate hash, exactness
flags, and a recomputable receipt.  Replay distinguishes
`live_graph_exact_attested` from `summary_commutative_exact`: the former is
historical provenance, while only the latter is rechecked from persisted
evidence.  Any altered edge, domain, summary hash, observation hash, or receipt
invalidates the witness.

## Source data and split

The only source-train games are:

- `bp35-0a0ad940`;
- `lp85-305b61c3`;
- `su15-4c352900`.

Allowed evidence is (a) newly executed source trajectories and (b) frozen
source-only replay transitions with complete provenance.  Historical
aggregates are comparators, not training examples.

Collection is fixed before execution:

| Split | Seeds | Resets per game/seed | Maximum actions/reset |
|---|---|---:|---:|
| discovery | 0, 1, 2 | 4 | 64 |
| leave-one-game-out confirmation | 3, 4, 5 | 4 | 64 |

The maximum is 4,608 actions.  An episode stops on positive progress,
`GAME_OVER`, or budget.  Total collection wall time is limited to 5,400 s.
Each confirmation unit contains exactly four resets total: two controlled by
the learned common posterior and two by the capacity-matched independent
posterior, with the arm order counterbalanced.  Thus the registered cross-fit
audit contains exactly nine game/seed units and the comparator does not double
the 4,608-action source budget.

Within each leave-one-game-out fold, the held-out game contributes neither
grammar nor priors.  The controller may update only from actions it actually
executes in that confirmation condition.  The final three-game fit occurs
only after every cross-fit prediction is immutable.

## QA gate before fit

Fit is prohibited if any check fails:

- code, source, split, manifest, environment metadata, or checksum drift;
- any validation, `ar25`, holdout, or game/seed identity contamination;
- duplicate physical `event_id`;
- persistent correspondence below 90% or fully ambiguous correspondence at
  or above 10%;
- failed exact transport round trip, entity-permutation invariance, or exact
  commutativity;
- universal learned predicate or prevalence outside `[0.5%, 95%]`;
- learned predicate support below 32 occurrences across at least two games;
- fewer than 80% evaluable nonterminal prefixes;
- fewer than 50% prefixes coherent in at least two frames.

Rare `level_complete`, `WIN`, and `GAME_OVER` events remain evaluation anchors
but cannot alone become learned targets.  A methodological correction after
QA requires an explicit preregistration amendment committed and pushed before
the first fit.  Splits and outcome gates cannot be silently changed.

## Registered controls and oracles

Every control is executed even after a scientific gate fails:

1. T10.1 behavior-frozen baseline;
2. capacity-matched independent posterior;
3. four single-frame lanes;
4. identity-only, no-transport, and deterministically permuted transport;
5. frame, binding, dynamics, goal, and option swaps;
6. early MAP collapse;
7. immediate no-op deduplication reproducing T10.1;
8. best executed sequence oracle;
9. grammar oracle;
10. transport, dynamics, goal, option, and complete-program oracles.

Every factor oracle is a paired causal comparison, not a presence check.  The
aligned factor and its typed swap or ablation are evaluated on the same
ordered physical bundles, observation count, particle/class capacity, and MDL
prior vector, with both positive and negative outcomes represented.  Passing
requires a strictly better grounded score for the aligned factor.  The option
oracle additionally requires compatibility on a progressing prefix after the
alternatives diverge.  The complete-program oracle requires a strict loss
under a separate intervention on each of `D`, `G`, `F`, `Tau`, and `A`.

The grammar oracle must progress on at least two source games and complete at
least two levels cumulatively, with zero errors, illegal actions, or
`GAME_OVER`.  Otherwise the exclusive verdict is
`MIXED_SEQUENCE_GRAMMAR_MISS`, irrespective of posterior scores.

The learned system must then:

- rank a progressing sequence top 8 in each positive fold, with median rank
  at most 4;
- recover at least 50% of oracle levels;
- be non-negative on at least two source games;
- beat the independent posterior in levels per 1,000 legal actions with a
  paired bootstrap 95% lower bound strictly above zero;
- degrade positively under no-transport and binding-swap controls;
- keep incremental game/seed probe accuracy at or below 0.10.

## Conditional paired validation

Validation is inaccessible until the source report is complete, checksummed,
and has status `PASS_T10_2_SOURCE_GATE`.  Only then compare behavior-frozen
T10.1 with frozen T10.2 on `re86`, `ls20`, and `sc25`:

- seeds 2101-2105;
- 14 resets per seed;
- 96 actions maximum per reset;
- 15 game-seed pairs and 20,160 actions maximum per controller;
- counterbalanced A/B order;
- posterior reset between pairs;
- no learning between seeds or controllers.

The T10.1 arm is a literal bounded projection of its frozen search: independent
one-step probes are followed, on separate resets, by the frozen
`path_successor` and `repeat_target` candidate macros.  It reads no historical
T10.1 outcome.  If the complete scan-plus-macro schedule cannot fit within the
14 registered resets, validation refuses before executing either arm; no
repeat-first approximation is substituted.

The primary metric is paired difference in levels per 1,000 legal actions.
Support requires all of:

- total advantage at least one level;
- at least two games non-negative;
- paired bootstrap 95% lower bound above zero;
- zero illegal actions and zero errors;
- `GAME_OVER` rate no worse than T10.1;
- at least 95% of planned budget completed outside registered stops;
- decision latency p95 <=750 ms and p99 <=2,500 ms;
- observation latency p95 <=500 ms and p99 <=3,000 ms;
- total validation wall time <=21,600 s.

## Resource gates

- compact ledger <=256 MiB;
- shard <=64 MiB;
- checkpoint/posterior <=10 MiB;
- derived file <=512 MiB;
- scratch and cache <=5 GiB each;
- repository <=12 GiB;
- resident memory <=8 GiB;
- free disk before an expensive job >=100 GiB.

## Exclusive verdict ladder

The first applicable verdict is final for T10.2:

1. `DATA_OR_PROVENANCE_INVALID`
2. `MIXED_SEQUENCE_GRAMMAR_MISS`
3. `FRAME_TRANSPORT_MISS`
4. `GOAL_OR_DYNAMICS_MISS`
5. `COMMON_POSTERIOR_MISS`
6. `OPTION_SYNTHESIS_MISS`
7. `SOURCE_GROUNDING_MISS`
8. `SOURCE_VALIDATION_TRANSFER_MISS`
9. `SAFETY_OR_RESOURCE_MISS`
10. `SAGE_T10_2_GAUGE_POSTERIOR_SUPPORTED`

No negative result triggers retuning or a second validation opening.  A new
attempt requires a new version and manifest.

## Publication sequence

1. Commit and push this protocol plus the code-bound source manifest before
   collection.
2. Publish any necessary pre-fit QA amendment before fitting.
3. Execute and report every registered source control.
4. Open paired validation if and only if source gates pass.
5. Publish the final report, checksums, latency/resource evidence, all
   negative controls, and an explicit artifact-exclusion inventory in the
   stacked draft PR.

## Amendment A1 - default bundle dispatch before collection

The first invocation of `collect` under manifest `018bff2e...` stopped on the
first source transition because the internal dispatch passed two
custom-builder-only keywords to the closed default bundle constructor.  The
environment step had already executed one legal action on the allowed `bp35`
source game, but bundle construction failed before an event could be sealed.
No output directory, ledger, projection, checkpoint, report, QA result, or fit
was created, and no evidence from that action is available to the learner.
The invocation is recorded as a technical `DATA_OR_PROVENANCE_INVALID` attempt,
not as the scientific verdict of the amended T10.2 experiment.

Amendment A1 makes only the following execution correction:

- call the default bundle constructor with its seven declared keyword
  arguments;
- retain the enriched context only for injected builders;
- add a regression test exercising the strict default signature;
- regenerate and publish the code-bound manifest before retrying `collect`.

The aborted action is included in the source action accounting.  The completed
collection plus this one action must remain at or below the registered ceiling
of 4,608.  Frames, splits, seeds, resets, gates, priors, controls, verdicts, and
validation conditions are unchanged.  This is an implementation amendment,
not a methodological retuning.
