# SAGE.T12.5b.4 — Target-local short-program utility

## Scientific status

T12.5b.3 is preserved as a signed negative result with status
`FAIL_T12_5B_3_COLLECTION_INTEGRITY_GATE`. Its files are intact and its fixed
36-trial schedule completed, but the preregistered zero-terminal condition
failed. The result also contains two scientific misses that are not promoted
past the integrity classification: no stage-3 progress affordance survived a
neutral detour and no hard one-step contrast existed.

T12.5b.4 is a separately frozen source-train experiment. It does not amend,
rerun, or reclassify T12.5b.3. It changes the unit of analysis from one action
to a bounded action program and makes objective value and terminal risk part
of the observed outcome.

Earlier broad target-local searches are also preserved. T12.4a.4d used 24,171
SDK calls and T12.4a.4d.1 used 36,481 without discovering a progress witness.
T12.5b.4 therefore does not launch another open-ended archive search.

## Question

> From a new exact state reached by one progress-neutral detour, can a short
> locally grounded action program produce safe level progress, form a real
> larger-magnitude contrast, and transfer from route lineage 8701 to 8705
> while the frozen causal-progress posterior ranks it above the distractor?

The primary outcome is objective progress (`level_delta > 0`). A terminal
outcome with no level progress is a valid unsafe effect, not collection
corruption. Labels never use the causal-progress score being evaluated.

## Parent-bound target

Freeze requires the exact signed T12.5b.3 manifest, receipt, trials,
affordance registry, contrast registry, and report. It verifies:

- parent status and classification are unchanged;
- every parent artifact checksum matches;
- the fixed schedule, exact-prefix, repetition, determinism, and budget checks
  passed;
- the only parent integrity-gate miss is the preregistered terminal condition;
- no transported one-step progress affordance or hard contrast was claimed.

The target context is selected from the sealed parent before any new ARC call.
`stage_3_action4_depth_1` is the unique shallowest context that is valid in
both lineages, has no terminal trial, and exposes at least two deterministic
actions in each lineage. Lineage 8701 is calibration because it exposes the
full `ACTION3/ACTION4/ACTION6` catalogue. Lineage 8705 is evaluation. The
shared transport catalogue is `ACTION3/ACTION4`; `ACTION6` remains in the
calibration matrix and is explicitly non-transportable rather than converted
to a zero-effect observation.

## Frozen candidate programs

Calibration executes every length-2 and length-3 sequence over
`ACTION3/ACTION4/ACTION6`, twice, from the same exact depth-1 context:

`(3^2 + 3^3) programs × 2 repetitions = 72 calibration trials`.

Every trial independently resets, replays the sealed route, replays the first
three option steps, executes one `ACTION4` detour, and then attempts its fixed
program. Execution stops on the first terminal state or positive level delta.
Unavailable actions remain missing. Raw frames are not persisted.

A program is executable only if all declared actions run. Repetitions must
agree on availability, projected effects, level delta, terminal state, and
completion. Candidate terminal failures are registered as deterministic risk
evidence and do not fail collection integrity by themselves.

## Calibration labels and selection

A calibration program is:

- **safe progress** when it is complete, deterministic, has
  `level_delta > 0`, and has no terminal failure;
- **safe non-progress** when it is complete, deterministic, non-terminal,
  and has `level_delta == 0`;
- **unsafe** when it reaches a terminal state without level progress;
- **missing** when the complete action program is not locally executable.

Only programs composed entirely of the shared transport catalogue may be
selected for evaluation. The progress program is the shortest safe-progress
program, with lexicographic action order as the fixed tie-break. The
distractor is the largest-magnitude safe non-progress program whose magnitude
is at least one greater than the selected progress program. These rules are
independent of the causal-progress score.

For each deterministic complete trace, the frozen posterior computes the
change in expected causal potential after the exact prefix. Calibration passes
only if the selected progress program has strictly greater causal gain than
the registered larger-magnitude distractor.

## Two-phase firewall

Calibration writes an immutable signed receipt and, only on a full pass, a
sealed evaluation registry. Evaluation is a separate manual command and is
not authorized by the initial freeze alone.

Evaluation runs exactly the registered progress program and distractor twice
each from lineage 8705:

`2 registered programs × 2 repetitions = 4 evaluation trials`.

The final gate requires exact replay and deterministic execution, transported
safe level progress for the progress program, stable safe non-progress for the
distractor, and a strictly higher causal gain for the progress program.

## Negative-result routes

Calibration outcomes are classified without retuning:

- malformed, incomplete, non-exact, nondeterministic, or over-budget
  collection → `FAIL_T12_5B_4_CALIBRATION_INTEGRITY_GATE`;
- no transport-eligible safe progress program →
  `FAIL_T12_5B_4_NO_LOCAL_PROGRESS_PROGRAM`;
- no registered larger-magnitude safe distractor →
  `FAIL_T12_5B_4_NO_HARD_UTILITY_CONTRAST`;
- contrast exists but causal gain is not strictly higher →
  `FAIL_T12_5B_4_CAUSAL_UTILITY_GATE`;
- all calibration gates pass → `PASS_T12_5B_4_CALIBRATION_GATE`.

Evaluation outcomes are classified separately:

- replay, registry, determinism, or budget failure →
  `FAIL_T12_5B_4_EVALUATION_INTEGRITY_GATE`;
- progress program does not transfer →
  `FAIL_T12_5B_4_PROGRESS_PROGRAM_TRANSFER_GATE`;
- distractor does not remain safe non-progress →
  `FAIL_T12_5B_4_DISTRACTOR_STABILITY_GATE`;
- causal gain does not transfer →
  `FAIL_T12_5B_4_CAUSAL_UTILITY_TRANSFER_GATE`;
- full pass → `PASS_T12_5B_4_LOCAL_PROGRAM_UTILITY_GATE`.

No same-version rerun, seed substitution, threshold change, action removal, or
post-hoc program addition is authorized after a miss.

## Bounds and claim boundary

- maximum 6,500 SDK calls for calibration;
- maximum 1,000 SDK calls for evaluation;
- maximum 7,500 total SDK calls;
- maximum 7,200 wall-clock seconds per physical phase;
- maximum 3 GiB of artifacts per phase;
- no raw frame persistence;
- source-train game `bp35` only;
- calibration lineage 8701 and evaluation lineage 8705 only.

A final pass authorizes only preparation of a separately frozen T12.5c
paired-control protocol. It does not authorize that run. Environment control,
source validation, holdout access, neural training, target-game transfer,
production use, and controller authority remain closed in every outcome.
