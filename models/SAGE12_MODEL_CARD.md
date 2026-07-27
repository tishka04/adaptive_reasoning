# SAGE12 semantic planner — model card

Status: software implementation complete; free-generation V1, constrained V2,
action-target V3, temporal mechanic V4, and clean replication V4.1 failed
closed. V4.1 stopped at its source-only preflight, before prospective
collection. V4.2 passed source preflight and collection but failed closed at
runtime before producing prospective metrics. The world model and EBM remain
untrained and unauthorized.

V4.1 repairs causal role resolution, adds leave-one-source-game-out Platt
calibration and thresholds, and compacts the Qwen contract below a verified
token budget. Structured and Qwen authority are separate: a structured pass
may authorize only a deterministic V5 protocol, while Qwen must pass its own
gates before LLM proposals can enter V5.

Those repairs passed their direct audits: role resolution was 0.9984,
calibrated source macro ECE was 0.0360, and compact Qwen prompts were 322–345
tokens. V4.1 nevertheless failed because `actor_displaced` had only 35
positives against the required 75 and static identity leakage was +0.1293
against a maximum +0.10. No Qwen generation or prospective evaluation ran.

V4.2 maps anchors to `occupied`, `free`, or `none` and excludes
`actor_displaced` from every model-facing and authority-bearing interface.
The old actor signal remains an audit-only count. A V4.2 pass may authorize
only a separately frozen V5 protocol for target creation, removal, and
movement; it does not authorize model fitting directly.

V4.2 passed all 11 source-preflight gates. The result authorizes only its
frozen fresh collection; prospective transfer, Qwen authority, V5, the world
model, EBM, and controller remain unvalidated and unauthorized.

The frozen prospective evaluator later stopped at `FAIL_RUNTIME_CLOSED`
because its public rule serializer did not map the structured engine's
generic `any` anchor. No structured or Qwen verdict exists. The partial
validation and Qwen artifacts are audit-only, and all downstream authority
remains closed.

The separately frozen V4 pilot now evaluates sequence-conditioned mechanic
induction from eight observed transitions. Its primary model is a bounded
structured Beta rule inducer; Qwen is a non-authoritative ablation. This
changes no current model or controller authority.

V4 subsequently failed closed despite strong predictive evidence. Structured
macro Brier was 0.0377 versus 0.0708 for the stronger local action-only
baseline, and macro-F1 gained 0.1526. Every validation game improved and
outcome shuffling removed 0.3987 Brier skill. Source actor-role quality failed
at 0.831 and macro ECE narrowly failed at 0.1056. Qwen prompts exceeded their
frozen input cap before generation. The world model and EBM therefore remain
untrained and unauthorized.

## Components

SAGE12 is a composite planner, not one monolithic neural network:

1. a local open-weight causal language model proposes typed semantic
   hypotheses;
2. a deterministic compiler grounds roles and legal actions;
3. a small abstract world model estimates semantic effect probabilities;
4. a transparent heuristic energy ranks bounded trajectories;
5. an optional tiny PyTorch pairwise EBM can replace the heuristic only after
   independent validation;
6. a receding-horizon controller may execute one action and then replans.

The repository includes a lazy `TransformersJSONModel` backend whose default
path is the local Qwen2.5 0.5B Instruct artifact. It permits automatic CUDA
placement but does not download weights. The exact proposal model and prompt
must be frozen in an experiment manifest before an empirical pilot.

## Inputs and outputs

Inputs are structural entities, role labels, size/aspect buckets, grounded
relations, legal action names/arguments, and a hierarchical semantic subgoal.
Game ID, raw grid hash, absolute-layout signature, future state, outcome,
policy arm, and support labels are excluded from model-facing inputs.

The proposal output is a strict JSON list of hypotheses with typed
preconditions/effects and `support=0`. The compiler output is a grounded
semantic option. The world model outputs bounded trajectories with probability
and uncertainty. Energy is lower-is-better. The controller output is either
the unchanged symbolic action or the first action of the best safe trajectory.

## Intended use

The intended use is cross-game, hypothesis-driven planning when current
symbolic knowledge has no protected successful route. Initial use is offline
and shadow evaluation. Bounded or active control is prohibited until the
corresponding protocol stages pass.

## Prohibited use

- treating LLM text or confidence as evidence;
- writing generated support into A32/A33 or symbolic belief stores;
- executing ungrounded or illegal actions;
- using SAGE12 to override observed danger or protected terminal competence;
- training or tuning on holdout/historical/regression-only games;
- activating a learned EBM merely because training loss decreased;
- executing an entire imagined trajectory open-loop.

## Training and evaluation

The frozen Stage A corpus contains 2,104 source-only executed transitions:
1,624 source-training and 480 source-validation rows. Qwen2.5 0.5B Instruct
was evaluated on 112 outcome-blind representative scenes and their 112
relation-shuffled controls. All 224 outputs failed the strict typed parser,
productive-mechanism recall@8 was zero, and the stronger action-only baseline
reached 0.895. Full scene signatures identified source-training games with
99.94% accuracy against a 9.85% majority baseline.

The laptop RTX 4050 was selected only after identical-decoding inference was
3.808x faster by median wall time than CPU. This accelerated proposal
evaluation; it did not change a quality gate. No semantic world-model
checkpoint or pairwise EBM checkpoint exists because Stage A failed and
authorized no later fitting. The pairwise EBM unit test still verifies
optimization mechanics only.

Constrained V2 reused the same source corpus with a frozen Qwen encoder, a
one-bit actor-interaction prompt, independent linear effect heads, and
code-rendered typed hypotheses. JSON, support-zero, grounding, and reduced
game-signature gates passed. Primary macro-F1 was 0.484 versus 0.549 for
action-only, relation shuffling improved it to 0.582, and `re86` transferred
at -0.237. V2 therefore also authorized no world-model fit.

Action-target V3 collected 4,000 fresh source-only transitions and scored four
independent observed effects anchored to the exact movement destination or
click target. Source-only preflight selected a coarse game-identity-controlled
projection and shallow gradient boosting. Validation macro-F1 was 0.232,
below action-only at 0.237 and the deterministic template at 0.371. The
primary gain was -0.140, target-shuffle degradation 0.0005, and macro ECE
0.397. JSON, support-zero, grounding, duplicate, training capacity, and
identity-leakage checks passed; eight data-quality or predictive gates failed.
V3 authorized no world-model fit.

## Safety and epistemic boundaries

- every generated hypothesis must enter with zero support;
- only observed transitions update semantic support/refutation;
- compiler rejection is fail-closed;
- danger filtering occurs before energy ranking;
- protected competence blocks intervention;
- bounded mode spends at most one probe per scene context per branch;
- missing proposal/world-model/energy gates downgrade bounded/active to
  shadow;
- active mode additionally requires its own gate;
- the default integrated mode is `off`.

## Known limitations

Entity identity is currently reconstructed per frame and may be unstable under
large scene rearrangements. The abstract world model assumes effects are
adequately represented by the bounded predicate vocabulary. The template
generator is only a baseline. In Stage A, every Qwen response was Markdown
fenced, fewer than half were JSON after removing only the fence, and none
matched the typed schema. More importantly, the supposedly structural entity,
relation, and action views all carried strong game signatures. Constrained
V2 repaired syntax and reduced leakage, but its single global interaction bit
reversed or collapsed its movement association across validation games.
Output constraints and invariance are therefore necessary but not sufficient;
the missing representation is likely anchored to the exact action target,
requested direction, and stable before/after object event. The heuristic
energy is hand-weighted. No claim of cross-game generalization, improved game
score, calibration, or safe live authority is made.

V3 added that anchoring but its selected coarse model view still collapsed to
only 26 unique source-training signatures. Its conditioned target shuffle
changed only 1.25% of validation rows, and shared signatures carried
substantially different effect rates between games. Stable actor matching also
failed on 540 training rows. The next credible repair therefore needs
persistent object identity and short transition histories from which to infer
a game mechanic; simply collecting more rows for the same global one-step
projection is not supported.

V4 demonstrates that a short observed history substantially repairs the
static-representation failure, especially for actor and target movement.
Creation and removal remain weak at the fixed classification threshold,
binding-shuffle sensitivity is modest, calibration remains above the gate,
and actor identity is still unstable in two source-training games. The result
supports temporal mechanic induction but makes no trajectory, ranking, score,
or live-control claim.

See `theory/sage12/README.md`,
`training/SAGE12_DATA_POLICY.md`, and
`reports/SAGE12_VALIDATION_PROTOCOL.md`. The full negative result is in
`reports/SAGE12_PROPOSAL_PILOT_RESULT.md`; the constrained repair result is in
`reports/SAGE12_CONSTRAINED_PILOT_V2_RESULT.md`; the action-target result is
in `reports/SAGE12_ACTION_TARGET_PILOT_V3_RESULT.md`.

## Runtime-safe target replication V4.2.1

V4.2.1 is frozen as a separate clean replication; it does not amend or rerun
V4.2. The observed state vocabulary remains `occupied`, `free`, and `none`.
Only the public structured-rule vocabulary adds the generic `any` anchor
already used by the internal rule engine. Qwen's concrete-anchor prompt and
schema are unchanged.

Before any prospective collection, the complete structured pipeline must
serialize all 1,911 source predictions and round-trip every exact, family,
concrete, and generic rule. V4.2 shards are forbidden; any authorized
V4.2.1 collection uses 768 fresh transitions under seeds 661, 709, 757, and
809. Predictions and a structured verdict are committed before Qwen starts,
and an automatic failure artifact closes every downstream authority on an
uncaught error.

No V4.2.1 result exists at this freeze checkpoint. The world model, EBM, and
controller remain untrained and unauthorized. See
`reports/SAGE12_TARGET_MECHANIC_RECOVERY_V4_2_1_PROTOCOL.md`.

The executed source rehearsal and preflight have since passed every frozen
check. Source calibrated Brier skill is +0.182060, macro-F1 gain +0.074908,
context skill +0.432771, macro-ECE 0.036452, and identity gain +0.038723.
This evidence authorizes only a fresh 768-transition V4.2.1 collection; it is
not a prospective transfer result and grants no model or controller authority.

The authorized collector has now produced and frozen 768 new prospective
rows, 256 per validation game, with balanced legal-action coverage and no
outcome adaptation. No prospective metric existed before publication of the
raw shards. All downstream authority remains closed pending the single frozen
evaluation.
