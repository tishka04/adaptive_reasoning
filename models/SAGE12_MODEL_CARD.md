# SAGE12 semantic planner — model card

Status: software implementation complete; free-generation V1, constrained V2,
and action-target V3 Stage A pilots failed closed. The world model and EBM
remain untrained and unauthorized.

The separately frozen V4 pilot now evaluates sequence-conditioned mechanic
induction from eight observed transitions. Its primary model is a bounded
structured Beta rule inducer; Qwen is a non-authoritative ablation. This
changes no current model or controller authority.

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

See `theory/sage12/README.md`,
`training/SAGE12_DATA_POLICY.md`, and
`reports/SAGE12_VALIDATION_PROTOCOL.md`. The full negative result is in
`reports/SAGE12_PROPOSAL_PILOT_RESULT.md`; the constrained repair result is in
`reports/SAGE12_CONSTRAINED_PILOT_V2_RESULT.md`; the action-target result is
in `reports/SAGE12_ACTION_TARGET_PILOT_V3_RESULT.md`.
