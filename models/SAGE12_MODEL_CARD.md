# SAGE12 semantic planner — model card

Status: software implementation complete; empirically untrained and
unauthorized.

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

No SAGE12 corpus has been collected and no SAGE12 model checkpoint exists.
The pairwise EBM implementation has six inputs and a default hidden width of
16. Its unit test verifies optimization mechanics only; this is not an
empirical result or promotion evidence. GPU training was not run for this
implementation because there is no preregistered SAGE12 training corpus yet.

The future device policy allows CUDA for the 0.5B local-model inference and
pairwise EBM when a frozen CPU/GPU timing comparison shows a real benefit.
Hardware choice cannot change the task definition, decoding, inputs, labels,
or gates.

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
generator is only a baseline. Strict JSON generation can produce zero
proposals, which is intentional. The heuristic energy is hand-weighted. No
claim of cross-game generalization, improved game score, calibration, or safe
live authority is made.

See `theory/sage12/README.md`,
`training/SAGE12_DATA_POLICY.md`, and
`reports/SAGE12_VALIDATION_PROTOCOL.md`.
