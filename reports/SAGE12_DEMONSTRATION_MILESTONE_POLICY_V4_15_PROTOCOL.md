# SAGE12 V4.15 — demonstration-conditioned milestone policy

Status at freeze: **protocol specified; no V4.15 result observed**.

## Question

V4.14 made terminal credit dense enough to learn from, but its independent
semantic-effect heads did not produce a competent live policy. V4.15 tests the
missing behavioral scaffold directly:

> Can complete human demonstrations teach an object-relative sequence policy
> that proposes plausible actions and milestones on unseen games, after which
> the semantic world model and EBM improve those proposals?

Every registered condition runs. A failed imitation or milestone diagnostic
cannot skip the global transfer panel or the bounded active panel.

## Frozen data boundary

The SAGE11 registry remains unchanged. V4.15 reuses the content-addressed
V4.14 protocol split:

- human demonstration training: `ar25`, `bp35`, `cd82`, `cn04`, `dc22`,
  `ft09`;
- untraced offline transfer: `g50t`, `ka59`, `lf52`, `lp85`, `sp80`, `su15`,
  `tr87`, `tu93`;
- bounded active validation: `re86`, `ls20`, `sc25`;
- final confirmation: the five `NEURO_HOLDOUT_V1` games, kept closed.

The manifest fingerprints every human JSONL file, the V4.14 manifest,
teacher, checkpoint metadata, transfer predictions, active baseline and
result. Model fields, splits, seeds, hyperparameters, thresholds and active
budgets are frozen before compilation or fitting.

## Causal demonstration compiler

Each non-reset human transition becomes one candidate-choice record. The
student sees only:

- a causal recurrent history ending before the current action;
- an identity-free object-relative graph for every candidate;
- previous observed semantic effects;
- a scalar desired return-to-go;
- a typed milestone token.

The executed action is included exactly. Negative candidates comprise every
other recorded legal action name plus deterministic object-centre click
anchors, capped at 16 total candidates. Candidates are sorted by semantic
checksum so their array position cannot reveal the answer.

Teacher-only fields retain source file, episode, state digests, human text,
post-action frame, raw colours, coordinates and persistent object IDs.

The next milestone is the earliest event within 64 actions among:

- `level_complete`;
- `path_opened`;
- `target_removed`;
- `target_created`;
- `target_moved`;
- `actor_approached_root`;
- other `productive`;
- `none_within_64`.

The teacher also supplies milestone distance and discounted suffix return.
The return is a temporal condition, not an immediate action label. Training
uses all transitions; success-conditioned policy loss is weighted up to 5×,
without deleting exploratory or failed play.

## Model

The policy uses:

- the V4.14 identity-free hashed object graph vocabulary;
- a 32-wide token embedding and 96-wide DeepSets graph encoder;
- a 128-unit GRU over executed graphs and observed effects;
- a candidate compatibility scorer;
- an eight-way milestone head;
- milestone-distance and suffix-return heads;
- a 16-wide milestone embedding used by the conditioned scorer.

Two policy losses share the encoder:

1. ordinary behavior cloning with milestone/return inputs zeroed;
2. success-conditioned cloning with the teacher milestone and observed
   return-to-go.

Milestone cross-entropy, distance regression and return regression are
auxiliary losses. At deployment the full learned lane uses its own predicted
milestone and requests return `1.0`; the oracle-milestone lane supplies the
teacher milestone only as a non-deployable upper bound.

Training uses AdamW, 30 epochs, learning rate `0.0015`, weight decay `0.0001`,
32-transition truncated histories and seed `5_150`. CUDA is selected only
when the same workload is at least 20% faster than CPU.

## Evaluation ladder

### Human leave-one-game-out

Every human game is predicted by a model trained on the other five. Report:

- candidate top-1 accuracy and negative log-likelihood;
- action-only and deterministic-template baselines;
- ordinary behavior cloning;
- learned-milestone success conditioning;
- oracle-milestone success conditioning;
- relation-shuffled and history-shuffled controls;
- milestone macro balanced accuracy and Brier;
- productive-prefix and within-16 accuracy;
- exact three-action imitation;
- game-signature leakage.

`BEHAVIOR_PRIOR_SUPPORTED` requires all of:

- learned success-conditioned top-1 gain over action-only has a positive 95%
  paired-bootstrap lower bound;
- nonnegative top-1 transfer on at least four of six games;
- relation shuffling has a positive 95% degradation lower bound;
- milestone balanced accuracy exceeds the majority baseline by at least
  `0.05`;
- output game-identity accuracy is at most `0.20` above its majority baseline.

These checks classify the component but do not gate later evaluation.

### Untraced transfer and EBM reranking

All V4.11 action panels run with:

- action-only;
- action-sequence-only;
- deterministic template;
- ordinary behavior policy;
- learned-milestone success policy;
- relation-shuffled success policy;
- success policy plus V4.14 temporal EBM;
- oracle-milestone policy plus temporal EBM;
- true-world learned EBM;
- exact oracle.

The EBM coefficient is fixed at `0.5` after per-panel z-normalization. Future
frames remain scoring-only.

`GLOBAL_RERANKING_SUPPORTED` requires:

- paired 95% utility-gain lower bound above zero versus the learned success
  policy;
- nonnegative transfer on at least five of eight games;
- at least one completion and at least 50% of oracle completion
  opportunities.

### Active validation

The V4.14 action-sequence baseline is reused by checksum because game, seeds,
budgets and controller are unchanged. Fresh runs execute:

- ordinary behavior policy;
- learned-milestone policy plus temporal EBM.

Each controller runs `re86`, `ls20`, `sc25`, seeds 0–2, budget 1,000 actions
and at most 14 resets. All legal candidates are scored. Report levels, WINs,
GAME_OVERs, illegal proposals, latency and paired progress. Non-zero live
level progress is required before any larger panel is recommended, but V4.15
remains descriptive and cannot open the holdout or promote authority.

## Reproducibility

The implementation exposes `freeze`, `compile`, `train`, `evaluate`, `active`
and `run-all`. It writes checksummed JSON/JSONL artifacts under
`training/sage12/demonstration_milestone_policy_v4_15`, focused tests, this
protocol, a result report whether positive or negative, and an updated SAGE12
README.

Only scoped V4.15 changes may be committed. Existing unrelated worktree
changes remain untouched.
