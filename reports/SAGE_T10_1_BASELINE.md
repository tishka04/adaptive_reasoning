# SAGE.T7-T10.1 published baseline

## Status

This document freezes the local SAGE.T line before SAGE.T10.2.  The immutable
parent is SAGE12 V4.19 commit
`64fec35a4211e0bbac896987d9efe1038b548f94`.  SAGE.T remains fail-closed by
default and no result in this baseline opens `ar25`, the final holdout, or
production authority.

The scientific endpoint is T10.1: 182 legal actions were executed on the
single frozen source-validation opening (`re86`, `ls20`, and `sc25`), with
zero errors, zero illegal actions, zero `GAME_OVER`, and zero completed levels.
All three games were diagnosed as `SEQUENCE_MISS`.  The next admissible test is
therefore mixed option induction under a joint relational-gauge posterior, not
another calibration of the T10.1 monolithic macros.

## Evidence chain

| Stage | Recorded status | Checksum |
|---|---|---|
| T9.6 productive abstention | `T9_6_PASSED` | `38624e011d9e24018fc54d5f2cecd8679cc8c93c9b028b1864c914ae7a8e7e0c` |
| T10.0 progress witness | `FAIL_CLOSED` | `47f76cbf99d7308224246a3ede768b0f0088e57dd254e50ae9f4fc478a880b30` |
| T10.0b source-train | `PASS_T10_0_AUTHORIZE_T10_1` | `a72bff60c6fec8a7fc1d3a7b4ecebabb5d84c39bc80bfc57ed641e0080075fbc` |
| T10.1 manifest | `FROZEN_BEFORE_T10_1_SOURCE_VALIDATION` | `4d1c4dc8b62973187ea5e1c52e698652fdaeb424ae481b56baded0c0b2b9c1a3` |
| T10.1 result | `FAIL_CLOSED` | `167649e5a0e27d63668ca20ae98c57dfd50dd469204ce53a7a5af488fae6348a` |

The T10.1 loader reproduced both registered checksums and verified the hashes
of `progress_witness_v10.py` and `source_validation_progress_v10_1.py` against
the frozen manifest.  The T10.0b parent report also passed its embedded
checksum and status check.

## Historical interpretation

- T7-T8 established the joint-program posterior, deterministic executor,
  counterfactual replay, and fail-closed shadow path.  Calibration and signal
  sufficiency remained negative.
- T9.0-T9.4 added reachability, terminal calibration, trajectory planning,
  bounded execution, and paired source-train authority.  T9.5 then failed
  source-validation; it did not open the holdout.
- T9.6 recovered safe source-train behavior through productive abstention.
- T10.0 failed to ground a progress witness.  T10.0b passed after the frozen
  source-train revision, with rank-one witnesses on `lp85` and `su15`.
- T10.1 transferred that frozen control family once and failed safely on all
  three source-validation games with `SEQUENCE_MISS`.

These results justify a new versioned experiment.  They do not justify
retuning against T10.1 validation outcomes or enabling production authority.

## Published scope

The baseline contains:

- the complete local `theory/sage_t` source and frozen manifests from T7
  through T10.1;
- the complete `tests/test_sage_t_*.py` regression set;
- the default-off integration in `theory/unified_cognitive_controller.py`;
- SAGE.T documentation and milestone notes;
- only the compact T10.0b and T10.1 machine reports;
- a reproducible, checksummed inventory of every local training artifact kept
  outside Git.

The integration does not mark any scientific gate as passed.  `off` remains
the default; the historical shadow/bounded/active modes retain their explicit
gate checks and safety vetoes.

## Artifact exclusion inventory

`training/sage_t/t10_1_omitted_artifacts_inventory.json` records 2,823 omitted
files totaling 163,575,010 bytes.  Its canonical inventory checksum is
`a2b4c26ca999d384d8728dcb3ce65632673963039f44b6c0b631e22d862a80d5`.

The excluded files are local raw transitions, JSONL condition rows, worker
logs, caches, partial reports, and other regenerable historical outputs.  They
were hashed in place and were not deleted.  Only the two compact T10 reports
(204,920 bytes total) are versioned.

## Verification

The baseline checks recorded on 2026-08-07 are:

- T10-focused suite: `9 passed`;
- complete SAGE.T regression suite (`tests/test_sage_t_*.py`): `212 passed`;
- repository-wide suite in the pinned ARC runtime: `2,101 passed`, with two
  pre-existing asset failures and no SAGE.T regression;
- manifest and result checksum reconstruction: passed;
- free disk before expensive work: 649,559,187,456 bytes;
- raw/regenerable artifacts deleted: `0`.

The two repository-wide failures are environmental: `training/checkpoints`
does not contain the pretrained world-model/EBM checkpoints expected by
`test_agent_runtime_config.py`, and
`models/qwen2_5_0.5b_instruct/model.safetensors` is absent for the V4.7
integration-manifest test.  They are the same optional-asset class documented
for V4.19 and do not touch the SAGE.T package or its tests.

## Frozen boundary for T10.2

T10.2 must branch from the published commit containing this baseline.  Its
collector and fit path may read only `bp35-0a0ad940`, `lp85-305b61c3`, and
`su15-4c352900`, plus source-only replays with full transition provenance.
The prior T10.1 results for `re86`, `ls20`, and `sc25` are motivation and a
behavior-frozen comparator only; they are not training data.
