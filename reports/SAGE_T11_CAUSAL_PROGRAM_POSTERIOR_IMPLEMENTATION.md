# SAGE.T11/T12 causal-program posterior implementation result

## Implemented

- Versioned complete causal-program, state, evidence, intervention, prediction,
  trace, and bundle contracts.
- Fail-closed two-slice compiler with derived graph, topological plan,
  descendant sets, type/domain checks, action availability, canonical hashes,
  and full serialization.
- One causal executor supporting symbolic, hybrid, neural, action, and internal
  `do` interventions.
- Robust A38T structured likelihood and diagnostics; A39T log-space posterior,
  MDL prior, bounded diversity and repair; A40T checksummed append-only memory.
- Exact-prefix bundle runner requiring preregistered prediction matrices and
  identical prefix hashes before each branch.
- Lexicographic route/progress/probe/exploration arbitration with bounded probe
  risk.
- Proposal-only adapters for SAGE.9p, M2/LLM payloads, SAGE.9o routes,
  T10.3.12f families, and ARC-LeWM modules.
- Shared graph-masked neural mechanism heads, ARC-LeWM observation encoder,
  calibrated observation-likelihood head, and the registered multi-term loss.
- Target `CausalSageTController`, usable through the existing controller
  injection point while default authority remains off/shadow.
- Split firewall, one-shot holdout guard, bounded diagnostics, protocol
  manifest, and historical `ft09` baselines.
- Experimental CLI with separately sealed rival-program and exact-prefix-plan
  inputs, clean-tree/code-bound freeze, paired baseline/full/ablation arms,
  replay-gated bounded authority, per-reset controller reconstruction, A40
  reload ablation, game-scoped validation memory, and signed reports/receipts.
- Prefix replay no longer imports `arcengine` for injected test environments;
  the real SDK reset is resolved only at the environment boundary.

## Executed validation

- New causal/replay/experiment suite: 28 passed.
- Targeted legacy-controller and unified-controller compatibility: 24 passed.
- Focused compatibility suite covering legacy SAGE.T, unified controller,
  SAGE.11 splits, and ARC-LeWM: 53 passed. The combined focused run passed all
  75 tests.
- Ruff on all new/changed causal and replay files: passed.
- Broad historical SAGE-prefixed regression:
  - group 1: 299 passed;
  - group 2: 188 passed;
  - group 3 first half: 204 passed;
  - group 3 durable-runtime half executed without an assertion failure but did
    not terminate under the ten-minute cumulative budget;
  - final SAGE.T group: 141 passed, one historical test deselected for the
    separately recorded defect below.

## Negative and unverified outcomes

- No real ARC active causal run was executed. `bp35`, frozen source validation,
  `ft09`, and the neural holdout remain unobserved for this implementation.
- The installed `arc_agi` package does not export `Arcade`; ten historical live
  acquisition tests therefore fail at environment setup. This is an optional
  SDK/environment limitation, not a causal-program assertion failure.
- `test_status_cli_emits_exactly_one_json_object` in the frozen T10.3.12f
  lineage fails because its existing default manifest path is absolute and
  ignores a temporary `--repo-root`. The checksummed T10.3.12f protocol/runtime
  were not modified to conceal this unrelated defect.
- Three older `sage_third_unknown_game` runners observe zero reproduced actions
  or an unresolved source probe in the current environment. They are outside
  the new SAGE.T causal package and were not promoted or reinterpreted.
- A single exhaustive test process timed out at 74%; deterministic smaller
  groups were used to obtain explicit results. The cumulative durable-runtime
  subgroup still exceeded its budget without printing a failing assertion.

## Scientific status

The implementation gates for contracts, execution, posterior behavior,
memory, replay fail-closure, shared neural mechanisms, and compatibility pass.
The breakthrough gate—real replayed intervention, posterior update, persisted
belief, and improved subsequent action—has not yet been observed in an ARC
environment. Authority and holdout access therefore remain closed.
