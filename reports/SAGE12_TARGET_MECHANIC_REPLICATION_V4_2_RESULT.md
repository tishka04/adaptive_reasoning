# SAGE12 target-mechanic replication V4.2 — final result

Date: 2026-07-27

Status: `FAIL_RUNTIME_CLOSED`

Failure checksum:
`17934d7b576ac11c36abcac6235e7bc259247f225f49edf5e05126971390be6a`

## Outcome

V4.2 passed its complete source preflight and collected the exact
preregistered 768 prospective transitions. The frozen evaluator then stopped
during prediction serialization before it wrote `predictions.jsonl` or
`pilot_result.json`.

The structured and Qwen verdicts are therefore unavailable. V4.2 authorizes
no V5 protocol, world model, EBM, controller, or Qwen-generated rule.

## Runtime failure

The structured engine can emit a generic rule whose internal anchor is
`any`. The frozen V4.2 public serializer translated only the three concrete
compatibility anchors:

- `occupied_object` to `occupied`;
- `empty` to `free`;
- `targetless` to `none`.

When prediction export reached a generic rule, `_public_rule(item.rule)`
looked up `any` in that incomplete mapping and raised `KeyError('any')`.
This is a serialization defect, not a failed metric gate.

The failure happened after prospective outcomes were opened. The frozen
protocol prohibits changing code, schema, prompt, or gates at that point.
V4.2 was therefore not patched or rerun.

## Preserved partial artifacts

The evaluator had already written the complete prospective window set and
both planned Qwen generation streams:

| Artifact | Rows | SHA-256 | Authority |
|---|---:|---|---|
| Validation windows | 576 | `07cddf4a90e7cdd84e9f118cc57dd01efef67c471d0b2bbd7a2008b23c9ef588` | none |
| Qwen outputs | 128 | `4c96b2f6a56719583d977ab07bef4e00dc374b24d8a7ac24e63118eafa5c493c` | none |
| Qwen shuffled outputs | 128 | `02b9f950fa39a74bcaf43d86c1946df88b164fb2d7dfe6ae2a285014f6823c8d` | none |

These files are published for audit only. Their existence does not imply
that JSON, grounding, recall, shuffle, calibration, transfer, or any other
gate passed. No partial metric is promoted.

## Scientific interpretation

V4.2 neither supports nor refutes the semantic-trajectory architecture. It
validly established that the source representation repair passed all frozen
capacity, leakage, calibration, and temporal-utility checks. The prospective
hypothesis test itself has no valid result because its frozen evaluator did
not complete.

The minimal successor must be a newly versioned replication. Before opening
fresh outcomes it should give rule anchors a distinct public vocabulary that
includes `any`, add a serialization test covering generic exact/family rules,
freeze unused policy seeds, and rerun the full preflight and collection
sequence. The already opened V4.2 shards cannot be reused as a clean gating
set.

## Validation and authority ledger

Before the source boundary, 85 targeted SAGE12 tests and all 1,752 repository
tests passed. The missing `any` case was not covered; the runtime failure is
the evidence for that test gap.

| Authority | Result |
|---|---|
| Structured V4.2 verdict | unavailable |
| Qwen verdict | unavailable |
| V5 protocol | unauthorized |
| World model | unauthorized |
| EBM | unauthorized |
| Controller | unauthorized |
