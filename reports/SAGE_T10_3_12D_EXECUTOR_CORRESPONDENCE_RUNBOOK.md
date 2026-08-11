# SAGE T10.3.12d — Operator runbook

Run from the repository root with the repository virtual environment. Do not
run `freeze` before static checks and focused regressions are green.

Order:

1. static checks and regressions;
2. `freeze` exactly once;
3. `audit-parent`;
4. `audit-trajectories`;
5. `preflight`;
6. `compile-executors` before any diagnostic action;
7. `status`, requiring zero authorized actions;
8. `active-diagnostic`;
9. `adjudicate`;
10. `report`.

Before `active-diagnostic`, the signed registry must report
`compiled_before_diagnostic_actions = true`,
`parent_outcomes_used_for_program_fit = false`,
`parent_grounded_paths_imported = false`, and
`historical_support_imported = 0`.

Stop immediately on code 2. If adjudication returns code 3, run only `report`
to seal the negative. Never replay an interrupted work scope or modify the
frozen protocol. The terminal report must retain `diagnostic_only`,
`confirmatory_evidence = false`, closed validation/holdout/sequence firewalls,
and zero production authority.

