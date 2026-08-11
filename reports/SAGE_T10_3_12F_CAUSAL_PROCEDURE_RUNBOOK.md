# SAGE T10.3.12f runbook

Run from the repository root with the ARC-AGI virtual environment.  Do not
freeze until lint, imports, focused tests, and the lineage regression pass.

## Frozen phase order

1. `freeze` exactly once;
2. `audit`;
3. `qa-source`;
4. `compile-prior`;
5. `evaluate-source`;
6. `preflight`;
7. `status` and verify virgin physical accounting;
8. `active-historical`;
9. `adjudicate`;
10. `report`.

Every command writes exactly one JSON object to stdout.  Capture the complete
stdout string in PowerShell and pass it directly to `ConvertFrom-Json`; do not
select the final character of a scalar string.

Before `active-historical`, verify that all offline reports are signed and
passed, the journal contains no intent or event, the collector lock is absent,
the compiled prior is transfer-safe, and every firewall is closed.

`active-historical` returns 0 only after all 144 work receipts and the journal
accounting are complete.  Scientific success or failure is returned by
`adjudicate`.  After code 3, run only `report`; never retry, change a scope,
retune a threshold, or promote a program.  Code 2 requires immediate stop and
no further write phase.

T10.3.13 is not part of this run.  A positive T10.3.12f report merely identifies
which predeclared candidate/control pair could be frozen later.  Opening the
protected panel requires a separate explicit authorization.

## PowerShell fail-closed execution

Run this first.  It performs only static/offline validation and freezes the
experiment; it does not enter the nine-game active matrix until the final
`active-historical` call below.

```powershell
$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path .).Path
$Py = Join-Path $Repo 'ARC-AGI-3-Agents\.venv\Scripts\python.exe'
$Module = 'theory.sage_t.t10_3_12f_runtime'

function Invoke-SagePhase {
    param([Parameter(Mandatory = $true)][string]$Phase)
    $Raw = ((& $Py -B -u -m $Module $Phase --repo-root $Repo 2>&1) | Out-String).Trim()
    $Code = $LASTEXITCODE
    try {
        $Payload = $Raw | ConvertFrom-Json
    }
    catch {
        throw "La phase $Phase n'a pas emis un objet JSON unique. Sortie: $Raw"
    }
    [pscustomobject]@{ Phase = $Phase; Code = $Code; Payload = $Payload; Raw = $Raw }
}

$LintFiles = @(
    'theory/sage_t/causal_procedure_v10_3_12f.py',
    'theory/sage_t/t10_3_12f_protocol.py',
    'theory/sage_t/t10_3_12f_runtime.py',
    'theory/sage_t/t10_3_13_protocol.py',
    'theory/sage_t/t10_3_13_runtime.py',
    'tests/test_sage_t_causal_procedure_v10_3_12f.py',
    'tests/test_sage_t_t10_3_12f_protocol.py',
    'tests/test_sage_t_t10_3_12f_runtime.py',
    'tests/test_sage_t_t10_3_13_protocol.py',
    'tests/test_sage_t_t10_3_13_runtime.py'
)
& $Py -B -m ruff check --no-cache @LintFiles
if ($LASTEXITCODE -ne 0) { throw 'Ruff a echoue; ne pas geler.' }

$FocusedTests = @(
    'tests/test_sage_t_causal_procedure_v10_3_12f.py',
    'tests/test_sage_t_t10_3_12f_protocol.py',
    'tests/test_sage_t_t10_3_12f_runtime.py',
    'tests/test_sage_t_t10_3_13_protocol.py',
    'tests/test_sage_t_t10_3_13_runtime.py'
)
& $Py -B -m pytest -q -p no:cacheprovider @FocusedTests
if ($LASTEXITCODE -ne 0) { throw 'Les tests cibles ont echoue; ne pas geler.' }

$LineageTests = @(
    'tests/test_sage_t_goal_directed_v10_3_9.py',
    'tests/test_sage_t_goal_directed_v10_3_10.py',
    'tests/test_sage_t_goal_directed_v10_3_11.py',
    'tests/test_sage_t_goal_directed_v10_3_12.py',
    'tests/test_sage_t_relational_program_v10_3_12.py',
    'tests/test_sage_t_factorial_invariants_v10_3_12b.py',
    'tests/test_sage_t_cross_game_transfer_v10_3_12c.py',
    'tests/test_sage_t_executor_correspondence_v10_3_12d.py',
    'tests/test_sage_t_closed_loop_successor_v10_3_12e.py',
    'tests/test_sage_t_t10_3_9_protocol.py',
    'tests/test_sage_t_t10_3_9_runtime.py',
    'tests/test_sage_t_t10_3_10_protocol.py',
    'tests/test_sage_t_t10_3_10_runtime.py',
    'tests/test_sage_t_t10_3_11_protocol.py',
    'tests/test_sage_t_t10_3_11_runtime.py',
    'tests/test_sage_t_t10_3_12_protocol.py',
    'tests/test_sage_t_t10_3_12_runtime.py',
    'tests/test_sage_t_t10_3_12b_protocol.py',
    'tests/test_sage_t_t10_3_12b_runtime.py',
    'tests/test_sage_t_t10_3_12c_protocol.py',
    'tests/test_sage_t_t10_3_12c_runtime.py',
    'tests/test_sage_t_t10_3_12d_protocol.py',
    'tests/test_sage_t_t10_3_12d_runtime.py',
    'tests/test_sage_t_t10_3_12e_protocol.py',
    'tests/test_sage_t_t10_3_12e_runtime.py'
)
& $Py -B -m pytest -q -p no:cacheprovider @LineageTests
if ($LASTEXITCODE -ne 0) { throw 'La non-regression T10.3.9-T10.3.12e a echoue; ne pas geler.' }

$Freeze = Invoke-SagePhase 'freeze'
$Freeze.Payload | ConvertTo-Json -Depth 12
if ($Freeze.Code -ne 0) { throw 'Freeze invalide.' }

foreach ($Phase in @('audit', 'qa-source', 'compile-prior', 'evaluate-source', 'preflight')) {
    $Result = Invoke-SagePhase $Phase
    $Result.Payload | ConvertTo-Json -Depth 20
    if ($Result.Code -eq 2) { throw "Defaut d'integrite pendant $Phase. Arret immediat." }
    if ($Result.Code -eq 3) {
        $Terminal = Invoke-SagePhase 'report'
        $Terminal.Payload | ConvertTo-Json -Depth 20
        throw "Gate scientifique manque pendant $Phase. Rapport scelle; aucune action active."
    }
    if ($Result.Code -ne 0) {
        throw ("Code inattendu pendant {0}: {1}" -f $Phase, $Result.Code)
    }
}

$Status = Invoke-SagePhase 'status'
$Status.Payload | ConvertTo-Json -Depth 20
if (
    $Status.Code -ne 0 -or
    $Status.Payload.accounting.authorized_actions -ne 0 -or
    $Status.Payload.accounting.sealed_events -ne 0 -or
    $Status.Payload.accounting.inflight_intents -ne 0 -or
    $Status.Payload.accounting.unresolved_intents -ne 0 -or
    -not $Status.Payload.accounting.equation_holds -or
    -not $Status.Payload.accounting.inflight_valid -or
    $Status.Payload.accounting.live_collector_lock -or
    $Status.Payload.holdout_opened -or
    $Status.Payload.t10_3_13_authorized -or
    $Status.Payload.production_authority
) {
    throw 'Etat non vierge ou firewall ouvert avant active-historical.'
}
```

Only after that block succeeds, launch the bounded historical collection and
seal its scientific verdict:

```powershell
$Active = Invoke-SagePhase 'active-historical'
$Active.Payload | ConvertTo-Json -Depth 20
if ($Active.Code -ne 0) {
    throw "Collecte historique invalide (code $($Active.Code)); ne pas relancer."
}

$Adjudication = Invoke-SagePhase 'adjudicate'
$Adjudication.Payload | ConvertTo-Json -Depth 30
if ($Adjudication.Code -eq 2) {
    throw "Defaut d'integrite a l'adjudication; ne pas produire de claim scientifique."
}

$Report = Invoke-SagePhase 'report'
$Report.Payload | ConvertTo-Json -Depth 30
if ($Report.Code -eq 2) { throw "Rapport terminal invalide." }

$TerminalPath = Join-Path $Repo 'training/sage_t/t10_3_12f_causal_procedure/terminal_report.json'
$Terminal = Get-Content -LiteralPath $TerminalPath -Raw | ConvertFrom-Json
if (
    $Terminal.manifest_checksum -ne $Freeze.Payload.manifest_checksum -or
    $Terminal.accounting.authorized_actions -ne $Terminal.accounting.sealed_events -or
    $Terminal.accounting.inflight_intents -ne 0 -or
    $Terminal.accounting.unresolved_intents -ne 0 -or
    -not $Terminal.accounting.equation_holds -or
    -not $Terminal.accounting.inflight_valid -or
    $Terminal.physical_actions_replayed -ne 0 -or
    $Terminal.legacy_fallback_actions -ne 0 -or
    $Terminal.holdout_opened -or
    $Terminal.t10_3_13_authorized -or
    $Terminal.ar25_opened -or
    $Terminal.production_authority
) {
    $Terminal | ConvertTo-Json -Depth 30
    throw 'Le rapport terminal viole la comptabilite ou un firewall.'
}

"Adjudication exit code: $($Adjudication.Code)"
"Report exit code: $($Report.Code)"
"Verdict: $($Terminal.verdict)"
```

An adjudication/report code 3 is a valid negative scientific result.  Do not
rerun it.  If and only if the terminal verdict is one of the two historical
PASS candidates, stop and request the separate T10.3.13 authorization; this
runbook intentionally contains no holdout-opening command.
