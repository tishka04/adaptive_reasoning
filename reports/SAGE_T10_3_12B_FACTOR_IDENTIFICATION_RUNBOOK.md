# SAGE.T10.3.12b — PowerShell runbook

Run from `C:\Users\coudr\projects\adaptive_reasoning`. The commands do not
authorize or execute new ARC actions.

```powershell
$Repo = (Resolve-Path .).Path
$Py = Join-Path $Repo 'ARC-AGI-3-Agents\.venv\Scripts\python.exe'
$Module = 'theory.sage_t.t10_3_12b_runtime'

& $Py -B -m pytest -p no:cacheprovider `
  tests/test_sage_t_factorial_invariants_v10_3_12b.py `
  tests/test_sage_t_t10_3_12b_protocol.py `
  tests/test_sage_t_t10_3_12b_runtime.py
if ($LASTEXITCODE -ne 0) { throw 'T10.3.12b focused tests failed before freeze' }

& $Py -B -m $Module freeze --repo-root $Repo
if ($LASTEXITCODE -ne 0) { throw 'T10.3.12b freeze failed' }

$Phases = @(
  'status',
  'audit-parent',
  'preflight',
  'materialize-variants',
  'compile-factors',
  'evaluate-interventions'
)

foreach ($Phase in $Phases) {
  & $Py -B -m $Module $Phase --repo-root $Repo
  $Code = $LASTEXITCODE
  if ($Code -eq 2) { throw "Integrity failure during $Phase" }
  if ($Code -eq 3) {
    Write-Host "Scientific gate miss during $Phase; no retuning is authorized."
    break
  }
}

& $Py -B -m $Module adjudicate --repo-root $Repo
$AdjudicationCode = $LASTEXITCODE
if ($AdjudicationCode -eq 2) { throw 'Integrity failure during adjudication' }

& $Py -B -m $Module report --repo-root $Repo
$ReportCode = $LASTEXITCODE
if ($ReportCode -eq 2) { throw 'Integrity failure while sealing terminal report' }

$TerminalPath = Join-Path $Repo 'training\sage_t\t10_3_12b_factorial_invariant_identification\terminal_report.json'
$Terminal = Get-Content -LiteralPath $TerminalPath -Raw | ConvertFrom-Json
$Terminal | ConvertTo-Json -Depth 8
$ReportCode
```

Code 0 is a completed positive factor-identification result. Code 3 is a
completed scientific miss. Code 2 is an integrity/provenance failure: stop
without running another writing phase. Any code or protocol change after
`freeze` requires a new suffix and cannot repair this run.

