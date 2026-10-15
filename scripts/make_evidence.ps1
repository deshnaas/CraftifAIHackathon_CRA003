<#
.SYNOPSIS
  Reproduce Sentinel's checks and demo evidence on Windows (PowerShell 5.1+).

.DESCRIPTION
  Uses only the documented project commands:
    1. create .venv if missing and install requirements.txt (skip with -SkipInstall)
    2. run pytest, ruff, mypy and a JavaScript syntax check of the dashboard (skip with -SkipChecks)
    3. regenerate the synthetic demo data and baseline  (python -m sentinel make-demo)
    4. run the demo timeline through the CLI            (python -m sentinel run ...)
    5. write a readable summary from the raw JSON       (scripts/summarize_run.py)
    6. record SHA-256 hashes of the outputs

  Everything goes to a NEW folder evidence\repro_<timestamp>. The script refuses to reuse an
  existing folder, so earlier evidence is never overwritten. It does not start or stop servers
  and never terminates other processes. Note: step 3 regenerates data\demo and data\baseline.json
  (generated, deterministic files), as `make-demo` always does.

  Dashboard screenshots are not automated here; see evidence\DEMO_REPORT.md for the capture procedure.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\make_evidence.ps1
#>
param(
    [string]$OutRoot = "evidence",
    [switch]$SkipInstall,
    [switch]$SkipChecks
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$out = Join-Path $OutRoot "repro_$stamp"
if (Test-Path $out) { Write-Error "Refusing to overwrite existing folder $out"; exit 1 }
New-Item -ItemType Directory -Path $out | Out-Null
$log = Join-Path $out "commands.log"
$py = Join-Path $root ".venv\Scripts\python.exe"
$failures = @()

function Write-Log([string]$text) { Write-Host $text; Add-Content -Path $log -Value $text -Encoding UTF8 }

function Invoke-Step([string]$title, [string]$exe, [string[]]$arguments) {
    Write-Log ""
    Write-Log "== $title"
    Write-Log ("> " + (Split-Path -Leaf $exe) + " " + ($arguments -join " "))
    & $exe @arguments 2>&1 | ForEach-Object { Write-Log "$_" }
    $code = $LASTEXITCODE
    Write-Log "exit code: $code"
    if ($code -ne 0) { $script:failures += "$title (exit $code)" }
}

Write-Log "Sentinel evidence reproduction - $stamp"
Write-Log ("PowerShell " + $PSVersionTable.PSVersion.ToString() + " on " + [System.Environment]::OSVersion.VersionString)

if (-not (Test-Path $py)) {
    Invoke-Step "Create virtual environment" "python" @("-m", "venv", ".venv")
}
if (-not $SkipInstall) {
    Invoke-Step "Install requirements" $py @("-m", "pip", "install", "-q", "-r", "requirements.txt")
}
Invoke-Step "Python version" $py @("--version")

if (-not $SkipChecks) {
    Invoke-Step "Tests" $py @("-m", "pytest", "-q", "-p", "no:cacheprovider")
    Invoke-Step "Ruff" (Join-Path $root ".venv\Scripts\ruff.exe") @("check", "sentinel", "tests", "scripts")
    Invoke-Step "mypy" (Join-Path $root ".venv\Scripts\mypy.exe") @("sentinel")
    $node = Get-Command node -ErrorAction SilentlyContinue
    if ($node) {
        $js = Join-Path $out "dashboard_script.js"
        $html = Get-Content -Raw -Encoding UTF8 "sentinel\static\index.html"
        $m = [regex]::Match($html, "<script>([\s\S]*)</script>")
        Set-Content -Path $js -Value $m.Groups[1].Value -Encoding UTF8
        Invoke-Step "Dashboard JavaScript syntax" $node.Source @("--check", $js)
        Remove-Item $js
    } else {
        Write-Log ""
        Write-Log "== Dashboard JavaScript syntax: SKIPPED (node not found on PATH)"
    }
}

Invoke-Step "Generate synthetic demo data and baseline" $py @("-m", "sentinel", "make-demo")
$runDir = Join-Path $out "cli_run"
Invoke-Step "Run demo timeline (CLI)" $py @("-m", "sentinel", "run", "data\demo\demo_timeline.mp4", "--out", $runDir)
if (Test-Path (Join-Path $runDir "incidents.json")) {
    Copy-Item "data\demo\manifest.json" (Join-Path $runDir "demo_manifest.json")
    Invoke-Step "Summarise run" $py @("scripts\summarize_run.py", $runDir, "--manifest", (Join-Path $runDir "demo_manifest.json"))
    Get-ChildItem -Recurse -File $runDir | Get-FileHash -Algorithm SHA256 |
        ForEach-Object { "{0}  {1}" -f $_.Hash.ToLower(), (Resolve-Path -Relative $_.Path) } |
        Set-Content -Path (Join-Path $out "SHA256SUMS.txt") -Encoding UTF8
}

Write-Log ""
if ($failures.Count -eq 0) {
    Write-Log "RESULT: all steps succeeded. Evidence folder: $out"
    exit 0
}
Write-Log ("RESULT: failed steps: " + ($failures -join "; "))
exit 1
