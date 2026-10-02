$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot

Set-Location $ProjectRoot

function Invoke-PythonCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$CommandArguments,

        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    Write-Host ""
    Write-Host "========================================"
    Write-Host $Description
    Write-Host "========================================"

    & python @CommandArguments

    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "Flexible Beam Controller-in-the-Loop"
Write-Host "Starting project quality gate..."
Write-Host "Project root: $ProjectRoot"

Invoke-PythonCommand `
    -Description "1. Ruff static analysis" `
    -CommandArguments @(
        "-m",
        "ruff",
        "check",
        "."
    )

Invoke-PythonCommand `
    -Description "2. Ruff formatting verification" `
    -CommandArguments @(
        "-m",
        "ruff",
        "format",
        "--check",
        "."
    )

New-Item `
    -ItemType Directory `
    -Path "reports" `
    -Force | Out-Null

Invoke-PythonCommand `
    -Description "3. Full automated test suite and coverage" `
    -CommandArguments @(
        "-m",
        "pytest",
        "-v",
        "--cov=actuators",
        "--cov=communication",
        "--cov=controllers",
        "--cov=embedded",
        "--cov=plant",
        "--cov=safety",
        "--cov=sensors",
        "--cov=simulation",
        "--cov-report=term-missing",
        "--cov-report=html:reports/coverage",
        "--cov-fail-under=90"
    )

Invoke-PythonCommand `
    -Description "4. Embedded timing experiment" `
    -CommandArguments @(
        "-m",
        "experiments.embedded_timing_control"
    )

Invoke-PythonCommand `
    -Description "5. CAN fault and safety experiment" `
    -CommandArguments @(
        "-m",
        "experiments.can_fault_control"
    )

$RequiredFiles = @(
    "data\embedded_timing_response.csv",
    "data\healthy_can_control.csv",
    "data\faulty_can_control.csv",
    "results\embedded_timing_comparison.png",
    "results\can_fault_control_comparison.png",
    "results\safety_state_timeline.png",
    "reports\coverage\index.html"
)

Write-Host ""
Write-Host "========================================"
Write-Host "6. Generated artifact verification"
Write-Host "========================================"

$MissingFiles = @()

foreach ($RequiredFile in $RequiredFiles) {
    if (Test-Path $RequiredFile) {
        Write-Host "[PASS] $RequiredFile"
    }
    else {
        Write-Host "[FAIL] $RequiredFile"
        $MissingFiles += $RequiredFile
    }
}

if ($MissingFiles.Count -gt 0) {
    throw (
        "Quality gate failed. Missing artifact count: " +
        $MissingFiles.Count
    )
}

Write-Host ""
Write-Host "========================================"
Write-Host "QUALITY GATE PASSED"
Write-Host "========================================"
Write-Host "Static analysis: passed"
Write-Host "Formatting:      passed"
Write-Host "Tests:           passed"
Write-Host "Coverage report: reports\coverage\index.html"
Write-Host "Experiments:     passed"
Write-Host "Artifacts:       verified"
Write-Host ""