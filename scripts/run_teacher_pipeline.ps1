param()

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Python not found at $python. Please create the .venv first."
}

$traceRaw = "artifacts/datasets/teacher_trace_raw.jsonl"
$summaryRaw = "artifacts/datasets/teacher_games_summary.jsonl"
$traceBalanced = "artifacts/datasets/teacher_trace_balanced.jsonl"
$summaryBalanced = "artifacts/datasets/teacher_games_balanced.jsonl"
$modelSlow = "artifacts/models/model_tiny_policy_slow.pt"

function Run-Step {
    param(
        [string]$Title,
        [string[]]$CommandArgs
    )
    Write-Host ""
    Write-Host "=== $Title ==="
    & $python @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Title (exit code: $LASTEXITCODE)"
    }
}

Push-Location $repoRoot
try {
    Run-Step "1/5 Export Raw Teacher Data (slow)" @(
        "scripts/export_teacher_data.py",
        "--games", "5000",
        "--workers", "8",
        "--trace-output", $traceRaw,
        "--summary-output", $summaryRaw
    )

    Run-Step "2/5 Curate Balanced Dataset" @(
        "scripts/curate_teacher_dataset.py",
        "--trace-input", $traceRaw,
        "--summary-input", $summaryRaw,
        "--trace-output", $traceBalanced,
        "--summary-output", $summaryBalanced,
        "--split-step", "40"
    )

    Run-Step "3/5 Train Slow Model" @(
        "scripts/train_policy_model.py",
        "--trace", $traceBalanced,
        "--mode", "slow",
        "--output", $modelSlow,
        "--epochs", "12",
        "--batch-size", "256",
        "--device", "cuda"
    )

    Run-Step "4/5 Eval vs Weak (slow)" @(
        "scripts/eval_model_ai.py",
        "--model", $modelSlow,
        "--opponent", "weak",
        "--games", "50",
        "--game-mode", "slow",
        "--device", "cuda"
    )

    Run-Step "5/5 Eval vs Baseline (slow)" @(
        "scripts/eval_model_ai.py",
        "--model", $modelSlow,
        "--opponent", "baseline",
        "--games", "50",
        "--game-mode", "slow",
        "--device", "cuda"
    )

    Write-Host ""
    Write-Host "Pipeline completed successfully."
}
finally {
    Pop-Location
}
