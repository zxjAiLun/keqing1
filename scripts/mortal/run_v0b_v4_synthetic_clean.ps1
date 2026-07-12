param(
    [switch]$PrepareOnly,
    [switch]$SkipAudit,
    [switch]$ResetState,
    [switch]$RunTraining,
    [switch]$RunFinalEval
)

$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Repo
$env:UV_PROJECT_ENVIRONMENT = ".venv-win"

$ExpDir = "artifacts\experiments\v4_synthetic_2026_06\V0b_v4_synthetic_clean_2026_07"
$LogDir = Join-Path $ExpDir "pipeline_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"

function Invoke-Logged {
    param([string]$Name, [string[]]$Command)
    $LogPath = Join-Path $LogDir "$RunStamp`_$Name.log"
    "[$(Get-Date -Format o)] START $Name" | Tee-Object -FilePath $LogPath -Append
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Command[0] @($Command[1..($Command.Count - 1)]) *>> $LogPath
        $ExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($ExitCode -ne 0) { throw "$Name failed with exit code $ExitCode. See $LogPath" }
    "[$(Get-Date -Format o)] DONE $Name" | Tee-Object -FilePath $LogPath -Append
}

$Prepare = @("uv", "run", "--no-sync", "python", "scripts/mortal/prepare_v0b_v4_synthetic_clean.py")
if ($ResetState) { $Prepare += "--reset-state" }
Invoke-Logged "00_prepare_v0b" $Prepare

if (-not $SkipAudit) {
    Invoke-Logged "01_dataset_audit" @("uv", "run", "--no-sync", "python", "scripts/mortal/audit_v4_synthetic_dataset.py")
}

if ($PrepareOnly) { exit 0 }

if ($RunTraining) {
    Invoke-Logged "02_train_v0b_continuous_15000" @(
        "uv", "run", "--no-sync", "python", "scripts/run_mortal_dqn_offline.py",
        "--config", "$ExpDir\config.toml", "--target-steps", "15000",
        "--device", "cuda", "--num-workers", "0", "--seed", "20260711", "--data-seed", "20260711",
        "--archive-steps", "2000,5000,10000,15000", "--archive-dir", "$ExpDir\checkpoints", "--log-every", "50"
    )
}

if ($RunFinalEval) {
    Invoke-Logged "03_eval_v0b_15000_500h" @(
        "uv", "run", "--no-sync", "python", "scripts/mortal/four_player_native.py",
        "--require-cuda", "--device", "cuda", "--seat-mode", "random", "--seed-start", "930000", "--seed-key", "8192",
        "--games", "500", "--native-batch-games", "100", "--progress-every", "100", "--rank-points", "90,45,0,-135",
        "--model", "model_v4=artifacts/model_v4_20240308_best_min.pth",
        "--model", "V1_74000=artifacts/experiments/v4_synthetic_2026_06/V1_v4_synthetic_warmstart_2026_06/checkpoints/mortal_v1_74000.pth",
        "--model", "T1_71000=artifacts/experiments/teacher_transfer_2026_05/T1_teacher_ce_01/mortal.pth",
        "--model", "V0b=$ExpDir\checkpoints\mortal_15000.pth",
        "--output-dir", "$ExpDir\eval_500h_v0b_15000"
    )
}
