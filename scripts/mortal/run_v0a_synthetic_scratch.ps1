param(
    [switch]$PrepareOnly,
    [switch]$SkipAudit,
    [switch]$ResetState,
    [switch]$RunSmoke,
    [switch]$RunStage1,
    [switch]$RunProbe
)

$ErrorActionPreference = "Stop"

$Repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Repo
$env:UV_PROJECT_ENVIRONMENT = ".venv-win"

$ExpDir = "artifacts\experiments\v4_synthetic_2026_06\V0a_v4_synthetic_scratch_2026_06"
$LogDir = Join-Path $ExpDir "pipeline_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"

function Invoke-Logged {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$Command
    )
    $LogPath = Join-Path $LogDir "$RunStamp`_$Name.log"
    $CommandLine = ($Command | ForEach-Object {
        if ($_ -match "\s") { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }) -join " "
    "[$(Get-Date -Format o)] START $Name" | Tee-Object -FilePath $LogPath -Append
    $CommandLine | Tee-Object -FilePath $LogPath -Append
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Command[0] @($Command[1..($Command.Count - 1)]) *>> $LogPath
        $ExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($ExitCode -ne 0) {
        throw "$Name failed with exit code $ExitCode. See $LogPath"
    }
    "[$(Get-Date -Format o)] DONE $Name" | Tee-Object -FilePath $LogPath -Append
}

$PrepareCommand = @(
    "uv", "run", "--no-sync", "python", "scripts/mortal/prepare_v0a_v4_synthetic_scratch.py"
)
if ($ResetState) {
    $PrepareCommand += "--reset-state"
}
Invoke-Logged "00_prepare_v0a_config" $PrepareCommand

if (-not $SkipAudit) {
    Invoke-Logged "01_dataset_audit_full" @(
        "uv", "run", "--no-sync", "python", "scripts/mortal/audit_v4_synthetic_dataset.py"
    )
}

if ($PrepareOnly) {
    exit 0
}

if ($RunSmoke) {
    Invoke-Logged "02_train_v0a_400_smoke" @(
        "uv", "run", "--no-sync", "python", "scripts/run_mortal_dqn_offline.py",
        "--config", "artifacts\experiments\v4_synthetic_2026_06\V0a_v4_synthetic_scratch_2026_06\config.toml",
        "--target-steps", "400",
        "--device", "cuda",
        "--num-workers", "0",
        "--log-every", "10"
    )
}

if ($RunStage1) {
    Invoke-Logged "03_train_v0a_2000" @(
        "uv", "run", "--no-sync", "python", "scripts/run_mortal_dqn_offline.py",
        "--config", "artifacts\experiments\v4_synthetic_2026_06\V0a_v4_synthetic_scratch_2026_06\config.toml",
        "--target-steps", "2000",
        "--device", "cuda",
        "--num-workers", "0"
    )
    New-Item -ItemType Directory -Force -Path (Join-Path $ExpDir "checkpoints") | Out-Null
    Copy-Item -Force (Join-Path $ExpDir "mortal.pth") (Join-Path $ExpDir "checkpoints\mortal_v0a_2000.pth")
    Invoke-Logged "04_eval_v0a_2000_100h" @(
        "uv", "run", "--no-sync", "python", "scripts/mortal/four_player_native.py",
        "--require-cuda", "--device", "cuda", "--seat-mode", "random",
        "--seed-start", "810000", "--seed-key", "8192", "--games", "100",
        "--progress-every", "25", "--rank-points", "90,45,0,-135",
        "--model", "70k=artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth",
        "--model", "80k_game=artifacts/mortal_training/checkpoints/mortal_default_80k_rejected_gate.pth",
        "--model", "T1_71000=artifacts/experiments/teacher_transfer_2026_05/T1_teacher_ce_01/mortal.pth",
        "--model", "V0a=artifacts\experiments\v4_synthetic_2026_06\V0a_v4_synthetic_scratch_2026_06\checkpoints\mortal_v0a_2000.pth",
        "--output-dir", "artifacts\experiments\v4_synthetic_2026_06\V0a_v4_synthetic_scratch_2026_06\eval_100h_v0a_2000"
    )
}

if ($RunProbe) {
    Invoke-Logged "05_train_v0a_10000" @(
        "uv", "run", "--no-sync", "python", "scripts/run_mortal_dqn_offline.py",
        "--config", "artifacts\experiments\v4_synthetic_2026_06\V0a_v4_synthetic_scratch_2026_06\config.toml",
        "--target-steps", "10000",
        "--device", "cuda",
        "--num-workers", "0"
    )
    New-Item -ItemType Directory -Force -Path (Join-Path $ExpDir "checkpoints") | Out-Null
    Copy-Item -Force (Join-Path $ExpDir "mortal.pth") (Join-Path $ExpDir "checkpoints\mortal_v0a_10000.pth")
    Invoke-Logged "06_eval_v0a_10000_250h" @(
        "uv", "run", "--no-sync", "python", "scripts/mortal/four_player_native.py",
        "--require-cuda", "--device", "cuda", "--seat-mode", "random",
        "--seed-start", "811000", "--seed-key", "8192", "--games", "250",
        "--progress-every", "25", "--rank-points", "90,45,0,-135",
        "--model", "70k=artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth",
        "--model", "80k_game=artifacts/mortal_training/checkpoints/mortal_default_80k_rejected_gate.pth",
        "--model", "T1_71000=artifacts/experiments/teacher_transfer_2026_05/T1_teacher_ce_01/mortal.pth",
        "--model", "V0a=artifacts\experiments\v4_synthetic_2026_06\V0a_v4_synthetic_scratch_2026_06\checkpoints\mortal_v0a_10000.pth",
        "--output-dir", "artifacts\experiments\v4_synthetic_2026_06\V0a_v4_synthetic_scratch_2026_06\eval_250h_v0a_10000"
    )
}
