param(
    [switch]$DataOnly,
    [switch]$SkipGeneration,
    [switch]$SkipAudit,
    [switch]$RunStage1,
    [switch]$RunFinal
)

$ErrorActionPreference = "Stop"

$Repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Repo
$env:UV_PROJECT_ENVIRONMENT = ".venv-win"

$ExpDir = "artifacts\experiments\v4_synthetic_2026_06\V1_v4_synthetic_warmstart_2026_06"
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

if (-not $SkipGeneration) {
    Invoke-Logged "01_selfplay_v4_12000h_1v3" @(
        "uv", "run", "--no-sync", "python", "scripts/mortal/one_vs_three_smoke.py",
        "--challenger", "artifacts/model_v4_20240308_best_min.pth",
        "--champion", "artifacts/model_v4_20240308_best_min.pth",
        "--device", "cuda",
        "--seed-start", "810000",
        "--seed-key", "8192",
        "--seed-count", "3000",
        "--progress-every", "25",
        "--rank-points", "90,45,0,-135",
        "--challenger-label", "challenger",
        "--champion-label", "champion",
        "--platform-model-label", "v4",
        "--output-dir", "artifacts\experiments\v4_synthetic_2026_06\V1_data\selfplay_v4_12000h_1v3",
        "--resume"
    )

    Invoke-Logged "02_mix_v4_70k_T1_4000h" @(
        "uv", "run", "--no-sync", "python", "scripts/mortal/four_player_native.py",
        "--require-cuda",
        "--device", "cuda",
        "--seat-mode", "random",
        "--seed-start", "830000",
        "--seed-key", "8192",
        "--games", "4000",
        "--progress-every", "25",
        "--rank-points", "90,45,0,-135",
        "--model", "v4_a=artifacts/model_v4_20240308_best_min.pth",
        "--model", "v4_b=artifacts/model_v4_20240308_best_min.pth",
        "--model", "70k=artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth",
        "--model", "T1_71000=artifacts/experiments/teacher_transfer_2026_05/T1_teacher_ce_01/mortal.pth",
        "--output-dir", "artifacts\experiments\v4_synthetic_2026_06\V1_data\mix_v4_70k_T1_4000h",
        "--resume"
    )

    Invoke-Logged "03_mix_v4_80k_T1_4000h" @(
        "uv", "run", "--no-sync", "python", "scripts/mortal/four_player_native.py",
        "--require-cuda",
        "--device", "cuda",
        "--seat-mode", "random",
        "--seed-start", "850000",
        "--seed-key", "8192",
        "--games", "4000",
        "--progress-every", "25",
        "--rank-points", "90,45,0,-135",
        "--model", "v4_a=artifacts/model_v4_20240308_best_min.pth",
        "--model", "v4_b=artifacts/model_v4_20240308_best_min.pth",
        "--model", "80k_game=artifacts/mortal_training/checkpoints/mortal_default_80k_rejected_gate.pth",
        "--model", "T1_71000=artifacts/experiments/teacher_transfer_2026_05/T1_teacher_ce_01/mortal.pth",
        "--output-dir", "artifacts\experiments\v4_synthetic_2026_06\V1_data\mix_v4_80k_T1_4000h",
        "--resume"
    )
}

if (-not $SkipAudit) {
    Invoke-Logged "04_dataset_audit_full" @(
        "uv", "run", "--no-sync", "python", "scripts/mortal/audit_v4_synthetic_dataset.py"
    )
}

if ($DataOnly) {
    exit 0
}

if ($RunStage1) {
    Invoke-Logged "05_train_v1_74000" @(
        "uv", "run", "--no-sync", "python", "scripts/run_mortal_dqn_offline.py",
        "--config", "artifacts\experiments\v4_synthetic_2026_06\V1_v4_synthetic_warmstart_2026_06\config.toml",
        "--target-steps", "74000",
        "--device", "cuda",
        "--num-workers", "0"
    )
    New-Item -ItemType Directory -Force -Path (Join-Path $ExpDir "checkpoints") | Out-Null
    Copy-Item -Force (Join-Path $ExpDir "mortal.pth") (Join-Path $ExpDir "checkpoints\mortal_v1_74000.pth")
    Invoke-Logged "06_eval_v1_74000_250h" @(
        "uv", "run", "--no-sync", "python", "scripts/mortal/four_player_native.py",
        "--require-cuda", "--device", "cuda", "--seat-mode", "random",
        "--seed-start", "790000", "--seed-key", "8192", "--games", "250",
        "--progress-every", "25", "--rank-points", "90,45,0,-135",
        "--model", "70k=artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth",
        "--model", "80k_game=artifacts/mortal_training/checkpoints/mortal_default_80k_rejected_gate.pth",
        "--model", "T1_71000=artifacts/experiments/teacher_transfer_2026_05/T1_teacher_ce_01/mortal.pth",
        "--model", "V1=artifacts\experiments\v4_synthetic_2026_06\V1_v4_synthetic_warmstart_2026_06\checkpoints\mortal_v1_74000.pth",
        "--output-dir", "artifacts\experiments\v4_synthetic_2026_06\V1_v4_synthetic_warmstart_2026_06\eval_250h_v1_74000"
    )
}

if ($RunFinal) {
    Invoke-Logged "07_train_v1_80000" @(
        "uv", "run", "--no-sync", "python", "scripts/run_mortal_dqn_offline.py",
        "--config", "artifacts\experiments\v4_synthetic_2026_06\V1_v4_synthetic_warmstart_2026_06\config.toml",
        "--target-steps", "80000",
        "--device", "cuda",
        "--num-workers", "0"
    )
    New-Item -ItemType Directory -Force -Path (Join-Path $ExpDir "checkpoints") | Out-Null
    Copy-Item -Force (Join-Path $ExpDir "mortal.pth") (Join-Path $ExpDir "checkpoints\mortal_v1_80000.pth")
    Invoke-Logged "08_eval_v1_80000_1000h" @(
        "uv", "run", "--no-sync", "python", "scripts/mortal/four_player_native.py",
        "--require-cuda", "--device", "cuda", "--seat-mode", "random",
        "--seed-start", "800000", "--seed-key", "8192", "--games", "1000",
        "--progress-every", "25", "--rank-points", "90,45,0,-135",
        "--model", "70k=artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth",
        "--model", "80k_game=artifacts/mortal_training/checkpoints/mortal_default_80k_rejected_gate.pth",
        "--model", "T1_71000=artifacts/experiments/teacher_transfer_2026_05/T1_teacher_ce_01/mortal.pth",
        "--model", "V1=artifacts\experiments\v4_synthetic_2026_06\V1_v4_synthetic_warmstart_2026_06\checkpoints\mortal_v1_80000.pth",
        "--output-dir", "artifacts\experiments\v4_synthetic_2026_06\V1_v4_synthetic_warmstart_2026_06\eval_1000h_v1_80000"
    )
}
