param(
    [switch]$PrepareOnly,
    [switch]$DataOnly,
    [switch]$RunTraining,
    [switch]$RunEvaluations
)

$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Repo
$env:UV_PROJECT_ENVIRONMENT = ".venv-win"

$ExpDir = "artifacts\experiments\model_pool_2026_07\V2_population_mixed_v4_warmstart_2026_07"
$DataRoot = "artifacts\experiments\model_pool_2026_07\V2_data"
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

Invoke-Logged "00_prepare_v2" @(
    "uv", "run", "--no-sync", "python", "scripts/mortal/prepare_v2_population_mixed_warmstart.py"
)

if ($PrepareOnly) { exit 0 }

$Pools = @(
    @{ Name = "v4_70k_t1_v0b"; Seed = "960000"; Models = @(
        "model_v4=artifacts/model_v4_20240308_best_min.pth",
        "70k=artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth",
        "T1_71000=artifacts/experiments/teacher_transfer_2026_05/T1_teacher_ce_01/mortal.pth",
        "V0b_15000=artifacts/experiments/v4_synthetic_2026_06/V0b_v4_synthetic_clean_2026_07/checkpoints/mortal_15000.pth"
    ) },
    @{ Name = "v4_70k_v1_80k"; Seed = "962000"; Models = @(
        "model_v4=artifacts/model_v4_20240308_best_min.pth",
        "70k=artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth",
        "V1_74000=artifacts/experiments/v4_synthetic_2026_06/V1_v4_synthetic_warmstart_2026_06/checkpoints/mortal_v1_74000.pth",
        "80k_game=artifacts/mortal_training/checkpoints/mortal_default_80k_rejected_gate.pth"
    ) },
    @{ Name = "v4_v0b_v1_t1"; Seed = "964000"; Models = @(
        "model_v4=artifacts/model_v4_20240308_best_min.pth",
        "V0b_15000=artifacts/experiments/v4_synthetic_2026_06/V0b_v4_synthetic_clean_2026_07/checkpoints/mortal_15000.pth",
        "V1_74000=artifacts/experiments/v4_synthetic_2026_06/V1_v4_synthetic_warmstart_2026_06/checkpoints/mortal_v1_74000.pth",
        "T1_71000=artifacts/experiments/teacher_transfer_2026_05/T1_teacher_ce_01/mortal.pth"
    ) }
)

foreach ($Pool in $Pools) {
    $Output = Join-Path $DataRoot "$($Pool.Name)_2000h"
    $Smoke = @(
        "uv", "run", "--no-sync", "python", "scripts/mortal/four_player_native.py",
        "--require-cuda", "--device", "cuda", "--seat-mode", "random",
        "--seed-start", $Pool.Seed, "--seed-key", "8192", "--games", "25",
        "--native-batch-games", "25", "--progress-every", "25", "--rank-points", "90,45,0,-135",
        "--output-dir", $Output, "--resume"
    )
    foreach ($Model in $Pool.Models) { $Smoke += @("--model", $Model) }
    Invoke-Logged "01_smoke_$($Pool.Name)" $Smoke

    $Full = @(
        "uv", "run", "--no-sync", "python", "scripts/mortal/four_player_native.py",
        "--require-cuda", "--device", "cuda", "--seat-mode", "random",
        "--seed-start", $Pool.Seed, "--seed-key", "8192", "--games", "2000",
        "--native-batch-games", "100", "--progress-every", "100", "--rank-points", "90,45,0,-135",
        "--output-dir", $Output, "--resume"
    )
    foreach ($Model in $Pool.Models) { $Full += @("--model", $Model) }
    Invoke-Logged "02_generate_$($Pool.Name)" $Full
}

Invoke-Logged "03_audit_v2_data" @(
    "uv", "run", "--no-sync", "python", "scripts/mortal/audit_population_synthetic_dataset.py",
    "--data-root", $DataRoot, "--output", "$ExpDir\dataset_audit.json"
)

if ($DataOnly) { exit 0 }

if ($RunTraining) {
    $StatePath = Join-Path $ExpDir "mortal.pth"
    $Train = @(
        "uv", "run", "--no-sync", "python", "scripts/run_mortal_dqn_offline.py",
        "--config", "$ExpDir\config.toml", "--target-steps", "74000",
        "--device", "cuda", "--num-workers", "0", "--seed", "20260712", "--data-seed", "20260712",
        "--archive-steps", "72000,74000", "--archive-dir", "$ExpDir\checkpoints", "--log-every", "50"
    )
    if (-not (Test-Path $StatePath)) {
        $Train += @(
            "--initialize-from", "artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth",
            "--initial-steps", "70000"
        )
    }
    Invoke-Logged "04_train_v2_74000" $Train
}

if ($RunEvaluations) {
    foreach ($Spec in @(
        @{ Step = "72000"; Games = "250"; Seed = "966000" },
        @{ Step = "74000"; Games = "500"; Seed = "967000" }
    )) {
        Invoke-Logged "05_eval_v2_$($Spec.Step)" @(
            "uv", "run", "--no-sync", "python", "scripts/mortal/four_player_native.py",
            "--require-cuda", "--device", "cuda", "--seat-mode", "random",
            "--seed-start", $Spec.Seed, "--seed-key", "8192", "--games", $Spec.Games,
            "--native-batch-games", "100", "--progress-every", "100", "--rank-points", "90,45,0,-135",
            "--model", "model_v4=artifacts/model_v4_20240308_best_min.pth",
            "--model", "70k=artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth",
            "--model", "T1_71000=artifacts/experiments/teacher_transfer_2026_05/T1_teacher_ce_01/mortal.pth",
            "--model", "V2=$ExpDir\checkpoints\mortal_$($Spec.Step).pth",
            "--output-dir", "$ExpDir\eval_$($Spec.Games)h_v2_$($Spec.Step)"
        )
    }
}
