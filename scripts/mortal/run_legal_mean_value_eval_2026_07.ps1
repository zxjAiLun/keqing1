param(
    [string]$ExperimentId = "legal_mean_value_ab_2026_07",
    [int[]]$Seeds = @(20260803, 20260804, 20260805),
    [int]$SeedStartBase = 1400000,
    [int]$Games = 1000,
    [int]$NativeBatchGames = 25,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Repo
$Python = Join-Path $Repo ".venv-win\Scripts\python.exe"
$ExpDir = Join-Path $Repo "artifacts\experiments\model_pool_2026_07\$ExperimentId"
$EvalDir = Join-Path $ExpDir "eval_1000h"
$LogDir = Join-Path $EvalDir "runner_logs"
$Parent = Join-Path $Repo "artifacts\mortal_training\checkpoints\mortal_default_70k_promoted_candidate.pth"
$External = Join-Path $Repo "artifacts\external_mortal_20240308_best_min.pth"

if (-not (Test-Path -LiteralPath $Python)) { throw "Windows venv Python is missing: $Python" }
if (-not (Test-Path -LiteralPath $Parent)) { throw "70k parent is missing: $Parent" }
if (-not (Test-Path -LiteralPath $External)) { throw "External Mortal reference is missing: $External" }
if ($Games -ne 1000) { throw "Formal evaluation must use exactly 1000 hanchans" }
if ($NativeBatchGames -ne 25) { throw "Formal evaluation must use native batch 25" }
if ($Seeds.Count -ne 3) { throw "Formal evaluation requires exactly three training seeds" }

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$SeedStarts = @{}
for ($Index = 0; $Index -lt $Seeds.Count; $Index++) {
    $SeedStarts[$Seeds[$Index]] = $SeedStartBase + (10000 * $Index)
}

function Invoke-Logged {
    param(
        [int]$TrainingSeed,
        [int]$EvalSeedStart,
        [string[]]$Arguments
    )
    $Name = "seed_$TrainingSeed"
    $LogPath = Join-Path $LogDir "$Name.log"
    "[$(Get-Date -Format o)] START training_seed=$TrainingSeed eval_seed_start=$EvalSeedStart games=$Games native_batch=$NativeBatchGames" | Tee-Object -FilePath $LogPath -Append
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python @Arguments 2>&1 | Tee-Object -FilePath $LogPath -Append
    $ExitCode = $LASTEXITCODE
    $ErrorActionPreference = $PreviousErrorActionPreference
    if ($ExitCode -ne 0) {
        throw "seed $TrainingSeed failed with exit code $ExitCode. See $LogPath"
    }
    "[$(Get-Date -Format o)] DONE training_seed=$TrainingSeed" | Tee-Object -FilePath $LogPath -Append
}

foreach ($Seed in $Seeds) {
    $RunDir = Join-Path $EvalDir "seed_$Seed"
    $Control = Join-Path $ExpDir "C_behavior_action_mc\seed_$Seed\mortal.pth"
    $Variant = Join-Path $ExpDir "V_legal_mean_mc\seed_$Seed\mortal.pth"
    if (-not (Test-Path -LiteralPath $Control)) { throw "control checkpoint is missing: $Control" }
    if (-not (Test-Path -LiteralPath $Variant)) { throw "variant checkpoint is missing: $Variant" }

    $Args = @(
        "scripts\mortal\four_player_native.py",
        "--model", "70k=$Parent",
        "--model", "ext_mortal=$External",
        "--model", "C_behavior_action_mc_$Seed=$Control",
        "--model", "V_legal_mean_mc_$Seed=$Variant",
        "--output-dir", $RunDir,
        "--device", "cuda",
        "--require-cuda",
        "--seed-start", "$($SeedStarts[$Seed])",
        "--seed-key", "8192",
        "--games", "$Games",
        "--seat-mode", "random",
        "--progress-every", "25",
        "--native-batch-games", "$NativeBatchGames",
        "--rank-points", "90,45,0,-135",
        "--profile"
    )
    if ($Resume) { $Args += "--resume" }
    Invoke-Logged -TrainingSeed $Seed -EvalSeedStart $SeedStarts[$Seed] -Arguments $Args
}

"legal_mean_value formal eval completed: games=$Games native_batch=$NativeBatchGames seeds=$($Seeds -join ',')" | Tee-Object -FilePath (Join-Path $LogDir "formal_eval.done")
