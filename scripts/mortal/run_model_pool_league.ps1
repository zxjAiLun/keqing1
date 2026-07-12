param(
    [switch]$Resume,
    [switch]$ReportOnly,
    [int]$MaxLineups = 0
)

$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Repo
$env:UV_PROJECT_ENVIRONMENT = ".venv-win"

$Root = "artifacts\experiments\model_pool_2026_07"
$LogDir = Join-Path $Root "pipeline_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $LogDir "$RunStamp`_run_balanced_league.log"

$Command = @("uv", "run", "--no-sync", "python", "scripts/mortal/run_model_pool_league.py")
if ($Resume) { $Command += "--resume" }
if ($ReportOnly) { $Command += "--report-only" }
if ($MaxLineups -gt 0) { $Command += @("--max-lineups", "$MaxLineups") }

"[$(Get-Date -Format o)] START model pool league" | Tee-Object -FilePath $LogPath -Append
$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    & $Command[0] @($Command[1..($Command.Count - 1)]) *>> $LogPath
    $ExitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $PreviousErrorActionPreference
}
if ($ExitCode -ne 0) { throw "model pool league failed with exit code $ExitCode. See $LogPath" }
"[$(Get-Date -Format o)] DONE model pool league" | Tee-Object -FilePath $LogPath -Append
