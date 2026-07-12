param(
    [int]$IntervalSeconds = 300,
    [string]$Output = "artifacts\experiments\v4_synthetic_2026_06\V1_v4_synthetic_warmstart_2026_06\pipeline_progress.log"
)

$ErrorActionPreference = "Stop"
$Repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Repo

$Pools = @(
    @{ Name = "selfplay_v4_legacy_3000h"; Logs = "artifacts\experiments\v4_synthetic_2026_06\V1_data\selfplay_v4_12000h_1v3\logs"; Pattern = "*_a.json.gz"; Target = 3000 },
    @{ Name = "selfplay_v4_unique_9000h"; Logs = "artifacts\experiments\v4_synthetic_2026_06\V1_data\selfplay_v4_unique_9000h\logs"; Pattern = "*.json.gz"; Target = 9000 }
)

New-Item -ItemType Directory -Force -Path (Split-Path $Output) | Out-Null

while ($true) {
    $Stamp = Get-Date -Format o
    $Rows = @()
    foreach ($Pool in $Pools) {
        $Files = @(Get-ChildItem -Path $Pool.Logs -Filter $Pool.Pattern -ErrorAction SilentlyContinue)
        $LastWrite = if ($Files.Count -gt 0) {
            ($Files | Sort-Object LastWriteTime -Descending | Select-Object -First 1).LastWriteTime.ToString("o")
        } else {
            "NA"
        }
        $Rows += "$($Pool.Name)=$($Files.Count)/$($Pool.Target) last=$LastWrite"
    }
    $Gpu = ""
    try {
        $Gpu = (nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader,nounits) -join "; "
    } catch {
        $Gpu = "nvidia-smi unavailable"
    }
    "$Stamp | $($Rows -join ' | ') | gpu=$Gpu" | Tee-Object -FilePath $Output -Append
    Start-Sleep -Seconds $IntervalSeconds
}
