$ErrorActionPreference = "Stop"
$LabRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $LabRoot "..\..")
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$LogDir = Join-Path $LabRoot "logs"
$LogPath = Join-Path $LogDir ("weekly-" + (Get-Date -Format "yyyy-MM-dd-HHmmss") + ".log")

Set-Location $ProjectRoot
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

if (-not (Test-Path -LiteralPath $Python)) {
    "Python not found: $Python" | Tee-Object -FilePath $LogPath -Append
    exit 1
}

$ErrorActionPreference = "Continue"
& $Python (Join-Path $LabRoot "run_weekly.py") 2>&1 |
    Tee-Object -FilePath $LogPath -Append
$ExitCode = $LASTEXITCODE

exit $ExitCode
