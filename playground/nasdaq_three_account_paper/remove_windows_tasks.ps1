$ErrorActionPreference = "Stop"

$TaskNames = @(
    "SuperTrendQuant-Nasdaq-7Account-Daily",
    "SuperTrendQuant-Nasdaq-7Account-Weekly",
    "SuperTrendQuant-Nasdaq-3Account-Daily",
    "SuperTrendQuant-Nasdaq-3Account-Weekly",
    "SuperTrendQuant-Nasdaq-D-Intraday"
)

foreach ($TaskName in $TaskNames) {
    $Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($null -ne $Task) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Output "Removed: $TaskName"
    }
    else {
        Write-Output "Not found: $TaskName"
    }
}
