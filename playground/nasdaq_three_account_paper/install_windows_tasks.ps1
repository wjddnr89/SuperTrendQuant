$ErrorActionPreference = "Stop"

$LabRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $LabRoot "..\..")).Path
$PowerShellExe = Join-Path $PSHOME "powershell.exe"
$CurrentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

$DailyTaskName = "SuperTrendQuant-Nasdaq-7Account-Daily"
$WeeklyTaskName = "SuperTrendQuant-Nasdaq-7Account-Weekly"
$DailyScript = Join-Path $LabRoot "run_daily.ps1"
$WeeklyScript = Join-Path $LabRoot "run_weekly.ps1"

$Principal = New-ScheduledTaskPrincipal `
    -UserId $CurrentUser `
    -LogonType Interactive `
    -RunLevel Limited

$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 15) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 3)

$DailyAction = New-ScheduledTaskAction `
    -Execute $PowerShellExe `
    -Argument "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$DailyScript`"" `
    -WorkingDirectory $ProjectRoot
$DailyTrigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Tuesday, Wednesday, Thursday, Friday, Saturday `
    -At ([datetime]::Today.AddHours(6).AddMinutes(30))

$WeeklyAction = New-ScheduledTaskAction `
    -Execute $PowerShellExe `
    -Argument "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$WeeklyScript`"" `
    -WorkingDirectory $ProjectRoot
$WeeklyTrigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Saturday `
    -At ([datetime]::Today.AddHours(7))

Register-ScheduledTask `
    -TaskName $DailyTaskName `
    -Action $DailyAction `
    -Trigger $DailyTrigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "Update the seven read-only Nasdaq paper accounts (A-G)." `
    -Force | Out-Null

Register-ScheduledTask `
    -TaskName $WeeklyTaskName `
    -Action $WeeklyAction `
    -Trigger $WeeklyTrigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "Generate the weekly seven-account paper report (A-G)." `
    -Force | Out-Null

foreach ($TaskName in @($DailyTaskName, $WeeklyTaskName)) {
    $Task = Get-ScheduledTask -TaskName $TaskName
    $Info = $Task | Get-ScheduledTaskInfo
    [pscustomobject]@{
        TaskName = $TaskName
        State = $Task.State
        NextRunTime = $Info.NextRunTime
        RunAs = $CurrentUser
    }
}
