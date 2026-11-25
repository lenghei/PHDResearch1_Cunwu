param(
    [Parameter(Mandatory=$true)][string]$Package,
    [Parameter(Mandatory=$true)][string]$DataRoot,
    [Parameter(Mandatory=$true)][string]$Output,
    [ValidateSet('formal','ablations','benchmark')][string]$Mode='formal',
    [ValidateSet('YOLO26s','YOLOv8s')][string]$Architecture,
    [string]$Authorization,
    [string]$CPUReceipt,
    [string]$Python='python',
    [switch]$Execute,
    [switch]$DryRun
)
$ErrorActionPreference='Stop'
if ($Execute -and $DryRun) { throw 'Choose Execute or DryRun.' }
if ($Execute -and (!$Authorization -or !$CPUReceipt)) { throw 'New arm-specific authorization and target CPU receipt are required.' }
$Entry=Join-Path $Package 'main/experiments/vm48_formal_rq4/runtime.py'
$Arguments=@('-B','-X','utf8',$Entry,'--package',$Package,'--data-root',$DataRoot,'--output',$Output,'--mode',$Mode)
if ($Architecture) { $Arguments+=@('--architecture',$Architecture) }
if ($Authorization) { $Arguments+=@('--authorization',$Authorization) }
if ($CPUReceipt) { $Arguments+=@('--cpu-receipt',$CPUReceipt) }
if ($Execute) { $Arguments+='--execute' } else { $Arguments+='--dry-run' }
& $Python @Arguments
exit $LASTEXITCODE
