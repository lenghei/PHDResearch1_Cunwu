param(
    [Parameter(Mandatory=$true)][string]$Package,
    [Parameter(Mandatory=$true)][string]$DataRoot,
    [Parameter(Mandatory=$true)][string]$TrainingParent,
    [string]$Authorization,
    [string]$CPUReceipt,
    [string]$ControlParent,
    [string]$Python='python',
    [switch]$Execute,
    [switch]$DryRun
)
$ErrorActionPreference='Stop'
if ($Execute -and $DryRun) { throw 'Choose Execute or DryRun.' }
if ($Execute -and (!$Authorization -or !$CPUReceipt)) { throw 'New evaluation authorization and target CPU receipt are required.' }
$Entry=Join-Path $Package 'main/experiments/vm48_formal_rq4/evaluation48.py'
$Arguments=@('-B','-X','utf8',$Entry,'--package',$Package,'--data-root',$DataRoot,'--training-parent',$TrainingParent)
if ($Authorization) { $Arguments+=@('--authorization',$Authorization) }
if ($CPUReceipt) { $Arguments+=@('--cpu-receipt',$CPUReceipt) }
if ($ControlParent) { $Arguments+=@('--control-parent',$ControlParent) }
if ($Execute) { $Arguments+='--execute' } else { $Arguments+='--dry-run' }
& $Python @Arguments
exit $LASTEXITCODE
