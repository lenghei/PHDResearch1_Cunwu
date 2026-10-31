$ErrorActionPreference = 'Stop'
$Outer = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$Python = Join-Path $Outer '.venv\Scripts\python.exe'
if (!(Test-Path $Python)) { throw "Python not found: $Python" }
& $Python -B (Join-Path $PSScriptRoot 'run.py') --preflight
if ($LASTEXITCODE -ne 0) { throw "CV261102 preflight failed with exit code $LASTEXITCODE" }
