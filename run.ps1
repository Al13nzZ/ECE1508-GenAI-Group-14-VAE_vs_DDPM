param(
    [ValidateSet("quick", "standard", "report")]
    [string]$Profile = "report"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Virtual environment missing. Create .venv and install requirements first."
}

$env:EXPERIMENT_PROFILE = $Profile
$env:PYTHONUNBUFFERED = "1"
Set-Location -LiteralPath $ProjectRoot
& $PythonExe complete_project.py
