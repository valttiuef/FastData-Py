$ErrorActionPreference = "Stop"

# Detect version & edition
$edition = $PSVersionTable.PSEdition
$ver     = $PSVersionTable.PSVersion

Write-Host "🔍 Running PowerShell $ver ($edition)"

# Get script folder and go to repo root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $scriptDir
Set-Location ..

# Choose venv folder name based on OS/edition (you can change this if you want)
$Venv = ".venv-windows"
$py = $env:PYTHON_BIN
if (-not $py) { $py = "py" }

# Create venv if it doesn't exist
if (-not (Test-Path $Venv)) {
    & $py -m venv $Venv
}

# Install deps
& "$Venv\Scripts\python.exe" -m pip install --upgrade pip
if (Test-Path "requirements.txt") {
    & "$Venv\Scripts\python.exe" -m pip install -r requirements.txt
}

Write-Host "✅ Windows env ready at $Venv"
