$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-Checked {
    param(
        [Parameter(Mandatory=$true)][string]$FilePath,
        [Parameter(ValueFromRemainingArguments=$true)][string[]]$Args
    )
    & $FilePath @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $FilePath $($Args -join ' ')"
    }
}

function Get-PythonCmd {
    if ($env:PYTHON_BIN) {
        $cmd = Get-Command $env:PYTHON_BIN -ErrorAction SilentlyContinue
        if ($cmd) { return @($env:PYTHON_BIN) }
    }
    if (Get-Command py -ErrorAction SilentlyContinue) { return @("py", "-3") }
    if (Get-Command python -ErrorAction SilentlyContinue) { return @("python") }
    throw "Python not found. Install Python 3.10+ or set PYTHON_BIN to python/py."
}

function Test-PipHealthy {
    param([string]$VenvPython)

    # This import is exactly what is failing in your trace
    $code = "import pip; import pip._internal; import pip._internal.operations.build.metadata as m; print('pip ok')"
    & $VenvPython -c $code | Out-Null
    return ($LASTEXITCODE -eq 0)
}

# --- Info ---
$edition = $PSVersionTable.PSEdition
$ver     = $PSVersionTable.PSVersion
Write-Host "🔍 Running PowerShell $ver ($edition)"

# Repo root (script located in scripts\setup.ps1)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot  = Resolve-Path (Join-Path $scriptDir "..")
Set-Location $repoRoot
Write-Host "📁 Repo root: $repoRoot"

$Venv = ".venv-windows"
$pyParts = Get-PythonCmd
Write-Host "🐍 System Python: $($pyParts -join ' ')"
Write-Host "🧪 Venv folder: $Venv"

$venvPy = Join-Path $repoRoot "$Venv\Scripts\python.exe"

# Create venv if missing
if (-not (Test-Path $venvPy)) {
    Write-Host "➡️ Creating venv..."
    Invoke-Checked $pyParts[0] ($pyParts[1..($pyParts.Count-1)] + @("-m","venv",$Venv))
}

# If pip is broken, rebuild the venv (most reliable fix)
if (-not (Test-PipHealthy -VenvPython $venvPy)) {
    Write-Host "⚠️ pip is broken in $Venv — rebuilding venv..."
    Remove-Item -Recurse -Force $Venv -ErrorAction SilentlyContinue

    Invoke-Checked $pyParts[0] ($pyParts[1..($pyParts.Count-1)] + @("-m","venv",$Venv))

    if (-not (Test-Path $venvPy)) {
        throw "Venv python not found after rebuild: $venvPy"
    }
}

# Bootstrap pip from stdlib and then upgrade tooling
Write-Host "🛠️ Bootstrapping pip (ensurepip)..."
Invoke-Checked $venvPy @("-m","ensurepip","--upgrade")

Write-Host "⬆️ Upgrading pip/setuptools/wheel (force clean reinstall)..."
Invoke-Checked $venvPy @("-m","pip","install","--upgrade","--force-reinstall","--no-cache-dir","pip","setuptools","wheel")

# Install dependencies
if (Test-Path "requirements.txt") {
    Write-Host "📦 Installing from requirements.txt..."
    Invoke-Checked $venvPy @("-m","pip","install","-r","requirements.txt")
}
elseif (Test-Path "pyproject.toml") {
    Write-Host "📦 Installing from pyproject.toml (editable)..."
    Invoke-Checked $venvPy @("-m","pip","install","-e",".")
}
else {
    Write-Host "⚠️ No requirements.txt or pyproject.toml found. Skipping dependency install."
}

Write-Host "✅ Windows env ready at $Venv"
Write-Host "   Activate: `"$Venv\Scripts\Activate.ps1`""