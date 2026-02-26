$ISCC = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"

if (-not (Test-Path $ISCC)) {
    Write-Error "ISCC not found at '$ISCC'."
    exit 1
}

& $ISCC .\scripts\FastDataSetup.iss
Write-Host "`nOutput should be in .\dist\FastData-Installer.exe"
