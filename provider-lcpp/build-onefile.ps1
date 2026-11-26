# Build single-file thinkfarm-provider.exe for Windows using PyInstaller.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".\venv\Scripts\pyinstaller.exe")) {
    Write-Host "[build-onefile] Installing pyinstaller into .\venv..."
    .\venv\Scripts\pip.exe install --quiet pyinstaller
}

.\venv\Scripts\pyinstaller.exe --clean --noconfirm thinkfarm-provider.spec

Write-Host ""
Write-Host "[build-onefile] Built single-file executable:"
Get-Item .\dist\thinkfarm-provider.exe | Format-Table Name, Length, LastWriteTime

