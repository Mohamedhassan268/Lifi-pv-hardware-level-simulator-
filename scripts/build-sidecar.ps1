# Build the backend into a single-file executable that Tauri will spawn.
#
# Usage (from repo root):
#   .\scripts\build-sidecar.ps1
#
# Output:
#   dist\lifi-backend.exe
#
# After building, the Tauri config (frontend/src-tauri/tauri.conf.json) picks
# this up via `bundle.externalBin` and ships it inside the installer.

$ErrorActionPreference = "Stop"

# Resolve the repo root regardless of where this script is invoked from.
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$python = Join-Path $repo "venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "venv not found at $python. Create it first: python -m venv venv"
}

# PowerShell 5.1 wraps native stderr in ErrorRecords, which makes `$?` lie.
# We use $LASTEXITCODE directly and invoke each command on its own line so
# splatting can't be misinterpreted.

Write-Host "==> Installing build requirements" -ForegroundColor Cyan
& $python -m pip install --quiet "pyinstaller>=6.5" "fastapi>=0.110" "uvicorn[standard]>=0.27"
if ($LASTEXITCODE -ne 0) {
    throw "pip install failed (exit code $LASTEXITCODE)"
}

Write-Host "==> Building lifi-backend.exe" -ForegroundColor Cyan
& $python -m PyInstaller backend\lifi-backend.spec --clean --noconfirm
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed (exit code $LASTEXITCODE)"
}

$exe = Join-Path $repo "dist\lifi-backend.exe"
if (-not (Test-Path $exe)) {
    throw "Expected $exe but it is missing"
}

Write-Host ""
Write-Host "Sidecar built: $exe" -ForegroundColor Green
Write-Host ""

# Copy alongside the Tauri project under the Rust target triple Tauri expects.
# `tauri build` looks for: src-tauri/binaries/lifi-backend-<target-triple><.exe>
$triple = $null
try {
    $rustcOut = & rustc -vV 2>&1 | Out-String
    $match = [regex]::Match($rustcOut, 'host:\s*(\S+)')
    if ($match.Success) { $triple = $match.Groups[1].Value }
} catch {}
if (-not $triple) { $triple = "x86_64-pc-windows-msvc" }

$dest = Join-Path $repo "frontend\src-tauri\binaries"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
$destExe = Join-Path $dest "lifi-backend-$triple.exe"
Copy-Item -Force $exe $destExe
Write-Host "Copied to: $destExe" -ForegroundColor Green
