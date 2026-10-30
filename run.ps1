# Quick PowerShell launcher for Driver Drowsiness Detection System
Set-Location $PSScriptRoot

if (Test-Path ".venv/bin/Activate.ps1") {
    . ./.venv/bin/Activate.ps1
}

Write-Host "🚀 Starting Driver Drowsiness Detection System on local server..." -ForegroundColor Cyan
streamlit run app.py
