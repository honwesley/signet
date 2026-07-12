$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$packageRoot = Join-Path $env:USERPROFILE "SIGNET-package"
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$models = Join-Path $repoRoot "models"

if (-not (Test-Path $python)) {
    throw "Virtual environment not found: $python"
}

$requiredModels = @(
    "hand_landmarker.task",
    "asl_classifier.joblib",
    "motion_classifier.joblib"
)

foreach ($model in $requiredModels) {
    $modelPath = Join-Path $models $model

    if (-not (Test-Path $modelPath)) {
        throw "Missing model: $modelPath"
    }
}

Stop-Process -Name SIGNET -Force -ErrorAction SilentlyContinue

Remove-Item `
    -Recurse `
    -Force `
    (Join-Path $packageRoot "build") `
    -ErrorAction SilentlyContinue

Remove-Item `
    -Recurse `
    -Force `
    (Join-Path $packageRoot "dist") `
    -ErrorAction SilentlyContinue

Push-Location $repoRoot

try {
    & $python -m PyInstaller `
        --noconfirm `
        --clean `
        --windowed `
        --onedir `
        --name SIGNET `
        --paths "$repoRoot\app" `
        --workpath "$packageRoot\build" `
        --distpath "$packageRoot\dist" `
        --add-data "$models;models" `
        --collect-all customtkinter `
        --collect-all mediapipe `
        --collect-all pyttsx3 `
        --collect-all sklearn `
        --collect-submodules sklearn `
        --collect-all scipy `
        --hidden-import pyttsx3.drivers.sapi5 `
        "$repoRoot\app\gui_app.py"

    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Build completed successfully."
Write-Host "Application:"
Write-Host "$packageRoot\dist\SIGNET\SIGNET.exe"