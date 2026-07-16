param(
    [switch]$SkipTorch
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\requirements.txt")) {
    throw "Run this script from the MultiSense project root."
}

if (-not (Test-Path ".\.venv")) {
    py -m venv .venv
}

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass | Out-Null
& ".\.venv\Scripts\Activate.ps1"

python -m pip install --upgrade pip setuptools wheel

if (-not $SkipTorch) {
    Write-Host "Installing current CUDA 12.8 PyTorch build..."
    python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
}

python -m pip install -r requirements.txt

if (-not (Test-Path ".\.env")) {
    Copy-Item ".\.env.example" ".\.env"
    Write-Host "Created .env. Add your GOOGLE_API_KEY before using Gemini."
}

Write-Host ""
Write-Host "Setup completed."
Write-Host "Place the final model at checkpoints\best_model.pt"
Write-Host "Then run:"
Write-Host "python app\demo.py --config config\config.yaml --checkpoint checkpoints\best_model.pt"
