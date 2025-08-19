Param(
    [string]$CondaEnvName = "ai_trading_py311"
)

$ErrorActionPreference = "Stop"

function Test-Command {
    param([string]$Name)
    $old = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    $null = Get-Command $Name
    $ok = $?
    $ErrorActionPreference = $old
    return $ok
}

Write-Host "==> Checking for conda..."
if (Test-Command conda) {
    Write-Host "==> Conda detected. Creating/updating environment: $CondaEnvName"
    conda env update -n $CondaEnvName -f environment.yml 2>$null
    if (-not $?) {
        conda env create -n $CondaEnvName -f environment.yml
    }
    conda activate $CondaEnvName
    python -m pip install -U pip setuptools wheel
    if (Test-Path "requirements.txt") {
        pip install -r requirements.txt
    }
} else {
    Write-Host "==> Conda not found. Falling back to Python venv."
    if (-not (Test-Command py)) {
        Write-Error "Python launcher 'py' not found. Install Python 3.11+ or install Miniconda."
    }
    if (-not (Test-Path ".venv")) {
        py -3.11 -m venv .venv
    }
    .\.venv\Scripts\Activate.ps1
    python -m pip install -U pip setuptools wheel
    if (Test-Path "requirements.txt") {
        pip install -r requirements.txt
    }
}

Write-Host "==> Environment ready."
Write-Host "==> Running Streamlit app..."
streamlit run streamlit_app.py
