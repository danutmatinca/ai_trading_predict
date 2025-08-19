#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="ai_trading_predict"
CONDA_ENV_NAME="ai_trading_py311"

echo "==> Checking for conda..."
if command -v conda >/dev/null 2>&1; then
    echo "==> Conda detected. Creating environment: ${CONDA_ENV_NAME}"
    # Use mamba if available for speed
    if command -v mamba >/dev/null 2>&1; then
        mamba env update -n "${CONDA_ENV_NAME}" -f environment.yml || mamba env create -n "${CONDA_ENV_NAME}" -f environment.yml
    else
        conda env update -n "${CONDA_ENV_NAME}" -f environment.yml || conda env create -n "${CONDA_ENV_NAME}" -f environment.yml
    fi
    # shellcheck disable=SC1091
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV_NAME}"
    python -m pip install -U pip setuptools wheel
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi
else
    echo "==> Conda not found. Falling back to Python venv."
    PYTHON_BIN="${PYTHON_BIN:-python3}"
    if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
        echo "Error: python3 not found. Install Python 3.11+ and retry." >&2
        exit 1
    fi
    echo "==> Creating virtual environment: .venv"
    "${PYTHON_BIN}" -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    python -m pip install -U pip setuptools wheel
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi
fi

echo "==> Environment ready."
echo "==> Running Streamlit app..."
exec streamlit run streamlit_app.py
