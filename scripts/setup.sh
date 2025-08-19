#!/bin/bash
# Setup-Skript für AI Trading Predict (Linux/macOS)

set -e

echo ">>> Erstelle Conda-Umgebung (ai_trading_py311)..."
conda env create -f environment.yml || true
conda activate ai_trading_py311

echo ">>> Installiere Python-Pakete..."
pip install -U pip setuptools wheel
pip install -r requirements.txt

echo ">>> Installation abgeschlossen."
echo ">>> Zum Starten der App: streamlit run streamlit_app.py"
