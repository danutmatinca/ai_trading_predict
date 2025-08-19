# Setup-Skript für AI Trading Predict (Windows PowerShell)

Write-Output ">>> Erstelle Conda-Umgebung (ai_trading_py311)..."
conda env create -f environment.yml
conda activate ai_trading_py311

Write-Output ">>> Installiere Python-Pakete..."
pip install -U pip setuptools wheel
pip install -r requirements.txt

Write-Output ">>> Installation abgeschlossen."
Write-Output ">>> Zum Starten der App: streamlit run streamlit_app.py"
