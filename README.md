# AI Trading Predict — LSTM (PyTorch & Streamlit)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aitradingpredict-mmsmvxttti9sw64fazr4ee.streamlit.app/)
<!-- Ersetze <DEINE-STREAMLIT-APP-URL> durch die echte Streamlit-Domain, falls du die App öffentlich gehostet hast. -->

Dieses Projekt dient ausschließlich Studien-, Forschungs- und Demonstrationszwecken.
Es handelt sich nicht um ein Finanzwerkzeug, darf nicht als solches verstanden oder verwendet werden und bietet keinerlei finanzielle, steuerliche oder rechtliche Beratung.
Jegliche Nutzung erfolgt auf eigene Verantwortung.
Der Autor übernimmt keine Haftung für Schäden oder Verluste, die aus der Anwendung des Programms entstehen.

---

## Projektbeschreibung

AI Trading Predict ist eine Anwendung zur Vorhersage von Kursen (Aktien, Währungen, Kryptowährungen) auf Basis von Long Short-Term Memory (LSTM)-Netzwerken.
Die App lädt historische Daten aus **Yahoo Finance** (`yfinance`) oder **Stooq** und erstellt darauf Prognosen.

### Einsatzmöglichkeiten
- Aktienkurse (z. B. `AAPL`, `TSLA`)
- Währungspaare (z. B. `EURUSD=X`, `USDJPY=X`)
- Kryptowährungen (z. B. `BTC-USD`, `ETH-USD`, `BTC-ETH`)

Die App unterstützt alle Symbole, die in `yfinance` oder Stooq verfügbar sind.

---

## Repository klonen (öffentlich)
**Direkt (empfohlen):**
```bash
# HTTPS
git clone https://github.com/danutmatinca/ai_trading_predict.git
cd ai_trading_predict

# (Alternative) SSH – setzt eingerichteten SSH-Key bei GitHub voraus
git clone git@github.com:danutmatinca/ai_trading_predict.git
cd ai_trading_predict

# (Alternative) GitHub CLI
gh repo clone danutmatinca/ai_trading_predict
cd ai_trading_predict
```

**Fork & Beitrag (optional):**
```bash
# 1) Auf GitHub "Fork" klicken, dann deinen Fork klonen:
git clone https://github.com/<dein-github-name>/ai_trading_predict.git
cd ai_trading_predict

# 2) Original als upstream eintragen (um Updates zu holen):
git remote add upstream https://github.com/danutmatinca/ai_trading_predict.git
git fetch upstream
git checkout main
git merge upstream/main   # oder: git rebase upstream/main
```

> Hinweis: Das Repo ist **öffentlich**. Forken/Klonen ist jedem eingeloggten GitHub-Nutzer möglich.  
> Lizenz: **Nicht-kommerziell** (siehe weiter unten).

---

## Schnellstart
Wenn Python-Umgebung und Abhängigkeiten bereits installiert sind:
```bash
streamlit run streamlit_app.py
# Browser: http://localhost:8501
```

---

## Installation und Nutzung

### Voraussetzungen
- Python ≥ 3.10 (empfohlen: 3.11)
- Git installiert
- Conda **oder** venv (empfohlen)

### Linux / macOS
```bash
# (im Projektordner)
# Conda-Umgebung
conda env create -f environment.yml
conda activate ai_trading_py311

# oder mit venv/pip
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

### Windows (PowerShell)
```powershell
# (im Projektordner)
# Conda-Umgebung
conda env create -f environment.yml
conda activate ai_trading_py311

# oder mit venv/pip
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

### Anwendung starten

Training (CLI):
```bash
python src/train.py --config configs/default.yaml
```

Vorhersage (CLI):
```bash
python src/predict.py --config configs/default.yaml
```

Streamlit-App (UI):
```bash
streamlit run streamlit_app.py
# Browser: http://localhost:8501
```

---

## Konzept und Technologien

- Backend: Python 3.11, **PyTorch** (LSTM)
- Frontend: **Streamlit** für die interaktive Weboberfläche
- Datenquellen: **yfinance**, **Stooq** (via `pandas-datareader`)
- Konfiguration: YAML, Logging, CSV-Export

---

## Projektstruktur
```
ai_trading_predict/
│
├── configs/                # Konfigurationsdateien
│   └── default.yaml
│
├── models/                 # Gespeicherte Modelle (.pth) – Artefakte
│   └── .gitkeep
│
├── outputs/                # Trainings-Logs, Vorhersagen, CSVs – Artefakte
│   └── .gitkeep
│
├── src/
│   ├── data/               # Datenhandling
│   │   └── datasets.py
│   ├── models/             # ML-Modelle
│   │   └── lstm.py
│   ├── utils/              # Hilfsfunktionen
│   │   └── config.py
│   ├── train.py            # Training des Modells
│   └── predict.py          # Vorhersagen
│
├── streamlit_app.py        # Web-App (Streamlit)
├── requirements.txt        # Python-Dependencies
├── environment.yml         # Conda-Umgebung
├── .gitignore
├── LICENSE
└── README.md
```

---

## Funktionen
- Datenimport aus Yahoo Finance & Stooq
- Wahl des Tickers/Intervalls/Spalte
- LSTM-Training (Fenstergröße, Epochen, Batch-Größe einstellbar)
- Prognose zukünftiger Werte
- Visualisierung (Historie + Forecast)
- CSV-Export
- Streamlit-Interface mit Fortschrittsanzeige

---

## LSTM – Kurz erklärt
**Long Short-Term Memory (LSTM)** ist ein spezielles rekurrentes neuronales Netzwerk (RNN) für Zeitreihen.
Durch interne Speicher- und Gate-Mechanismen kann ein LSTM sowohl kurzfristige Schwankungen als auch längerfristige Trends erfassen und ist damit gut für Finanzzeitreihen geeignet.

---

## Skripte
- `src/train.py` – Trainiert ein LSTM auf historischen Daten
- `src/predict.py` – Führt Vorhersagen mit trainiertem Modell aus
- `streamlit_app.py` – Interaktive Oberfläche für Training & Prognosen
- `setup.sh` / `setup.ps1` – Optionale Setup-Skripte zum Einrichten/Starten

---

## Screenshots
Beispiele der Streamlit-Anwendung (falls vorhanden):
![Screenshot 1](docs/screenshot1.png)
![Screenshot 2](docs/screenshot2.png)
![Screenshot 3](docs/screenshot3.png)

---

## Urheberrecht und Lizenz
© 2025 Danut Matinca. Alle Rechte vorbehalten.

Lizenz: **PolyForm Noncommercial 1.0.0** (Volltext siehe `LICENSE`).  
Nicht-kommerzielle Nutzung ist erlaubt; **kommerzielle Nutzung ist untersagt**, sofern nicht schriftlich genehmigt.
SPDX-Identifier: `Polyform-Noncommercial-1.0.0`.
