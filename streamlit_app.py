# SPDX-License-Identifier: Polyform-Noncommercial-1.0.0

from __future__ import annotations

import sys
import subprocess
import io
import pickle
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import streamlit as st
import yfinance as yf

# --------------
# Streamlit-Layout zuerst setzen
# --------------
st.set_page_config(page_title="AI Trading Forecast", layout="wide")

# -------------
# Projektpfade (damit wir src/* importieren können)
# -------------
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.config import load_config
from models.lstm import LSTMForecaster

# ---------------
# Helfer
# ---------------
def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in str(name))

def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = {**base}
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out

def write_cfg_tmp(base_cfg_path: str, overrides: Dict[str, Any]) -> Path:
    """YAML laden, Felder überschreiben, temporäre YAML schreiben, Pfad zurückgeben."""
    base = load_config(base_cfg_path)
    merged = deep_update(base, overrides)
    import yaml  # lazy import
    tmp = ROOT / "outputs" / f"_tmp_cfg_{sanitize(overrides['data']['ticker'])}.yaml"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, allow_unicode=True, sort_keys=False)
    return tmp

# ---------------
# Datenbeschaffung
# ---------------
def fetch_series_yf(ticker: str, start: str | None, end: str | None,
                    interval: str, column: str) -> pd.Series:
    df = yf.download(
        ticker, start=start, end=end, interval=interval,
        auto_adjust=False, group_by="column", progress=False
    )
    # MultiIndex -> abflachen (Einzelticker)
    if isinstance(df.columns, pd.MultiIndex):
        if len(df.columns.levels[-1]) == 1:
            df.columns = df.columns.get_level_values(0)
    else:
        # 'Close AAPL' -> 'Close'
        ren = {c: c.split(" ")[0] for c in df.columns if isinstance(c, str) and " " in c}
        if ren:
            df = df.rename(columns=ren)

    wanted = column
    if wanted not in df.columns and wanted == "Adj Close" and "Close" in df.columns:
        wanted = "Close"
    if wanted not in df.columns:
        raise ValueError(f"Spalte '{column}' nicht gefunden. Verfügbar: {list(df.columns)}")

    s = df[wanted].dropna().astype("float32")
    s.name = f"{ticker}_{wanted}"
    return s


def fetch_series_stooq(ticker: str, column: str) -> pd.Series:
    try:
        from pandas_datareader import data as pdr
    except ImportError:
        raise ImportError("Bitte installiere pandas-datareader: pip install pandas-datareader")
    df = pdr.DataReader(ticker, "stooq").sort_index()
    wanted = "Close" if column == "Adj Close" else column
    if wanted not in df.columns:
        raise ValueError(f"Spalte '{column}' nicht gefunden. Verfügbar: {list(df.columns)}")
    s = df[wanted].dropna().astype("float32")
    s.name = f"{ticker}_{wanted}"
    return s


def fetch_series_from_source(ticker: str, start: str | None, end: str | None,
                             interval: str, column: str, source: str) -> pd.Series:
    if (source or "yfinance").lower() == "stooq":
        return fetch_series_stooq(ticker, column)
    return fetch_series_yf(ticker, start, end, interval, column)


def build_future_index(s: pd.Series, horizon: int, interval: str) -> pd.DatetimeIndex:
    freq = getattr(s.index, "freq", None) or pd.infer_freq(s.index)
    if freq:
        from pandas.tseries.frequencies import to_offset
        start = s.index[-1] + to_offset(freq)
        return pd.date_range(start=start, periods=horizon, freq=freq)

    iv = interval.lower()
    if "wk" in iv:
        step = pd.Timedelta(weeks=1)
    elif "mo" in iv:
        step = pd.DateOffset(months=1)
    else:
        step = pd.Timedelta(days=1)

    cur = s.index[-1]
    idx = []
    for _ in range(horizon):
        cur = cur + step
        idx.append(cur)
    return pd.DatetimeIndex(idx)


def load_model_and_scaler_from_cfg(cfg_path: Path):
    cfg = load_config(str(cfg_path))
    model_path = Path(cfg["paths"]["best_model"])
    model_dir = Path(cfg["paths"]["model_dir"])

    state = torch.load(model_path, map_location="cpu")
    model = LSTMForecaster(
        input_size=1,
        hidden_size=int(cfg["model"]["hidden_size"]),
        num_layers=int(cfg["model"]["num_layers"]),
        dropout=float(cfg["model"]["dropout"]),
        horizon=int(cfg["window"]["horizon"]),
    )
    model.load_state_dict(state)
    model.eval()

    with (model_dir / "scaler.pkl").open("rb") as f:
        scaler = pickle.load(f)
    return cfg, model, scaler


def make_forecast(series: pd.Series, model: torch.nn.Module, scaler,
                  window: int, horizon: int) -> np.ndarray:
    if len(series) < window:
        raise ValueError(f"Zu wenig Datenpunkte ({len(series)}) für window={window}.")
    last_window = series.values[-window:].reshape(-1, 1)
    last_scaled = scaler.transform(last_window).astype("float32")
    x = torch.from_numpy(last_scaled[None, :, :])
    with torch.no_grad():
        pred_scaled = model(x).cpu().numpy().reshape(-1)
    inv = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
    return inv


# ----------------
st.markdown(
    '''
    <p style="font-size:18px;">Dieses Projekt dient ausschließlich Studien-, Forschungs- und Demonstrationszwecken!!!</p>
    ''',
    unsafe_allow_html=True
)
# Kopfbereich
# ----------------
st.title("📈📉 AI Stock & Crypto Prediction")
st.caption("Interaktive App: Daten laden → LSTM trainieren → Vorhersage visualisieren")

# ---------------
# Sidebar – Konfiguration
# ---------------

st.sidebar.title("⚙️ Einstellungen")
cfg_path_in = st.sidebar.text_input("Konfiguration (YAML)", "configs/default.yaml")
data_source = st.sidebar.selectbox("Datenquelle", ["yfinance", "stooq"], index=0)

# Mini-Check Build-Tools (gegen _distutils_hack-Fehler)
try:
    import setuptools  # noqa
except Exception:
    st.warning("Setuptools fehlt/alt. In Env installieren: `python -m pip install -U pip setuptools wheel`")

# Defaults aus YAML ziehen
try:
    cfg_defaults = load_config(cfg_path_in)
except Exception:
    cfg_defaults = {
        "data": {"ticker": "AAPL", "interval": "1d", "column": "Adj Close", "start": "2015-01-01", "end": None},
        "window": {"size": 60, "horizon": 5},
        "training": {"epochs": 30, "batch_size": 64, "device": "cpu"},
        "paths": {"model_dir": "models", "best_model": "models/best_model.pth", "outputs_dir": "outputs"},
        "model": {"hidden_size": 64, "num_layers": 2, "dropout": 0.1},
    }

ticker = st.sidebar.text_input("Ticker", cfg_defaults["data"]["ticker"])
interval = st.sidebar.selectbox("Intervall", ["1d", "1wk", "1mo"],
                                index=["1d", "1wk", "1mo"].index(cfg_defaults["data"]["interval"]))
column = st.sidebar.selectbox("Preisspalte", ["Adj Close", "Close"],
                              index=0 if cfg_defaults["data"]["column"] == "Adj Close" else 1)
window = st.sidebar.number_input("Fenstergröße (Vergangenheit)", 10, 365*3,
                                 int(cfg_defaults["window"]["size"]), 5)
horizon = st.sidebar.number_input("Horizont (Zukunft)", 1, 90,
                                  int(cfg_defaults["window"]["horizon"]), 1)
epochs = st.sidebar.number_input("Epochen (Training)", 1, 2000,
                                 int(cfg_defaults["training"]["epochs"]), 1)
batch = st.sidebar.number_input("Batch‑Größe (Training)", 1, 8192,
                                int(cfg_defaults["training"]["batch_size"]), 1)
device = st.sidebar.selectbox("Device", ["cpu", "cuda"],
                              index=0 if str(cfg_defaults["training"]["device"]).lower() != "cuda" else 1)

st.sidebar.markdown("---")
col_btn1, col_btn2 = st.sidebar.columns(2)
train_btn = col_btn1.button("️️🏃‍♂️‍➡️ Trainieren")
run_btn = col_btn2.button("🔮 Vorhersagen")
st.sidebar.caption("Modelle werden ticker‑spezifisch unter `models/<TICKER>/` gespeichert.")
st.sidebar.markdown("---")

# Laufzeit‑Config (ticker‑spezifische Pfade)
safe_t = sanitize(ticker)
model_dir_t = ROOT / "models" / safe_t
overrides_common = {
    "data": {"ticker": ticker, "interval": interval, "column": column},
    "window": {"size": int(window), "horizon": int(horizon)},
    "training": {"epochs": int(epochs), "batch_size": int(batch), "device": device},
    "paths": {
        "model_dir": str(model_dir_t),
        "best_model": str(model_dir_t / "best_model.pth"),
        "outputs_dir": str(ROOT / "outputs"),
    },
}

# --------------
# Historie laden & plotten
# --------------
try:
    s = fetch_series_from_source(
        ticker,
        cfg_defaults["data"].get("start"),
        cfg_defaults["data"].get("end"),
        interval,
        column,
        data_source,
    )
    st.subheader(f"Historie – {ticker}")
    st.line_chart(s.tail(400), height=240)
except Exception as e:
    st.error(f"Daten konnten nicht geladen werden: {e}")
    st.stop()

# --------------
# Trainieren – mit horizontaler Progressbar + Live-Logs
# --------------
if train_btn:
    st.subheader(f"Training – {ticker}")
    cfg_tmp_path = write_cfg_tmp(cfg_path_in, overrides_common)
    model_dir_t.mkdir(parents=True, exist_ok=True)

    # Widgets
    status = st.status("Starte Training …", expanded=True)
    prog_bar = st.progress(0)           # <-- horizontaler Balken
    prog_text = st.empty()
    log_box = st.empty()

    # Subprocess aufrufen
    cmd = [sys.executable, str(SRC / "train.py"), "--config", str(cfg_tmp_path)]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as p:

        total_epochs = int(epochs) if epochs else 1
        cur_epoch = 0
        logs = []

        for line in p.stdout or []:
            logs.append(line.rstrip("\n"))
            log_box.code("\n".join(logs[-80:]), language="bash")

            sline = line.strip()
            # Zeilen wie "[003] train_loss=..." parsen
            if sline.startswith("[") and "]" in sline:
                tag = sline.split("]", 1)[0].lstrip("[")
                if tag.isdigit():
                    cur_epoch = int(tag)
                    pct = int(min(100, max(0, cur_epoch/  max(1, total_epochs) * 100)))
                    prog_bar.progress(pct)                         # <-- Balken-Update
                    prog_text.text(f"Epoche {cur_epoch}/{total_epochs}  |  Fortschritt: {pct}%")

        ret = p.wait()

    if ret == 0:
        prog_bar.progress(100)
        status.update(label=f"Fertig. Modell gespeichert: {model_dir_t}", state="complete")
        st.success(f"Training beendet. Modell: `{model_dir_t / 'best_model.pth'}`")
        # Cache leeren, damit beim Forecast neu geladen wird
        try:
            st.cache_resource.clear()
        except Exception:
            pass
    else:
        status.update(label="Training fehlgeschlagen – bitte Logs prüfen.", state="error")
        st.stop()

# ------------
# Forecast
# ------------
if run_btn:
    try:
        cfg_tmp_path = write_cfg_tmp(cfg_path_in, overrides_common)
        cfg, model, scaler = load_model_and_scaler_from_cfg(cfg_tmp_path)

        # Mini-Progress für die Vorhersage
        f_prog = st.progress(0)
        f_prog.text("Skaliere Eingabefenster …")

        forecast_vals = make_forecast(s, model, scaler, int(window), int(horizon))
        f_prog.progress(60)
        fidx = build_future_index(s, int(horizon), interval)
        f_prog.progress(100)

        st.subheader("Vorhersage")
        df_forecast = pd.DataFrame({"timestamp": fidx, "forecast": forecast_vals})
        st.dataframe(df_forecast, use_container_width=True, height=240)

        # Plot: letzte Window-Punkte + Forecast
        hist = s.tail(int(window))
        chart_df = pd.concat(
            [hist.rename("History"), pd.Series(forecast_vals, index=fidx, name="Forecast")],
            axis=1,
        )
        st.line_chart(chart_df, height=320)

        # Download
        out_csv = df_forecast.to_csv(index=False).encode("utf-8")
        st.download_button("📥 CSV herunterladen", out_csv,
                           file_name=f"forecast_{safe_t}.csv", mime="text/csv")

    except FileNotFoundError:
        st.warning("Kein Modell für diesen Ticker gefunden. Bitte zuerst **Trainieren** klicken.")
    except Exception as e:
        st.exception(e)
st.caption("Dieses Projekt dient ausschließlich Studien-, Forschungs- und Demonstrationszwecken. "
           "Es handelt sich nicht um ein Finanzwerkzeug, darf nicht als solches verstanden oder verwendet werden und bietet keinerlei finanzielle, "
           "steuerliche oder rechtliche Beratung. Jegliche Nutzung erfolgt auf eigene Verantwortung. Der Autor übernimmt keine Haftung für Schäden "
           "oder Verluste, die aus der Anwendung des Programms entstehen.")