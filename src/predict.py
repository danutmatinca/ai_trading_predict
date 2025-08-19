# SPDX-License-Identifier: Polyform-Noncommercial-1.0.0

from __future__ import annotations
# import argparse
# from pathlib import Path
# import numpy as np
import pandas as pd
# import torch
import yfinance as yf
# import pickle

from utils.config import load_config
from models.lstm import LSTMForecaster
# import matplotlib.pyplot as plt

def fetch_series(ticker: str, start: str | None, end: str | None, interval: str, column: str) -> pd.Series:
    # Hole Daten so, dass 'Adj Close' existiert und Spalten sauber gruppiert sind
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,      # <— wichtig: damit 'Adj Close' wieder vorhanden ist
        group_by="column",      # <— Spaltenlayout stabil halten
        progress=False
    )

    # Falls MultiIndex oder an den Ticker angehängte Namen (z.B. 'Close AAPL') vorkommen:
    if isinstance(df.columns, pd.MultiIndex):
        # Bei Einzelticker: Ticker-Ebene verwerfen -> einfache Spaltennamen
        if len(df.columns.levels[-1]) == 1:
            df.columns = df.columns.get_level_values(0)
    else:
        # Namen wie 'Close AAPL' -> auf 'Close' kürzen
        rename_map = {}
        for c in df.columns:
            if isinstance(c, str) and " " in c:
                rename_map[c] = c.split(" ")[0]
        if rename_map:
            df = df.rename(columns=rename_map)

    # Fallback: Wenn 'Adj Close' fehlt, nimm 'Close'
    wanted = column
    if wanted not in df.columns and wanted == "Adj Close" and "Close" in df.columns:
        wanted = "Close"

    if wanted not in df.columns:
        raise ValueError(f"Spalte '{column}' nicht in Daten gefunden. Verfügbare Spalten: {list(df.columns)}")

    s = df[wanted].dropna()
    s.name = f"{ticker}_{wanted}"
    return s

def main(cfg_path: str):
#    import numpy as np
    import pandas as pd
    import torch
    import pickle
    from pathlib import Path
    import matplotlib.pyplot as plt

    cfg = load_config(cfg_path)

    # --- Device wählen (GPU wenn explizit "cuda" und verfügbar, sonst CPU) ---
    want_cuda = str(cfg["training"]["device"]).lower() == "cuda"
    device = torch.device("cuda" if (want_cuda and torch.cuda.is_available()) else "cpu")

    # --- Modell & Scaler laden ---
    best_model_path = Path(cfg["paths"]["best_model"])
    state = torch.load(best_model_path, map_location="cpu")

    model = LSTMForecaster(
        input_size=1,
        hidden_size=int(cfg["model"]["hidden_size"]),
        num_layers=int(cfg["model"]["num_layers"]),
        dropout=float(cfg["model"]["dropout"]),
        horizon=int(cfg["window"]["horizon"]),
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    scaler_path = Path(cfg["paths"]["model_dir"]) / "scaler.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # --- Zeitreihe laden ---
    s = fetch_series(
        cfg["data"]["ticker"],
        cfg["data"]["start"],
        cfg["data"]["end"],
        cfg["data"]["interval"],
        cfg["data"]["column"],
    ).astype("float32")

    window = int(cfg["window"]["size"])
    horizon = int(cfg["window"]["horizon"])

    if len(s) < window:
        raise ValueError(f"Zu wenig Datenpunkte ({len(s)}) für window={window}.")

    # --- Letztes Fenster skalieren & vorhersagen ---
    last_window = s.values[-window:].reshape(-1, 1)                          # (W, 1)
    last_window_scaled = scaler.transform(last_window).astype("float32")     # (W, 1)

    x = torch.from_numpy(last_window_scaled[None, :, :]).to(device)          # (1, W, 1)
    with torch.no_grad():
        pred_scaled = model(x).cpu().numpy().reshape(-1)                     # (horizon,)

    # --- Zurückskalieren ---
    forecast_values = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)

    # --- Forecast-Index bestimmen ---
    # bevorzugt aus echter Frequenz; sonst anhand cfg["data"]["interval"]
    freq = getattr(s.index, "freq", None) or pd.infer_freq(s.index)
    if freq is not None:
        # ersten Zukunftszeitpunkt = letztes Datum + eine Einheit
        from pandas.tseries.frequencies import to_offset
        start_ts = s.index[-1] + to_offset(freq)
        forecast_index = pd.date_range(start=start_ts, periods=horizon, freq=freq)
    else:
        iv = str(cfg["data"]["interval"]).lower()
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
        forecast_index = pd.DatetimeIndex(idx)

    # --- CSV schreiben ---
    out_dir = Path(cfg["paths"]["outputs_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame({
        "timestamp": forecast_index,
        "forecast": forecast_values
    })

    csv_path = out_dir / "forecast.csv"
    df_out.to_csv(csv_path, index=False)
    print("Vorhersage gespeichert:", csv_path)

    # --- Plot: letzte window.size Punkte (History) vs. Forecast ---
    hist_idx = s.index[-window:]
    hist_vals = s.values[-window:]

    plt.figure(figsize=(10, 5))
    plt.plot(hist_idx, hist_vals, label="History")
    plt.plot(forecast_index, forecast_values, label="Forecast")
    plt.title(f"Forecast {cfg['data']['ticker']}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    if __name__ == "__main__":
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, default="configs/default.yaml")
        args = parser.parse_args()
        main(args.config)
