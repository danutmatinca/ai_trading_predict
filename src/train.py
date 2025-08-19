# SPDX-FileCopyrightText: © 2025 Danut Matinca
# SPDX-License-Identifier: Polyform-Noncommercial-1.0.0

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import yfinance as yf

from utils.config import load_config
from data.datasets import TimeSeriesWindowDataset
from models.lstm import LSTMForecaster

def set_seed(seed: int):
    import random, os
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

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


def prepare_scaler(kind: str):
    if kind == "standard":
        return StandardScaler()
    elif kind == "minmax":
        return MinMaxScaler()
    else:
        raise ValueError("features.normalize muss 'standard' oder 'minmax' sein.")

def split_series(arr: np.ndarray, val_ratio: float):
    n = len(arr)
    n_val = int(round(n * val_ratio))
    n_train = n - n_val
    train, val = arr[:n_train], arr[n_train:]
    return train, val

def train_loop(model, train_loader, val_loader, device, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        train_loss = total / len(train_loader.dataset)

        # Validate
        model.eval()
        vtotal = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                vtotal += loss.item() * xb.size(0)
        val_loss = vtotal / len(val_loader.dataset)

        print(f"[{epoch:03d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_state, best_val

def main(cfg_path: str):
    cfg = load_config(cfg_path)

    seed = int(cfg["training"]["seed"])
    set_seed(seed)

    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() or cfg["training"]["device"] == "cpu" else "cpu")
    print("Device:", device)

    # 1) Daten laden
    s = fetch_series(
        cfg["data"]["ticker"],
        cfg["data"]["start"],
        cfg["data"]["end"],
        cfg["data"]["interval"],
        cfg["data"]["column"],
    )
    values = s.values.astype(np.float32).reshape(-1, 1)

    # 2) Skalierung
    scaler = prepare_scaler(cfg["features"]["normalize"])
    # nur auf Trainingsteil fitten -> deswegen split vor dem Fit sinnvoll:
    val_ratio = float(cfg["split"]["val_ratio"])
    n = len(values)
    n_val = int(round(n * val_ratio))
    n_train = n - n_val
    train_vals = values[:n_train]
    val_vals = values[n_train:]

    scaler.fit(train_vals)
    values_scaled = scaler.transform(values).astype(np.float32)

    # 3) Fenster bilden
    window_size = int(cfg["window"]["size"])
    horizon = int(cfg["window"]["horizon"])

    from data.datasets import make_windows
    X, y = make_windows(values_scaled.squeeze(-1), window_size, horizon)
    # split identisch auf X/y abbilden
    X_train, y_train = X[: n_train - window_size - horizon + 1], y[: n_train - window_size - horizon + 1]
    X_val, y_val = X[n_train - window_size - horizon + 1 :], y[n_train - window_size - horizon + 1 :]

    train_ds = TimeSeriesWindowDataset(series=values_scaled.squeeze(-1)[:n_train], window_size=window_size, horizon=horizon)
    val_ds   = TimeSeriesWindowDataset(series=values_scaled.squeeze(-1)[n_train-window_size-horizon+1:], window_size=window_size, horizon=horizon)
    # Hinweis: Obige Konstruktion hält Train/Val strikt zeitlich getrennt.

    # Bessere, klare Variante: direkt DataLoader über X_train/y_train usw. erstellen.
    # Um es sauber zu halten, wandeln wir hier direkt:
    import torch.utils.data as tud
    train_loader = tud.DataLoader(tud.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                                  batch_size=int(cfg["training"]["batch_size"]), shuffle=True)
    val_loader   = tud.DataLoader(tud.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                                  batch_size=int(cfg["training"]["batch_size"]), shuffle=False)

    # 4) Modell
    model = LSTMForecaster(
        input_size=1,
        hidden_size=int(cfg["model"]["hidden_size"]),
        num_layers=int(cfg["model"]["num_layers"]),
        dropout=float(cfg["model"]["dropout"]),
        horizon=horizon,
    ).to(device)

    # 5) Training
    best_state, best_val = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=int(cfg["training"]["epochs"]),
        lr=float(cfg["training"]["lr"]),
    )

    # 6) Speichern
    model_dir = Path(cfg["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = Path(cfg["paths"]["best_model"])
    torch.save(best_state, best_model_path)

    # Scaler speichern (per numpy)
    import pickle
    with open(model_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Log
    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    with open(outputs_dir / "training_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Ticker: {cfg['data']['ticker']}\n")
        f.write(f"Best Val MSE: {best_val:.6f}\n")
        f.write(f"Window={window_size}, Horizon={horizon}\n")
        f.write(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}\n")

    print(f"Fertig. Bestes Modell: {best_model_path} (val_mse={best_val:.6f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
