# SPDX-License-Identifier: Polyform-Noncommercial-1.0.0

from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset

def make_windows(x: np.ndarray, window_size: int, horizon: int):
    X, y = [], []
    for i in range(len(x) - window_size - horizon + 1):
        X.append(x[i : i + window_size])
        y.append(x[i + window_size : i + window_size + horizon])
    X = np.array(X, dtype=np.float32)[..., None]  # shape: (N, W, 1)
    y = np.array(y, dtype=np.float32)            # shape: (N, H)
    return X, y

class TimeSeriesWindowDataset(Dataset):
    def __init__(self, series: np.ndarray, window_size: int, horizon: int):
        X, y = make_windows(series, window_size, horizon)
        self.X = torch.from_numpy(X)  # (N, W, 1)
        self.y = torch.from_numpy(y)  # (N, H)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
