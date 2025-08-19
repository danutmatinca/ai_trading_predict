# SPDX-License-Identifier: Polyform-Noncommercial-1.0.0

from __future__ import annotations
import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, horizon: int):
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0.0,
                            batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, horizon)
        )

    def forward(self, x):
        # x: (B, W, 1)
        out, (h_n, c_n) = self.lstm(x)  # out: (B, W, H)
        last = out[:, -1, :]            # (B, H) — Zustand nach letztem Zeitschritt
        preds = self.head(last)         # (B, horizon)
        return preds
