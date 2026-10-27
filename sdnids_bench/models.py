"""Benchmark models. Every neural model takes [B, L, d] and returns logits [B, C].
Tabular models (RF, XGBoost) take the flattened window [B, L*d]."""
from __future__ import annotations

import math
import torch
import torch.nn as nn

NEURAL = ["MLP", "CNN1D", "LSTM", "GRU", "BiLSTM", "CNN-LSTM", "Transformer"]
TABULAR = ["RF", "XGBoost"]
ALL_MODELS = TABULAR + NEURAL
FAMILY = {"RF": "Tree ensemble", "XGBoost": "Tree ensemble", "MLP": "Feed-forward / convolutional",
          "CNN1D": "Feed-forward / convolutional", "LSTM": "Recurrent", "GRU": "Recurrent", "BiLSTM": "Recurrent",
          "CNN-LSTM": "Recurrent", "Transformer": "Attention"}
FAMILY_ORDER = ["Tree ensemble", "Feed-forward / convolutional", "Recurrent", "Attention"]


class MLP(nn.Module):
    def __init__(self, d, L, C, hidden=128, dropout=0.2, **_):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(d * L, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, C))

    def forward(self, x):
        return self.net(x)


class CNN1D(nn.Module):
    def __init__(self, d, L, C, hidden=64, dropout=0.2, kernel=3, **_):
        super().__init__()
        k = min(kernel, L) if L > 1 else 1
        self.conv = nn.Sequential(nn.Conv1d(d, hidden, k, padding="same"), nn.ReLU(),
                                  nn.Conv1d(hidden, hidden, k, padding="same"), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, C))

    def forward(self, x):
        h = self.conv(x.transpose(1, 2))          # B, H, L
        return self.head(h[:, :, -1])             # causal read-out at the current flow


class RNN(nn.Module):
    def __init__(self, d, L, C, cell="LSTM", hidden=64, layers=1, dropout=0.2, bidirectional=False, **_):
        super().__init__()
        rnn = nn.LSTM if cell == "LSTM" else nn.GRU
        self.rnn = rnn(d, hidden, num_layers=layers, batch_first=True, bidirectional=bidirectional,
                       dropout=dropout if layers > 1 else 0.0)
        out = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out, C))
        self.bi = bidirectional

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.head(h[:, -1])


class CNNLSTM(nn.Module):
    def __init__(self, d, L, C, hidden=64, dropout=0.2, kernel=3, **_):
        super().__init__()
        k = min(kernel, L) if L > 1 else 1
        self.conv = nn.Sequential(nn.Conv1d(d, hidden, k, padding="same"), nn.ReLU())
        self.rnn = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, C))

    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.rnn(h)
        return self.head(h[:, -1])


class Transformer(nn.Module):
    def __init__(self, d, L, C, hidden=64, layers=2, heads=4, dropout=0.1, **_):
        super().__init__()
        self.inp = nn.Linear(d, hidden)
        pe = torch.zeros(max(L, 1), hidden)
        pos = torch.arange(max(L, 1)).unsqueeze(1)
        div = torch.exp(torch.arange(0, hidden, 2) * (-math.log(10000.0) / hidden))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
        enc = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=2 * hidden, dropout=dropout,
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Linear(hidden, C)
        mask = torch.triu(torch.ones(max(L, 1), max(L, 1), dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x):
        h = self.inp(x) + self.pe[:, : x.shape[1]]
        h = self.enc(h, mask=self.mask[: x.shape[1], : x.shape[1]])
        return self.head(h[:, -1])


def build_neural(name: str, d: int, L: int, C: int, hp: dict) -> nn.Module:
    if name == "MLP":
        return MLP(d, L, C, **hp)
    if name == "CNN1D":
        return CNN1D(d, L, C, **hp)
    if name == "LSTM":
        return RNN(d, L, C, cell="LSTM", **hp)
    if name == "GRU":
        return RNN(d, L, C, cell="GRU", **hp)
    if name == "BiLSTM":
        return RNN(d, L, C, cell="LSTM", bidirectional=True, **hp)
    if name == "CNN-LSTM":
        return CNNLSTM(d, L, C, **hp)
    if name == "Transformer":
        return Transformer(d, L, C, **hp)
    raise ValueError(name)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
