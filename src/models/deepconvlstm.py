"""Advanced model: DeepConvLSTM (Ordonez & Roggen, 2016) + additive temporal attention.

4 conv layers extract local features, 2 LSTM layers model temporal dynamics, and an
attention pool (our own implementation) replaces 'last hidden state'. Setting
attention=False gives the ablation 'no attention' (mean pooling over time instead).
"""
import torch.nn as nn

from src.models.attention import AdditiveAttentionPool


class DeepConvLSTM(nn.Module):
    def __init__(self, in_channels, n_classes, seq_len, filters=64, kernel=5,
                 hidden=128, lstm_layers=2, dropout=0.3, attention=True):
        super().__init__()
        convs, c = [], in_channels
        for _ in range(4):
            convs += [nn.Conv1d(c, filters, kernel, padding=kernel // 2),
                      nn.BatchNorm1d(filters), nn.ReLU()]
            c = filters
        self.convs = nn.Sequential(*convs)
        self.lstm = nn.LSTM(filters, hidden, num_layers=lstm_layers, batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.use_attention = attention
        self.pool = AdditiveAttentionPool(hidden) if attention else None
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x):
        h, _ = self.lstm(self.convs(x).transpose(1, 2))      # (B, T, H)
        z = self.pool(h)[0] if self.use_attention else h.mean(dim=1)
        return self.head(z)
