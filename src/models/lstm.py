"""Classic model 2: (bi)LSTM / GRU over the raw window."""
import torch.nn as nn


class RNNClassifier(nn.Module):
    def __init__(self, in_channels, n_classes, seq_len, hidden=128, layers=2,
                 cell="lstm", bidirectional=True, dropout=0.3):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(in_channels, hidden, num_layers=layers, batch_first=True,
                       bidirectional=bidirectional, dropout=dropout if layers > 1 else 0.0)
        out = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out, n_classes))

    def forward(self, x):                       # (B, C, T) -> (B, T, C)
        h, _ = self.rnn(x.transpose(1, 2))
        return self.head(h.mean(dim=1))         # mean over time
