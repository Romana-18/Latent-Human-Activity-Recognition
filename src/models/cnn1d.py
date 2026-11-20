"""Classic model 1: 1D-CNN (3 conv blocks + global average pooling)."""
import torch.nn as nn


def block(cin, cout, k, drop):
    return nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout),
                         nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(drop))


class CNN1D(nn.Module):
    def __init__(self, in_channels, n_classes, seq_len, width=64, kernel=5, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(block(in_channels, width, kernel, dropout),
                                      block(width, width * 2, kernel, dropout),
                                      block(width * 2, width * 2, kernel, dropout))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Linear(width * 2, n_classes))

    def forward(self, x):          # x: (B, C, T)
        return self.head(self.features(x))
