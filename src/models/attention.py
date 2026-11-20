"""Attention components WRITTEN FROM SCRATCH (requirement 5) - no nn.MultiheadAttention.

1. MultiHeadSelfAttention : scaled dot-product attention  softmax(QK^T / sqrt(d_k)) V
                            (Vaswani et al., 2017, "Attention Is All You Need")
2. AdditiveAttentionPool  : learned temporal attention pooling over a sequence
                            (Bahdanau et al., 2015 style additive scoring)
"""
import math

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.h, self.dk = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, return_attn=False):          # x: (B, T, D)
        B, T, D = x.shape
        q, k, v = self.qkv(x).reshape(B, T, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)   # (B, h, T, T)
        attn = self.drop(torch.softmax(scores, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)       # concat heads
        out = self.proj(out)
        return (out, attn) if return_attn else out


class AdditiveAttentionPool(nn.Module):
    """h: (B, T, D) -> weighted sum over time (B, D); weights are returned for plotting."""

    def __init__(self, dim, hidden=64):
        super().__init__()
        self.W = nn.Linear(dim, hidden)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, h):
        w = torch.softmax(self.v(torch.tanh(self.W(h))).squeeze(-1), dim=1)   # (B, T)
        return (w.unsqueeze(-1) * h).sum(dim=1), w
