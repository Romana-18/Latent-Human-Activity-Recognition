"""Modern model: Transformer encoder for time series (patch embedding + [CLS] token).

Uses our own MultiHeadSelfAttention. Supports self-supervised pre-training by masked
patch reconstruction (like MAE/BERT on signals) -> used for the pretrained-vs-scratch study.
"""
import torch
import torch.nn as nn

from src.models.attention import MultiHeadSelfAttention


class EncoderLayer(nn.Module):
    def __init__(self, d, heads, ff, dropout):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = MultiHeadSelfAttention(d, heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.n1(x)))     # pre-LN residual blocks
        return x + self.drop(self.ff(self.n2(x)))


class TSTransformer(nn.Module):
    def __init__(self, in_channels, n_classes, seq_len, d_model=64, heads=4, layers=2,
                 patch=8, ff_mult=2, dropout=0.2):
        super().__init__()
        assert seq_len % patch == 0
        self.patch, self.n_patches = patch, seq_len // patch
        self.embed = nn.Conv1d(in_channels, d_model, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.randn(1, self.n_patches + 1, d_model) * 0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, d_model * ff_mult, dropout)
                                     for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)
        self.recon = nn.Linear(d_model, in_channels * patch)   # only used for pre-training
        self.in_channels = in_channels

    def _encode(self, tokens):
        tokens = torch.cat([self.cls.expand(tokens.size(0), -1, -1), tokens], dim=1) + self.pos
        for l in self.layers:
            tokens = l(tokens)
        return self.norm(tokens)

    def forward(self, x):                                    # x: (B, C, T)
        tok = self.embed(x).transpose(1, 2)                  # (B, P, D)
        return self.head(self._encode(tok)[:, 0])

    def pretrain_loss(self, x, mask_ratio=0.5):
        """Mask random patches in embedding space and reconstruct their raw values."""
        B, C, T = x.shape
        tok = self.embed(x).transpose(1, 2)
        mask = torch.rand(B, self.n_patches, device=x.device) < mask_ratio
        tok = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(tok), tok)
        rec = self.recon(self._encode(tok)[:, 1:])           # (B, P, C*patch)
        target = x.reshape(B, C, self.n_patches, self.patch).permute(0, 2, 1, 3).reshape(B, self.n_patches, -1)
        return ((rec - target) ** 2)[mask].mean()
