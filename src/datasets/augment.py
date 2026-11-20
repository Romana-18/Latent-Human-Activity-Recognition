"""GPU batch augmentations for (B, C, T) tensors: jitter, scaling, time-warp."""
import torch
import torch.nn.functional as F


def jitter(x, sigma):
    return x + sigma * torch.randn_like(x)


def scaling(x, sigma):
    return x * (1 + sigma * torch.randn(x.shape[0], x.shape[1], 1, device=x.device))


def time_warp(x, sigma, knots=4):
    B, C, T = x.shape
    speed = 1 + sigma * torch.randn(B, 1, knots + 2, device=x.device)
    speed = F.interpolate(speed, size=T, mode="linear", align_corners=True).clamp(min=0.2)
    cum = torch.cumsum(speed, dim=-1)
    cum = (cum - cum[..., :1]) / (cum[..., -1:] - cum[..., :1])      # 0..1, monotonic
    gx = (cum * 2 - 1).squeeze(1)                                     # (B, T)
    grid = torch.stack([gx, torch.zeros_like(gx)], dim=-1).unsqueeze(1)  # (B,1,T,2)
    out = F.grid_sample(x.unsqueeze(2), grid, mode="bilinear",
                        padding_mode="border", align_corners=True)
    return out.squeeze(2)


def augment_batch(x, aug_cfg):
    """aug_cfg e.g. {'jitter': 0.05, 'scaling': 0.1, 'time_warp': 0.2}; 0/None = off."""
    if not aug_cfg:
        return x
    if aug_cfg.get("jitter"):
        x = jitter(x, aug_cfg["jitter"])
    if aug_cfg.get("scaling"):
        x = scaling(x, aug_cfg["scaling"])
    if aug_cfg.get("time_warp"):
        x = time_warp(x, aug_cfg["time_warp"])
    return x
