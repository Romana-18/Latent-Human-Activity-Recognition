from dataclasses import dataclass, field

import numpy as np


@dataclass
class HARData:
    splits: dict            # split -> (X[N,C,T] normalised, y, subject)
    classes: list
    channel_names: list
    mean: np.ndarray        # per-channel train mean  (C,)
    std: np.ndarray
    seq_len: int = 0
    in_channels: int = 0


def get_data(cfg):
    """Load a dataset, select channel groups, normalise with TRAIN statistics only."""
    name = cfg["dataset"]
    if name == "uci_har":
        from src.datasets.uci_har import get_splits
    elif name == "wisdm":
        from src.datasets.wisdm import get_splits
    elif name == "synthetic":
        from src.datasets.synthetic import get_splits
    else:
        raise ValueError(f"unknown dataset {name}")
    splits, classes, chan_names = get_splits()

    wanted = cfg.get("channels")
    if wanted and name != "wisdm":  # channel groups, e.g. total_acc / body_acc / body_gyro
        idx = [i for i, c in enumerate(chan_names) if c.rsplit("_", 1)[0] in wanted]
    else:
        idx = list(range(len(chan_names)))
    chan_names = [chan_names[i] for i in idx]

    out = {}
    for k, (X, y, s) in splits.items():
        out[k] = [X[:, idx, :].astype(np.float32), y.astype(np.int64), s]
    mean = out["train"][0].mean(axis=(0, 2))
    std = out["train"][0].std(axis=(0, 2)) + 1e-6
    for k in out:
        out[k][0] = (out[k][0] - mean[None, :, None]) / std[None, :, None]
        out[k] = tuple(out[k])
    X0 = out["train"][0]
    return HARData(out, classes, chan_names, mean, std, X0.shape[2], X0.shape[1])
