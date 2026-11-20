"""Print split sizes + class distribution (for Table 1 and the dataset section).
    python -m src.dataset_stats --dataset uci_har
"""
import argparse

import numpy as np

from src.datasets import get_data
from src.utils import apply_overrides, load_config

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="uci_har", choices=["uci_har", "wisdm", "synthetic"])
a = ap.parse_args()
cfg = load_config("configs/cnn1d.yaml")
cfg = apply_overrides(cfg, {"dataset": a.dataset, **({"channels": None} if a.dataset == "wisdm" else {})})
d = get_data(cfg)
print(f"{a.dataset}: window = {d.seq_len} samples, {d.in_channels} channels {d.channel_names}")
for k, (X, y, s) in d.splits.items():
    print(f"\n{k}: {len(X)} windows, {len(np.unique(s))} subjects {sorted(np.unique(s).tolist())}")
    for c, name in enumerate(d.classes):
        print(f"   {name:<22s} {int((y == c).sum()):6d}  ({100 * (y == c).mean():.1f}%)")
