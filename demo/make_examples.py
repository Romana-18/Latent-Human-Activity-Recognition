"""Create one example CSV per activity from the TEST set -> demo/examples/*.csv
    python demo/make_examples.py --dataset uci_har
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets import get_data  # noqa: E402
from src.utils import ROOT, apply_overrides, load_config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="uci_har")
a = ap.parse_args()
cfg = load_config(os.path.join(ROOT, "configs", "cnn1d.yaml"))
if a.dataset == "wisdm":
    cfg = apply_overrides(cfg, {"dataset": "wisdm", "channels": None})
data = get_data(cfg)
X, y, s = data.splits["test"]
out = os.path.join(ROOT, "demo", "examples", a.dataset)
os.makedirs(out, exist_ok=True)
for c, name in enumerate(data.classes):
    i = np.flatnonzero(y == c)[0]
    raw = (X[i] * data.std[:, None] + data.mean[:, None]).T      # (T, C)
    pd.DataFrame(raw, columns=data.channel_names).to_csv(
        os.path.join(out, f"{name.lower().replace(' ', '_')}.csv"), index=False)
print("examples written to", out)
