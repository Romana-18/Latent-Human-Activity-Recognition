"""Synthetic data used ONLY for smoke tests (checks the code runs end to end).
Never report results from it."""
import numpy as np

CLASSES = ["A", "B", "C", "D", "E", "F"]
CHANNEL_NAMES = [f"{s}_{a}" for s in ["total_acc", "body_acc", "body_gyro"] for a in "xyz"]


def get_splits(n_subjects=12, per=12, T=128, seed=0):
    rng = np.random.RandomState(seed)
    X, y, subj = [], [], []
    t = np.linspace(0, 2 * np.pi, T)
    for s in range(n_subjects):
        offset = rng.randn(9, 1) * 0.2
        for c in range(6):
            for _ in range(per):
                freq = 1 + c
                sig = np.sin(freq * t[None] + rng.rand(9, 1) * 6) * (0.5 + 0.2 * c)
                X.append(sig + offset + 0.3 * rng.randn(9, T)); y.append(c); subj.append(s)
    X, y, subj = np.array(X, np.float32), np.array(y), np.array(subj)
    pick = lambda m: (X[m], y[m], subj[m])
    return ({"train": pick(subj < 8), "val": pick((subj >= 8) & (subj < 10)),
             "test": pick(subj >= 10)}, CLASSES, CHANNEL_NAMES)
