"""UCI HAR (Anguita et al., 2013) - raw inertial signals, subject-wise splits.

The official release already separates subjects: 21 train subjects / 9 test subjects.
We carve a *validation* set out of the 21 train subjects (4 subjects, fixed seed),
so train / val / test never share a subject (no leakage from overlapping windows).
"""
import glob
import os

import numpy as np
import pandas as pd

from src.utils import ROOT

RAW_DIR = os.path.join(ROOT, "data", "raw", "uci_har")
CACHE = os.path.join(ROOT, "data", "processed", "uci_har.npz")
SIGNALS = ["total_acc", "body_acc", "body_gyro"]
AXES = ["x", "y", "z"]
CHANNEL_NAMES = [f"{s}_{a}" for s in SIGNALS for a in AXES]
VAL_SUBJECTS = 4
SPLIT_SEED = 42  # fixed for ALL models and ALL seeds -> identical split everywhere


def _find_root():
    hits = glob.glob(os.path.join(RAW_DIR, "**", "Inertial Signals"), recursive=True)
    if not hits:
        raise FileNotFoundError(
            "UCI HAR not found. Run: python data/download_uci_har.py")
    return os.path.dirname(os.path.dirname(hits[0]))  # .../UCI HAR Dataset


def _read_matrix(path):
    return pd.read_csv(path, sep=r"\s+", header=None).values.astype(np.float32)


def _load_split(root, split):
    chans = []
    for s in SIGNALS:
        for a in AXES:
            p = os.path.join(root, split, "Inertial Signals", f"{s}_{a}_{split}.txt")
            chans.append(_read_matrix(p))
    X = np.stack(chans, axis=1)  # (N, 9, 128)
    y = np.loadtxt(os.path.join(root, split, f"y_{split}.txt"), dtype=int) - 1
    subj = np.loadtxt(os.path.join(root, split, f"subject_{split}.txt"), dtype=int)
    return X, y, subj


def load_raw():
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return {k: z[k] for k in z.files}
    root = _find_root()
    Xtr, ytr, str_ = _load_split(root, "train")
    Xte, yte, ste = _load_split(root, "test")
    names = [l.split(" ", 1)[1].strip().replace("_", " ").title()
             for l in open(os.path.join(root, "activity_labels.txt")).read().strip().splitlines()]
    out = dict(Xtr=Xtr, ytr=ytr, str=str_, Xte=Xte, yte=yte, ste=ste, classes=np.array(names))
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **out)
    return out


def get_splits():
    d = load_raw()
    subjects = np.unique(d["str"])
    rng = np.random.RandomState(SPLIT_SEED)
    val_subj = rng.choice(subjects, size=VAL_SUBJECTS, replace=False)
    is_val = np.isin(d["str"], val_subj)
    splits = {
        "train": (d["Xtr"][~is_val], d["ytr"][~is_val], d["str"][~is_val]),
        "val": (d["Xtr"][is_val], d["ytr"][is_val], d["str"][is_val]),
        "test": (d["Xte"], d["yte"], d["ste"]),
    }
    # hard guarantee: no subject appears in two splits
    sets = {k: set(v[2].tolist()) for k, v in splits.items()}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) \
        and not (sets["val"] & sets["test"]), "subject leakage between splits!"
    return splits, list(d["classes"]), CHANNEL_NAMES
