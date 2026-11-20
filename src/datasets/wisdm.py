"""WISDM Activity Prediction v1.1 (Kwapisz et al., 2011): phone accelerometer, 20 Hz, 36 users.

Windows of 80 samples (4 s), 50 % overlap, built inside contiguous (user, activity)
segments so a window never mixes two activities. Split is by USER (no leakage).
"""
import os
import re

import numpy as np

from src.utils import ROOT

RAW_FILE = os.path.join(ROOT, "data", "raw", "wisdm", "WISDM_ar_v1.1_raw.txt")
CACHE = os.path.join(ROOT, "data", "processed", "wisdm.npz")
CLASSES = ["Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing"]
CHANNEL_NAMES = ["acc_x", "acc_y", "acc_z"]
WIN, STEP = 80, 40
SPLIT_SEED = 42
N_VAL, N_TEST = 6, 8
_REC = re.compile(r"^\s*(\d+)\s*,\s*(\w+)\s*,\s*(\d+)\s*,\s*(-?[\d.eE+-]+)\s*,\s*(-?[\d.eE+-]+)\s*,\s*(-?[\d.eE+-]+)\s*;?\s*$")


def _parse():
    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError("WISDM not found. Run: python data/download_wisdm.py")
    text = open(RAW_FILE, errors="ignore").read().replace("\n", "")
    recs = []
    for chunk in text.split(";"):
        m = _REC.match(chunk)
        if m and m.group(2) in CLASSES:
            recs.append((int(m.group(1)), CLASSES.index(m.group(2)),
                         float(m.group(4)), float(m.group(5)), float(m.group(6))))
    return np.array(recs, dtype=np.float64)


def _windows(rec):
    X, y, subj = [], [], []
    users, acts = rec[:, 0].astype(int), rec[:, 1].astype(int)
    # contiguous runs of the same (user, activity)
    change = np.r_[True, (users[1:] != users[:-1]) | (acts[1:] != acts[:-1])]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], len(rec)]
    for s, e in zip(starts, ends):
        seg = rec[s:e, 2:5]
        for i in range(0, len(seg) - WIN + 1, STEP):
            X.append(seg[i:i + WIN].T)
            y.append(acts[s]); subj.append(users[s])
    return np.array(X, np.float32), np.array(y), np.array(subj)


def get_splits():
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        X, y, subj = z["X"], z["y"], z["subj"]
    else:
        X, y, subj = _windows(_parse())
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.savez_compressed(CACHE, X=X, y=y, subj=subj)
    users = np.unique(subj)
    perm = np.random.RandomState(SPLIT_SEED).permutation(users)
    test_u, val_u = perm[:N_TEST], perm[N_TEST:N_TEST + N_VAL]
    m_test, m_val = np.isin(subj, test_u), np.isin(subj, val_u)
    m_train = ~(m_test | m_val)
    sp = {"train": m_train, "val": m_val, "test": m_test}
    splits = {k: (X[m], y[m], subj[m]) for k, m in sp.items()}
    return splits, CLASSES, CHANNEL_NAMES
