"""Shared evaluation for ALL models (metrics, latency, checkpoint loading).

CLI:  python -m src.evaluate --checkpoint checkpoints/uci_har__cnn1d__seed0.pt
"""
import argparse
import copy
import time

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, precision_recall_fscore_support)


def compute_metrics(y_true, y_pred, n_classes):
    labels = list(range(n_classes))
    p, r, f, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "per_class": {"precision": p.tolist(), "recall": r.tolist(), "f1": f.tolist(),
                      "support": sup.tolist()},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


@torch.no_grad()
def predict_proba(model, X, device, bs=512):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = torch.as_tensor(X[i:i + bs], dtype=torch.float32, device=device)
        out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def latency_ms(model, x1, device, n=100, warmup=10):
    """Mean latency (ms) to classify ONE window (batch size 1)."""
    model.eval()
    x1 = torch.as_tensor(x1[:1], dtype=torch.float32, device=device)
    for _ in range(warmup):
        model(x1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        model(x1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def cpu_latency_ms(model, x1, n=100):
    return latency_ms(copy.deepcopy(model).cpu(), x1, torch.device("cpu"), n=n)


class Predictor:
    """Loads any checkpoint (torch or sklearn) and predicts from RAW (un-normalised) windows."""

    def __init__(self, path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if path.endswith(".joblib"):
            import joblib
            ck = joblib.load(path)
            self.kind, self.model = "sklearn", ck["model"]
        else:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            from src.models import build_model
            self.kind = "torch"
            self.model = build_model(ck["cfg"]["model"], ck["in_channels"], len(ck["classes"]),
                                     ck["seq_len"], **ck["cfg"].get("model_args", {}))
            self.model.load_state_dict(ck["state_dict"])
            self.model.to(self.device).eval()
        self.cfg, self.mean, self.std = ck["cfg"], ck["mean"], ck["std"]
        self.classes, self.seq_len = list(ck["classes"]), ck["seq_len"]
        self.in_channels, self.channel_names = ck["in_channels"], ck["channel_names"]

    def predict_proba(self, X_raw):
        X = (X_raw - self.mean[None, :, None]) / self.std[None, :, None]
        if self.kind == "sklearn":
            from src.models.baseline import extract_features
            return self.model.predict_proba(extract_features(X))
        return predict_proba(self.model, X, self.device)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    a = ap.parse_args()
    from src.datasets import get_data
    pr = Predictor(a.checkpoint)
    data = get_data(pr.cfg)
    X, y, _ = data.splits[a.split]
    X_raw = X * data.std[None, :, None] + data.mean[None, :, None]
    m = compute_metrics(y, pr.predict_proba(X_raw).argmax(1), len(pr.classes))
    print(f"{a.split}: macro-F1={m['macro_f1']:.4f}  acc={m['accuracy']:.4f}  "
          f"balanced-acc={m['balanced_acc']:.4f}")
    for c, f1 in zip(pr.classes, m["per_class"]["f1"]):
        print(f"  {c:<22s} F1={f1:.3f}")
