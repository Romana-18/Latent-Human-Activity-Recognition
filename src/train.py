"""Train ONE configuration with ONE seed, evaluate, save results/checkpoint/predictions.

    python -m src.train --config configs/cnn1d.yaml --seed 0
    python -m src.train --config configs/cnn1d.yaml --seed 0 --override train.lr=0.0005
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from src.datasets import get_data
from src.datasets.augment import augment_batch
from src.evaluate import compute_metrics, cpu_latency_ms, latency_ms, predict_proba
from src.models import build_model
from src.models.baseline import build_baseline, extract_features
from src.utils import (ROOT, apply_overrides, exp_name, hardware_string, load_config, set_seed)

RESULTS = os.path.join(ROOT, "experiments", "results")
CKPT = os.path.join(ROOT, "checkpoints")


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def _macro_f1(y, p, n):
    return f1_score(y, p, average="macro", labels=list(range(n)), zero_division=0)


# ----------------------------------------------------------------------------- baseline
def run_baseline(cfg, seed, data, save_ckpt, evaluate_test=True):
    n = len(data.classes)
    Xtr, ytr, _ = data.splits["train"]
    t0 = time.time()
    clf = build_baseline(cfg.get("baseline", "rf"), seed, **cfg.get("model_args", {}))
    clf.fit(extract_features(Xtr), ytr)
    train_time = time.time() - t0
    res = {}
    for split in (["val", "test"] if evaluate_test else ["val"]):
        X, y, s = data.splits[split]
        probs = clf.predict_proba(extract_features(X))
        res[split] = (compute_metrics(y, probs.argmax(1), n), probs)
    Xt = data.splits["test" if evaluate_test else "val"][0]
    t = time.perf_counter()
    for i in range(50):
        clf.predict_proba(extract_features(Xt[i:i + 1]))
    infer = (time.perf_counter() - t) / 50 * 1000
    name = exp_name(cfg, seed)
    tmp = os.path.join(CKPT, name + ".joblib")
    os.makedirs(CKPT, exist_ok=True)
    joblib.dump(dict(model=clf, cfg=cfg, mean=data.mean, std=data.std, classes=data.classes,
                     seq_len=data.seq_len, in_channels=data.in_channels,
                     channel_names=data.channel_names), tmp)
    size_mb = os.path.getsize(tmp) / 1e6
    if not save_ckpt:
        os.remove(tmp)
    est = clf[-1]
    n_params = int(sum(t.tree_.node_count for t in est.estimators_)) if hasattr(est, "estimators_") \
        else int(getattr(est, "support_vectors_", np.zeros((0, 1))).size)
    cost = dict(params=n_params, params_note="tree nodes (RF) / support-vector values (SVM)",
                model_size_mb=size_mb, train_time_s=train_time, epochs_run=1,
                infer_ms_gpu=None, infer_ms_cpu=infer, peak_mem_mb=None, hardware=hardware_string())
    return res, cost, []


# ----------------------------------------------------------------------------- deep nets
def _batches(n, bs, shuffle, device):
    idx = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def run_nn(cfg, seed, data, save_ckpt, verbose, evaluate_test=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = len(data.classes)
    tc = cfg["train"]
    model = build_model(cfg["model"], data.in_channels, n, data.seq_len,
                        **cfg.get("model_args", {})).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    T = {k: (torch.as_tensor(v[0], device=device), torch.as_tensor(v[1], device=device))
         for k, v in data.splits.items() if evaluate_test or k != "test"}
    Xtr, ytr = T["train"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    # ---- optional self-supervised pre-training (transformer only): masked patch reconstruction
    pre = cfg.get("pretrain", {})
    if pre.get("enabled"):
        opt = torch.optim.AdamW(model.parameters(), lr=pre.get("lr", 1e-3), weight_decay=1e-4)
        for ep in range(pre.get("epochs", 30)):
            model.train()
            for b in _batches(len(Xtr), tc["batch_size"], True, device):
                loss = model.pretrain_loss(Xtr[b], pre.get("mask_ratio", 0.5))  # labels NOT used
                opt.zero_grad(); loss.backward(); opt.step()
            if verbose and ep % 10 == 0:
                print(f"  [pretrain] epoch {ep} loss {loss.item():.4f}")

    # ---- supervised training
    weight = None
    if tc.get("class_weight") == "balanced":
        cnt = torch.bincount(ytr, minlength=n).float()
        weight = (cnt.sum() / (n * cnt.clamp(min=1))).to(device)
    crit = nn.CrossEntropyLoss(weight=weight, label_smoothing=tc.get("label_smoothing", 0.0))
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc.get("weight_decay", 1e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["epochs"])
    best, best_state, bad, history = -1, None, 0, []
    for ep in range(tc["epochs"]):
        model.train()
        tl = 0.0
        for b in _batches(len(Xtr), tc["batch_size"], True, device):
            xb = augment_batch(Xtr[b], cfg.get("augment"))
            loss = crit(model(xb), ytr[b])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl += loss.item() * len(b)
        sched.step()
        pv = predict_proba(model, data.splits["val"][0], device).argmax(1)
        vf1 = _macro_f1(data.splits["val"][1], pv, n)
        history.append(dict(epoch=ep, train_loss=tl / len(Xtr), val_macro_f1=vf1))
        if verbose:
            print(f"  epoch {ep:03d} loss {tl / len(Xtr):.4f} val-macroF1 {vf1:.4f}")
        if vf1 > best:
            best, bad = vf1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tc.get("patience", 8):
                break
    train_time = time.time() - t_start
    model.load_state_dict(best_state)        # model selection on VALIDATION only

    res = {}
    for split in (["val", "test"] if evaluate_test else ["val"]):
        X, y, s = data.splits[split]
        probs = predict_proba(model, X, device)
        res[split] = (compute_metrics(y, probs.argmax(1), n), probs)

    Xt = data.splits["test" if evaluate_test else "val"][0]
    name = exp_name(cfg, seed)
    path = os.path.join(CKPT, name + ".pt")
    os.makedirs(CKPT, exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), cfg=cfg, mean=data.mean, std=data.std,
                    classes=data.classes, seq_len=data.seq_len, in_channels=data.in_channels,
                    channel_names=data.channel_names), path)
    size_mb = os.path.getsize(path) / 1e6
    if not save_ckpt:
        os.remove(path)
    cost = dict(params=n_params, model_size_mb=size_mb, train_time_s=train_time,
                epochs_run=len(history),
                infer_ms_gpu=latency_ms(model, Xt, device) if device.type == "cuda" else None,
                infer_ms_cpu=cpu_latency_ms(model, Xt),
                peak_mem_mb=torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else None,
                hardware=hardware_string())
    return res, cost, history


# ----------------------------------------------------------------------------- entry point
def run(cfg, seed, save_ckpt=True, save_results=True, verbose=True, evaluate_test=True):
    set_seed(seed)
    data = get_data(cfg)
    if cfg["model"] == "baseline":
        res, cost, history = run_baseline(cfg, seed, data, save_ckpt, evaluate_test)
    else:
        res, cost, history = run_nn(cfg, seed, data, save_ckpt, verbose, evaluate_test)
    name = exp_name(cfg, seed)
    out = dict(name=cfg["name"], dataset=cfg["dataset"], seed=seed, exp=name, config=cfg,
               classes=data.classes, val=res["val"][0], test=res["test"][0] if evaluate_test else None, cost=cost,
               history=history, group=cfg.get("group", "main"))
    if save_results:
        if evaluate_test:
            out_dir = RESULTS
        else:
            out_dir = os.path.join(
                ROOT, "experiments", "validation_only",
                cfg.get("group", "other")
            )
        os.makedirs(out_dir, exist_ok=True)

        with open(os.path.join(out_dir, name + ".json"), "w") as f:
            json.dump(_jsonable(out), f, indent=1)

        if evaluate_test:
            yt, st = data.splits["test"][1], data.splits["test"][2]
            np.savez_compressed(
                os.path.join(out_dir, name + "_preds.npz"),
                y_true=yt,
                subject=st,
                y_pred=res["test"][1].argmax(1),
                probs=res["test"][1]
            )
    if verbose:
        test_msg = (f"TEST macro-F1 {out['test']['macro_f1']:.4f} acc {out['test']['accuracy']:.4f}"
                    if evaluate_test else "TEST not evaluated")
        print(f"[{name}] val macro-F1 {out['val']['macro_f1']:.4f} | {test_msg} | "
              f"{cost['train_time_s']:.0f}s")
    return out


def parse_overrides(items):
    ov = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        ov[k] = v
    return ov


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--override", nargs="*", help="dotted.key=value, e.g. train.lr=0.0005")
    a = ap.parse_args()
    run(apply_overrides(load_config(a.config), parse_overrides(a.override)), a.seed)
