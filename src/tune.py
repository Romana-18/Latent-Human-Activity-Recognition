"""Hyper-parameter study: SAME random-search budget (n_trials) for every model.

Each trial trains with seed 0 and is scored on the VALIDATION set only (the test set is
never touched). The best setting is written back into the model's config file, so all
later runs (3 seeds, ablations...) use the tuned values. The full trial table is saved
to experiments/tuning/<model>.json -> this is your hyper-parameter study table/figure.

    python -m src.tune --configs configs/cnn1d.yaml configs/lstm.yaml ... --trials 8
"""
import argparse
import json
import os

import numpy as np
import yaml

from src.train import run
from src.utils import ROOT, apply_overrides, load_config

SPACE = {
    "baseline": {"model_args.n_estimators": ("choice", [100, 200, 300, 500])},
    "nn": {"train.lr": ("log", 3e-4, 3e-3), "train.batch_size": ("choice", [32, 64, 128]),
           "train.weight_decay": ("log", 1e-5, 1e-2), "model_args.dropout": ("uni", 0.1, 0.5)},
}


def sample(space, rng):
    out = {}
    for k, spec in space.items():
        if spec[0] == "choice":
            out[k] = spec[1][rng.randint(len(spec[1]))]
        elif spec[0] == "log":
            out[k] = float(np.exp(rng.uniform(np.log(spec[1]), np.log(spec[2]))))
        else:
            out[k] = float(rng.uniform(spec[1], spec[2]))
    return out


def tune(config_path, trials, seed=0):
    cfg = load_config(config_path)
    space = SPACE["baseline" if cfg["model"] == "baseline" else "nn"]
    rng = np.random.RandomState(1234)           # same sampler seed for every model
    rows = []
    for t in range(trials):
        ov = sample(space, rng)
        r = run(apply_overrides(cfg, ov), seed, save_ckpt=False, save_results=False, verbose=False, evaluate_test=False)
        rows.append(dict(trial=t, params=ov, val_macro_f1=r["val"]["macro_f1"],
                         train_time_s=r["cost"]["train_time_s"]))
        print(f"[{cfg['name']}] trial {t}: val macro-F1 {rows[-1]['val_macro_f1']:.4f}  {ov}")
    best = max(rows, key=lambda r: r["val_macro_f1"])
    os.makedirs(os.path.join(ROOT, "experiments", "tuning"), exist_ok=True)
    with open(os.path.join(ROOT, "experiments", "tuning", f"{cfg['name']}.json"), "w") as f:
        json.dump(dict(name=cfg["name"], trials=rows, best=best), f, indent=1)
    # write best values back into the (un-inherited) config file
    raw = yaml.safe_load(open(config_path))
    for k, v in best["params"].items():
        d = raw
        parts = k.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    with open(config_path, "w") as f:
        yaml.safe_dump(raw, f, sort_keys=False)
    print(f"[{cfg['name']}] BEST val macro-F1 {best['val_macro_f1']:.4f} -> {best['params']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--trials", type=int, default=8)
    a = ap.parse_args()
    for c in a.configs:
        tune(c, a.trials)
