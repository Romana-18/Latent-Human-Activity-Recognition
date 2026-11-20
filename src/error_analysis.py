"""Error analysis from saved test predictions (needs experiments/results/*_preds.npz).

    python -m src.error_analysis --dataset uci_har --models deepconvlstm_attn cnn1d
Writes: experiments/tables/table6_errors_<model>.{csv,md}, per_subject_errors_<model>.csv,
        figures: confusion_<model>.png, failure_examples_<model>.png, error_overlap.png
The 'Likely cause' column is left as TODO on purpose: look at the example plots and
WRITE the explanation yourselves (this is what the viva will ask).
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.datasets import get_data
from src.utils import ROOT

RES = os.path.join(ROOT, "experiments", "results")
TAB = os.path.join(ROOT, "experiments", "tables")
FIG = os.path.join(ROOT, "experiments", "figures")


def load_preds(dataset, model):
    out = {}
    for p in sorted(glob.glob(os.path.join(RES, f"{dataset}__{model}__seed*_preds.npz"))):
        out[int(p.split("seed")[1].split("_")[0])] = np.load(p)
    return out


def analyse(dataset, model, seed, classes, data):
    P = load_preds(dataset, model)
    if seed not in P:
        print(f"no predictions for {model} seed {seed}"); return None
    d = P[seed]; y, p, s, pr = d["y_true"], d["y_pred"], d["subject"], d["probs"]
    X = data.splits["test"][0]; n = len(classes)
    # confusion matrix (row-normalised)
    cm = np.zeros((n, n), int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    plt.figure(figsize=(0.9 * n + 3, 0.8 * n + 2))
    plt.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    plt.xticks(range(n), classes, rotation=40, ha="right"); plt.yticks(range(n), classes)
    for i in range(n):
        for j in range(n):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="w" if cm[i, j] > cm.max() / 2 else "k", fontsize=8)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"{model} (seed {seed})")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, f"confusion_{dataset}_{model}.png"), dpi=200); plt.close()
    # Table 6: error types
    rows = []
    for a in range(n):
        for b in range(n):
            if a != b and cm[a, b] > 0:
                idx = np.flatnonzero((y == a) & (p == b))
                ex = idx[np.argmax(pr[idx, b])]            # most confident wrong example
                rows.append({"Error type (true -> predicted)": f"{classes[a]} -> {classes[b]}", "Count": int(cm[a, b]),
                             "% of true class": f"{100 * cm[a, b] / cm[a].sum():.1f}", "Example (test idx, subject)": f"{ex}, subj {s[ex]}",
                             "Confidence": f"{pr[ex, b]:.2f}", "Likely cause": "TODO - write after viewing failure_examples figure",
                             "_idx": ex, "_a": a, "_b": b})
    t = pd.DataFrame(rows).sort_values("Count", ascending=False)
    t.drop(columns=["_idx", "_a", "_b"]).to_csv(os.path.join(TAB, f"table6_errors_{dataset}_{model}.csv"), index=False)
    t.drop(columns=["_idx", "_a", "_b"]).head(10).to_markdown(os.path.join(TAB, f"table6_errors_{dataset}_{model}.md"), index=False) \
        if hasattr(pd.DataFrame, "to_markdown") and _has_tabulate() else None
    print(f"\n== {model} (seed {seed}) top confusions ==\n", t.drop(columns=["_idx", "_a", "_b"]).head(8).to_string(index=False))
    # failure examples: the top-3 confusions, mean-centred signal of one example each
    top = t.head(3)
    if len(top):
        fig, ax = plt.subplots(1, len(top), figsize=(5 * len(top), 3.2), squeeze=False)
        for k, (_, r) in enumerate(top.iterrows()):
            win = X[int(r["_idx"])] * data.std[:, None] + data.mean[:, None]
            for c in range(min(3, win.shape[0])):
                ax[0, k].plot(win[c], label=data.channel_names[c])
            ax[0, k].set_title(f"true {classes[int(r['_a'])]} / pred {classes[int(r['_b'])]}", fontsize=9)
        ax[0, 0].legend(fontsize=7); plt.tight_layout()
        plt.savefig(os.path.join(FIG, f"failure_examples_{dataset}_{model}.png"), dpi=200); plt.close()
    # per-subject error rate
    ps = pd.DataFrame({"subject": s, "err": (y != p)}).groupby("subject").err.agg(["mean", "sum", "count"]).reset_index()
    ps.columns = ["subject", "error_rate", "errors", "windows"]
    ps.to_csv(os.path.join(TAB, f"per_subject_errors_{dataset}_{model}.csv"), index=False)
    print("per-subject error rate:\n", ps.round(3).to_string(index=False))
    ok = pr.max(1)[y == p].mean(); bad = pr.max(1)[y != p].mean()
    print(f"mean confidence: correct {ok:.3f} vs wrong {bad:.3f}  (over-confident errors? compare)")
    return (y != p)


def _has_tabulate():
    try:
        import tabulate  # noqa
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="uci_har")
    ap.add_argument("--models", nargs="+", default=["deepconvlstm_attn", "cnn1d", "transformer_scratch"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    any_json = sorted(glob.glob(os.path.join(RES, f"{a.dataset}__{a.models[0]}__seed*.json")))
    cfg = json.load(open(any_json[0]))["config"]
    data = get_data(cfg)
    wrong = {}
    for m in a.models:
        w = analyse(a.dataset, m, a.seed, data.classes, data)
        if w is not None:
            wrong[m] = w
    if len(wrong) > 1:
        W = np.array(list(wrong.values()))
        print(f"\nTest windows misclassified by ALL {len(wrong)} models: {int(W.all(0).sum())} "
              f"| by at least one: {int(W.any(0).sum())} | total {W.shape[1]}")
        plt.figure(figsize=(5, 3)); plt.bar(["by >=1 model", "by ALL models"], [W.any(0).sum(), W.all(0).sum()])
        plt.ylabel("# test windows misclassified"); plt.tight_layout()
        plt.savefig(os.path.join(FIG, f"error_overlap_{a.dataset}.png"), dpi=200)
