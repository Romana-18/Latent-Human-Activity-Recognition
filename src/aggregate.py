"""Turn experiments/results/*.json into the report tables (CSV + Markdown) and figures.

    python -m src.aggregate
Outputs -> experiments/tables/ and experiments/figures/
Tables: table2_architectures, table3_hyperparameters, table4_main_results, per_class_f1,
        table5_ablation, data_strategy, table7_cost, tuning_summary, hypotheses_summary
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.utils import ROOT

RES = os.path.join(ROOT, "experiments", "results")
TAB = os.path.join(ROOT, "experiments", "tables")
FIG = os.path.join(ROOT, "experiments", "figures")
TUNE = os.path.join(ROOT, "experiments", "tuning")
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)

ARCH = {  # Table 2 descriptive part (edit if you change the models)
    "rf_baseline": ("Baseline (classical ML)", "hand-crafted features (~11 stats/channel)", "Random Forest, 300 trees", "No"),
    "cnn1d": ("Classic - CNN", "raw window (C x T)", "3 x [Conv1D-BN-ReLU-MaxPool] + GAP + FC", "No"),
    "bilstm": ("Classic - RNN", "raw window (T x C)", "2-layer BiLSTM (128) + mean pool + FC", "No"),
    "transformer_scratch": ("Modern - Transformer", "raw window -> 8-sample patches", "patch Conv embed + [CLS] + 2 x (own MHSA + FFN)", "No (scratch)"),
    "transformer_ssl_pretrained": ("Modern - Transformer", "raw window -> 8-sample patches", "same as above", "Yes (self-supervised masked patch reconstruction)"),
    "deepconvlstm_attn": ("Advanced - DeepConvLSTM", "raw window (C x T)", "4 x Conv1D + 2-layer LSTM (128) + additive attention pool", "No"),
}
ABLATION = {  # variant -> (what changed, reference model)
    "dcl_no_attention": ("Attention pooling removed (mean pooling)", "deepconvlstm_attn"),
    "dcl_no_gyro": ("Gyroscope modality removed", "deepconvlstm_attn"),
    "dcl_no_augmentation": ("Augmentation removed", "deepconvlstm_attn"),
    "transformer_ssl_pretrained": ("Self-supervised pre-training ADDED (scratch is the reference)", "transformer_scratch"),
}


def load():
    rows = []
    for p in sorted(glob.glob(os.path.join(RES, "*.json"))):
        r = json.load(open(p))
        if r["dataset"] == "synthetic" and not os.environ.get("INCLUDE_SYNTHETIC"):
            continue
        row = dict(dataset=r["dataset"], name=r["name"], group=r["group"], seed=r["seed"],
                   **{f"test_{k}": r["test"][k] for k in ["macro_f1", "weighted_f1", "balanced_acc", "accuracy", "kappa"]},
                   val_macro_f1=r["val"]["macro_f1"], f1_per_class=r["test"]["per_class"]["f1"],
                   classes=r["classes"], cfg=r["config"], history=r["history"], **r["cost"])
        rows.append(row)
    return pd.DataFrame(rows)


def ms(v):
    v = np.asarray(v, float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def ci95(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return float("nan")
    return stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


def fmt(v, d=4):
    m, s = ms(v)
    return f"{m:.{d}f} ± {s:.{d}f}"


def save(df, name):
    df.to_csv(os.path.join(TAB, name + ".csv"), index=False)
    cols = list(df.columns)
    md = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in df.iterrows():
        md += "| " + " | ".join(str(r[c]) for c in cols) + " |\n"
    open(os.path.join(TAB, name + ".md"), "w").write(md)
    print(f"\n### {name}\n{md}")


def table_main(df):
    out = []
    for (ds, name), g in df[df.group == "main"].groupby(["dataset", "name"]):
        out.append({"Dataset": ds, "Model": name, "Seeds": len(g),
                    "Macro-F1 (mean ± SD)": fmt(g.test_macro_f1),
                    "Macro-F1 95% CI ±": f"{ci95(g.test_macro_f1):.4f}",
                    "Balanced acc": fmt(g.test_balanced_acc), "Accuracy": fmt(g.test_accuracy),
                    "Weighted-F1": fmt(g.test_weighted_f1), "Kappa": fmt(g.test_kappa)})
    t = pd.DataFrame(out)
    save(t, "table4_main_results")
    # per-class F1
    pc = []
    for (ds, name), g in df[df.group == "main"].groupby(["dataset", "name"]):
        arr = np.array(g.f1_per_class.tolist())
        row = {"Dataset": ds, "Model": name}
        for i, c in enumerate(g.classes.iloc[0]):
            row[c] = f"{arr[:, i].mean():.3f} ± {arr[:, i].std(ddof=1) if len(arr) > 1 else 0:.3f}"
        pc.append(row)
    save(pd.DataFrame(pc), "per_class_f1")


def table_cost(df):
    out = []
    for (ds, name), g in df[df.group.isin(["main", "ablation"])].groupby(["dataset", "name"]):
        f = lambda c, d=1: "n/a" if g[c].isna().all() else f"{g[c].mean():.{d}f}"
        out.append({"Dataset": ds, "Model": name, "Params": f"{int(g.params.mean()):,}",
                    "Size (MB)": f("model_size_mb", 2), "Train time (s)": f("train_time_s"),
                    "Epochs run": f("epochs_run"), "Infer CPU (ms/window)": f("infer_ms_cpu", 2),
                    "Infer GPU (ms/window)": f("infer_ms_gpu", 2), "Peak GPU mem (MB)": f("peak_mem_mb", 0),
                    "Hardware": g.hardware.iloc[0]})
    save(pd.DataFrame(out), "table7_cost")


def table_arch(df):
    out = []
    for name, (fam, inp, layers, pre) in ARCH.items():
        g = df[(df.name == name)]
        p = f"{int(g.params.mean()):,}" if len(g) else "(run first)"
        out.append({"Model": name, "Family": fam, "Input": inp, "Main layers": layers,
                    "Parameters": p, "Pretrained": pre})
    save(pd.DataFrame(out), "table2_architectures")


def table_hparams(df):
    out = []
    for name, g in df[df.dataset == df.dataset.iloc[0]].groupby("name"):
        c = g.cfg.iloc[0]; t = c.get("train", {}); m = c.get("model_args", {}); a = c.get("augment") or {}
        out.append({"Model": name, "Optimizer": "AdamW" if "train" in c else "n/a (sklearn)",
                    "LR": t.get("lr", "-"), "Batch": t.get("batch_size", "-"),
                    "Max epochs (early-stop patience)": f"{t.get('epochs', '-')} ({t.get('patience', '-')})",
                    "Regularization": f"wd={t.get('weight_decay', '-')}, dropout={m.get('dropout', '-')}",
                    "Loss": "CrossEntropy" + (f" (ls={t['label_smoothing']})" if t.get("label_smoothing") else "")
                            + (" class-weighted" if t.get("class_weight") == "balanced" else "") if t else "Gini (RF)",
                    "Augmentation": ", ".join(f"{k}={v}" for k, v in a.items() if v) or "none"})
    save(pd.DataFrame(out), "table3_hyperparameters")


def table_ablation(df):
    out = []
    for ds in df.dataset.unique():
        d = df[df.dataset == ds]
        for var, (what, ref) in ABLATION.items():
            gv, gr = d[d.name == var], d[d.name == ref]
            if len(gv) and len(gr):
                dm = gv.test_macro_f1.mean() - gr.test_macro_f1.mean()
                p = stats.ttest_ind(gv.test_macro_f1, gr.test_macro_f1, equal_var=False).pvalue if len(gv) > 1 and len(gr) > 1 else float("nan")
                out.append({"Dataset": ds, "Variant": var, "Change": what, "Reference": ref,
                            "Macro-F1 variant": fmt(gv.test_macro_f1), "Macro-F1 reference": fmt(gr.test_macro_f1),
                            "Δ vs reference": f"{dm:+.4f}", "Welch p (n=3, indicative)": f"{p:.3f}"})
    if out:
        save(pd.DataFrame(out), "table5_ablation")
    ds_out = []
    for ds in df.dataset.unique():
        d = df[(df.dataset == ds) & (df.group == "data")]
        base = d[d.name == "cnn_aug_none"].test_macro_f1
        for name, g in d.groupby("name"):
            ds_out.append({"Dataset": ds, "Strategy": name, "Macro-F1": fmt(g.test_macro_f1),
                           "Balanced acc": fmt(g.test_balanced_acc),
                           "Δ vs no augmentation": f"{g.test_macro_f1.mean() - base.mean():+.4f}" if len(base) else "n/a"})
    if ds_out:
        save(pd.DataFrame(ds_out), "data_strategy")


def table_tuning():
    out = []
    for p in sorted(glob.glob(os.path.join(TUNE, "*.json"))):
        r = json.load(open(p))
        v = [t["val_macro_f1"] for t in r["trials"]]
        out.append({"Model": r["name"], "Trials": len(v), "Val F1 min": f"{min(v):.4f}", "Val F1 max": f"{max(v):.4f}",
                    "Best params": json.dumps({k.split('.')[-1]: round(x, 5) if isinstance(x, float) else x
                                               for k, x in r["best"]["params"].items()})})
    if out:
        save(pd.DataFrame(out), "tuning_summary")


def hypotheses(df):
    lines = ["# Numbers for your hypothesis verdicts (YOU write the verdict + explanation)\n"]
    for ds in df.dataset.unique():
        d = df[df.dataset == ds]
        get = lambda n: d[d.name == n].test_macro_f1
        lines.append(f"\n## {ds}\n")
        main = d[d.group == "main"]
        if len(main):
            best = main.groupby("name").test_macro_f1.mean().sort_values(ascending=False)
            lines.append(f"- Ranking by mean macro-F1: " + ", ".join(f"{k} ({v:.4f})" for k, v in best.items()))
        def cmp(a, b, label):
            if len(get(a)) and len(get(b)):
                p = stats.ttest_ind(get(a), get(b), equal_var=False).pvalue if len(get(a)) > 1 and len(get(b)) > 1 else float("nan")
                lines.append(f"- {label}: {a} {get(a).mean():.4f} vs {b} {get(b).mean():.4f} "
                             f"(diff {get(a).mean() - get(b).mean():+.4f}, Welch p={p:.3f}, n=3 -> indicative only)")
        cmp("cnn1d", "rf_baseline", "H1 (deep sequence models > handcrafted baseline)")
        cmp("deepconvlstm_attn", "rf_baseline", "H1")
        cmp("transformer_scratch", "cnn1d", "H2 (Transformer from scratch vs CNN)")
        cmp("transformer_ssl_pretrained", "transformer_scratch", "H2b (SSL pretraining helps Transformer?)")
        cmp("cnn_aug_all", "cnn_aug_none", "H3 (augmentation helps)")
        cmp("cnn_aug_jitter", "cnn_aug_none", "H3 (jitter only)")
    open(os.path.join(TAB, "hypotheses_summary.md"), "w").write("\n".join(lines))
    print("\n".join(lines))


def figures(df):
    for ds in df.dataset.unique():
        d = df[(df.dataset == ds) & (df.group == "main")]
        if d.empty:
            continue
        g = d.groupby("name").test_macro_f1.agg(["mean", "std"]).sort_values("mean")
        plt.figure(figsize=(7, 3.8)); plt.barh(g.index, g["mean"], xerr=g["std"].fillna(0), capsize=4, color="#4c78a8")
        plt.xlabel("Test macro-F1 (mean ± SD over seeds)"); plt.title(f"Model comparison - {ds}")
        plt.xlim(max(0, g["mean"].min() - 0.1), 1.0); plt.tight_layout()
        plt.savefig(os.path.join(FIG, f"model_comparison_{ds}.png"), dpi=200); plt.close()
        # learning curves (seed 0)
        plt.figure(figsize=(7, 4))
        for name, h in d[d.seed == 0].groupby("name"):
            hist = h.history.iloc[0]
            if hist:
                plt.plot([e["val_macro_f1"] for e in hist], label=name)
        plt.xlabel("epoch"); plt.ylabel("validation macro-F1"); plt.legend(); plt.title(f"Learning curves (seed 0) - {ds}")
        plt.tight_layout(); plt.savefig(os.path.join(FIG, f"learning_curves_{ds}.png"), dpi=200); plt.close()
        # per-class heatmap
        rows = {n: np.array(x.f1_per_class.tolist()).mean(0) for n, x in d.groupby("name")}
        cls = d.classes.iloc[0]
        plt.figure(figsize=(1.2 * len(cls) + 2, 0.6 * len(rows) + 1.5))
        plt.imshow(np.array(list(rows.values())), cmap="viridis", vmin=0.5, vmax=1, aspect="auto")
        plt.xticks(range(len(cls)), cls, rotation=35, ha="right"); plt.yticks(range(len(rows)), rows.keys())
        for i, r in enumerate(rows.values()):
            for j, v in enumerate(r):
                plt.text(j, i, f"{v:.2f}", ha="center", va="center", color="w", fontsize=8)
        plt.colorbar(label="F1"); plt.title(f"Per-class F1 - {ds}"); plt.tight_layout()
        plt.savefig(os.path.join(FIG, f"per_class_f1_{ds}.png"), dpi=200); plt.close()
        # ablation / data strategy bars
        for grp, fname in [("ablation", "ablation"), ("data", "data_strategy")]:
            a = df[(df.dataset == ds) & (df.group.isin([grp] + (["main"] if grp == "ablation" else [])))]
            if grp == "ablation":
                keep = {"deepconvlstm_attn", "transformer_scratch"} | set(ABLATION)
                a = a[a.name.isin(keep)]
            if grp == "data" and a.empty:
                continue
            if a.empty or len(a.name.unique()) < 2:
                continue
            g = a.groupby("name").test_macro_f1.agg(["mean", "std"]).sort_values("mean")
            plt.figure(figsize=(7, 0.5 * len(g) + 1.5)); plt.barh(g.index, g["mean"], xerr=g["std"].fillna(0), capsize=4, color="#e45756")
            plt.xlabel("Test macro-F1"); plt.title(f"{fname} - {ds}"); plt.xlim(max(0, g["mean"].min() - 0.05), 1.0)
            plt.tight_layout(); plt.savefig(os.path.join(FIG, f"{fname}_{ds}.png"), dpi=200); plt.close()
    # tuning plot
    plt.figure(figsize=(6, 3.5)); labels, data = [], []
    for p in sorted(glob.glob(os.path.join(TUNE, "*.json"))):
        r = json.load(open(p)); labels.append(r["name"]); data.append([t["val_macro_f1"] for t in r["trials"]])
    if data:
        plt.boxplot(data, labels=labels); plt.xticks(rotation=25, ha="right"); plt.ylabel("val macro-F1")
        plt.title("Hyper-parameter search (same budget per model)"); plt.tight_layout()
        plt.savefig(os.path.join(FIG, "tuning_boxplot.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    df = load()
    if df.empty:
        raise SystemExit("No results found in experiments/results/. Run src.run_all first.")
    table_main(df); table_cost(df); table_arch(df); table_hparams(df)
    table_ablation(df); table_tuning(); hypotheses(df); figures(df)
    print("\nTables ->", TAB, "\nFigures ->", FIG)
