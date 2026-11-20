"""Run a whole suite of experiments x seeds (skips runs whose result file already exists).

    python -m src.run_all --suite main --seeds 0 1 2
    python -m src.run_all --suite data ablation --seeds 0 1 2
    python -m src.run_all --suite main data --dataset wisdm     # second dataset
Suites: main | data | ablation
"""
import argparse
import glob
import os

from src.train import RESULTS, run
from src.utils import ROOT, apply_overrides, exp_name, load_config

C = os.path.join(ROOT, "configs")
SUITES = {
    "main": ["baseline_rf.yaml", "cnn1d.yaml", "lstm.yaml", "transformer.yaml", "deepconvlstm.yaml"],
    "data": sorted(os.path.basename(p) for p in glob.glob(os.path.join(C, "data_strategy", "*.yaml"))),
    "ablation": sorted(os.path.basename(p) for p in glob.glob(os.path.join(C, "ablation", "*.yaml"))),
}
SUBDIR = {"main": "", "data": "data_strategy", "ablation": "ablation"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", nargs="+", default=["main"], choices=list(SUITES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--dataset", default="uci_har", choices=["uci_har", "wisdm", "synthetic"])
    ap.add_argument("--epochs", type=int, default=None, help="override max epochs (testing only)")
    ap.add_argument("--force", action="store_true", help="re-run even if result exists")
    a = ap.parse_args()
    for suite in a.suite:
        for fn in SUITES[suite]:
            cfg = load_config(os.path.join(C, SUBDIR[suite], fn))
            if a.dataset == "wisdm":
                if cfg.get("only_dataset") == "uci_har":
                    continue                      # e.g. gyroscope ablation: WISDM has no gyro
                cfg = apply_overrides(cfg, {"dataset": "wisdm", "channels": None})
            elif a.dataset == "synthetic":
                cfg = apply_overrides(cfg, {"dataset": "synthetic"})
            if a.epochs and cfg["model"] != "baseline":
                cfg = apply_overrides(cfg, {"train.epochs": a.epochs})
            for seed in a.seeds:
                name = exp_name(cfg, seed)

                if suite == "main":
                    out_dir = RESULTS
                    evaluate_test = True
                else:
                    out_dir = os.path.join(
                        ROOT, "experiments", "validation_only",
                        cfg.get("group", suite)
                    )
                    evaluate_test = False

                result_path = os.path.join(out_dir, name + ".json")
                if not a.force and os.path.exists(result_path):
                    print("skip (exists):", result_path)
                    continue

                run(cfg, seed, evaluate_test=evaluate_test)
