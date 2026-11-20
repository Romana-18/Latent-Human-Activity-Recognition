"""Live demo (Gradio). Upload a CSV of sensor readings -> predicted activity + probabilities.

    python demo/app.py                 # then open the printed local URL
CSV format: one row per time step, one column per sensor channel, in the same order as the
model's training channels (shown in the UI). Header row optional. >= one window of rows;
longer recordings are cut into overlapping windows and the predictions are aggregated.
"""
import glob
import os
import sys

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.evaluate import Predictor  # noqa: E402

CKPTS = sorted(glob.glob(os.path.join(ROOT, "checkpoints", "*.pt")) + glob.glob(os.path.join(ROOT, "checkpoints", "*.joblib")))
CKPTS = [c for c in CKPTS if "synthetic" not in c]
_cache = {}

# The project's two datasets have a fixed channel count, so a CSV with 9 columns is
# UCI HAR data and a CSV with 3 columns is WISDM data.
COLUMNS_TO_DATASET = {9: "UCI HAR", 3: "WISDM"}
PREFIX_TO_DATASET = {"uci_har": "UCI HAR", "wisdm": "WISDM"}
PREFIX = {"UCI HAR": "uci_har", "WISDM": "wisdm"}


def ckpt_dataset(ckpt_name):
    """Dataset a checkpoint was trained on, read from its file name prefix."""
    for prefix, label in PREFIX_TO_DATASET.items():
        if os.path.basename(ckpt_name).startswith(prefix):
            return label
    return None


def get_model(path):
    if path not in _cache:
        _cache[path] = Predictor(path)
    return _cache[path]


def read_csv(file):
    df = pd.read_csv(file if isinstance(file, str) else file.name, header=None)
    first = df.iloc[0].astype(str).tolist()
    try:
        [float(v) for v in first]
    except ValueError:                       # first row is a header
        df = df.iloc[1:]
    return df.astype(float).values            # (T, C)


def predict(file, ckpt_name):
    if file is None:
        raise gr.Error("Upload a CSV file first.")
    if not ckpt_name:
        raise gr.Error("No checkpoint found in checkpoints/ - train a model first.")
    pr = get_model(os.path.join(ROOT, "checkpoints", ckpt_name))
    sig = read_csv(file)
    model_ds, data_ds = ckpt_dataset(ckpt_name), COLUMNS_TO_DATASET.get(sig.shape[1])
    if model_ds and data_ds and model_ds != data_ds:
        raise gr.Error(
            f"Dataset mismatch: '{os.path.basename(ckpt_name)}' is trained on {model_ds} "
            f"({pr.in_channels} channels), but this CSV has {sig.shape[1]} columns, which is "
            f"{data_ds} data. Choose a {data_ds} checkpoint (a file starting with "
            f"'{PREFIX[data_ds]}__'), or upload a {model_ds} CSV with {pr.in_channels} columns.")
    if sig.shape[1] != pr.in_channels:
        raise gr.Error(f"This model expects {pr.in_channels} columns {pr.channel_names}, got {sig.shape[1]}.")
    T = pr.seq_len
    if sig.shape[0] < T:
        raise gr.Error(f"Need at least {T} rows (one window), got {sig.shape[0]}.")
    starts = list(range(0, sig.shape[0] - T + 1, max(1, T // 2)))
    W = np.stack([sig[s:s + T].T for s in starts]).astype(np.float32)      # (N, C, T)
    probs = pr.predict_proba(W)
    mean_p = probs.mean(0)
    label = {c: float(p) for c, p in zip(pr.classes, mean_p)}
    table = pd.DataFrame({"window": range(len(starts)), "start_row": starts,
                          "prediction": [pr.classes[i] for i in probs.argmax(1)],
                          "confidence": probs.max(1).round(3)})
    fig, ax = plt.subplots(figsize=(7, 3))
    for c in range(min(3, sig.shape[1])):
        ax.plot(sig[:, c], label=pr.channel_names[c])
    ax.legend(fontsize=7); ax.set_xlabel("time step"); plt.tight_layout()
    return label, table, fig


def channel_info(ckpt_name):
    if not ckpt_name:
        return "No checkpoints found."
    pr = get_model(os.path.join(ROOT, "checkpoints", ckpt_name))
    return f"Window = {pr.seq_len} rows; {pr.in_channels} columns in this order: {', '.join(pr.channel_names)}"


examples = sorted(glob.glob(os.path.join(ROOT, "demo", "examples", "**", "*.csv"), recursive=True))
with gr.Blocks(title="Wearable Human Activity Recognition") as demo:
    gr.Markdown("# Wearable Human Activity Recognition\nUpload a CSV of inertial sensor readings; the trained model predicts the activity.")
    default = next((os.path.basename(c) for c in CKPTS if "deepconvlstm" in c), os.path.basename(CKPTS[0]) if CKPTS else None)
    ckpt = gr.Dropdown([os.path.basename(c) for c in CKPTS], value=default, label="Model checkpoint")
    info = gr.Markdown(channel_info(default) if default else "")
    f = gr.File(label="Sensor CSV", file_types=[".csv"])
    btn = gr.Button("Predict", variant="primary")
    with gr.Row():
        out_label = gr.Label(label="Predicted activity (mean over windows)")
        out_plot = gr.Plot(label="Signal")
    out_table = gr.Dataframe(label="Per-window predictions")
    ckpt.change(channel_info, ckpt, info)
    btn.click(predict, [f, ckpt], [out_label, out_table, out_plot])
    if examples:
        gr.Examples(examples=[[e] for e in examples], inputs=[f], label="Examples from the test set")

if __name__ == "__main__":
    demo.launch(share=False)
