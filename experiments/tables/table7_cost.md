# Table 7 — Computational Cost

Values are reported as mean ± sample standard deviation across three seeds (0, 1, 2).

All experiments were recorded on an NVIDIA Tesla T4 GPU with CPU architecture reported as x86_64.

Inference latency for neural models measures one-window forward-pass latency after 10 warm-up iterations, averaged over 100 iterations.
Random Forest CPU latency includes feature extraction and prediction; therefore, it is not directly comparable to neural-network forward-pass latency.
Random Forest parameter counts are reported as total tree nodes, not trainable neural parameters.

| Dataset | Model | Parameters / tree nodes | Checkpoint (MB) | Training time (s) | Epochs | CPU latency (ms/window) | GPU latency (ms/window) | Peak GPU memory (MB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| UCI_HAR | Random Forest | 119355 tree nodes | 13.50 ± 0.10 | 13.71 ± 0.21 | 1.0 | 75.74 ± 3.25 | N/A | N/A |
| UCI_HAR | CNN1D | 127494 | 0.52 ± 0.00 | 13.59 ± 4.29 | 16.0 | 0.80 ± 0.19 | 0.90 ± 0.28 | 120.92 ± 0.14 |
| UCI_HAR | BiLSTM | 539142 | 2.16 ± 0.00 | 28.95 ± 3.26 | 16.7 | 5.31 ± 1.17 | 1.48 ± 0.00 | 672.18 ± 0.00 |
| UCI_HAR | Transformer (scratch) | 78030 | 0.33 ± 0.00 | 18.04 ± 1.19 | 12.3 | 0.96 ± 0.04 | 1.43 ± 0.07 | 94.11 ± 0.00 |
| UCI_HAR | DeepConvLSTM + Attention | 305606 | 1.24 ± 0.00 | 26.58 ± 14.45 | 19.7 | 3.50 ± 0.56 | 1.27 ± 0.18 | 350.88 ± 0.00 |
| WISDM | Random Forest | 364043 tree nodes | 40.90 ± 0.03 | 36.08 ± 2.31 | 1.0 | 77.41 ± 8.14 | N/A | N/A |
| WISDM | CNN1D | 125574 | 0.51 ± 0.00 | 50.48 ± 28.48 | 22.0 | 0.74 ± 0.16 | 0.83 ± 0.17 | 77.36 ± 0.00 |
| WISDM | BiLSTM | 532998 | 2.14 ± 0.00 | 92.49 ± 16.76 | 31.0 | 3.83 ± 0.57 | 1.03 ± 0.00 | 441.70 ± 0.00 |
| WISDM | Transformer (scratch) | 71454 | 0.30 ± 0.00 | 97.32 ± 47.50 | 23.7 | 1.29 ± 0.27 | 1.75 ± 0.21 | 60.16 ± 0.00 |
| WISDM | DeepConvLSTM + Attention | 303686 | 1.23 ± 0.00 | 68.26 ± 30.86 | 19.3 | 2.43 ± 0.23 | 1.07 ± 0.07 | 231.75 ± 0.00 |

## Notes

- Training time includes the training procedure recorded by the experiment script.
- GPU memory is the recorded peak allocated CUDA memory; it is not system-wide RAM usage.
- N/A indicates a metric that was not measured or does not apply.
- Checkpoint size refers to the saved model artifact.
- Synthetic smoke-test results are excluded from this comparison.
