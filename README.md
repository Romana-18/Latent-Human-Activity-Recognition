# Wearable Human Activity Recognition (HAR)
## Deep Learning Final Project — AI403, Fall 2026

**Team Name:** Latent

**Team Members:**
1. Asmaa Reda — 202302036
2. Sondos Alaa — 202302031
3. Donia Sayed — 202303000
4. Aya Mohamed Saieed — 202303640
5. Hager Mahmoud Mohamed Ramadan — 202303727

**Selected Idea:** #9 — Wearable Human Activity Recognition (inertial sensors → activity)

---
## Google Colab Notebook

The complete Google Colab notebook used to run the experiments, train and evaluate the models, generate result tables and figures, and launch the live demo is available here:

[**Open the Human Activity Recognition – Final Project Notebook**](https://colab.research.google.com/drive/1AX_5f8LvdUvHhtNUnUyiLRuNaPcW_h3p?usp=sharing)

## Large Files

Trained model checkpoints and other large project files are available on Google Drive:

[Access Project Files and Model Checkpoints](https://drive.google.com/drive/folders/1NzH95IteW3bbGT_5PogUwXKDdg6Qr_j2?usp=sharing)

The folder contains trained model checkpoints and other required large files that are not included in the submission ZIP.

Raw datasets are not included in the submission archive.

Access is provided to the instructor and teaching assistants.
## Demo Video

[Watch the Wearable Human Activity Recognition Demo](https://drive.google.com/file/d/1TzoyNhO7MjAdAprbu7HzzDxoh38bFpWN/view?usp=sharing)


## 1. Task and Main Results

The goal of this project is to classify short windows of wearable inertial sensor signals into six human activities.

We compare five approaches:

- Random Forest using hand-crafted features (baseline).
- 1D Convolutional Neural Network (CNN1D).
- Bidirectional LSTM (BiLSTM).
- Transformer encoder trained from scratch.
- DeepConvLSTM with attention.

All models are evaluated using subject-independent data splits and three random seeds per model and dataset. Macro-F1 is the primary evaluation metric.

### Best Test Results

Test Macro-F1 (mean ± sample standard deviation over three seeds):

- **UCI HAR:** CNN1D — **93.34 ± 1.92%**
- **WISDM:** Random Forest — **85.42 ± 0.05%**

### Main Test Results

| Model | UCI HAR Macro-F1 | WISDM Macro-F1 |
|---|---:|---:|
| Random Forest | 91.98 ± 0.04% | **85.42 ± 0.05%** |
| CNN1D | **93.34 ± 1.92%** | 82.07 ± 1.05% |
| BiLSTM | 90.80 ± 0.41% | 84.48 ± 1.21% |
| Transformer (scratch) | 91.18 ± 1.29% | 75.60 ± 1.60% |
| DeepConvLSTM + Attention | 91.26 ± 1.11% | 82.39 ± 5.00% |

### Main Observations

- CNN1D achieved the highest Macro-F1 on UCI HAR.
- Random Forest achieved the highest Macro-F1 on WISDM, followed by BiLSTM.
- The scratch Transformer had the lowest Macro-F1 on WISDM among the five evaluated models.
- DeepConvLSTM + Attention showed relatively high seed-to-seed variability on WISDM.

Detailed experiment records are available under:

```text
experiments/results/
```

---

## 2. Environment Setup

Create and activate a Python virtual environment:

```bash
python -m venv .venv
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The project can also be executed using Google Colab or Kaggle, subject to the required environment and dataset availability.

---

## 3. Dataset Download and Preprocessing

The project uses two public Human Activity Recognition datasets:

- **UCI HAR:** Smartphone inertial sensor recordings.
- **WISDM v1.1:** Wearable/mobile-device accelerometer recordings.

Dataset documentation, preprocessing details, subject-independent split information, and licensing notes are provided in:

```text
docs/dataset_documentation.md
```

Preprocessing uses training-set statistics for normalization to avoid information leakage from validation and test sets.

Both datasets follow subject-independent splitting to evaluate generalization to unseen participants.

Raw datasets are not included in the submission archive. Use the dataset download instructions and scripts provided in the repository after verifying their availability.

---

## 4. Training Commands

The project uses configuration files under `configs/` to define model and experiment settings.

The intended individual training commands are:

```bash
python -m src.train --config configs/baseline_rf.yaml --seed 0
python -m src.train --config configs/cnn1d.yaml --seed 0
python -m src.train --config configs/lstm.yaml --seed 0
python -m src.train --config configs/transformer.yaml --seed 0
python -m src.train --config configs/deepconvlstm.yaml --seed 0
```

The WISDM equivalents live in `configs/wisdm/`:

```bash
python -m src.train --config configs/wisdm/wisdm_rf.yaml --seed 0
python -m src.train --config configs/wisdm/wisdm_cnn1d.yaml --seed 0
python -m src.train --config configs/wisdm/wisdm_lstm.yaml --seed 0
python -m src.train --config configs/wisdm/wisdm_transformer.yaml --seed 0
python -m src.train --config configs/wisdm/wisdm_deepconvlstm.yaml --seed 0
```

If the SSL configuration and implementation are included, the Transformer pre-training experiment can also be run using its corresponding configuration.

The reported main comparison uses three seeds (0, 1, and 2) for each of the five models on both datasets.

Experiment results are stored in:

```text
experiments/results/
```

Before running the commands, verify that the referenced configuration files and Python scripts exist in the submitted repository.

Optional orchestration commands should only be used if their corresponding scripts are included and tested.

---

## 5. Evaluation

The evaluation implementation is available at:

```text
src/evaluate.py
```

Example evaluation command:

```bash
python -m src.evaluate --checkpoint checkpoints/uci_har__deepconvlstm_attn__seed0.pt
```

The main evaluation metrics include:

- Macro-F1 (primary metric).
- Weighted-F1.
- Balanced accuracy.
- Accuracy.
- Cohen's kappa.

Results are reported as mean ± sample standard deviation over three seeds.

Per-class precision, recall, F1-score, and confusion matrices are used for more detailed error analysis.

---

## 6. Demo

The Gradio demo is available at:

```text
demo/app.py
```

Run the demo using:

```bash
python demo/app.py
```

The interface accepts a sensor CSV file and displays the predicted activity and class probabilities.

Validation examples for both UCI HAR and WISDM were tested successfully.

Example files, where included, are available under:

```text
demo/examples/validation/
```

The demo supports the project's trained model checkpoints, including `.pt` and `.joblib` files.

---

## 7. Hardware and Computational Cost

The experiments were conducted using Google Colab with an NVIDIA Tesla T4 GPU and an x86_64 CPU environment.

Computational cost was measured for all five models across both UCI HAR and WISDM datasets.

The recorded measurements include:

- Training time per run.
- Inference latency.
- Model/checkpoint size.
- Model complexity.
- Peak GPU memory usage, where applicable.

Detailed computational-cost results, including mean and standard deviation across the three seeds, are available in:

```text
experiments/tables/table7_cost.md
```

Random Forest inference latency includes feature extraction and prediction. Neural-network inference latency measures a single-window forward pass after warm-up iterations.

For Random Forest, model complexity is reported as the total number of tree nodes rather than the number of trainable neural-network parameters.

GPU memory measurements are not applicable to the CPU-based Random Forest implementation.

Detailed CPU model and system RAM specifications were not available in the recorded experiment logs and are therefore not reported.

---

## 8. Large Files and Shared Drive

Model checkpoints and processed data that are too large for the submission ZIP should be stored in Google Drive.

**Shared Drive Link:** still missing. The accessible folder link has not been added yet and must be inserted here before the final submission. Until then, the model checkpoints are available in `checkpoints/` inside the repository.

The shared folder should contain the required model checkpoints and any necessary processed data that are not included in the ZIP archive.

Raw datasets should not be uploaded unless explicitly required.

Make sure the instructor and teaching assistants have permission to access the shared folder.

---

## 9. Credits and References

### Datasets

- UCI HAR — Anguita et al. (2013), ESANN.
- WISDM v1.1 — Kwapisz et al. (2011), SIGKDD Explorations.

### Architectures and Methods

- DeepConvLSTM — Ordonez and Roggen (2016), Sensors.
- Transformer — Vaswani et al. (2017), NeurIPS.
- Additive Attention — Bahdanau et al. (2015), ICLR.
- Masked Autoencoders — He et al. (2022), CVPR.

### Implementation and Libraries

- PyTorch.
- scikit-learn.
- pandas.
- NumPy.
- SciPy.
- Matplotlib.
- Gradio.

Custom implementations and adaptations should be clearly identified, and their original sources should be cited.

### AI Assistance Disclosure

ChatGPT was used for selected troubleshooting, code and documentation review, and assistance with written explanations. The project team reviewed and verified the submitted implementation and reported results.

---

## 10. Repository Structure

The main project repository is organized into the following directories:

```text
har_final_submission/
├── README.md
├── requirements.txt
├── configs/
├── data/
├── src/
├── experiments/
│   ├── results/
│   ├── validation_only/
│   └── tables/
├── checkpoints/
├── demo/
├── report/
├── contributions/
├── docs/
└── presentation.pptx
```

The repository also includes the project source code, experiment configurations, evaluation outputs, documentation, and demo resources.

---

## 11. Submission Notes

The final submission includes the project source code, dataset documentation, experiment results, model checkpoints, demo resources, presentation slides, and technical report.

The final ZIP archive should contain the required project files while excluding raw datasets and unnecessary temporary files.

The technical report PDF and individual contribution logs should be added before creating the final submission archive.

The shared Google Drive link should be updated after uploading the required large files.

