# Report and Viva Guide
## Wearable Human Activity Recognition (HAR)
### AI403 — Deep Learning Final Project, Fall 2026
### Team: Latent

---

# Part 1: Technical Report Guide

## 1. Report Requirements

The final technical report should be approximately 15–25 pages and explain the problem, datasets, methodology, experiments, results, limitations, and conclusions.

All reported results must match the experiment files included in the project repository.

## 2. Recommended Report Structure

### 2.1 Abstract
Summarize:
- The Human Activity Recognition problem.
- The two datasets used: UCI HAR and WISDM v1.1.
- The five evaluated models.
- The main findings.
- The practical significance of the comparison.

### 2.2 Introduction
Explain wearable Human Activity Recognition and its applications in:
- Fitness monitoring.
- Healthcare and assisted living.
- Context-aware mobile applications.
- Activity-aware wearable devices.

Introduce the challenges of recognizing activities from sensor signals.

### 2.3 Problem Statement and Objectives
The task is to classify a fixed-length window of inertial sensor measurements into one of six activity classes.

The project objectives are:
- Compare classical machine learning and deep learning approaches.
- Evaluate generalization to unseen subjects.
- Compare performance across two public datasets.
- Investigate the effects of selected model components and data strategies.
- Build a working demonstration interface.

### 2.4 Research Hypotheses
The following hypotheses can guide the discussion. Their verdicts must be based on the actual experimental results.

1. CNN-based temporal models can achieve competitive or better performance than a classical Random Forest baseline.
2. Model performance differs across datasets because of differences in sensor characteristics, participants, and activity distributions.
3. Attention mechanisms and data augmentation may affect generalization, but their benefits should be verified experimentally rather than assumed.

### 2.5 Related Work
Discuss relevant research on:
- Classical feature-based activity recognition.
- CNN-based sensor classification.
- Recurrent models such as LSTM and BiLSTM.
- Transformer architectures for time-series data.
- DeepConvLSTM and attention mechanisms.

Cite the original papers and any adapted implementations.

### 2.6 Datasets
Describe both datasets.

**UCI HAR**
- Smartphone inertial sensor recordings.
- Six activity classes.
- Nine sensor channels in the project's processed representation.
- Window length: 128 samples.

**WISDM v1.1**
- Accelerometer recordings.
- Six activity classes.
- Three channels: acc_x, acc_y, acc_z.
- Sampling frequency: 20 Hz.
- Window length: 80 samples.

Refer to `dataset_documentation.md` for exact split counts, subject IDs, class distributions, preprocessing, and licensing information.

### 2.7 Data Preprocessing
Explain:
- Sensor signal processing.
- Window construction.
- Label encoding.
- Normalization using training-set statistics.
- Subject-independent train, validation, and test splits.

Discuss how the split strategy helps prevent subject overlap between the partitions.

### 2.8 Methodology and Models
Describe the five approaches:

1. Random Forest using hand-crafted features — classical baseline.
2. CNN1D — temporal convolutional network.
3. BiLSTM — bidirectional recurrent model.
4. Transformer encoder trained from scratch.
5. DeepConvLSTM with attention — convolutional and recurrent feature extraction with attention.

Include architecture diagrams or tables where useful. Report actual architecture and parameter details from the implementation.

### 2.9 Experimental Setup
Document:
- Training and validation strategy.
- Subject-independent splitting.
- Random seeds: 0, 1, and 2.
- Optimizer and learning rate.
- Batch size and number of epochs.
- Loss function.
- Regularization and augmentation.
- Early stopping and model selection.
- Hardware and software environment.

Use the actual configuration files and experiment logs as the source of these details.

### 2.10 Main Results
Present the test Macro-F1 results as mean ± sample standard deviation over three seeds.

| Model | UCI HAR Macro-F1 | WISDM Macro-F1 |
|---|---:|---:|
| Random Forest | 91.98 ± 0.04% | 85.42 ± 0.05% |
| CNN1D | 93.34 ± 1.92% | 82.07 ± 1.05% |
| BiLSTM | 90.80 ± 0.41% | 84.48 ± 1.21% |
| Transformer (scratch) | 91.18 ± 1.29% | 75.60 ± 1.60% |
| DeepConvLSTM + Attention | 91.26 ± 1.11% | 82.39 ± 5.00% |

Discuss why the best model differs between the datasets.

### 2.11 Validation-Only Experiments
Discuss augmentation and ablation experiments separately from the main test comparison.

Important: the data-strategy and ablation experiments described in the project are validation-only experiments. Do not present their validation scores as test results.

Explain the observed changes and possible reasons without claiming that a validation improvement proves better test generalization.

### 2.12 Error Analysis
Discuss the main confusion patterns:

**UCI HAR**
- Sitting and Standing are among the more difficult activity pairs.
- Random Forest showed stronger performance on these classes.

**WISDM**
- Upstairs and Downstairs are difficult to distinguish.
- Walking can also be confused with stair-related activities.

Use actual confusion matrices and example predictions from the saved experiment outputs.

### 2.13 Computational Cost
Use:

`experiments/tables/table7_cost.md`

Discuss:
- Training time.
- Inference latency.
- Model size.
- Model complexity.
- GPU memory, where measured.
- Hardware environment.

Clarify that Random Forest complexity is represented by tree nodes, not neural-network trainable parameters.

### 2.14 Demo
Describe the Gradio application in `demo/app.py`.

Explain:
- Input sensor CSV.
- Model/checkpoint selection.
- Predicted activity.
- Class probabilities.

Include screenshots of the working demo if available.

### 2.15 Limitations and Ethical Considerations
Discuss:
- Limited number of datasets and participants.
- Differences between dataset collection conditions.
- Sensor placement and device variability.
- Class imbalance.
- Potential errors in real-world activity recognition.
- Privacy considerations when collecting wearable sensor data.
- The fact that predictions should not be treated as medical diagnoses.

### 2.16 Conclusion and Future Work
Summarize the findings and suggest:
- Testing on additional datasets.
- More robust cross-device evaluation.
- Self-supervised representation learning.
- Better handling of confusing activity pairs.
- Real-time deployment and energy-efficiency evaluation.

### 2.17 References
Include complete bibliographic references for datasets, architectures, and methods. Verify publication details before submission.

---

# Part 2: Viva Preparation Guide

## 3. Important Questions and Suggested Answers

### Q1. What is Human Activity Recognition?
Human Activity Recognition is the task of identifying a person's physical activity from sensor measurements, such as accelerometer and gyroscope signals.

### Q2. Why did you choose this problem?
It has practical applications in wearable devices, fitness monitoring, healthcare support, and context-aware systems. It also provides a useful time-series classification problem for comparing different machine learning architectures.

### Q3. What is the input and output of your system?
The input is a fixed-length window of sensor measurements. The output is one of six activity classes.

### Q4. Why did you use two datasets?
UCI HAR and WISDM provide different sensor characteristics and collection conditions. Using both helps evaluate how model performance changes across datasets.

### Q5. Why use subject-independent splitting?
It evaluates whether a model generalizes to participants it did not see during training. This is more realistic than randomly splitting windows from the same participants across train and test.

### Q6. What is Macro-F1?
Macro-F1 is the arithmetic mean of the F1-score calculated separately for each class. It gives every class equal weight, making it useful when class frequencies are imbalanced.

### Q7. Why include Random Forest?
It provides a classical machine-learning baseline using hand-crafted features. It helps determine whether deep learning provides an advantage over a simpler approach.

### Q8. Why use CNN1D?
CNN1D can learn local temporal patterns from sensor sequences using convolutional filters.

### Q9. What is the advantage of BiLSTM?
A BiLSTM processes a sequence in both forward and backward directions, allowing it to use temporal context from both directions within the input window.

### Q10. What is a Transformer?
A Transformer uses attention mechanisms to model relationships between elements in a sequence. In this project, the Transformer was trained from scratch.

### Q11. What does attention do in DeepConvLSTM?
Attention assigns different importance weights to learned temporal features, helping the model focus on relevant parts of the sequence.

### Q12. Which model performed best?
CNN1D achieved the highest test Macro-F1 on UCI HAR at 93.34 ± 1.92%. Random Forest achieved the highest test Macro-F1 on WISDM at 85.42 ± 0.05%.

### Q13. Why did the best model differ between datasets?
The datasets differ in sensor channels, sampling conditions, participant characteristics, and class distributions. These differences can favor different model families.

### Q14. What is data leakage?
Data leakage occurs when information from validation or test data influences training or model selection inappropriately. It can produce overly optimistic evaluation results.

### Q15. How did you reduce data leakage?
The project uses subject-independent partitions and computes normalization statistics from training data only. Augmentation is applied to training batches. Validation is used for model selection.

### Q16. What is the purpose of ablation studies?
Ablation studies remove or change selected components to investigate their contribution to model behavior.

### Q17. Did augmentation improve performance?
The validation-only experiments showed that scaling achieved the highest validation Macro-F1 among the tested UCI HAR data strategies. This does not establish a test-set improvement.

### Q18. What were the main classification errors?
For UCI HAR, Sitting and Standing were difficult to distinguish. For WISDM, Upstairs and Downstairs were among the most confused activities.

### Q19. Why is computational cost important?
A model with high accuracy may still be unsuitable for wearable devices if it requires too much memory, computation, or inference time.

### Q20. What are the limitations of your project?
The evaluation uses two datasets and a limited set of participants and sensor configurations. Performance may change under different devices, placements, and real-world conditions.

---

## 4. Individual Viva Preparation

Every team member should be ready to explain:

- The overall project objective.
- The dataset and preprocessing pipeline.
- The model or implementation they contributed to.
- The experiments and results they worked on.
- The challenges they encountered.
- The limitations of the project.

Each member should understand the full pipeline, not only their individual task.

## 5. Final Viva Checklist

Before the presentation and viva, make sure you can:

- Explain the input shape and output classes.
- Explain the subject-independent split.
- Describe all five models.
- Explain Macro-F1 and why it was selected.
- Compare the best results on both datasets.
- Interpret confusion matrices.
- Explain the validation-only ablations and data strategies.
- Discuss computational cost.
- Run the Gradio demo.
- Explain the limitations and possible future improvements.

**Important:** These are preparation questions and suggested answers. Team members should understand the actual implementation and be able to explain their own contributions honestly.

---

# Part 3: Submission Consistency Checklist

Before final submission, verify that:

- The README matches the actual repository.
- Dataset documentation matches the processed data and split files.
- Main results match the saved experiment records.
- Validation-only results are clearly labeled.
- The computational-cost table is included.
- Checkpoints correspond to the reported experiments.
- Demo instructions work in the submitted environment.
- Presentation claims match the report.
- Individual contribution logs are completed honestly.
- The final report includes references and limitations.
- The final ZIP excludes raw datasets and unnecessary temporary files.

