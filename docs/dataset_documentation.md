# Dataset Documentation
## Wearable Human Activity Recognition (HAR) — AI403 Final Project

This document describes the two public datasets used in the project, their sensor inputs, windowing, subject-independent splits, class distributions, preprocessing, and experimental roles.

## 1. UCI HAR — Primary Dataset

| Item | Description |
|---|---|
| Dataset name | Human Activity Recognition Using Smartphones (UCI HAR) |
| Source | [UCI Machine Learning Repository — Dataset 240](https://archive.ics.uci.edu/dataset/240) |
| Reference | Anguita, D., Ghio, A., Oneto, L., Parra, X., and Reyes-Ortiz, J. L. (2013). *A Public Domain Dataset for Human Activity Recognition Using Smartphones*. ESANN. |
| Participants | 30 volunteers |
| Device placement | Waist-mounted smartphone |
| Sensors | 3-axis accelerometer and 3-axis gyroscope |
| Sampling frequency | 50 Hz |
| Activities | Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying |
| Window length | 128 samples (2.56 seconds) |
| Window overlap | 50% |
| Model input | 9 channels × 128 time steps |

### Channels Used

The deep-learning models use the nine raw inertial-signal channels:

1. `total_acc_x`
2. `total_acc_y`
3. `total_acc_z`
4. `body_acc_x`
5. `body_acc_y`
6. `body_acc_z`
7. `body_gyro_x`
8. `body_gyro_y`
9. `body_gyro_z`

These are raw time-series channels, not the 561 engineered features supplied with the original UCI HAR dataset. The Random Forest baseline computes hand-crafted features from the same normalized signal windows.

### Subject-Independent Split

The original UCI HAR training partition contains 21 subjects, while the official test partition contains 9 subjects. Four of the 21 original training subjects are held out for validation using the project's subject-level split procedure (`RandomState(42)`). Therefore, the final train, validation, and test subject sets are disjoint.

| Split | Subjects | Windows |
|---|---:|---:|
| Train | 17 | 5,879 |
| Validation | 4 | 1,473 |
| Test | 9 | 2,947 |
| **Total** | **30** | **10,299** |

Subject IDs reported by the dataset statistics script:

- **Train:** 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 26, 28, 29, 30
- **Validation:** 1, 3, 25, 27
- **Test:** 2, 4, 9, 10, 12, 13, 18, 20, 24

### UCI HAR Class Distribution

| Activity | Train windows | Validation windows | Test windows |
|---|---:|---:|---:|
| Walking | 942 | 284 | 496 |
| Walking Upstairs | 845 | 228 | 471 |
| Walking Downstairs | 786 | 200 | 420 |
| Sitting | 1,052 | 234 | 491 |
| Standing | 1,106 | 268 | 532 |
| Laying | 1,148 | 259 | 537 |
| **Total** | **5,879** | **1,473** | **2,947** |

### Preprocessing

- The nine raw inertial-signal channels are loaded for each window.
- Per-channel normalization uses the mean and standard deviation calculated from the training split only.
- The normalization statistics are stored with the model checkpoints and reused for validation, test, and demo inference.
- Training-time augmentation (jitter, scaling, and time-warping, where enabled) is applied to training batches only.
- Validation and test data are not augmented.
- The Random Forest baseline derives hand-crafted features from the same normalized signal windows.

### Leakage Prevention

The split is subject-independent: no subject appears in more than one of train, validation, or test. This is important because adjacent windows overlap by 50%; randomly splitting windows could place highly similar windows from the same person in different partitions and produce overly optimistic evaluation results.

The test partition is reserved for final evaluation. Model selection and early stopping use validation data.

### Licence

The project notes the UCI dataset page's stated licence information as CC BY 4.0. The exact licence wording and attribution requirements should be checked against the current UCI dataset page before final publication or redistribution.

## 2. WISDM v1.1 — Second Benchmark Dataset

| Item | Description |
|---|---|
| Dataset name | WISDM v1.1 |
| Source | [WISDM Dataset Page](https://www.cis.fordham.edu/wisdm/dataset.php) |
| Raw file | `WISDM_ar_v1.1_raw.txt` |
| Reference | Kwapisz, J. R., Weiss, G. M., and Moore, S. A. (2011). *Activity Recognition using Cell Phone Accelerometers*. SIGKDD Explorations, 12(2). |
| Participants | 36 users |
| Sensor | Phone accelerometer |
| Channels | 3 axes: `acc_x`, `acc_y`, `acc_z` |
| Sampling frequency | 20 Hz |
| Activities | Walking, Jogging, Upstairs, Downstairs, Sitting, Standing |
| Window length | 80 samples (4 seconds) |
| Window overlap | 50% |
| Model input | 3 channels × 80 time steps |

### Subject-Independent Split

WISDM is split by user, with 22 users for training, 6 for validation, and 8 for testing. The split uses the project's subject-level procedure (`RandomState(42)`), and the three sets contain disjoint users.

| Split | Users | Windows |
|---|---:|---:|
| Train | 22 | 16,068 |
| Validation | 6 | 3,956 |
| Test | 8 | 6,537 |
| **Total** | **36** | **26,561** |

User IDs reported by the dataset statistics script:

- **Train:** 2, 3, 4, 6, 7, 8, 11, 12, 15, 16, 19, 20, 21, 23, 24, 25, 26, 28, 29, 30, 33, 34
- **Validation:** 1, 5, 9, 10, 18, 35
- **Test:** 13, 14, 17, 22, 27, 31, 32, 36

### WISDM Class Distribution

| Activity | Train windows | Validation windows | Test windows |
|---|---:|---:|---:|
| Walking | 6,347 | 1,754 | 2,268 |
| Jogging | 4,881 | 1,340 | 2,109 |
| Upstairs | 1,591 | 310 | 992 |
| Downstairs | 1,371 | 291 | 671 |
| Sitting | 1,083 | 113 | 267 |
| Standing | 795 | 148 | 230 |
| **Total** | **16,068** | **3,956** | **6,537** |

### Preprocessing

- The raw WISDM accelerometer stream is parsed into user/activity sequences.
- Windows contain 80 consecutive samples with 50% overlap and are formed within contiguous runs of the same user and activity.
- The model input contains the three accelerometer axes.
- Normalization uses training-split statistics only; the same stored statistics are applied to validation, test, and demo inputs.
- Training-time augmentation, when enabled, is applied only to training batches. Validation and test data are not augmented.

### Licence and Use

The project source describes WISDM v1.1 as available for research use. A formal licence statement was not confirmed in the project documentation. Check the current WISDM dataset page and its terms before redistribution, and do not describe the dataset as having a specific open-source licence unless the source explicitly confirms it.

## 3. Dataset Comparison

| Dataset | Source | Total windows | Input format | Classes | Split strategy | Experimental role |
|---|---|---:|---|---:|---|---|
| UCI HAR | UCI Machine Learning Repository | 10,299 | 9 × 128 | 6 | Subject-independent: 17 train / 4 validation / 9 test | Primary benchmark |
| WISDM v1.1 | WISDM dataset page | 26,561 | 3 × 80 | 6 | User-independent: 22 train / 6 validation / 8 test | Second benchmark |

WISDM is evaluated as a separate benchmark. It has a different sensor configuration, sampling frequency, window length, and activity distribution from UCI HAR. The experiments therefore assess model behavior on two datasets independently; they do not constitute cross-dataset transfer learning.

## 4. Reproducibility and Verification

The split sizes, subject/user IDs, input dimensions, and class counts in this document were obtained by running:

```bash
python -m src.dataset_stats --dataset uci_har
python -m src.dataset_stats --dataset wisdm
```

The corresponding implementation is `src/dataset_stats.py`, which loads the project data through `src.datasets.get_data`.

Before final submission, verify that:

- The split implementation keeps subject/user IDs disjoint across train, validation, and test.
- Normalization statistics are fitted on training data only.
- Augmentation is restricted to training data.
- The dataset source pages and licence/terms are checked and cited in the final technical report.
- Raw datasets are not included in the submission ZIP unless the course explicitly requests them.

## 5. Dataset References

1. Anguita, D., Ghio, A., Oneto, L., Parra, X., and Reyes-Ortiz, J. L. (2013). *A Public Domain Dataset for Human Activity Recognition Using Smartphones*. ESANN.
2. Kwapisz, J. R., Weiss, G. M., and Moore, S. A. (2011). *Activity Recognition using Cell Phone Accelerometers*. SIGKDD Explorations, 12(2).
3. UCI Machine Learning Repository. [Human Activity Recognition Using Smartphones (Dataset 240)](https://archive.ics.uci.edu/dataset/240).
4. WISDM. [WISDM Dataset Page](https://www.cis.fordham.edu/wisdm/dataset.php).