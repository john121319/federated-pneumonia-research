# Evaluating FedAvg and FedProx for Federated Chest-Radiograph Classification Under Non-IID Client Heterogeneity

**Author:** Yohannes Alelign Biresaw
**Status:** Completed independent research project
**Research areas:** Federated Learning · Deep Learning · Medical Imaging · Non-IID Learning · Distributed Machine Learning · Trustworthy AI

> **Research-use notice:** This repository investigates pneumonia-associated lung-opacity classification from frontal chest radiographs for research and educational purposes. It is not a clinically validated diagnostic system and must not be used for patient care.

---

## Research Outputs

* **Research manuscript:** [Federated Pneumonia Research Manuscript](paper/Federated_Pneumonia_Research_Manuscript.pdf)
* **Markdown manuscript:** [Manuscript source](paper/Federated_Pneumonia_Research_Manuscript.md)
* **Graduate research statement:** [Graduate Research Statement](docs/Graduate_Research_Statement.pdf)
* **Research profile:** [Research Profile](docs/Research_Profile.pdf)
* **Experimental results:** [Results](results/)
* **Figures:** [Research Figures](results/figures/)
* **Result tables:** [Research Tables](results/tables/)
* **Reproduction code:** [Experiments](experiments/)
* **Core implementation:** [Source Code](src/)

---

## Overview

Federated learning enables multiple clients to collaboratively train machine-learning models without directly centralizing their raw data. However, real-world clients rarely contain identically distributed data. Differences in patient populations, disease prevalence, acquisition settings, and other factors can create statistical heterogeneity that may degrade federated optimization and predictive performance.

This project investigates the effect of controlled client heterogeneity on federated chest-radiograph classification.

Using the **RSNA Pneumonia Detection Challenge 2018 dataset**, the study compares:

* centralized training;
* Federated Averaging (**FedAvg**) under approximately IID data;
* FedAvg under moderate non-IID heterogeneity;
* FedAvg under severe non-IID heterogeneity;
* Federated Proximal optimization (**FedProx**) under moderate non-IID heterogeneity; and
* FedProx under severe non-IID heterogeneity.

The study emphasizes patient-aware data separation, reproducible experimentation, multi-seed evaluation, validation-only model selection, direct measurement of client drift, and a frozen final-test protocol.

---

## Main Finding

Increasing client heterogeneity progressively reduced federated predictive performance, particularly **PR-AUC**.

FedProx substantially reduced client and global model-update magnitudes relative to FedAvg, demonstrating stronger control of local optimization drift. However, this reduction in update magnitude did **not consistently translate into improved ROC-AUC or PR-AUC on unseen patients**.

FedAvg retained a small PR-AUC advantage under the evaluated non-IID conditions, while FedProx achieved slightly higher balanced accuracy and F1-score.

This finding illustrates an important distinction:

> **Reducing optimization drift does not automatically guarantee improved generalization performance.**

---

## Final Test PR-AUC Comparison

![Final test PR-AUC comparison](results/figures/final_test_pr_auc_comparison.png)

---

## Research Contributions

This project contributes the following:

1. Developed a patient-aware federated-learning pipeline using the RSNA Pneumonia Detection Challenge dataset.

2. Constructed patient-exclusive training, validation, and test partitions to prevent patient-level information leakage.

3. Created patient-exclusive simulated federated clients while ensuring that all examinations from the same patient remained assigned to a single client.

4. Evaluated FedAvg under approximately IID, moderately non-IID, and severely non-IID client distributions.

5. Evaluated FedProx under the same moderate and severe non-IID conditions.

6. Conducted the primary experiments across three independent random seeds: `11`, `22`, and `33`.

7. Measured client and global model-update magnitudes to directly characterize optimization drift.

8. Used validation-only checkpoint selection and validation-only decision-threshold selection.

9. Performed a frozen final-test evaluation only after model paths, checkpoints, model hashes, validation-report hashes, and thresholds had been fixed.

10. Found that FedProx substantially reduced update magnitudes without consistently improving held-out ROC-AUC or PR-AUC.

---

## Research Questions

The study addresses four main questions:

1. **How does increasing non-IID client heterogeneity affect federated predictive performance?**

2. **Does FedProx reduce client drift relative to FedAvg?**

3. **Does reducing client-update magnitude translate into improved predictive performance on unseen patients?**

4. **How does federated training compare with a centralized reference model under the same patient-aware data split?**

---

## Dataset

The project uses the **RSNA Pneumonia Detection Challenge 2018 dataset**.

* Dataset: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
* RSNA challenge information: https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/rsna-pneumonia-detection-challenge-2018

| Dataset property                 |  Count |
| -------------------------------- | -----: |
| Annotation rows                  | 30,227 |
| Unique radiographic examinations | 26,684 |
| Original patients                | 11,452 |
| Positive examinations            |  6,012 |
| Negative examinations            | 20,672 |

Multiple bounding-box annotations may correspond to the same radiograph. These rows were therefore collapsed into a single examination-level binary target.

The RSNA-to-NIH mapping information was used to recover original patient identifiers for patient-aware splitting.

**Raw DICOM images are not redistributed in this repository.**

---

## Task Definition

The study performs examination-level binary classification:

* `1` — pneumonia-associated lung opacity
* `0` — no pneumonia-associated lung opacity

If a radiograph contains one or more positive opacity annotations, the examination is assigned a positive target.

The target represents the research labeling used in the challenge and must not be interpreted as a definitive clinical diagnosis.

---

## Patient-Aware Data Splitting

A central methodological requirement of this project was preventing examinations belonging to the same patient from appearing across training, validation, and test sets.

The final patient-exclusive split was:

| Split      | Images | Patients | Positive | Negative |
| ---------- | -----: | -------: | -------: | -------: |
| Training   | 18,981 |    8,148 |    4,289 |   14,692 |
| Validation |  3,841 |    1,658 |      846 |    2,995 |
| Test       |  3,862 |    1,646 |      877 |    2,985 |

**Patient overlap across training, validation, and test sets: 0**

This design reduces the risk that patient-specific information is unintentionally shared across experimental partitions.

---

## Image Preprocessing

The DICOM preprocessing pipeline performs the following operations:

1. Loads the original DICOM pixel data.
2. Applies DICOM rescale slope and intercept.
3. Replaces non-finite pixel values.
4. Corrects `MONOCHROME1` grayscale polarity when required.
5. Performs robust intensity clipping.
6. Normalizes image intensity.
7. Resizes images to `128 × 128`.
8. Stores preprocessed images in a lossless 16-bit grayscale PNG cache.

The preprocessing pipeline was validated both numerically and visually before the complete image cache was generated.

A representative preprocessing figure is available here:

![Representative RSNA preprocessing examples](results/figures/rsna_preprocessing_examples.png)

---

## Model Architecture

The same compact convolutional neural network was used across the primary experimental conditions.

The architecture contains:

* convolutional blocks with `32`, `64`, `128`, and `256` filters;
* batch normalization;
* ReLU activation;
* max pooling after the first three convolutional blocks;
* global average pooling;
* dropout with rate `0.30`; and
* a sigmoid binary-classification output.

**Total trainable model size:** 389,537 parameters.

### Main Training Configuration

| Setting                | Value                   |
| ---------------------- | ----------------------- |
| Input shape            | `128 × 128 × 1`         |
| Optimizer              | Adam                    |
| Learning rate          | `0.0005`                |
| Batch size             | `32`                    |
| Federated clients      | `5`                     |
| Communication rounds   | `20`                    |
| Local epochs per round | `1`                     |
| Random seeds           | `11`, `22`, `33`        |
| FedProx coefficient    | `μ = 0.01`              |
| Checkpoint criterion   | Validation PR-AUC       |
| Threshold selection    | Validation Youden index |

---

## Federated Learning Setup

Five simulated federated clients were constructed exclusively from the training partition.

All examinations belonging to an individual patient remained assigned to the same federated client.

Three client-distribution conditions were investigated.

### Approximately IID

Training data were distributed approximately evenly across clients while preserving patient exclusivity.

### Moderate Non-IID

A Dirichlet label-skew partition with:

```text
alpha = 0.5
```

was used to introduce moderate differences in class distribution between clients.

### Severe Non-IID

A stronger Dirichlet label-skew configuration with:

```text
alpha = 0.1
```

was used to produce more severe client heterogeneity.

Smaller Dirichlet concentration values produce stronger differences between client label distributions.

---

## FedAvg

Federated Averaging follows the standard client-server training process:

1. The server initializes the global model.
2. The global model is distributed to participating clients.
3. Each client trains locally using its private partition.
4. Clients return their updated model parameters.
5. The server aggregates client models using data-size-weighted averaging.
6. The aggregated model becomes the global model for the next communication round.

The process is repeated for 20 communication rounds.

---

## FedProx

FedProx extends federated local training by adding a proximal penalty that discourages each client's local model from moving excessively far from the current global model.

The experiments used:

```text
mu = 0.01
```

The purpose of this comparison was to investigate whether controlling local-client drift under heterogeneous data improves optimization behavior and predictive performance.

---

## Model Selection and Evaluation Protocol

Research evaluation was separated into three stages.

### 1. Training

Models were trained only using the training partition.

### 2. Validation

The validation partition was used for:

* checkpoint selection;
* comparison during development; and
* decision-threshold selection.

The primary checkpoint-selection metric was **validation PR-AUC**.

### 3. Final Test

The test set remained isolated from training, checkpoint selection, and threshold selection.

Once the experimental configuration had been finalized, the selected models and thresholds were frozen before final evaluation.

---

# Final Frozen Test Results

Values represent **mean ± sample standard deviation** across seeds `11`, `22`, and `33`.

| Condition                |             ROC-AUC |              PR-AUC |   Balanced Accuracy |            F1-score |
| ------------------------ | ------------------: | ------------------: | ------------------: | ------------------: |
| Centralized              | **0.8295 ± 0.0040** | **0.5800 ± 0.0057** | **0.7510 ± 0.0037** | **0.5745 ± 0.0107** |
| FedAvg IID               |     0.8179 ± 0.0030 |     0.5605 ± 0.0042 |     0.7453 ± 0.0048 |     0.5695 ± 0.0068 |
| FedAvg moderate non-IID  |     0.8131 ± 0.0027 | **0.5553 ± 0.0065** |     0.7355 ± 0.0032 |     0.5541 ± 0.0055 |
| FedAvg severe non-IID    |     0.8070 ± 0.0005 | **0.5440 ± 0.0043** |     0.7312 ± 0.0020 |     0.5529 ± 0.0029 |
| FedProx moderate non-IID |     0.8112 ± 0.0020 |     0.5505 ± 0.0040 | **0.7375 ± 0.0050** | **0.5608 ± 0.0036** |
| FedProx severe non-IID   | **0.8076 ± 0.0019** |     0.5428 ± 0.0092 | **0.7335 ± 0.0030** | **0.5581 ± 0.0076** |

---

## Interpretation of Final Results

### Effect of Heterogeneity

For FedAvg, performance decreased as the client distribution became increasingly heterogeneous.

PR-AUC changed from:

```text
FedAvg IID:
0.5605

FedAvg moderate non-IID:
0.5553

FedAvg severe non-IID:
0.5440
```

This provides empirical evidence that stronger client label heterogeneity made the federated classification problem more difficult under the evaluated configuration.

### FedAvg vs FedProx

FedProx did not consistently improve discrimination metrics.

Under moderate non-IID:

```text
FedAvg PR-AUC  = 0.5553
FedProx PR-AUC = 0.5505
```

Under severe non-IID:

```text
FedAvg PR-AUC  = 0.5440
FedProx PR-AUC = 0.5428
```

However, FedProx achieved slightly higher balanced accuracy and F1-score under both non-IID conditions.

This indicates that the effect of FedProx depended on the evaluation metric and that better control of optimization drift did not automatically produce higher ranking-based predictive performance.

---

# Client-Drift Analysis

Model-update magnitudes were measured directly to investigate the effect of the FedProx proximal term on federated optimization.

At validation-selected checkpoints, FedProx substantially reduced both global and average client-update magnitudes.

| Non-IID condition        | Global-update reduction | Mean client-update reduction |
| ------------------------ | ----------------------: | ---------------------------: |
| Moderate (`alpha = 0.5`) |                  51.27% |                       52.08% |
| Severe (`alpha = 0.1`)   |                  48.83% |                       43.53% |

These results show that FedProx strongly constrained model movement relative to FedAvg.

However, the final predictive results demonstrate that **smaller updates alone were not sufficient to guarantee improved generalization**.

Relevant figures:

* [Global update comparison](results/figures/fedavg_fedprox_global_update_l2_comparison.png)
* [Mean client update comparison](results/figures/fedavg_fedprox_mean_client_update_l2_comparison.png)
* [FedAvg vs FedProx PR-AUC](results/figures/fedavg_fedprox_pr_auc_comparison.png)
* [FedAvg vs FedProx ROC-AUC](results/figures/fedavg_fedprox_roc_auc_comparison.png)
* [FedAvg vs FedProx balanced accuracy](results/figures/fedavg_fedprox_balanced_accuracy_comparison.png)
* [FedAvg vs FedProx F1-score](results/figures/fedavg_fedprox_f1_score_comparison.png)

---

# Experimental Discipline

## Multiple Random Seeds

The principal experiments were evaluated using:

```text
11
22
33
```

Using multiple seeds reduces dependence on a single random initialization or partition realization and allows the variability of the observed results to be reported.

---

## Validation-Only Selection

The final test set was not used to choose:

* hyperparameters;
* checkpoints;
* thresholds; or
* preferred models.

Checkpoint and threshold decisions were made using validation data.

---

## Frozen Final Evaluation

Before final test evaluation, the following were frozen:

* model paths;
* model hashes;
* validation-report hashes;
* selected checkpoints; and
* validation-selected thresholds.

The final protocol files are stored under:

```text
results/tables/final_test_protocol.json
results/tables/final_test_report.json
results/tables/final_test_aggregate_results.csv
results/tables/final_test_evaluation_completed.lock
```

The final test results should **not** be repeatedly rerun to choose more favorable outcomes.

Any future extension should use validation data, a newly defined independent test protocol, or an external dataset.

---

# Repository Structure

```text
federated-pneumonia-research/
│
├── README.md
├── CITATION.cff
├── config.py
├── requirements.txt
│
├── src/
│   └── Core model, data, preprocessing, and federated-learning utilities
│
├── experiments/
│   └── Dataset auditing, preprocessing, training, evaluation,
│       FedAvg, FedProx, and analysis scripts
│
├── research/
│   └── Research documentation and supporting material
│
├── results/
│   ├── figures/
│   └── tables/
│
├── paper/
│   ├── Federated_Pneumonia_Research_Manuscript.md
│   ├── Federated_Pneumonia_Research_Manuscript.pdf
│   └── references.bib
│
├── proposal/
│   └── Research proposal material
│
├── docs/
│   ├── Graduate_Research_Statement.md
│   ├── Graduate_Research_Statement.pdf
│   ├── Research_Profile.md
│   └── Research_Profile.pdf
│
└── portfolio/
    └── Academic_and_Research_Projects.md
```

Raw DICOM images, cached image datasets, large trained models, raw predictions, temporary logs, local environments, and personal documents are intentionally excluded from the public repository.

---

# Reproduction Workflow

Run commands from the repository root.

## 1. Create the Environment

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. Obtain the Dataset

Download the RSNA Pneumonia Detection Challenge dataset separately.

Raw dataset files are not included in this repository.

Configure the local dataset locations using the project configuration before beginning the experiments.

---

## 3. Audit the Raw Dataset

```bash
python -m experiments.audit_rsna
```

This verifies the raw RSNA dataset structure, annotation counts, examination identifiers, and required metadata.

---

## 4. Build the Patient-Aware Manifest

```bash
python -m experiments.build_manifest
```

This creates patient-exclusive training, validation, and test manifests.

---

## 5. Validate DICOM Preprocessing

```bash
python -m experiments.validate_preprocessing
```

This validates the image preprocessing pipeline before full cache generation.

---

## 6. Generate the Image Cache

```bash
python -m experiments.cache_images
```

This converts the original DICOM examinations into the validated `128 × 128` image representation used during training.

---

## 7. Create Federated Client Partitions

```bash
python -m experiments.create_partitions
```

This constructs the approximately IID and controlled non-IID client partitions.

---

## 8. Verify the Data and Model Pipeline

```bash
python -m experiments.smoke_test_pipeline
```

This performs a small end-to-end check of the TensorFlow data and model pipeline.

---

## 9. Federated Training

Available FedAvg options can be inspected using:

```bash
python -m experiments.train_fedavg --help
```

Available FedProx options can be inspected using:

```bash
python -m experiments.train_fedprox --help
```

The exact configurations used for the reported experiments should be reproduced according to the experiment scripts and stored result metadata.

---

# Evaluation Metrics

The study reports multiple metrics because no single metric fully characterizes performance under class imbalance.

### ROC-AUC

Measures ranking performance across classification thresholds.

### PR-AUC

Measures the precision-recall trade-off and is particularly useful when the positive class is less common.

### Sensitivity

Measures the proportion of positive examinations correctly identified.

### Specificity

Measures the proportion of negative examinations correctly identified.

### Balanced Accuracy

Averages sensitivity and specificity so that performance on both classes contributes equally.

### F1-score

Combines precision and recall through their harmonic mean.

---

# Limitations

This study has several important limitations.

1. **Simulated federated clients**
   Clients were created from partitions of one public dataset and do not represent independent hospitals or institutions.

2. **Primary heterogeneity mechanism**
   The main controlled non-IID mechanism was label-distribution skew. Real federated medical systems may also experience acquisition, demographic, hardware, geographic, and institutional heterogeneity.

3. **Single primary CNN architecture**
   The study did not compare multiple neural-network architectures.

4. **Single FedProx coefficient**
   The primary comparison used `μ = 0.01`. A broader coefficient sweep may produce different optimization behavior.

5. **One local epoch**
   Each communication round used one local epoch.

6. **Twenty communication rounds**
   Longer training may alter convergence behavior.

7. **Three random seeds**
   Three seeds support descriptive multi-run comparison but are insufficient for strong claims of statistical significance.

8. **Different FedAvg and FedProx local-training implementations**
   This implementation difference should be considered when interpreting direct optimization comparisons.

9. **No external clinical validation**
   The models were not evaluated on a separate hospital dataset.

10. **Federated learning is not itself a privacy guarantee**
    Keeping raw data decentralized does not automatically prevent information leakage or malicious-client attacks.

---

# Future Research Direction: Secure and Trustworthy Federated Learning

The completed work focuses on **statistical heterogeneity and federated optimization**.

A natural future extension is to investigate what happens when client heterogeneity exists together with **adversarial or compromised participants**.

A central future research question is:

> **How can robust federated aggregation distinguish malicious model updates from legitimate updates produced by naturally heterogeneous non-IID clients?**

Potential future topics include:

* model-poisoning attacks;
* label-flipping attacks;
* Byzantine or malicious clients;
* coordinate-wise median aggregation;
* trimmed-mean aggregation;
* Krum and Multi-Krum;
* geometric-median aggregation;
* trust-based aggregation;
* benign-client false rejection;
* worst-client predictive performance;
* attack-aware evaluation;
* calibration under attack; and
* privacy-security trade-offs when servers inspect individual client updates.

This direction connects the project's experience with non-IID federated learning to broader research in **cybersecurity, trustworthy machine learning, adversarial machine learning, and secure distributed AI**.

> **Important:** This section describes planned future research. The current repository contains no completed malicious-client, poisoning-attack, or robust-aggregation experiments.

---

# Research Significance

The central lesson from this project is that federated-learning performance cannot be evaluated only by whether optimization appears more stable.

Under the experimental conditions studied here:

```text
Increasing heterogeneity
        ↓
Reduced predictive performance

FedProx
        ↓
Reduced client/global update magnitude

But
        ↓
Reduced update magnitude did not consistently improve
ROC-AUC or PR-AUC
```

This motivates further research into the interaction between:

* statistical heterogeneity;
* optimization stability;
* model generalization;
* fairness across clients;
* security;
* privacy; and
* adversarial robustness.

---

# Responsible Use

This repository is intended exclusively for research, education, and methodological experimentation.

The models and results in this repository:

* are not clinically validated;
* are not approved medical devices;
* must not be used to diagnose pneumonia;
* must not be used for treatment decisions;
* must not be used for clinical triage; and
* must not be used for patient management.

---

# Research Status

**Completed:**

* dataset audit;
* patient-aware manifest construction;
* preprocessing validation;
* complete image caching;
* centralized reference experiments;
* approximately IID FedAvg experiments;
* moderate non-IID FedAvg experiments;
* severe non-IID FedAvg experiments;
* moderate non-IID FedProx experiments;
* severe non-IID FedProx experiments;
* multi-seed evaluation;
* client-drift analysis;
* validation-only model selection;
* frozen final-test evaluation;
* result aggregation;
* figures and tables; and
* independent research manuscript.

**Planned future work:**

* adversarial-client experiments;
* poisoning attacks;
* robust aggregation; and
* secure/trustworthy federated-learning research.

---

# Citation

If you use or reference this repository, please cite:

```bibtex
@misc{biresaw2026federatedpneumonia,
  author       = {Yohannes Alelign Biresaw},
  title        = {Federated Learning Under Non-IID Client Heterogeneity for Chest-Radiograph Classification},
  year         = {2026},
  howpublished = {Independent research repository},
  url          = {https://github.com/john121319/federated-pneumonia-research}
}
```

A machine-readable citation file is also available in [`CITATION.cff`](CITATION.cff).

---

# Author

**Yohannes Alelign Biresaw**

BSc in Electrical and Computer Engineering
Haramaya University, Ethiopia

**Research interests:** Cybersecurity · Trustworthy Machine Learning · Federated Learning · Adversarial Machine Learning · Privacy-Preserving Machine Learning · Secure Distributed AI

**Current research experience:** Federated medical-image classification under controlled non-IID client heterogeneity.

**Future research direction:** Security, robustness, and trustworthiness of federated and distributed machine-learning systems.

---

## Disclaimer

This is an independent research project and is not presented as a peer-reviewed publication or a clinically validated medical system.
