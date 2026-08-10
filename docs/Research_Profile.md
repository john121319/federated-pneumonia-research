# Research Profile - Yohannes Alelign Biresaw

**BSc in Electrical and Computer Engineering, Haramaya University**  
**GPA:** 3.63/4.0  
**Email:** yohannes.sch.ca@gmail.com  
**GitHub:** https://github.com/john121319/federated-pneumonia-research

## Focused research direction

My research direction centres on the **security, privacy, and robustness of distributed machine-learning systems**. My strongest current research evidence is a completed federated medical-imaging study, which provides a practical foundation for future work on adversarial clients, robust aggregation, and trustworthy evaluation under legitimate non-IID variation.

## Completed research project

**Evaluating FedAvg and FedProx for Federated Chest-Radiograph Classification Under Non-IID Client Heterogeneity**

I developed and executed a patient-aware experimental pipeline using 26,684 RSNA examinations linked to 11,452 original patients. The study compared centralized learning, FedAvg, and FedProx across IID, moderate non-IID, and severe non-IID conditions; repeated experiments across three random seeds; selected models and thresholds using validation data only; and evaluated 18 frozen model-threshold pairs once on the held-out test set.

| Condition | ROC-AUC | PR-AUC | Balanced accuracy | F1-score |
| --- | ---: | ---: | ---: | ---: |
| Centralized | 0.8295 +/- 0.0040 | 0.5800 +/- 0.0057 | 0.7510 +/- 0.0037 | 0.5745 +/- 0.0107 |
| FedAvg IID | 0.8179 +/- 0.0030 | 0.5605 +/- 0.0042 | 0.7453 +/- 0.0048 | 0.5695 +/- 0.0068 |
| FedAvg moderate | 0.8131 +/- 0.0027 | 0.5553 +/- 0.0065 | 0.7355 +/- 0.0032 | 0.5541 +/- 0.0055 |
| FedAvg severe | 0.8070 +/- 0.0005 | 0.5440 +/- 0.0043 | 0.7312 +/- 0.0020 | 0.5529 +/- 0.0029 |
| FedProx moderate | 0.8112 +/- 0.0020 | 0.5505 +/- 0.0040 | 0.7375 +/- 0.0050 | 0.5608 +/- 0.0036 |
| FedProx severe | 0.8076 +/- 0.0019 | 0.5428 +/- 0.0092 | 0.7335 +/- 0.0030 | 0.5581 +/- 0.0076 |

Stronger client heterogeneity reduced federated performance. FedProx reduced client and global update magnitudes by roughly one-half, but it did not consistently improve held-out ranking performance. This distinction between optimization stability and predictive generalization motivates the proposed security-focused extension.

## Proposed MSc direction

**Secure and Robust Federated Learning Under Heterogeneous and Adversarial Clients**

The proposed research will examine whether robust aggregation can limit model-poisoning attacks without rejecting honest clients whose updates are unusual because their data are legitimately non-IID. Evaluation will include attack success, global and worst-client utility, benign-client false rejection, calibration, and computational cost.

## Core research questions

- How does legitimate label or acquisition heterogeneity change the geometry of honest client updates?
- Which robust aggregation methods preserve minority-client utility while limiting model-poisoning attacks?
- How often do security defenses reject honest clients because their data are different rather than malicious?

## Research preparation

- Python, TensorFlow/Keras, NumPy, pandas, scikit-learn, Git, and GitHub.
- DICOM preprocessing, CNN development, class-imbalance evaluation, threshold selection, and experiment tracking.
- FedAvg, FedProx, IID/non-IID partitioning, client-drift analysis, multi-seed evaluation, and frozen-test controls.
- Professional experience in IT support, networking, Windows systems, and practical troubleshooting.
- Embedded-systems foundations through Arduino-based simulation projects in robotics and agricultural monitoring.

## Research strengths

- Evidence-oriented experimental design that connects security claims with measurable utility and fairness outcomes.
- Careful handling of patient leakage, validation selection, class imbalance, and reproducibility controls.
- Practical systems background that supports implementation, troubleshooting, and research infrastructure.
- Willingness to report mixed or negative findings without overstating conclusions.

## Selected academic projects

- Simulation-based Arduino-controlled obstacle-avoidance robotic vehicle for industrial applications.
- Simulation-based Arduino-controlled agricultural quadcopter for crop and seed-health monitoring.

## Potential contribution to a research group

I can contribute a working federated medical-imaging pipeline, verified experiment records, and a disciplined approach to evaluation. Under supervision, I aim to deepen the mathematical and security analysis, implement robust aggregation and attack models, and develop the work into a thesis and publishable study.

## What I am seeking

A funded thesis-based MSc opportunity with a supervisor working on federated learning security, adversarial machine learning, privacy-aware machine learning, trustworthy AI, or distributed-system security. I am prepared to strengthen the mathematical foundations of my work while contributing a reproducible experimental pipeline and careful empirical analysis.
