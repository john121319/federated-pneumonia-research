# Robust Federated Chest-Radiograph Learning Under Institutional Heterogeneity: Optimization, Calibration, and Client-Level Reliability

**Prospective MSc Thesis Proposal**  
**Prepared by:** Yohannes Alelign Biresaw  
**Research area:** trustworthy and privacy-preserving machine learning

## Proposal Summary

Federated learning allows institutions to train a shared model without pooling their raw data, but its performance can become unstable when participating clients have different data distributions. My completed research project showed that stronger label heterogeneity reduced chest-radiograph performance and that FedProx reduced client drift by approximately one-half without consistently improving PR-AUC.

The proposed MSc thesis will investigate when optimization stability becomes practically useful. It will compare FedAvg and FedProx across several proximal strengths and local-training budgets, introduce controlled label and view-position heterogeneity, and evaluate calibration, threshold transfer, worst-client performance, and external generalization where feasible.

## 1. Background and Rationale

Federated learning supports collaborative model training without requiring each institution to transfer raw data to a central repository [1], [3]-[5]. It is attractive for medical imaging because data sharing is constrained by privacy, governance, ownership, and institutional policy. However, keeping data local does not remove statistical heterogeneity. When clients optimize different local distributions, their updates may diverge and the global model may perform unevenly across institutions.

FedProx was introduced to reduce this divergence by penalizing large departures from the current global model [2]. My completed project confirmed that the method can substantially reduce update magnitudes, but it also showed that lower model movement is not automatically equivalent to better generalization.

## 2. Preliminary Work and Motivation

As preparation for the proposed research, I completed a patient-aware study using the RSNA Pneumonia Detection Challenge dataset. The project included 26,684 examinations linked to 11,452 original patients, patient-exclusive train/validation/test splits, five-client IID and non-IID partitions, three random seeds, validation-only selection, and one frozen final test evaluation of 18 models.

| Condition | ROC-AUC | PR-AUC | Balanced accuracy | F1-score |
|---|---:|---:|---:|---:|
| Centralized | 0.8295 ± 0.0040 | 0.5800 ± 0.0057 | 0.7510 ± 0.0037 | 0.5745 ± 0.0107 |
| FedAvg IID | 0.8179 ± 0.0030 | 0.5605 ± 0.0042 | 0.7453 ± 0.0048 | 0.5695 ± 0.0068 |
| FedAvg moderate | 0.8131 ± 0.0027 | 0.5553 ± 0.0065 | 0.7355 ± 0.0032 | 0.5541 ± 0.0055 |
| FedAvg severe | 0.8070 ± 0.0005 | 0.5440 ± 0.0043 | 0.7312 ± 0.0020 | 0.5529 ± 0.0029 |
| FedProx moderate | 0.8112 ± 0.0020 | 0.5505 ± 0.0040 | 0.7375 ± 0.0050 | 0.5608 ± 0.0036 |
| FedProx severe | 0.8076 ± 0.0019 | 0.5428 ± 0.0092 | 0.7335 ± 0.0030 | 0.5581 ± 0.0076 |

The mixed result motivates a deeper question: when does reduced client drift produce practically meaningful improvements rather than only smaller model updates?

## 3. Research Gap and Novelty

Global AUC values can hide poor calibration, weak threshold transfer, or uneven performance on the smallest client. The proposed study will jointly examine client heterogeneity, optimization stability, global discrimination, and client-level reliability.

**Proposed novelty:** test whether reduced update magnitude is associated with better calibration, threshold transfer, and worst-client performance across controlled forms of medical-imaging heterogeneity.

## 4. Aim

To develop and evaluate a reproducible federated chest-radiograph framework that explains how institutional heterogeneity and optimization choices affect discrimination, calibration, threshold transfer, and client-level reliability.

## 5. Objectives

1. Extend the verified patient-aware baseline without reusing the frozen test set for model development.
2. Compare FedAvg and FedProx across prespecified proximal coefficients and local-epoch settings.
3. Model label skew and AP/PA view-position skew.
4. Measure global performance, calibration, threshold transfer, and worst-client behavior.
5. Test whether update magnitudes are associated with predictive generalization.
6. Conduct external validation where compatible data, licensing, and supervision permit.

## 6. Research Questions

1. How do label skew and AP/PA view-position skew affect global and client-level performance?
2. How sensitive is FedProx to the proximal coefficient and local-training duration?
3. When does lower client drift correspond to better PR-AUC, calibration, threshold transfer, or worst-client performance?
4. How well do validation-selected thresholds transfer across heterogeneous clients and external data?

## 7. Proposed Methodology

The RSNA dataset will remain the primary resource. Experiments will include approximately IID clients, Dirichlet label skew, AP/PA view-position skew, and a combined scenario where practical. FedAvg and FedProx will be compared using proximal coefficients of 0.001, 0.01, and 0.1 and local-training budgets of one and three epochs.

Primary outcomes will include PR-AUC, ROC-AUC, balanced accuracy, F1-score, log loss, calibration error, threshold-transfer performance, worst-client performance, and client/global update magnitudes. Model development will use validation data only. Final configurations will be frozen before held-out evaluation, and the existing frozen test set will not be repeatedly reused for development.

## 8. Expected Contributions

- A patient-aware framework separating label-related and view-related heterogeneity.
- A practical analysis of FedProx strength and local-training duration.
- Evidence connecting update stability with calibration, threshold transfer, and client-level reliability.
- A transparent evaluation protocol that limits test-set reuse.
- A foundation for later security and privacy extensions such as secure aggregation, differential privacy, or threat-aware federated learning.

## 9. Preparation and Research Fit

Through the completed project, I have developed practical experience in patient-aware medical-image preparation, federated optimization, controlled non-IID partitioning, multi-seed evaluation, validation-based model selection, experiment tracking, and scientific interpretation.

This proposal is a focused MSc pathway within my broader interest in secure and trustworthy intelligent systems, including artificial intelligence and machine learning, cybersecurity, applied cryptography, privacy-preserving computation, and secure distributed systems.

## 10. Ethics and Responsible Use

The research will use public, de-identified datasets and will not collect new patient information. The models will remain research prototypes. Institutional ethics, data-access, security, and responsible-use requirements will be confirmed with the supervisor.

## 11. Indicative 12-Month Plan

| Months | Activities |
|---|---|
| 1-2 | Literature review, supervisor refinement, ethics and access checks. |
| 3-4 | Implement label-skew and view-skew scenarios. |
| 5-6 | Run FedAvg/FedProx coefficient and local-epoch experiments. |
| 7-8 | Calibration, threshold-transfer, and worst-client analysis. |
| 9 | External validation where feasible. |
| 10 | Consolidate the reproducibility package. |
| 11 | Write the thesis and manuscript. |
| 12 | Revision, defense preparation, and submission. |

## References

1. McMahan B, et al. Communication-Efficient Learning of Deep Networks from Decentralized Data. PMLR, 2017.
2. Li T, et al. Federated Optimization in Heterogeneous Networks. MLSys, 2020.
3. Rieke N, et al. The Future of Digital Health with Federated Learning. npj Digital Medicine, 2020.
4. Kaissis GA, et al. Secure, Privacy-Preserving and Federated Machine Learning in Medical Imaging. Nature Machine Intelligence, 2020.
5. Sheller MJ, et al. Federated Learning in Medicine. Scientific Reports, 2020.
6. Shih G, et al. Augmenting the NIH Chest Radiograph Dataset with Expert Annotations of Possible Pneumonia. Radiology: AI, 2019.
