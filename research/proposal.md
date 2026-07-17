# Evaluating FedAvg and FedProx for Federated Classification of
# Pneumonia-Associated Lung Opacity Under Non-IID Data

## Background

Deep-learning systems have shown potential for supporting the
interpretation of chest radiographs. However, medical imaging data are
often distributed across healthcare institutions and may be difficult to
combine centrally because of privacy, governance and data-ownership
requirements.

Federated learning enables multiple clients to train a shared model
without transferring their raw images to a central server. Federated
Averaging trains local models and produces a global model through
sample-weighted parameter averaging. However, clients may possess
different disease prevalences and sample quantities. This statistical
heterogeneity can cause local models to move in different optimization
directions and can reduce global-model performance.

FedProx extends Federated Averaging by adding a proximal term to each
client's local objective. This term limits how far a local model moves
from the current global model and may improve training under
heterogeneous client data.

## Research problem

Many introductory federated-learning studies divide one dataset equally
among clients and evaluate only overall accuracy. Such experiments do
not adequately reveal how increasing heterogeneity affects medical
sensitivity, convergence stability or client-level performance.

A controlled and reproducible comparison of centralized learning,
local-only learning, FedAvg and FedProx is therefore required.

## Aim

This study aims to evaluate the effect of IID and non-IID client data on
federated classification of pneumonia-associated lung opacity and to
determine whether FedProx provides more stable performance than FedAvg.

## Research questions

1. How closely does FedAvg approach centralized performance under IID
   client data?
2. How does increasing label-distribution heterogeneity affect global
   and client-level classification performance?
3. Does FedProx outperform FedAvg under moderate and severe non-IID
   conditions?
4. How does increasing local training from one epoch to three epochs
   affect convergence under severe non-IID data?
5. How does heterogeneity affect the difference between the best- and
   worst-performing clients?

## Dataset

The study will use the labelled training collection from the RSNA
Pneumonia Detection Challenge. DICOM images, image-level classes,
bounding-box annotations and the official mapping to the original NIH
dataset will be used to construct a patient-aware research manifest.

The task will be binary classification. Images categorized as Lung
Opacity will receive label 1, while Normal and No Lung Opacity / Not
Normal images will receive label 0.

## Methodology

The data will be divided into patient-grouped training, validation and
test partitions. Five simulated federated clients will be created using
three distribution settings: IID, moderate non-IID label skew with a
Dirichlet concentration parameter of 0.5, and severe non-IID label skew
with a concentration parameter of 0.1.

A lightweight convolutional neural network will be used consistently
across all experiments. The study will compare pooled centralized
training, local-only training, FedAvg and FedProx. Federated experiments
will use twenty communication rounds and one local epoch per round. A
separate ablation will compare one and three local epochs.

Experiments will be repeated with at least three random seeds. Methods
will be evaluated using accuracy, precision, sensitivity, specificity,
F1-score, balanced accuracy, ROC-AUC, PR-AUC, false-negative count and
client-level performance.

## Expected contribution

The study is expected to provide a reproducible analysis of how
controlled statistical heterogeneity affects federated medical-image
classification. It will determine the conditions under which FedAvg
approaches centralized performance and whether FedProx provides a
practical improvement under stronger heterogeneity.

## Limitations

The federated clients will be simulated from a single public source and
will not represent genuine independent hospitals. The study will not
provide a clinical diagnostic system or a formal privacy guarantee.
The classification target represents radiographic lung opacity
suspicious for pneumonia rather than definitive clinical pneumonia.

## Future work

Future work may extend the study to independent institutional datasets,
personalized federated learning, privacy-preserving mechanisms and
cybersecurity threats such as malicious-client model poisoning.