# Evaluating FedAvg and FedProx for Federated Chest-Radiograph Classification Under Non-IID Client Heterogeneity

**Yohannes Alelign Biresaw**  
BSc in Electrical and Computer Engineering, Haramaya University, Ethiopia  
Email: yohannes.sch.ca@gmail.com  
Repository: https://github.com/john121319/federated-pneumonia-research

## Abstract

**Background:** Federated learning enables several institutions to train a shared model while retaining raw data locally, but differences among client datasets can produce client drift and weaken global performance.  
**Objective:** This study evaluated how controlled client heterogeneity affected chest-radiograph classification and whether FedProx's reduction of model movement translated into better held-out prediction than FedAvg.  
**Methods:** The RSNA Pneumonia Detection Challenge data were consolidated into 26,684 examination-level labels and linked to 11,452 original patients. Patient-exclusive training, validation, and test splits contained 18,981, 3,841, and 3,862 images. A 389,537-parameter CNN was evaluated under centralized training, FedAvg IID, FedAvg moderate and severe non-IID, and FedProx moderate and severe non-IID conditions. Federated experiments used five simulated clients, 20 rounds, one local epoch, three random seeds, and Dirichlet label-skew partitions with `alpha = 0.5` and `alpha = 0.1`. Validation PR-AUC selected checkpoints, validation predictions selected thresholds, and 18 model-threshold pairs were frozen before a single final test evaluation.  
**Results:** Centralized training achieved the strongest overall test performance (ROC-AUC 0.8295 ± 0.0040; PR-AUC 0.5800 ± 0.0057). Increasing heterogeneity reduced federated performance, particularly PR-AUC. Under both non-IID conditions, FedAvg retained a small PR-AUC advantage, whereas FedProx achieved slightly higher balanced accuracy and F1-score. At validation-selected checkpoints, FedProx reduced global update magnitude by 48.83%-51.27% and mean client update magnitude by 43.53%-52.08%.  
**Conclusions:** FedProx clearly constrained client drift, but lower update magnitude did not consistently improve held-out ranking performance. FedAvg and FedProx were broadly comparable predictively and offered different strengths under the studied settings.

**Keywords:** federated learning; FedAvg; FedProx; non-IID data; medical imaging; chest radiography; lung opacity; client drift

## 1. Introduction

Deep learning for medical imaging benefits from large and diverse datasets, yet clinical data are commonly separated across institutions, jurisdictions, and governance systems. Federated learning offers a data-local training pattern in which clients optimize models locally and exchange model updates rather than raw examples [1,4-6]. This arrangement can support collaboration where unrestricted data pooling is not feasible, but it does not remove statistical or systems challenges.

Hospital datasets may differ in disease prevalence, imaging protocol, view position, equipment, patient characteristics, referral patterns, and annotation practice. Such non-independent and identically distributed (non-IID) data cause clients to optimize different local objectives. FedAvg remains the standard federated baseline because of its simplicity and communication efficiency [1], but local updates can drift under heterogeneity. FedProx adds a proximal penalty that limits the distance between each local model and the current global model [2]. SCAFFOLD addresses the same broad problem through control variates that correct client drift [3]. These approaches make clear that optimization stability is central to federated learning, but they do not imply that smaller updates will always produce better generalization.

Medical federated-learning studies have shown the feasibility of cross-institutional collaboration while also emphasizing governance, privacy, and generalization concerns [4-6]. Recent chest-radiograph work has examined class heterogeneity and collaborative pneumonia classification [11,12]. However, direct comparisons of optimization movement and held-out predictive performance remain useful, especially when evaluation is patient-aware and class imbalance is considered.

This study therefore asked whether FedProx's intended optimization effect could be observed under controlled non-IID chest-radiograph partitions and whether that effect improved final predictive performance. The project also prioritized evaluation discipline: patients were exclusive across splits and clients, experiments were repeated across three seeds, model and threshold selection used validation data only, and the final test protocol was frozen before evaluation.

### Research questions

1. How does increasing client heterogeneity affect ROC-AUC, PR-AUC, balanced accuracy, and F1-score?
2. Does FedProx reduce local-client and global-model update magnitudes relative to FedAvg?
3. Do smaller update magnitudes translate into better held-out predictive performance?
4. How large is the difference between centralized and federated training under the same patient-aware dataset split?

## 2. Materials and Methods

### 2.1 Study design

Six conditions were evaluated: centralized training; FedAvg with approximately IID clients; FedAvg with moderate non-IID clients; FedAvg with severe non-IID clients; FedProx with moderate non-IID clients; and FedProx with severe non-IID clients. Each condition was repeated with seeds 11, 22, and 33. The centralized model provided a reference for unconstrained access to the full training split.

### 2.2 Dataset construction and target definition

The RSNA challenge files contained 30,227 annotation rows representing 26,684 unique chest-radiograph examinations [7,13]. Positive examinations could appear in multiple rows because each opacity bounding box was recorded separately. Rows were consolidated by examination identifier to create one binary label per radiograph. The final dataset contained 6,012 positive and 20,672 negative examinations.

The official RSNA-to-NIH mapping linked RSNA identifiers to 11,452 original patients. The task was examination-level binary classification of pneumonia-associated lung opacity. A positive label indicates an opacity annotation associated with possible pneumonia; it does not constitute a complete clinical diagnosis.

### 2.3 Patient-aware splitting

Patients, rather than individual images, were assigned to training, validation, and test sets. No patient appeared in more than one split.

| Split | Images | Patients | Positive | Negative |
| --- | ---: | ---: | ---: | ---: |
| Training | 18,981 | 8,148 | 4,289 | 14,692 |
| Validation | 3,841 | 1,658 | 846 | 2,995 |
| Test | 3,862 | 1,646 | 877 | 2,985 |

Patient-aware splitting was necessary because some patients had multiple examinations. Random image-level splitting could otherwise place related studies from the same patient in both development and evaluation data.

### 2.4 DICOM preprocessing

DICOM pixel arrays were transformed using rescale slope and intercept. Non-finite values were replaced, `MONOCHROME1` images were inverted, intensities were clipped using robust percentiles, and values were normalized to `[0,1]`. Images were resized to `128 x 128` grayscale and cached as lossless 16-bit PNG files. Modest rotation, translation, zoom, and contrast augmentation were applied during training. Horizontal and vertical flips were excluded.

![Representative RSNA preprocessing examples](../results/figures/rsna_preprocessing_examples.png)

### 2.5 CNN architecture and optimization

The model contained four convolutional blocks with 32, 64, 128, and 256 filters. Each block used `3 x 3` convolutions, batch normalization, and ReLU activation. Max pooling followed the first three blocks. Global average pooling, dropout of 0.30, and a sigmoid output completed the network. The model contained 389,537 parameters.

Training used binary cross-entropy and Adam at a learning rate of 0.0005 [10]. Batch size was 32. Global class weights were 0.645964 for the negative class and 2.212754 for the positive class.

### 2.6 Federated client construction

Five simulated clients were created from the training split. All examinations belonging to one patient remained on the same client. Three partition types were used: approximately IID; moderate Dirichlet label skew with `alpha = 0.5`; and severe label skew with `alpha = 0.1`. Every training examination was assigned exactly once, and patient overlap between clients was zero.

These clients were controlled research partitions derived from one public dataset. They should not be interpreted as independent hospitals.

### 2.7 FedAvg and FedProx

For FedAvg, all clients initialized from the current global model, trained locally for one epoch, and returned complete model weights. The server computed a sample-size-weighted average. Batch-normalization moving statistics were included, while optimizer state was reset for each client and was not aggregated.

FedProx used the same server aggregation and training budget but added a proximal penalty over trainable variables. The coefficient was fixed at `mu = 0.01`. Non-trainable batch-normalization moving statistics were excluded from the proximal term.

### 2.8 Model selection and final-test protocol

Federated models trained for 20 communication rounds with full participation. Centralized models trained for up to 20 epochs. For each condition and seed, validation PR-AUC selected the checkpoint. A threshold was then selected from validation predictions using Youden's index.

Before final evaluation, 18 model paths, model hashes, validation-report hashes, checkpoints, and thresholds were recorded. The held-out test set was evaluated once. No threshold tuning, model selection, or hyperparameter choice used test data.

Metrics included ROC-AUC, PR-AUC, average precision, accuracy, precision, sensitivity, specificity, F1-score, balanced accuracy, log loss, and confusion-matrix counts. Client and global update L2 magnitudes were measured at validation-selected checkpoints. Results are summarized as mean ± sample standard deviation across three seeds. Because `n = 3` offers limited inferential power, comparisons are descriptive and are not presented as statistical-significance claims.

## 3. Results

### 3.1 Validation behaviour

Validation performance declined as FedAvg moved from IID to moderate and severe non-IID partitions. PR-AUC decreased from 0.5949 in the IID condition to 0.5853 under moderate heterogeneity and 0.5672 under severe heterogeneity. ROC-AUC, balanced accuracy, and F1-score followed the same general direction.

FedProx produced validation metrics close to FedAvg. Under moderate non-IID data, FedProx was slightly lower in ROC-AUC and PR-AUC and slightly higher in F1-score. Under severe non-IID data, FedProx was slightly higher in ROC-AUC, PR-AUC, and F1-score, while balanced accuracy was nearly unchanged. The differences were small.

| Condition | ROC-AUC | PR-AUC | Balanced accuracy | F1-score |
|---|---:|---:|---:|---:|
| Centralized | 0.8359 ± 0.0054 | 0.6015 ± 0.0134 | 0.7595 ± 0.0065 | 0.5762 ± 0.0214 |
| FedAvg IID | 0.8251 ± 0.0041 | 0.5949 ± 0.0053 | 0.7518 ± 0.0047 | 0.5688 ± 0.0076 |
| FedAvg moderate | 0.8181 ± 0.0033 | 0.5853 ± 0.0050 | 0.7450 ± 0.0030 | 0.5577 ± 0.0019 |
| FedAvg severe | 0.8122 ± 0.0014 | 0.5672 ± 0.0045 | 0.7414 ± 0.0035 | 0.5576 ± 0.0037 |
| FedProx moderate | 0.8167 ± 0.0029 | 0.5821 ± 0.0053 | 0.7432 ± 0.0049 | 0.5602 ± 0.0032 |
| FedProx severe | 0.8133 ± 0.0013 | 0.5691 ± 0.0058 | 0.7410 ± 0.0029 | 0.5602 ± 0.0103 |

### 3.2 Client-drift behaviour

The clearest FedProx effect appeared in update magnitude. Under moderate non-IID data, mean global update magnitude at the selected checkpoint fell from 4.7289 with FedAvg to 2.2906 with FedProx, a reduction of 51.27%. Mean client update magnitude fell from 10.5574 to 5.0459, a reduction of 52.08%.

Under severe non-IID data, FedProx reduced global update magnitude by 48.83% and mean client update magnitude by 43.53%.

| Heterogeneity | Algorithm | Global update L2 | Mean client update L2 | Global reduction | Client reduction |
|---|---|---:|---:|---:|---:|
| Moderate | FedAvg | 4.7289 ± 0.3516 | 10.5574 ± 2.0092 | - | - |
| Moderate | FedProx | 2.2906 ± 0.1452 | 5.0459 ± 0.8901 | 51.27% | 52.08% |
| Severe | FedAvg | 4.2901 ± 0.2812 | 16.8235 ± 3.0364 | - | - |
| Severe | FedProx | 2.1940 ± 0.1347 | 9.3554 ± 0.7318 | 48.83% | 43.53% |

![Global update comparison](../results/figures/fedavg_fedprox_global_update_l2_comparison.png)

![Mean client update comparison](../results/figures/fedavg_fedprox_mean_client_update_l2_comparison.png)

### 3.3 Frozen final test

The final evaluation included 3,862 images from 1,646 patients, with 877 positive and 2,985 negative examinations. The frozen protocol confirmed zero overlap with training and validation patients.

| Condition | ROC-AUC | PR-AUC | Balanced accuracy | F1-score |
| --- | ---: | ---: | ---: | ---: |
| Centralized | 0.8295 ± 0.0040 | 0.5800 ± 0.0057 | 0.7510 ± 0.0037 | 0.5745 ± 0.0107 |
| FedAvg IID | 0.8179 ± 0.0030 | 0.5605 ± 0.0042 | 0.7453 ± 0.0048 | 0.5695 ± 0.0068 |
| FedAvg moderate non-IID | 0.8131 ± 0.0027 | 0.5553 ± 0.0065 | 0.7355 ± 0.0032 | 0.5541 ± 0.0055 |
| FedAvg severe non-IID | 0.8070 ± 0.0005 | 0.5440 ± 0.0043 | 0.7312 ± 0.0020 | 0.5529 ± 0.0029 |
| FedProx moderate non-IID | 0.8112 ± 0.0020 | 0.5505 ± 0.0040 | 0.7375 ± 0.0050 | 0.5608 ± 0.0036 |
| FedProx severe non-IID | 0.8076 ± 0.0019 | 0.5428 ± 0.0092 | 0.7335 ± 0.0030 | 0.5581 ± 0.0076 |

Centralized training achieved the strongest overall results. FedAvg IID was the strongest federated condition. Under moderate non-IID data, FedAvg achieved higher ROC-AUC and PR-AUC, while FedProx achieved higher balanced accuracy and F1-score. Under severe non-IID data, ROC-AUC was nearly equal with a small FedProx advantage, FedAvg retained a small PR-AUC advantage, and FedProx again achieved higher balanced accuracy and F1-score.

![Final test PR-AUC](../results/figures/final_test_pr_auc_comparison.png)

![Final test ROC-AUC](../results/figures/final_test_roc_auc_comparison.png)

## 4. Discussion

### 4.1 Heterogeneity was the dominant pattern

The most consistent result was the effect of heterogeneity itself. Both federated algorithms lost ROC-AUC and PR-AUC as client label distributions became more uneven. The decline in PR-AUC was especially important because the positive class was less common and PR-AUC was the primary selection metric [9]. This finding is consistent with the broader literature on client drift and with recent chest-radiograph research showing that class heterogeneity is a substantial federated-learning challenge [3,11].

### 4.2 Optimization stability and predictive generalization were not equivalent

FedProx achieved its intended optimization effect by reducing client and global update magnitudes by roughly one-half. The size of that reduction did not produce a comparable increase in PR-AUC or ROC-AUC. A method can therefore stabilize optimization without improving the ranking of unseen examples.

Several explanations remain plausible. The selected `mu` may have constrained useful local adaptation as well as harmful drift. One local epoch may already have limited divergence, leaving less room for FedProx to improve prediction. The compact CNN and 20-round budget may also interact with the proximal term. These explanations should be tested prospectively rather than inferred from one configuration.

### 4.3 Threshold-dependent metrics captured a different aspect of behaviour

FedProx achieved slightly higher balanced accuracy and F1-score despite slightly lower PR-AUC. This is not contradictory. ROC-AUC and PR-AUC assess ranking across thresholds, whereas balanced accuracy and F1-score depend on a single validation-selected operating point. The result reinforces the importance of reporting both threshold-free and threshold-dependent metrics.

### 4.4 Centralized and federated learning answer different constraints

The centralized model remained strongest overall because it optimized directly over the complete training distribution. Federated learning addresses a different requirement: collaboration when raw data cannot be pooled. The relevant question is not whether federation always exceeds an unconstrained centralized reference, but whether its utility is acceptable for the governance, security, and privacy context.

### 4.5 Strengths

The study used original patient identifiers to prevent leakage, preserved patient integrity inside federated clients, repeated every condition across three seeds, reported both ranking and operating-point metrics, measured update magnitudes directly, and froze the final test protocol before evaluation. The interpretation also preserves the mixed evidence rather than declaring a universal winner.

### 4.6 Limitations

The clients were simulated from one public dataset rather than collected from independent hospitals. The main heterogeneity mechanism was label skew; equipment, demographic, temporal, and annotation shifts were not modeled directly. One CNN, one FedProx coefficient, one local epoch, and 20 rounds were used. Three seeds support descriptive comparison but are insufficient for strong inferential claims. FedAvg used the Keras fitting path, whereas FedProx required a custom gradient loop. Finally, federated learning alone does not provide formal privacy or security guarantees.

### 4.7 Future research

The next optimization-focused experiments should test prespecified `mu` values, additional local-epoch budgets, view-position or acquisition-skew partitions, calibration, and worst-client performance. A second dataset would strengthen external validation.

A distinct security-focused extension should introduce malicious clients and compare robust aggregation methods under the same legitimate non-IID conditions. The central security question is whether defenses can reject model-poisoning updates without excluding honest clients whose updates are unusual because of valid data heterogeneity. Such work should define a clear threat model and report benign-client false rejection, malicious-client detection, utility loss, worst-client performance, and computational overhead.

## 5. Conclusion

Increasing client heterogeneity reduced federated chest-radiograph performance, particularly PR-AUC. FedProx substantially constrained client and global model movement, but it did not consistently improve held-out ROC-AUC or PR-AUC. FedAvg produced slightly better ranking performance, whereas FedProx provided stronger drift control and small gains in balanced accuracy and F1-score. Under the studied configuration, the algorithms offered different strengths rather than a single universal winner.

## Data and Code Availability

Source code, configuration files, aggregate results, figures, and the frozen final-test protocol are available at https://github.com/john121319/federated-pneumonia-research. Raw RSNA DICOM images are not redistributed.

## Ethics and Responsible Use

The project used a public, de-identified research dataset and collected no new patient data. The models are research prototypes and must not be used for diagnosis, treatment, triage, or patient management.

## Author Contributions

Yohannes Alelign Biresaw developed and executed the research pipeline, prepared the data, implemented and ran the experiments, verified saved outputs, analyzed the results, prepared the figures and manuscript, and maintained the repository.

## Funding and Conflicts of Interest

This project was completed without dedicated external research funding. The author declares no conflict of interest.

## References

[1] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. Aguera y Arcas, 'Communication-Efficient Learning of Deep Networks from Decentralized Data,' Proceedings of Machine Learning Research, vol. 54, pp. 1273-1282, 2017.

[2] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, 'Federated Optimization in Heterogeneous Networks,' Proceedings of Machine Learning and Systems, vol. 2, pp. 429-450, 2020.

[3] S. P. Karimireddy, S. Kale, M. Mohri, S. Reddi, S. Stich, and A. T. Suresh, 'SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,' Proceedings of Machine Learning Research, vol. 119, pp. 5132-5143, 2020.

[4] N. Rieke et al., 'The Future of Digital Health with Federated Learning,' npj Digital Medicine, vol. 3, article 119, 2020.

[5] G. A. Kaissis, M. R. Makowski, D. Rueckert, and R. F. Braren, 'Secure, Privacy-Preserving and Federated Machine Learning in Medical Imaging,' Nature Machine Intelligence, vol. 2, pp. 305-311, 2020.

[6] M. J. Sheller et al., 'Federated Learning in Medicine: Facilitating Multi-Institutional Collaborations without Sharing Patient Data,' Scientific Reports, vol. 10, article 12598, 2020.

[7] G. Shih et al., 'Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert Annotations of Possible Pneumonia,' Radiology: Artificial Intelligence, vol. 1, no. 1, e180041, 2019.

[8] X. Wang et al., 'ChestX-ray8: Hospital-Scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases,' CVPR, pp. 2097-2106, 2017.

[9] T. Saito and M. Rehmsmeier, 'The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets,' PLOS ONE, vol. 10, no. 3, e0118432, 2015.

[10] D. P. Kingma and J. Ba, 'Adam: A Method for Stochastic Optimization,' International Conference on Learning Representations, 2015.

[11] P. Kulkarni, A. Kanhere, P. H. Yi, and V. S. Parekh, 'From Isolation to Collaboration: Federated Class-Heterogeneous Learning for Chest X-Ray Classification,' Proceedings of Machine Learning for Health, vol. 259, pp. 623-635, 2025.

[12] A. Mabrouk, R. P. Diaz Redondo, M. Abd Elaziz, and M. Kayed, 'Ensemble Federated Learning: An Approach for Collaborative Pneumonia Diagnosis,' Applied Soft Computing, vol. 144, article 110500, 2023.

[13] Radiological Society of North America, 'RSNA Pneumonia Detection Challenge 2018,' official challenge and dataset-mapping resource.
