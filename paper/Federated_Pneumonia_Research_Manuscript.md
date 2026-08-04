# Evaluating FedAvg and FedProx for Federated Classification of Pneumonia-Associated Lung Opacity Under Non-IID Data

**Yohannes Alelign Biresaw**  
BSc in Electrical and Computer Engineering, Haramaya University, Ethiopia  
Email: yohannes.sch.ca@gmail.com  
Repository: https://github.com/john121319/federated-pneumonia-research


## Abstract

Federated learning allows several institutions to train a shared model without pooling their raw data, but its performance can deteriorate when client datasets differ. This research project examined that problem in chest-radiograph classification by comparing centralized training, Federated Averaging (FedAvg), and Federated Proximal optimization (FedProx) under controlled IID and non-IID conditions. The RSNA Pneumonia Detection Challenge dataset was consolidated into 26,684 examination-level labels and linked to original NIH patient identifiers. Patient-exclusive training, validation, and test splits contained 18,981, 3,841, and 3,862 images, respectively, with zero patient overlap. A compact 389,537-parameter convolutional neural network was trained across three random seeds. Federated experiments used five simulated clients, 20 communication rounds, one local epoch, and Dirichlet label-skew partitions with alpha values of 0.5 and 0.1. Models and classification thresholds were selected only on validation data; 18 model-threshold pairs were then frozen and evaluated once on the held-out test set. Centralized learning achieved the strongest overall performance (ROC-AUC 0.8295 +/- 0.0040; PR-AUC 0.5800 +/- 0.0057). Among federated conditions, FedAvg IID performed best overall. Increasing heterogeneity reduced predictive performance, particularly PR-AUC. Under non-IID data, FedAvg retained slightly higher PR-AUC, whereas FedProx achieved slightly higher balanced accuracy and F1-score. FedProx reduced best-checkpoint global-update magnitude by about 49-51% and mean client-update magnitude by about 44-52%. These findings show that FedProx controlled client drift effectively, but the resulting optimization stability did not consistently improve held-out ranking performance. The study therefore supports a careful conclusion: FedAvg and FedProx were broadly comparable predictively, while FedProx offered a clearer advantage in limiting model movement under heterogeneous client data.

**Keywords:** federated learning; FedAvg; FedProx; non-IID data; medical imaging; chest radiography; lung opacity; client drift

## 1. Introduction

Deep learning systems for medical imaging usually benefit from large, diverse datasets. In practice, however, healthcare data are divided across hospitals, countries, and governance systems. Moving all records into one central repository can be limited by privacy obligations, data ownership, institutional policy, technical cost, and patient trust. Federated learning was developed for this kind of setting: clients train locally and share model updates rather than raw examples [1], [3].

The appeal of federated learning does not remove the difficulty of learning from different institutions. Hospital datasets can vary in disease prevalence, referral patterns, imaging equipment, view position, patient characteristics, and labeling practice. These differences create statistical heterogeneity, commonly described as non-independent and identically distributed, or non-IID, data. When clients optimize different local objectives, their model updates can point in different directions and cause client drift.

FedAvg is the standard starting point for federated optimization [1]. It is simple and communication-efficient, but it can struggle when local data distributions diverge. FedProx modifies the local objective by adding a proximal penalty that discourages each client model from moving too far from the current global model [2]. The method is designed to improve stability under heterogeneity, yet smaller updates do not necessarily guarantee better predictive performance on unseen patients.

This distinction motivated the present study. Rather than asking only which algorithm produced a higher score, the experiment measured both predictive behavior and optimization movement. The goal was to determine whether FedProx's control of client drift translated into better generalization for pneumonia-associated lung-opacity classification.

The project also placed unusual emphasis on evaluation discipline. Patients were kept exclusive across training, validation, and test sets; federated clients were patient-exclusive; experiments were repeated across three seeds; validation data alone determined checkpoints and thresholds; and the final test protocol was frozen before the test manifest was opened.

### Research questions

1. How does increasing client heterogeneity affect federated ROC-AUC, PR-AUC, balanced accuracy, and F1-score?
2. Does FedProx reduce local-client and global-model update magnitudes relative to FedAvg?
3. Do smaller update magnitudes translate into better held-out predictive performance?
4. How large is the difference between centralized and federated training under the same patient-aware dataset split?

## 2. Related Work

### Federated learning and FedAvg

FedAvg alternates between local optimization and server-side weighted averaging [1]. At each communication round, the server sends the current global model to selected clients. Each client trains on local data and returns updated model parameters. The server then weights each client by its number of training examples and produces the next global model. This design reduces the need to move raw data, but it introduces sensitivity to client participation, local training duration, sample imbalance, and statistical heterogeneity.

Federated learning is not automatically equivalent to privacy. Model updates can still reveal information, and a production system may require secure aggregation, differential privacy, authentication, audit controls, and threat modeling [3], [4]. The present study evaluates the data-local training pattern and optimization behavior; it does not claim cryptographic privacy.

### FedProx and heterogeneous optimization

FedProx was proposed to make federated optimization more tolerant of systems and statistical heterogeneity [2]. Its local objective adds a penalty proportional to the squared distance between local parameters and the round's global parameters. The penalty is controlled by the coefficient mu. A small value may have little effect, while a large value can prevent useful local adaptation. Therefore, FedProx should be evaluated for a specific dataset and training regime rather than assumed to dominate FedAvg.

In medical settings, this question is especially important because client distributions can differ for legitimate clinical reasons. Previous research has demonstrated the feasibility of federated learning across medical institutions while emphasizing generalizability, governance, and privacy challenges [3]-[5].

### Chest radiography and the RSNA challenge

The RSNA Pneumonia Detection Challenge added expert annotations of possible pneumonia-related lung opacity to a subset of the NIH chest-radiograph collection [6], [7]. The challenge itself included localization, but the present study consolidated the annotations into an examination-level binary task. A positive label indicates pneumonia-associated lung opacity, not a definitive clinical diagnosis of pneumonia. This wording matters because chest-radiograph appearance alone is not equivalent to a complete clinical assessment.

The positive class was less common than the negative class. For that reason, PR-AUC was treated as the primary model-selection metric. Precision-recall analysis is often more informative than ROC analysis when the positive class is relatively uncommon [8].

## 3. Materials and Methods

### Experimental design

The project compared six final conditions: centralized training; FedAvg with approximately IID clients; FedAvg with moderate non-IID clients; FedAvg with severe non-IID clients; FedProx with moderate non-IID clients; and FedProx with severe non-IID clients. Every condition was repeated using seeds 11, 22, and 33.

The centralized model provided a reference for training on all available data in one location. The federated conditions used the same CNN architecture, global class weights, image size, batch size, learning rate, communication-round budget, and local-epoch budget.

### Dataset construction and target definition

The raw RSNA files contained 30,227 annotation rows representing 26,684 unique chest-radiograph examinations. Some positive examinations appeared in multiple rows because they contained more than one opacity bounding box. Rows were therefore consolidated by examination identifier to create one binary target per image.

After consolidation, the dataset contained 6,012 positive examinations and 20,672 negative examinations. The official RSNA-to-NIH mapping associated RSNA image identifiers with 11,452 original patients. This mapping made patient-aware splitting possible.

### Patient-aware data splitting

Patients, rather than individual images, were assigned to training, validation, and test sets. The final split contained 18,981 training images from 8,148 patients, 3,841 validation images from 1,658 patients, and 3,862 test images from 1,646 patients. No original patient appeared in more than one split.

This design was necessary because a patient could have multiple examinations. Image-level random splitting would risk allowing related studies from the same patient to appear in both development and evaluation sets.

| Split | Images | Patients | Positive | Negative | Positive fraction |
|---|---|---|---|---|---|
| Training | 18,981 | 8,148 | 4,289 | 14,692 | 0.2260 |
| Validation | 3,841 | 1,658 | 846 | 2,995 | 0.2203 |
| Test | 3,862 | 1,646 | 877 | 2,985 | 0.2271 |

### DICOM preprocessing

DICOM pixel arrays were converted using rescale slope and intercept. Non-finite values were replaced, and MONOCHROME1 images were inverted so that image intensity followed a consistent visual direction. Robust percentile clipping reduced the effect of extreme values, after which images were normalized to [0, 1] and resized to 128 x 128 pixels.

To reduce repeated DICOM decoding during training, preprocessed images were cached as lossless 16-bit grayscale PNG files. Training-only augmentation used small rotations, translations, zoom changes, and contrast variation. Horizontal and vertical flips were excluded because arbitrary mirroring is not anatomically neutral for chest radiographs.

![Representative preprocessed RSNA images](../results/figures/rsna_preprocessing_examples.png)

*Figure 1. Representative preprocessed RSNA chest radiographs.*

### CNN architecture and optimization

The CNN contained four convolutional blocks with 32, 64, 128, and 256 filters. Each block used a 3 x 3 convolution, batch normalization, and ReLU activation. Max pooling followed the first three blocks. Global average pooling, dropout of 0.30, and a sigmoid output completed the model. The network contained 389,537 parameters.

Training used binary cross-entropy and Adam with a learning rate of 0.0005 [9]. Batch size was 32. Global class weights were 0.645964 for the negative class and 2.212754 for the positive class. Using the same global weights for every client avoided adding a second client-specific weighting mechanism to the comparison.

### Federated client construction

Five simulated clients were created from the training set. All examinations belonging to one patient were assigned to the same client. Three partition types were used: approximately IID; moderate Dirichlet label skew with alpha = 0.5; and severe Dirichlet label skew with alpha = 0.1. Lower alpha produced more uneven client label distributions.

Every training examination was assigned exactly once within each partition. Patient overlap between clients was zero. The clients were simulations derived from one public dataset and should not be interpreted as real hospitals.

### FedAvg and FedProx

For FedAvg, all five clients initialized from the same global model at each round, trained locally for one epoch, and returned complete model weights. The server computed a sample-size-weighted average. Batch-normalization moving statistics were included in aggregation, while optimizer state was reset for each local client and was not aggregated.

FedProx used the same server aggregation and training budget, with one addition: a proximal penalty over trainable variables. The coefficient was fixed at mu = 0.01 for the primary study. Non-trainable batch-normalization moving statistics were excluded from the proximal term.

### Model selection, thresholds, and final test

Federated models were trained for 20 rounds with full participation of five clients and one local epoch per round. Centralized models were trained for up to 20 epochs. For each condition and seed, the checkpoint with the highest validation PR-AUC was selected. A classification threshold was then chosen from validation predictions using Youden's index.

Before final evaluation, 18 model paths, model hashes, validation-report hashes, checkpoints, and thresholds were recorded in a frozen protocol. The test set was evaluated once. No model selection, threshold tuning, or hyperparameter choice was performed on test data.

Metrics included ROC-AUC, PR-AUC, average precision, accuracy, precision, sensitivity, specificity, F1-score, balanced accuracy, log loss, and confusion-matrix counts. Client-update and global-update L2 magnitudes were recorded at the validation-selected checkpoints. Means and sample standard deviations summarize three seeds. Because n = 3 offers limited inferential power, results are interpreted descriptively rather than as claims of statistical significance.

| Parameter | Value |
|---|---|
| Image size | 128 x 128 x 1 |
| CNN parameters | 389,537 |
| Clients | 5 |
| Federated rounds | 20 |
| Local epochs | 1 |
| Batch size | 32 |
| Optimizer | Adam |
| Learning rate | 0.0005 |
| Seeds | 11, 22, 33 |
| FedProx mu | 0.01 |
| Primary selection metric | Validation PR-AUC |
| Threshold selection | Validation Youden index |

## 4. Results

### Validation behavior

Validation results showed a clear decline as FedAvg moved from IID to moderate and severe non-IID partitions. PR-AUC fell from 0.5949 in the IID condition to 0.5853 under moderate heterogeneity and 0.5672 under severe heterogeneity. The same direction appeared in ROC-AUC, balanced accuracy, and F1-score, although the size of the change differed by metric.

FedProx produced validation metrics close to FedAvg. Under moderate non-IID data, FedProx was slightly lower in ROC-AUC and PR-AUC and slightly higher in F1-score. Under severe non-IID data, FedProx was slightly higher in ROC-AUC, PR-AUC, and F1-score, while balanced accuracy was nearly unchanged. These differences were small and were not treated as evidence of statistical superiority.

| Condition | ROC-AUC | PR-AUC | Balanced accuracy | F1-score |
|---|---|---|---|---|
| Centralized | 0.8359 +/- 0.0054 | 0.6015 +/- 0.0134 | 0.7595 +/- 0.0065 | 0.5762 +/- 0.0214 |
| FedAvg IID | 0.8251 +/- 0.0041 | 0.5949 +/- 0.0053 | 0.7518 +/- 0.0047 | 0.5688 +/- 0.0076 |
| FedAvg moderate | 0.8181 +/- 0.0033 | 0.5853 +/- 0.0050 | 0.7450 +/- 0.0030 | 0.5577 +/- 0.0019 |
| FedAvg severe | 0.8122 +/- 0.0014 | 0.5672 +/- 0.0045 | 0.7414 +/- 0.0035 | 0.5576 +/- 0.0037 |
| FedProx moderate | 0.8167 +/- 0.0029 | 0.5821 +/- 0.0053 | 0.7432 +/- 0.0049 | 0.5602 +/- 0.0032 |
| FedProx severe | 0.8133 +/- 0.0013 | 0.5691 +/- 0.0058 | 0.7410 +/- 0.0029 | 0.5602 +/- 0.0103 |

### Client-drift behavior

The clearest FedProx effect appeared in update magnitude. Under moderate non-IID data, the mean global update at the selected checkpoint fell from 4.7289 with FedAvg to 2.2906 with FedProx, a reduction of 51.27%. Mean client update magnitude fell from 10.5574 to 5.0459, a reduction of 52.08%.

Under severe non-IID data, FedProx reduced global update magnitude by 48.83% and mean client update magnitude by 43.53%. Severe heterogeneity increased client movement for both algorithms, but FedProx continued to constrain it substantially.

| Heterogeneity | Algorithm | Global update L2 | Mean client update L2 | Global reduction | Client reduction |
|---|---|---|---|---|---|
| Moderate | FedAvg | 4.7289 +/- 0.3516 | 10.5574 +/- 2.0092 | - | - |
| Moderate | FedProx | 2.2906 +/- 0.1452 | 5.0459 +/- 0.8901 | 51.27% | 52.08% |
| Severe | FedAvg | 4.2901 +/- 0.2812 | 16.8235 +/- 3.0364 | - | - |
| Severe | FedProx | 2.1940 +/- 0.1347 | 9.3554 +/- 0.7318 | 48.83% | 43.53% |

![Global update comparison](../results/figures/fedavg_fedprox_global_update_l2_comparison.png)

![Mean client update comparison](../results/figures/fedavg_fedprox_mean_client_update_l2_comparison.png)

### Frozen final test

The final evaluation included 3,862 images from 1,646 patients, with 877 positive and 2,985 negative examinations. The protocol confirmed zero overlap with training and validation patients. Threshold tuning on the test set was disabled, model selection on the test set was disabled, and the final evaluation was not repeated.

Centralized training achieved the strongest overall results: ROC-AUC 0.8295, PR-AUC 0.5800, balanced accuracy 0.7510, and F1-score 0.5745. FedAvg IID was the strongest federated condition overall, showing that similar client distributions made the federated optimization problem easier.

Under moderate non-IID data, FedAvg achieved higher ROC-AUC and PR-AUC, while FedProx achieved higher balanced accuracy and F1-score. Under severe non-IID data, the ROC-AUC values were nearly equal, with a small FedProx advantage; FedAvg retained a small PR-AUC advantage; and FedProx again achieved higher balanced accuracy and F1-score.

The final test therefore did not identify a single predictive winner. The ranking-oriented primary metric favored FedAvg slightly, while threshold-dependent balanced accuracy and F1-score favored FedProx.

| Condition | ROC-AUC | PR-AUC | Balanced accuracy | F1-score |
|---|---|---|---|---|
| Centralized | 0.8295 +/- 0.0040 | 0.5800 +/- 0.0057 | 0.7510 +/- 0.0037 | 0.5745 +/- 0.0107 |
| FedAvg IID | 0.8179 +/- 0.0030 | 0.5605 +/- 0.0042 | 0.7453 +/- 0.0048 | 0.5695 +/- 0.0068 |
| FedAvg moderate | 0.8131 +/- 0.0027 | 0.5553 +/- 0.0065 | 0.7355 +/- 0.0032 | 0.5541 +/- 0.0055 |
| FedAvg severe | 0.8070 +/- 0.0005 | 0.5440 +/- 0.0043 | 0.7312 +/- 0.0020 | 0.5529 +/- 0.0029 |
| FedProx moderate | 0.8112 +/- 0.0020 | 0.5505 +/- 0.0040 | 0.7375 +/- 0.0050 | 0.5608 +/- 0.0036 |
| FedProx severe | 0.8076 +/- 0.0019 | 0.5428 +/- 0.0092 | 0.7335 +/- 0.0030 | 0.5581 +/- 0.0076 |

| Partition | Delta ROC-AUC | Delta PR-AUC | Delta balanced accuracy | Delta F1 |
|---|---|---|---|---|
| Moderate | -0.0019 | -0.0049 | +0.0020 | +0.0067 |
| Severe | +0.0006 | -0.0012 | +0.0023 | +0.0052 |

![Final test PR-AUC](../results/figures/final_test_pr_auc_comparison.png)

![Final test ROC-AUC](../results/figures/final_test_roc_auc_comparison.png)

## 5. Discussion

### Heterogeneity reduced predictive performance

The most consistent pattern was not the difference between FedAvg and FedProx, but the effect of heterogeneity itself. Both algorithms lost ROC-AUC and PR-AUC when the client partition became more uneven. PR-AUC showed the clearest decline, which is important because the positive class was less common and PR-AUC was the primary selection metric.

This result is consistent with the idea that local clients learn from narrower and less representative label distributions. Even when the global model receives updates from every client, those updates may reflect different local trade-offs between sensitivity and specificity.

### Optimization stability and predictive generalization were not the same

FedProx achieved its intended optimization effect: it reduced both global and client update magnitudes by roughly one-half. Yet this substantial reduction did not lead to a matching increase in PR-AUC or ROC-AUC. The finding is useful because it separates two questions that are often treated as one. A method can stabilize training without improving the final ranking of unseen examples.

Several explanations are plausible. The chosen mu value may have constrained useful local adaptation as well as harmful drift. One local epoch may already have limited divergence, leaving less room for FedProx to improve prediction. The compact CNN and 20-round budget may also interact with the proximal term. These possibilities require targeted follow-up experiments rather than post-hoc certainty.

### Why threshold-dependent metrics favored FedProx

FedProx produced slightly higher balanced accuracy and F1-score under both non-IID conditions, despite slightly lower PR-AUC. This is not contradictory. ROC-AUC and PR-AUC evaluate ranking across many thresholds, whereas balanced accuracy and F1-score depend on one validation-selected threshold. A model can have a slightly weaker global ranking but a more useful operating point after threshold transfer.

For clinical research, that distinction matters. Ranking metrics are valuable for comparing discrimination, but deployment decisions also depend on sensitivity, specificity, calibration, and the cost of errors. The current study does not support clinical deployment; it does show why a single metric should not be used to summarize all behavior.

### Centralized and federated learning

The centralized model remained strongest overall. This result should not be interpreted as a failure of federated learning. Centralized optimization had direct access to the full training distribution and avoided client partitioning. Federated learning addresses a different constraint: collaboration when raw data cannot be pooled. The relevant question is therefore whether the performance trade-off is acceptable for the governance and privacy context, not whether federation always exceeds an unconstrained centralized reference.

### Strengths

The study's main strengths are methodological rather than architectural. It used original patient identifiers to prevent leakage, preserved patient integrity inside federated clients, repeated every condition across three seeds, reported both ranking and threshold-dependent metrics, measured update magnitudes directly, and froze the final test protocol before evaluation. The conclusion also preserves the mixed nature of the evidence instead of declaring a winner based on one metric.

### Limitations

First, the five clients were simulated from one public dataset rather than collected from independent hospitals. The partitions therefore represent controlled heterogeneity, not the full complexity of real institutional differences.

Second, the main non-IID mechanism was label skew. Feature shift, equipment variation, demographic shift, annotation practice, and temporal change were not modeled directly.

Third, the main study used one CNN, one FedProx coefficient, one local epoch, and 20 communication rounds. Different architectures or optimization budgets could change the relative behavior.

Fourth, three seeds are sufficient for a more reliable descriptive comparison than a single run, but they are not enough for strong statistical-significance claims.

Fifth, the FedAvg local training path used the standard Keras fitting mechanism, whereas FedProx required a custom gradient loop. The shared model, optimizer settings, data, and training budget reduce but do not eliminate implementation differences.

Finally, federated learning alone does not guarantee privacy or security. Secure aggregation, differential privacy, adversarial robustness, and communication analysis were outside the scope of this experiment.

### Future work

A focused extension should test several mu values, additional local-epoch settings, and feature-skew partitions based on view position or acquisition proxies. Client-level performance and calibration should be reported alongside global metrics. A second chest-radiograph dataset would provide a stronger test of external generalization, subject to access, licensing, and supervisor approval.

A later systems-oriented study could add secure aggregation or differential privacy and measure the resulting trade-offs in utility, privacy, communication cost, and training stability.

## 6. Conclusion

This project evaluated centralized learning, FedAvg, and FedProx for patient-aware classification of pneumonia-associated lung opacity under controlled federated heterogeneity. Increasing non-IID severity reduced predictive performance, with the clearest deterioration in PR-AUC. FedProx substantially constrained client and global model movement, but it did not consistently improve held-out ROC-AUC or PR-AUC. FedAvg retained a small PR-AUC advantage, while FedProx achieved small improvements in balanced accuracy and F1-score. The most defensible conclusion is therefore not that one algorithm won, but that they offered different strengths: FedAvg produced slightly better ranking performance in this experiment, whereas FedProx provided stronger control of client drift. This distinction provides a practical basis for future research on calibration, client-level reliability, richer forms of heterogeneity, and privacy-preserving deployment.

## Data and Code Availability

The source code, configuration files, aggregate results, figures, and frozen final-test protocol are available at https://github.com/john121319/federated-pneumonia-research. Raw RSNA DICOM images are not redistributed.

## Ethics and Responsible Use

The project used a public, de-identified research dataset and collected no new patient data. Formal institutional requirements should be confirmed before university or journal submission. The models are research prototypes and must not be used for diagnosis, treatment, triage, or patient management.

## Author Contributions

Yohannes Alelign Biresaw developed and executed the research pipeline, prepared the data, implemented and ran the experiments, verified the saved outputs, analyzed the results, prepared the figures and manuscript, and maintained the repository.

## Funding and Conflicts of Interest

This project was completed without dedicated external research funding. The author declares no conflict of interest.

## References

[1] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. Aguera y Arcas, 'Communication-Efficient Learning of Deep Networks from Decentralized Data,' Proceedings of Machine Learning Research, vol. 54, pp. 1273-1282, 2017. https://proceedings.mlr.press/v54/mcmahan17a.html

[2] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, 'Federated Optimization in Heterogeneous Networks,' Proceedings of Machine Learning and Systems, vol. 2, pp. 429-450, 2020. https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html

[3] N. Rieke et al., 'The Future of Digital Health with Federated Learning,' npj Digital Medicine, vol. 3, article 119, 2020. doi:10.1038/s41746-020-00323-1.

[4] G. A. Kaissis, M. R. Makowski, D. Rueckert, and R. F. Braren, 'Secure, Privacy-Preserving and Federated Machine Learning in Medical Imaging,' Nature Machine Intelligence, vol. 2, pp. 305-311, 2020. doi:10.1038/s42256-020-0186-1.

[5] M. J. Sheller et al., 'Federated Learning in Medicine: Facilitating Multi-Institutional Collaborations without Sharing Patient Data,' Scientific Reports, vol. 10, article 12598, 2020. doi:10.1038/s41598-020-69250-1.

[6] G. Shih et al., 'Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert Annotations of Possible Pneumonia,' Radiology: Artificial Intelligence, vol. 1, no. 1, e180041, 2019. doi:10.1148/ryai.2019180041.

[7] X. Wang et al., 'ChestX-ray8: Hospital-Scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases,' in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 2097-2106, 2017. doi:10.1109/CVPR.2017.369.

[8] T. Saito and M. Rehmsmeier, 'The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets,' PLOS ONE, vol. 10, no. 3, e0118432, 2015. doi:10.1371/journal.pone.0118432.

[9] D. P. Kingma and J. Ba, 'Adam: A Method for Stochastic Optimization,' International Conference on Learning Representations, 2015.

[10] Radiological Society of North America, 'RSNA Pneumonia Detection Challenge (2018),' official challenge and dataset-mapping resource. https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018
