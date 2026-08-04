# Research Profile and Selected Project - Yohannes Alelign Biresaw

**BSc in Electrical and Computer Engineering, Haramaya University**  
**GPA:** 3.63/4.0  
**Email:** yohannes.sch.ca@gmail.com  
**GitHub:** https://github.com/john121319

## Research Direction

My long-term research interest is **secure and trustworthy intelligent systems**. I am interested in artificial intelligence and machine learning, cybersecurity, applied cryptography, privacy-preserving computation, secure distributed systems, and medical-image analysis. The common thread is the development of systems that remain accurate, robust, secure, and privacy-aware when they operate on sensitive, distributed, or heterogeneous data.

My completed federated-learning project is the strongest current evidence of my research preparation, but it does not limit my future direction to one algorithm or one application area. The research skills developed through the project are transferable to a wider range of AI, machine-learning, cybersecurity, and privacy problems.

## Completed Research Project

**Evaluating FedAvg and FedProx for Federated Classification of Pneumonia-Associated Lung Opacity Under Non-IID Data**

I developed a patient-aware pipeline using the RSNA Pneumonia Detection Challenge dataset and compared centralized learning, FedAvg, and FedProx under IID and controlled non-IID conditions.

- 26,684 chest-radiograph examinations linked to 11,452 original patients.
- Patient-exclusive training, validation, and test splits with zero patient overlap.
- Five simulated clients under IID, moderate non-IID, and severe non-IID conditions.
- Three random seeds and validation-only model and threshold selection.
- One frozen final test evaluation of 18 selected models.

## Main Results

| Condition | ROC-AUC | PR-AUC | Balanced accuracy | F1-score |
|---|---:|---:|---:|---:|
| Centralized | 0.8295 | 0.5800 | 0.7510 | 0.5745 |
| FedAvg IID | 0.8179 | 0.5605 | 0.7453 | 0.5695 |
| FedAvg moderate | 0.8131 | 0.5553 | 0.7355 | 0.5541 |
| FedAvg severe | 0.8070 | 0.5440 | 0.7312 | 0.5529 |
| FedProx moderate | 0.8112 | 0.5505 | 0.7375 | 0.5608 |
| FedProx severe | 0.8076 | 0.5428 | 0.7335 | 0.5581 |

Increasing client heterogeneity reduced federated performance, particularly PR-AUC. FedAvg retained a small PR-AUC advantage, while FedProx achieved slightly higher balanced accuracy and F1-score and reduced client and global update magnitudes by approximately one-half. The central conclusion is that optimization stability and predictive generalization were related but not identical.

## Broader Graduate Research Interests

- Artificial intelligence and machine learning: robustness, generalization, calibration, trustworthy AI, and real-world applications.
- Cybersecurity and AI security: detection and defense, adversarial robustness, model integrity, and secure system design.
- Cryptography and privacy: applied cryptography, privacy-preserving computation, secure aggregation, and protection of distributed learning systems.
- Secure distributed systems: privacy, integrity, communication, and threat-aware collaboration across devices or institutions.
- Medical and high-stakes AI: careful evaluation, uncertainty, fairness, and responsible use.

## Graduate Goal

I am seeking a thesis-based master's opportunity where I can work on a focused problem aligned with the advisor's expertise while developing stronger foundations in mathematics, algorithms, security, machine learning, and experimental research.
